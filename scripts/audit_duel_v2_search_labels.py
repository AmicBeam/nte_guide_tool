#!/usr/bin/env python3
"""Audit archived five-cross ends for leftover action points.

Replays stored action traces only. Does not start a game, train, or export a model.
"""
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.modules.card_game.engine.duel_v2.flow import (  # noqa: E402
    apply_action, card_cost, card_is_free_instant, legal_actions, new_game, normal_attack_payment,
)
from app.modules.card_game.engine.duel_v2.projection import preview  # noqa: E402
from app.modules.card_game.engine.duel_v2.state import damage_immune, hero, other  # noqa: E402
from app.modules.card_game.rl.cross_lineup import identity  # noqa: E402
from app.modules.card_game.rl.search_label_audit import (  # noqa: E402
    ap_bucket, classify_leftover, count_ends, question_rows, turn_stage,
)

KIND = {'end_turn': 0, 'attack': 1, 'play_card': 2, 'ultimate': 3, 'mulligan': 4, 'choose': 5}
LABEL = {
    'starter': '创生', 'weave-rush': '覆纹', 'quick-rush': '快攻', 'zhenhong': '真红', 'murk': '浊燃',
    'explained': '可解释', 'suspicious': '可疑', 'insufficient': '证据不足',
}
STAGE = {
    'first_opening': '先手首回合', 'second_opening': '后手首回合',
    'own_2_to_4': '自己第2–4回合', 'own_5_to_8': '自己第5–8回合', 'own_9_plus': '自己第9回合及以后',
}


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def config_payload():
    return dict(
        question='Do archived Gumbel labels end turns while 2 action points and a legal attack or card remain?',
        corpora=dict(
            training='five-cross formal episodes; both sides act by Gumbel search_choice',
            argmax='frozen evaluation with empty search_sides; pure-network actions',
            search='frozen evaluation with search_sides a and b; Gumbel actions, no per-action value stored',
        ),
        denominator='one playing-phase selected end_turn, keyed by corpus, game id and decision step',
        pi_preference='recorded on the same decision when a policy row exists; not counted as another turn',
        buckets=['0', '1', '2', '3+'],
        opening_ap='global turn 1 refreshes 1 point; every later turn refreshes 2 plus delayed extra points',
        ultimate_ap='engine start_ultimate spends no action points in this rule hash',
        leftover_response='points still held at the end decision remain available for the opponent turn responses; leaving 2 is not the 1-point response reserve',
        classes=dict(
            explained='every spend is a knockdown, an immune front that takes 0, or there is no action that spends a point',
            suspicious='at least one attack or battle card spends a point, the logged attack amount is positive, the attacker survives, and the logged counter is 0. This is the outgoing attack number, not confirmed hit-point loss',
            insufficient='a payable non-battle card, a surviving counter trade, or a zero-damage combat remains',
        ),
        not_a_verdict='suspicious does not mean the end was worse; explained does not mean the line was strong',
        value_comparison='archived rows store pi and visit_pi only; per-action completed Q is absent',
        serving_export=False, formal_run=False, cuda=False, new_games=False,
    )


def load_json_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def preset(decks, side):
    return decks[side]['id']


def card_of(state, side, action):
    if action.get('type') != 'play_card':
        return None
    return next(card for card in state['sides'][side]['hand'] if card['instance_id'] == action['card_id'])


def spend_cost(state, side, action):
    kind = action.get('type')
    if kind == 'attack':
        payment = normal_attack_payment(state, side, action['character_id'])
        return None if payment is None else int(payment['ap'])
    if kind == 'play_card':
        card = card_of(state, side, action)
        if card_is_free_instant(state, side, card):
            return 0
        return int(card_cost(state, side, card))
    if kind == 'ultimate':
        return 0
    return None


def option_view(state, side, action, foe_front, foe_immune):
    kind = action['type']
    cost = spend_cost(state, side, action)
    card = card_of(state, side, action)
    view = dict(kind=kind, ap_cost=cost, card_type=(card or {}).get('type'), card_id=(card or {}).get('card_id'),
                card_name=(card or {}).get('name'), character_id=action.get('character_id') or (card or {}).get('character_id'),
                damage=0, counter=0, survives=True, immune_target=False, preview=False)
    if cost is None or kind not in ('attack', 'play_card'):
        return view
    if kind == 'play_card' and card and card.get('type') != 'battle' and cost == 0:
        return view
    if kind == 'play_card' and card and card.get('type') != 'battle':
        return view
    try:
        shown = preview(state, side, action)
    except Exception as exc:
        view['preview_error'] = type(exc).__name__
        return view
    view['preview'] = True
    view['damage'] = int(shown.get('attack') or 0)
    view['counter'] = int(shown.get('counter') or 0)
    cid = view['character_id']
    before = hero(state, side, cid)['hp'] if cid else 0
    view['survives'] = before + int(shown.get('自身生命变化') or 0) > 0
    target = shown.get('攻击目标') or ''
    front_name = hero(state, other(side), foe_front)['name'] if foe_front else ''
    view['immune_target'] = bool(foe_immune and foe_front and target == front_name and view['damage'] == 0)
    view['target'] = target
    view['self_hp_delta'] = int(shown.get('自身生命变化') or 0)
    view['foe_player_hp_delta'] = int(shown.get('对手玩家生命变化') or 0)
    return view


def policy_fields(row, action):
    if not row or 'pi' not in row:
        return dict(pi_end_first=None, label_aligned=None)
    selected = int(row['selected'])
    candidates = row['c']
    pi = row['pi']
    if not 0 <= selected < len(candidates) or len(pi) < len(candidates):
        return dict(pi_end_first=None, label_aligned=False)
    aligned = int(candidates[selected][0]) == KIND.get(action['type'])
    top = max(range(len(candidates)), key=lambda index: pi[index])
    end_indexes = [index for index, candidate in enumerate(candidates) if int(candidate[0]) == KIND['end_turn']]
    end_index = end_indexes[0] if len(end_indexes) == 1 else None
    visit = row.get('visit_pi') or []
    return dict(
        pi_end_first=bool(int(candidates[top][0]) == KIND['end_turn']),
        pi_top_kind=int(candidates[top][0]),
        label_aligned=aligned,
        search_budget=row.get('search_budget'),
        end_pi=None if end_index is None else float(pi[end_index]),
        end_visit=None if end_index is None or end_index >= len(visit) else float(visit[end_index]),
        legal_encoded=len(candidates),
    )


def replay_game(corpus, game_id, seed, first, decks, actions, rows_by_step, episode_path, replay_path):
    state = new_game(seed=int(seed), decks=decks, first_side=first)
    records = []
    failures = []
    for step, item in enumerate(actions):
        side = item['side']
        action = item['action']
        if state['phase'] == 'finished':
            failures.append(dict(game_id=game_id, step=step, error='action after finish'))
            break
        if action.get('type') == 'end_turn' and state['phase'] == 'playing':
            foe = other(side)
            front = state['sides'][foe].get('front')
            immune = bool(front) and damage_immune(state, foe, front)
            legal = [choice for choice in legal_actions(state, side) if choice.get('type') != 'concede']
            interesting = [choice for choice in legal if choice.get('type') in ('attack', 'play_card', 'ultimate')]
            ap = int(state['sides'][side]['ap'])
            need_preview = ap == 2 and any(choice.get('type') in ('attack', 'play_card') for choice in interesting)
            options = [option_view(state, side, choice, front, immune) if need_preview else dict(
                kind=choice['type'], ap_cost=spend_cost(state, side, choice),
                card_type=(card_of(state, side, choice) or {}).get('type') if choice['type'] == 'play_card' else None,
                card_id=(card_of(state, side, choice) or {}).get('card_id') if choice['type'] == 'play_card' else None,
            ) for choice in interesting]
            attacks = [option for option in options if option['kind'] == 'attack']
            plays = [option for option in options if option['kind'] == 'play_card']
            paying = [option for option in options if int(option.get('ap_cost') or 0) >= 1 and option['kind'] in ('attack', 'play_card')]
            own_turn = int(state['sides'][side]['turn_count'])
            policy = policy_fields(rows_by_step.get(step), action)
            record = dict(
                corpus=corpus, game_id=game_id, seed=int(seed), step=step, side=side,
                seat='first' if side == state['first_side'] else 'second',
                preset=preset(decks, side), opponent=preset(decks, foe),
                turn=int(state['turn']), own_turn=own_turn, stage=turn_stage(own_turn, state['turn']),
                phase=state['phase'], selected='end_turn', ap=ap, bucket=ap_bucket(ap),
                has_attack_or_play=bool(attacks or plays),
                paying_attack=sum(option['kind'] == 'attack' and int(option.get('ap_cost') or 0) >= 1 for option in options),
                paying_play=sum(option['kind'] == 'play_card' and int(option.get('ap_cost') or 0) >= 1 for option in options),
                free_attack=sum(option['kind'] == 'attack' and int(option.get('ap_cost') or 0) == 0 for option in options),
                free_play=sum(option['kind'] == 'play_card' and int(option.get('ap_cost') or 0) == 0 for option in options),
                battle_play=sum(option['kind'] == 'play_card' and option.get('card_type') == 'battle' for option in options),
                ultimate=sum(option['kind'] == 'ultimate' for option in options),
                enemy_front=front,
                enemy_front_zhenhong_immune=bool(immune and front == 'zhenhong'),
                enemy_front_immune=bool(immune),
                our_front=state['sides'][side].get('front'),
                our_front_hp=(hero(state, side, state['sides'][side]['front'])['hp'] if state['sides'][side].get('front') else None),
                episode=str(episode_path) if episode_path else None,
                replay=str(replay_path) if replay_path and Path(replay_path).exists() else None,
                **policy,
            )
            if ap == 2 and (attacks or plays):
                label, reasons = classify_leftover(paying or [option for option in options if option['kind'] in ('attack', 'play_card')])
                record['label'] = label
                record['reasons'] = reasons
                record['options'] = [{key: option.get(key) for key in (
                    'kind', 'ap_cost', 'card_type', 'card_id', 'card_name', 'character_id',
                    'damage', 'counter', 'survives', 'immune_target', 'target', 'preview_error')}
                    for option in options if option['kind'] in ('attack', 'play_card')]
            records.append(record)
        try:
            state = apply_action(state, side, action)
        except Exception as exc:
            failures.append(dict(game_id=game_id, step=step, error=type(exc).__name__, detail=str(exc)[:180]))
            break
    return records, failures


def non_end_pi(corpus, game_id, seed, actions, rows_by_step):
    """Decisions whose improved policy ranks end first while the played action is not end."""
    found = []
    for step, item in enumerate(actions):
        row = rows_by_step.get(step)
        if not row or 'pi' not in row or item['action'].get('type') == 'end_turn':
            continue
        fields = policy_fields(row, item['action'])
        if fields.get('pi_end_first'):
            found.append(dict(corpus=corpus, game_id=game_id, seed=int(seed), step=step,
                              selected=item['action'].get('type'), preset=row.get('key')))
    return found


def nest_count(rows, *fields):
    counts = Counter()
    for row in rows:
        counts[tuple(row[field] for field in fields)] += 1
    return [{'keys': dict(zip(fields, key)), 'n': value} for key, value in sorted(counts.items())]


def summarize(ends, pi_other):
    training = [row for row in ends if row['corpus'] == 'training']
    asked = question_rows(ends)
    by_label = Counter(row.get('label') for row in asked)
    return dict(
        ends=len(ends),
        by_corpus=nest_count(ends, 'corpus', 'bucket'),
        by_preset_seat_stage_opponent=nest_count(ends, 'corpus', 'preset', 'seat', 'stage', 'opponent', 'bucket'),
        training_with_attack_or_play=nest_count(
            [row for row in training if row['has_attack_or_play']], 'bucket'),
        training_paying_spend=nest_count(
            [row for row in training if row['paying_attack'] or row['paying_play']], 'bucket'),
        question_ap2_with_attack_or_play=len(asked),
        question_by_preset=nest_count(asked, 'preset', 'seat', 'opponent', 'label'),
        question_labels=dict(by_label),
        question_reasons=nest_count(asked, 'label', 'reasons') if False else [
            {'label': row.get('label'), 'reasons': row.get('reasons'), 'n': value}
            for (row_label, reasons), value in Counter(
                (row.get('label'), tuple(row.get('reasons') or ())) for row in asked).items()
            for row in [dict(label=row_label, reasons=list(reasons))]
        ],
        pi_end_on_real_ends=dict(
            ranked=sum(row.get('pi_end_first') is True for row in training),
            not_ranked=sum(row.get('pi_end_first') is False for row in training),
            missing=sum(row.get('pi_end_first') is None for row in training),
        ),
        pi_end_but_other_action=len(pi_other),
        zhenhong_immune_front_on_ap2=sum(row.get('enemy_front_zhenhong_immune') for row in asked),
        label_misaligned=sum(row.get('label_aligned') is False for row in training),
    )


def render(summary, failures, coverage):
    labels = summary['question_labels']
    lines = [
        '# 五套交叉训练的结束回合标签审计',
        '',
        '只回放 `five-cross-20260924` 已归档的动作序列。没有新对局，没有训练，没有 CUDA，没有正式长训，也没有使用已淘汰的分层打分头。当前规则哈希与该次运行记录的哈希一致，因此回放能执行归档动作。',
        '',
        '一个完整回合只计一次：对局、决策序号上的那一次实际结束。策略 `pi` 把结束排第一但实际走了别的动作，另列，不计入回合分母。终结在这版规则里不支付行动力。先手全局第 1 回合只刷新 1 点；其后每回合刷新 2 点。结束时剩下的点数会留给对方回合的响应，留 1 点和留 2 点不是同一件事。',
        '',
        '## 剩余行动力',
        '',
        '分母是各推理方式里实际点下结束回合的次数。',
        '',
        '| 推理 | 0 点 | 1 点 | 2 点 | 3 点及以上 | 合计 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    by_corpus = defaultdict(Counter)
    for item in summary['by_corpus']:
        by_corpus[item['keys']['corpus']][item['keys']['bucket']] = item['n']
    for corpus in ('training', 'argmax', 'search'):
        counts = by_corpus[corpus]
        total = sum(counts.values())
        lines.append('| {name} | {a} | {b} | {c} | {d} | {t} |'.format(
            name={'training': '训练 Gumbel', 'argmax': '纯网络', 'search': '冻结评估 Gumbel'}[corpus],
            a=counts['0'], b=counts['1'], c=counts['2'], d=counts['3+'], t=total))
    lines.extend([
        '',
        '按构筑、先后手、自己的回合段和对手的拆分在 `summary.json` 的 `by_preset_seat_stage_opponent`。',
        '',
        '## 留 2 点且仍有出击或出牌',
        '',
        f"训练对局里，实际结束时恰好剩下 2 点、并且当时还有合法出击或出牌的回合是 **{summary['question_ap2_with_attack_or_play']}**。",
        f"其中可解释 {labels.get('explained', 0)}，可疑 {labels.get('suspicious', 0)}，证据不足 {labels.get('insufficient', 0)}。",
        f"敌方前排正是免伤中的真红：{summary['zhenhong_immune_front_on_ap2']} 次。",
        '',
        '可解释只包括：没有会支付行动力的动作，或每一个会支付的出击／战斗牌都会把进攻者反击倒地，或打在免伤前排上且实际伤害为 0。可疑只包括：存在一次日志反击为 0、进攻者还站着、并且日志攻击数字大于 0 的出击或战斗牌。这个数字是打出去的攻击，不是已经扣掉的生命。付费战术或弧盘没有模拟后续收益，归入证据不足。',
        '',
        '## 搜索有没有留下比较',
        '',
        coverage,
        '',
        f"训练结束回合里，改进策略 `pi` 也把结束排第一的有 {summary['pi_end_on_real_ends']['ranked']} 次，没有排第一的有 {summary['pi_end_on_real_ends']['not_ranked']} 次。另有 {summary['pi_end_but_other_action']} 个决策是 `pi` 把结束排第一、实际动作却不是结束；这些不是已结束的回合。",
        '',
        '现有记录不能证明结束比出击或出牌更好，也不能证明更差。`pi` 和 `visit_pi` 混着先验与访问，归档没有逐动作的 completed Q。搜索选中也不是正确标签。',
        '',
        '## 回放',
        '',
        f"未能跑完的对局：{len(failures)}。标签列和实际动作对不上的训练结束回合：{summary['label_misaligned']}。",
        '',
        '逐回合记录在 `turns.jsonl`。留 2 点且有出击或出牌的分类和追溯字段在同一行的 `label`、`game_id`、`seed`、`step`、`replay`。',
        '',
    ])
    return '\n'.join(lines)


def main():
    formal = Path('artifacts/rl-evals/five-cross-20260924/formal')
    out = Path('artifacts/rl-evals/policy-recovery-20260924/search-label-audit')
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'config.json', config_payload())
    write(out / 'source-sha.json', dict(
        rule_hash=identity(),
        files={str(path): sha256_file(ROOT / path) for path in (
            'scripts/audit_duel_v2_search_labels.py',
            'app/modules/card_game/rl/search_label_audit.py',
            'app/modules/card_game/engine/duel_v2/flow.py',
            'app/modules/card_game/rl/episode_replay.py',
            'app/modules/card_game/rl/information_search.py',
        )},
    ))
    ends = []
    pi_other = []
    failures = []
    sources = [('training', formal / 'episodes', formal / 'episodes', None)]
    # Evaluation traces carry the same action list as a training episode, plus a public replay when one was exported.
    jobs = []
    for path in sorted((formal / 'episodes').glob('*.json.gz')):
        data = load_json_gz(path)
        jobs.append(('training', data['job']['id'], data['job']['seed'], data['job']['first'], data['job']['decks'],
                     data['actions'], {row['step']: row for row in data['rows'] if 'pi' in row}, path, None))
    for corpus, folder in (('argmax', formal / 'argmax/evaluation'), ('search', formal / 'search/evaluation')):
        for path in sorted((folder / 'raw').glob('*.json.gz')):
            data = load_json_gz(path)
            replay = folder / 'replays' / f"{data['id']}.json.gz"
            jobs.append((corpus, data['id'], data['seed'], data['first'], data['decks'], data['actions'], {}, path, replay))
    print(f'replay {len(jobs)} games', flush=True)
    for index, job in enumerate(jobs, start=1):
        corpus, game_id, seed, first, decks, actions, rows, episode, replay = job
        try:
            records, failed = replay_game(corpus, game_id, seed, first, decks, actions, rows, episode, replay)
        except Exception as exc:
            failures.append(dict(game_id=game_id, step=None, error=type(exc).__name__, detail=str(exc)[:180]))
            continue
        ends.extend(records)
        failures.extend(failed)
        if corpus == 'training':
            pi_other.extend(non_end_pi(corpus, game_id, seed, actions, rows))
        if index % 25 == 0:
            print(f'{index} ends {len(ends)} failures {len(failures)}', flush=True)
    ends = count_ends(ends)
    summary = summarize(ends, pi_other)
    summary['replay_failures'] = len(failures)
    write(out / 'summary.json', summary)
    write(out / 'replay-failures.json', failures)
    write(out / 'pi-end-other-action.json', pi_other)
    with (out / 'turns.jsonl').open('w', encoding='utf-8') as handle:
        for row in ends:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    asked = question_rows(ends)
    samples = {label: [row for row in asked if row.get('label') == label] for label in ('explained', 'suspicious', 'insufficient')}
    write(out / 'samples.json', {label: items for label, items in samples.items()})
    coverage = (
        '训练策略行保存了 `pi`、`visit_pi`、`search_budget` 和选中下标，没有逐动作访问次数以外的价值，也没有 completed Q。'
        '冻结评估的 `search_stats` 只有模拟次数、合法动作数和根候选覆盖数。纯网络评估没有搜索统计。'
        '因此不能从归档比较「结束」和「出击／出牌」的后续收益。'
    )
    (out / 'report.md').write_text(render(summary, failures, coverage), encoding='utf-8')
    print('question', summary['question_ap2_with_attack_or_play'], summary['question_labels'], flush=True)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
