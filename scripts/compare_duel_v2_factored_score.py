#!/usr/bin/env python3
"""Screen a factored action score against the coupled scorer on archived games.

Same episodes, same four policy passes, and the same two predeclared learning
rates. Confirmation seeds are not scored. Overall play rate is not a pass.
This does not start self-play, export a model, or open formal training.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.modules.card_game.rl.factored_action_score import (  # noqa: E402
    SELECTION_KEYS, FactoredActionScore, factored_beats, scheme_passes, screening_counts,
)
from scripts.diagnose_duel_v2_policy_margin import (  # noqa: E402
    aggregate, attach, fit_policy, score_rows, sha256_file,
)
from scripts.fit_duel_v2_policy_recovery import holdout_seeds, load_games, split_games, write  # noqa: E402

KEYS = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')
LEARNING_RATES = (1.8e-4, 1.5e-4)
LABEL = {
    'starter': '创生', 'weave-rush': '覆纹', 'quick-rush': '快攻', 'zhenhong': '真红', 'murk': '浊燃',
}
# Coupled replay of offline-diagnosis/margin-by-lr.json. A mismatch means this
# screen is not the same budget and must not be read as a pass or a fail.
ARCHIVED_PLAY_TOP1 = {
    ('zhenhong', 1.5e-4): 0.29328621908127206,
    ('zhenhong', 1.8e-4): 0.4717314487632509,
    ('starter', 1.8e-4): 0.09223057644110276,
    ('weave-rush', 1.8e-4): 0.8131672597864769,
    ('quick-rush', 1.8e-4): 0.96760710553814,
    ('murk', 1.8e-4): 0.9854817421909371,
}


def config_payload():
    return dict(
        question='Does separating action type from the within-type card or actor cut attack/end decisions called play and raise exact-card hits?',
        pass_rule=dict(
            attack_called_play='strictly fewer than the coupled scorer on the same rows',
            end_called_play='strictly fewer than the coupled scorer on the same rows',
            same_card='strictly more target-play rows whose argmax card name matches',
            play_rate='recorded only; not a pass criterion',
            collapse='play_top or end_top at or above 95% of scored rows fails',
            eligible_teams=list(SELECTION_KEYS),
            scheme='one predeclared learning rate must pass every eligible team',
            in_sample_teams=['quick-rush', 'murk'],
            in_sample_cannot_pass=True,
            zero_denominator_fails=True,
        ),
        score=dict(
            play='kind_head[play] + card_head[card]',
            other='kind_head[kind] + actor_head[actor]',
            coupled_candidate_network='frozen and not added to the logit',
            new_heads='zero-initialized linears on the state trunk',
        ),
        learning_rates=list(LEARNING_RATES),
        passes=4, batch=128, minimum_batch=32, seed=20260924, self_imitation=0, optimizer='Adam',
        selection_split='original holdout from fit_duel_v2_policy_recovery.split_games every=5',
        confirmation_scored=False,
        independent_confirmation='尚无新的独立确认数据',
        serving_export=False, formal_run=False, cuda=False,
    )


def load_initial(key, initial):
    import torch
    from app.modules.card_game.rl.cross_runtime import create_network
    manifest = json.loads((initial / f'{key}.json').read_text(encoding='utf-8'))
    net = create_network(int(manifest['hidden']), 'cpu')
    own = net.state_dict()
    with np.load(initial / f'{key}.npz', allow_pickle=False) as arrays:
        for name in own:
            if name not in arrays.files:
                continue
            value = torch.from_numpy(arrays[name].copy())
            if own[name].shape != value.shape:
                raise ValueError(f'{key} {name} shape mismatch')
            own[name] = value
    net.load_state_dict(own)
    return net


def frozen_digest(net):
    digest = hashlib.sha256()
    for name, parameter in net.named_parameters():
        if 'cand_net' in name or name.endswith('score.weight') or name.endswith('score.bias') or '.score.' in name:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_factored(key, learning_rate, passes, train_rows, initial):
    import torch
    from app.modules.card_game.rl.outcome_runtime import update
    net = load_initial(key, initial)
    scored = FactoredActionScore(net)
    before = frozen_digest(net)
    opt = torch.optim.Adam([parameter for parameter in scored.parameters() if parameter.requires_grad], lr=learning_rate)
    policy = [row for row in train_rows if 'pi' in row]
    seen = np.zeros(len(policy), dtype=np.int32)
    losses = []
    rng = np.random.default_rng(20260924)
    for _ in range(passes):
        order = rng.permutation(len(policy))
        for start in range(0, len(policy), 128):
            batch_index = order[start:start + 128]
            if len(batch_index) < 32:
                continue
            seen[batch_index] += 1
            stats = update(scored, opt, [policy[int(index)] for index in batch_index], self_imitation=0)
            losses.append(dict(policy_loss=stats['policy_loss'], value_loss=stats['value_loss'],
                               policy_samples=stats['policy_samples'], samples=stats['samples']))
    reuse = dict(rows=int(seen.size), min=int(seen.min()) if seen.size else 0,
                 max=int(seen.max()) if seen.size else 0,
                 mean=float(seen.mean()) if seen.size else 0, never=int(np.sum(seen == 0)))
    heads = {name: float(parameter.detach().abs().mean())
             for name, parameter in scored.named_parameters()
             if name.startswith(('kind_head', 'card_head', 'actor_head'))}
    return scored, losses, reuse, dict(frozen_unchanged=frozen_digest(net) == before, head_mean_abs=heads)


def pack(records, losses, reuse):
    counts = screening_counts(records)
    summary = aggregate(records)
    kept = {name: summary.get(name) for name in (
        'n', 'play_top1', 'target_play_top1', 'same_kind', 'same_card_among_both_play', 'both_play',
        'mean_cross_entropy', 'mean_play_mass', 'mean_target_play_mass')}
    return dict(
        counts=counts, aggregate=kept, reuse=reuse,
        mean_policy_loss=float(np.mean([item['policy_loss'] for item in losses])) if losses else None,
        mean_value_loss=float(np.mean([item['value_loss'] for item in losses])) if losses else None,
    )


def reasons(cell):
    if not cell['eligible']:
        return ['原选择集没有策略行，拟合集内成绩不能放行']
    found = []
    base, new = cell['baseline']['counts'], cell['factored']['counts']
    if new['attack_called_play'] >= base['attack_called_play']:
        found.append('攻击被改判成出牌的次数没有下降')
    if new['end_called_play'] >= base['end_called_play']:
        found.append('结束回合被改判成出牌的次数没有下降')
    if new['same_card'] <= base['same_card']:
        found.append('目标出牌时同一张牌的命中次数没有上升')
    if new['play_top'] >= 0.95 * new['n']:
        found.append('argmax 塌成永远出牌')
    if new['end_top'] >= 0.95 * new['n']:
        found.append('argmax 塌成永远结束回合')
    if not found and not cell['beats']:
        found.append('对照没有同时满足三项放行条件')
    return found


def rate_text(count, total):
    if not total:
        return f'{count}/0'
    return f'{count}/{total}（{count / total:.1%}）'


def render_report(payload):
    lines = [
        '# 分层动作打分的离线对照',
        '',
        '这一步只在已归档的 `five-cross-20260924` 训练 episode 上筛选打分方式。没有新对局，没有导出模型，没有 CUDA 短测，正式长训保持关闭。确认种子没有参与打分，也没有参与取舍。归档里没有未使用的根种子，因此**尚无新的独立确认数据**。',
        '',
        '## 放行规则',
        '',
        '规则在第一次优化之前写入 `config.json`。同一学习率、同样 4 遍策略行、批大小 128、丢弃短于 32 的尾批、Adam、自模仿系数 0。分层头把动作类别和类别内的选择分开：出牌是类别分加上该牌的分，其他动作是类别分加上行动者的分。原来的候选网络冻结，并且不加进这个 logit。',
        '',
        '一组对照放行，必须同时做到：目标为攻击时被改判成出牌的次数下降，目标为结束回合时被改判成出牌的次数下降，目标为出牌时 argmax 打中同一张牌的次数上升。总体出牌比例只记录，不作为放行依据。出牌或结束回合的 argmax 达到该批决策的 95% 也失败。创生、覆纹、真红必须在同一个预声明学习率上同时通过。快攻和浊燃的原选择集没有行，只记拟合集内成绩，不能放行。',
        '',
        '## 对照',
        '',
        '| 构筑 | 学习率 | 评分集 | 攻击被改判成出牌 | 结束回合被改判成出牌 | 同一张牌命中 | 出牌 top-1 | 是否优于原模型 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for cell in payload['cells']:
        base, new = cell['baseline']['counts'], cell['factored']['counts']
        lines.append(
            f"| {LABEL[cell['key']]} | {cell['learning_rate']} | {cell['scored_split']} | "
            f"{rate_text(base['attack_called_play'], base['attack_n'])} → {rate_text(new['attack_called_play'], new['attack_n'])} | "
            f"{rate_text(base['end_called_play'], base['end_n'])} → {rate_text(new['end_called_play'], new['end_n'])} | "
            f"{rate_text(base['same_card'], base['target_play_n'])} → {rate_text(new['same_card'], new['target_play_n'])} | "
            f"{base['play_top'] / base['n']:.1%} → {new['play_top'] / new['n']:.1%} | "
            f"{'是' if cell['beats'] else '否'} |"
        )
    lines.extend(['', '## 结论', ''])
    if payload['scheme_passes']:
        lines.append('分层打分在预声明的同一学习率上，让创生、覆纹、真红三项计数同时优于原耦合打分。这只说明该方案可以进入下一步采集，不能把它写入训练器，也不能当作五套确认。')
    else:
        lines.append('分层打分没有在同一个预声明学习率上让创生、覆纹、真红同时减少两类误判并提高同一张牌命中。这个头不接入训练器，也不为它采集新对局。')
    lines.append('')
    traded = []
    for cell in payload['cells']:
        if not cell['eligible']:
            continue
        base, new = cell['baseline']['counts'], cell['factored']['counts']
        if new['same_card'] > base['same_card'] and (
                new['attack_called_play'] > base['attack_called_play'] or new['end_called_play'] > base['end_called_play']):
            traded.append(f"{LABEL[cell['key']]} {cell['learning_rate']}")
    if traded:
        lines.append('留出三套里，同一张牌命中上升的对照同时让攻击或结束回合被改判成出牌的次数上升：' + '、'.join(traded) + '。命中变多来自更多决策被排成出牌，不是两类误判一起下降。')
        lines.append('')
    lines.append('快攻和浊燃没有原选择集行，表中的拟合集内数字不能补成留出结论。')
    lines.append('')
    lines.append('下一步若要确认一个已经通过本筛选的方案，需要新采集覆盖五套的独立对局。五套确认通过之后才做 CUDA 短测和吞吐预检。本次没有采集新对局，正式长训继续关闭。')
    lines.append('')
    if payload['replay_ok']:
        lines.append('原耦合模型在相同预算下的出牌 top-1 与已归档的 `margin-by-lr.json` 一致，这次对照使用的是同一条拟合。')
    else:
        lines.append('原耦合模型的重放和已归档 `margin-by-lr.json` 不一致。这次数字不能用来放行或否定方案，需要先核对拟合是否漂移。')
    lines.extend(['', '## 分项', ''])
    for cell in payload['cells']:
        why = '；'.join(cell['reasons']) if cell['reasons'] else '三项计数同时改善，且没有塌缩'
        base, new = cell['baseline']['aggregate'], cell['factored']['aggregate']
        lines.append(
            f"- {LABEL[cell['key']]} {cell['learning_rate']}：{why}。"
            f"类别一致率 {base['same_kind']:.1%} → {new['same_kind']:.1%}，"
            f"双方都出牌时的同一张牌比例 {base['same_card_among_both_play']:.1%} → {new['same_card_among_both_play']:.1%} "
            f"（双方都出牌 {base['both_play']} → {new['both_play']}）。"
            f"冻结的候选网络{'未变' if cell['factored_probe']['frozen_unchanged'] else '发生了变化'}。"
        )
    lines.extend(['', '## 仍待验证', ''])
    for item in payload['open_questions']:
        lines.append(f'- {item}')
    lines.append('')
    return '\n'.join(lines)


def main():
    episodes = Path('artifacts/rl-evals/five-cross-20260924/formal/episodes')
    initial = Path('artifacts/rl-evals/five-cross-20260924/formal/initial')
    out = Path('artifacts/rl-evals/policy-recovery-20260924/offline-diagnosis/factored-score')
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'config.json', config_payload())
    write(out / 'source-sha.json', dict(files={
        str(path): sha256_file(ROOT / path) for path in (
            'app/modules/card_game/rl/factored_action_score.py',
            'scripts/compare_duel_v2_factored_score.py',
            'scripts/diagnose_duel_v2_policy_margin.py',
            'app/modules/card_game/rl/outcome_runtime.py',
            'app/modules/card_game/rl/cross_runtime.py',
        )}, initial_zhenhong=sha256_file(initial / 'zhenhong.npz')))
    games = load_games(episodes)
    fit_games, selection_games, selection_seeds = split_games(games)
    confirmation_seeds = holdout_seeds((game['seed'] for game in fit_games), every=5)
    saved = json.loads(Path('artifacts/rl-evals/policy-recovery-20260924/calibration-plan.json').read_text())
    if sorted(confirmation_seeds) != sorted(saved['confirmation_seeds']):
        raise SystemExit('Confirmation seed list does not match the already used calibration plan')
    trainable = [game for game in fit_games if game['seed'] not in confirmation_seeds]
    overlap = set(selection_seeds) & confirmation_seeds & {game['seed'] for game in trainable}
    if overlap:
        raise SystemExit('Selection or confirmation seeds leaked into the fit')
    train_rows = attach(trainable)
    selection_rows = attach(selection_games)
    by_key_train = {key: [row for row in train_rows if row['key'] == key] for key in KEYS}
    by_key_selection = {key: [row for row in selection_rows if row['key'] == key] for key in KEYS}
    write(out / 'split-audit.json', dict(
        fit_episodes=len(trainable),
        fit_seeds=len({game['seed'] for game in trainable}),
        selection_episodes=len(selection_games),
        selection_seeds=len(selection_seeds),
        confirmation_seeds_excluded=len(confirmation_seeds),
        policy_rows={key: len(by_key_train[key]) for key in KEYS},
        selection_policy_rows={key: len(by_key_selection[key]) for key in KEYS},
    ))
    cells = []
    replay = []
    flags = {rate: {} for rate in LEARNING_RATES}
    for rate in LEARNING_RATES:
        for key in KEYS:
            print(f'fit {key} {rate}', flush=True)
            baseline_net, baseline_losses, baseline_reuse = fit_policy(key, rate, 4, by_key_train[key], initial)
            eligible = key in SELECTION_KEYS and bool(by_key_selection[key])
            scored_rows = by_key_selection[key] if by_key_selection[key] else by_key_train[key]
            scored_split = 'selection' if by_key_selection[key] else 'fit_in_sample'
            baseline_records = score_rows(baseline_net, scored_rows)
            expected = ARCHIVED_PLAY_TOP1.get((key, rate))
            played = aggregate(baseline_records)['play_top1']
            if expected is not None:
                matches = math.isclose(played, expected, rel_tol=0, abs_tol=1e-9)
                replay.append(dict(key=key, learning_rate=rate, archived=expected, replay=played, matches=matches))
                write(out / 'replay-check.json', replay)
                if not matches:
                    write(out / 'comparison.json', dict(cells=cells, replay=replay, scheme_passes=False, replay_ok=False))
                    raise SystemExit(f'Coupled replay drifted for {key} at {rate}: {played} vs {expected}')
            factored_net, factored_losses, factored_reuse, probe = fit_factored(key, rate, 4, by_key_train[key], initial)
            factored_records = score_rows(factored_net, scored_rows)
            baseline = pack(baseline_records, baseline_losses, baseline_reuse)
            factored = pack(factored_records, factored_losses, factored_reuse)
            beats = bool(eligible and factored_beats(baseline['counts'], factored['counts']) and probe['frozen_unchanged'])
            if eligible:
                flags[rate][key] = beats
            cell = dict(
                key=key, learning_rate=rate, scored_split=scored_split, eligible=eligible, beats=beats,
                baseline=baseline, factored=factored, factored_probe=probe,
            )
            cell['reasons'] = reasons(cell)
            cells.append(cell)
            print(key, rate, 'beats' if beats else 'no',
                  f"attack {baseline['counts']['attack_called_play']}->{factored['counts']['attack_called_play']}",
                  f"end {baseline['counts']['end_called_play']}->{factored['counts']['end_called_play']}",
                  f"card {baseline['counts']['same_card']}->{factored['counts']['same_card']}",
                  flush=True)
            write(out / 'comparison.json', dict(cells=cells, replay=replay, scheme_passes=False, replay_ok=all(item['matches'] for item in replay)))
    passed = scheme_passes(flags) and all(item['matches'] for item in replay)
    open_questions = [
        '尚无新的独立确认数据。已使用的确认种子没有参与这次筛选。',
        '快攻和浊燃没有原选择集策略行，不能用拟合集内成绩代替留出。',
        '类别头和卡牌头仍读同一个状态主干，这次没有把主干拆开。',
        '通过本筛选之后，才采集覆盖五套的新独立对局做确认。',
        '五套确认通过之后才做 CUDA 短测和吞吐预检。正式长训继续关闭。',
    ]
    payload = dict(cells=cells, replay=replay, scheme_passes=passed, replay_ok=all(item['matches'] for item in replay),
                   flags={str(rate): flags[rate] for rate in flags}, open_questions=open_questions)
    write(out / 'comparison.json', payload)
    write(out / 'open-questions.json', open_questions)
    (out / 'report.md').write_text(render_report(payload), encoding='utf-8')
    print('scheme_passes', passed, flush=True)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
