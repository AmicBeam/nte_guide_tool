"""On-demand offline policy analysis and selected public replay export.

This module never trains, never constructs an optimizer, and never writes
training logs. Games use the official pure-Python V2 engine with first-turn
draw and escalation enabled. Policies see only public views plus legal actions.
"""
from __future__ import annotations

import json
import re
import itertools
from random import Random
from secrets import randbits
import time
from copy import deepcopy
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Mapping

from app.modules.card_game.content.duel_v2 import CARDS, STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import acting_side, apply_action, new_game, observe
from app.modules.card_game.engine.duel_v2.replay_log import build_match_payload, leak_markers
from app.modules.card_game.rl.capability import (
    PUBLIC_KIT_CARDS,
    PUBLIC_CHARACTERS,
    assert_public_encoder_deck,
    is_exact_preset,
    sample_public_deck,
)
from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
from app.modules.card_game.rl.replay_export import public_file_ok, write_replay_json


PRESETS = ('starter', 'weave-rush')
ANALYSIS_CARD_IDS=(*PUBLIC_KIT_CARDS,*(cid for cid,c in CARDS.items()
                   if c.get('derived') and c.get('character_id') in PUBLIC_CHARACTERS))
Policy = Callable[[dict[str, Any], tuple[dict[str, Any], ...]], int]

ANALYSIS_LABEL = (
    'On-demand offline analysis of extra bounded official games. '
    'Small sample. Per-card usage/win numbers are associations, not causal balance.'
)
HARD_MAX_PAIRS = 64
HARD_MAX_ACTIONS = 2000
HARD_MAX_SECONDS = 600
DEFAULT_SEED_BASE = None


def _preset_deck(deck_id: str) -> dict[str, Any]:
    if deck_id not in PRESETS:
        raise ValueError(f'Learner deck must be starter or weave-rush, got {deck_id!r}')
    deck = next(item for item in STARTER_DECKS if item['id'] == deck_id)
    payload = assert_public_encoder_deck(deck)
    if not is_exact_preset(payload, deck_id):
        raise ValueError(f'Learner deck {deck_id} is not the official public preset')
    return payload


def resolve_numeric_model_dir(path: str | Path, *, deck_id: str | None = None) -> tuple[Path, str]:
    """Locate a numeric .npz + adjacent manifest. Never loads weights or .pt files."""
    model_path = Path(path).expanduser()
    suffix = model_path.suffix.lower()
    if suffix == '.pt':
        raise ValueError(
            'PyTorch .pt checkpoints are not supported. '
            'Export numeric CPU files with scripts/export_duel_v2_public.py first.'
        )
    if suffix == '.npz':
        resolved_deck = model_path.stem
        if deck_id is not None and deck_id != resolved_deck:
            raise ValueError(f'Model file {model_path.name} does not match deck {deck_id}')
        manifest = model_path.with_suffix('.json')
        if not model_path.is_file():
            raise ValueError(f'Missing numeric model file: {model_path}')
        if not manifest.is_file():
            raise ValueError(f'Missing adjacent manifest: {manifest.name}')
        return model_path.parent, resolved_deck
    if model_path.is_dir():
        if deck_id is None:
            raise ValueError('A directory model path requires an explicit deck id')
        npz = model_path / f'{deck_id}.npz'
        manifest = model_path / f'{deck_id}.json'
        if not npz.is_file() or not manifest.is_file():
            raise ValueError(f'Directory must contain {deck_id}.npz and adjacent {deck_id}.json')
        return model_path, deck_id
    raise ValueError('Model path must be a numeric .npz file or a directory containing one')


def load_public_deck(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding='utf-8'))
    else:
        payload = dict(source)
    return assert_public_encoder_deck(payload)


def frozen_model_policy(model):
    """Wrap FrozenModel.select_public_action(view, actions) -> index. Do not recode the model."""
    selector = getattr(model, 'select_public_action', None)
    if not callable(selector):
        raise TypeError('FrozenModel.select_public_action(view, actions) is required')

    def policy(view, actions):
        return selector(view, actions)

    manifest=getattr(model,'manifest',{})
    policy.model_metadata={'sha256':getattr(model,'version',None),
                           'schema':getattr(model,'schema',None),
                           'rule_hash':manifest.get('runtime_rule_hash') or (manifest.get('rule_identity') or {}).get('rule_hash'),
                           'deck':manifest.get('deck'),'opening_policy':manifest.get('opening_policy','keep_all'),
                           'capability':(manifest.get('capability') or {}).get('kind','legacy_exact_mirror')}
    return policy


def _empty_record() -> dict[str, int]:
    return {'games': 0, 'win': 0, 'loss': 0, 'draw': 0, 'error': 0, 'truncated': 0}


def _tally(record: dict[str, int], game: Mapping[str, Any]) -> None:
    record['games'] += 1
    if game.get('error'):
        record['error'] += 1
        record['truncated'] += 1
        return
    if game.get('truncated') and not game.get('terminated'):
        record['truncated'] += 1
        return
    winner = game.get('winner')
    if winner == 'a':
        record['win'] += 1
    elif winner == 'b':
        record['loss'] += 1
    else:
        record['draw'] += 1


def _legal_from_view(view: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = view.get('legal_actions') or []
    actions = [deepcopy(entry.get('action')) for entry in entries if isinstance(entry, dict)]
    actions = [action for action in actions if isinstance(action, dict) and action.get('type')]
    playable = [action for action in actions if action.get('type') != 'concede']
    return playable or actions


def _choose(policy: Policy, view: dict[str, Any], actions: list[dict[str, Any]]) -> int:
    if not actions:
        raise ValueError('Policy received no legal actions')
    index = policy(deepcopy(view), tuple(deepcopy(action) for action in actions))
    if isinstance(index, bool) or not isinstance(index, Integral):
        raise ValueError('Policy must return an integral candidate index')
    chosen = int(index)
    if not 0 <= chosen < len(actions):
        raise ValueError('Policy returned an out-of-range action index')
    return chosen


def _card_id_from_view(view: Mapping[str, Any], side: str, instance_id: str | None) -> str | None:
    if not instance_id:
        return None
    hand = ((view.get('sides') or {}).get(side) or {}).get('hand') or []
    for card in hand:
        if isinstance(card, dict) and card.get('instance_id') == instance_id:
            return card.get('card_id')
    pending = view.get('pending_choice') or {}
    for choice in pending.get('choices') or []:
        card = (choice or {}).get('card') if isinstance(choice, dict) else None
        if isinstance(choice, dict) and choice.get('id') == instance_id:
            if isinstance(card, dict) and card.get('card_id'):
                return card.get('card_id')
            return choice.get('card_id')
        if isinstance(card, dict) and card.get('instance_id') == instance_id:
            return card.get('card_id')
    return None


def _append_action(recorded, *, side, action, card_id=None):
    payload = deepcopy(dict(action))
    item = {
        'seq': len(recorded) + 1,
        'side': side,
        'type': payload.get('type'),
        'action': payload,
    }
    if card_id:
        item['card_id'] = card_id
    recorded.append(item)
    return item


def _opponent_team_key(deck: Mapping[str, Any]) -> str:
    return ','.join(sorted(str(item) for item in (deck.get('character_ids') or [])))


def _public_payload(opening, final, *, game_id):
    payload = build_match_payload(
        opening, final,
        room_code=game_id,
        name_a='分析策略',
        name_b='对手策略',
        source='offline-analysis',
        learning_side='a',
    )
    for key in ('seed', 'rng', 'agent', 'opponent'):
        payload.pop(key, None)
    payload['visibility']='spectator_public_v1'
    # Existing participant replays retain that participant's own hand. A shareable
    # offline recording is a spectator view: both private hands stay hidden.
    def spectator(value):
        if isinstance(value,list):return [spectator(item) for item in value]
        if not isinstance(value,dict):return value
        result={key:spectator(item) for key,item in value.items() if key!='policy_state'}
        if isinstance(value.get('hand'),list):
            result['hand']=[spectator(card) if isinstance(card,dict) and
                            (card.get('copy') or card.get('revealed') or card.get('derived'))
                            else {'hidden':True} for card in value['hand']]
        if isinstance(value.get('pending_choice'),dict):
            result['pending_choice']['choices']=[]
            result['pending_choice']['prompt']='正在选择'
        return result
    raw_by_seq={e['seq']:e for e in final.get('events',[])}
    from app.modules.card_game.engine.duel_v2.replay_log import public_replay_event
    for viewer,view in payload['views'].items():
        view['opening_board']=spectator(view['opening_board'])
        for event in view['events']:
            raw=raw_by_seq.get(event.get('seq'),{})
            if raw.get('private_side'):
                public_raw={k:v for k,v in raw.items() if k not in ('private_card','private_side','card')}
                visible=public_replay_event(public_raw,viewer)
                event.pop('card',None)
                event['text']=visible['text']
            if event.get('type')=='mulligan':event.pop('card_ids',None)
            if isinstance(event.get('patch'),dict):event['patch']=spectator(event['patch'])
    public_file_ok(payload)
    leaks = leak_markers(payload)
    if leaks:
        raise ValueError('Replay payload would leak hidden keys: ' + ', '.join(leaks[:8]))
    dumped = json.dumps(payload, ensure_ascii=False)
    for marker in ('"seed"', '"rng"', '"deck":'):
        if marker in dumped:
            raise ValueError(f'Public replay leaked {marker}')
    return payload


def _card_usage_bucket():
    return {
        card_id: {
            'card_id': card_id,
            'name': (CARDS.get(card_id) or {}).get('name') or card_id,
            'plays': 0,
            'games_used': 0,
            'wins_when_used': 0,
            'losses_when_used': 0,
            'draws_when_used': 0,
            'truncated_when_used': 0,
            'errors_when_used': 0,
        }
        for card_id in ANALYSIS_CARD_IDS
    }


def _play_game(*, policy, opponent_policy, learner_deck, opponent_deck, seed, first_side, max_actions, deadline, clock):
    decks = {'a': deepcopy(dict(learner_deck)), 'b': deepcopy(dict(opponent_deck))}
    info = {
        'seed': seed,
        'first_side': first_side,
        'position': 'first' if first_side == 'a' else 'second',
        'learner_deck_id': learner_deck.get('id'),
        'opponent_deck': {
            'id': opponent_deck.get('id'),
            'name': opponent_deck.get('name'),
            'character_ids': list(opponent_deck.get('character_ids') or []),
            'card_ids': list(opponent_deck.get('card_ids') or []),
        },
        'actions': [],
        'learner_card_ids': [],
        'opponent_card_ids': [],
        'winner': None,
        'reason': '',
        'terminated': False,
        'truncated': False,
        'truncation_reason': '',
        'error': None,
        'event_count': 0,
    }
    try:
        state = new_game(
            seed=seed,
            decks=decks,
            first_side=first_side,
            skip_mulligan=False,
            escalation=True,
        )
    except Exception as exc:  # noqa: BLE001
        info['error'] = f'{type(exc).__name__}: {exc}'
        info['truncated'] = True
        info['truncation_reason'] = 'exception'
        return info
    opening = deepcopy(state)
    from .report_telemetry import GameTelemetry
    telemetry = GameTelemetry(state)
    recorded = info['actions']
    try:
        while state.get('phase') != 'finished':
            if deadline is not None and clock() >= deadline:
                info['truncated'] = True
                info['truncation_reason'] = 'max_seconds'
                break
            if len(recorded) >= max_actions:
                info['truncated'] = True
                info['truncation_reason'] = 'max_actions'
                break
            side = acting_side(state)
            if side not in ('a', 'b'):
                raise ValueError('Nonterminal game has no acting side')
            view = observe(state, side)
            actions = _legal_from_view(view)
            telemetry.before(state,side,actions)
            chooser = policy if side == 'a' else opponent_policy
            index = _choose(chooser, view, actions)
            action = actions[index]
            card_id = None
            if action.get('type') == 'play_card':
                card_id = _card_id_from_view(view, side, action.get('card_id'))
            elif action.get('type') == 'choose':
                card_id = _card_id_from_view(view, side, action.get('choice_id'))
            _append_action(recorded, side=side, action=action, card_id=card_id)
            previous_seq=state.get('event_seq',0)
            previous_state=state
            state = apply_action(state, side, action)
            telemetry.after(previous_state,state,action,side)
            for event in state.get('events',[]):
                if event.get('seq',0)>previous_seq and event.get('type')=='play':
                    played=(event.get('card') or {}).get('card_id')
                    if played:
                        key='learner_card_ids' if event.get('side')=='a' else 'opponent_card_ids'
                        info[key].append(played)
        info['telemetry'] = telemetry.finish(state)
        info['winner'] = state.get('winner')
        info['reason'] = str(state.get('reason') or '')
        info['terminated'] = state.get('phase') == 'finished'
        info['event_count'] = int(state.get('event_seq') or 0)
        info['public_payload'] = _public_payload(opening, state, game_id='pending')
    except Exception as exc:  # noqa: BLE001
        info['error'] = f'{type(exc).__name__}: {exc}'
        info['truncated'] = True
        info['truncation_reason'] = info.get('truncation_reason') or 'exception'
        try:
            info['public_payload'] = _public_payload(opening, state, game_id='pending')
            info['event_count'] = int(state.get('event_seq') or 0)
        except Exception:
            info.pop('public_payload', None)
    return info


def _write_report_md(path: Path, summary: Mapping[str, Any]) -> None:
    model=summary.get('model') or {}
    lines=['# 离线对局分析','','## 本轮设置','',
           f"- 打牌模型标识：{summary['preset']}",
           f"- 本轮构筑：{summary['learner_build']['name']}；角色：{', '.join(summary['learner_build']['character_ids'])}",
           f"- 模型版本：{model.get('sha256') or '自定义策略（无权重版本）'}",
           f"- 对局：{summary['games']} 场；请求 {summary['pairs']} 对先后手配对。",
           '- 正式 Python 规则：先手首回合抽牌、白热化开启。',
           '- 这是额外评估，不更新模型参数。卡牌使用与胜负的关联不等于卡牌的因果强度。',
           '', '## 先后手结果','',
           '| 位置 | 尝试 | 胜 | 负 | 平 | 异常 | 截断 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for position,name in (('first','先手'),('second','后手')):
        r=summary['by_position'][position]
        lines.append(f"| {name} | {r['games']} | {r['win']} | {r['loss']} | {r['draw']} | {r['error']} | {r['truncated']} |")
    lines += ['', '## 对手编队','', '| 编队 | 场次 | 胜 | 负 | 平 | 异常 | 截断 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for team,r in summary['by_opponent_team'].items():
        lines.append(f"| {team} | {r['games']} | {r['win']} | {r['loss']} | {r['draw']} | {r['error']} | {r['truncated']} |")
    lines += ['', '## 双方卡牌使用','',
              '统计成功打出的卡牌，包括自动响应。学习方使用本轮指定构筑，出牌覆盖不以全卡池为目标；未使用不代表卡牌弱。',
              '“使用时胜局”按各自所属方胜负统计，同方同局同名牌仅计一场关联对局。', '',
              '| 卡牌 | AI 出牌次数 | AI 使用局数 | AI 使用时胜局 | 对手出牌次数 | 对手使用局数 | 对手使用时胜局 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for cid in ANALYSIS_CARD_IDS:
        a=summary['card_usage'][cid];b=summary['opponent_card_usage'][cid]
        lines.append(f"| {cid} · {a['name']} | {a['plays']} | {a['games_used']} | {a['wins_when_used']} | {b['plays']} | {b['games_used']} | {b['wins_when_used']} |")
    lines += ['', '## 对局索引','',
              '| game_id | 先后手 | 胜方 | 结束 | 截断 | 异常 | 操作数 | 公开录像 |',
              '| --- | --- | --- | --- | --- | --- | ---: | --- |']
    for g in summary['games_index']:
        replay=g.get('public_replay') or ''
        link=f'[JSON]({replay})' if replay else '无'
        error=str(g.get('error') or '').replace('|','/')
        lines.append(f"| {g['game_id']} | {g['position']} | {g.get('winner') or ''} | {g['terminated']} | {g['truncated']} | {error} | {g['action_count']} | {link} |")
    lines += ['', '原始种子、构筑和操作记录保存在 raw/；public/ 录像不包含暗牌或牌序。',
              '可通过 --export-game 选择一局导出；导出直接读取已记录的公开录像，不重新模拟。', '']
    path.write_text('\n'.join(lines),encoding='utf-8')


def run_offline_analysis(
    output: str | Path,
    policy: Policy,
    *,
    deck: str = 'starter',
    opponent_policy: Policy | None = None,
    opponent_deck: Mapping[str, Any] | None = None,
    learner_deck: Mapping[str, Any] | None = None,
    pairs: int = 1,
    max_seconds: float = 30,
    max_actions: int = 80,
    seed_base: int | None = DEFAULT_SEED_BASE,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    if not callable(policy):
        raise TypeError('policy must be callable')
    foe = opponent_policy or VisibleEngineRulePolicy()
    if not callable(foe):
        raise TypeError('opponent_policy must be callable')
    if not isinstance(pairs, int) or isinstance(pairs, bool) or not 1 <= pairs <= HARD_MAX_PAIRS:
        raise ValueError(f'pairs must be 1..{HARD_MAX_PAIRS}')
    if not isinstance(max_actions, int) or isinstance(max_actions, bool) or not 1 <= max_actions <= HARD_MAX_ACTIONS:
        raise ValueError(f'max_actions must be 1..{HARD_MAX_ACTIONS}')
    if not 0 < float(max_seconds) <= HARD_MAX_SECONDS:
        raise ValueError(f'max_seconds must be in (0, {HARD_MAX_SECONDS}]')
    baseline_deck = _preset_deck(deck)
    learner_deck = load_public_deck(learner_deck) if learner_deck is not None else baseline_deck
    seed_base = randbits(31) if seed_base is None else int(seed_base)
    explicit_opponent = load_public_deck(opponent_deck) if opponent_deck is not None else None
    output_dir = Path(output)
    if output_dir.exists():
        raise ValueError(f'output dir must be new, not clobber: {output_dir}')
    raw_dir = output_dir / 'raw'
    public_dir = output_dir / 'public'
    raw_dir.mkdir(parents=True)
    public_dir.mkdir(parents=True)

    ticker = clock or time.monotonic
    deadline = ticker() + float(max_seconds)
    usage = _card_usage_bucket()
    opponent_usage = _card_usage_bucket()
    by_position = {'first': _empty_record(), 'second': _empty_record()}
    by_team = {}
    games_index = []
    games_path = raw_dir / 'games.jsonl'
    (raw_dir/'metadata.json').write_text(json.dumps({'seed_base':seed_base,'pairs':pairs})+'\n',encoding='utf-8')
    teams=list(itertools.combinations(PUBLIC_CHARACTERS,4))
    Random(seed_base).shuffle(teams)

    for pair in range(pairs):
        if ticker()>=deadline:break
        seed = int(seed_base) + pair
        sampled = explicit_opponent or sample_public_deck(seed,character_ids=teams[pair%len(teams)])
        for first_side, position in (('a', 'first'), ('b', 'second')):
            if ticker()>=deadline:break
            game_id = f'{deck}-{pair:02d}-{position}'
            result = _play_game(
                policy=policy,
                opponent_policy=foe,
                learner_deck=learner_deck,
                opponent_deck=sampled,
                seed=seed,
                first_side=first_side,
                max_actions=max_actions,
                deadline=deadline,
                clock=ticker,
            )
            result['game_id'] = game_id
            result['pair'] = pair
            result['position'] = position
            payload = result.pop('public_payload', None)
            if payload is not None:
                payload['room_code'] = game_id
                public_file_ok(payload)
                public_path = public_dir / f'{game_id}.json'
                write_replay_json(public_path, payload)
                result['event_count'] = max(
                    int(result.get('event_count') or 0),
                    len(((payload.get('views') or {}).get('a') or {}).get('events') or []),
                )
                result['public_replay'] = str(public_path.relative_to(output_dir))
            raw_record = {
                'game_id': game_id,
                'pair': pair,
                'position': position,
                'seed': result['seed'],
                'first_side': first_side,
                'learner_deck_id': learner_deck.get('id'),
                'opponent_deck': result['opponent_deck'],
                'actions': result['actions'],
                'telemetry':result.get('telemetry'),
                'learner_card_ids': result['learner_card_ids'],
                'opponent_card_ids': result['opponent_card_ids'],
                'winner': result.get('winner'),
                'reason': result.get('reason'),
                'terminated': result.get('terminated'),
                'truncated': result.get('truncated'),
                'truncation_reason': result.get('truncation_reason'),
                'error': result.get('error'),
            }
            (raw_dir / f'{game_id}.json').write_text(
                json.dumps(raw_record, ensure_ascii=False) + "\n", encoding="utf-8")
            with games_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")

            team_key = _opponent_team_key(sampled)
            by_team.setdefault(team_key, _empty_record())
            _tally(by_position[position], result)
            _tally(by_team[team_key], result)
            used = {card_id for card_id in result['learner_card_ids'] if card_id in usage}
            for card_id in result['learner_card_ids']:
                if card_id in usage:
                    usage[card_id]['plays'] += 1
            for card_id in used:
                usage[card_id]['games_used'] += 1
                if result.get('error'):
                    usage[card_id]['errors_when_used'] += 1
                    usage[card_id]['truncated_when_used'] += 1
                elif result.get('truncated') and not result.get('terminated'):
                    usage[card_id]['truncated_when_used'] += 1
                elif result.get('winner') == 'a':
                    usage[card_id]['wins_when_used'] += 1
                elif result.get('winner') == 'b':
                    usage[card_id]['losses_when_used'] += 1
                else:
                    usage[card_id]['draws_when_used'] += 1
            for card_id in result['opponent_card_ids']:
                if card_id in opponent_usage:opponent_usage[card_id]['plays']+=1
            for card_id in set(result['opponent_card_ids']) & set(opponent_usage):
                row=opponent_usage[card_id];row['games_used']+=1
                if result.get('error'):row['errors_when_used']+=1;row['truncated_when_used']+=1
                elif result.get('truncated'):row['truncated_when_used']+=1
                elif result.get('winner')=='b':row['wins_when_used']+=1
                elif result.get('winner')=='a':row['losses_when_used']+=1
                else:row['draws_when_used']+=1
            games_index.append({
                'game_id': game_id,
                'pair': pair,
                'position': position,
                'first_side': first_side,
                'winner': result.get('winner'),
                'terminated': bool(result.get('terminated')),
                'truncated': bool(result.get('truncated')),
                'truncation_reason': result.get('truncation_reason') or '',
                'error': result.get('error'),
                'action_count': len(result.get('actions') or []),
                'final_turn':(result.get('telemetry') or {}).get('final_turn'),
                'action_types': sorted({item.get('type') for item in result.get('actions') or [] if item.get('type')}),
                'event_count': result.get('event_count') or 0,
                'opponent_team': list(sampled.get('character_ids') or []),
                'public_replay': result.get('public_replay'),
                'raw_log': str((raw_dir / f'{game_id}.json').relative_to(output_dir)),
            })

    played = [card_id for card_id in PUBLIC_KIT_CARDS if usage[card_id]['plays']]
    summary = {
        'ok': True,
        'label': ANALYSIS_LABEL,
        'preset': deck,
        'learner_build':deepcopy(learner_deck),
        'learner_is_preset':is_exact_preset(learner_deck,deck),
        'model':deepcopy(getattr(policy,'model_metadata',{'type':type(policy).__name__})),
        'opponent_model':deepcopy(getattr(foe,'model_metadata',{'type':type(foe).__name__})),
        'sampled_opponent':explicit_opponent is None,
        'complete':len(games_index)==pairs*2 and all(g['terminated'] and not g['truncated'] and not g['error'] for g in games_index),
        'pairs': pairs,
        'games': len(games_index),
        'max_actions': max_actions,
        'max_seconds': float(max_seconds),
        'engine': {
            'name': 'official-python',
            'first_turn_draw': True,
            'escalation': True,
            'skip_mulligan': False,
        },
        'by_position': by_position,
        'by_opponent_team': by_team,
        'card_usage': usage,
        'opponent_card_usage':opponent_usage,
        'coverage': {
            'public_characters': 6,
            'public_cards': len(PUBLIC_KIT_CARDS),
            'played': len(played),
            'opponent_played':sum(bool(opponent_usage[c]['plays']) for c in PUBLIC_KIT_CARDS),
            'derived_cards':len(ANALYSIS_CARD_IDS)-len(PUBLIC_KIT_CARDS),
            'both_sides_played':sum(bool(usage[c]['plays'] or opponent_usage[c]['plays']) for c in PUBLIC_KIT_CARDS),
            'unplayed': [card_id for card_id in PUBLIC_KIT_CARDS if card_id not in played],
        },
        'games_index': games_index,
        'caveats': [
            'Small sample; not a balance claim.',
            'Per-card wins_when_used is an association in this sample, not a causal card-strength result.',
            'Seeds and constructed decks are only in raw logs, not in shareable public replays.',
        ],
    }
    (output_dir / 'report.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report_md(output_dir / 'report.md', summary)
    return summary


def load_recorded_public_replay(source: str | Path, game_id: str) -> dict[str, Any]:
    if not re.fullmatch(r'[A-Za-z0-9_-]+',game_id):
        raise ValueError('Invalid game_id')
    root = Path(source).resolve()
    if not (root/'public').resolve().is_relative_to(root):
        raise ValueError('Public replay directory points outside the report')
    public_path = root / 'public' / f'{game_id}.json'
    if not public_path.is_file():
        report_path = root / 'report.json'
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding='utf-8'))
            match = next((row for row in report.get('games_index') or [] if row.get('game_id') == game_id), None)
            if match and match.get('public_replay'):
                public_path = root / match['public_replay']
        if not public_path.is_file():
            raise ValueError(f'No recorded public replay for {game_id} in {root}')
    if not public_path.resolve().is_relative_to((root/'public').resolve()):
        raise ValueError('Replay path is outside the recorded public directory')
    payload = json.loads(public_path.read_text(encoding='utf-8'))
    if payload.get('visibility')!='spectator_public_v1':
        raise ValueError('Run analysis again to obtain a spectator-public recording')
    if payload.get('seed') is not None:
        raise ValueError('Recorded public replay unexpectedly contains seed')
    public_file_ok(payload)
    return payload


def export_game_replay(source: str | Path, game_id: str, replay_output: str | Path) -> dict[str, Any]:
    """Copy a recorded already-public payload. No model load and no game rerun."""
    payload = load_recorded_public_replay(source, game_id)
    path = write_replay_json(Path(replay_output), payload)
    return {'ok': True, 'game_id': game_id, 'replay': str(path), 'event_count': payload.get('last_seq')}
