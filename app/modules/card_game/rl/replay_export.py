"""Convert RL matches into website-playable public replay JSON."""
from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha1
from pathlib import Path
from typing import Any

from .adapter import DuelAdapter
from .backend import V2Backend
from .encoding import V2Encoder
from .policies import VisibleEngineRulePolicy
from .transfer import default_training_deck

from app.modules.card_game.engine.duel_v2.replay_log import (
    assemble_replay_game,
    build_match_payload,
    leak_markers,
)


def replay_code_for(*parts: Any) -> str:
    digest = sha1('|'.join(str(part) for part in parts).encode('utf-8')).hexdigest()
    return ('R' + digest[:5]).upper()


def payload_from_adapter(
    adapter: DuelAdapter, *,
    agent: str = 'trained',
    opponent: str = 'rule',
    seed: int = 0,
    learning_side: str = 'a',
) -> dict[str, Any]:
    if adapter._opening_state is None or adapter._state is None:
        raise ValueError('Adapter has no opening or final state to export')
    name_a = '训练策略' if learning_side == 'a' else '规则AI'
    name_b = '规则AI' if learning_side == 'a' else '训练策略'
    if agent == 'random':
        if learning_side == 'a':
            name_a = '随机策略'
        else:
            name_b = '随机策略'
    elif agent == 'rule':
        if learning_side == 'a':
            name_a = '规则AI'
        else:
            name_b = '规则AI'
    code = replay_code_for(agent, opponent, seed, learning_side)
    payload = build_match_payload(
        adapter._opening_state, adapter._state,
        room_code=code, name_a=name_a, name_b=name_b,
        source='rl-eval', learning_side=learning_side,
    )
    payload['seed'] = seed
    payload['agent'] = agent
    payload['opponent'] = opponent
    return payload


def write_replay_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + '\n', encoding='utf-8')
    return path


def _action_index(legal: tuple[dict, ...], saved: dict) -> int:
    payload = saved.get('action') if isinstance(saved.get('action'), dict) else saved
    for index, action in enumerate(legal):
        if action == payload:
            return index
    raise ValueError('Saved learner action is not in the reconstructed legal set')


def replay_from_eval_record(record: dict[str, Any], *, deck: dict[str, Any] | None = None) -> dict[str, Any]:
    from .budget import TrainBudget, CountingBackend

    learning_side = str(record.get('learning_side') or 'a')
    seed = int(record.get('seed') or 0)
    recorded = record.get('deck') if isinstance(record.get('deck'), dict) else None
    if deck is not None:
        used = deepcopy(deck)
    elif recorded and recorded.get('character_ids') and recorded.get('card_ids'):
        from app.modules.card_game.content.duel_v2 import validate_deck
        used = validate_deck({
            'id': recorded.get('id') or 'eval',
            'name': recorded.get('name') or 'eval',
            'character_ids': list(recorded['character_ids']),
            'card_ids': list(recorded['card_ids']),
        })
    else:
        used = default_training_deck()
    budget = TrainBudget(max_raw_steps=100_000, max_games=8, max_seconds=600)
    adapter = DuelAdapter(
        CountingBackend(V2Backend(), budget), V2Encoder(),
        learning_side=learning_side, opponent=VisibleEngineRulePolicy(),
        max_game_actions=1000, max_opponent_actions=100, allow_concede=False,
        capture_opening=True,
    )
    from .capability import matchup_decks
    if record.get('opponent_policy', 'VisibleEngineRulePolicy') != 'VisibleEngineRulePolicy':
        raise ValueError('Replay export requires the recorded frozen opponent policy')
    decks = matchup_decks(learning_side=learning_side, learner_deck=used,
                          opponent_deck=record.get('opponent_deck') or used)
    options = {'decks': decks}
    if record.get('first_side') in ('a', 'b'):
        options['first_side'] = record['first_side']
    adapter.reset(seed=seed, **options)
    for saved in record.get('actions') or []:
        decision = adapter.decision()
        if adapter._outcome().terminated or not decision.actions:
            break
        index = _action_index(decision.actions, saved)
        adapter.step(index, revision=decision.revision)
    payload = payload_from_adapter(
        adapter,
        agent=str(record.get('agent') or 'trained'),
        opponent=str(record.get('opponent') or 'rule'),
        seed=seed,
        learning_side=learning_side,
    )
    payload['reconstructed'] = True
    payload['winner'] = record.get('winner') or payload.get('winner')
    return payload


def export_eval_games(source: Path, output: Path | None = None, *, deck: dict[str, Any] | None = None) -> list[Path]:
    source = Path(source)
    if source.is_dir():
        jsonl = source / 'eval_games.jsonl'
        output = output or (source / 'replays')
    else:
        jsonl = source
        output = output or (jsonl.parent / 'replays')
    if not jsonl.is_file():
        raise ValueError(f'No eval_games.jsonl at {jsonl}')
    written: list[Path] = []
    for line in jsonl.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        payload = replay_from_eval_record(record, deck=deck)
        name = f"{payload['room_code']}-{record.get('agent') or 'agent'}-{record.get('learning_side') or 'a'}.json"
        written.append(write_replay_json(output / name, payload))
    return written


def public_file_ok(payload: dict[str, Any]) -> None:
    leaks = leak_markers(payload)
    if leaks:
        raise ValueError('Replay file would leak hidden keys: ' + ', '.join(leaks[:8]))
    views = payload.get('views') or {}
    for side, view in views.items():
        assemble_replay_game(view['opening_board'], view.get('events') or [])
        if side not in ('a', 'b'):
            raise ValueError('Replay views must be a/b')
