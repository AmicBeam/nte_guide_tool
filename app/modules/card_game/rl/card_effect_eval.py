"""Stochastic legal-action selfplay coverage baseline. Not RL training or balance proof."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from random import Random
import threading
import time
import traceback
from typing import Any, Callable

from app.modules.card_game.content.duel_v2.catalog import (
    CARD_ID_ORDER,
    CARDS,
    CHARACTER_ORDER,
    STARTER_DECK,
    validate_deck,
)

LABEL = (
    'stochastic legal-action selfplay coverage baseline; '
    'NOT RL-trained strength or balance proof'
)
DESIGN_VERSION = 'V2.4'
HARD_MAX_RAW_STEPS = 10000
HARD_MAX_GAMES = 128
HARD_MAX_SECONDS = 1800
PER_GAME_ACTIONS = 500
SIDES = ('a', 'b')
ACTION_WEIGHTS = {
    'play_card': 16, 'choose': 12, 'mulligan': 8, 'attack': 6,
    'ultimate': 6, 'cycle': 4, 'end_turn': 1, 'concede': 0,
}
REPO_ROOT = Path(__file__).resolve().parents[4]


def coverage_deck() -> dict[str, Any]:
    character_ids = list(STARTER_DECK['character_ids'])
    allowed = set(character_ids)
    return validate_deck({
        'id': 'coverage_all8', 'name': 'coverage_all8',
        'character_ids': character_ids,
        'card_ids': [card_id for card_id in CARD_ID_ORDER
                     if CARDS[card_id]['character_id'] in allowed and not CARDS[card_id].get('derived')],
    })


def starter_deck() -> dict[str, Any]:
    return validate_deck(deepcopy(STARTER_DECK))


def check_payload() -> dict[str, Any]:
    return {
        'ok': True, 'training_enabled': False, 'model': 0, 'training_updates': 0,
        'label': LABEL, 'rules_version': 'duel_v2', 'design_version': DESIGN_VERSION,
        'card_ids': list(CARD_ID_ORDER),
        'budget_caps': {
            'max_raw_steps': HARD_MAX_RAW_STEPS, 'max_games': HARD_MAX_GAMES,
            'max_seconds': HARD_MAX_SECONDS, 'per_game_actions': PER_GAME_ACTIONS,
        },
    }


def verify_manifest(path: Path, *, root: Path | None = None) -> int:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    root = Path(root or REPO_ROOT)
    if isinstance(payload.get('files'), dict):
        items = list(payload['files'].items())
    else:
        raw = payload.get('entries') or payload.get('files') or []
        items = [(item['path'], item.get('sha256') or item.get('digest')) for item in raw]
    for rel, expected in items:
        digest = sha256((root / rel).read_bytes()).hexdigest()
        if digest != str(expected).lower():
            raise ValueError(f'manifest SHA-256 mismatch: {rel}')
    return len(items)


def find_manifest(start: Path | None = None) -> Path | None:
    for candidate in (Path(start or Path.cwd()) / 'source-manifest.json', REPO_ROOT / 'source-manifest.json'):
        if candidate.is_file():
            return candidate
    return None


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def legal_action_dicts(observation: dict[str, Any]) -> list[dict[str, Any]]:
    return [item['action'] for item in observation.get('legal_actions') or []
            if isinstance(item, dict) and isinstance(item.get('action'), dict)]


def select_weighted_action(legal_actions: list[dict[str, Any]], rng: Random) -> dict[str, Any]:
    if not legal_actions:
        raise ValueError('policy received empty legal_actions')
    pool = [a for a in legal_actions if a.get('type') != 'concede'] or list(legal_actions)
    weights = [max(ACTION_WEIGHTS.get(str(a.get('type')), 1), 0) for a in pool]
    if sum(weights) <= 0:
        return deepcopy(pool[rng.randrange(len(pool))])
    return deepcopy(rng.choices(pool, weights=weights, k=1)[0])


def _zeros() -> dict[str, int]:
    return {card_id: 0 for card_id in CARD_ID_ORDER}


def _effect_zeros() -> dict[str, dict[str, int]]:
    return {card_id: {} for card_id in CARD_ID_ORDER}


def _visible_card(container: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(container, dict):
        return None
    card_id = container.get('card_id')
    if card_id not in CARD_ID_ORDER:
        return None
    return {
        'card_id': card_id, 'copy': bool(container.get('copy')),
        'instance_id': container.get('instance_id'),
    }


def _card_from_observation(observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    instance_id = action.get('card_id')
    if not instance_id:
        return None
    side = observation.get('viewer_side')
    for card in ((observation.get('sides') or {}).get(side) or {}).get('hand') or []:
        if isinstance(card, dict) and card.get('instance_id') == instance_id:
            return _visible_card(card)
    resolving = observation.get('resolving_card') or {}
    if resolving.get('instance_id') == instance_id:
        return _visible_card(resolving)
    return None


def _default_engine():
    from app.modules.card_game.engine.duel_v2 import acting_side, apply_action, new_game, observe

    class Bound:
        new_game = staticmethod(new_game)
        acting_side = staticmethod(acting_side)
        observe = staticmethod(observe)
        apply_action = staticmethod(apply_action)

    return Bound()


def apply_with_timeout(engine, state, side, action, timeout: float | None, clock) -> Any:
    if timeout is not None and timeout <= 0:
        raise TimeoutError('apply_action watchdog: no wall time remaining')
    if timeout is None:
        return engine.apply_action(state, side, action)
    holder: dict[str, Any] = {}

    def _run() -> None:
        try:
            holder['state'] = engine.apply_action(state, side, action)
        except Exception as exc:  # noqa: BLE001
            holder['error'] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError('apply_action watchdog timeout')
    if 'error' in holder:
        raise holder['error']
    return holder['state']


class CardEffectEvaluator:
    def __init__(
        self, output_dir: Path, *, engine=None, clock=None,
        max_raw_steps: int = HARD_MAX_RAW_STEPS, max_games: int = HARD_MAX_GAMES,
        max_seconds: int = HARD_MAX_SECONDS, per_game_actions: int = PER_GAME_ACTIONS,
        base_seed: int = 0, apply_timeout: float | None = None,
        rng_factory: Callable[[int], Random] | None = None,
    ) -> None:
        if not (1 <= max_raw_steps <= HARD_MAX_RAW_STEPS):
            raise ValueError('max_raw_steps exceeds hard cap')
        if not (1 <= max_games <= HARD_MAX_GAMES):
            raise ValueError('max_games exceeds hard cap')
        if not (1 <= max_seconds <= HARD_MAX_SECONDS):
            raise ValueError('max_seconds exceeds hard cap')
        if not (1 <= per_game_actions <= PER_GAME_ACTIONS):
            raise ValueError('per_game_actions exceeds hard cap')
        self.output_dir = Path(output_dir)
        self.engine = engine or _default_engine()
        self.clock = clock or time
        self.max_raw_steps = max_raw_steps
        self.max_games = max_games
        self.max_seconds = max_seconds
        self.per_game_actions = per_game_actions
        self.base_seed = base_seed
        self.apply_timeout = apply_timeout
        self.rng_factory = rng_factory or Random
        self.started = self.clock.monotonic()
        self.raw_steps = 0
        self.games_started = 0
        self.invalid_actions = 0
        self.exceptions = 0
        self.timeouts = 0
        self.card_uses = _zeros()
        self.copy_uses = _zeros()
        self.card_effect_events = _effect_zeros()
        self.event_counts: Counter[str] = Counter()
        self.winners = {'a': 0, 'b': 0, 'draw': 0, 'truncated': 0}
        self.by_first_side = {side: dict(self.winners) for side in SIDES}
        self.first_player = {'win': 0, 'loss': 0, 'draw': 0, 'truncated': 0}
        self.stopped_reason = ''
        self.error = None
        self.phase = 'starting'
        self.last_extra: dict[str, Any] = {}

    def elapsed(self) -> float:
        return max(0.0, self.clock.monotonic() - self.started)

    def coverage(self) -> dict[str, str]:
        return {
            card_id: 'OBSERVED' if self.card_effect_events[card_id] else 'NOT_COVERED'
            for card_id in CARD_ID_ORDER
        }

    def snapshot(self, **extra: Any) -> dict[str, Any]:
        payload = {
            'ok': self.error is None, 'phase': self.phase, 'pid': os.getpid(),
            'raw_steps': self.raw_steps, 'games_started': self.games_started,
            'elapsed_seconds': round(self.elapsed(), 4), 'model': 0, 'training_updates': 0,
            'label': LABEL, 'design_version': DESIGN_VERSION,
            'budget': {
                'max_raw_steps': self.max_raw_steps, 'max_games': self.max_games,
                'max_seconds': self.max_seconds, 'per_game_actions': self.per_game_actions,
            },
            'winners': dict(self.winners), 'by_first_side': deepcopy(self.by_first_side),
            'first_player': dict(self.first_player), 'card_uses': dict(self.card_uses),
            'copy_uses': dict(self.copy_uses), 'event_counts': dict(self.event_counts),
            'card_effect_events': deepcopy(self.card_effect_events), 'coverage': self.coverage(),
            'invalid_actions': self.invalid_actions, 'exceptions': self.exceptions,
            'timeouts': self.timeouts, 'stopped_reason': self.stopped_reason, 'error': self.error,
        }
        payload.update(extra)
        return payload

    def write_status(self, **extra: Any) -> None:
        atomic_write(self.output_dir / 'status.json', self.snapshot(**extra))

    def write_summary(self, **extra: Any) -> None:
        payload = self.snapshot(**extra)
        payload['note'] = (
            'card_uses counts play_card applications; observed effect events are separate and '
            'only attributed with a public card field or pending-choice continuation. '
            'NOT_COVERED means no attributed effect event. Proxy coverage only; not every '
            'mechanic was tested. Hidden deck order is not stored.'
        )
        atomic_write(self.output_dir / 'summary.json', payload)

    def write_config(self) -> None:
        atomic_write(self.output_dir / 'config.json', {
            'budget': {
                'max_raw_steps': self.max_raw_steps, 'max_games': self.max_games,
                'max_seconds': self.max_seconds, 'per_game_actions': self.per_game_actions,
            },
            'pid': os.getpid(), 'model': 0, 'training_updates': 0, 'label': LABEL,
        })

    def record_result(self, first_side: str, winner: str) -> None:
        bucket = winner if winner in self.winners else 'truncated'
        self.winners[bucket] += 1
        self.by_first_side[first_side][bucket] += 1
        if bucket in ('truncated', 'draw'):
            self.first_player[bucket] += 1
        elif winner == first_side:
            self.first_player['win'] += 1
        else:
            self.first_player['loss'] += 1

    def fail(self, exc: BaseException, *, timeout: bool = False, invalid: bool = False) -> None:
        self.exceptions += 1
        if timeout:
            self.timeouts += 1
            self.stopped_reason = self.stopped_reason or 'timeout'
        if invalid:
            self.invalid_actions += 1
        self.error = f'{type(exc).__name__}: {exc}'
        self.phase = 'failed'
        (self.output_dir / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        self.write_status()
        self.write_summary()

    def decks_for(self, index: int) -> tuple[str, dict[str, Any]]:
        kind = 'starter' if index < self.max_games / 2 else 'coverage_all8'
        return kind, starter_deck() if kind == 'starter' else coverage_deck()

    def _log_step(self, action: dict[str, Any], event_types: list[str]) -> bool:
        if self.raw_steps == 1 or self.raw_steps % 10 == 0:
            return True
        if action.get('type') in ('play_card', 'choose', 'ultimate', 'cycle'):
            return True
        return any(kind not in {'turn', 'move', 'finish'} for kind in event_types)

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_config()
        self.phase = 'running'
        try:
            while self.games_started < self.max_games and not self.stopped_reason:
                if self.raw_steps >= self.max_raw_steps:
                    self.stopped_reason = 'max_raw_steps'
                    break
                if self.elapsed() >= self.max_seconds:
                    self.stopped_reason = 'max_seconds'
                    break
                self._run_game()
            if not self.stopped_reason:
                self.stopped_reason = 'max_games'
            if self.error is None:
                self.phase = 'finished'
            self.write_status(**self.last_extra)
            self.write_summary(**self.last_extra)
            return self.snapshot(**self.last_extra)
        except TimeoutError as exc:
            self.fail(exc, timeout=True)
            return self.snapshot()
        except ValueError as exc:
            self.fail(exc, invalid='不能执行' in str(exc) or 'invalid' in str(exc).lower())
            return self.snapshot()
        except Exception as exc:  # noqa: BLE001
            self.fail(exc)
            return self.snapshot()

    def _budget(self) -> str:
        if self.raw_steps >= self.max_raw_steps:
            return 'max_raw_steps'
        if self.elapsed() >= self.max_seconds:
            return 'max_seconds'
        return ''

    def _run_game(self) -> None:
        index = self.games_started
        first_side = SIDES[index % 2]
        kind, deck = self.decks_for(index)
        seed = self.base_seed + index
        rng = self.rng_factory(seed)
        self.games_started += 1
        state = self.engine.new_game(
            seed=seed, decks={'a': deepcopy(deck), 'b': deepcopy(deck)},
            first_side=first_side, skip_mulligan=False,
        )
        pending = None
        game_steps = 0
        last_seq = len(state.get('events') or [])
        while True:
            hit = self._budget()
            if hit:
                self.stopped_reason = hit
                self.record_result(first_side, 'truncated')
                return
            if game_steps >= self.per_game_actions:
                self.record_result(first_side, 'truncated')
                return
            if state.get('phase') == 'finished' or state.get('winner') is not None:
                winner = state.get('winner') or 'draw'
                self.record_result(first_side, winner if winner in ('a', 'b', 'draw') else 'truncated')
                return
            side = self.engine.acting_side(state)
            if side not in SIDES:
                self.record_result(first_side, 'truncated')
                return
            observation = self.engine.observe(state, side)
            hit = self._budget()
            if hit:
                self.stopped_reason = hit
                self.record_result(first_side, 'truncated')
                return
            action = select_weighted_action(legal_action_dicts(observation), rng)
            usage = _card_from_observation(observation, action) if action.get('type') == 'play_card' else None
            remaining = self.max_seconds - self.elapsed()
            if remaining <= 0:
                self.stopped_reason = 'max_seconds'
                self.record_result(first_side, 'truncated')
                return
            timeout = remaining if self.apply_timeout is None else min(remaining, self.apply_timeout)
            state = apply_with_timeout(self.engine, state, side, action, timeout, self.clock)
            self.raw_steps += 1
            game_steps += 1
            extra = {
                'game_index': index, 'deck_kind': kind, 'first_side': first_side,
                'last_action_type': action.get('type'), 'last_side': side,
                'sample_phase': state.get('phase'), 'sample_turn': state.get('turn'),
            }
            self.last_extra = extra
            self.write_status(**extra)
            if usage:
                self.card_uses[usage['card_id']] += 1
                if usage.get('copy'):
                    self.copy_uses[usage['card_id']] += 1
                pending = usage
            elif action.get('type') != 'choose':
                pending = None
            events = list(state.get('events') or [])
            new_events = events[last_seq:]
            last_seq = len(events)
            event_types = []
            for event in new_events:
                kind_name = str(event.get('type') or 'unknown')
                event_types.append(kind_name)
                self.event_counts[kind_name] += 1
                event_card = _visible_card(event.get('card') if isinstance(event, dict) else None)
                if event_card:
                    pending = event_card
                if kind_name == 'play':
                    continue
                reliable = event_card or (pending if action.get('type') in ('play_card', 'choose') else None)
                if reliable:
                    bucket = self.card_effect_events[reliable['card_id']]
                    bucket[kind_name] = bucket.get(kind_name, 0) + 1
            if state.get('phase') != 'choice':
                pending = None
            if self._log_step(action, event_types):
                append_jsonl(self.output_dir / 'progress.jsonl', {
                    'raw_steps': self.raw_steps, 'game_index': index, 'side': side,
                    'action_type': action.get('type'),
                    'card_id': (usage or pending or {}).get('card_id') if isinstance(usage or pending, dict) else None,
                    'copy': bool((usage or {}).get('copy')), 'event_types': event_types,
                    'phase': state.get('phase'), 'winner': state.get('winner'),
                    'deck_kind': kind, 'first_side': first_side,
                })


def run_evaluation(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    return CardEffectEvaluator(output_dir, **kwargs).run()
