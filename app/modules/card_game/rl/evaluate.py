"""Fixed-seed strategy evaluation. Not card-effect coverage and not a balance claim."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Any, Callable

from .adapter import DuelAdapter
from .backend import V2Backend
from .budget import BudgetStop, CountingBackend, TrainBudget
from .encoding import V2Encoder
from .capability import matchup_decks, preset_opponent_deck, sample_public_deck
from .policies import MaskablePolicy, RandomMaskedPolicy, VisibleEngineRulePolicy
from .transfer import default_training_deck


EVAL_SEED_BASE = 20_260_912


def _play_match(budget: TrainBudget, learner, opponent, *, seed: int, learning_side: str,
                max_game_actions: int = 400, max_opponent_actions: int = 80,
                deck: dict[str, Any] | None = None,
                opponent_deck: dict[str, Any] | None = None,
                sample_opponent: bool = False) -> dict[str, Any]:
    backend = CountingBackend(V2Backend(), budget)
    adapter = DuelAdapter(
        backend, V2Encoder(), learning_side=learning_side, opponent=opponent,
        max_game_actions=max_game_actions, max_opponent_actions=max_opponent_actions,
        allow_concede=False,
    )
    used = deepcopy(deck or default_training_deck())
    foe = deepcopy(sample_public_deck(seed) if sample_opponent else (opponent_deck or used))
    decks = matchup_decks(learning_side=learning_side, learner_deck=used, opponent_deck=foe)
    actions: list[dict[str, Any]] = []
    transition = adapter.reset(seed=seed, decks=decks, first_side='a')
    info = {
        'seed': seed, 'learning_side': learning_side, 'first_side': None,
        'opponent_policy': type(opponent).__name__,
        'winner': None, 'truncated': False, 'terminated': False,
        'truncation_reason': '', 'actions': actions, 'error': None,
        'deck': {
            'id': used.get('id'), 'name': used.get('name'),
            'character_ids': list(used.get('character_ids') or []),
            'card_ids': list(used.get('card_ids') or []),
        },
        'opponent_deck': {
            'id': foe.get('id'), 'name': foe.get('name'),
            'character_ids': list(foe.get('character_ids') or []),
            'card_ids': list(foe.get('card_ids') or []),
        },
    }
    state = adapter._state or {}
    info['first_side'] = state.get('first_side') or state.get('active_side')
    if transition.terminated or transition.truncated:
        info['terminated'] = transition.terminated
        info['truncated'] = transition.truncated
        info['winner'] = transition.info.get('winner')
        info['truncation_reason'] = transition.info.get('truncation_reason') or ''
        info['reward'] = transition.reward
        return info
    while True:
        decision = transition.decision
        try:
            index = int(learner(deepcopy(decision.observation), deepcopy(decision.actions)))
            action = decision.actions[index]
            actions.append({'side': learning_side, 'type': action.get('type'),
                            'card_id': action.get('card_id'), 'action': deepcopy(action)})
            transition = adapter.step(index, revision=decision.revision)
        except BudgetStop as exc:
            info['truncated'] = True
            info['truncation_reason'] = exc.reason
            info['reward'] = 0.0
            return info
        except Exception as exc:  # noqa: BLE001
            info['error'] = f'{type(exc).__name__}: {exc}'
            info['truncated'] = True
            info['truncation_reason'] = 'exception'
            return info
        if transition.terminated or transition.truncated:
            info['terminated'] = transition.terminated
            info['truncated'] = transition.truncated
            info['winner'] = transition.info.get('winner')
            info['truncation_reason'] = transition.info.get('truncation_reason') or ''
            info['reward'] = transition.reward
            info['terminal_reward'] = transition.info.get('terminal_reward')
            info['episode_shaping'] = transition.info.get('episode_shaping')
            info['episode_score'] = transition.info.get('episode_score')
            info['game_actions'] = transition.info.get('game_actions')
            return info


def _empty_record() -> dict[str, int]:
    return {'win': 0, 'loss': 0, 'draw': 0, 'truncated': 0, 'exception': 0, 'games': 0}


def _tally(record: dict[str, int], match: dict[str, Any], learning_side: str) -> None:
    record['games'] += 1
    if match.get('error'):
        record['exception'] += 1
        record['truncated'] += 1
        return
    if match.get('truncated') and not match.get('terminated'):
        record['truncated'] += 1
        return
    winner = match.get('winner')
    if winner == 'draw' or winner in (None, ''):
        record['draw'] += 1
    elif winner == learning_side:
        record['win'] += 1
    else:
        record['loss'] += 1


def evaluate_strategies(
    output: Path, budget: TrainBudget, *, model=None, encoder=None,
    matches_per_seat: int = 6, log: Callable[[str], None] | None = None,
    deck: dict[str, Any] | None = None,
    opponent=None, opponent_deck: dict[str, Any] | None = None,
    sample_opponent: bool = False,
) -> dict[str, Any]:
    log = log or print
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    games_path = output / 'eval_games.jsonl'
    random_policy = RandomMaskedPolicy(Random(EVAL_SEED_BASE))
    rule = opponent or VisibleEngineRulePolicy()
    trained = MaskablePolicy(model, encoder) if model is not None and encoder is not None else None
    agents = [('random', random_policy), ('rule', rule)]
    if trained is not None:
        agents.append(('trained', trained))
    eval_opponent_deck = deepcopy(opponent_deck or (None if sample_opponent else deck) or preset_opponent_deck('starter'))
    summary: dict[str, Any] = {
        'ok': True, 'label': 'fixed-seed strategy evaluation; small sample, not a balance claim',
        'eval_seed_base': EVAL_SEED_BASE, 'matches_per_seat': matches_per_seat,
        'agents': [name for name, _ in agents],
        'results': {}, 'by_seat': {}, 'exceptions': 0,
        'training_deck_id': (deck or default_training_deck()).get('id'),
        'opponent_deck_id': eval_opponent_deck.get('id'),
        'sample_opponent': bool(sample_opponent),
        'independent_eval_policy': type(rule).__name__,
    }
    for name, policy in agents:
        if name == 'rule':
            continue
        key = f'{name}_vs_rule'
        summary['results'][key] = _empty_record()
        summary['by_seat'][key] = {'a': _empty_record(), 'b': _empty_record()}
        for seat in ('a', 'b'):
            for offset in range(matches_per_seat):
                if budget.remaining_reason() or budget.games_exhausted():
                    summary['stopped_reason'] = budget.remaining_reason() or 'max_games'
                    break
                seed = EVAL_SEED_BASE + offset
                log(f'phase=eval agent={name} seat={seat} seed={seed} raw_steps={budget.raw_steps}')
                match = _play_match(
                    budget, policy, rule, seed=seed, learning_side=seat, deck=deck,
                    opponent_deck=eval_opponent_deck, sample_opponent=sample_opponent,
                )
                match['agent'] = name
                match['opponent'] = 'rule'
                with games_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(match, ensure_ascii=False, default=str) + '\n')
                _tally(summary['results'][key], match, seat)
                _tally(summary['by_seat'][key][seat], match, seat)
                if match.get('error'):
                    summary['exceptions'] += 1
            else:
                continue
            break
    summary['budget'] = budget.snapshot()
    summary['note'] = (
        'Small-sample strategy comparison against the frozen public rule opponent. '
        'Not evidence of balance. Card-effect coverage is owned by the separate eval task.'
    )
    (output / 'eval_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    return summary
