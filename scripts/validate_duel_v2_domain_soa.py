#!/usr/bin/env python3
"""Offline domain-codec conformance on natural formal-engine trajectories.

The formal engine generates diagnostic inputs; this is NOT a SoA game loop,
GPU benchmark, model evaluation, or training. Unsupported states are saved
and fail the run rather than being repaired or silently discarded.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


KEYS = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk', 'collapse', 'lingke')
PRESENTATION_KEYS = ('events', '_public_board')


def _check_types(expected, actual, path='root'):
    # Dict keys, lists and entity class are checked independently of JSON text.
    if type(expected) is not type(actual):
        raise AssertionError(f'{path}: type {type(actual).__name__} != {type(expected).__name__}')
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f'{path}: key set changed')
        for key in expected:
            _check_types(expected[key], actual[key], path+'.'+str(key))
    elif isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise AssertionError(f'{path}: length changed')
        for index, (left, right) in enumerate(zip(expected, actual)):
            _check_types(left, right, path+f'[{index}]')
    elif expected != actual:
        raise AssertionError(f'{path}: value changed')


def _check_aliases(expected, actual):
    left_to_right, right_to_left = {}, {}
    def walk(left, right):
        if isinstance(left, (dict, list)):
            old = left_to_right.setdefault(id(left), id(right))
            if old != id(right):
                raise AssertionError('A shared mutable reference was split')
            old = right_to_left.setdefault(id(right), id(left))
            if old != id(left):
                raise AssertionError('Independent mutable objects were aliased')
        if isinstance(left, dict):
            for key in left:
                walk(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            for a, b in zip(left, right):
                walk(a, b)
    walk(expected, actual)


def execute(output, *, games=14, max_decisions=1000, seconds=600):
    from app.modules.card_game.content.duel_v2 import STARTER_DECKS
    from app.modules.card_game.engine.duel_v2 import new_game, acting_side, legal_actions, apply_action
    from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoALayout
    from app.modules.card_game.rl.batched_duel.domain_soa_codec import encode_domain_states, decode_domain_states
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    decks = {d['id']: d for d in STARTER_DECKS}
    layout = DomainSoALayout(game_capacity=1)
    records, checked, failures = [], 0, []
    allocated_bytes = 0
    for game in range(games):
        first = 'a' if game % 2 == 0 else 'b'
        # Both first-side directions for each cyclic matchup, then repeat seeds.
        pair = (game // 2) % len(KEYS)
        left, right = KEYS[pair], KEYS[(pair+1) % len(KEYS)]
        seed = 10160000 + game
        chooser = random.Random(seed ^ 0x1016)
        state = new_game(seed=seed, decks={'a': decks[left], 'b': decks[right]}, first_side=first)
        moves = []
        record = dict(game=game, a=left, b=right, seed=seed, first_side=first,
                      natural_terminal=False, codec_failed=False, decisions=0)
        for decision in range(max_decisions+1):
            if time.monotonic()-begun >= seconds:
                record['stop'] = 'deadline'
                break
            try:
                packed = encode_domain_states([state], layout, presentation='drop')
                restored = decode_domain_states(packed)[0]
                expected = {k: v for k, v in state.items() if k not in PRESENTATION_KEYS}
                _check_types(expected, restored)
                _check_aliases(expected, restored)
                checked += 1
                allocated_bytes = max(allocated_bytes, packed.allocated_bytes)
            except Exception as exc:
                failure = dict(game=game, decision=decision, type=type(exc).__name__, error=str(exc))
                failures.append(failure)
                record['codec_failed'] = True
                record['stop'] = 'codec_unsupported_or_mismatch'
                with gzip.open(output/f'failure-{game:03d}.json.gz', 'wt', encoding='utf-8') as stream:
                    json.dump(dict(failure=failure, state=state, actions=moves), stream, ensure_ascii=False)
                break
            if state['phase'] == 'finished':
                record.update(natural_terminal=True, winner=state['winner'], turn=state['turn'], stop='natural_terminal')
                break
            if decision >= max_decisions:
                record['stop'] = 'decision_limit'
                break
            actor = acting_side(state)
            actions = [a for a in legal_actions(state, actor) if a['type'] != 'concede']
            if not actions:
                raise RuntimeError('Nonterminal formal state has no diagnostic action')
            action = chooser.choice(actions)
            moves.append(dict(side=actor, action=action))
            state = apply_action(state, actor, action)
            record['decisions'] += 1
        records.append(record)
        if time.monotonic()-begun >= seconds:
            break
    complete = len(records) == games and all(r['natural_terminal'] for r in records) and not failures
    report = dict(protocol='domain_soa_codec_natural_inputs_v1', passed=complete,
        actual_backend='formal_cpu_input_generator_and_domain_codec',
        formal_engine_is_input_generator=True, soa_rules_executed=False,
        soa_state_complete=False, batched_rules_complete=False, gpu_qualified=False,
        cuda_search32=False, model_quality_approved=False,
        presentation_keys_explicitly_excluded=list(PRESENTATION_KEYS),
        planned_games=games, attempted_games=len(records),
        natural_input_games=sum(r['natural_terminal'] for r in records), soa_natural_games=0,
        checked_states=checked, physical_decisions=sum(r['decisions'] for r in records),
        max_codec_allocated_bytes=allocated_bytes, seconds=time.monotonic()-begun,
        games=records, failures=failures)
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--games', type=int, default=14)
    parser.add_argument('--max-decisions', type=int, default=1000)
    parser.add_argument('--seconds', type=int, default=600)
    args = parser.parse_args(argv)
    if not 1 <= args.games <= 140 or not 1 <= args.max_decisions <= 10000 or not 1 <= args.seconds <= 3600:
        parser.error('Bounded games/decisions/time required')
    if not args.run:
        print(json.dumps(dict(mode='plan', games=args.games, keys=KEYS,
            scope='formal-engine natural input generation plus typed codec roundtrip',
            soa_rules_executed=False, gpu_qualified=False), ensure_ascii=False, indent=2))
        return 0
    if not args.output:
        parser.error('--run requires a new --output directory')
    report = execute(args.output, games=args.games, max_decisions=args.max_decisions, seconds=args.seconds)
    print(json.dumps({k: report[k] for k in ('passed', 'natural_input_games', 'soa_natural_games', 'checked_states',
        'physical_decisions', 'seconds', 'soa_rules_executed', 'gpu_qualified')}, indent=2))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
