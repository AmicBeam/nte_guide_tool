#!/usr/bin/env python3
"""Bounded seven-team CUDA rule/encoding/sampler parity; no weights or training."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--program', type=Path, help='CPython 3.10 frozen seven-team program')
    parser.add_argument('--output', type=Path, help='Fresh artifact directory')
    parser.add_argument('--seconds', type=float, default=120)
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 600:
        parser.error('--seconds must be in 1..600')
    if args.run and (args.program is None or args.output is None):
        parser.error('--run requires --program and a fresh --output')
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.run:
        print(json.dumps({'runtime': 'seven', 'mode': 'config_only', 'device_verified': False,
                          'games_started': 0, 'torch_imported': 'torch' in sys.modules,
                          'seconds': args.seconds, 'full_1600_games_complete': False}))
        return 0
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no CPU rule fallback')
    if args.output.exists():
        raise FileExistsError('Fresh output directory required')
    import numpy as np
    from uuid import uuid4
    from app.modules.card_game.rl.batched_search.cooperative import OperationQuery, RequestExpired
    from copy import deepcopy
    from app.modules.card_game.content.duel_v2 import STARTER_DECKS, validate_deck
    from app.modules.card_game.engine.duel_v2 import new_game, apply_action
    from app.modules.card_game.engine.duel_v2.simulation import simulate_action
    from app.modules.card_game.rl import seven_runtime
    from app.modules.card_game.rl.covered_rollout import rule_decision
    from app.modules.card_game.rl.information_search import sample_world
    from app.modules.card_game.rl.league_rollout import clean
    from app.modules.card_game.rl.batched_duel.gpu_rule_backend import GpuRuleBackend
    from app.modules.card_game.rl.batched_duel.gpu_rule_capacity import plan_gpu_rule_capacity
    from app.modules.card_game.rl.batched_duel.gpu_rule_program import load_program
    from app.modules.card_game.rl.batched_search.evaluator import runtime_matrix_keys

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    program = load_program(args.program, runtime='seven')
    plan = plan_gpu_rule_capacity(14, 14, runtime='seven', query_batch_rows=14,
                                  memory_budget_bytes=14 * 1024**3)
    if not plan['valid']:
        raise ValueError(plan['rejection_reason'])
    deadline = time.monotonic() + args.seconds
    report = {'runtime': 'seven', 'device_verified': False, 'full_1600_games_complete': False,
              'scope': 'bounded parity only; no throughput or complete-game claim',
              'program_identity': program.identity, 'capacity_plan': plan, 'cases': []}
    args.output.mkdir(parents=True)
    try:
        with GpuRuleBackend(program, runtime='seven', capacity_plan=plan,
                            guard_standalone_entities=True) as backend:
            rows = []
            for key in runtime_matrix_keys('seven'):
                deck = validate_deck(deepcopy(next(d for d in STARTER_DECKS if d['id'] == key)))
                for mulligan in (True, False):
                    state = new_game(seed=2026100900 + len(rows),
                                     decks={'a': deck, 'b': deepcopy(deck)}, skip_mulligan=not mulligan)
                    clean(state)
                    rows.append({'key': key, 'mulligan': mulligan, 'state': state,
                                 'game_id': f'{key}-{int(mulligan)}'})
            refs = backend.import_states([row['state'] for row in rows],
                                         [row['game_id'] for row in rows], [0] * len(rows))
            sides = [ref.actor for ref in refs]

            def queries(method, extras):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Bounded CUDA parity time expired')
                report['stage'] = method
                pending = [OperationQuery(uuid4().hex, backend, method, (ref, *extra), (), deadline=deadline)
                           for ref, extra in zip(refs, extras)]
                responses = backend.resolve_operations(pending)
                if any(isinstance(response.value, RequestExpired) for response in responses):
                    raise TimeoutError('Bounded CUDA parity query expired')
                return [response.value for response in responses]

            decisions = queries('decision', [(side,) for side in sides])
            for row, side, (actions, (x, c)) in zip(rows, sides, decisions):
                expected_actions, (xx, cc) = seven_runtime.decision(row['state'], side)
                assert actions == expected_actions, (row['key'], 'legal action order')
                np.testing.assert_array_equal(x, xx)
                np.testing.assert_array_equal(c, cc)
            teachers = queries('rule_decision', [(side,) for side in sides])
            for row, side, (actions, choice) in zip(rows, sides, teachers):
                expected_actions, expected_choice = rule_decision(row['state'], side)
                assert actions == expected_actions, (row['key'], 'teacher legal order')
                assert choice == expected_choice, (row['key'], 'teacher choice')
            seeds = [2026100950 + i for i in range(len(rows))]
            sampled = queries('sample_world', list(zip(sides, seeds)))
            for row, side, seed, actual in zip(rows, sides, seeds, backend.export_states(sampled)):
                expected = sample_world(row['state'], side, seed, runtime=seven_runtime)
                assert actual == expected, (row['key'], 'sample world')
            actions = [decision[0][0] for decision in decisions]
            stepped = queries('step', list(zip(sides, actions)))
            for row, side, action, actual in zip(rows, sides, actions, backend.export_states(stepped)):
                expected = simulate_action(row['state'], side, action)
                clean(expected)
                assert actual == expected, (row['key'], 'simulation')
            report['stage'] = 'commit'
            committed = backend.commit_actions([(ref, side, action, 1, None)
                                                for ref, side, action in zip(refs, sides, actions)])
            for row, side, action, actual, decision in zip(rows, sides, actions,
                                                          backend.export_states(committed), decisions):
                assert actual == apply_action(row['state'], side, action), (row['key'], 'physical action')
                report['cases'].append({'key': row['key'], 'mulligan': row['mulligan'], 'passed': True,
                                        'teacher_passed': True, 'candidates': len(decision[0])})
            backend.release([*refs, *sampled, *stepped, *committed])
            report['backend_stats'] = backend.stats()
        report['device_verified'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output / 'seven-cuda-parity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
