#!/usr/bin/env python3
"""Check or run the CUDA-only Duel V2 bulk rule backend.

Default mode is configuration/check only: capture/lower on CPython 3.10 so
entry blockers are real, write nothing, and do not import Torch. Explicit
--run requires actual CUDA, a fresh output directory, bounded cases/seconds,
and fail-closed unsupported entries. This is not a full-duel readiness claim.
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUPPORTED_FLAGS = (
    '--check              configuration/capability only (default); writes nothing, no Torch',
    '--run                strict CUDA execution of a bounded light-entry parity case',
    '--output DIR         fresh output directory required by --run; refused if it exists',
    '--entries NAMES      comma-separated GPU rule entry subset',
    '--load PATH          load an explicit frozen program JSON instead of capturing',
    '--max-cases N        bounded --run cases (default 2)',
    '--max-seconds N      bounded --run wall time (default 30)',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Supported flags:\n  ' + '\n  '.join(SUPPORTED_FLAGS),
    )
    parser.add_argument('--check', action='store_true',
                        help='Configuration and capability only (default). Does not write files or import Torch.')
    parser.add_argument('--run', action='store_true',
                        help='Execute bounded CUDA light-entry parity. Requires actual CUDA and a fresh --output directory.')
    parser.add_argument('--output', type=Path,
                        help='Fresh output directory for --run. Refused if the path already exists.')
    parser.add_argument('--load', type=Path,
                        help='Load an explicit frozen program JSON instead of capturing source.')
    parser.add_argument('--entries', type=str,
                        help='Comma-separated GPU rule entry subset.')
    parser.add_argument('--max-cases', type=int, default=2,
                        help='Bounded number of --run games. Default 2.')
    parser.add_argument('--max-seconds', type=float, default=30.0,
                        help='Bounded --run wall time in seconds. Default 30.')
    args = parser.parse_args(argv)
    if args.run and args.check:
        parser.error('--check and --run cannot be combined')
    if args.run and args.output is None:
        parser.error('--run requires a fresh --output directory')
    if args.output is not None and not args.run:
        parser.error('--output is only valid with --run')
    if not args.run:
        args.check = True
    return args


def parse_entries(raw):
    if raw is None:
        return None
    names = tuple(item.strip() for item in raw.split(',') if item.strip())
    if not names:
        raise SystemExit('Empty --entries is not a valid subset')
    return names


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Refusing to overwrite ' + str(path))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def run_cuda(args, requested):
    from app.modules.card_game.content.duel_v2.catalog import STARTER_DECKS
    from app.modules.card_game.engine.duel_v2 import new_game
    from app.modules.card_game.rl.batched_duel import gpu_rule_backend as gb
    from app.modules.card_game.rl.batched_duel import gpu_rule_entries as entries
    from app.modules.card_game.rl.batched_duel import gpu_rule_program as gp

    import torch
    if not torch.cuda.is_available():
        raise SystemExit('Actual CUDA required for --run; no CPU VM fallback')
    output = Path(args.output)
    if output.exists():
        raise SystemExit('Fresh output directory required; refusing to overwrite ' + str(output))
    if type(args.max_cases) is not int or isinstance(args.max_cases, bool) or args.max_cases < 1 or args.max_cases > 8:
        raise SystemExit('Bounded --max-cases must be in 1..8')
    if (
        type(args.max_seconds) not in (int, float)
        or isinstance(args.max_seconds, bool)
        or args.max_seconds <= 0
        or args.max_seconds > 120
    ):
        raise SystemExit('Bounded --max-seconds must be in (0, 120]')
    names = requested if requested is not None else entries.LIGHT_ENTRY_NAMES
    if args.load is not None:
        program = gp.load_program(args.load, expected_entries=names, runtime='seven')
    else:
        program = gp.capture_gpu_rule_program(names, runtime='seven')
    started = time.monotonic()
    report = {
        'mode': 'run',
        'supports_full_duel_rules': False,
        'device_verified': False,
        'actual_backend': 'cuda',
        'hidden_cpu_fallback': False,
        'cuda_available': True,
        'device': str(torch.cuda.current_device()),
        'cases': [],
        'decision': 'not_attempted',
    }
    collapse = next(item for item in STARTER_DECKS if item['id'] == 'collapse')
    lingke = next(item for item in STARTER_DECKS if item['id'] == 'lingke')
    with gb.GpuRuleBackend(program, entries=names, runtime='seven', max_games=args.max_cases, max_states=max(8, args.max_cases * 4)) as backend:
        report['backend_id'] = backend.backend_id
        report['capability'] = backend.capability_report()
        for index in range(args.max_cases):
            if time.monotonic() - started > args.max_seconds:
                raise SystemExit('Bounded --run time exceeded')
            state = new_game(
                seed=20261009 + index, first_side='a' if index % 2 == 0 else 'b',
                skip_mulligan=True, decks={'a': collapse, 'b': lingke},
            )
            ref = backend.import_states([state], ['run-%d' % index], [0])[0]
            actor = backend.acting_side(ref)
            finished = backend.finished(ref)
            value_a = backend.terminal_value(ref, 'a')
            cleaned = backend.clean(ref)
            exported = backend.export_states([cleaned])[0]
            row = {
                'game_id': ref.game_id,
                'acting_side': actor,
                'oracle_acting_side': entries.acting_side(state),
                'finished': finished,
                'oracle_finished': entries.finished(state),
                'terminal_value_a': value_a,
                'oracle_terminal_value_a': entries.terminal_value(state, 'a'),
                'clean_events': exported.get('events'),
            }
            if (
                row['acting_side'] != row['oracle_acting_side']
                or row['finished'] != row['oracle_finished']
                or row['terminal_value_a'] != row['oracle_terminal_value_a']
                or row['clean_events'] != []
            ):
                raise SystemExit('CUDA light-entry oracle mismatch: ' + json.dumps(row))
            report['cases'].append(row)
            if index == 0:
                try:
                    backend.decision(ref, actor or 'a')
                    report['decision'] = 'executed'
                except gb.GpuRuleCapabilityError as exc:
                    report['decision'] = 'fail_closed'
                    report['decision_error'] = str(exc)
            backend.release([ref, cleaned])
        report['stats'] = backend.stats()
        report['elapsed_seconds'] = time.monotonic() - started
    output.mkdir(parents=True)
    _write_json(output / 'gpu_rule_backend_run.json', report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main(argv=None):
    args = parse_args(argv)
    requested = parse_entries(args.entries)
    if args.check:
        from app.modules.card_game.rl.batched_duel.gpu_rule_backend import inspect_backend_configuration
        if 'torch' in sys.modules:
            raise RuntimeError('Torch must not be imported by GPU rule backend check')
        status = inspect_backend_configuration(requested, runtime='seven')
        status['torch_imported'] = 'torch' in sys.modules
        if status['torch_imported']:
            raise RuntimeError('GPU rule backend check imported Torch')
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    return run_cuda(args, requested)


if __name__ == '__main__':
    raise SystemExit(main())
