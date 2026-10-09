#!/usr/bin/env python3
"""Compile or check a frozen Duel V2 GPU-rule source program.

Default mode is configuration/check only. Explicit --build writes a fresh
frozen program and identity/source manifest. Source capture requires CPython
3.10; CPython 3.13 loads --load frozen inputs with source/hash/entry checks.
This script does not import Torch and does not claim CUDA rule execution.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUPPORTED_FLAGS = (
    '--check          configuration/validation only (default); writes nothing',
    '--build          capture and write a frozen program; requires CPython 3.10',
    '--output DIR     fresh output directory for --build; refused if it exists',
    '--load PATH      load an explicit frozen program JSON (Windows 3.13 path)',
    '--entries NAMES  comma-separated entry subset for diagnosis; never full coverage',
    '--skip-lower     skip numeric VM lowering after capture/load',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Supported flags:\n  ' + '\n  '.join(SUPPORTED_FLAGS),
    )
    parser.add_argument('--check', action='store_true',
                        help='Configuration and validation only (default). Does not write files or import Torch.')
    parser.add_argument('--build', action='store_true',
                        help='Capture canonical formal functions and write a frozen program. Requires CPython 3.10 and a fresh --output directory.')
    parser.add_argument('--output', type=Path,
                        help='Fresh output directory for --build. Refused if the path already exists.')
    parser.add_argument('--load', type=Path,
                        help='Load and validate an explicit frozen program JSON. Windows CPython 3.13 uses this instead of recapturing host bytecode.')
    parser.add_argument('--entries', type=str,
                        help='Comma-separated GPU rule entry subset for diagnosis. A subset is never labelled full coverage.')
    parser.add_argument('--skip-lower', action='store_true',
                        help='Validate capture/load without lowering a numeric VM image.')
    args = parser.parse_args(argv)
    if args.build and args.load:
        parser.error('--build and --load cannot be combined')
    if args.build and args.output is None:
        parser.error('--build requires a fresh --output directory')
    if args.output is not None and not args.build:
        parser.error('--output is only valid with --build')
    if not args.build and not args.load:
        args.check = True
    return args


def parse_entries(raw):
    if raw is None:
        return None
    names = tuple(item.strip() for item in raw.split(',') if item.strip())
    if not names:
        raise SystemExit('Empty --entries is not a valid subset')
    return names


def main(argv=None):
    args = parse_args(argv)
    from app.modules.card_game.rl.batched_duel import gpu_rule_program as gp
    requested = parse_entries(args.entries)
    if 'torch' in sys.modules:
        raise RuntimeError('Torch must not be imported by GPU rule program check/build')
    if args.build:
        program, status = gp.build_gpu_rule_program(requested, lower=not args.skip_lower)
        paths = gp.save_program(program, args.output, status=status)
        payload = dict(mode='build', paths=paths, status=status, torch_imported='torch' in sys.modules)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if status.get('compile_ready') or status.get('unresolved') is not None else 1
    if args.load:
        expected = requested
        program = gp.load_program(args.load, expected_entries=expected)
        status = gp.inspect_gpu_rule_program(
            expected if expected is not None else tuple(program.entries),
            program=program,
            lower=not args.skip_lower,
        )
        payload = dict(mode='load', path=str(args.load), status=status, torch_imported='torch' in sys.modules)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if status.get('error') is None else 1
    payload = gp.check_configuration(requested)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
