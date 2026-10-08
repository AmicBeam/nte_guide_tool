#!/usr/bin/env python3
"""Dry-run checker for the isolated old-teacher preservation experiment.

Default is input validation only: SHA/schema/build of repository fixed_ten_v1
teachers, named-column mapping onto current cross roots, paired-root seed
splits, and the predeclared KL grid. It never trains, never writes output, and
never edits serving files.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.modules.card_game.rl.cross_teacher_distillation import (  # noqa: E402
    ARCHIVED_EPISODES, KL_GRID, REPO_TEACHERS, dry_run,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teachers', type=Path, default=REPO_TEACHERS)
    parser.add_argument('--episodes', type=Path, default=ARCHIVED_EPISODES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kl-grid', nargs='*', type=float, default=list(KL_GRID))
    parser.add_argument('--run', action='store_true',
                        help='Rejected: this checker does not train in the worker turn')
    args = parser.parse_args(argv)
    if args.run:
        raise SystemExit('This checker is dry-run only; it does not train')
    result = dry_run(
        teachers=args.teachers, episodes=args.episodes, output=args.output,
        kl_grid=tuple(args.kl_grid),
    )
    if Path(result['output']).exists() or result['output_created'] or result['training']:
        raise SystemExit('Dry-run must not create output or train')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
