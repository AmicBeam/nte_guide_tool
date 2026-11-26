#!/usr/bin/env python3
"""Collect bounded offline SoA conformance inputs for public-17 / seven-key duels.

Default is plan-only. Explicit --run writes a NEW artifacts/rl-evals directory.
This is not a SoA implementation, trainer, evaluator, or GPU benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from app.modules.card_game.rl.batched_duel.domain_soa_cases import (
        KEYS,
        MECHANISM_MODULES,
        execute,
        plan_report,
        validate_output_dir,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--games', type=int, default=14)
    parser.add_argument('--max-decisions', type=int, default=1000)
    parser.add_argument('--seconds', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=10160000)
    args = parser.parse_args(argv)
    if not 1 <= args.games <= 140 or not 1 <= args.max_decisions <= 10000 or not 1 <= args.seconds <= 3600:
        parser.error('Bounded games/decisions/time required')
    if not args.run:
        print(json.dumps(plan_report(games=args.games, max_decisions=args.max_decisions,
                                     seconds=args.seconds, seed=args.seed),
                         ensure_ascii=False, indent=2))
        return 0
    if not args.output:
        parser.error('--run requires a new --output directory under artifacts/rl-evals')
    try:
        validate_output_dir(args.output)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    report = execute(args.output, games=args.games, max_decisions=args.max_decisions,
                     seconds=args.seconds, seed=args.seed)
    print(json.dumps({
        'passed': report['passed'],
        'natural_input_games': report['natural_input_games'],
        'physical_decisions': report['physical_decisions'],
        'mechanism_tests': report['mechanism_tests'],
        'mechanism_skipped': report['mechanism_skipped'],
        'mechanism_failures': report['mechanism_failures'],
        'mechanism_errors': report['mechanism_errors'],
        'mechanism_commits': report['mechanism_commits'],
        'mechanism_rejections': report['mechanism_rejections'],
        'mechanism_excluded': report['mechanism_excluded'],
        'missing_coverage': report['missing_coverage'],
        'seconds': report['seconds'],
        'soa_rules_executed': report['soa_rules_executed'],
        'gpu_qualified': report['gpu_qualified'],
        'keys': list(KEYS),
        'mechanism_modules': list(MECHANISM_MODULES),
    }, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
