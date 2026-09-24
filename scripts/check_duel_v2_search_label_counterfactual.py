#!/usr/bin/env python3
"""Dry-run checker for the next independent paired-root search-label experiment.

Validates a new seed reservation, the five-preset roster, both initiatives,
model/build/rule identities and a fresh output path. Does not create the output
directory, play a game, train, or export a model. Runtime stays disabled.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.modules.card_game.rl.search_label_counterfactual import (  # noqa: E402
    DEFAULT_LEDGER, PAIRED_ROOT_GAP, PAIRED_ROOT_RUNTIME, dry_run,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-reservation', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-ledger', type=Path, default=DEFAULT_LEDGER)
    parser.add_argument('--rule-hash', help='Override live rule identity; dry-run still checks file SHA and schema')
    parser.add_argument('--run', action='store_true', help='Disabled; paired-root runtime is not implemented')
    args = parser.parse_args(argv)
    if args.run or PAIRED_ROOT_RUNTIME:
        raise SystemExit('Paired-root runtime is disabled. ' + PAIRED_ROOT_GAP)
    result = dry_run(seed_reservation=args.seed_reservation, models=args.models,
                     output=args.output, ledger=args.seed_ledger, rule_hash=args.rule_hash)
    if Path(result['output']).exists() or result['output_created'] or result['games_played']:
        raise SystemExit('Dry-run must not create output or play a game')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
