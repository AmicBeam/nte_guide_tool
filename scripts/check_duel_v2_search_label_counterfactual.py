#!/usr/bin/env python3
"""Evaluation-only paired-root checker for five-preset search labels.

Default is a dry-run: it validates a new seed reservation, the ten cross-pairs,
both initiatives, model/build/rule identities and a fresh output path. It does
not create the output directory, play a game, train, or export a model.

Explicit --run enables a bounded evaluation of forced end_turn versus a paid
alternative on cloned information-set worlds. Continuations are not a full-game
win rate and never create an optimizer.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.modules.card_game.rl.search_label_counterfactual import (  # noqa: E402
    DEFAULT_LEDGER, MAX_WALL_SECONDS, PAIRED_ROOT_GAP, PAIRED_ROOT_RUNTIME,
    dry_run, evaluate_paired_roots,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-reservation', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-ledger', type=Path, default=DEFAULT_LEDGER)
    parser.add_argument('--run-id', help='Allow this reservation when it already exists in the shared ledger')
    parser.add_argument('--rule-hash', help='Override live rule identity; still checks file SHA and schema')
    parser.add_argument('--seconds', type=int, default=MAX_WALL_SECONDS)
    parser.add_argument('--max-games', type=int, default=20)
    parser.add_argument('--max-roots', type=int, default=20)
    parser.add_argument('--include-non-end', action='store_true',
                        help='Also compare eligible roots where Gumbel did not choose end_turn')
    parser.add_argument('--run', action='store_true', help='Create a new output path and evaluate paired roots')
    args = parser.parse_args(argv)
    common = dict(seed_reservation=args.seed_reservation, models=args.models, output=args.output,
                  ledger=args.seed_ledger, rule_hash=args.rule_hash, run_id=args.run_id)
    if not args.run:
        result = dry_run(**common)
        if Path(result['output']).exists() or result['output_created'] or result['games_played']:
            raise SystemExit('Dry-run must not create output or play a game')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if not PAIRED_ROOT_RUNTIME:
        raise SystemExit('Paired-root runtime is disabled. ' + PAIRED_ROOT_GAP)
    result = evaluate_paired_roots(**common, seconds=args.seconds, max_games=args.max_games,
                                   max_roots=args.max_roots,
                                   require_end_selected=not args.include_non_end)
    if result.get('optimizer') or result.get('training') or result.get('serving_export'):
        raise SystemExit('Paired-root evaluation must not train or export')
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


if __name__ == '__main__':
    main()
