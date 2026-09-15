#!/usr/bin/env python3
"""On-demand offline analysis of a frozen numeric policy. Never trains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.modules.card_game.rl.offline_analysis import (
    HARD_MAX_ACTIONS,
    HARD_MAX_PAIRS,
    HARD_MAX_SECONDS,
    PRESETS,
    export_game_replay,
    frozen_model_policy,
    load_public_deck,
    resolve_numeric_model_dir,
    run_offline_analysis,
)
from app.modules.card_game.rl.policies import VisibleEngineRulePolicy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='analyze_duel_v2_policy.py',
        description='Run extra bounded official games for a frozen numeric policy, or export a recorded public replay.',
    )
    parser.add_argument('--model', type=Path, help='Numeric .npz or directory with adjacent JSON manifest')
    parser.add_argument('--opponent-model', type=Path, help='Optional second numeric .npz / directory')
    parser.add_argument('--output', type=Path, help='New report directory')
    parser.add_argument('--deck', choices=PRESETS, default='starter')
    parser.add_argument('--opponent-deck', type=Path, help='Optional explicit public deck JSON')
    parser.add_argument('--learner-deck',type=Path,help='Optional card-tuning candidate JSON for independent analysis')
    parser.add_argument('--pairs', type=int, default=1)
    parser.add_argument('--max-seconds', type=float, default=30)
    parser.add_argument('--max-actions', type=int, default=80)
    parser.add_argument('--seed', type=int, help='Optional reproducible seed; otherwise random and saved only in raw metadata')
    parser.add_argument('--export-game', help='Recorded game_id to export without rerunning')
    parser.add_argument('--source', type=Path, help='Existing report directory for --export-game')
    parser.add_argument('--replay-output', type=Path, help='Website-playable public replay JSON path')
    return parser


def _load_frozen(path: Path, deck_id: str | None):
    directory, resolved = resolve_numeric_model_dir(path, deck_id=deck_id)
    from app.modules.card_game.engine.ai.advanced_model import FrozenModel
    model = FrozenModel(directory, resolved)
    return frozen_model_policy(model)


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.export_game:
        if args.model or args.opponent_model:
            parser.error('--export-game works without loading models')
        if args.source is None or args.replay_output is None:
            parser.error('--export-game requires --source REPORTDIR and --replay-output FILE')
        result = export_game_replay(args.source, args.export_game, args.replay_output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.model is None or args.output is None:
        parser.error('analysis requires --model PATH and --output NEWDIR, or --export-game')
    if not 1 <= args.pairs <= HARD_MAX_PAIRS:
        parser.error(f'--pairs must be 1..{HARD_MAX_PAIRS}')
    if not 1 <= args.max_actions <= HARD_MAX_ACTIONS:
        parser.error(f'--max-actions must be 1..{HARD_MAX_ACTIONS}')
    if not 0 < args.max_seconds <= HARD_MAX_SECONDS:
        parser.error(f'--max-seconds must be in (0, {HARD_MAX_SECONDS}]')
    if args.output.exists():
        parser.error(f'output dir must be new, not clobber: {args.output}')
    try:
        policy = _load_frozen(args.model, args.deck)
        opponent = VisibleEngineRulePolicy()
        if args.opponent_model is not None:
            opponent = _load_frozen(args.opponent_model, None if args.opponent_model.suffix=='.npz' else args.deck)
        opponent_deck = load_public_deck(args.opponent_deck) if args.opponent_deck else None
        learner_deck = load_public_deck(args.learner_deck) if args.learner_deck else None
    except ValueError as exc:
        parser.error(str(exc))
    summary = run_offline_analysis(
        args.output,
        policy,
        deck=args.deck,
        opponent_policy=opponent,
        opponent_deck=opponent_deck,
        learner_deck=learner_deck,
        pairs=args.pairs,
        max_seconds=args.max_seconds,
        max_actions=args.max_actions,
        seed_base=args.seed,
    )
    print(json.dumps({
        'ok': True,
        'output': str(args.output),
        'games': summary.get('games'),
        'by_position': summary.get('by_position'),
        'coverage': summary.get('coverage'),
        'label': summary.get('label'),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
