#!/usr/bin/env python3
"""V2 MaskablePPO training CLI. --check stays metadata-only; --train runs a budgeted loop."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HARD_MAX_RAW_STEPS = 500000
HARD_MAX_GAMES = 4096
HARD_MAX_SECONDS = 7200
EVAL_RESERVE_STEPS = 20000
EVAL_RESERVE_GAMES = 48
EVAL_RESERVE_SECONDS = 600
MATCHES_PER_SEAT = 8


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='V2 RL: --check inspects config; --train runs MaskablePPO vs frozen rule AI.',
    )
    parser.add_argument('--check', action='store_true', help='Validate metadata; no game or model')
    parser.add_argument('--check-config', metavar='JSON', help='Validate inspect-only metadata JSON')
    parser.add_argument('--train', action='store_true', help='Run budgeted MaskablePPO training then eval')
    parser.add_argument('--export-replays', type=Path,
                        help='Convert eval_games.jsonl (file or train dir) into public replay JSON')
    parser.add_argument('--enable-training', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--deck', help='default/starter 创生预组, public-sample, other official public preset id, or deck JSON path')
    parser.add_argument('--opponent-deck', dest='opponent_deck', help='Independent opponent deck: starter/weave-rush or JSON path')
    parser.add_argument('--sample-public', action='store_true', help='Keep --deck fixed and sample public opponent decks each episode')
    parser.add_argument('--swap', help='Replace cards in the training deck, e.g. N03:N05,Z04:Z08')
    parser.add_argument('--freeze-generic', action='store_true',
                        help='When resuming, freeze board/generic-action weights and train card identity')
    parser.add_argument('--generic-lr-mult', type=float, default=1.0,
                        help='Learning-rate multiplier for generic weights; ignored if --freeze-generic')
    parser.add_argument('--kl-coef', type=float, default=0.0,
                        help='KL to the frozen previous policy on shared actions; requires --resume')
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--max-raw-steps', type=int, default=HARD_MAX_RAW_STEPS)
    parser.add_argument('--max-games', type=int, default=HARD_MAX_GAMES)
    parser.add_argument('--max-seconds', type=int, default=HARD_MAX_SECONDS)
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-epochs', type=int, default=2)
    parser.add_argument('--n-envs', type=int, default=None,
                        help='Parallel game processes. Default is CPU count, capped at 32. Max 128.')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-matches-per-seat', type=int, default=MATCHES_PER_SEAT)
    return parser


def _require_budget(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 1 <= args.max_raw_steps <= HARD_MAX_RAW_STEPS:
        parser.error(f'--max-raw-steps must be 1..{HARD_MAX_RAW_STEPS}')
    if not 1 <= args.max_games <= HARD_MAX_GAMES:
        parser.error(f'--max-games must be 1..{HARD_MAX_GAMES}')
    if not 1 <= args.max_seconds <= HARD_MAX_SECONDS:
        parser.error(f'--max-seconds must be 1..{HARD_MAX_SECONDS}')
    if args.n_steps < 8 or args.batch_size < 8:
        parser.error('--n-steps and --batch-size must be >= 8')
    from app.modules.card_game.rl.training_config import MAX_N_ENVS, default_n_envs
    if args.n_envs is None:
        args.n_envs = default_n_envs()
    if args.n_envs < 1 or args.n_envs > MAX_N_ENVS:
        parser.error(f'--n-envs must be 1..{MAX_N_ENVS}')
    if args.generic_lr_mult < 0:
        parser.error('--generic-lr-mult must be >= 0')
    if args.kl_coef < 0:
        parser.error('--kl-coef must be >= 0')
    if args.learning_rate <= 0:
        parser.error('--learning-rate must be > 0')
    if args.freeze_generic and not args.resume:
        parser.error('--freeze-generic requires --resume')
    if args.kl_coef > 0 and not args.resume:
        parser.error('--kl-coef requires --resume')


def _prepare_output(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Path:
    if args.output is None:
        parser.error('--output is required for --train')
    output = args.output.expanduser().resolve()
    if args.resume:
        output.mkdir(parents=True, exist_ok=True)
        return output
    if output.exists():
        parser.error(f'output dir must be new, not clobber: {output}')
    output.mkdir(parents=True)
    return output


def _log_to(output: Path):
    log_path = output / 'train.log'

    def log(message: str) -> None:
        line = message if message.endswith('\n') else message + '\n'
        sys.stdout.write(line)
        sys.stdout.flush()
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(line)
            handle.flush()

    return log


def run_training(args: argparse.Namespace, output: Path) -> int:
    from app.modules.card_game.rl.budget import TrainBudget
    from app.modules.card_game.rl.checkpoint import load_training_checkpoint
    from app.modules.card_game.rl.encoding import V2Encoder
    from app.modules.card_game.rl.evaluate import evaluate_strategies
    from app.modules.card_game.rl.training import train_maskable_ppo
    from app.modules.card_game.rl.transfer import apply_card_swaps, parse_card_swaps, resolve_opponent_deck, resolve_training_deck

    log = _log_to(output)
    (output / 'train.pid').write_text(str(os.getpid()), encoding='utf-8')
    reserve_steps = min(EVAL_RESERVE_STEPS, max(1, args.max_raw_steps // 4))
    reserve_games = min(EVAL_RESERVE_GAMES, max(1, args.max_games // 4))
    reserve_seconds = min(EVAL_RESERVE_SECONDS, max(30, args.max_seconds // 6))
    if args.max_raw_steps <= reserve_steps + 64:
        reserve_steps = max(1, args.max_raw_steps // 5)
    if args.max_games <= reserve_games + 4:
        reserve_games = max(1, args.max_games // 5)
    budget = TrainBudget(
        max_raw_steps=args.max_raw_steps, max_games=args.max_games, max_seconds=args.max_seconds,
        reserve_raw_steps=reserve_steps, reserve_games=reserve_games, reserve_seconds=reserve_seconds,
    )
    try:
        deck = apply_card_swaps(resolve_training_deck(args.deck), parse_card_swaps(args.swap))
        opponent_deck = resolve_opponent_deck(args.opponent_deck) if args.opponent_deck else deck
        if args.sample_public:
            from app.modules.card_game.rl.capability import COMPILED_PRESETS, is_exact_preset
            if not any(is_exact_preset(deck, preset_id) for preset_id in COMPILED_PRESETS):
                raise ValueError('--sample-public requires a fixed starter/weave-rush learner preset')
            if args.opponent_deck:
                raise ValueError('--sample-public cannot be combined with --opponent-deck')
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = {
        'pid': os.getpid(), 'budget': budget.snapshot(),
        'n_steps': args.n_steps, 'batch_size': args.batch_size, 'n_epochs': args.n_epochs,
        'n_envs': args.n_envs, 'hidden_dim': args.hidden_dim,
        'seed': args.seed, 'resume': str(args.resume) if args.resume else None,
        'eval_matches_per_seat': args.eval_matches_per_seat,
        'deck_id': deck.get('id'), 'deck_card_ids': list(deck.get('card_ids') or []),
        'opponent_deck_id': opponent_deck.get('id'),
        'sample_public': bool(args.sample_public),
        'swap': args.swap, 'freeze_generic': bool(args.freeze_generic),
        'generic_lr_mult': args.generic_lr_mult, 'kl_coef': args.kl_coef,
        'learning_rate': args.learning_rate,
        'label': 'MaskablePPO vs frozen public rule AI; not a balance claim',
    }
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    log(f'phase=start pid={os.getpid()} output={output} reserve_steps={reserve_steps}'
        f' reserve_games={reserve_games} reserve_seconds={reserve_seconds}')
    stats = train_maskable_ppo(
        output, budget, start_seed=args.seed, n_steps=args.n_steps,
        batch_size=args.batch_size, n_epochs=args.n_epochs,
        n_envs=args.n_envs, hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate, resume_from=args.resume, log=log,
        deck=deck, freeze_generic=bool(args.freeze_generic),
        generic_lr_mult=args.generic_lr_mult, kl_coef=args.kl_coef,
        opponent_deck=opponent_deck, sample_public=bool(args.sample_public),
    )
    budget.begin_eval()
    model, _meta = load_training_checkpoint(output)
    log(f'phase=eval start n_updates={stats.get("n_updates")} remaining_raw={budget.max_raw_steps - budget.raw_steps}')
    evaluate_strategies(
        output, budget, model=model, encoder=V2Encoder(),
        matches_per_seat=args.eval_matches_per_seat, log=log, deck=deck,
        opponent_deck=opponent_deck, sample_opponent=bool(args.sample_public),
    )
    log('phase=done')
    return 0 if int(stats.get('n_updates') or 0) > 0 else 1


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    train = bool(args.train or args.enable_training)
    if args.check or args.check_config:
        from app.modules.card_game.rl.checkpoint import read_checkpoint_metadata, validate_checkpoint
        try:
            metadata = read_checkpoint_metadata(args.check_config) if args.check_config else validate_checkpoint()
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps({'ok': True, 'training_enabled': False, 'checkpoint': metadata},
                         ensure_ascii=False, indent=2))
        return 0
    if args.export_replays:
        from app.modules.card_game.rl.replay_export import export_eval_games
        try:
            written = export_eval_games(args.export_replays)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps({
            'ok': True,
            'count': len(written),
            'replays': [str(path) for path in written],
        }, ensure_ascii=False, indent=2))
        return 0
    if not train:
        parser.error('Use --check / --check-config JSON, --export-replays DIR, or --train --output DIR')
    _require_budget(parser, args)
    output = _prepare_output(parser, args)
    try:
        return run_training(args, output)
    except Exception as exc:  # noqa: BLE001
        log_path = output / 'train.log'
        message = f'phase=failed {type(exc).__name__}: {exc}\n'
        sys.stderr.write(message)
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(message)
        (output / 'status.json').write_text(json.dumps({
            'ok': False, 'phase': 'failed', 'error': f'{type(exc).__name__}: {exc}',
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
