#!/usr/bin/env python3
"""Configuration inspection only. Training is deliberately unavailable."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description='V2 RL framework: training is disabled; inspect configuration only.')
    parser.add_argument('--check', action='store_true', help='Validate and print current metadata; no game/model creation')
    parser.add_argument('--check-config', metavar='JSON', help='Validate existing metadata JSON; no model loading')
    args = parser.parse_args(argv)
    if not args.check and not args.check_config:
        parser.error('Training is disabled. Use --check or --check-config JSON; neither runs a game.')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.modules.card_game.rl.checkpoint import validate_checkpoint, read_checkpoint_metadata
    try:
        metadata = read_checkpoint_metadata(args.check_config) if args.check_config else validate_checkpoint()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({'ok': True, 'training_enabled': False, 'checkpoint': metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
