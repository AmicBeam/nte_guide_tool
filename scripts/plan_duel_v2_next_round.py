#!/usr/bin/env python3
"""Materialize the declared next-round request without training or reserving seeds."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from app.modules.card_game.rl.next_training import plan
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New JSON request under artifacts/rl-evals')
    args = parser.parse_args()
    payload = plan()
    text = json.dumps(payload, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        output = args.output.resolve()
        root = (ROOT / 'artifacts' / 'rl-evals').resolve()
        if root not in output.parents:
            parser.error('Output must be under artifacts/rl-evals')
        if output.exists():
            parser.error('Preserve existing run requests; choose a new output')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding='utf-8')
        print(output)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
