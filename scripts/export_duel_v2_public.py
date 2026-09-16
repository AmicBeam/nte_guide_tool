#!/usr/bin/env python3
"""Export one public .pt checkpoint as NumPy weights for offline evaluation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--model',type=Path)
    p.add_argument('--approve-report',type=Path)
    p.add_argument('--serving-build',type=Path,help='Explicit evaluated fixed-character bot allocation')
    args=p.parse_args(argv)
    from app.modules.card_game.rl.public_export import export_checkpoint,approve_candidate
    if args.approve_report:
        if not args.model or args.checkpoint or args.output:p.error('Use --model NPZ --approve-report JSON')
        result=approve_candidate(args.model,args.approve_report,serving_build=json.loads(args.serving_build.read_text(encoding='utf-8')) if args.serving_build else None)
    else:
        if not args.checkpoint or not args.output or args.model or args.serving_build:p.error('Use --checkpoint PT --output NEWDIR')
        result=export_checkpoint(args.checkpoint,args.output)
    print(json.dumps(result,ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
