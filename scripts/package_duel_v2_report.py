#!/usr/bin/env python3
"""Package a completed full-chain report for transfer to Mac. Never trains or publishes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.modules.card_game.rl.report_bundle import package_report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(package_report(args.source,args.output),ensure_ascii=False))


if __name__=='__main__':main()
