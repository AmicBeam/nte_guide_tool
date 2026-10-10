#!/usr/bin/env python3
"""Plan or execute exact complete-game hybrid CUDA collection; no learning."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path)
    parser.add_argument('--source', type=Path,
                        help='Isolated frozen source checkout for a before/after variant')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline', type=Path, nargs='+')
    parser.add_argument('--optimized', type=Path, nargs='+')
    args = parser.parse_args()
    if args.source:
        # Use the exact same timing harness with an older frozen implementation,
        # without modifying that checkout or importing its private artifacts.
        import importlib.util
        sys.path.insert(0, str(args.source.resolve()))
        name = 'app.modules.card_game.rl.collection_benchmark'
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parents[1] / 'app/modules/card_game/rl/collection_benchmark.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    from app.modules.card_game.rl.collection_benchmark import prepare_case, run_case, compare_receipts, read_receipt
    if args.baseline or args.optimized:
        if not args.baseline or not args.optimized or args.case or args.execute:
            parser.error('Comparison requires both receipt lists, without --case/--execute')
        result = compare_receipts([read_receipt(p) for p in args.baseline],
                                  [read_receipt(p) for p in args.optimized])
    else:
        if not args.case:
            parser.error('--case is required')
        case = json.loads(args.case.read_text())
        config, jobs = prepare_case(case)
        if args.execute:
            if not args.output:
                parser.error('Execution requires --output')
            result = run_case(case, args.output)
        else:
            result = {'execution_started': False, 'planned_games': len(jobs),
                      'config': config, 'gpu_fivefold_accepted': False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
