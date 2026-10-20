"""Prepare the Windows Mctx Q-profile comparison; execution is explicit."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-cpu', action='store_true')
    mode.add_argument('--verify-cuda', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error('Use a fresh output directory; do not overwrite evidence')
    report = dict(
        mode='plan', executed=False, profiles=['natural_wdl', 'mctx_default'],
        algorithm='gumbel', simulations=32, gumbel_candidates=16,
        rescale_values={'natural_wdl': False, 'mctx_default': True},
        value_scale=.1, maxvisit_init=50., epsilon=1e-8,
        runtime='NumPy / PyTorch CUDA; official JAX is an isolated oracle only',
        windows_cuda_required=True, cuda_qualified=False, quality_approved=False,
        training_started=False, website_model_updated=False,
        stages=[
            dict(name='official_numeric', requirement='same numeric trees and fixed Gumbel; compare visits/Q/action/targets'),
            dict(name='cuda_components', command='python scripts/prepare_duel_v2_mctx.py --verify-cuda --output artifacts/rl-evals/mctx-cuda-components',
                 requirement='actual CUDA statistics and repeated graph replay; not full rules or throughput'),
            dict(name='full_rules', requirement='new same-source CUDA rule/search proof, auxiliary scope and value calibration; begin with 8 real roots'),
            dict(name='frozen_ab', requirement='same weights/builds/teacher/world sampling; shared-ledger new paired seeds and equal seats; freeze sample counts before execution'),
            dict(name='learning_ab', requirement='same initial arrays and equal complete-game/update budgets; separate training, selection and confirmation seeds'),
            dict(name='confirmation', requirement='new independent seeds; report pure policy and searched W/L/D, incomplete games, time and real denominators'),
        ],
        excluded_from_learning=['user case 012', 'selection games', 'final confirmation games'],
        pending=['actual Windows CUDA execution', 'full-rule search qualification',
                 'frozen model/build bindings and new evaluation seed reservation',
                 'predeclared game/update budgets and absolute deadline',
                 'independent strength comparison'],
    )
    exit_code = 0
    if args.verify_cpu or args.verify_cuda:
        if args.output is None:
            parser.error('Explicit --output required for verification evidence')
        try:
            report['verification_attempted'] = True
            import torch
            torch.set_num_threads(1)
            if args.verify_cuda and not torch.cuda.is_available():
                raise RuntimeError('Actual CUDA unavailable; no CPU fallback')
            from app.modules.card_game.rl.mctx_validation import verify_statistics
            report['executed'] = True
            checked = verify_statistics(device='cuda' if args.verify_cuda else 'cpu',
                                        capture=args.verify_cuda)
            report.update(mode='cuda_components' if args.verify_cuda else 'cpu_reference',
                          executed=True, result=checked, passed=checked['passed'])
        except Exception as exc:
            report.update(mode='failed', passed=False,
                          error=str(exc), error_type=type(exc).__name__)
            exit_code = 2
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
