"""Plan-first official Mctx comparison fixture. Never installs JAX or Mctx."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _write(report, output):
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'report.json').write_text(text + '\n', encoding='utf8')
    print(text)
    return text


def main(args):
    from app.modules.card_game.rl.batched_search.mctx_reference import (
        compare_records, load_records, make_bandit_fixture, plan_report, probe_official_mctx,
    )
    if args.plan and not args.run:
        report = plan_report()
        _write(report, args.output)
        return 0
    payload = load_records(args.records) if args.records else make_bandit_fixture()
    for root in payload['roots']:
        root['q_scale'] = args.q_profile
    report = compare_records(
        payload, exact=args.exact, run_official=True,
        statistical_trials=args.statistical_trials,
    )
    report['mode'] = 'run'
    report['install_attempted'] = False
    probe = probe_official_mctx()
    official_cases = [case.get('official_mctx') or {} for case in report.get('cases') or []]
    official_ran = [case for case in official_cases if case.get('status') == 'ran']
    all_ran = bool(official_cases) and len(official_ran) == len(official_cases)
    unavailable = [case.get('skip_reason') or case.get('status') for case in official_cases
                   if case.get('status') != 'ran']
    report['official_summary'] = dict(
        status='ran' if all_ran else 'incomplete',
        passed=all_ran and all(case.get('passed') for case in official_ran),
        skip_reason=None if all_ran else (unavailable or [probe.skip_reason or 'no_official_cases']),
        jax_available=probe.jax_available, mctx_available=probe.mctx_available,
        install_attempted=False,
    )
    if all_ran and args.exact:
        report['official_summary']['passed'] = all(case.get('passed') for case in official_ran)
    else:
        report['official_summary']['passed'] = False
    _write(report, args.output)
    if args.require_official and not report['official_summary']['passed']:
        return 2
    if args.exact and not report.get('passed'):
        return 1
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Probe availability only; default if --run is omitted')
    parser.add_argument('--run', action='store_true', help='Replay the built-in or provided numeric records')
    parser.add_argument('--exact', dest='exact', action='store_true', default=None,
                        help='Require explicit Gumbel data and exact comparable axes')
    parser.add_argument('--no-exact', dest='exact', action='store_false',
                        help='Do not claim exact equality; still replay records')
    parser.add_argument('--records', type=Path, help='JSON recurrent records or recorded tree')
    parser.add_argument('--output', type=Path, help='Directory for report.json')
    parser.add_argument('--statistical-trials', type=int, default=0)
    parser.add_argument('--q-profile', choices=('natural_wdl', 'mctx_default'), default='natural_wdl',
                        help='Explicit Q profile; historical records keep natural_wdl by default')
    parser.add_argument('--require-official', action='store_true',
                        help='Fail if official Mctx did not run and match; never installs')
    args = parser.parse_args()
    if args.run and args.plan:
        parser.error('Choose --plan or --run')
    if not args.run:
        args.plan = True
    if args.exact is None:
        args.exact = bool(args.run)
    if args.statistical_trials < 0 or args.statistical_trials > 64:
        parser.error('Bounded statistical trials required')
    if args.records is not None:
        args.records = args.records.resolve()
    if args.output is not None:
        args.output = args.output.resolve()
    sys.exit(main(args))
