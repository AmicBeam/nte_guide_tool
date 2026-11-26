"""Explicit offline phase-policy migration; never trains, deploys or grants readiness."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Write a new isolated numeric export bank')
    args = parser.parse_args()
    from app.modules.card_game.rl.seven_schedule import ROSTER
    plan = dict(policy_layout='phase_isolated_v1', source=str(args.source.resolve()),
                output=str(args.output.resolve()), keys=list(ROSTER), training_started=False,
                cuda_verified=False, quality_approved=False)
    if not args.apply:
        print(json.dumps(plan, indent=2))
        return
    if args.output.exists() or args.source.resolve() == args.output.resolve():
        parser.error('A new output directory separate from the source is required')
    from app.modules.card_game.rl.phase_runtime import migrate, model
    from app.modules.card_game.rl.seven_runtime import model as old_model
    # Validate the entire input before writing the first team.
    for key in ROSTER:
        old_model(args.source, key)
    for key in ROSTER:
        migrate(args.source, args.output, key)
    plan['models'] = {k: model(args.output, k).version for k in ROSTER}
    (args.output / 'migration.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
