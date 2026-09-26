#!/usr/bin/env python3
"""Fresh CPU games prove exact preservation, not stronger trained policies."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--seconds', type=int, default=300)
    args = parser.parse_args()
    if not 30 <= args.seconds <= 600:
        parser.error('Bounded CPU verification must use 30..600 seconds')
    config = dict(question='Does zero residual preserve every old legal-action logit in real current-rule games?',
                  keys=['starter', 'weave-rush', 'quick-rush'], inference='pure',
                  learning=False, search=False, optimizer_steps=0, website_approval=False,
                  strength_claim=False, seconds=args.seconds)
    if not args.run:
        print(json.dumps(dict(dry_run=True, config=config))); return
    if args.output.exists():
        raise ValueError('Output directory must be new')
    import torch
    from app.modules.card_game.rl import preserved_policy, recovery_runtime
    from app.modules.card_game.rl.build_acceptance import claim_seeds
    from scripts.fit_duel_v2_policy_recovery import write
    torch.set_num_threads(2); torch.manual_seed(20260927)
    deadline = time.time() + args.seconds
    config['started_at'] = datetime.now(timezone.utc).isoformat()
    config['seeds'] = claim_seeds(ROOT / 'artifacts/rl-seed-ledger.json', str(args.output.resolve()))
    decks = {}
    for key in config['keys']:
        net, decks[key] = preserved_policy.create_network(key)
        preserved_policy.export(net, args.output / 'models', key, decks[key],
                                dict(initialization='exact_frozen_policy_zero_residual', updates=0))
    config['models'] = {key: recovery_runtime.model(args.output / 'models', key).version for key in decks}
    write(args.output / 'config.json', config)
    games = []
    pair_id = 0
    keys = config['keys']
    for i, left in enumerate(keys):
        for right in keys[i:]:
            seed = config['seeds']['confirmation'] + pair_id
            pair_id += 1
            for first in ('a', 'b'):
                job = dict(seed=seed, first=first, pure=True, require_preservation=True,
                           decks={'a': decks[left], 'b': decks[right]}, deadline=deadline,
                           policies={'a': (str(args.output / 'models'), left),
                                     'b': (str(args.output / 'models'), right)})
                result = recovery_runtime.search_game(job)
                counts = Counter((r['side'], r['action']['type']) for r in result['episode_actions'])
                summary = dict(seed=seed, first=first, left=left, right=right,
                               complete=result['complete'], winner=result.get('winner'),
                               decisions=result.get('decisions', 0),
                               preserved_roots=result.get('preserved_roots', 0),
                               play_counts={side: counts[side, 'play_card'] for side in ('a', 'b')})
                games.append(summary)
                with (args.output / 'games.jsonl').open('a') as handle:
                    handle.write(json.dumps(summary) + '\n')
                # Private actual action sequence, not a fabricated public replay.
                write(args.output / 'actions' / f'{pair_id}-{first}.json', result['episode_actions'])
                print(json.dumps(summary), flush=True)
                if not result['complete']:
                    write(args.output / 'result.json', dict(complete=False, reason=result['reason'], games=games))
                    return
    write(args.output / 'result.json', dict(
        complete=True, planned=12, completed=len(games),
        preserved_roots=sum(r['preserved_roots'] for r in games),
        plays=sum(sum(r['play_counts'].values()) for r in games),
        exact_preservation=True, strength_improved=False, website_approval=False, games=games))


if __name__ == '__main__':
    main()
