#!/usr/bin/env python3
"""GPU-resident PPO on the frozen 创生预组 env. Does not replace CPU train_duel_v2.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Short GPU PPO loop for the starter-deck tensor env.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n', type=int, default=256)
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--updates', type=int, default=4)
    parser.add_argument('--device', default=None)
    parser.add_argument('--hidden', type=int, default=256)
    args = parser.parse_args(argv)
    import torch
    from app.modules.card_game.rl.gpu_duel.env import GpuDuelEnv
    from app.modules.card_game.rl.gpu_duel.observe import compact_observe
    from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer, collect_rollout, ppo_update, rollout_to_batch

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = GpuDuelEnv(args.n, device=device)
    env.reset(seeds=list(range(args.n)))
    state, cand, _mask = compact_observe(env.state)
    net = CompactScorer(state.shape[-1], cand.shape[-1], hidden=args.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    log = []
    for update in range(1, args.updates + 1):
        rollout = collect_rollout(env, net, args.horizon)
        batch = rollout_to_batch(rollout)
        ppo_update(net, opt, batch, minibatch=min(1024, batch[0].shape[0]))
        done = float(env.outcome()[0].float().mean())
        row = {'update': update, 'done_frac': done, 'reward_mean': float(rollout['reward'].mean())}
        log.append(row)
        print(json.dumps(row), flush=True)
    torch.save({'model': net.state_dict(), 'n': args.n}, output / 'gpu_policy.pt')
    (output / 'gpu_train.json').write_text(json.dumps(log, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
