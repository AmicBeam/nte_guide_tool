#!/usr/bin/env python3
"""Short end-to-end learner-steps/s: GPU tensor env vs CPU multiprocess trainer."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def gpu_bench(n: int, horizon: int, hidden: int) -> dict:
    import torch
    from app.modules.card_game.rl.gpu_duel.env import GpuDuelEnv
    from app.modules.card_game.rl.gpu_duel.observe import compact_observe
    from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer, collect_rollout, ppo_update, rollout_to_batch

    env = GpuDuelEnv(n, device='cpu')
    env.reset(seeds=list(range(n)), native=True)
    state, cand, _mask = compact_observe(env.state)
    net_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = CompactScorer(state.shape[-1], cand.shape[-1], hidden=hidden).to(net_device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    t0 = time.perf_counter()
    rollout = collect_rollout(env, net, horizon, observe_fn=compact_observe)
    ppo_update(net, opt, rollout_to_batch(rollout), epochs=1, minibatch=min(256, n * horizon))
    elapsed = time.perf_counter() - t0
    steps = n * horizon
    return {
        'backend': 'gpu_tensor_env+cuda_scorer' if net_device == 'cuda' else 'gpu_tensor_env+cpu_scorer',
        'n': n, 'horizon': horizon, 'learner_steps': steps,
        'seconds': round(elapsed, 3),
        'learner_steps_per_s': round(steps / elapsed, 2) if elapsed else 0,
        'net_device': net_device,
    }


def cpu_bench(n_envs: int, n_steps: int, output: Path) -> dict:
    from argparse import Namespace
    from app.modules.card_game.rl.budget import TrainBudget
    from app.modules.card_game.rl.training import train_maskable_ppo
    from app.modules.card_game.rl.transfer import default_training_deck

    output.mkdir(parents=True, exist_ok=True)
    budget = TrainBudget(max_raw_steps=max(512, n_envs * n_steps * 4), max_games=16, max_seconds=90)
    t0 = time.perf_counter()
    result = train_maskable_ppo(
        output, budget,
        n_steps=n_steps, batch_size=min(256, n_envs * n_steps), n_epochs=1,
        n_envs=n_envs, hidden_dim=64, start_seed=0, deck=default_training_deck(),
        log=lambda _msg: None,
    )
    elapsed = time.perf_counter() - t0
    steps = int(result.get('num_timesteps') or 0)
    return {
        'backend': 'cpu_multiprocess+cuda_scorer',
        'n_envs': n_envs, 'n_steps': n_steps,
        'learner_steps': steps,
        'seconds': round(elapsed, 3),
        'learner_steps_per_s': round(steps / elapsed, 2) if elapsed else 0,
        'env_steps_per_sec': result.get('env_steps_per_sec'),
        'raw_steps_per_sec': result.get('raw_steps_per_sec'),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-n', type=int, default=64)
    parser.add_argument('--gpu-horizon', type=int, default=8)
    parser.add_argument('--cpu-envs', type=int, default=8)
    parser.add_argument('--cpu-steps', type=int, default=32)
    parser.add_argument('--skip-cpu', action='store_true')
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'gpu': gpu_bench(args.gpu_n, args.gpu_horizon, hidden=64)}
    (output / 'compare.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if not args.skip_cpu:
        try:
            report['cpu'] = cpu_bench(args.cpu_envs, args.cpu_steps, output / 'cpu-run')
            gpu_sps = report['gpu']['learner_steps_per_s']
            cpu_sps = report['cpu']['learner_steps_per_s']
            report['ratio_gpu_over_cpu'] = round(gpu_sps / cpu_sps, 3) if cpu_sps else None
        except Exception as exc:  # noqa: BLE001
            report['cpu_error'] = f'{type(exc).__name__}: {exc}'
        report['note'] = (
            'Same 创生预组 and frozen rule opponent. GPU uses the integer tensor env; '
            'CPU uses the multiprocess Python engine. Not a 10x claim.'
        )
    (output / 'compare.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
