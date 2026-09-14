#!/usr/bin/env python3
"""GPU-resident PPO: 1024 concurrent starter-deck games vs a GPU rule opponent."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='GPU PPO for 创生/覆纹快攻 tensor env.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--updates', type=int, default=8)
    parser.add_argument('--minibatch', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--device', default=None)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--deck', default='starter', help='starter or weave-rush; ignored when --matchup cross')
    parser.add_argument('--matchup', default='mirror', choices=('mirror', 'cross'),
                        help='mirror uses --deck on both sides; cross is 创生 vs 覆纹快攻')
    parser.add_argument('--resume', type=Path, help='load gpu_policy.pt from a previous GPU run')
    parser.add_argument('--compact', action='store_true',
                        help='compact observe (not website-encoder compatible)')
    parser.add_argument('--eval-games', type=int, default=2)
    parser.add_argument('--skip-eval', action='store_true')
    args = parser.parse_args(argv)
    import torch
    from app.modules.card_game.rl.encoding import encoder_metadata
    from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK
    from app.modules.card_game.rl.gpu_duel.env import GpuDuelEnv
    from app.modules.card_game.rl.gpu_duel.observe import compact_observe
    from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer, collect_rollout, ppo_update, rollout_to_batch
    from app.modules.card_game.rl.rule_ir.starter import validate_starter_ir

    validate_starter_ir()
    net_device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # Integer rule tensors of a few thousand rows launch faster on CPU than as many tiny CUDA kernels.
    env_device = 'cpu'
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = GpuDuelEnv(args.n, device=env_device, deck=args.deck, matchup=args.matchup)
    t0 = time.perf_counter()
    env.reset(seeds=list(range(args.n)), native=True)
    if args.compact:
        observe_fn = compact_observe
        state, cand, _mask = observe_fn(env.state)
        net = CompactScorer(state.shape[-1], cand.shape[-1], hidden=args.hidden).to(net_device)
    else:
        from app.modules.card_game.rl.gpu_duel.encoder_obs import encoder_observe
        from app.modules.card_game.rl.gpu_duel.policy import GpuCandidatePolicy
        observe_fn = encoder_observe
        state, cand, _mask = observe_fn(env.state)
        net = GpuCandidatePolicy(state.shape[-1], cand.shape[-1], hidden=args.hidden).to(net_device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=net_device, weights_only=False)
        net.load_state_dict(ckpt['model'])
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    log = []
    peak = 0.0
    for update in range(1, args.updates + 1):
        start = time.perf_counter()
        rollout = collect_rollout(env, net, args.horizon, observe_fn=observe_fn)
        batch = rollout_to_batch(rollout)
        ppo_update(net, opt, batch, epochs=args.epochs, minibatch=min(args.minibatch, batch[0].shape[0]))
        done, winner = env.outcome()
        elapsed = time.perf_counter() - start
        steps = args.n * args.horizon
        if str(net_device).startswith('cuda'):
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**20)
        row = {
            'update': update,
            'n': args.n,
            'horizon': args.horizon,
            'learner_steps': steps,
            'seconds': round(elapsed, 3),
            'learner_steps_per_s': round(steps / elapsed, 2) if elapsed else 0,
            'done_frac': float(done.float().mean()),
            'reward_mean': float(rollout['reward'].mean()),
            'env_device': env_device,
            'net_device': str(net_device),
            'gpu_lock': GPU_LOCK,
            'deck': args.deck,
            'matchup': args.matchup,
            'peak_cuda_mib': round(peak, 1),
            'encoder_compatible': not args.compact,
        }
        log.append(row)
        print(json.dumps(row), flush=True)
        torch.save({
            'model': net.state_dict(),
            'n': args.n,
            'update': update,
            'hidden': args.hidden,
            'starter_lock': GPU_LOCK,
            'gpu_lock': GPU_LOCK,
            'deck': args.deck,
            'matchup': args.matchup,
            'state_dim': state.shape[-1],
            'cand_dim': cand.shape[-1],
            'encoder_compatible': not args.compact,
            'encoder': encoder_metadata() if not args.compact else None,
        }, output / 'gpu_policy.pt')
        (output / 'gpu_train.json').write_text(json.dumps({
            'total_seconds': round(time.perf_counter() - t0, 3),
            'encoder_compatible': not args.compact,
            'log': log,
        }, indent=2) + '\n', encoding='utf-8')
    if not args.compact and not args.skip_eval and args.eval_games > 0:
        from app.modules.card_game.rl.gpu_duel.eval_python import eval_gpu_policy_on_python
        from app.modules.card_game.rl.gpu_duel.catalog import preset_by_id
        eval_deck = preset_by_id('starter' if args.matchup == 'cross' else args.deck)
        eval_report = eval_gpu_policy_on_python(
            output / 'gpu_policy.pt', n_games=args.eval_games, device='cpu', deck=eval_deck)
        (output / 'python_eval.json').write_text(json.dumps(eval_report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'python_eval': eval_report}, default=str), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
