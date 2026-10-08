#!/usr/bin/env python3
"""Stage-1 fit of the archived five-cross episodes. Does not start self-play.

The pass line is the precheck proposed in
docs/duel-v2-gumbel-policy-recovery-plan.md. It is written to the output
config before any optimizer step. Weights are not exported for serving.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.modules.card_game.rl.league_schema import ACT_END, ACT_PLAY

KEYS = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')
GATES = dict(
    cross_entropy_drop_nats=0.10,
    play_top1_tolerance=0.10,
    collapse_fraction=0.95,
    source='docs/duel-v2-gumbel-policy-recovery-plan.md',
    note='Precheck proposal for this failure, not a paper guarantee or a strength result.',
)


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def holdout_seeds(seeds, every=5):
    ordered = tuple(sorted(set(int(seed) for seed in seeds)))
    if every < 2:
        raise ValueError('Holdout stride must keep a fit set')
    return frozenset(seed for index, seed in enumerate(ordered) if index % every == 0)


def _kind(row):
    return np.asarray(row['c'], dtype=np.int64)[:, 0]


def target_rank(row):
    kind = _kind(row)
    pi = np.asarray(row['pi'], dtype=np.float64)
    top = int(np.argmax(pi))
    return dict(play=bool(kind[top] == ACT_PLAY), end=bool(kind[top] == ACT_END), legal_play=bool(np.any(kind == ACT_PLAY)))


def load_games(episode_dir):
    games = []
    for path in sorted(Path(episode_dir).glob('*.json.gz')):
        payload = json.loads(gzip.open(path, 'rt', encoding='utf-8').read())
        job = payload['job']
        if job.get('training') is False:
            raise ValueError('Frozen evaluation episode entered the fit set')
        rows = []
        for row in payload['rows']:
            item = dict(row)
            item['x'] = np.asarray(row['x'], dtype=np.float32)
            item['z'] = int(row['z'])
            if 'pi' in row:
                item['c'] = np.asarray(row['c'], dtype=np.float32)
                item['pi'] = np.asarray(row['pi'], dtype=np.float64)
                item['selected'] = int(row['selected'])
            rows.append(item)
        games.append(dict(seed=int(job['seed']), first=job['first'], left=job['left'], right=job['right'],
                          episode=path.name, rows=rows))
    if not games:
        raise ValueError('No archived training episodes')
    return games


def audit(games):
    per_key = {}
    problems = []
    for key in KEYS:
        rows = [row for game in games for row in game['rows'] if row['key'] == key and 'pi' in row]
        play_rows = []
        target_play = selected_play = 0
        for row in rows:
            kind = _kind(row)
            pi = row['pi']
            if len(pi) != len(row['c']) or not np.isfinite(pi).all() or np.any(pi < 0) or not np.isclose(pi.sum(), 1):
                problems.append(dict(key=key, error='policy label'))
                continue
            chosen = row['selected']
            if not 0 <= chosen < len(kind):
                problems.append(dict(key=key, error='selected index'))
                continue
            if int(row['z']) not in (-1, 0, 1):
                problems.append(dict(key=key, error='terminal label'))
                continue
            if np.any(kind == ACT_PLAY):
                play_rows.append(row)
                ranked = target_rank(row)
                target_play += ranked['play']
                selected_play += int(kind[chosen] == ACT_PLAY)
        per_key[key] = dict(policy_rows=len(rows), play_legal=len(play_rows), target_play_top1=target_play,
                            selected_play=selected_play)
    paired = {}
    for game in games:
        paired.setdefault(game['seed'], set()).add(game['first'])
    return dict(episodes=len(games), seeds=len(paired), both_seats=sum(len(v) == 2 for v in paired.values()),
                keys=per_key, problems=len(problems))


def split_games(games, every=5):
    held = holdout_seeds((game['seed'] for game in games), every=every)
    train = [game for game in games if game['seed'] not in held]
    hold = [game for game in games if game['seed'] in held]
    if set(game['seed'] for game in train) & set(game['seed'] for game in hold):
        raise ValueError('A paired seed crossed the fit boundary')
    return train, hold, sorted(held)


def rows_for(games, key):
    return [row for game in games for row in game['rows'] if row['key'] == key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--key', default='zhenhong')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--only', choices=('all', 'd', 'c-mid'))
    args = parser.parse_args()
    if args.only is None:
        args.only = 'all'
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config = dict(stage='offline_full_network_ablation', key=args.key, gates=GATES,
                  variants=['A_reproduce_3e-5', 'B_policy_passes_1_4_8', 'C_lr_1e-4_1e-3_on_4_passes'],
                  self_imitation=0, batch=128, reproduce_steps=67, no_serving_export=True,
                  episodes=str(args.episodes), initial=str(args.initial))
    write(out / 'config.json', config)
    games = load_games(args.episodes)
    report = audit(games)
    train, hold, held = split_games(games)
    report.update(fit_episodes=len(train), holdout_episodes=len(hold), holdout_seeds=len(held))
    write(out / 'audit.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.audit_only:
        return
    import torch
    from app.modules.card_game.rl.cross_runtime import create_network
    from app.modules.card_game.rl.outcome_runtime import update

    def load_net():
        manifest = json.loads((args.initial / f'{args.key}.json').read_text(encoding='utf-8'))
        net = create_network(int(manifest['hidden']), 'cpu')
        own = net.state_dict()
        with np.load(args.initial / f'{args.key}.npz', allow_pickle=False) as arrays:
            for name in own:
                if name not in arrays.files:
                    continue
                value = torch.from_numpy(arrays[name].copy())
                if own[name].shape != value.shape:
                    raise ValueError(f'Shape mismatch for {name}')
                own[name] = value
        net.load_state_dict(own)
        return net

    def fresh(lr):
        net = load_net()
        opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
        return net, opt

    def grad_check(net, batch):
        from app.modules.card_game.rl.league_learning import tensors
        selected = [row for row in batch if 'pi' in row]
        xx, c, mask = tensors([(row['x'], row['c']) for row in selected], 'cpu')
        policy, _ = net(xx, c, mask)
        target = torch.zeros_like(policy)
        for index, row in enumerate(selected):
            target[index, :len(row['pi'])] = torch.as_tensor(row['pi'])
        loss = -(target * policy.log_softmax(-1).masked_fill(~mask, 0)).sum(-1).mean()
        net.zero_grad()
        loss.backward()
        norms = {}
        for name, parameter in net.named_parameters():
            if parameter.grad is None:
                continue
            norms[name] = float(parameter.grad.detach().norm())
        needed = [name for name in norms if name.startswith(('cand_net', 'score', 'state_net'))]
        if not needed or any(not np.isfinite(norms[name]) or norms[name] == 0 for name in needed):
            raise ValueError('Policy gradient did not reach candidate, score, and state networks')
        return dict(loss=float(loss.detach()), norms={name: norms[name] for name in needed})

    @torch.no_grad()
    def evaluate(net, rows):
        policy_rows = [row for row in rows if 'pi' in row and np.any(_kind(row) == ACT_PLAY)]
        if not policy_rows:
            raise ValueError('Holdout has no legal play rows')
        total = play = target = end = 0
        entropy = cross = uniform = 0.0
        for start in range(0, len(policy_rows), 64):
            batch = policy_rows[start:start + 64]
            from app.modules.card_game.rl.league_learning import tensors
            xx, c, mask = tensors([(row['x'], row['c']) for row in batch], 'cpu')
            scores, _ = net(xx, c, mask)
            probs = scores.softmax(-1)
            for index, row in enumerate(batch):
                width = len(row['pi'])
                pred = probs[index, :width].cpu().numpy()
                kind = _kind(row)
                top = int(np.argmax(pred))
                total += 1
                play += int(kind[top] == ACT_PLAY)
                end += int(kind[top] == ACT_END)
                target += int(target_rank(row)['play'])
                logged = np.log(np.clip(pred, 1e-12, 1))
                cross += float(-(row['pi'] * logged).sum())
                entropy += float(-(pred * logged).sum())
                uniform += float(np.log(width))
        return dict(play_legal=total, model_play_top1=play / total, target_play_top1=target / total,
                    model_end_top1=end / total, cross_entropy=cross / total, model_entropy=entropy / total,
                    uniform_entropy=uniform / total,
                    cross_entropy_drop=uniform / total - cross / total)

    def passes(metrics):
        drop = metrics['cross_entropy_drop'] >= GATES['cross_entropy_drop_nats']
        close = abs(metrics['model_play_top1'] - metrics['target_play_top1']) <= GATES['play_top1_tolerance']
        collapsed = max(metrics['model_play_top1'], metrics['model_end_top1']) >= GATES['collapse_fraction']
        return bool(drop and close and not collapsed)

    train_rows = rows_for(train, args.key)
    hold_rows = rows_for(hold, args.key)
    net, _ = fresh(3e-5)
    policy_batch = [row for row in train_rows if 'pi' in row][:32]
    checked = grad_check(net, policy_batch)
    write(out / 'gradient-check.json', checked)
    results = {}
    if args.only != 'all' and (out / 'ablation.json').is_file():
        results = json.loads((out / 'ablation.json').read_text(encoding='utf-8'))

    def run_named(name, lr, mode, steps=0, passes_n=0, prepare=None):
        net, opt = fresh(lr)
        if prepare is not None:
            opt = prepare(net, opt, lr)
        rng = np.random.default_rng(20260924)
        policy = [row for row in train_rows if 'pi' in row]
        taken = 0
        if mode == 'mixed':
            for _ in range(steps):
                pick = rng.integers(0, len(train_rows), size=128)
                update(net, opt, [train_rows[int(i)] for i in pick], self_imitation=0)
                taken += 1
        else:
            for _ in range(passes_n):
                order = rng.permutation(len(policy))
                for start in range(0, len(policy), 128):
                    batch = [policy[int(i)] for i in order[start:start + 128]]
                    if len(batch) < 32:
                        continue
                    update(net, opt, batch, self_imitation=0)
                    taken += 1
        metrics = evaluate(net, hold_rows)
        metrics.update(steps=taken, learning_rate=lr, passed=passes(metrics))
        results[name] = metrics
        write(out / 'ablation.json', results)
        print(name, json.dumps(metrics, ensure_ascii=False))
        return metrics['passed']

    def normal_candidate(net, opt, lr):
        for module in (net.cand_net, net.score):
            for layer in module.modules():
                if isinstance(layer, torch.nn.Linear):
                    layer.reset_parameters()
        opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
        return opt

    def attach_residual(net, opt, lr):
        from app.modules.card_game.rl.cross_grounded import CATEGORIES
        kind_n, card_n = CATEGORIES[0][1], CATEGORIES[2][1]
        residual = torch.nn.Linear(kind_n + card_n, 1)
        torch.nn.init.zeros_(residual.weight)
        torch.nn.init.zeros_(residual.bias)
        base = net.forward

        def forward(x, c, mask):
            scores, value = base(x, c, mask)
            kind = torch.nn.functional.one_hot(c[..., 0].long().clamp(0, kind_n - 1), kind_n).to(x.dtype)
            card = torch.nn.functional.one_hot((c[..., 2].long() + 1).clamp(0, card_n - 1), card_n).to(x.dtype)
            scores = scores + residual(torch.cat((kind, card), -1)).squeeze(-1)
            return scores, value

        net.forward = forward
        opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad] + list(residual.parameters()), lr=lr)
        net._residual = residual
        return opt

    def run_all():
        run_named('A_reproduce_3e-5', 3e-5, 'mixed', steps=67)
        for count in (1, 4, 8):
            run_named(f'B_policy_passes_{count}', 3e-5, 'policy', passes_n=count)
        any_b = any(results[name]['passed'] for name in results if name.startswith('B_'))
        if not any_b:
            for lr, label in ((1e-4, '1e-4'), (1e-3, '1e-3')):
                run_named(f'C_passes_4_lr_{label}', lr, 'policy', passes_n=4)

    if args.only == 'all':
        run_all()
    elif args.only == 'c-mid':
        run_named('C_passes_4_lr_3e-4', 3e-4, 'policy', passes_n=4)
    else:
        run_named('D_normal_candidate_init', 3e-5, 'policy', passes_n=4, prepare=normal_candidate)
        run_named('D_raw_kind_card_residual', 3e-5, 'policy', passes_n=4, prepare=attach_residual)
    passed = [name for name, item in results.items() if item['passed']]
    moved = [name for name, item in results.items() if item['model_play_top1'] >= 0.2 and name.startswith('C_')]
    if passed:
        verdict = 'retain_representation_gate_passed'
    elif moved:
        verdict = 'retain_representation_calibrate_steps_and_lr'
    else:
        verdict = 'fix_representation_before_self_play'
    write(out / 'verdict.json', dict(verdict=verdict, results=results, serving_export=False))
    print(verdict)


if __name__ == '__main__':
    main()
