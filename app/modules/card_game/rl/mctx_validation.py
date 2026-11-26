"""Bounded A/B statistics checks; no games, training or implicit CUDA fallback."""
from contextlib import nullcontext
import numpy as np
from .mctx_q import completed_q


def cases():
    rng = np.random.default_rng(7311902)
    result = []
    for count in (1, 2, 3, 7, 12, 17, 20, 4, 9):
        legal = np.zeros(24, dtype=bool)
        positions = np.sort(rng.choice(24, count, replace=False))
        legal[positions] = True
        logits = rng.normal(size=count)
        if count == 12:
            logits[0] = -14.
        prior = np.zeros(24)
        exp = np.exp(logits - logits.max())
        prior[legal] = exp / exp.sum()
        payoffs = np.zeros(24)
        payoffs[legal] = (np.full(count, .2) if count == 4 else
                          np.linspace(.2, .2 + 1e-10, count) if count == 9 else
                          np.linspace(-.8, .8, count))
        noise = np.zeros(24)
        noise[legal] = rng.gumbel(size=count)
        result.append(dict(prior=prior, legal=legal, payoffs=payoffs,
                           noise=noise, value=.1, requested=1 if count == 1 else 32))
    return result


def verify_statistics(*, device='cpu', capture=False):
    import torch
    from .information_search import Node
    from .recovery_search import choose_root, improved_policy
    from .search_policy import visit_schedule
    from .batched_search.device_statistics import DeviceStatisticsPool
    target = torch.device(device)
    if target.type not in ('cpu', 'cuda'):
        raise ValueError('Only explicit CPU reference or actual CUDA allowed')
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA unavailable; no CPU fallback')
    if capture and target.type != 'cuda':
        raise ValueError('CUDA capture requires actual CUDA')
    fixtures = cases()
    rows, width = len(fixtures), 24
    def tensor(value, dtype):
        return torch.as_tensor(value, dtype=dtype, device=target)
    slots = tensor(np.arange(rows), torch.int64)
    owners = slots + 1
    priors = tensor(np.stack([c['prior'] for c in fixtures]), torch.float64)
    values = tensor([c['value'] for c in fixtures], torch.float64)
    legal = tensor(np.stack([c['legal'] for c in fixtures]), torch.bool)
    noise = tensor(np.stack([c['noise'] for c in fixtures]), torch.float64)
    payoffs = tensor(np.stack([c['payoffs'] for c in fixtures]), torch.float64)
    schedules = [visit_schedule(min(int(c['legal'].sum()), 16, c['requested']), c['requested'])
                 for c in fixtures]
    targets = [tensor([s[min(step, len(s)-1)] for s in schedules], torch.int64) for step in range(32)]
    active = [tensor([step < c['requested'] for c in fixtures], torch.bool) for step in range(32)]
    path_mask = torch.ones((rows, 1), dtype=torch.bool, device=target)
    reports = []
    origin_stream = torch.cuda.current_stream(target) if capture else None
    for profile in ('natural_wdl', 'mctx_default'):
        context = torch.cuda.stream(torch.cuda.Stream(device=target)) if capture else nullcontext()
        with context:
            if capture:
                torch.cuda.current_stream(target).wait_stream(origin_stream)
            pool = DeviceStatisticsPool(rows, width, device=str(target), q_scale=profile,
                allow_cpu_reference=target.type == 'cpu', max_batch=rows, max_path_len=1)
            written = pool.write_nodes(slots, owners, priors, values, legal)
            generations = written.generations
            errors = torch.zeros(rows, dtype=torch.bool, device=target)
            def execute():
                # Reset actual storage before each warmup/capture/replay, retaining leases.
                reset = pool.clear_visits(slots, generations)
                errors.copy_(reset.error)
                for step in range(32):
                    selected = pool.select_root(slots, generations, noise, targets[step], active[step])
                    reward = payoffs.gather(1, selected.actions[:, None]).squeeze(1)
                    backed = pool.backup(slots[:, None], generations[:, None], selected.actions[:, None],
                                         path_mask, reward, owners, active[step])
                    errors.logical_or_(selected.error | backed.error)
                q = pool.completed_q(slots, generations)
                pi = pool.improved_policy(slots, generations)
                final = pool.select_final(slots, generations, noise)
                inside = pool.select_interior(slots, generations)
                errors.logical_or_(q.error | pi.error | final.error | inside.error)
                return pool.snapshot(slots, generations), q, pi, final, inside
            if capture:
                execute()
                execute()
                torch.cuda.synchronize(target)
                graph = torch.cuda.CUDAGraph()
                with pool.capture_scope(), torch.cuda.graph(graph, stream=torch.cuda.current_stream(target)):
                    actual = execute()
                graph.replay()
                graph.replay()
                torch.cuda.synchronize(target)
            else:
                actual = execute()
            if bool(written.error.any() | errors.any() | actual[0].error.any()):
                raise ValueError('Actual statistics returned errors')
            snapshot, q, pi, final, interior = actual
            checked = []
            for row, case in enumerate(fixtures):
                mask = case['legal']
                node = Node.create(case['prior'][mask], case['value'])
                noise_row = case['noise'][mask]
                for step in range(case['requested']):
                    selected = choose_root(node, noise_row, schedules[row][step], profile)
                    node.visits[selected] += 1
                    node.total[selected] += case['payoffs'][mask][selected]
                expected_q = completed_q(node.prior, node.visits, node.total, node.raw_value,
                                          np.ones(len(node.prior), bool), q_scale=profile)
                expected_pi = improved_policy(node, profile)
                expected_choice = int(np.flatnonzero(mask)[choose_root(node, noise_row, int(node.visits.max()), profile)])
                expected_interior = int(np.flatnonzero(mask)[np.argmax(expected_pi - node.visits / (1 + node.visits.sum()))])
                np.testing.assert_array_equal(snapshot.visits[row].cpu().numpy()[mask], node.visits)
                np.testing.assert_allclose(q.completed_q[row].cpu().numpy()[mask], expected_q, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(pi.policy[row].cpu().numpy()[mask], expected_pi, atol=1e-10, rtol=1e-10)
                if int(final.actions[row]) != expected_choice or int(interior.actions[row]) != expected_interior:
                    raise ValueError('Final or interior action differs from NumPy reference')
                if (pi.policy[row].cpu().numpy()[~mask] != 0).any():
                    raise ValueError('Padding received policy mass')
                checked.append(dict(legal=int(mask.sum()), simulations=case['requested'], action=expected_choice))
            reports.append(dict(q_scale=profile, passed=True, cases=checked))
    return dict(passed=True, device=str(target), cuda_executed=target.type == 'cuda',
                capture_replayed=bool(capture), profiles=reports,
                full_rule_search_qualified=False, training_qualified=False, quality_approved=False)
