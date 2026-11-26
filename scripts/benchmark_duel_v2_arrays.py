"""Bounded component timings; never a full-search or full-game qualification."""
from pathlib import Path
import argparse
from dataclasses import fields, asdict
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def measure(operation, device, iterations):
    import numpy as np
    import torch
    for _ in range(3):
        output = operation()
    graph = None
    if device.type == 'cuda':
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = operation()
        graph.replay()
        torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        if graph is not None:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / 1000)
        else:
            begin = time.perf_counter()
            output = operation()
            times.append(time.perf_counter() - begin)
    return dict(median_seconds=float(np.median(times)), p95_seconds=float(np.percentile(times, 95)),
                min_seconds=min(times), iterations=iterations, cuda_graph=graph is not None), output


def run(args):
    import numpy as np
    import torch
    from app.modules.card_game.content.duel_v2 import STARTER_DECKS
    from app.modules.card_game.engine.duel_v2 import new_game, observe, acting_side
    from app.modules.card_game.rl import seven_lineup
    from app.modules.card_game.rl.batched_duel import gpu_seven_candidates as encoder
    from app.modules.card_game.rl.batched_search.compact_device_candidates import CompactDeviceCandidates
    from app.modules.card_game.rl.batched_search.array_zhenhong_prior import ArrayZhenhongPrior
    from app.modules.card_game.rl.rule_ir.lower_array import lower_array
    from app.modules.card_game.rl.rule_ir.combat_slice import COMBAT_SLICE
    from app.modules.card_game.rl.rule_ir.schema import FIELDS, FIELD_INDEX
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no fallback')
        device = torch.device('cuda', torch.cuda.current_device())
    elif device.type != 'cpu':
        raise ValueError('Only CUDA or explicit CPU reference allowed')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    guide = ArrayZhenhongPrior(device=device, allow_cpu_reference=device.type == 'cpu')
    kernel = lower_array(COMBAT_SLICE)
    decks = {d['id']: d for d in STARTER_DECKS}
    report = dict(scope='array components only; no full32 search or natural-game throughput claim',
                  device=str(device), torch=torch.__version__, prior_version=guide.version,
                  prior_weights=asdict(guide.weights), guide_extra_simulations=0,
                  full_search_verified=False, natural_games=0, training_updates=0, rows=[])
    if device.type == 'cuda':
        report['gpu_name'] = torch.cuda.get_device_name(device)
    for count in args.rows:
        states = [new_game(seed=args.seed+i, first_side='ab'[i%2], skip_mulligan=True,
                    decks={'a': decks[seven_lineup.KEYS[(i-i%2)%7]],
                           'b': decks[seven_lineup.KEYS[(i+1-i%2)%7]]}) for i in range(count)]
        views = [observe(state, acting_side(state), include_previews=False) for state in states]
        actions = [[entry['action'] for entry in view['legal_actions']
                    if entry['action']['type'] != 'concede'] for view in views]
        observations = torch.tensor(np.stack([seven_lineup.encode(v, a)[0]
                                   for v, a in zip(views, actions)]), device=device)
        raw = encoder.extract_raw_tokens(views, actions, capacity=800)
        raw = encoder.RawCandidateTokens(**{f.name: torch.as_tensor(getattr(raw, f.name), device=device)
            for f in fields(raw) if f.name != 'request_ids'})
        compact = CompactDeviceCandidates(raw, device=device, allow_cpu_reference=device.type == 'cpu')
        active = torch.ones(count, dtype=torch.bool, device=device)
        timing, result = measure(lambda: guide.score_device(observations, compact, raw.mask, active),
                                 device, args.iterations)
        if bool(result.errors.any().cpu()):
            raise RuntimeError('Array prior component rejected real-root inputs')
        values = torch.zeros((count, len(FIELDS)), dtype=torch.int32, device=device)
        for key, value in (('a_hp', 5), ('b_hp', 5), ('a_atk', 2), ('b_atk', 2),
                           ('a_front', 1), ('b_front', 1), ('pa_hp', 30), ('pb_hp', 30)):
            values[:, FIELD_INDEX[key]] = value
        sides = torch.arange(count, device=device, dtype=torch.int64) % 2
        combat_timing, combat_result = measure(lambda: kernel(values, sides, active), device, args.iterations)
        if bool(combat_result.errors.any().cpu()):
            raise RuntimeError('Shared IR component failed')
        report['rows'].append(dict(roots=count, complete_candidate_capacity=800,
            guide=timing, shared_ir_combat_component=combat_timing,
            guide_applicable_rows=int(result.applicable.sum().cpu()),
            nonzero_hints=int((result.shifts != 0).sum().cpu()),
            guide_errors=result.errors.cpu().tolist(),
            compact_token_bytes=compact.allocated_bytes))
    data = json.dumps(report, ensure_ascii=False, indent=2)
    (args.output/'report.json').write_text(data, encoding='utf8')
    (args.output/'integrity.json').write_text(json.dumps({
        'report.json': dict(bytes=len(data.encode()), sha256=hashlib.sha256(data.encode()).hexdigest())},
        indent=2), encoding='utf8')
    print(data)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--rows', type=int, nargs='+', default=[8, 32, 64])
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--seed', type=int, default=202610161601)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if any(n < 1 or n > 1600 for n in args.rows) or not 1 <= args.iterations <= 1000:
        p.error('Bounded rows 1..1600 and iterations 1..1000 required')
    run(args)
