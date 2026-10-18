"""Deadline supervision for resumable CUDA H3 graphs in one live arena.

The device owns start/search/commit phases. The host replays the same bounded
chunk and reads aggregate progress/errors; it never chooses or advances roots.
All raw, policy, value, RNG and statistics archives remain device-resident until
natural completion. Partial roots, guard exits and deadlines export no labels.
"""
import time
from collections import deque


def _guard_failed(loop):
    # Result masks must not hide a real conditional-graph guard at a planned
    # phase boundary. These statuses are never cleared or retried here.
    for name in ('_outer', '_inner', '_chunk', '_start_gate', '_commit_gate'):
        runner = getattr(loop, name, None)
        errors = getattr(runner, 'errors', None)
        if errors is not None and bool(errors.any().cpu()):
            return True
    return False


def _progress(result, previous, max_decisions):
    errors = result.errors.cpu().tolist()
    steps = result.steps.cpu().tolist()
    complete = result.complete.cpu().tolist()
    truncated = result.truncated.cpu().tolist()
    if (not errors or len({len(errors), len(steps), len(complete), len(truncated)}) != 1
            or any(type(v) is not int or v != 0 for v in errors)
            or any(type(v) is not bool for v in complete + truncated)
            or any(truncated)):
        raise RuntimeError('H3 chunk rule/guard/truncation failure; no joint learning or resume')
    if any(type(v) is not int or not 0 <= v <= max_decisions for v in steps):
        raise ValueError('H3 chunk exceeded original physical decision ceiling')
    if previous is not None:
        if len(previous['steps']) != len(steps):
            raise ValueError('H3 persistent cohort size changed')
        for now, before, done, was_done in zip(steps, previous['steps'], complete, previous['complete_rows']):
            if now < before or now > before + 1 or (was_done and (now != before or not done)):
                raise ValueError('H3 chunk restarted or skipped a physical prefix')
    return dict(steps=steps, complete_rows=complete, errors=errors, truncated=truncated,
                natural_games=sum(complete))


def replay_h3_chunks(collector, loop, metadata):
    """Resume only intact device phases; return only after all natural games.

    A root can span many chunks without incrementing physical steps. No
    initialize/reset/restore is called after capture, and no partial tape export
    occurs here. Final full-prefix independent proof remains the collector's job.
    """
    absolute = min(job['deadline'] for job in metadata.jobs)
    stop = time.monotonic() + absolute - time.time()
    if _guard_failed(loop):
        raise RuntimeError('Prior H3 conditional-graph guard cannot be resumed')
    previous = _progress(loop.result_device(), None, collector.max_decisions)
    if any(previous['steps']) or any(previous['complete_rows']):
        raise ValueError('H3 collection must begin from its original captured initialization')
    records = deque(maxlen=32)
    count, replay_seconds = 0, 0.0
    torch = collector.executor.torch
    while True:
        # Preserve the same original absolute deadline, plus a monotonic bound
        # so wall-clock corrections cannot extend this collection.
        remaining = min(absolute - time.time(), stop - time.monotonic())
        if remaining < collector.seconds + collector.proof_seconds:
            raise TimeoutError('Original H3 deadline exhausted; partial chunks have no learning labels')
        collector._memory()
        start = time.monotonic()
        loop.replay_chunk()
        torch.cuda.synchronize()
        count += 1
        elapsed = time.monotonic() - start
        replay_seconds += elapsed
        result = loop.result_device()
        diag = _progress(result, previous, collector.max_decisions)
        ticks = int(loop.chunk_search_ticks_device.cpu().item())
        if ticks < 0 or ticks > collector.chunk_search_ticks:
            raise ValueError('H3 chunk exceeded its declared search tick budget')
        diag.update(chunk_search_ticks=ticks, chunk_seconds=elapsed, device_used_bytes=collector._memory())
        records.append(diag)
        collector.last_diagnostics.update(diag, chunk_replays=count, replay_seconds=replay_seconds,
            chunk_search_tick_limit=collector.chunk_search_ticks, recent_chunks=list(records),
            complete=False, optimizer_steps=0)
        if elapsed > collector.seconds:
            collector.last_diagnostics['chunk_timed_out'] = True
            raise RuntimeError('H3 chunk exceeded original replay budget; no joint labels')
        if _guard_failed(loop):
            raise RuntimeError('Actual H3 conditional-graph guard failed; prefix cannot be resumed')
        if collector.on_chunk is not None:
            from copy import deepcopy
            collector.on_chunk(deepcopy(collector.last_diagnostics))
        if all(diag['complete_rows']):
            return result
        previous = diag
