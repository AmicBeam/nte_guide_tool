"""Bounded CUDA-only DeviceSoARules / DeviceGameLoop acceptance CLI.

Default prints a plan and does not import CUDA, load models, or write files.
--run requires a frozen native image, frozen models, teachers, and a new
artifacts/rl-evals directory. Search stays Gumbel32/cap16/h0/network-WDL/
policy opponents with the array Zhenhong root prior. Not training.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
KEYS = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk', 'collapse', 'lingke')
SEARCH = dict(
    simulations=32, gumbel_candidates=16, terminal_horizon=0,
    teacher_mode='network_value', opponent_mode='policy',
    prior='array_zhenhong', extra_guidance_simulations=0, cpu_fallback=False,
    world_sampling='per_decision')
STAGE_SOURCE = r"""
__device__ unsigned long long ticks(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
extern "C" __global__ void begin_span(long long* start,long long* live,const unsigned char* active,int rows,int site){
 if(blockIdx.x||threadIdx.x)return;long long n=0;for(int i=0;i<rows;i++)n+=active[i]!=0;live[site]=n;start[site]=(long long)ticks();
}
extern "C" __global__ void end_span(const long long* start,const long long* live,long long* elapsed,long long* calls,long long* active_calls,long long* active_rows,int site){
 if(blockIdx.x||threadIdx.x)return;elapsed[site]+=(long long)ticks()-start[site];calls[site]++;active_calls[site]+=live[site]>0;active_rows[site]+=live[site];
}
"""


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--program', type=Path)
    p.add_argument('--models', type=Path)
    p.add_argument('--teachers', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--rows', type=int, default=8)
    p.add_argument('--max-decisions', type=int, default=1)
    p.add_argument('--require-natural', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=202610161600)
    p.add_argument('--setup-seconds', type=int, default=600)
    p.add_argument('--execution-seconds', type=int, default=900)
    p.add_argument('--proof-seconds', type=int, default=300)
    p.add_argument('--max-actions', type=int, default=800)
    p.add_argument('--inner-max-ticks', type=int, default=10000)
    p.add_argument('--rule-steps', type=int, default=48000000)
    return p


def plan(args):
    return dict(
        mode='plan',
        scope='bounded CUDA DeviceSoARules DeviceGameLoop acceptance; not training or CPU fallback',
        rows=args.rows, max_decisions=args.max_decisions,
        require_natural=bool(args.require_natural), device=str(args.device),
        warmup=args.warmup, repeats=args.repeats, presets=list(KEYS),
        execution_budget_scope='per_trial',proof_budget_scope='after_each_batch',
        typed_state_slots=args.rows*6+8,
        mixed_first_sides=True, skip_mulligan=True, search=dict(SEARCH),
        cuda_imported=False, models_loaded=False, files_created=False,
        learning_updates=0, gpu_qualified=False, passed=False,
        output=None if args.output is None else str(args.output))


def _bounded(parser, name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        parser.error('Bounded %s in %s..%s required' % (name, low, high))


def validate_bounds(parser, args):
    _bounded(parser, 'rows', args.rows, 1, 128)
    _bounded(parser, 'max-decisions', args.max_decisions, 1, 400)
    _bounded(parser, 'warmup', args.warmup, 0, 8)
    _bounded(parser, 'repeats', args.repeats, 1, 8)
    _bounded(parser, 'setup-seconds', args.setup_seconds, 1, 600)
    _bounded(parser, 'execution-seconds', args.execution_seconds, 1, 86400)
    _bounded(parser, 'proof-seconds', args.proof_seconds, 1, 86400)
    _bounded(parser, 'max-actions', args.max_actions, 1, 800)
    _bounded(parser, 'inner-max-ticks', args.inner_max_ticks, 1, 100000)
    _bounded(parser, 'rule-steps', args.rule_steps, 1, 2 ** 30)
    if type(args.seed) is not int or not 0 <= args.seed <= (1 << 63)-args.rows:
        parser.error('Nonnegative int64 seed range must include every independent row')
    if not str(args.device).startswith('cuda'):
        parser.error('CUDA-only acceptance; CPU fallback is forbidden')


def validate_run(parser, args):
    if args.program is None or args.models is None or args.teachers is None or args.output is None:
        parser.error('--run requires --program --models --teachers --output')
    args.program = args.program.resolve()
    args.models = args.models.resolve()
    args.teachers = args.teachers.resolve()
    args.output = args.output.resolve()
    if not args.program.is_file():
        parser.error('Frozen native image --program is required')
    if not args.models.is_dir():
        parser.error('Frozen --models directory is required')
    if not args.teachers.is_dir():
        parser.error('Frozen --teachers directory is required')
    try:
        args.output.relative_to((ROOT/'artifacts'/'rl-evals').resolve())
    except ValueError:
        parser.error('--output must be a new artifacts/rl-evals directory')
    if args.output.exists():
        parser.error('Use a new output directory')


def parameters_for(rows, seed):
    jobs = []
    for i in range(rows):
        jobs.append(dict(seed=seed + i, first_side='ab'[i % 2], skip_mulligan=True,
                         escalation=True, keys={'a': KEYS[i % 7], 'b': KEYS[(i + 1) % 7]}))
    return jobs


def compact_summary(report):
    keys = ('passed', 'phase', 'rows', 'max_decisions', 'natural_games', 'truncated_games',
            'error_rows', 'require_natural', 'one_step_cannot_qualify_natural',
            'capture_seconds', 'replay_wall_seconds', 'replay_gpu_seconds',
            'setup_seconds', 'complete_root_decisions')
    return {key: report[key] for key in keys if key in report}


def save_report(output, report, phase, started):
    report['phase'] = phase
    report['seconds'] = time.monotonic() - started
    tmp = output / 'report.json.tmp'
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    tmp.replace(output / 'report.json')


class StageTiming:
    """Device globaltimer spans compiled in this CLI; no extra module."""

    def __init__(self, device, sites=64):
        import torch
        from app.modules.card_game.rl.batched_duel.cuda_runtime import CudaModule
        self.device = device
        self.names, self.sites, self.ids = [], sites, {}
        self.start = torch.zeros(sites, dtype=torch.int64, device=device)
        self.live = torch.zeros_like(self.start)
        self.elapsed = torch.zeros_like(self.start)
        self.calls = torch.zeros_like(self.start)
        self.active_calls = torch.zeros_like(self.start)
        self.active_rows = torch.zeros_like(self.start)
        self.module = CudaModule(
            STAGE_SOURCE,
            {'begin_span': ['int64*', 'int64*', 'uint8*', 'int32', 'int32'],
             'end_span': ['int64*', 'int64*', 'int64*', 'int64*', 'int64*', 'int64*', 'int32']},
            device=str(device))

    def call_mask(self, name, mask, fn, *args, **kwargs):
        import torch
        if name not in self.ids:
            self.ids[name] = len(self.names)
            if len(self.names) >= self.sites:
                raise RuntimeError('Stage timing site capacity exceeded')
            self.names.append(name)
        site = self.ids[name]
        mask = mask.view(torch.uint8).contiguous()
        self.module.launch('begin_span', [self.start, self.live, mask, mask.numel(), site],
                           grid=1, block=1, capture_safe=True)
        out = fn(*args, **kwargs)
        self.module.launch(
            'end_span',
            [self.start, self.live, self.elapsed, self.calls, self.active_calls, self.active_rows, site],
            grid=1, block=1, capture_safe=True)
        return out

    def after_batch(self):
        return [dict(name=name, site=i, seconds=int(self.elapsed[i].cpu()) / 1e9,
                     calls=int(self.calls[i].cpu()), active_calls=int(self.active_calls[i].cpu()),
                     active_rows=int(self.active_rows[i].cpu()))
                for i, name in enumerate(self.names)]

    def close(self):
        self.module.close()


def action_store(device, rows):
    from app.modules.card_game.rl.batched_duel.device_graph_storage import CompactDeviceGraphStore
    return CompactDeviceGraphStore(
        slots=rows * 6 + 128, node_pages=8192, edge_pages=8192, payload_pages=4096,
        node_page_bytes=4096, edge_page_bytes=4096, payload_page_bytes=4096,
        max_node_pages_per_slot=64, max_edge_pages_per_slot=64, max_payload_pages_per_slot=32,
        device=device)


def unpack_graph(graph, mask=None):
    import numpy as np
    from app.modules.card_game.rl.batched_duel.codec import unpack_states
    from app.modules.card_game.rl.batched_duel.schema import GraphLayout, PackedStateBatch
    arrays = {name: getattr(graph, name).detach().cpu().numpy()
              for name in ('nodes', 'edges', 'payload', 'counts', 'roots')}
    ids = list(range(graph.rows)) if mask is None else [i for i, on in enumerate(mask) if on]
    output = [None] * graph.rows
    if ids:
        layout = GraphLayout(len(ids), graph.layout.max_nodes, graph.layout.max_edges,
                             graph.layout.max_payload_bytes)
        packed = PackedStateBatch(
            layout, **{name: np.ascontiguousarray(value[ids]) for name, value in arrays.items()})
        for i, value in zip(ids, unpack_states(packed)):
            output[i] = value
    return output


def host_states(store, roots):
    from app.modules.card_game.rl.batched_duel.record_soa import unpack_game_states
    return unpack_game_states(store.gather_device(roots, shared_reads=True).to_host())


def memory_snapshot(torch, store, rules, actions, raw, light):
    return dict(
        allocated=int(torch.cuda.memory_allocated()), reserved=int(torch.cuda.memory_reserved()),
        typed_store=int(store.allocated_bytes), workspace=int(rules.workspace_bytes),
        action_store=int(actions.allocated_bytes), raw_tape=int(raw.allocated_bytes),
        policy_tape=int(getattr(light, 'allocated_bytes', 0)))


def install_root_diagnostics(cohort,torch,rows,width):
    B,shared=rows,{'max_actions':width}
    device=cohort.stats.device
    cohort.diagnostics={name:torch.zeros((B,shared['max_actions']),dtype=torch.float64,device=device)
                      for name in ('q','prior','mean','gumbel')}
    cohort.diagnostics['value']=torch.zeros(B,dtype=torch.float64,device=device)
    cohort.diagnostics['errors']=torch.zeros(B,dtype=torch.bool,device=device)
    original_snapshot=cohort.stats.snapshot
    def snapshot(*args,**kwargs):
        result=original_snapshot(*args,**kwargs)
        q=cohort.stats.completed_q(*args,**kwargs)
        active=kwargs.get('active')
        if active is None:
            raise RuntimeError('Root diagnostic snapshot requires the exact final mask')
        values=dict(q=q.completed_q,prior=result.prior,
            mean=result.total/result.visits.clamp_min(1),gumbel=cohort.search._noise)
        for name,value in values.items():
            cohort.diagnostics[name].copy_(torch.where(active[:,None],value,cohort.diagnostics[name]))
        cohort.diagnostics['value'].copy_(torch.where(active,result.value,cohort.diagnostics['value']))
        cohort.diagnostics['errors'].logical_or_(active & (result.error|q.error))
        return result
    cohort.stats.snapshot=snapshot


class IsolatedCohort:
    """Own store/tape/search/loop. Recreated per timing trial or acceptance run."""

    def __init__(self, shared, *, max_decisions, seconds):
        self.started=time.monotonic()
        import numpy as np
        import torch
        from app.modules.card_game.rl.batched_duel.device_soa_storage import DeviceSoAStore
        from app.modules.card_game.rl.batched_search.device_soa_rules import DeviceSoARules, create_soa_search
        from app.modules.card_game.rl.batched_search.device_raw_action_tape import DeviceRawActionTape
        from app.modules.card_game.rl.batched_search.device_replay_policy_tape import DeviceReplayPolicyTape
        from app.modules.card_game.rl.batched_search.device_game_policy_sink import DeviceGamePolicySink
        from app.modules.card_game.rl.batched_search.device_physical_action_capture import RawActionPolicySink
        from app.modules.card_game.rl.batched_search.device_random import DevicePCG64
        from app.modules.card_game.rl.batched_search.device_root_random import DeviceRootRandomPlan
        from app.modules.card_game.rl.batched_search.device_seeds import DeviceRootSeeds
        from app.modules.card_game.rl.batched_search.device_statistics import DeviceStatisticsPool
        from app.modules.card_game.rl.batched_search.device_info_key import DeviceInformationKey
        from app.modules.card_game.rl.batched_search.device_games import DeviceGameLoop
        self.shared, self.max_decisions, self.closed = shared, max_decisions, False
        B, device = shared['rows'], shared['device']
        self.store = DeviceSoAStore(shared['schema'], B * 6 + 8, device=device)
        self.actions = action_store(device, B)
        self.raw = DeviceRawActionTape(
            B, B * max_decisions, B * max_decisions * 1024, B * max_decisions * 1024,
            B * max_decisions * 4096, device=device)
        self.light = DeviceReplayPolicyTape(
            B, B * max_decisions, B * max_decisions * shared['max_actions'], shared['max_actions'],
            128 << 20,
            device=device)
        self.rules = DeviceSoARules(
            shared['executor'], self.store, self.actions, rows=B,
            max_actions=shared['max_actions'], raw_tape=self.raw)
        self.random = DevicePCG64(
            [np.random.PCG64(shared['seed'] + i).state for i in range(B)],
            device=device, max_rows=B, max_draws=shared['max_actions'],
            max_choices=shared['max_actions'])
        self.random_plan = DeviceRootRandomPlan(
            self.random, [shared['seed'] + i for i in range(B)],
            max_decisions=max_decisions, max_bytes=64 << 20)
        self.seeds = DeviceRootSeeds(device=device, max_rows=B)
        self.keys = DeviceInformationKey(device=device, max_rows=B)
        self.stats = DeviceStatisticsPool(
            B * 33, shared['max_actions'], device=device, max_batch=B, max_path_len=10)
        self.search = create_soa_search(
            self.rules, shared['inference'], shared['inference'], self.stats, self.keys,
            rows=B, random=self.random, max_actions=shared['max_actions'], zhenhong_prior=True,
            noise=True, max_depth=10, value_batch_rows=min(8, B), queued_row_work=True,
            row_work_seconds=seconds, reuse_root_policy_value=True,
            group_ready_leaves=True, reuse_interior_observation=True)
        policy = DeviceGamePolicySink(self.light)
        self.sink = RawActionPolicySink(policy, self.raw)
        self.loop = DeviceGameLoop(
            self.rules, self.search, self.sink, self.seeds, max_decisions=max_decisions,
            random_plan=self.random_plan, device=device, inner_max_ticks=shared['inner_max_ticks'],
            seconds=seconds)
        self.rules.bind_game_loop(self.loop)
        policy.bind_game_loop(self.loop)
        self.timing = None
        self.root_prior_separated = False
        self.guided_row_calls=torch.zeros(1,dtype=torch.int64,device=device)
        self.nonzero_guided_row_calls=torch.zeros_like(self.guided_row_calls)
        install_root_diagnostics(self,torch,B,shared['max_actions'])
        try:
            self.timing = StageTiming(self.loop.device)
            for name in ('sample_world_device', 'decision_device', 'step_device',
                         'clean_device', 'metadata_device', 'value_observation_device'):
                original_rule = getattr(self.rules, name)
                def timed_rule(states, *args, _name=name, _original=original_rule, **kwargs):
                    return self.timing.call_mask(_name, states.active,
                        _original, states, *args, **kwargs)
                setattr(self.rules, name, timed_rule)
            prior = self.search.prior_guide
            original_prior = prior.score_device

            def timed_prior(observations, candidates, valid, mask, _original=original_prior):
                result=self.timing.call_mask(
                    'root_zhenhong_prior', mask.view(torch.uint8),
                    _original, observations, candidates, valid, mask)
                self.guided_row_calls.add_(result.applicable.long().sum().reshape(1))
                self.nonzero_guided_row_calls.add_(((result.shifts!=0).any(-1)&result.applicable).long().sum().reshape(1))
                return result

            prior.score_device = timed_prior
            self.root_prior_separated = True
            span = len(self.timing.names)
            self.timing.ids['complete_gumbel32'] = span
            self.timing.names.append('complete_gumbel32')
            original_start = self.loop._start_search
            original_commit = self.loop._commit_turn

            def start_span(_original=original_start):
                mask = self.loop._present.view(torch.uint8).contiguous()
                self.timing.module.launch(
                    'begin_span', [self.timing.start, self.timing.live, mask, B, span],
                    grid=1, block=1, capture_safe=True)
                return _original()

            def finish_span(_original=original_commit):
                self.timing.module.launch(
                    'end_span',
                    [self.timing.start, self.timing.live, self.timing.elapsed,
                     self.timing.calls, self.timing.active_calls, self.timing.active_rows, span],
                    grid=1, block=1, capture_safe=True)
                return _original()

            self.loop._start_search = start_span
            self.loop._commit_turn = finish_span
        except Exception:
            if self.timing is not None:
                self.timing.close()
                self.timing = None
            self.root_prior_separated = False
            if 'original_prior' in locals():
                prior.score_device=original_prior
            if 'original_start' in locals():
                self.loop._start_search=original_start
                self.loop._commit_turn=original_commit

    def close(self):
        if self.closed:
            return
        self.closed = True
        for obj in (self.timing, self.loop, self.search, self.rules, self.raw, self.actions,
                    self.random, self.keys, self.seeds):
            closer = getattr(obj, 'close', None)
            if callable(closer):
                closer()


def warmup_inference(inference, loop, rows, max_actions):
    import torch
    device = loop.device
    wx = torch.zeros((rows, inference.feature_dim), dtype=torch.float32, device=device)
    wc = torch.zeros((rows, max_actions, inference.candidate_dim), dtype=torch.float32, device=device)
    wm = torch.zeros((rows, max_actions), dtype=torch.bool, device=device)
    wm[:, 0] = True
    wi = torch.full((rows,), inference.keys.index('starter'), dtype=torch.int64, device=device)
    wa = torch.ones((rows, 1), dtype=torch.float32, device=device)
    streams = (torch.cuda.current_stream(device), loop._outer.capture_stream,
               loop._outer.body_stream, loop._inner.body_stream)
    empty = torch.ones(rows, dtype=torch.bool, device=device)
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            inference.predict_device(wx, wc, wm, wi, wa)
            inference.predict_device(wx, wc[:, :0], wm[:, :0], wi, wa, value_only=empty)
        stream.synchronize()


def run_cohort(cohort, shared, *, deadline, proof_deadline=None, verify=True):
    import torch
    from app.modules.card_game.engine.duel_v2 import new_game
    B, device = shared['rows'], shared['device']
    progress = shared.get('progress', lambda phase: None)
    if time.monotonic() >= deadline:
        raise TimeoutError('Execution budget exhausted before capture')
    progress('initializing_cohort')
    warmup_inference(shared['inference'], cohort.loop, B, shared['max_actions'])
    roots = cohort.rules.initialize_device(shared['parameters'])
    torch.cuda.synchronize()
    if bool((roots.errors != 0).any().cpu()):
        raise AssertionError(['initialize errors', roots.errors.cpu().tolist()])
    initial = host_states(cohort.store, roots)
    expected = [new_game(**row) for row in shared['engine_parameters']]
    from scripts.validate_duel_v2_native_soa import compare_boundary
    for actual,reference in zip(initial,expected):
        compare_boundary(actual,reference,'CUDA initial boundary')
    seed_tensor = torch.arange(shared['seed'], shared['seed'] + B, dtype=torch.int64, device=device)
    ids = torch.tensor(
        [[shared['inference'].keys.index(job['keys']['a']),
          shared['inference'].keys.index(job['keys']['b'])] for job in shared['jobs']],
        dtype=torch.int64, device=device)
    train = torch.zeros((B, 2), dtype=torch.bool, device=device)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    progress('capturing_cohort')
    capture_started = time.monotonic()
    cohort.loop.capture(roots, seed_tensor, ids, ids, train)
    capture_seconds = time.monotonic() - capture_started
    if time.monotonic() >= deadline:
        raise TimeoutError('Execution budget exhausted during capture')
    progress('executing_cohort')
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    replay_started = time.monotonic()
    start_event.record()
    cohort.loop.replay()
    end_event.record()
    end_event.synchronize()
    replay_wall = time.monotonic() - replay_started
    replay_gpu = start_event.elapsed_time(end_event) / 1000.0
    progress('reading_cohort_result')
    result = cohort.loop.result_device()
    searched = cohort.search.result_device()
    stage = cohort.timing.after_batch() if cohort.timing is not None else []
    payload = dict(
        cohort_setup_seconds=capture_started-cohort.started,
        capture_seconds=capture_seconds, replay_wall_seconds=replay_wall,
        replay_gpu_seconds=replay_gpu,
        per_root_requested=searched.requested.cpu().tolist(),
        per_root_completed=searched.completed.cpu().tolist(),
        per_root_world_samples=cohort.search.world_samples.cpu().tolist(),
        per_root_search_errors=searched.errors.cpu().tolist(),
        per_root_game_errors=result.errors.cpu().tolist(),
        per_root_steps=result.steps.cpu().tolist(),
        per_root_complete=result.complete.cpu().tolist(),
        per_root_truncated=result.truncated.cpu().tolist(),
        complete_root_decisions=int(((searched.completed == searched.requested)
                                    & (searched.errors == 0)).sum().cpu()),
        game_error_rows=int((result.errors != 0).sum().cpu()),
        search_error_rows=int((searched.errors != 0).sum().cpu()),
        natural_games=int(result.complete.sum().cpu()),
        truncated_games=int(result.truncated.sum().cpu()),
        error_rows=int(((result.errors != 0) | (searched.errors != 0)).sum().cpu()),
        inner_ticks=int(cohort.loop._inner.counter[0].cpu()),
        outer_ticks=int(cohort.loop._outer.counter[0].cpu()),
        stage_timings=stage, root_prior_separated=cohort.root_prior_separated,
        memory=memory_snapshot(torch, cohort.store, cohort.rules, cohort.actions,
                               cohort.raw, cohort.light),
        graph_executed=int(cohort.loop._outer.counter[0].cpu()) > 0)
    payload['peak_device_search_roots']=int(searched.peak_device_roots.cpu())
    if 'output' in shared:
        (shared['output'] / 'last-replay.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf8')
    if time.monotonic()>deadline:
        raise TimeoutError('Per-trial CUDA execution budget exceeded')
    if not payload['graph_executed']:
        raise AssertionError('Closed nested CUDA WHILE did not execute')
    if payload['game_error_rows'] or payload['search_error_rows']:
        raise AssertionError(['batch errors', payload])
    if bool((cohort.raw.errors_device != 0).any().cpu()):
        raise AssertionError(['raw tape errors', cohort.raw.errors_device.cpu().tolist()])
    payload['root_prior_applicable_row_calls']=int(cohort.guided_row_calls.cpu()[0])
    payload['root_prior_nonzero_row_calls']=int(cohort.nonzero_guided_row_calls.cpu()[0])
    final = None
    if verify:
        progress('verifying_cohort')
        final = host_states(cohort.store, result.states)
        payload['postbatch'] = verify_postbatch(
            cohort, shared, initial, final, result, searched,time.monotonic()+shared['proof_seconds'])
    if not payload['root_prior_separated']:
        payload['root_prior_timing'] = 'included_in_complete_gumbel32_not_separated'
    return payload, initial, final, result, searched


def verify_postbatch(cohort, shared, initial, final, result, searched, deadline):
    import hashlib as _hashlib
    import numpy as np
    import torch
    from app.modules.card_game.engine.duel_v2 import apply_action, acting_side, observe
    from app.modules.card_game.engine.duel_v2.state import finished
    from app.modules.card_game.rl.league_rollout import clean
    from app.modules.card_game.rl.seven_lineup import encode
    from app.modules.card_game.rl.batched_duel.device_rule_calls import _empty_graph
    B, raw, light = shared['rows'], cohort.raw, cohort.light
    counts = result.steps.cpu().tolist()
    if counts != raw.steps_device.cpu().tolist():
        raise AssertionError('Raw tape step counts drifted from physical game steps')
    raw_count = int(raw._record_count.cpu())
    policy_count = int(light._record_count.cpu())
    if raw_count != sum(counts):
        raise AssertionError(['raw record count', raw_count, sum(counts)])
    raw_map, policy_map = {}, {}
    for tape, count, mapping in ((raw, raw_count, raw_map), (light, policy_count, policy_map)):
        games = tape._rec_game[:count].cpu().tolist()
        steps = tape._rec_step[:count].cpu().tolist()
        for index, pair in enumerate(zip(games, steps)):
            if pair in mapping:
                raise AssertionError(['duplicate tape record', pair])
            mapping[pair] = index
    if policy_count!=raw_count or set(raw_map) != set(policy_map):
        raise AssertionError('Raw and policy tape coordinates diverged')
    expected = list(initial)
    last_roots=[None]*B
    records = []
    graph = _empty_graph(torch, cohort.rules.action_layout, cohort.rules.device)
    for step in range(max(counts) if counts else 0):
        if time.monotonic() >= deadline:
            raise TimeoutError('Postbatch proof budget exhausted')
        ids = [raw_map.get((row, step), raw.max_records) for row in range(B)]
        on = torch.tensor([step < counts[row] for row in range(B)], dtype=torch.bool, device=raw.device)
        errors = raw.gather_into_device(torch.tensor(ids, dtype=torch.int64, device=raw.device), graph, on)
        torch.cuda.synchronize()
        if bool((errors != 0).any().cpu()):
            raise AssertionError(['raw gather errors', errors.cpu().tolist()])
        actions = unpack_graph(graph, [step < counts[row] for row in range(B)])
        for row in range(B):
            if step >= counts[row]:
                continue
            rid = ids[row]
            if (int(raw._rec_version[rid].cpu())!=expected[row]['version']
                    or int(raw._rec_owner[rid].cpu())!=row
                    or int(raw._rec_seed[rid].cpu())!=shared['jobs'][row]['seed']):
                raise AssertionError(['raw owner/version/seed mismatch',row,step])
            actor_id = int(raw._rec_actor[rid].cpu())
            actor = 'a' if actor_id == 0 else 'b'
            if actor != acting_side(expected[row]):
                raise AssertionError(['raw actor mismatch', row, step, actor])
            view = observe(expected[row], actor, include_previews=False)
            legal = [entry['action'] for entry in view['legal_actions']
                     if entry['action']['type'] != 'concede']
            action = actions[row]
            if action not in legal:
                raise AssertionError(['raw action not legal', row, step])
            info_hash = None
            chosen = legal.index(action)
            if (row, step) in policy_map:
                pid = policy_map[(row, step)]
                x, candidates = encode(view, legal)
                digest = _hashlib.sha256(x.tobytes() + candidates.tobytes()).digest()
                words = np.frombuffer(digest, dtype='>u4').astype(np.int64).tolist()
                info_hash = light._rec_info[pid].cpu().tolist()
                if info_hash != words:
                    raise AssertionError(['information hash mismatch', row, step])
                if int(light._rec_count[pid].cpu()) != len(legal):
                    raise AssertionError(['policy legal count', row, step])
                if int(light._rec_chosen[pid].cpu()) != chosen:
                    raise AssertionError(['policy chosen mismatch', row, step])
                if int(light._rec_actor[pid].cpu()) != actor_id:
                    raise AssertionError(['policy actor mismatch', row, step])
                requested=int(light._rec_requested[pid].cpu())
                completed=int(light._rec_completed[pid].cpu())
                if requested!=(1 if len(legal)==1 else 32) or completed!=requested:
                    raise AssertionError(['incomplete physical search budget',row,step,requested,completed])
            last_roots[row]=expected[row]
            expected[row] = clean(apply_action(expected[row], actor, action))
            records.append(dict(
                game=row, seed=shared['jobs'][row]['seed'], step=step, actor=actor,
                raw_action=action, legal_count=len(legal), chosen=chosen, info_hash=info_hash,
                requested=requested,completed=completed))
    from scripts.validate_duel_v2_native_soa import compare_boundary
    for left,right in zip(final,expected):
        compare_boundary(left,right,'CUDA raw replay final boundary')
    terminals = [bool(finished(state)) for state in expected]
    if terminals != result.complete.cpu().tolist():
        raise AssertionError(['natural terminal mismatch', terminals, result.complete.cpu().tolist()])
    if 'output' in shared:
        # Save device evidence before invoking the independent Python oracle so
        # a proof/export failure can be investigated without replaying the GPU batch.
        snapshot=dict(states=last_roots, counts=counts, policy_map=[[*key,value] for key,value in policy_map.items()],
            diagnostics={name:value.cpu().tolist() for name,value in cohort.diagnostics.items()},
            tape={name:getattr(light,name).cpu().tolist() for name in
                  ('_rec_count','_rec_offset','_packed_visits','_packed_pi','_rec_chosen')},
            raw_records=records, raw_replay_verified=True)
        path=shared['output']/'postbatch-device-proof.json'
        path.write_text(json.dumps(snapshot,ensure_ascii=False),encoding='utf8')
    numerical=verify_last_searches(cohort,shared,last_roots,counts,policy_map,deadline)
    return dict(verified=True, records=records, natural_terminals=terminals,
                raw_records=raw_count, policy_records=policy_count,
                numerical_root_parity=numerical,
                information_hashes_verified=policy_count == raw_count and raw_count > 0)



class PostbatchValueOracle:
    """Recompute NN forwards after replay; never read search values or device states.

    The Python rule/search oracle uses the same FP32 CUDA arithmetic, separately
    checked against frozen NumPy WDL at the inference component's tolerance.
    This prevents amplifying permitted forward rounding by the Q-scale factor.
    """
    def __init__(self, bank, policies):
        import torch
        self.t = torch
        self.bank = bank
        self.policies = policies
        self.cache = {}
        self.calls = 0
        self.max_abs_error = 0.

    def policy(self, key):
        oracle = self
        base = self.policies[key]
        class Policy:
            def scores(self, x, c):
                return base.scores(x, c)
            def wdl(self, x, is_actor):
                return oracle.wdl(key, x, is_actor)
            def scores_value(self, x, c):
                p = self.wdl(x, True)
                return self.scores(x, c), float(p[2] - p[0])
        return Policy()

    def wdl(self, key, x, is_actor):
        import numpy as np
        cache_key = (key, bool(is_actor), x.tobytes())
        if cache_key not in self.cache:
            t = self.t; bank = self.bank; n = min(2, bank.max_rows)
            xx = t.as_tensor(np.stack([x] * n), device=bank.device)
            candidates = t.empty((n, 0, bank.candidate_dim), device=bank.device)
            valid = t.empty((n, 0), dtype=t.bool, device=bank.device)
            ids = t.full((n,), bank.keys.index(key), dtype=t.int64, device=bank.device)
            context = t.full((n, 1), float(is_actor), device=bank.device)
            value_only = t.ones(n, dtype=t.bool, device=bank.device)
            prediction = bank.predict_device(xx, candidates, valid, ids, context,
                                             value_only=value_only)
            if bool(prediction.invalid_rows.any().cpu()):
                raise AssertionError('Invalid independent postbatch value forward')
            actual = prediction.wdl[0].cpu().numpy()
            expected = self.policies[key].wdl(x, is_actor)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5,
                                       err_msg='Frozen NumPy/CUDA WDL component parity')
            self.max_abs_error = max(self.max_abs_error, float(np.abs(actual - expected).max()))
            self.calls += 1
            self.cache[cache_key] = actual
        return self.cache[cache_key]


def verify_last_searches(cohort,shared,states,counts,policy_map,deadline):
    """Postbatch CPU oracle only; no search decision or rule result is supplied."""
    import numpy as np
    from app.modules.card_game.engine.duel_v2 import acting_side
    from app.modules.card_game.rl import seven_runtime
    from app.modules.card_game.rl.recovery_search import search
    from app.modules.card_game.rl.batched_search.array_zhenhong_prior import ArrayZhenhongPrior
    guide=ArrayZhenhongPrior(device='cpu',allow_cpu_reference=True)
    diagnostics={name:value.detach().cpu().numpy() for name,value in cohort.diagnostics.items()}
    rows=[]
    forwards=PostbatchValueOracle(shared['inference'],shared['models'])
    for row,state in enumerate(states):
        if state is None or time.monotonic()>=deadline:
            raise TimeoutError('Full numerical search proof incomplete')
        actor=acting_side(state);step=counts[row]-1
        seed=(1<<256)|int.from_bytes(hashlib.sha256(
            ('recovery:%s:%s:%s'%(shared['jobs'][row]['seed'],step,actor)).encode()).digest(),'big')
        policies={side:forwards.policy(key) for side,key in shared['jobs'][row]['keys'].items()}
        result=search(state,actor,policies,seed=seed,runtime=seven_runtime,simulations=32,
            gumbel_candidates=16,terminal_horizon=0,teacher_mode='network',max_depth=10,
            noise=True,q_scale='natural_wdl',root_prior=guide,
            world_sampling='per_decision',
            deadline=time.time()+max(0,deadline-time.monotonic()))
        if not result['complete']:
            raise AssertionError('CPU numerical oracle failed to finish its declared budget')
        pid=policy_map[(row,step)]
        tape=cohort.light
        count=int(tape._rec_count[pid].cpu());offset=int(tape._rec_offset[pid].cpu())
        visits=tape._packed_visits[offset:offset+count].cpu().numpy()
        pi=tape._packed_pi[offset:offset+count].cpu().numpy()
        checks=dict(visits=np.array_equal(visits,result['visits']),
            pi=np.allclose(pi,result['pi'],rtol=1e-5,atol=1e-6),
            completed_q=np.allclose(diagnostics['q'][row,:count],result['root_diagnostics']['completed_q'],rtol=1e-5,atol=1e-6),
            prior=np.allclose(diagnostics['prior'][row,:count],result['root_prior'],rtol=1e-5,atol=1e-6),
            mean=np.allclose(diagnostics['mean'][row,:count],result['mean_values'],rtol=1e-5,atol=1e-6),
            value=np.isclose(diagnostics['value'][row],result['value'],rtol=1e-5,atol=1e-6),
            gumbel=np.allclose(diagnostics['gumbel'][row,:count],result['root_gumbel'],rtol=1e-12,atol=1e-12),
            choice=int(tape._rec_chosen[pid].cpu())==result['search_choice'],
            snapshot=not diagnostics['errors'][row])
        checks={key:bool(value) for key,value in checks.items()}
        rows.append(dict(row=row,step=step,seed=str(seed),checks=checks,visits=visits.tolist(),
            pi=pi.tolist(),completed_q=diagnostics['q'][row,:count].tolist(),
            reference_completed_q=np.asarray(result['root_diagnostics']['completed_q']).tolist(),
            value_forward_checks=forwards.calls,
            max_numpy_cuda_wdl_abs_error=forwards.max_abs_error,
            reference_value_backend='independent_postbatch_cuda_forward_numpy_checked'))
        if not all(checks.values()):
            raise AssertionError(['CUDA complete32 numerical parity',rows[-1]])
    return rows


def run(args):
    import torch
    from contextlib import ExitStack
    from app.modules.card_game.rl import seven_runtime, cross_teacher_distillation
    from app.modules.card_game.rl.batched_duel.native_array_program import load_native_array_program
    from app.modules.card_game.rl.batched_duel.vm_image import lower_program
    from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
    from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
    from app.modules.card_game.rl.batched_duel.native_soa_cuda import CudaSoARules
    from app.modules.card_game.rl.batched_search.device_value_only_inference import DeviceValueOnlyInference
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no host loop fallback')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise RuntimeError('CUDA-only acceptance; CPU fallback is forbidden')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    torch.cuda.set_device(device)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(
        mode='run',
        scope='bounded CUDA DeviceSoARules DeviceGameLoop acceptance; not training or CPU fallback',
        passed=False, rows=args.rows, max_decisions=args.max_decisions,
        require_natural=bool(args.require_natural), device=str(device),
        warmup=args.warmup, repeats=args.repeats, seed=args.seed, presets=list(KEYS),
        search=dict(SEARCH), learning_updates=0, gpu_qualified=False,
        one_step_cannot_qualify_natural=True, skip_mulligan=True, mixed_first_sides=True)
    report['static'] = dict(
        search=dict(SEARCH), rows=args.rows, max_decisions=args.max_decisions,
        max_actions=args.max_actions, inner_max_ticks=args.inner_max_ticks,
        rule_steps=args.rule_steps, seed=args.seed, warmup=args.warmup, repeats=args.repeats,
        require_natural=bool(args.require_natural), skip_mulligan=True, mixed_first_sides=True,
        presets=list(KEYS), cpu_fallback=False, training=False)

    def checkpoint(phase):
        save_report(args.output, report, phase, started)

    checkpoint('start')
    try:
        jobs = parameters_for(args.rows, args.seed)
        cross_teacher_distillation.REPO_TEACHERS = args.teachers
        models = {key: seven_runtime.SevenModel(args.models, key) for key in KEYS}
        report['weight_sha256'] = {
            key: hashlib.sha256((args.models / (key + '.npz')).read_bytes()).hexdigest()
            for key in models}
        report['model_versions'] = {key: models[key].version for key in models}
        program = load_native_array_program(args.program, repo_root=ROOT)
        image = lower_program(program)
        schema = compile_execution_schema()
        report.update(program_identity=program.identity, image_identity=image.identity,
                      schema_identity=schema.identity, gpu_name=torch.cuda.get_device_name(device),
                      torch=torch.__version__)
        checkpoint('program')
        executor = CudaSoARules(image, schema, device=str(device),
                                limits=VmLimits(steps=args.rule_steps))
        report.update(native_source_sha256=executor.source_sha256,
                      cuda_compilation_profile=executor.module.compilation_profile,
                      cuda_compile_options=list(executor.module.compile_options),
                      rule_threads_per_block=executor.threads)
        inference = DeviceValueOnlyInference(
            models, device=str(device), max_rows=args.rows,
            max_candidate_elements=args.rows * args.max_actions * 1576)
        engine_parameters, parameters = [], []
        for job in jobs:
            decks = {'a': models[job['keys']['a']].serving_deck,
                     'b': models[job['keys']['b']].serving_deck}
            row = dict(seed=job['seed'], decks=decks, first_side=job['first_side'],
                       skip_mulligan=True, escalation=True)
            engine_parameters.append(row)
            parameters.append(row)
        shared = dict(
            rows=args.rows, device=str(device), schema=schema, executor=executor,
            output=args.output, progress=checkpoint,
            inference=inference, seed=args.seed, max_actions=args.max_actions,models=models,
            proof_seconds=args.proof_seconds,
            inner_max_ticks=args.inner_max_ticks, jobs=jobs, parameters=parameters,
            engine_parameters=engine_parameters)
        report['setup_seconds'] = time.monotonic() - started
        if report['setup_seconds'] > args.setup_seconds:
            raise TimeoutError('Setup hard deadline exceeded before capture')
        checkpoint('setup')
        timings = []
        with ExitStack() as cleanup:
            cleanup.callback(inference.close)
            cleanup.callback(executor.close)
            for trial in range(args.warmup + args.repeats):
                exec_deadline=time.monotonic()+args.execution_seconds
                with ExitStack() as trial_cleanup:
                    cohort = IsolatedCohort(shared, max_decisions=1, seconds=args.execution_seconds)
                    trial_cleanup.callback(cohort.close)
                    payload, initial, final, result, searched = run_cohort(
                        cohort, shared, deadline=exec_deadline, verify=True)
                    if payload['per_root_requested']!=[32]*args.rows or payload['per_root_completed']!=[32]*args.rows:
                        raise AssertionError('One-step timing requires every independent root to finish exactly32')
                    if payload['per_root_world_samples'] != [1]*args.rows:
                        raise AssertionError('Each physical decision must sample exactly one hidden world')
                    timings.append(dict(
                        trial=trial, warmup=trial < args.warmup, max_decisions=1,
                        capture_seconds=payload['capture_seconds'],
                        replay_wall_seconds=payload['replay_wall_seconds'],
                        replay_gpu_seconds=payload['replay_gpu_seconds'],
                        complete_root_decisions=payload['complete_root_decisions'],
                        per_root_requested=payload['per_root_requested'],per_root_completed=payload['per_root_completed'],
                        per_root_world_samples=payload['per_root_world_samples'],
                        graph_executed=payload['graph_executed'],
                        root_prior_separated=payload['root_prior_separated'],
                        stage_timings=payload['stage_timings'], memory=payload['memory']))
                    report['one_step_timings'] = timings
                    checkpoint('one_step_timing')
                # A callback bound to a cohort retains its large state/work buffers.
                # Drop that callback and all device receipts before allocating the next trial.
                del cohort, result, searched
                gc.collect()
                torch.cuda.empty_cache()
            measured = [row for row in timings if not row['warmup']]
            if len(measured) != args.repeats:
                raise AssertionError('One-step repeats incomplete')
            report['one_step_repeat_seconds'] = [row['replay_gpu_seconds'] for row in measured]
            report['one_step_capture_seconds'] = [row['capture_seconds'] for row in measured]
            if args.max_decisions != 1:
                exec_deadline=time.monotonic()+args.execution_seconds
                acceptance = IsolatedCohort(
                    shared, max_decisions=args.max_decisions, seconds=args.execution_seconds)
                cleanup.callback(acceptance.close)
                payload, initial, final, result, searched = run_cohort(
                    acceptance, shared, deadline=exec_deadline)
            report.update(payload)
            report['per_root'] = [dict(
                row=i, seed=jobs[i]['seed'], first=jobs[i]['first_side'], decks=jobs[i]['keys'],
                requested=payload['per_root_requested'][i],
                completed=payload['per_root_completed'][i],
                search_errors=payload['per_root_search_errors'][i],
                game_errors=payload['per_root_game_errors'][i],
                steps=payload['per_root_steps'][i], complete=payload['per_root_complete'][i],
                truncated=payload['per_root_truncated'][i], natural=payload['per_root_complete'][i])
                for i in range(args.rows)]
            (args.output / 'raw-replay.json').write_text(
                json.dumps(dict(initial=initial, final=final, records=payload['postbatch']['records']),
                           ensure_ascii=False), encoding='utf8')
            if args.require_natural and payload['natural_games'] != args.rows:
                raise AssertionError('require-natural rejected truncated or incomplete games')
            report['natural_claim_from_step1'] = False
            report['passed'] = (
                payload['graph_executed'] and payload['game_error_rows'] == 0
                and payload['search_error_rows'] == 0 and payload['postbatch']['verified']
                and (payload['natural_games'] == args.rows if args.require_natural else True))
            checkpoint('complete')
        report['torch_peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        report['torch_peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved())
        checkpoint(report.get('phase', 'complete'))
        if not report['passed']:
            raise SystemExit(1)
    except BaseException:
        report['passed'] = False
        report['error'] = traceback.format_exc()
        if torch.cuda.is_available():
            report['torch_peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
            report['torch_peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved())
        checkpoint('failed')
        raise
    finally:
        save_report(args.output, report, report.get('phase', 'failed'), started)
        print(json.dumps(compact_summary(report), ensure_ascii=False, indent=2))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_bounds(parser, args)
    if not args.run:
        print(json.dumps(plan(args), ensure_ascii=False, indent=2))
        return 0
    validate_run(parser, args)
    run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
