"""Collector-side CUDA DeviceFrozenRules composition. Not a qualified executor.

Binds an already-allocated same-source frozen VM, stable compact state/action
arenas, and prepared shared scratch into the DeviceRuleExecutor surface used by
DeviceGumbelSearch / DeviceGameLoop. Helper call plans capture one static site
once; GPU replay reuses those pointers and the shared VM workspace sequentially.

This module never claims full duel-rule readiness, receipts, or 1600-root proof.
"""
from dataclasses import dataclass

from ..batched_duel.cuda_runtime import CudaModule
from ..batched_duel.cuda_vm import CudaVm, VmCapabilityError
from ..batched_duel.device_graph_storage import CompactDeviceGraphStore
from ..batched_duel.device_rule_calls import ALLOWED_ENTRIES, DeviceRuleGraphCalls
from ..batched_duel.device_rule_lanes import DeviceRuleLanes
from ..batched_duel.device_rule_presentation import DeviceRulePresentation
from ..batched_duel.device_rule_protocol import DeviceDecision, DeviceStateBatch
from ..batched_duel.device_rule_results import DeviceRuleResultReader
from ..batched_duel.vm_heap import ERROR_INDEX_OUT_OF_BOUNDS, emit_vm_heap_source
from .compact_device_candidates import CompactDeviceCandidates


TEACHER_HORIZON = 3
ROOT_MAX_ACTIONS = 800
TEACHER_INDEX_ERROR = 154
OPERATIONS = (
    'metadata_device',
    'decision_device',
    'rule_decision_device',
    'value_observation_device',
    'sample_world_device',
    'step_device',
    'clean_device',
    'teacher_continue_device',
    'release_device',
)
REQUIRED_CALL_ENTRIES = (
    'clean', 'teacher_continue', 'sample_world', 'simulate_action', 'apply_action',
)
REQUIRED_PRESENTATION_ENTRIES = (
    'acting_side', 'finished', 'public_legal_decision', 'terminal_value',
)
REQUIRED_TEACHER_ENTRY = 'rule_decision'
DEFAULT_SITES = {
    'metadata': 9,
    'decision': 2,
    'rule_decision': 1,
    'value': 4,
    'sample_world': 1,
    'apply_action': 1,
    'simulate_action': 1,
    'clean': 2,
    'teacher_continue': 1,
}

_INT_SPEC = (
    'int32*', 'int32*', 'uint8*', 'int32*', 'int32*', 'int32*', 'uint8*',
    'int32', 'int32', 'int32', 'int32*', 'int32*', 'uint8*',
    'int64*', 'int32*', 'int32', 'int32', 'int32', 'int32',
)
_INT_SOURCE = emit_vm_heap_source() + r'''
using namespace duel_vm_heap;
const int RESULT_INVALID=154;
__device__ inline int integer64(Heap* h,int ref,long long* result){
 NodeView n;if(!get_node(h,ref,&n)||n.tag!=TAG_INT||n.count<1)return 0;
 const unsigned char* p=get_payload_ptr(h,&n);if(!p)return 0;
 int negative=(p[n.count-1]&128)!=0;
 unsigned long long bits=negative?~0ULL:0ULL;
 for(int i=0;i<n.count&&i<8;i++)bits=(bits&~(255ULL<<(8*i)))|((unsigned long long)p[i]<<(8*i));
 if(((bits>>63)!=0)!=negative)return 0;
 for(int i=8;i<n.count;i++)if(p[i]!=(negative?255:0))return 0;
 *result=(long long)bits;return 1;
}
extern "C" __global__ void decode_int64(
 int* nodes,int* edges,unsigned char* payload,int* counts,const int* cn,const int* ce,const unsigned char* cp,
 int cnc,int cec,int cpc,const int* refs,const int* input_errors,const unsigned char* active,
 long long* values,int* errors,int rows,int maxn,int maxe,int maxp){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
 values[row]=-1;errors[row]=input_errors[row];if(!active[row]||errors[row])return;
 int* c=counts+row*3;
 if(c[0]<1||c[0]>maxn||c[1]<0||c[1]>maxe||c[2]<0||c[2]>maxp){errors[row]=RESULT_INVALID;return;}
 Heap h;init_heap(&h,nodes+(long long)row*maxn*4,edges+(long long)row*maxe*2,
  payload+(long long)row*maxp,c,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);
 long long value=0;
 if(!integer64(&h,refs[row],&value)||value<0){errors[row]=RESULT_INVALID;return;}
 values[row]=value;
}
'''


class FrozenRuleGap(RuntimeError):
    """Required helper or frozen-VM capability is missing. Not a readiness receipt."""


def source_teacher_horizon():
    """CPU-static horizon of the canonical frozen teacher_continue source."""
    from ..batched_duel.gpu_rule_entries import teacher_continue
    consts = teacher_continue.__code__.co_consts
    if TEACHER_HORIZON not in consts:
        return None
    return TEACHER_HORIZON


def inspect_device_frozen_rule_dependencies(calls, presentation, action_store):
    """Exact missing capabilities for primary integration. Never grants ready."""
    gaps = []
    if not isinstance(calls, DeviceRuleGraphCalls) or getattr(calls, 'closed', True):
        gaps.append('live DeviceRuleGraphCalls (or PhysicalActionRecordingCalls subclass) required')
        return gaps
    if not isinstance(presentation, DeviceRulePresentation) or getattr(presentation, 'closed', True):
        gaps.append('live DeviceRulePresentation required')
        return gaps
    if getattr(presentation, 'calls', None) is not calls:
        gaps.append('presentation.calls must be the injected DeviceRuleGraphCalls object')
    if getattr(calls, 'supports_full_duel_rules', None) is True:
        gaps.append('injected calls must not claim supports_full_duel_rules')
    if getattr(presentation, 'supports_full_duel_rules', None) is True:
        gaps.append('injected presentation must not claim supports_full_duel_rules')
    vm = getattr(calls, 'vm', None)
    live_vm = isinstance(vm, CudaVm) and not vm.closed
    if not live_vm:
        gaps.append('live same-source frozen CUDA VM required')
    call_device = getattr(calls, 'device', None)
    presentation_device = getattr(presentation, 'device', None)
    vm_device = getattr(vm, 'device', None)
    if getattr(call_device, 'type', None) != 'cuda' or getattr(vm_device, 'type', None) != 'cuda':
        gaps.append('actual CUDA required; no CPU DeviceFrozenRules fallback')
    if live_vm and (presentation_device != call_device or vm_device != call_device):
        gaps.append('VM, graph calls and presentation must share the CUDA device')
    if getattr(presentation, 'rows', None) != getattr(calls, 'rows', None):
        gaps.append('presentation.rows must match calls.rows')
    max_actions = getattr(presentation, 'max_actions', None)
    if max_actions != ROOT_MAX_ACTIONS:
        gaps.append('presentation.max_actions is %s, required root maxM %s'
                    % (max_actions, ROOT_MAX_ACTIONS))
    if not isinstance(action_store, CompactDeviceGraphStore) or getattr(action_store, 'closed', True):
        gaps.append('live compact action graph store required')
    else:
        if action_store is getattr(calls, 'state_store', None):
            gaps.append('action arena must be independent from the state arena')
        if getattr(action_store, 'device', None) != call_device:
            gaps.append('action arena must share the CUDA device')
    if not callable(getattr(calls, 'prepare_call', None)):
        gaps.append('DeviceRuleGraphCalls.prepare_call missing')
    missing_allowed = [name for name in REQUIRED_CALL_ENTRIES if name not in ALLOWED_ENTRIES]
    if missing_allowed:
        gaps.append('DeviceRuleGraphCalls.ALLOWED_ENTRIES missing %s' % ','.join(missing_allowed))
    for name in ('prepare_metadata', 'prepare_decision', 'prepare_value', 'release_decision_device'):
        if not callable(getattr(presentation, name, None)):
            gaps.append('DeviceRulePresentation.%s missing' % name)
    if not callable(getattr(calls, 'release_device', None)):
        gaps.append('DeviceRuleGraphCalls.release_device missing')
    if not isinstance(getattr(presentation, 'reader', None), DeviceRuleResultReader):
        gaps.append('DeviceRulePresentation.reader DeviceRuleResultReader required')
    entries = getattr(getattr(vm, 'image', None), 'entries', {}) or {}
    for name in REQUIRED_CALL_ENTRIES + REQUIRED_PRESENTATION_ENTRIES + (REQUIRED_TEACHER_ENTRY,):
        if name not in entries:
            gaps.append('missing frozen VM entry %s' % name)
            continue
        if not live_vm:
            continue
        try:
            cap = vm.capability(name)
        except (TypeError, ValueError, RuntimeError, VmCapabilityError) as exc:
            gaps.append('frozen VM entry %s capability error: %s' % (name, exc))
            continue
        if not cap.get('ready'):
            gaps.append('frozen VM entry %s is not ready: %s' % (name, cap.get('missing')))
    horizon = source_teacher_horizon()
    if horizon != TEACHER_HORIZON:
        gaps.append('canonical teacher_continue source is not globally configured horizon %s'
                    % TEACHER_HORIZON)
    return gaps


def _positive_sites(sites):
    if sites is None:
        chosen = dict(DEFAULT_SITES)
    else:
        if not isinstance(sites, dict):
            raise TypeError('sites must be a mapping of operation to static capture quota')
        unknown = sorted(set(sites) - set(DEFAULT_SITES))
        if unknown:
            raise ValueError('Unknown DeviceFrozenRules sites: ' + ','.join(unknown))
        chosen = dict(DEFAULT_SITES)
        chosen.update(sites)
    for name, value in chosen.items():
        if type(value) is not int or value < 1:
            raise ValueError('Positive static site quota required for ' + name)
    return chosen


class _PlanQueue:
    def __init__(self, name, plans):
        self.name = name
        self.plans = list(plans)
        self.index = 0

    def consume(self):
        if self.index >= len(self.plans):
            raise RuntimeError('No remaining prepared %s capture site' % self.name)
        plan = self.plans[self.index]
        self.index += 1
        return plan


class _ActiveRuleLanes(DeviceRuleLanes):
    """Skip a whole scratch queue when its device cohort has no live leases.

    The enclosing plan resets its borrowed outputs before dispatch. Preserve
    errored active leases for its usual checked error propagation.
    """
    def capture_dispatch(self, states, operation):
        if self.closed:
            raise RuntimeError('Rule scratch dispatcher closed')
        self._validate_states(states, self.rows)
        if not callable(operation):
            raise TypeError('Static operation callback required')
        if not self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Rule dispatch must be embedded in CUDA capture')

        def initialize():
            self.cursor.zero_()
            self.pending.copy_(states.active.any().reshape(1).to(self.t.uint8))
            self.processed.zero_()
            self.peak.zero_()

        def body():
            local = self.select_states_device(states)
            operation(local, self.ids, self.valid)
            self.module.launch('advance_rule_lanes',
                [self.cursor, self.pending, self.rows, self.lanes],
                grid=1, block=1, capture_safe=True)

        self.loop.capture_loop(self.pending, body, initialize=initialize)


def _active_plan(plan):
    """Specialize collector-owned plans without changing the shared helper."""
    old = plan.lanes
    replacement = _ActiveRuleLanes(old.rows, old.lanes, device=str(old.device),
                                  seconds=old.loop.budget_ns / 1e9)
    plan.lanes = replacement
    old.close()
    return plan


class DeviceTeacherChoicePlan:
    """Launch frozen rule_decision and decode the teacher index on CUDA."""

    def __init__(self, rules):
        c = rules.calls
        p = rules.presentation
        t = c.t
        self.rules = rules
        self.captured = False
        self.closed = False
        self.lanes = _ActiveRuleLanes(c.rows, c.lanes, device=c.device, seconds=c.seconds)
        self.refs = t.empty((c.lanes, 2), dtype=t.int32, device=c.device)
        self.load_errors = t.empty(c.lanes, dtype=t.int32, device=c.device)
        self.bind_errors = t.empty(c.lanes, dtype=t.int32, device=c.device)
        self.actor_rows = t.empty(c.lanes, dtype=t.int64, device=c.device)
        self.local_choices = t.empty(c.lanes, dtype=t.int64, device=c.device)
        self.local_errors = t.empty(c.lanes, dtype=t.int32, device=c.device)
        self.pair = p.reader.prepare_pair_workspace(c.lanes)
        self.choices = t.full((c.rows,), -1, dtype=t.int64, device=c.device)
        self.errors = t.zeros(c.rows, dtype=t.int32, device=c.device)

    def close(self):
        if not self.closed:
            self.lanes.close()
            self.closed = True

    def capture_device(self, states, observers):
        rules = self.rules
        c = rules.calls
        p = rules.presentation
        t = c.t
        if self.closed or rules.closed or c.closed or p.closed:
            raise RuntimeError('Teacher choice plan closed')
        if self.captured:
            raise RuntimeError('Teacher choice plan already captured')
        if not t.cuda.is_current_stream_capturing():
            raise RuntimeError('Embed teacher choice in CUDA capture')
        c._validate_states(states, c.rows, pool=c.state_store)
        c._exact(observers, t.int64, (c.rows,), 'teacher observers')
        self.choices.fill_(-1)
        self.errors.copy_(states.errors)

        def operation(local, ids, valid):
            gathered = c.state_store.gather_into_device(local, c.input_graph)
            errors = local.errors | gathered.errors
            active = (local.active & (errors == 0)).view(t.uint8)
            c.vm.load_graph_into_device(c.workspace, c.input_graph, active=active, errors=self.load_errors)
            errors = errors | self.load_errors
            active = (local.active & (errors == 0)).view(t.uint8)
            self.actor_rows.copy_(t.where(valid.bool(), observers[ids], 0))
            c.binder.bind_into_device(
                c.large, self.actor_rows, active, argument_refs=self.refs, errors=self.bind_errors)
            errors = errors | self.bind_errors
            active = (local.active & (errors == 0)).view(t.uint8)
            call = c.vm.launch_into_device(
                c.workspace, REQUIRED_TEACHER_ENTRY, argument_refs=self.refs,
                active=active, gpu_deadlines=c.deadlines)
            errors = errors | call.statuses[:, 0]
            active = (local.active & (errors == 0)).view(t.uint8)
            pair = p.reader.pair_into_device(call, active=active, output=self.pair)
            errors = errors | pair.errors
            active = (local.active & (errors == 0)).view(t.uint8)
            rules._decode_teacher_index(
                call.arena, pair.second, errors.contiguous(), active,
                self.local_choices, self.local_errors)
            errors = errors | self.local_errors
            self.lanes.scatter_tensor_device(self.local_choices, self.choices)
            self.lanes.scatter_tensor_device(errors, self.errors)

        self.lanes.capture_dispatch(states, operation)
        guard = (self.lanes.loop.errors != 0).to(t.int32)[0] * 152
        self.errors.bitwise_or_(states.active.to(t.int32) * guard)
        self.captured = True
        return self.choices, self.errors


@dataclass
class _TeacherDecisionSite:
    decision: object
    choice: object
    combined_errors: object

    def capture_device(self, states, observers):
        decision = self.decision.capture_device(states, observers)
        choices, errors = self.choice.capture_device(states, observers)
        combined = self.combined_errors
        combined.copy_(decision.errors)
        combined.bitwise_or_(errors)
        width = int(decision.action_refs.shape[1])
        in_range = (choices >= 0) & (choices < decision.counts) & (choices < width)
        safe = choices.clamp(0, max(width - 1, 0))
        picked = decision.valid.gather(1, safe.unsqueeze(1)).squeeze(1)
        live = states.active & (combined == 0)
        bad = live & (~in_range | ~picked)
        combined.bitwise_or_(bad.to(combined.dtype) * ERROR_INDEX_OUT_OF_BOUNDS)
        return decision, choices, combined


class DeviceFrozenRules:
    """CUDA-only RuleExecutor composition over prepared helper plans.

    Dependencies (already created; this class does not construct the VM):
      calls: DeviceRuleGraphCalls or PhysicalActionRecordingCalls
      presentation: DeviceRulePresentation wrapping the same calls
      action_store: independent CompactDeviceGraphStore for legal action graphs
      sites: per-operation static CUDA capture quotas (see DEFAULT_SITES)

    Operations: metadata, full legal decision, teacher rule_decision, fixed-viewer
    value, sample_world int64[B,5], physical apply vs hypothetical simulate,
    clean, horizon-3 teacher_continue, checked release.
    """

    supports_full_duel_rules = False
    host_rule_callbacks = False
    teacher_horizon = TEACHER_HORIZON
    root_max_actions = ROOT_MAX_ACTIONS
    operations = OPERATIONS
    execution_kind = 'device_frozen_rules_composition'

    def __init__(self, calls, presentation, *, action_store, sites=None):
        if not isinstance(calls, DeviceRuleGraphCalls) or calls.closed:
            raise TypeError('Live DeviceRuleGraphCalls (or PhysicalActionRecordingCalls subclass) required')
        if not isinstance(presentation, DeviceRulePresentation) or presentation.closed:
            raise TypeError('Live DeviceRulePresentation required')
        if not isinstance(action_store, CompactDeviceGraphStore) or action_store.closed:
            raise TypeError('Live compact action graph store required')
        gaps = inspect_device_frozen_rule_dependencies(calls, presentation, action_store)
        if gaps:
            raise FrozenRuleGap('Frozen DeviceFrozenRules gap: ' + '; '.join(gaps))
        t = calls.t
        if t.cuda.is_current_stream_capturing():
            raise RuntimeError('Allocate DeviceFrozenRules scratch and plans before CUDA capture')
        self.calls = calls
        self.presentation = presentation
        self.action_store = action_store
        self.vm = calls.vm
        self.t = t
        self.device = calls.device
        self.rows = calls.rows
        self.sites = _positive_sites(sites)
        self.closed = False
        self.gaps = ()
        self._decoder = None
        self._queues = {}
        self._teacher_sites = []
        self._owned = []
        try:
            self._decoder = CudaModule(
                _INT_SOURCE, {'decode_int64': _INT_SPEC}, device=str(self.device))
            self._owned.append(self._decoder)
            self._queues['metadata'] = _PlanQueue(
                'metadata', [_active_plan(presentation.prepare_metadata()) for _ in range(self.sites['metadata'])])
            decisions = []
            for _ in range(self.sites['decision']):
                plan = _active_plan(presentation.prepare_decision(action_store))
                self._require_compact_decision_plan(plan)
                decisions.append(plan)
            self._queues['decision'] = _PlanQueue('decision', decisions)
            self._queues['value'] = _PlanQueue(
                'value', [_active_plan(presentation.prepare_value()) for _ in range(self.sites['value'])])
            for entry in ('sample_world', 'apply_action', 'simulate_action', 'clean', 'teacher_continue'):
                plans = [_active_plan(calls.prepare_call(entry)) for _ in range(self.sites[entry])]
                self._queues[entry] = _PlanQueue(entry, plans)
            teacher = []
            for _ in range(self.sites['rule_decision']):
                decision = _active_plan(presentation.prepare_decision(action_store))
                self._require_compact_decision_plan(decision)
                choice = DeviceTeacherChoicePlan(self)
                self._owned.append(choice)
                combined = t.zeros(self.rows, dtype=t.int32, device=self.device)
                teacher.append(_TeacherDecisionSite(decision, choice, combined))
            self._queues['rule_decision'] = _PlanQueue('rule_decision', teacher)
            self._teacher_sites = teacher
        except BaseException:
            self.close()
            raise
        self.prepared = {
            name: len(queue.plans) for name, queue in self._queues.items()
        }

    def close(self):
        if self.closed:
            return
        for item in self._owned:
            closer = getattr(item, 'close', None)
            if callable(closer):
                closer()
        self._owned.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _require_compact_decision_plan(self, plan):
        tokens = getattr(getattr(plan, 'candidates', None), 'tokens', None)
        if not isinstance(getattr(plan, 'candidates', None), CompactDeviceCandidates):
            raise FrozenRuleGap('decision plan must keep compact tokens, not dense BxMxC clones')
        if getattr(plan, 'action_types', None) is None or getattr(plan, 'action_refs', None) is None:
            raise FrozenRuleGap('decision plan missing action_types/action_refs')
        t = self.t
        if plan.action_types.dtype != t.int64 or plan.action_refs.dtype != t.int32:
            raise FrozenRuleGap('decision action_types must be int64 and action_refs int32')
        if plan.action_types.shape != (self.rows, ROOT_MAX_ACTIONS):
            raise FrozenRuleGap('root decision maxM must be %s full tokens' % ROOT_MAX_ACTIONS)
        if tokens is None:
            raise FrozenRuleGap('compact candidate tokens required')

    def _consume(self, name):
        if self.closed or self.calls.closed or self.presentation.closed:
            raise RuntimeError('DeviceFrozenRules closed')
        if not self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('DeviceFrozenRules methods must be embedded in CUDA capture')
        return self._queues[name].consume()

    def _decode_teacher_index(self, arena, refs, input_errors, active, values, errors):
        if self.closed or self._decoder is None:
            raise RuntimeError('Teacher index decoder closed')
        self.vm._validate_device_graph(arena)
        t = self.t
        rows = arena.rows
        for value, dtype in ((refs, t.int32), (input_errors, t.int32), (active, t.uint8),
                             (values, t.int64), (errors, t.int32)):
            if (not isinstance(value, t.Tensor) or value.device != self.device
                    or value.dtype != dtype or value.shape != (rows,) or not value.is_contiguous()):
                raise ValueError('Exact teacher-index vectors required')
        b = self.vm._buffers
        layout = arena.layout
        self._decoder.launch(
            'decode_int64',
            [arena.nodes, arena.edges, arena.payload, arena.counts,
             b['nodes'], b['edges'], b['payload'],
             len(self.vm.image.nodes), len(self.vm.image.edges), len(self.vm.image.payload),
             refs, input_errors, active, values, errors,
             rows, layout.max_nodes, layout.max_edges, layout.max_payload_bytes],
            grid=(rows + 127) // 128, block=128, capture_safe=True)
        return values, errors

    def metadata_device(self, states):
        return self._consume('metadata').capture_device(states)

    def decision_device(self, states, observers):
        decision = self._consume('decision').capture_device(states, observers)
        if not isinstance(decision.candidates, CompactDeviceCandidates):
            raise FrozenRuleGap('decision_device must return compact candidates')
        return decision

    def rule_decision_device(self, states, observers):
        return self._consume('rule_decision').capture_device(states, observers)

    def value_observation_device(self, states, viewers):
        return self._consume('value').capture_device(states, viewers)

    def sample_world_device(self, states, viewers, seeds):
        self.calls._exact(seeds, self.t.int64, (self.rows, 5), 'seeds')
        return self._consume('sample_world').capture_device(states, actors=viewers, seeds=seeds)

    def step_device(self, states, actors, decision, action_indices, *, physical=False):
        if type(physical) is not bool:
            raise TypeError('step_device physical must be an explicit bool')
        if not isinstance(decision, DeviceDecision):
            raise TypeError('DeviceDecision required')
        entry = 'apply_action' if physical else 'simulate_action'
        return self._consume(entry).capture_device(
            states, actors=actors, decision=decision, action_indices=action_indices)

    def clean_device(self, states):
        return self._consume('clean').capture_device(states)

    def teacher_continue_device(self, states, viewers, *, horizon=3):
        if type(horizon) is not int or horizon != TEACHER_HORIZON:
            raise ValueError(
                'Frozen teacher_continue is globally configured horizon %s; refusing horizon %r'
                % (TEACHER_HORIZON, horizon))
        return self._consume('teacher_continue').capture_device(states)

    def release_device(self, states):
        if self.closed or self.calls.closed:
            raise RuntimeError('DeviceFrozenRules closed')
        if not isinstance(states, DeviceStateBatch):
            raise TypeError('DeviceStateBatch required')
        self.calls.release_device(states)

