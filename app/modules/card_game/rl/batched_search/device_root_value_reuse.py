"""Opt-in source-bound root WDL reuse; never a general value cache.

Only a fresh canonical own-view, nonterminal root can qualify. The caller
must retain the ordinary fixed-viewer value path for every ineligible row.
This helper has no full-rule/search/readiness qualification.
"""
from ..batched_duel.device_rule_presentation import DeviceRulePresentation
from ..batched_duel.gpu_rule_projection import GraphProjectionKernel
from ..batched_duel.cuda_vm import CudaVm
from ..batched_duel.device_rule_calls import DeviceRuleGraphCalls
from .device_frozen_rules import DeviceFrozenRules
from .device_int32_presentation import DeviceInt32Presentation, _NarrowingProjection
from .device_inference import DeviceSevenInference
from .device_memory_bounded_inference import DeviceMemoryBoundInference
from .device_value_only_inference import DeviceValueOnlyInference

_CANONICAL_ENTRIES = (
    'func:app.modules.card_game.rl.batched_duel.gpu_rule_entries:public_legal_decision:',
    'func:app.modules.card_game.rl.batched_duel.gpu_rule_entries:seven_public_legal_decision:',
    'func:app.modules.card_game.rl.batched_duel.gpu_public_decision_reuse:public_legal_decision_reuse:',
)


def inspect_root_value_reuse(rules, inference, auxiliary):
    """Bind the current verified frozen public entry and identical projection."""
    gaps = []
    if type(rules) is not DeviceFrozenRules or getattr(rules, 'closed', True):
        return ('exact live DeviceFrozenRules required',)
    if type(inference) not in (DeviceSevenInference, DeviceMemoryBoundInference, DeviceValueOnlyInference):
        gaps.append('known frozen numerical inference required')
    if auxiliary is not inference:
        gaps.append('identical policy/value numerical bank required')
    p, c = rules.presentation, rules.calls
    if not isinstance(c, DeviceRuleGraphCalls) or not isinstance(c.vm, CudaVm):
        return tuple(gaps + ['actual frozen CUDA rule calls required'])
    if type(p) not in (DeviceRulePresentation, DeviceInt32Presentation) or p.calls is not c or p.vm is not c.vm:
        gaps.append('same canonical presentation/VM required')
    projection = p.projection
    if type(projection) is _NarrowingProjection: projection = projection.projection
    if type(projection) is not GraphProjectionKernel or projection.runtime != 'seven':
        gaps.append('exact shared seven observation projection required')
    image = c.vm.image
    image.verify()
    fid = image.entries.get('public_legal_decision')
    if fid is None or not image.function_ids[fid].startswith(_CANONICAL_ENTRIES):
        gaps.append('canonical frozen public decision entry required')
    if c.closed or p.closed or c.vm.closed or projection.closed or getattr(inference, 'device', None) != rules.device:
        gaps.append('live common device bindings required')
    return tuple(gaps)


def root_value_rows(roots, decision, meta, prediction, viewers, policy_ids, auxiliary_ids,
                    active, *, allow_cpu_reference=False):
    """Device row conditions only; caller must first establish source binding."""
    import torch as t
    d = viewers.device; b = viewers.shape[0]
    if d.type != 'cuda' and allow_cpu_reference is not True:
        raise ValueError('Actual CUDA required; CPU is an explicit mask oracle')
    vectors = [(viewers, t.int64), (policy_ids, t.int64), (auxiliary_ids, t.int64),
               (active, t.bool), (meta.actors, t.int64), (meta.finished, t.bool)]
    for state in (roots, decision.states):
        vectors += [(getattr(state, n), t.int64) for n in ('slots', 'generations', 'owners')]
        vectors += [(state.active, t.bool), (state.errors, t.int32)]
    vectors += [(decision.errors, t.int32), (meta.errors, t.int32),
                (prediction.invalid_rows, t.bool), (decision.counts, t.int64)]
    for value, dtype in vectors:
        if value.device != d or value.dtype != dtype or value.shape != (b,):
            raise ValueError('Exact bound-device root vector required')
    if prediction.wdl.device != d or prediction.wdl.dtype != t.float32 or prediction.wdl.shape != (b, 3):
        raise ValueError('Exact float32 WDL rows required')
    if decision.observations.device != d or decision.observations.dtype != t.float32 or decision.observations.shape[0] != b:
        raise ValueError('Exact root observation rows required')
    eligible = (active & roots.active & decision.states.active & ~meta.finished
                & (meta.actors == viewers) & (viewers >= 0) & (viewers <= 1)
                & (policy_ids == auxiliary_ids) & (roots.slots >= 0) & (roots.owners >= 0)
                & (roots.errors == 0) & (decision.errors == 0) & (decision.states.errors == 0)
                & (meta.errors == 0) & ~prediction.invalid_rows
                & (decision.counts > 0) & t.isfinite(prediction.wdl).all(-1)
                & t.isfinite(decision.observations).all(-1))
    for name in ('slots', 'generations', 'owners'):
        eligible &= getattr(roots, name) == getattr(decision.states, name)
    if roots.pool is not decision.states.pool: eligible.zero_()
    # Preserve the existing float32 subtraction BEFORE float64 conversion.
    value = (prediction.wdl[:, 2] - prediction.wdl[:, 0]).to(t.float64)
    return t.where(eligible, value, 0.), eligible


class DeviceRootValueReuse:
    def __init__(self, rules, inference, auxiliary):
        self.rules = rules
        self.gaps = inspect_root_value_reuse(rules, inference, auxiliary)
        self.enabled = not self.gaps

    def value_device(self, roots, decision, meta, prediction, viewers, policy_ids,
                     auxiliary_ids, active):
        r = self.rules; t = r.t
        if not self.enabled: raise RuntimeError('Root value binding gap: ' + '; '.join(self.gaps))
        if not t.cuda.is_current_stream_capturing() or r.closed:
            raise RuntimeError('Root reuse requires live canonical capture')
        queue = r._queues['decision']
        if queue.index != 1 or not queue.plans or decision.candidates is not queue.plans[0].candidates:
            raise RuntimeError('Fresh dedicated root decision required')
        if decision.observations is not queue.plans[0].observations:
            raise RuntimeError('Fresh root public observation required')
        return root_value_rows(roots, decision, meta, prediction, viewers, policy_ids,
                               auxiliary_ids, active)
