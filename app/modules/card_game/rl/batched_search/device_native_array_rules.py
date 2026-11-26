"""Closed-search composition over native compiled CUDA state-array kernels.

Storage/leases and typed numerical encoders are shared with the checked device
path. Game-rule control flow is fully compiled, without an uploaded instruction
stream. This wiring does not substitute for actual CUDA/full-game preflight.
"""
from ..batched_duel.native_array_cuda import CudaNativeArray
from .device_frozen_rules import DeviceFrozenRules
from .device_view_sampler import DeviceViewSampleCalls, DeviceViewValidatedSampler
from .device_physical_action_capture import PhysicalActionRecordingCalls
from .array_zhenhong_prior import ArrayZhenhongPrior
from .device_search import DeviceGumbelSearch


class NativeArrayRuleCalls(DeviceViewSampleCalls, PhysicalActionRecordingCalls):
    """Native physical-action recording plus mandatory two-view world guards."""
    def __init__(self, executor, *args, **kwargs):
        if not isinstance(executor, CudaNativeArray):
            raise TypeError('Actual native compiled CUDA array executor required')
        super().__init__(executor, *args, **kwargs)


class DeviceNativeArrayRules(DeviceFrozenRules):
    execution_kind = 'device_native_array_rules_composition'
    supports_full_duel_rules = False
    teacher_horizon = 0

    def __init__(self, calls, presentation, *, action_store, sites=None, guard_tile=8):
        if not isinstance(calls, NativeArrayRuleCalls) or not isinstance(calls.vm, CudaNativeArray):
            raise TypeError('Native array calls required; a labeled VM is not a native executor')
        if tuple(calls.vm._buffers['instructions'].shape) != (1, 4):
            raise ValueError('Native executor must not upload a bytecode stream')
        super().__init__(calls, presentation, action_store=action_store,
                         sites=sites, require_teacher=False)
        try:
            self._native_sampler = DeviceViewValidatedSampler(self, guard_tile=guard_tile)
            self._owned.append(self._native_sampler)
        except BaseException:
            self.close()
            raise

    def sample_world_device(self, states, viewers, seeds):
        return self._native_sampler.capture_device(states, viewers, seeds)


def create_native_array_search(rules, inference, auxiliary, statistics, keys, *,
                               rows, random, max_actions=800, zhenhong_prior=True,
                               prior_weights=None, **options):
    """Bind complete Gumbel32, pure WDL and the simulation-free root prior.

    A disabled prior is an explicit ablation. There is no optional root planner
    or leaf/opponent rule teacher in this branch. The existing game loop keeps
    raw action tapes, full policy labels and fixed-viewer terminal accounting.
    """
    if not isinstance(rules, DeviceNativeArrayRules):
        raise TypeError('Native array rule composition required')
    if type(zhenhong_prior) is not bool:
        raise TypeError('Explicit prior ablation required')
    reserved = {'teacher_mode', 'terminal_horizon', 'opponent_mode', 'prior_guide',
                'simulations', 'gumbel_candidates', 'device'}
    if reserved & set(options):
        raise ValueError('Native array search fixes full32/cap16/network-WDL/policy-opponent semantics')
    prior = ArrayZhenhongPrior(device=rules.device, weights=prior_weights) if zhenhong_prior else None
    return DeviceGumbelSearch(rules, inference, auxiliary, statistics, keys,
        rows=rows, max_actions=max_actions, simulations=32, gumbel_candidates=16,
        teacher_mode='network_value', terminal_horizon=0, opponent_mode='policy',
        prior_guide=prior, random=random, device=str(rules.device), **options)
