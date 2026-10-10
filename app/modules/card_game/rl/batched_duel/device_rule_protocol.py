"""Tensor ownership contract for a future closed device search driver.

This file defines data/signatures, not an executor or readiness evidence.
Implementations must retain formal rules, fail closed, and carry per-row errors
on device until a whole bounded batch ends. CPU parity stays outside the loop.
"""
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class DeviceStateBatch:
    pool: Any
    slots: Any          # int64[B], device state slots
    generations: Any   # int64[B], stale output protection
    owners: Any        # int64[B], immutable root/game ownership
    active: Any        # bool[B], preserve inactive rows without allocating
    errors: Any        # int32[B], sticky fail-closed execution errors


@dataclass(frozen=True)
class DeviceMeta:
    actors: Any        # int64[B]: 0=a, 1=b, -1=no decision
    phases: Any        # int32[B]: a versioned code table, not host strings
    winners: Any       # int64[B]: 0=a, 1=b, -1=draw, -2=not finished
    finished: Any      # bool[B]
    versions: Any      # int64[B], formal physical state revision
    errors: Any        # int32[B]


@dataclass(frozen=True)
class DeviceDecision:
    states: DeviceStateBatch
    action_pool: Any   # retains the filtered legal action graphs on device
    action_slots: Any  # int64[B], action graph slots in action_pool
    action_generations: Any  # int64[B], action graph generations
    action_refs: Any   # int32[B,M], filtered physical action refs; no concede
    action_types: Any  # int64[B,M], frozen formal type codes, never host strings
    counts: Any        # int64[B], full legal lengths, no truncation
    valid: Any         # bool[B,M]
    observations: Any  # float32[B,D], acting observer public information
    candidates: Any    # float32[B,M,C], original named order/identity
    errors: Any        # int32[B], includes overflow/unsupported errors


@dataclass(frozen=True)
class DeviceValueObservation:
    observations: Any  # float32[B,D], fixed viewer's public information
    viewer_is_actor: Any  # float32[B,1], value context does not switch owner
    terminal: Any      # bool[B]
    terminal_value: Any  # float64[B], actual viewer terminal WDL only
    errors: Any        # int32[B]


class DeviceRuleExecutor(Protocol):
    """All row selectors, masks, arguments and results remain on CUDA.

    Static host shape checks/setup are permitted. These calls must be capturable
    or invoked by a compiled device driver: no scalar extraction, host action
    dictionaries, callbacks into Python rules or per-root CPU queue dispatch.
    Immutable inputs may share an arena only through checked generation leases.
    """
    def metadata_device(self, states: DeviceStateBatch) -> DeviceMeta: ...

    def decision_device(self, states: DeviceStateBatch, observers: Any) -> DeviceDecision: ...

    def rule_decision_device(self, states: DeviceStateBatch, observers: Any) -> tuple[DeviceDecision, Any, Any]:
        """Original teacher legal ordering, int64[B] choices, int32[B] errors.

        Both root and current decision action leases survive until consumption;
        action graphs cannot alias scratch overwritten by a later decision.
        """
        ...

    def sample_world_device(self, states: DeviceStateBatch, viewers: Any,
                            seeds: Any) -> DeviceStateBatch:
        """seeds=int64[B,5] unsigned little-endian limbs, including bit 256.

        The current root hash seed plus simulation stride is wider than uint64;
        do not truncate it before the formal CPython-compatible MT sampler.
        """
        ...

    def step_device(self, states: DeviceStateBatch, actors: Any,
                    decision: DeviceDecision, action_indices: Any,
                    *, physical: bool = False) -> DeviceStateBatch:
        """Indices=int64[B]; same live owner/generation, preserve inputs.

        Static physical chooses formal apply_action; hypothetical chooses formal
        simulate_action. The selected device action graph is never exported.
        """
        ...

    def clean_device(self, states: DeviceStateBatch) -> DeviceStateBatch: ...

    def value_observation_device(self, states: DeviceStateBatch,
                                  viewers: Any) -> DeviceValueObservation: ...

    def teacher_continue_device(self, states: DeviceStateBatch, viewers: Any,
                                  *, horizon: int = 3) -> DeviceStateBatch:
        """Same original public rule teacher, original step/clean and horizon."""
        ...

    def release_device(self, states: DeviceStateBatch) -> None:
        """Checked slot/generation/owner release, including failed search rows.

        Search errors stop rule execution but must not leak previously granted
        valid leases. Inactive unallocated rows remain no-ops.
        """
        ...
