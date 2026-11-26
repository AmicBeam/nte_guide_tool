"""Collector-side raw physical action capture through the frozen GPU call path."""
from ..batched_duel.device_rule_calls import DeviceRuleGraphCalls
from .device_raw_action_tape import DeviceRawActionTape


class PhysicalActionRecordingCalls(DeviceRuleGraphCalls):
    """Record only apply_action, after exact raw selection and before GPU execution."""

    def __init__(self, vm, state_store, *, raw_tape, **options):
        if not isinstance(raw_tape, DeviceRawActionTape):
            raise TypeError('Bounded raw action tape required')
        if raw_tape.rows != options.get('rows') or raw_tape.device != getattr(vm, 'device', None):
            raise ValueError('Raw tape must match the exact GPU call cohort')
        super().__init__(vm, state_store, **options)
        self.raw_tape = raw_tape
        self.physical_loop = None

    def bind_game_loop(self, loop):
        if self.physical_loop is not None:
            raise ValueError('Bind the physical game loop once')
        if loop.rows != self.rows or loop.device != self.device:
            raise ValueError('Physical loop must match the GPU call cohort')
        self.physical_loop = loop

    def _bind_step(self, plan, local):
        refs = super()._bind_step(plan, local)
        if plan.entry == 'apply_action':
            if self.physical_loop is None:
                raise RuntimeError('Bind physical counters before capture')
            errors = self.raw_tape.append_device(
                self.selected_action, self._lanes.ids, self.lane_owners,
                self._pick(self.physical_loop.steps_device),
                self._pick(self.physical_loop.source_versions_device),
                self.lane_actors, self.lane_active.bool())
            self._or_errors(errors)
        return refs


class RawActionPolicySink:
    """Require successful raw archiving of every physical step before terminal labels."""

    supports_full_duel_rules = False

    def __init__(self, policy_sink, raw_tape):
        for name in ('begin_device', 'append_device', 'finish_device'):
            if not callable(getattr(policy_sink, name, None)):
                raise TypeError('Bound physical policy sink required')
        self.policy_sink, self.raw_tape = policy_sink, raw_tape

    def begin_device(self, roots, policy_ids, auxiliary_ids, seeds, train_sides, active):
        raw_errors = self.raw_tape.begin_device(roots.owners, seeds)
        policy_errors = self.policy_sink.begin_device(
            roots, policy_ids, auxiliary_ids, seeds, train_sides, active)
        return self.raw_tape.t.where(active, raw_errors | policy_errors, 0)

    def append_device(self, roots, actors, steps, result, active):
        errors = self.policy_sink.append_device(roots, actors, steps, result, active)
        return errors | self.raw_tape.t.where(active, self.raw_tape.errors_device, 0)

    def finish_device(self, roots, winners, steps, complete, errors):
        archive_errors = self.raw_tape.errors_device
        mismatch = roots.active & (steps != self.raw_tape.steps_device)
        failed = errors | archive_errors | mismatch.to(errors.dtype)
        policy_errors = self.policy_sink.finish_device(roots, winners, steps, complete, failed)
        return self.raw_tape.t.where(roots.active, policy_errors | failed, 0)
