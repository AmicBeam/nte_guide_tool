"""Root-only captured Zhenhong guidance; no CPU rule callback or leaf rollout."""
from types import SimpleNamespace
from ..batched_duel.device_rule_protocol import DeviceStateBatch
from .device_frozen_rules import _ActiveRuleLanes


class DeviceZhenhongGuide:
    supports_full_duel_rules = False
    host_rule_callbacks = False
    mode = 'zhenhong_setup_v3_fixed24'

    def __init__(self, rules, *, policy_id, calls=None):
        from ..batched_duel.device_rule_calls import DeviceRuleGraphCalls
        c = rules.calls if calls is None else calls
        if (not isinstance(c, DeviceRuleGraphCalls) or c.closed
                or c.vm is not rules.calls.vm or c.state_store is not rules.calls.state_store
                or c.rows != rules.calls.rows):
            raise ValueError('Guide workspace must share the exact VM, roots and rows')
        if 'zhenhong_guide' not in c.vm.image.entries:
            raise ValueError('Frozen root guide entry required')
        if type(policy_id) is not int or policy_id < 0:
            raise ValueError('Explicit Zhenhong policy identity required')
        self.policy_id = policy_id
        self.rules, self.calls, self.t = rules, c, c.t
        self.rows, self.device = c.rows, c.device
        self.lanes = _ActiveRuleLanes(c.rows, c.lanes, device=c.device, seconds=c.seconds)
        self.choices = c.t.full((c.rows,), -1, dtype=c.t.int64, device=c.device)
        self.errors = c.t.zeros(c.rows, dtype=c.t.int32, device=c.device)
        self.local_choices = c.t.empty(c.lanes, dtype=c.t.int64, device=c.device)
        self.local_errors = c.t.empty(c.lanes, dtype=c.t.int32, device=c.device)
        self.captured = self.closed = False

    def close(self):
        if not self.closed:
            self.lanes.close()
            self.closed = True

    def capture_result(self, states, actors, result, policy_ids, active):
        from dataclasses import replace
        t = self.t
        ids = policy_ids.gather(1, actors.clamp(0, 1)[:, None])[:, 0]
        eligible = (ids == self.policy_id) & active & result.complete
        self.baseline = result.actions.clone()
        self.original_policy = result.policy.clone()
        chosen, errors = self.capture_device(states, actors, result.decision,
                                              self.baseline, eligible.contiguous())
        return replace(result, actions=chosen, errors=result.errors | errors)

    def capture_device(self, states, actors, decision, baseline, eligible):
        c, t = self.calls, self.t
        if self.closed or self.captured or not t.cuda.is_current_stream_capturing():
            raise RuntimeError('One live guide site inside CUDA capture required')
        c._validate_inputs('simulate_action', states, actors=actors, seeds=None,
                           decision=decision, action_indices=baseline)
        c._exact(eligible, t.bool, (c.rows,), 'guide eligible')
        self.choices.copy_(baseline)
        self.errors.copy_(states.errors | decision.errors)
        active = states.active & eligible & (self.errors == 0)
        request = DeviceStateBatch(states.pool, states.slots, states.generations,
                                  states.owners, active.contiguous(), self.errors)
        plan = SimpleNamespace(entry='zhenhong_guide', lanes=self.lanes, actors=actors, decision=decision,
                               action_indices=baseline)

        def operation(local, ids, valid):
            # Reuse the checked physical-action argument binding: immutable root
            # state, exact selected raw action, owner/generation and index guard.
            c._lanes = self.lanes
            c.live_active.copy_(local.active)
            c.lane_owners.copy_(local.owners)
            c.lane_errors.copy_(local.errors)
            c._refresh_active()
            loaded = c.state_store.gather_into_device(local, c.input_graph)
            c._or_errors(loaded.errors)
            c.vm.load_graph_into_device(c.workspace, c.input_graph,
                                       active=c.lane_active, errors=c.load_errors)
            c._or_errors(c.load_errors)
            refs = c._bind_step(plan, local)
            call = c.vm.launch_into_device(c.workspace, 'zhenhong_guide',
                argument_refs=refs, active=c.lane_active, gpu_deadlines=c.deadlines)
            c._or_errors(call.statuses[:, 0])
            self.rules._decode_teacher_index(call.arena, call.outputs,
                c.lane_errors.contiguous(), c.lane_active,
                self.local_choices, self.local_errors)
            c._or_errors(self.local_errors)
            self.lanes.scatter_tensor_device(self.local_choices, self.choices)
            self.lanes.scatter_tensor_device(c.lane_errors, self.errors)

        # A separate guide workspace shares the original absolute deadline.
        if c is not self.rules.calls:
            c.deadlines.copy_(self.rules.calls.deadlines.amin().expand_as(c.deadlines))
        self.lanes.capture_dispatch(request, operation)
        # Lane scatter includes inactive real rows; preserve their search action.
        self.choices.copy_(t.where(active, self.choices, baseline))
        self.errors.bitwise_or_(active.to(t.int32) * (self.lanes.loop.errors != 0).to(t.int32)[0] * 152)
        safe = self.choices.clamp(0, decision.valid.shape[1] - 1)
        bad = active & ((self.choices < 0) | (self.choices >= decision.counts)
                        | ~decision.valid.gather(1, safe[:, None])[:, 0])
        self.errors.bitwise_or_(bad.to(t.int32) * 154)
        self.captured = True
        return self.choices, self.errors


def mix_guided_policy(policy, baseline, chosen, *, coefficient=.5):
    """Keep search target separate; only changed rows receive guide targets."""
    import torch
    if not 0 < coefficient <= 1:
        raise ValueError('Guide coefficient must be in (0,1]')
    guide = torch.nn.functional.one_hot(chosen, policy.shape[1]).to(policy.dtype)
    mixed = (1-coefficient)*policy + coefficient*guide
    return torch.where((chosen != baseline)[:, None], mixed, policy)
