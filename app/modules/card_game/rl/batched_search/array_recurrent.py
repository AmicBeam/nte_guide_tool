"""Batched array transition/evaluation boundary for Mctx-style expansion.

State embeddings are fixed-layout integer arrays, not Python entity graphs or
bytecode-VM heaps. The registry lowers shared rule IR at setup. Coverage is
explicit: registering a combat slice does not implement a seven-team game.
Information-set world sampling remains outside this recurrent boundary.
"""
from dataclasses import dataclass


# Bits 1/2 are reserved by lower_array for actor/overflow errors.
UNSUPPORTED_ACTION, INVALID_ROW, EVALUATION_ERROR = 4, 8, 16


@dataclass(frozen=True)
class ArrayState:
    values: object        # int32[B,F], includes every private/RNG rule field
    actors: object        # int64[B], actual next decision owner
    finished: object      # bool[B]
    winners: object       # int64[B], -2 ongoing, -1 draw, 0/1 winner
    errors: object        # int32[B], sticky


@dataclass(frozen=True)
class ArrayEvaluation:
    logits: object        # float32[B,M], actor's own information only
    valid: object         # bool[B,M], complete legal candidate set
    wdl: object           # float32[B,3], loss/draw/win for fixed viewers
    errors: object        # int32[B]


@dataclass(frozen=True)
class ArrayRecurrentOutput:
    state: ArrayState
    reward: object        # float32[B], actual newly reached terminal only
    discount: object      # float32[B], 1 continuing / 0 absorbing or error
    prior_logits: object
    valid: object
    value: object         # float32[B], P(win)-P(loss), fixed viewer
    errors: object


class ArrayRuleRegistry:
    """Static masked dispatch of shared IR programs over an entire batch.

    An action ID selects a declared rule program, not a truncated legal ordinal.
    Legality/coverage masks must be established by the concrete rule adapter.
    No fallback to a VM, CPU rules, or partially executed unsupported actions.
    """
    supports_full_duel_rules = False

    def __init__(self, programs):
        from ..rule_ir.lower_array import lower_array
        from ..rule_ir.schema import FIELDS
        if not isinstance(programs, dict) or not programs:
            raise ValueError('Nonempty action-ID to shared-IR program registry required')
        if any(type(k) is not int or k < 0 for k in programs):
            raise ValueError('Nonnegative integer action IDs required')
        layouts = [tuple(p.fields or FIELDS) for p in programs.values()]
        if any(layout != layouts[0] for layout in layouts):
            raise ValueError('All programs must use the same complete array layout')
        self.fields = layouts[0]
        self.kernels = tuple((key, lower_array(program)) for key, program in sorted(programs.items()))

    def step(self, values, actors, actions, active):
        import torch
        from ..rule_ir.lower_array import ArrayRuleResult
        if (not isinstance(actions, torch.Tensor) or actions.dtype != torch.int64
                or actions.device != values.device or actions.shape != (values.shape[0],)):
            raise ValueError('Resident int64 action IDs required')
        supported = torch.zeros_like(active)
        errors = torch.zeros_like(actions, dtype=torch.int32)
        output = values.clone()
        for action_id, kernel in self.kernels:
            selected = active & (actions == action_id)
            result = kernel(values, actors, selected)
            errors.bitwise_or_(result.errors)
            good = selected & (result.errors == 0)
            output = torch.where(good[:, None], result.states, output)
            supported = supported | selected
        errors.bitwise_or_((active & ~supported).to(torch.int32) * UNSUPPORTED_ACTION)
        return ArrayRuleResult(output, errors)


class SharedIrTransition:
    """Concrete registry-to-recurrent adapter for declared integer layouts.

    Shared combat IR uses winner=2 for draw; the recurrent ABI uses -1.
    If a program changes decision owner it MUST store ``actor_field``. Without
    that field, same-player continuation is retained, never flipped by depth.
    This adapter is a rule-slice integration, not a complete duel executor.
    """
    supports_full_duel_rules = False

    def __init__(self, registry, *, finished_field='done', winner_field='winner',
                 actor_field=None):
        if not isinstance(registry, ArrayRuleRegistry):
            raise TypeError('Shared array rule registry required')
        self.registry = registry
        self.finished_column = registry.fields.index(finished_field)
        self.winner_column = registry.fields.index(winner_field)
        self.actor_column = None if actor_field is None else registry.fields.index(actor_field)

    def __call__(self, state, actions, active):
        import torch
        old_done = state.values[:, self.finished_column]
        old_winner = state.values[:, self.winner_column].long()
        old_winner = torch.where(old_winner == 2, -1, old_winner)
        inconsistent = ((old_done != 0) != state.finished)
        inconsistent = inconsistent | ((old_done < 0) | (old_done > 1))
        inconsistent = inconsistent | (state.finished & (old_winner != state.winners))
        if self.actor_column is not None:
            inconsistent = inconsistent | (~state.finished &
                (state.values[:, self.actor_column] != state.actors))
        errors = state.errors | ((active & inconsistent).to(torch.int32) * INVALID_ROW)
        result = self.registry.step(state.values, state.actors, actions, active & (errors == 0))
        errors = errors | result.errors
        good = active & (errors == 0)
        values = torch.where(good[:, None], result.states, state.values)
        finished = values[:, self.finished_column] != 0
        winners = values[:, self.winner_column].long()
        winners = torch.where(winners == 2, -1, winners)
        winners = torch.where(finished, winners, -2)
        actors = state.actors if self.actor_column is None else values[:, self.actor_column].long()
        return ArrayState(values, actors, finished, winners, errors)


class ArrayRecurrentFn:
    """One batch of rule transitions followed by one batch of evaluation.

    ``transition(state, action_ids, active)`` returns ArrayState.
    ``evaluate(state, fixed_viewers, active)`` returns ArrayEvaluation: policy
    uses each current actor's view, while value uses fixed_viewers. A concrete
    seven-team implementation must additionally prove those information bounds.
    Callables are prepared before capture; neither is a per-root host callback.
    This Torch boundary borrows Mctx's structure, not its JAX binary interface.
    """
    supports_full_duel_rules = False

    def __init__(self, transition, evaluate, *, max_actions, device='cuda', allow_cpu_reference=False):
        import torch
        if not callable(transition) or not callable(evaluate):
            raise TypeError('Batched array transition and evaluation required')
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError('Positive complete legal action capacity required')
        self.t = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type != 'cuda' and not (self.device.type == 'cpu' and allow_cpu_reference):
            raise ValueError('CUDA or explicit CPU array reference required')
        self.transition, self.evaluate, self.max_actions = transition, evaluate, max_actions

    def _tensor(self, value, dtype, shape):
        if (not isinstance(value, self.t.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != tuple(shape)):
            raise ValueError('Exact resident recurrent tensor shape/dtype/device required')

    def _state(self, state, rows, width=None):
        t = self.t
        if not isinstance(state, ArrayState):
            raise TypeError('ArrayState required')
        if width is None:
            if not isinstance(state.values, t.Tensor) or state.values.ndim != 2:
                raise ValueError('Integer state matrix required')
            width = state.values.shape[1]
        self._tensor(state.values, t.int32, (rows, width))
        for name, dtype in (('actors', t.int64), ('finished', t.bool),
                            ('winners', t.int64), ('errors', t.int32)):
            self._tensor(getattr(state, name), dtype, (rows,))
        return width

    def __call__(self, state, actions, viewers, active):
        t = self.t
        rows = state.values.shape[0]
        width = self._state(state, rows)
        for value, dtype in ((actions, t.int64), (viewers, t.int64), (active, t.bool)):
            self._tensor(value, dtype, (rows,))
        metadata_bad = (((state.actors < 0) | (state.actors > 1)) & ~state.finished
                        | ((state.winners < -1) | (state.winners > 1)) & state.finished
                        | (state.winners != -2) & ~state.finished
                        | (viewers < 0) | (viewers > 1))
        errors = state.errors | ((metadata_bad & active).to(t.int32) * INVALID_ROW)
        continuing = active & (errors == 0) & ~state.finished
        # Protect persistent/tree embeddings even if a backend uses in-place
        # stores within its disposable transition workspace.
        owned = ArrayState(*(getattr(state, name).clone() for name in
                             ('values', 'actors', 'finished', 'winners', 'errors')))
        advanced = self.transition(owned, actions, continuing)
        self._state(advanced, rows, width)
        errors = errors | t.where(continuing, advanced.errors, 0)
        bad_next = (((advanced.actors < 0) | (advanced.actors > 1)) & ~advanced.finished
                    | ((advanced.winners < -1) | (advanced.winners > 1)) & advanced.finished
                    | (advanced.winners != -2) & ~advanced.finished)
        errors = errors | ((continuing & bad_next).to(t.int32) * INVALID_ROW)
        accepted = continuing & (errors == 0)
        values = t.where(accepted[:, None], advanced.values, state.values)
        actors = t.where(accepted, advanced.actors, state.actors)
        finished = t.where(accepted, advanced.finished, state.finished)
        winners = t.where(accepted, advanced.winners, state.winners)
        next_state = ArrayState(values, actors, finished, winners, errors)
        evaluate_rows = active & (errors == 0) & ~finished
        prediction = self.evaluate(next_state, viewers, evaluate_rows)
        if not isinstance(prediction, ArrayEvaluation):
            raise TypeError('ArrayEvaluation required')
        self._tensor(prediction.logits, t.float32, (rows, self.max_actions))
        self._tensor(prediction.valid, t.bool, (rows, self.max_actions))
        self._tensor(prediction.wdl, t.float32, (rows, 3))
        self._tensor(prediction.errors, t.int32, (rows,))
        bad_eval = (~t.isfinite(prediction.wdl).all(-1)
                    | ((prediction.wdl < 0) | (prediction.wdl > 1)).any(-1)
                    | ((prediction.wdl.sum(-1) - 1.).abs() > 1e-5)
                    | ~prediction.valid.any(-1)
                    | (prediction.valid & ~t.isfinite(prediction.logits)).any(-1)
                    | (prediction.errors != 0))
        errors = errors | ((evaluate_rows & bad_eval).to(t.int32) * EVALUATION_ERROR)
        good = active & (errors == 0)
        # Evaluation errors roll back this attempted transition as well.
        next_state = ArrayState(t.where(good[:, None], values, state.values),
            t.where(good, actors, state.actors), t.where(good, finished, state.finished),
            t.where(good, winners, state.winners), errors)
        reached = good & continuing & finished
        reward = t.where(reached, t.where(winners == viewers, 1.,
                         t.where(winners == -1, 0., -1.)), 0.).to(t.float32)
        live = good & ~finished
        value = t.where(live, prediction.wdl[:, 2] - prediction.wdl[:, 0], 0.)
        valid = prediction.valid & live[:, None]
        # A terminal embedding is absorbing. Its dummy zero action is only for
        # tree arithmetic; it must never be submitted as a physical legal move.
        dummy = t.arange(self.max_actions, device=self.device)[None, :] == 0
        valid = valid | (good & finished)[:, None] & dummy
        logits = t.where(live[:, None], prediction.logits, 0.).masked_fill(~valid, -float('inf'))
        return ArrayRecurrentOutput(next_state, reward, live.to(t.float32), logits, valid, value, errors)
