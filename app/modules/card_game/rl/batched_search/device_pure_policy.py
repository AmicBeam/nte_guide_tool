"""Device-only legal argmax for frozen evaluation; never Gumbel labels.

This numerical driver requires the same capturable rule adapter as search.
Its zero simulation counts cannot pass DeviceTrainingTape's search-label gate.
"""
from .device_search import DeviceGumbelResult, RULE_ERROR, INFERENCE_ERROR, METADATA_ERROR
from ..batched_duel.device_rule_protocol import DeviceStateBatch


class DevicePurePolicy:
    supports_full_duel_rules = False
    algorithm = 'argmax'
    counts_as_search = False
    noise = False

    def __init__(self, rules, inference, *, rows, max_actions, device='cuda',
                 allow_cpu_reference=False, allow_synthetic=False):
        import torch
        for value in (rows, max_actions):
            if type(value) is not int or value < 1:
                raise ValueError('Positive bounded pure-policy dimensions required')
        if type(allow_cpu_reference) is not bool or type(allow_synthetic) is not bool:
            raise TypeError('Explicit reference and synthetic controls required')
        self.t = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type != 'cuda' and not (self.device.type == 'cpu' and
                allow_cpu_reference and allow_synthetic):
            raise ValueError('CUDA required; CPU is an explicit synthetic oracle only')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no host argmax fallback')
        for name in ('metadata_device', 'decision_device'):
            if not callable(getattr(rules, name, None)):
                raise ValueError('Missing device rule interface: ' + name)
        if inference.device != self.device or not callable(getattr(inference, 'predict_device', None)):
            raise ValueError('Bound-device inference required')
        self.rules, self.inference = rules, inference
        self.allow_synthetic = allow_synthetic
        self.rows, self.max_actions = rows, max_actions
        self._unfinished = torch.zeros(rows, dtype=torch.uint8, device=self.device)
        self._result = None

    def _exact(self, value, dtype, shape):
        if (not isinstance(value, self.t.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != shape or not value.is_contiguous()):
            raise ValueError('Exact bound-device pure-policy tensor required')

    def initialize(self, roots, actors, world_seeds, policy_ids, auxiliary_ids):
        t = self.t
        if not isinstance(roots, DeviceStateBatch):
            raise TypeError('Checked device root states required')
        for value in (roots.slots, roots.generations, roots.owners, actors):
            self._exact(value, t.int64, (self.rows,))
        self._exact(roots.active, t.bool, (self.rows,))
        self._exact(roots.errors, t.int32, (self.rows,))
        self._exact(policy_ids, t.int64, (self.rows, 2))
        meta = self.rules.metadata_device(roots)
        self._exact(meta.actors, t.int64, (self.rows,))
        self._exact(meta.finished, t.bool, (self.rows,))
        self._exact(meta.errors, t.int32, (self.rows,))
        admitted = (roots.active & (roots.errors == 0) & (meta.errors == 0)
                    & ~meta.finished & (meta.actors == actors) & (actors >= 0) & (actors <= 1))
        selected = DeviceStateBatch(roots.pool, roots.slots, roots.generations, roots.owners,
                                   admitted, roots.errors)
        decision = self.rules.decision_device(selected, actors)
        self._exact(decision.counts, t.int64, (self.rows,))
        self._exact(decision.valid, t.bool, (self.rows, self.max_actions))
        self._exact(decision.errors, t.int32, (self.rows,))
        if (not isinstance(decision.observations, t.Tensor)
                or decision.observations.device != self.device
                or decision.observations.dtype != t.float32
                or decision.observations.ndim != 2 or decision.observations.shape[0] != self.rows
                or not decision.observations.is_contiguous()):
            raise ValueError('Bound-device public observation matrix required')
        prefix = t.arange(self.max_actions, device=self.device)[None, :] < decision.counts[:, None]
        bad = ((actors < 0) | (actors > 1) | (actors != meta.actors) | meta.finished
               | (roots.slots < 0) | (roots.generations < 1) | (roots.owners < 0)
               | (decision.counts < 1) | (decision.counts > self.max_actions)
               | (decision.valid != prefix).any(-1)
               | ~t.isfinite(decision.observations).all(-1))
        states = getattr(decision, 'states', None)
        if isinstance(states, DeviceStateBatch):
            if states.pool is not roots.pool:
                raise ValueError('Decision and physical state arenas must match')
            bad = (bad | ~states.active | (states.errors != 0)
                   | (states.slots != roots.slots) | (states.generations != roots.generations)
                   | (states.owners != roots.owners))
        elif not self.allow_synthetic:
            raise ValueError('Formal pure policy requires the decision state lease')
        errors = roots.errors.clone()
        errors.bitwise_or_(roots.active.to(t.int32) *
            (((meta.errors != 0) | (decision.errors != 0)).to(t.int32) * RULE_ERROR
             | bad.to(t.int32) * METADATA_ERROR))
        ids = policy_ids.gather(1, actors.clamp(0, 1)[:, None])[:, 0].contiguous()
        context = t.ones((self.rows, 1), dtype=t.float32, device=self.device)
        live = roots.active & (errors == 0)
        if isinstance(decision.candidates, t.Tensor):
            prediction = self.inference.predict_device(decision.observations,
                decision.candidates, decision.valid, ids, context)
        else:
            prediction = decision.candidates.predict_device(self.inference,
                decision.observations, decision.valid, ids, context, live)
        self._exact(prediction.logits, t.float32, (self.rows, self.max_actions))
        self._exact(prediction.invalid_rows, t.bool, (self.rows,))
        finite = (t.isfinite(prediction.logits) | ~decision.valid).all(-1)
        errors.bitwise_or_((live & (prediction.invalid_rows | ~finite)).to(t.int32) * INFERENCE_ERROR)
        live = roots.active & (errors == 0)
        logits = prediction.logits.masked_fill(~decision.valid, -float('inf'))
        logits = t.where(live[:, None], logits, 0.)
        actions = t.where(live, logits.argmax(-1), -1)
        policy = t.where(live[:, None] & decision.valid, logits.softmax(-1), 0.)
        zero = t.zeros(self.rows, dtype=t.int64, device=self.device)
        self._unfinished.zero_()
        self._result = DeviceGumbelResult(decision, actions, policy, t.zeros_like(policy),
            t.zeros_like(policy, dtype=t.int32), zero, zero.clone(), live, errors,
            zero.clone(), t.zeros((self.rows, 10), dtype=t.int64, device=self.device),
            t.zeros((), dtype=t.int64, device=self.device))

    @property
    def unfinished_device(self):
        return self._unfinished

    def tick(self):
        # Gives an otherwise empty captured inner body a static device operation.
        self._unfinished.zero_()

    def result_device(self):
        if self._result is None:
            raise RuntimeError('Pure-policy cohort not initialized')
        return self._result
