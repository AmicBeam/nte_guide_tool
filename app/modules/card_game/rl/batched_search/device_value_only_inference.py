# Device inference that evaluates only the value tower when action width is 0.
# This skips candidate scores; it does not claim 1600-batch readiness.

from .device_inference import DeviceInferenceResult
from .device_memory_bounded_inference import DeviceMemoryBoundInference


class DeviceValueOnlyInference(DeviceMemoryBoundInference):
    # Override predict_device only for static m=0 with an explicit value mask.
    # All other shapes, including omitted value_only, keep the parent path.

    def predict_device(self, x, candidates, valid, model_index, value_actor, *, value_only=None):
        t = self.torch
        static_empty = (
            value_only is not None
            and isinstance(candidates, t.Tensor)
            and candidates.ndim >= 2
            and candidates.shape[1] == 0
        )
        if not static_empty:
            return super().predict_device(
                x, candidates, valid, model_index, value_actor, value_only=value_only
            )
        for tensor in (x, candidates, valid, model_index, value_actor):
            if not isinstance(tensor, t.Tensor) or tensor.device != self.device:
                raise ValueError('Inputs must already be on the bound device')
        b, m = candidates.shape[:2]
        if (x.dtype != t.float32 or candidates.dtype != t.float32 or valid.dtype != t.bool
                or model_index.dtype != t.int64 or value_actor.dtype != t.float32
                or x.shape != (b, self.feature_dim) or candidates.shape != (b, m, self.candidate_dim)
                or valid.shape != (b, m) or model_index.shape != (b,) or value_actor.shape != (b, 1)
                or not 1 <= b <= self.max_rows or candidates.numel() > self.max_candidate_elements):
            raise ValueError('Exact bounded device input shapes and dtypes required')
        if self.device.type == 'cuda' and t.backends.cuda.matmul.allow_tf32:
            raise ValueError('Explicit FP32 inference requires TF32 disabled')
        if value_only.device != self.device or value_only.dtype != t.bool or value_only.shape != (b,):
            raise ValueError('Device value-only mask required')
        if x.requires_grad or candidates.requires_grad or value_actor.requires_grad:
            raise RuntimeError(
                "DeviceValueOnlyInference.predict_device rejects gradient-bearing inputs"
            )
        invalid = ((model_index < 0) | (model_index >= len(self.keys))
                   | ~t.isfinite(x).all(-1) | ~t.isfinite(candidates).all(-1).all(-1)
                   | ~((value_actor == 0) | (value_actor == 1)).all(-1))
        invalid = invalid | (~value_only & ~valid.any(-1)) | (value_only & valid.any(-1))
        ids = model_index.clamp(0, len(self.keys) - 1)
        with t.inference_mode():
            representation = self._tower(x, ids, 'value_net')
            wdl = self._linear(t.cat((representation, value_actor), -1), ids, 'wdl').softmax(-1)
            invalid = invalid | ~t.isfinite(wdl).all(-1)
            logits = t.empty((b, 0), dtype=t.float32, device=self.device)
        return DeviceInferenceResult(logits, wdl, invalid)

