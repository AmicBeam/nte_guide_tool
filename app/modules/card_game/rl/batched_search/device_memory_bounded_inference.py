# Device inference that reuses one full-size work tensor in layer-norm.
# This reduces full-size temporaries in _norm. It does not claim 1600-batch
# readiness or higher throughput.

from .device_inference import DeviceSevenInference


class DeviceMemoryBoundInference(DeviceSevenInference):
    # Override only _norm; inherited predict and model math stay put.

    def _norm(self, x, ids, name, active=False):
        torch = self.torch
        if x.requires_grad:
            raise RuntimeError(
                "DeviceMemoryBoundInference._norm rejects gradient-bearing inputs"
            )
        with torch.inference_mode():
            weight = self.weights[name + ".weight"].index_select(0, ids)
            bias = self.weights[name + ".bias"].index_select(0, ids)
            if x.ndim == 3:
                weight = weight[:, None, :]
                bias = bias[:, None, :]
            work = torch.empty_like(x)
            if active:
                enabled = self.weights[name + ".active"].index_select(0, ids)
                if x.ndim == 3:
                    enabled = enabled[:, None, :]
                count = enabled.sum(-1, keepdim=True)
                torch.mul(x, enabled, out=work)
                mean = work.sum(-1, keepdim=True) / count
                torch.sub(x, mean, out=work)
                work.square_()
                work.mul_(enabled)
                variance = work.sum(-1, keepdim=True) / count
            else:
                mean = x.mean(-1, keepdim=True)
                torch.sub(x, mean, out=work)
                work.square_()
                variance = work.mean(-1, keepdim=True)
            torch.sub(x, mean, out=work)
            work.mul_(torch.rsqrt(variance + 1e-5))
            work.mul_(weight)
            work.add_(bias)
            return work
