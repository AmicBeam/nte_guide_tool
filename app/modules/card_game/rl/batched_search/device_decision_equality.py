"""Exact device observation/candidate equality. Not a rule executor.

Compares float32 observations and full valid ordered encodings. This is not a
hash, epsilon check, readiness claim, or duel-rule step.
"""
from ..batched_duel import gpu_seven_candidates as encoder
from ..batched_duel.device_rule_protocol import DeviceDecision
from .compact_device_candidates import CompactDeviceCandidates


def compare_decisions_device(before, after, active, *, allow_cpu_reference=False):
    t = encoder._require_torch()
    if type(before) is not DeviceDecision or type(after) is not DeviceDecision:
        raise TypeError('DeviceDecision required')
    left, right = before.candidates, after.candidates
    if type(left) is not CompactDeviceCandidates or type(right) is not CompactDeviceCandidates:
        raise TypeError('CompactDeviceCandidates required')
    if (left._rows != right._rows or left._max_candidates != right._max_candidates
            or left.cand_dim != right.cand_dim or left.device != right.device
            or left.template_table is not right.template_table):
        raise ValueError('Stable compact row/M/C/device/template required')
    dev = left.device
    if dev.type != 'cuda' and not (dev.type == 'cpu' and allow_cpu_reference is True):
        raise ValueError('CUDA required; CPU is an explicit encoding oracle only')
    b, m, c = left._rows, left._max_candidates, left.cand_dim
    _require(t, active, t.bool, (b,), dev, 'active')
    obs_b, obs_a = before.observations, after.observations
    if (not isinstance(obs_b, t.Tensor) or not isinstance(obs_a, t.Tensor)
            or obs_b.device != dev or obs_a.device != dev
            or obs_b.dtype != t.float32 or obs_a.dtype != t.float32
            or obs_b.ndim != 2 or obs_a.ndim != 2 or obs_b.shape[0] != b
            or obs_a.shape != obs_b.shape or obs_b.shape[1] < 1):
        raise ValueError('Exact bound-device observations required')
    _require(t, before.counts, t.int64, (b,), dev, 'counts')
    _require(t, after.counts, t.int64, (b,), dev, 'counts')
    _require(t, before.valid, t.bool, (b, m), dev, 'valid', 2)
    _require(t, after.valid, t.bool, (b, m), dev, 'valid', 2)
    _require(t, before.errors, t.int32, (b,), dev, 'errors')
    _require(t, after.errors, t.int32, (b,), dev, 'errors')
    scratch = left.scratch_rows if left.scratch_rows < right.scratch_rows else right.scratch_rows
    errors = t.zeros(b, dtype=t.int32, device=dev)
    slots = t.arange(m, device=dev)
    for start in range(0, b, scratch):
        stop = start + scratch if start + scratch < b else b
        on = active[start:stop]
        packed_b, enc_b = left.encode_slice_device(start, stop, on)
        packed_a, enc_a = right.encode_slice_device(start, stop, on)
        if packed_b.candidates.shape != (stop - start, m, c) or packed_a.candidates.shape != packed_b.candidates.shape:
            raise ValueError('encode_slice_device must keep full candidate width')
        local = t.where(on, before.errors[start:stop], 0)
        local = t.where(on & (after.errors[start:stop] != 0) & (local == 0), after.errors[start:stop], local)
        local = t.where(on & (enc_b != 0) & (local == 0), enc_b, local)
        local = t.where(on & (enc_a != 0) & (local == 0), enc_a, local)
        count_b = before.counts[start:stop]
        count_a = after.counts[start:stop]
        token_b = left.tokens.counts[start:stop]
        token_a = right.tokens.counts[start:stop]
        count_bad = ((count_b != count_a) | (count_b < 1) | (count_b > m)
                     | (count_b != token_b) | (count_a != token_a)
                     | (count_b != packed_b.counts) | (count_a != packed_a.counts))
        prefix_b = slots[None, :] < count_b[:, None]
        prefix_a = slots[None, :] < count_a[:, None]
        valid_b = before.valid[start:stop]
        valid_a = after.valid[start:stop]
        mask_bad = ((valid_b != prefix_b) | (valid_a != prefix_a) | (valid_b != valid_a)
                    | (left.tokens.mask[start:stop] != prefix_b)
                    | (right.tokens.mask[start:stop] != prefix_a)
                    | (packed_b.mask != prefix_b) | (packed_a.mask != prefix_a)
                    | (packed_b.mask != packed_a.mask)).any(-1)
        obs_l = obs_b[start:stop]
        obs_r = obs_a[start:stop]
        obs_bad = ((~t.isfinite(obs_l)).any(-1) | (~t.isfinite(obs_r)).any(-1)
                   | (obs_l != obs_r).any(-1))
        need = (prefix_b | prefix_a)[:, :, None]
        finite = t.isfinite(packed_b.candidates) & t.isfinite(packed_a.candidates)
        cand_bad = (need & ((~finite) | (packed_b.candidates != packed_a.candidates))).any(-1).any(-1)
        mismatch = count_bad | mask_bad | obs_bad | cand_bad
        errors[start:stop] = t.where(on & mismatch & (local == 0), 1, local)
    return errors


def _require(t, value, dtype, shape, device, name, ndim=1):
    if (not isinstance(value, t.Tensor) or value.device != device
            or value.dtype != dtype or value.ndim != ndim or value.shape != shape):
        raise ValueError('Exact bound-device ' + name + ' required')
