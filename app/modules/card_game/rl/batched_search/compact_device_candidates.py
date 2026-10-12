"""Row-sliced dense seven-candidate encoding over compact resident tokens.

Holds full-B, full-M gpu_seven_candidates.RawCandidateTokens and materializes
at most scratch_rows dense [local, M, 1576] encodings at a time. Logical
.shape is (B, M, CAND_DIM) without allocating that matrix. This wrapper is
not a duel-rule executor.

allocated_bytes counts compact raw token tensors only. The shared card
template is reported separately. Peak encode/predict/hash scratch is not
reserved here; measure it on the actual device allocator.
"""
from dataclasses import fields

from ..batched_duel import gpu_seven_candidates as encoder
from .device_inference import DeviceInferenceResult
from .device_info_key import DeviceKeyResult


_TOKEN_TENSORS = tuple(field.name for field in fields(encoder.RawCandidateTokens)
                       if field.name != 'request_ids')


class CompactDeviceCandidates:
    supports_full_duel_rules = False

    def __init__(self, tokens, *, scratch_rows=32, device='cuda',
                 allow_cpu_reference=False, template_table=None):
        t = encoder._require_torch()
        if type(scratch_rows) is not int or scratch_rows < 1:
            raise ValueError('Positive static scratch row bound required')
        if not isinstance(tokens, encoder.RawCandidateTokens):
            raise TypeError('Device RawCandidateTokens required')
        dev = encoder._resolve_device(device)
        if dev.type != 'cuda' and not (dev.type == 'cpu' and allow_cpu_reference is True):
            raise ValueError('CUDA required; CPU is an explicit encoding oracle only')
        integer_names = ('int_tokens', 'mulligan_cards', 'counts', 'order_cards', 'select_cards')
        float_names = ('float_tokens', 'item_flags', 'turn_counts', 'extra_ap')
        for name in _TOKEN_TENSORS:
            value = getattr(tokens, name)
            allowed = ((t.int32, t.int64) if name in integer_names else
                       (t.float32, t.float64) if name in float_names else (t.bool,))
            if not isinstance(value, t.Tensor) or value.device != dev or value.dtype not in allowed:
                raise ValueError('Exact pre-resident candidate token dtype/device required: ' + name)
        int_tok = tokens.int_tokens
        if int_tok.ndim != 3 or int_tok.shape[2] != 7:
            raise ValueError('Candidate token shape must be N,M,7')
        n, m = int_tok.shape[:2]
        if n < 1 or m < 1:
            raise ValueError('Nonempty static candidate capacity required')
        expected = dict(float_tokens=(n, m, 8), item_flags=(n, m, 2), turn_counts=(n,),
                        counts=(n,), mask=(n, m), order_cards=(n, m, 2, encoder.ORDER_SLOTS),
                        extra_ap=(n, m))
        for name, shape in expected.items():
            if getattr(tokens, name).shape != shape:
                raise ValueError('Candidate token buffers are misaligned: ' + name)
        for name in ('mulligan_cards', 'select_cards'):
            value = getattr(tokens, name)
            if value.ndim != 3 or value.shape[:2] != (n, m) or value.shape[2] > encoder.MAX_MULLIGAN_SELECTIONS:
                raise ValueError('Bounded card-selection token shape required')
        if template_table is None:
            if (dev.type == 'cuda' and t.cuda.is_current_stream_capturing()
                    and str(dev) not in encoder._TEMPLATE_CACHE):
                raise RuntimeError('Prepare card template table before CUDA capture')
            template_table = encoder.get_card_template_table(dev)
        if (not isinstance(template_table, t.Tensor) or template_table.device != dev
                or template_table.dtype != t.float32
                or template_table.shape != (encoder.NUM_CARDS + 1, 6)):
            raise ValueError('Exact resident static card template table required')
        self.t = t
        self.device = dev
        self.allow_cpu_reference = allow_cpu_reference is True
        self.scratch_rows = scratch_rows
        self.tokens = tokens
        self.request_ids = ()
        self._template_table = template_table
        self._rows = n
        self._max_candidates = m
        self._row_copy_module=None
        if dev.type=='cuda':
            from .device_masked_row_copy import prepare_masked_row_copy
            self._row_copy_module=prepare_masked_row_copy(dev)

    @property
    def ndim(self):
        return 3

    @property
    def shape(self):
        return (self._rows, self._max_candidates, encoder.CAND_DIM)

    @property
    def batch_size(self):
        return self._rows

    @property
    def max_candidates(self):
        return self._max_candidates

    @property
    def cand_dim(self):
        return encoder.CAND_DIM

    @property
    def template_table(self):
        return self._template_table

    @property
    def allocated_bytes(self):
        return sum(getattr(self.tokens, name).numel() *
                   getattr(self.tokens, name).element_size()
                   for name in _TOKEN_TENSORS)

    @property
    def template_allocated_bytes(self):
        table = self._template_table
        return int(table.numel() * table.element_size())

    def clone(self):
        copied = encoder.RawCandidateTokens(
            **{name: getattr(self.tokens, name).clone() for name in _TOKEN_TENSORS},
            request_ids=(),
        )
        return CompactDeviceCandidates(
            copied, scratch_rows=self.scratch_rows, device=self.device,
            allow_cpu_reference=self.allow_cpu_reference,
            template_table=self._template_table,
        )

    def copy_masked_device(self, other, mask):
        t = self.t
        if type(other) is not CompactDeviceCandidates:
            raise TypeError('CompactDeviceCandidates required')
        if (other._rows != self._rows or other._max_candidates != self._max_candidates
                or other.device != self.device or other.template_table is not self.template_table):
            raise ValueError('Stable compact row/M/device layout required; this is not truncation')
        if (not isinstance(mask, t.Tensor) or mask.device != self.device
                or mask.dtype != t.bool or mask.shape != (self._rows,)):
            raise ValueError('Device boolean row copy mask required')
        for name in _TOKEN_TENSORS:
            dest = getattr(self.tokens, name)
            src = getattr(other.tokens, name)
            if src.shape != dest.shape or src.dtype != dest.dtype or src.device != dest.device:
                raise ValueError('Stable compact token layout required')
            view = mask if dest.ndim == 1 else mask.view((-1,) + (1,) * (dest.ndim - 1))
            independent=(src.untyped_storage().data_ptr()!=dest.untyped_storage().data_ptr()
                         or src.data_ptr()==dest.data_ptr())
            if self.device.type=='cuda' and src.is_contiguous() and dest.is_contiguous() and mask.is_contiguous() and independent:
                from .device_masked_row_copy import copy_masked_rows_device
                copy_masked_rows_device(src,dest,mask,self._row_copy_module)
            else:
                dest.copy_(t.where(view,src,dest))
        return t.zeros(self._rows, dtype=t.int32, device=self.device)

    def encode_slice_device(self, start, stop, active):
        if type(start) is not int or type(stop) is not int:
            raise ValueError('Static capture-time slice bounds required')
        if not 0 <= start < stop <= self._rows:
            raise ValueError('Slice is outside the compact row range')
        local = stop - start
        if local > self.scratch_rows:
            raise ValueError('Slice exceeds scratch_rows; this is not truncation')
        t = self.t
        if (not isinstance(active, t.Tensor) or active.device != self.device
                or active.dtype != t.bool or active.shape != (local,)):
            raise ValueError('Device boolean row activity required')
        sliced = encoder.RawCandidateTokens(
            **{name: getattr(self.tokens, name)[start:stop] for name in _TOKEN_TENSORS},
            request_ids=(),
        )
        return encoder.encode_tokens_device(
            sliced, device=self.device, allow_cpu_reference=self.allow_cpu_reference,
            active=active, template_table=self._template_table,
        )

    def predict_device(self, inference, observations, valid, model_ids, value_actor, active):
        t = self.t
        b, m = self._rows, self._max_candidates
        if (not isinstance(observations, t.Tensor) or observations.device != self.device
                or observations.dtype != t.float32 or observations.ndim != 2
                or observations.shape[0] != b or observations.shape[1] < 1):
            raise ValueError('Exact bound-device observations required')
        self._require_tensor(valid, t.bool, (b, m), 'valid', ndim=2)
        self._require_tensor(model_ids, t.int64, (b,), 'model_ids')
        self._require_tensor(value_actor, t.float32, (b, 1), 'value_actor', ndim=2)
        self._require_tensor(active, t.bool, (b,), 'active')
        if getattr(inference, 'device', None) != self.device:
            raise ValueError('Inference device must match compact candidates')
        logits = t.zeros((b, m), dtype=t.float32, device=self.device)
        wdl = t.zeros((b, 3), dtype=t.float32, device=self.device)
        invalid = t.zeros(b, dtype=t.bool, device=self.device)
        for start, stop in self._slices():
            local_active = active[start:stop]
            packed, enc_err = self.encode_slice_device(start, stop, local_active)
            local_valid = valid[start:stop]
            mismatch = local_active & (local_valid != packed.mask).any(-1)
            mismatch = mismatch | (local_active & (local_valid != self.tokens.mask[start:stop]).any(-1))
            result = inference.predict_device(
                observations[start:stop], packed.candidates, local_valid,
                model_ids[start:stop], value_actor[start:stop],
            )
            usable = local_active & (enc_err == 0) & ~mismatch & ~result.invalid_rows
            logits[start:stop] = t.where(usable[:, None], result.logits, 0.)
            wdl[start:stop] = t.where(usable[:, None], result.wdl, 0.)
            invalid[start:stop] = local_active & ~usable
        return DeviceInferenceResult(logits, wdl, invalid)

    def hash_device(self, information_keys, observations, counts, active):
        t = self.t
        b = self._rows
        if (not isinstance(observations, t.Tensor) or observations.device != self.device
                or observations.dtype != t.float32 or observations.ndim != 2
                or observations.shape[0] != b or observations.shape[1] < 1):
            raise ValueError('Exact bound-device observations required')
        self._require_tensor(counts, t.int64, (b,), 'counts')
        self._require_tensor(active, t.bool, (b,), 'active')
        if getattr(information_keys, 'device', None) != self.device:
            raise ValueError('Information-key device must match compact candidates')
        keys = t.zeros((b, 8), dtype=t.int64, device=self.device)
        errors = t.zeros(b, dtype=t.int32, device=self.device)
        for start, stop in self._slices():
            local_active = active[start:stop]
            packed, enc_err = self.encode_slice_device(start, stop, local_active)
            local_counts = counts[start:stop]
            count_mismatch = local_active & (local_counts != self.tokens.counts[start:stop])
            hashed = information_keys.hash(
                observations[start:stop].contiguous(), packed.candidates.contiguous(),
                local_counts.contiguous(), local_active.contiguous(),
            )
            local_errors = hashed.errors
            local_errors = t.where((enc_err != 0) & (local_errors == 0), enc_err, local_errors)
            local_errors = t.where(count_mismatch & (local_errors == 0), 1, local_errors)
            local_keys = t.where((local_errors != 0)[:, None], 0, hashed.keys)
            keys[start:stop] = local_keys
            errors[start:stop] = local_errors
        return DeviceKeyResult(keys, errors)

    def _slices(self):
        return [(start, min(start + self.scratch_rows, self._rows))
                for start in range(0, self._rows, self.scratch_rows)]

    def _require_tensor(self, value, dtype, shape, name, ndim=1):
        t = self.t
        if (not isinstance(value, t.Tensor) or value.device != self.device
                or value.dtype != dtype or value.ndim != ndim or value.shape != shape):
            raise ValueError('Exact bound-device ' + name + ' required')

