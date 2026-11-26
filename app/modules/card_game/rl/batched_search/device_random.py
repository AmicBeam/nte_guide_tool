"""GPU-resident capture-safe NumPy PCG64 primitive for Gumbel search.

This is an RNG primitive only. It does not run games, 1600-root search, or
GPU training. Importing this module does not import Torch or CUDA.

Setup expands arbitrary-width seeds through NumPy PCG64 / SeedSequence on
CPU, then uploads the verified 128-bit state, increment, and uint32 cache.
The hot path advances that state on CUDA: raw 64-bit outputs, uniform
doubles, Gumbel noise, categorical choice, and bounded integers. Inactive rows are not
advanced. CPU is an explicit numerical oracle only.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping, Sequence, Tuple

import numpy as np


KIND = 'numpy_pcg64_xsl_rr'
SOURCE_VERSION = 'duel_v2_device_pcg64_v1'
MASK64 = (1 << 64) - 1
MASK128 = (1 << 128) - 1
MASK32 = (1 << 32) - 1
PROB_ATOL = sqrt(np.finfo(np.float64).eps)

PCG64_OK = 0
PCG64_ERR_PROB = 1
PCG64_ERR_EMPTY = 2
PCG64_ERR_SCALE = 3
PCG64_ERR_STATE = 5
PCG64_ERR_BOUNDS = 6
PCG64_ERR_REJECTION = 7

_CUDA_SOURCE = r'''
/* duel_v2_device_pcg64_v1: NumPy PCG64 XSL-RR, 128-bit state/inc, uint64 limbs. */
extern "C" __global__ void pcg64_reset(
 unsigned long long* state_hi, unsigned long long* state_lo,
 unsigned long long* inc_hi, unsigned long long* inc_lo,
 unsigned int* has_uint32, unsigned int* uinteger,
 const unsigned long long* next_hi, const unsigned long long* next_lo,
 const unsigned long long* next_inc_hi, const unsigned long long* next_inc_lo,
 const unsigned int* next_has_uint32, const unsigned int* next_uinteger,
 const unsigned char* active, int* errors, int rows) {
 int row = blockIdx.x * blockDim.x + threadIdx.x;
 if (row >= rows) return;
 errors[row] = 0;
 if (!active[row]) return;
 unsigned long long hi=next_hi[row], lo=next_lo[row];
 unsigned long long ihi=next_inc_hi[row], ilo=next_inc_lo[row];
 unsigned int has=next_has_uint32[row], cached=next_uinteger[row];
 if (!(ilo & 1ULL) || has > 1u) { errors[row]=5; return; }
 state_hi[row]=hi; state_lo[row]=lo; inc_hi[row]=ihi; inc_lo[row]=ilo;
 has_uint32[row]=has; uinteger[row]=cached;
}
__device__ __forceinline__ unsigned long long pcg_checked_add(
 unsigned long long a, unsigned long long b, unsigned int* carry) {
 unsigned long long s = a + b;
 *carry = (unsigned int)(s < a);
 return s;
}
__device__ __forceinline__ void pcg64_step(
 unsigned long long* s_hi, unsigned long long* s_lo,
 unsigned long long inc_hi, unsigned long long inc_lo) {
 const unsigned long long MUL_HI = 2549297995355413924ULL;
 const unsigned long long MUL_LO = 4865540595714422341ULL;
 unsigned long long lo = (*s_lo) * MUL_LO;
 unsigned long long hi = __umul64hi(*s_lo, MUL_LO);
 hi = hi + ((*s_lo) * MUL_HI) + ((*s_hi) * MUL_LO);
 unsigned int carry = 0;
 lo = pcg_checked_add(lo, inc_lo, &carry);
 hi = hi + inc_hi + (unsigned long long)carry;
 *s_lo = lo;
 *s_hi = hi;
}
__device__ __forceinline__ unsigned long long pcg64_output(
 unsigned long long hi, unsigned long long lo) {
 unsigned long long xsl = hi ^ lo;
 unsigned int rot = (unsigned int)(hi >> 58);
 return (xsl >> rot) | (xsl << ((-rot) & 63u));
}
__device__ __forceinline__ unsigned long long pcg64_next64(
 unsigned long long* s_hi, unsigned long long* s_lo,
 unsigned long long inc_hi, unsigned long long inc_lo) {
 pcg64_step(s_hi, s_lo, inc_hi, inc_lo);
 return pcg64_output(*s_hi, *s_lo);
}
__device__ __forceinline__ double pcg64_next_double(
 unsigned long long* s_hi, unsigned long long* s_lo,
 unsigned long long inc_hi, unsigned long long inc_lo) {
 return (double)(pcg64_next64(s_hi, s_lo, inc_hi, inc_lo) >> 11) *
  (1.0 / 9007199254740992.0);
}
extern "C" __global__ void pcg64_raw64(
 unsigned long long* state_hi, unsigned long long* state_lo,
 const unsigned long long* inc_hi, const unsigned long long* inc_lo,
 const unsigned char* active, unsigned long long* out, int* errors,
 int rows, int draws, int max_draws) {
 int row = blockIdx.x * blockDim.x + threadIdx.x;
 if (row >= rows) return;
 errors[row] = 0;
 for (int d = 0; d < draws; d++) out[(long long)row * max_draws + d] = 0ULL;
 if (active[row] == 0) return;
 unsigned long long s_hi = state_hi[row];
 unsigned long long s_lo = state_lo[row];
 unsigned long long i_hi = inc_hi[row];
 unsigned long long i_lo = inc_lo[row];
 for (int d = 0; d < draws; d++) {
  out[(long long)row * max_draws + d] = pcg64_next64(&s_hi, &s_lo, i_hi, i_lo);
 }
 state_hi[row] = s_hi;
 state_lo[row] = s_lo;
}
extern "C" __global__ void pcg64_uniform(
 unsigned long long* state_hi, unsigned long long* state_lo,
 const unsigned long long* inc_hi, const unsigned long long* inc_lo,
 const unsigned char* active, double* out, int* errors,
 int rows, int draws, int max_draws) {
 int row = blockIdx.x * blockDim.x + threadIdx.x;
 if (row >= rows) return;
 errors[row] = 0;
 for (int d = 0; d < draws; d++) out[(long long)row * max_draws + d] = 0.0;
 if (active[row] == 0) return;
 unsigned long long s_hi = state_hi[row];
 unsigned long long s_lo = state_lo[row];
 unsigned long long i_hi = inc_hi[row];
 unsigned long long i_lo = inc_lo[row];
 for (int d = 0; d < draws; d++) {
  out[(long long)row * max_draws + d] =
   pcg64_next_double(&s_hi, &s_lo, i_hi, i_lo);
 }
 state_hi[row] = s_hi;
 state_lo[row] = s_lo;
}
extern "C" __global__ void pcg64_gumbel(
 unsigned long long* state_hi, unsigned long long* state_lo,
 const unsigned long long* inc_hi, const unsigned long long* inc_lo,
 const unsigned char* active, double* out, int* errors,
 int rows, int draws, int max_draws, double loc, double scale) {
 int row = blockIdx.x * blockDim.x + threadIdx.x;
 if (row >= rows) return;
 errors[row] = 0;
 for (int d = 0; d < draws; d++) out[(long long)row * max_draws + d] = 0.0;
 if (active[row] == 0) return;
 if (!(scale >= 0.0) || scale != scale) {
  errors[row] = 3;
  return;
 }
 unsigned long long s_hi = state_hi[row];
 unsigned long long s_lo = state_lo[row];
 unsigned long long i_hi = inc_hi[row];
 unsigned long long i_lo = inc_lo[row];
 for (int d = 0; d < draws; d++) {
  double ux=1.0;
  for(int retry=0;retry<128 && ux>=1.0;retry++) ux=1.0-pcg64_next_double(&s_hi,&s_lo,i_hi,i_lo);
  if(ux>=1.0){errors[row]=4;return;}
  double value = loc - scale * log(-log(ux));
  out[(long long)row * max_draws + d] = value;
 }
 state_hi[row] = s_hi;
 state_lo[row] = s_lo;
}
extern "C" __global__ void pcg64_gumbel_valid(
 unsigned long long* state_hi, unsigned long long* state_lo,
 const unsigned long long* inc_hi, const unsigned long long* inc_lo,
 const unsigned char* active, double* out, int* errors,
 int rows, int draws, int max_draws, double loc, double scale,const unsigned char* valid) {
 int row = blockIdx.x * blockDim.x + threadIdx.x;
 if (row >= rows) return;
 errors[row] = 0;
 for (int d = 0; d < draws; d++) out[(long long)row * max_draws + d] = 0.0;
 if (active[row] == 0) return;
 if (!(scale >= 0.0) || scale != scale) {
  errors[row] = 3;
  return;
 }
 unsigned long long s_hi = state_hi[row];
 unsigned long long s_lo = state_lo[row];
 unsigned long long i_hi = inc_hi[row];
 unsigned long long i_lo = inc_lo[row];
 for (int d = 0; d < draws; d++) {
  if(!valid[(long long)row*draws+d])continue;
  double ux=1.0;
  for(int retry=0;retry<128 && ux>=1.0;retry++) ux=1.0-pcg64_next_double(&s_hi,&s_lo,i_hi,i_lo);
  if(ux>=1.0){errors[row]=4;return;}
  double value = loc - scale * log(-log(ux));
  out[(long long)row * max_draws + d] = value;
 }
 state_hi[row] = s_hi;
 state_lo[row] = s_lo;
}
extern "C" __global__ void pcg64_categorical(
 unsigned long long* state_hi, unsigned long long* state_lo,
 const unsigned long long* inc_hi, const unsigned long long* inc_lo,
 const unsigned char* active, const double* probs, const long long* counts,
 long long* indices, int* errors, int rows, int max_choices) {
 int row = blockIdx.x * blockDim.x + threadIdx.x;
 if (row >= rows) return;
 errors[row] = 0;
 indices[row] = 0;
 if (active[row] == 0) return;
 long long n = counts[row];
 if (n < 1 || n > max_choices) {
  errors[row] = 2;
  indices[row] = -1;
  return;
 }
 double sum = 0.0;
 int bad = 0;
 for (int i = 0; i < (int)n; i++) {
  double p = probs[(long long)row * max_choices + i];
  if (!(p >= 0.0)) bad = 1;
  sum += p;
 }
 if (bad || fabs(sum - 1.0) > 1.4901161193847656e-8) {
  errors[row] = 1;
  indices[row] = -1;
  return;
 }
 unsigned long long s_hi = state_hi[row];
 unsigned long long s_lo = state_lo[row];
 unsigned long long i_hi = inc_hi[row];
 unsigned long long i_lo = inc_lo[row];
 double u = pcg64_next_double(&s_hi, &s_lo, i_hi, i_lo);
 double cdf = 0.0;
 long long choice = n - 1;
 for (int i = 0; i < (int)n; i++) {
  cdf += probs[(long long)row * max_choices + i];
  if ((cdf / sum) > u) { choice = i; break; }
 }
 indices[row] = choice;
 state_hi[row] = s_hi;
 state_lo[row] = s_lo;
}
__device__ __forceinline__ unsigned int pcg64_next32(
 unsigned long long* s_hi, unsigned long long* s_lo,
 unsigned long long inc_hi, unsigned long long inc_lo,
 unsigned int* has, unsigned int* cached) {
 if (*has) { *has=0u; return *cached; }
 unsigned long long value=pcg64_next64(s_hi,s_lo,inc_hi,inc_lo);
 *has=1u; *cached=(unsigned int)(value>>32);
 return (unsigned int)value;
}
extern "C" __global__ void pcg64_bounded_integers(
 unsigned long long* state_hi, unsigned long long* state_lo,
 const unsigned long long* inc_hi, const unsigned long long* inc_lo,
 unsigned int* has_uint32, unsigned int* uinteger,
 const long long* highs, const unsigned char* active,
 long long* indices, int* errors, int rows, int max_retries) {
 int row=blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=rows)return;
 indices[row]=0; errors[row]=0;
 if(!active[row])return;
 long long high=highs[row];
 if(high<1||high>2147483647LL){indices[row]=-1;errors[row]=6;return;}
 unsigned long long hi=state_hi[row],lo=state_lo[row];
 unsigned long long ihi=inc_hi[row],ilo=inc_lo[row];
 unsigned int has=has_uint32[row],cached=uinteger[row];
 if(!(ilo&1ULL)||has>1u){indices[row]=-1;errors[row]=5;return;}
 if(high==1)return;
 unsigned int bound=(unsigned int)high;
 unsigned int threshold=(0u-bound)%bound;
 // Commit all mutable limbs/cache only after an accepted Lemire draw.
 // Rejection exhaustion is an explicit failure, never a biased choice.
 for(int retry=0;retry<max_retries;retry++){
  unsigned int value=pcg64_next32(&hi,&lo,ihi,ilo,&has,&cached);
  unsigned long long product=(unsigned long long)value*bound;
  if((unsigned int)product<threshold)continue;
  indices[row]=(long long)(product>>32);
  state_hi[row]=hi;state_lo[row]=lo;
  has_uint32[row]=has;uinteger[row]=cached;
  return;
 }
 indices[row]=-1;errors[row]=7;
}
'''

_KERNEL_SPECS = {
    'pcg64_reset': ('uint64*',) * 4 + ('uint32*',) * 2
        + ('uint64*',) * 4 + ('uint32*',) * 2 + ('uint8*', 'int32*', 'int32'),
    'pcg64_raw64': (
        'uint64*', 'uint64*', 'uint64*', 'uint64*', 'uint8*', 'uint64*', 'int32*',
        'int32', 'int32', 'int32',
    ),
    'pcg64_uniform': (
        'uint64*', 'uint64*', 'uint64*', 'uint64*', 'uint8*', 'float64*', 'int32*',
        'int32', 'int32', 'int32',
    ),
    'pcg64_gumbel': (
        'uint64*', 'uint64*', 'uint64*', 'uint64*', 'uint8*', 'float64*', 'int32*',
        'int32', 'int32', 'int32', 'float64', 'float64',
    ),
    'pcg64_categorical': (
        'uint64*', 'uint64*', 'uint64*', 'uint64*', 'uint8*', 'float64*', 'int64*',
        'int64*', 'int32*', 'int32', 'int32',
    ),
}

_KERNEL_SPECS['pcg64_gumbel_valid'] = _KERNEL_SPECS['pcg64_gumbel'] + ('uint8*',)
_KERNEL_SPECS['pcg64_bounded_integers'] = (
    ('uint64*',) * 4 + ('uint32*',) * 2
    + ('int64*', 'uint8*', 'int64*', 'int32*', 'int32', 'int32'))

def _reject_bool(value: Any, name: str) -> None:
    if type(value) is bool:
        raise TypeError(name + ' rejects bool')


def _as_int(value: Any, name: str) -> int:
    _reject_bool(value, name)
    if isinstance(value, np.integer):
        value = int(value)
    if type(value) is not int:
        raise TypeError(name + ' must be an int')
    return value


def verify_numpy_pcg64_state(state: Mapping[str, Any]) -> dict:
    """Accept a NumPy PCG64 state dict with full 128-bit limbs and uint32 cache."""
    if not isinstance(state, Mapping):
        raise TypeError('NumPy PCG64 state mapping required')
    if state.get('bit_generator') != 'PCG64':
        raise ValueError('NumPy PCG64 (XSL-RR) required; not PCG64DXSM or another generator')
    inner = state.get('state')
    if not isinstance(inner, Mapping) or 'state' not in inner or 'inc' not in inner:
        raise ValueError('PCG64 state/inc required')
    packed = {}
    for key in ('state', 'inc'):
        packed[key] = _as_int(inner[key], key)
        if not 0 <= packed[key] <= MASK128:
            raise ValueError('Full 128-bit PCG ' + key + ' required; not a truncated seed')
    if not packed['inc'] & 1:
        raise ValueError('PCG64 increment must be odd')
    packed['has_uint32'] = _as_int(state.get('has_uint32'), 'has_uint32')
    packed['uinteger'] = _as_int(state.get('uinteger'), 'uinteger')
    if packed['has_uint32'] not in (0, 1):
        raise ValueError('has_uint32 must be 0 or 1')
    if not 0 <= packed['uinteger'] <= MASK32:
        raise ValueError('uinteger must fit in uint32')
    packed['bit_generator'] = 'PCG64'
    return packed


def prepare_numpy_pcg64_states(seeds: Sequence[Any]) -> Tuple[dict, ...]:
    """CPU SeedSequence / PCG64 expansion for one state per root.

    Integer seeds may be wider than 256 bits. This never truncates them.
    Existing verified PCG64 state dicts are copied. This is setup, not the
    hot path.
    """
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise TypeError('Sequence of seeds or verified PCG64 states required')
    if len(seeds) < 1:
        raise ValueError('Nonempty batch of PCG64 roots required')
    out = []
    for item in seeds:
        if isinstance(item, Mapping):
            packed = verify_numpy_pcg64_state(item)
            out.append({
                'bit_generator': 'PCG64',
                'state': {'state': packed['state'], 'inc': packed['inc']},
                'has_uint32': packed['has_uint32'],
                'uinteger': packed['uinteger'],
            })
            continue
        if type(item) is bool:
            raise TypeError('PCG64 seed rejects bool')
        if isinstance(item, np.integer):
            item = int(item)
        if type(item) is not int:
            raise TypeError('PCG64 seed must be an int or verified state dict')
        if item < 0:
            raise ValueError('PCG64 seed must be a non-negative integer')
        bg = np.random.PCG64(item)
        out.append({
            'bit_generator': 'PCG64',
            'state': {
                'state': int(bg.state['state']['state']),
                'inc': int(bg.state['state']['inc']),
            },
            'has_uint32': int(bg.state['has_uint32']),
            'uinteger': int(bg.state['uinteger']),
        })
    return tuple(out)


def _limbs(value: int) -> Tuple[int, int]:
    return (value >> 64) & MASK64, value & MASK64


def _join(hi: int, lo: int) -> int:
    return ((hi & MASK64) << 64) | (lo & MASK64)


def numpy_pcg64_state_from_limbs(state_hi, state_lo, inc_hi, inc_lo, has_uint32, uinteger) -> dict:
    return {
        'bit_generator': 'PCG64',
        'state': {'state': _join(int(state_hi), int(state_lo)),
                  'inc': _join(int(inc_hi), int(inc_lo))},
        'has_uint32': int(has_uint32) & 1,
        'uinteger': int(uinteger) & MASK32,
    }


def _generator_from_packed(packed: Mapping[str, int]) -> np.random.Generator:
    inner = verify_numpy_pcg64_state(packed)
    bg = np.random.PCG64()
    bg.state = {
        'bit_generator': 'PCG64',
        'state': {'state': inner['state'], 'inc': inner['inc']},
        'has_uint32': inner['has_uint32'],
        'uinteger': inner['uinteger'],
    }
    return np.random.Generator(bg)


@dataclass(frozen=True)
class DevicePCG64State:
    state_hi: Any
    state_lo: Any
    inc_hi: Any
    inc_lo: Any
    has_uint32: Any
    uinteger: Any


@dataclass(frozen=True)
class DeviceRandomU64:
    values: Any
    errors: Any
    state: DevicePCG64State


@dataclass(frozen=True)
class DeviceRandomF64:
    values: Any
    errors: Any
    state: DevicePCG64State


@dataclass(frozen=True)
class DeviceCategorical:
    indices: Any
    errors: Any
    state: DevicePCG64State


class DevicePCG64:
    """Batched NumPy-PCG64 state on device with masked hot-path draws.

    supports_full_duel_rules is always false. Workspace size is a bound, not
    a completed-search or training claim.
    """

    supports_full_duel_rules = False
    kind = KIND
    version = SOURCE_VERSION

    def __init__(self, states: Sequence[Any], *, device='cuda', max_rows=None,
                 max_draws=256, max_choices=256, allow_cpu_reference=False):
        import torch
        prepared = prepare_numpy_pcg64_states(states)
        loaded = len(prepared)
        if max_rows is None:
            max_rows = loaded
        for value, name in ((max_rows, 'max_rows'), (max_draws, 'max_draws'),
                            (max_choices, 'max_choices')):
            _reject_bool(value, name)
            if type(value) is not int or value < 1 or value >= 2 ** 31:
                raise ValueError('Positive int32 bounded ' + name + ' required')
        if loaded > max_rows:
            raise ValueError('PCG64 batch exceeds max_rows; this is not truncation')
        self.t = torch
        self.device = torch.device(device)
        self.max_rows = max_rows
        self.max_draws = max_draws
        self.max_choices = max_choices
        self.module = None
        self._loaded = 0
        if self.device.type == 'cpu' and allow_cpu_reference is True:
            self._allocate()
            self.load_states(prepared)
            return
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; CPU is an explicit PCG64 oracle only')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        from ..batched_duel.cuda_runtime import CudaModule
        self.module = CudaModule(_CUDA_SOURCE, _KERNEL_SPECS, device=str(self.device))
        self._allocate()
        self.load_states(prepared)

    def _allocate(self):
        t, z = self.t, dict(device=self.device)
        self._state_hi = t.zeros(self.max_rows, dtype=t.uint64, **z)
        self._state_lo = t.zeros(self.max_rows, dtype=t.uint64, **z)
        self._inc_hi = t.zeros(self.max_rows, dtype=t.uint64, **z)
        self._inc_lo = t.zeros(self.max_rows, dtype=t.uint64, **z)
        self._has_uint32 = t.zeros(self.max_rows, dtype=t.uint32, **z)
        self._uinteger = t.zeros(self.max_rows, dtype=t.uint32, **z)
        self._out_u64 = t.zeros((self.max_rows, self.max_draws), dtype=t.uint64, **z)
        self._out_f64 = t.zeros((self.max_rows, self.max_draws), dtype=t.float64, **z)
        self._indices = t.zeros(self.max_rows, dtype=t.int64, **z)
        self._errors = t.zeros(self.max_rows, dtype=t.int32, **z)
        self._reset_errors = t.zeros(self.max_rows, dtype=t.int32, **z)
        self._counts = t.zeros(self.max_rows, dtype=t.int64, **z)
        self._probs = t.zeros((self.max_rows, self.max_choices), dtype=t.float64, **z)

    @property
    def state(self) -> DevicePCG64State:
        return DevicePCG64State(
            self._state_hi, self._state_lo, self._inc_hi, self._inc_lo,
            self._has_uint32, self._uinteger,
        )

    @property
    def stream(self):
        return None if self.module is None else self.module.stream

    @property
    def loaded_rows(self) -> int:
        return self._loaded

    @property
    def allocated_bytes(self) -> int:
        names = ('_state_hi', '_state_lo', '_inc_hi', '_inc_lo', '_has_uint32',
                 '_uinteger', '_out_u64', '_out_f64', '_indices', '_errors',
                 '_counts', '_probs', '_reset_errors')
        return sum(getattr(self, name).numel() * getattr(self, name).element_size()
                   for name in names)

    def load_states(self, states: Sequence[Any]) -> DevicePCG64State:
        prepared = prepare_numpy_pcg64_states(states)
        n = len(prepared)
        if n > self.max_rows:
            raise ValueError('PCG64 batch exceeds max_rows; this is not truncation')
        state_hi = np.zeros(self.max_rows, dtype=np.uint64)
        state_lo = np.zeros(self.max_rows, dtype=np.uint64)
        inc_hi = np.zeros(self.max_rows, dtype=np.uint64)
        inc_lo = np.zeros(self.max_rows, dtype=np.uint64)
        has_uint32 = np.zeros(self.max_rows, dtype=np.uint32)
        uinteger = np.zeros(self.max_rows, dtype=np.uint32)
        for i, packed in enumerate(prepared):
            inner = verify_numpy_pcg64_state(packed)
            state_hi[i], state_lo[i] = _limbs(inner['state'])
            inc_hi[i], inc_lo[i] = _limbs(inner['inc'])
            has_uint32[i] = inner['has_uint32']
            uinteger[i] = inner['uinteger']
        t = self.t
        self._state_hi.copy_(t.as_tensor(state_hi, device=self.device))
        self._state_lo.copy_(t.as_tensor(state_lo, device=self.device))
        self._inc_hi.copy_(t.as_tensor(inc_hi, device=self.device))
        self._inc_lo.copy_(t.as_tensor(inc_lo, device=self.device))
        self._has_uint32.copy_(t.as_tensor(has_uint32, device=self.device))
        self._uinteger.copy_(t.as_tensor(uinteger, device=self.device))
        self._loaded = n
        return self.state

    def export_numpy_states(self) -> Tuple[dict, ...]:
        """Host-side reconstruction for tests and setup. Not a hot-path API."""
        n = self._loaded
        parts = [tensor[:n].detach().cpu().numpy() for tensor in (
            self._state_hi, self._state_lo, self._inc_hi, self._inc_lo,
            self._has_uint32, self._uinteger,
        )]
        return tuple(numpy_pcg64_state_from_limbs(*row) for row in zip(*parts))

    def reset_device(self, state: DevicePCG64State, active):
        """Reset actual decision rows from device-gathered NumPy state limbs.

        SeedSequence expansion belongs to batch setup. The caller selects the
        prepared state by the actual game step and actor on device; this call
        neither expands seeds nor advances randomness. Returns int32 row errors
        (5 for an even increment or invalid cache flag). Invalid and inactive
        rows preserve all six old fields. CPU execution is an explicit oracle.
        """
        rows = self._active(active)
        t = self.t
        if not isinstance(state, DevicePCG64State):
            raise TypeError('DevicePCG64State required')
        fields = (state.state_hi, state.state_lo, state.inc_hi, state.inc_lo,
                  state.has_uint32, state.uinteger)
        for value, dtype in zip(fields, (t.uint64,) * 4 + (t.uint32,) * 2):
            if (not isinstance(value, t.Tensor) or value.device != self.device
                    or value.dtype != dtype or value.shape != (rows,)
                    or not value.is_contiguous()):
                raise ValueError('Exact contiguous device PCG64 state vectors required')
        if self.device.type == 'cpu':
            # Copy each row's inputs before writes so self.state is also valid.
            self._reset_errors.zero_()
            targets = (self._state_hi, self._state_lo, self._inc_hi, self._inc_lo,
                       self._has_uint32, self._uinteger)
            for row in range(rows):
                if not bool(active[row]):
                    continue
                values = [int(value[row].item()) for value in fields]
                if not values[3] & 1 or values[4] > 1:
                    self._reset_errors[row] = PCG64_ERR_STATE
                    continue
                for target, value in zip(targets, values):
                    target[row] = value
        else:
            self._launch('pcg64_reset', [
                self._state_hi, self._state_lo, self._inc_hi, self._inc_lo,
                self._has_uint32, self._uinteger, *fields,
                active.view(t.uint8), self._reset_errors, rows,
            ])
        return self._reset_errors[:rows]

    def _active(self, active):
        t = self.t
        if (not isinstance(active, t.Tensor) or active.device != self.device
                or active.dtype != t.bool or active.ndim != 1 or not active.is_contiguous()):
            raise ValueError('Contiguous device bool active mask required')
        rows = active.shape[0]
        if rows != self._loaded or not 0 < rows <= self.max_rows:
            raise ValueError('Active mask must cover the loaded PCG64 batch')
        return rows

    def _positive_draws(self, draws):
        _reject_bool(draws, 'draws')
        if type(draws) is not int or not 1 <= draws <= self.max_draws:
            raise ValueError('draws must be an int in 1..max_draws')
        return draws

    def _launch(self, name, args):
        capture = self.t.cuda.is_current_stream_capturing()
        event = self.module.launch(
            name, args, grid=(self.max_rows + 127) // 128, block=128,capture_safe=capture,
        )
        if event is not None:
            self.t.cuda.current_stream(self.device).wait_event(event)
        return event

    def _packed_row(self, row: int) -> dict:
        return numpy_pcg64_state_from_limbs(
            int(self._state_hi[row].item()), int(self._state_lo[row].item()),
            int(self._inc_hi[row].item()), int(self._inc_lo[row].item()),
            int(self._has_uint32[row].item()), int(self._uinteger[row].item()),
        )

    def _store_packed(self, row: int, generator: np.random.Generator) -> None:
        packed = verify_numpy_pcg64_state(generator.bit_generator.state)
        hi, lo = _limbs(packed['state'])
        ihi, ilo = _limbs(packed['inc'])
        self._state_hi[row] = hi
        self._state_lo[row] = lo
        self._inc_hi[row] = ihi
        self._inc_lo[row] = ilo
        self._has_uint32[row] = packed['has_uint32']
        self._uinteger[row] = packed['uinteger']

    def _oracle_raw64(self, active, draws):
        rows = active.shape[0]
        values = self._out_u64[:rows, :draws]
        errors = self._errors[:rows]
        values.zero_()
        errors.zero_()
        on = active.numpy()
        for row in range(rows):
            if not on[row]:
                continue
            generator = _generator_from_packed(self._packed_row(row))
            raw = generator.bit_generator.random_raw(draws)
            values[row].copy_(self.t.as_tensor(np.asarray(raw, dtype=np.uint64)))
            self._store_packed(row, generator)
        return DeviceRandomU64(values, errors, self.state)

    def _oracle_uniform(self, active, draws):
        rows = active.shape[0]
        values = self._out_f64[:rows, :draws]
        errors = self._errors[:rows]
        values.zero_()
        errors.zero_()
        on = active.numpy()
        for row in range(rows):
            if not on[row]:
                continue
            generator = _generator_from_packed(self._packed_row(row))
            values[row].copy_(self.t.as_tensor(generator.random(draws), dtype=self.t.float64))
            self._store_packed(row, generator)
        return DeviceRandomF64(values, errors, self.state)

    def _oracle_gumbel(self, active, draws, loc, scale):
        rows = active.shape[0]
        values = self._out_f64[:rows, :draws]
        errors = self._errors[:rows]
        values.zero_()
        errors.zero_()
        on = active.numpy()
        for row in range(rows):
            if not on[row]:
                continue
            if not (scale >= 0.0) or scale != scale:
                errors[row] = PCG64_ERR_SCALE
                continue
            generator = _generator_from_packed(self._packed_row(row))
            values[row].copy_(self.t.as_tensor(
                generator.gumbel(loc=loc, scale=scale, size=draws), dtype=self.t.float64,
            ))
            self._store_packed(row, generator)
        return DeviceRandomF64(values, errors, self.state)

    def _oracle_categorical(self, probs, counts, active):
        rows = active.shape[0]
        indices = self._indices[:rows]
        errors = self._errors[:rows]
        indices.zero_()
        errors.zero_()
        on = active.numpy()
        p = probs.numpy()
        nvec = counts.numpy()
        for row in range(rows):
            if not on[row]:
                continue
            n = int(nvec[row])
            if n < 1 or n > self.max_choices or n > p.shape[1]:
                errors[row] = PCG64_ERR_EMPTY
                indices[row] = -1
                continue
            sl = p[row, :n]
            if (not np.isfinite(sl).all()) or np.any(sl < 0) or abs(float(sl.sum()) - 1.0) > PROB_ATOL:
                errors[row] = PCG64_ERR_PROB
                indices[row] = -1
                continue
            generator = _generator_from_packed(self._packed_row(row))
            indices[row] = int(generator.choice(n, p=sl))
            self._store_packed(row, generator)
        return DeviceCategorical(indices, errors, self.state)

    def raw64(self, active, draws=1) -> DeviceRandomU64:
        rows = self._active(active)
        draws = self._positive_draws(draws)
        if self.device.type == 'cpu':
            return self._oracle_raw64(active, draws)
        self._launch('pcg64_raw64', [
            self._state_hi, self._state_lo, self._inc_hi, self._inc_lo,
            active.view(self.t.uint8), self._out_u64, self._errors,
            rows, draws, self.max_draws,
        ])
        return DeviceRandomU64(self._out_u64[:rows, :draws], self._errors[:rows], self.state)

    def uniform(self, active, draws=1) -> DeviceRandomF64:
        rows = self._active(active)
        draws = self._positive_draws(draws)
        if self.device.type == 'cpu':
            return self._oracle_uniform(active, draws)
        self._launch('pcg64_uniform', [
            self._state_hi, self._state_lo, self._inc_hi, self._inc_lo,
            active.view(self.t.uint8), self._out_f64, self._errors,
            rows, draws, self.max_draws,
        ])
        return DeviceRandomF64(self._out_f64[:rows, :draws], self._errors[:rows], self.state)

    def gumbel(self, active, draws=1, loc=0.0, scale=1.0) -> DeviceRandomF64:
        rows = self._active(active)
        draws = self._positive_draws(draws)
        _reject_bool(loc, 'loc')
        _reject_bool(scale, 'scale')
        loc = float(loc)
        scale = float(scale)
        if loc != loc or scale != scale:
            raise ValueError('Finite Gumbel loc/scale required')
        if scale < 0.0:
            raise ValueError('Gumbel scale must be non-negative')
        if self.device.type == 'cpu':
            return self._oracle_gumbel(active, draws, loc, scale)
        self._launch('pcg64_gumbel', [
            self._state_hi, self._state_lo, self._inc_hi, self._inc_lo,
            active.view(self.t.uint8), self._out_f64, self._errors,
            rows, draws, self.max_draws, loc, scale,
        ])
        return DeviceRandomF64(self._out_f64[:rows, :draws], self._errors[:rows], self.state)

    def categorical(self, probs, counts, active) -> DeviceCategorical:
        rows = self._active(active)
        t = self.t
        if (not isinstance(probs, t.Tensor) or probs.device != self.device
                or probs.dtype != t.float64 or probs.ndim != 2 or not probs.is_contiguous()):
            raise ValueError('Contiguous device float64 probability rows required')
        if (not isinstance(counts, t.Tensor) or counts.device != self.device
                or counts.dtype != t.int64 or counts.ndim != 1 or not counts.is_contiguous()):
            raise ValueError('Contiguous device int64 categorical counts required')
        if probs.shape[0] != rows or counts.shape[0] != rows:
            raise ValueError('Categorical rows must match the active PCG64 batch')
        if probs.shape[1] > self.max_choices or probs.shape[1] < 1:
            raise ValueError('Bounded categorical width required')
        if self.device.type == 'cpu':
            return self._oracle_categorical(probs, counts, active)
        self._probs.zero_()
        self._counts.zero_()
        self._probs[:rows, :probs.shape[1]].copy_(probs)
        self._counts[:rows].copy_(counts)
        self._launch('pcg64_categorical', [
            self._state_hi, self._state_lo, self._inc_hi, self._inc_lo,
            active.view(self.t.uint8), self._probs, self._counts,
            self._indices, self._errors, rows, self.max_choices,
        ])
        return DeviceCategorical(self._indices[:rows], self._errors[:rows], self.state)

    def gumbel_device(self, valid, active) -> DeviceRandomF64:
        rows=self._active(active)
        t=self.t
        if (not isinstance(valid,t.Tensor) or valid.device!=self.device or valid.dtype!=t.bool
                or valid.ndim!=2 or valid.shape[0]!=rows or not 0<valid.shape[1]<=self.max_draws):
            raise ValueError('Exact device legal mask required')
        draws=valid.shape[1]
        if self.device.type=='cpu':
            values=self._out_f64[:rows,:draws];errors=self._errors[:rows]
            values.zero_();errors.zero_()
            for row in range(rows):
                if not bool(active[row]):continue
                mask=valid[row].numpy()
                generator=_generator_from_packed(self._packed_row(row))
                values[row,valid[row]]=t.as_tensor(generator.gumbel(size=int(mask.sum())),dtype=t.float64)
                self._store_packed(row,generator)
            return DeviceRandomF64(values,errors,self.state)
        self._launch('pcg64_gumbel_valid',[
            self._state_hi,self._state_lo,self._inc_hi,self._inc_lo,
            active.view(t.uint8),self._out_f64,self._errors,rows,draws,self.max_draws,0.0,1.0,
            valid.to(t.uint8).contiguous()])
        return DeviceRandomF64(self._out_f64[:rows,:draws],self._errors[:rows],self.state)

    def categorical_device(self, probs, active, *, counts=None):
        if counts is None:
            counts=self.t.full((probs.shape[0],),probs.shape[1],dtype=self.t.int64,device=self.device)
        return self.categorical(probs,counts,active)

    def integers_device(self, highs, active, *, max_retries=128) -> DeviceCategorical:
        """One exact NumPy PCG64 integers(high) draw per admitted row, low=0.

        Supports int64 highs in 1..2**31-1, using the uint32 Lemire branch.
        high=1 and inactive/invalid rows consume no randomness. Bounds error6
        and rejection-budget error7 return index -1; budget failure rolls back
        all six state fields. Errors must abort the caller's trajectory.
        CPU mode is only the explicitly constructed numerical oracle.
        """
        rows = self._active(active)
        t = self.t
        if (not isinstance(highs, t.Tensor) or highs.device != self.device
                or highs.dtype != t.int64 or highs.shape != (rows,) or not highs.is_contiguous()):
            raise ValueError('Exact contiguous device int64 upper bounds required')
        _reject_bool(max_retries, 'max_retries')
        if type(max_retries) is not int or not 1 <= max_retries <= 1024:
            raise ValueError('Explicit bounded rejection budget in 1..1024 required')
        if self.device.type == 'cpu':
            indices, errors = self._indices[:rows], self._errors[:rows]
            indices.zero_(); errors.zero_()
            for row in range(rows):
                if not bool(active[row]):
                    continue
                high = int(highs[row].item())
                if not 1 <= high <= 2**31-1:
                    indices[row] = -1; errors[row] = PCG64_ERR_BOUNDS
                    continue
                if not int(self._inc_lo[row].item()) & 1 or int(self._has_uint32[row].item()) > 1:
                    indices[row] = -1; errors[row] = PCG64_ERR_STATE
                    continue
                if high == 1:
                    continue
                generator = _generator_from_packed(self._packed_row(row))
                state = generator.bit_generator.state
                has, cached = state['has_uint32'], state['uinteger']
                threshold = ((-high) & MASK32) % high
                for _ in range(max_retries):
                    if has:
                        value, has = cached, 0
                    else:
                        word = int(generator.bit_generator.random_raw())
                        value, has, cached = word & MASK32, 1, word >> 32
                    product = value * high
                    if (product & MASK32) < threshold:
                        continue
                    state = generator.bit_generator.state
                    state['has_uint32'], state['uinteger'] = has, cached
                    generator.bit_generator.state = state
                    self._store_packed(row, generator)
                    indices[row] = product >> 32
                    break
                else:
                    indices[row] = -1; errors[row] = PCG64_ERR_REJECTION
        else:
            self._launch('pcg64_bounded_integers', [
                self._state_hi, self._state_lo, self._inc_hi, self._inc_lo,
                self._has_uint32, self._uinteger, highs, active.view(t.uint8),
                self._indices, self._errors, rows, max_retries,
            ])
        return DeviceCategorical(self._indices[:rows], self._errors[:rows], self.state)

    def close(self) -> None:
        if self.module is not None:
            self.module.close()
            self.module = None
