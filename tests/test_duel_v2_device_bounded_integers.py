"""Actual CUDA kernel source through a CPU shim; no GPU/driver qualification."""
import ast
import copy
import ctypes
import inspect
import os
from pathlib import Path
import tempfile
import textwrap
import unittest

import numpy as np

from app.modules.card_game.rl.batched_search import device_random as random
from app.modules.card_game.rl.batched_duel.native_array_cpu import compile_host_native_library


class HostBank:
    def __init__(self, library, states):
        self.library = library
        packed = [random.verify_numpy_pcg64_state(s) for s in states]
        self.fields = [np.asarray(values, dtype=dtype) for values, dtype in zip(zip(*[
            (*random._limbs(s['state']), *random._limbs(s['inc']), s['has_uint32'], s['uinteger'])
            for s in packed]), [np.uint64]*4+[np.uint32]*2)]
        self.rows = len(states)
    def states(self):
        return tuple(random.numpy_pcg64_state_from_limbs(*row) for row in zip(*self.fields))
    def integers(self, highs, active, retries=128):
        high = np.asarray(highs, dtype=np.int64)
        mask = np.asarray(active, dtype=np.uint8)
        out = np.full(self.rows, -999, np.int64);errors = np.full(self.rows, -999, np.int32)
        arrays = [*self.fields, high, mask, out, errors]
        self.library.host_integers(*[ctypes.c_void_p(a.ctypes.data) for a in arrays], self.rows, retries)
        return out, errors
    def primitive(self, name, active, draws):
        mask = np.asarray(active, dtype=np.uint8)
        out = np.zeros((self.rows, draws), np.uint64 if name == 'raw' else np.float64)
        errors = np.zeros(self.rows, np.int32)
        arrays = [*self.fields[:4], mask, out, errors]
        getattr(self.library, 'host_'+name)(*[ctypes.c_void_p(a.ctypes.data) for a in arrays], self.rows, draws)
        np.testing.assert_array_equal(errors, 0)
        return out


def generator(state):
    bitgen = np.random.PCG64();bitgen.state = copy.deepcopy(state)
    return np.random.Generator(bitgen)


def forced_zero_state():
    state = copy.deepcopy(np.random.PCG64(42).state)
    multiplier = (2549297995355413924 << 64) | 4865540595714422341
    state['state']['state'] = (-state['state']['inc'] * pow(multiplier, -1, 1 << 128)) % (1 << 128)
    state['has_uint32'], state['uinteger'] = 0, 0x12345678
    return state


class BoundedIntegerHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='pcg64_bounded_host_')
        preamble = '''#include <cmath>
#define __device__
#define __global__
#define __forceinline__ inline
struct Dim { int x; }; static Dim blockIdx={0},threadIdx={0},blockDim={1};
inline unsigned long long __umul64hi(unsigned long long a,unsigned long long b){
 unsigned long long low=(unsigned int)a*(unsigned long long)(unsigned int)b;
 unsigned long long mid1=(unsigned int)a*(b>>32),mid2=(a>>32)*(unsigned int)b;
 unsigned long long carry=(low>>32)+(unsigned int)mid1+(unsigned int)mid2;
 return (a>>32)*(b>>32)+(mid1>>32)+(mid2>>32)+(carry>>32);
}
'''
        wrapper = '''
extern "C" void host_integers(unsigned long long* hi,unsigned long long* lo,
 const unsigned long long* ihi,const unsigned long long* ilo,unsigned int* has,unsigned int* cached,
 const long long* high,const unsigned char* active,long long* out,int* errors,int rows,int retries){
 for(int row=0;row<rows+3;row++){blockIdx.x=row;pcg64_bounded_integers(hi,lo,ihi,ilo,has,cached,high,active,out,errors,rows,retries);}
}
extern "C" void host_raw(unsigned long long* hi,unsigned long long* lo,
 const unsigned long long* ihi,const unsigned long long* ilo,const unsigned char* active,
 unsigned long long* out,int* errors,int rows,int draws){
 for(int row=0;row<rows+3;row++){blockIdx.x=row;pcg64_raw64(hi,lo,ihi,ilo,active,out,errors,rows,draws,draws);}
}
extern "C" void host_uniform(unsigned long long* hi,unsigned long long* lo,
 const unsigned long long* ihi,const unsigned long long* ilo,const unsigned char* active,
 double* out,int* errors,int rows,int draws){
 for(int row=0;row<rows+3;row++){blockIdx.x=row;pcg64_uniform(hi,lo,ihi,ilo,active,out,errors,rows,draws,draws);}
}
'''
        try:
            cls.library = compile_host_native_library(preamble+random._CUDA_SOURCE+wrapper, Path(cls.temp.name), timeout=30)
            cls.library.host_integers.argtypes = [ctypes.c_void_p]*10+[ctypes.c_int]*2
            cls.library.host_integers.restype = None
            for name in ('host_raw','host_uniform'):
                getattr(cls.library,name).argtypes = [ctypes.c_void_p]*7+[ctypes.c_int]*2
                getattr(cls.library,name).restype = None
        except BaseException:
            cls.temp.cleanup();raise
    @classmethod
    def tearDownClass(cls):
        if os.name == 'nt':
            import _ctypes
            _ctypes.FreeLibrary(cls.library._handle)
        cls.library = None
        cls.temp.cleanup()

    def test_kernel_exact_cold_hot_caches_wide_seeds_masks_and_edges(self):
        highs = [1,2,3,7,800,2**30+1,1431655766,2**31-1]*8
        states = []
        for i in range(len(highs)):
            state = copy.deepcopy(np.random.PCG64([0,1,42,(1<<300)+7][i%4]).state)
            if i%3:
                state['has_uint32'] = 1
                state['uinteger'] = [0,1,0x12345678,0xFFFFFFFF][i%4]
            states.append(state)
        active = [i%5 != 0 for i in range(len(highs))]
        bank = HostBank(self.library,states)
        expected = [generator(s) for s in states]
        actual, errors = bank.integers(highs,active)
        choices = [int(g.integers(high)) if live else 0 for g, high, live in zip(expected,highs,active)]
        np.testing.assert_array_equal(actual,choices);np.testing.assert_array_equal(errors,0)
        self.assertEqual(bank.states(),tuple(g.bit_generator.state for g in expected))

    def test_rejections_are_exact_or_fail_transactionally_without_fake_draws(self):
        cold = forced_zero_state()
        hot = copy.deepcopy(np.random.PCG64(12).state);hot.update(has_uint32=1,uinteger=0)
        for state, retries in ((cold,1),(cold,2),(hot,1)):
            bank = HostBank(self.library,[state]);before = bank.states()
            out, errors = bank.integers([3],[True],retries)
            self.assertEqual(out.tolist(),[-1]);self.assertEqual(errors.tolist(),[random.PCG64_ERR_REJECTION])
            self.assertEqual(bank.states(),before)
            expected = generator(state)
            out, errors = bank.integers([3],[True])
            self.assertEqual(out.tolist(),[int(expected.integers(3))]);self.assertEqual(errors.tolist(),[0])
            self.assertEqual(bank.states(),(expected.bit_generator.state,))
        bank = HostBank(self.library,[cold]);expected = generator(cold)
        out, errors = bank.integers([3],[True],3)
        self.assertEqual(out.tolist(),[int(expected.integers(3))]);self.assertEqual(errors.tolist(),[0])
        self.assertEqual(bank.states(),(expected.bit_generator.state,))

    def test_invalid_bounds_inactive_and_singleton_preserve_all_six_fields(self):
        high = [0,-1,2**31,2**63-1,-2**63,1,0]
        states = [forced_zero_state() for _ in high]
        bank = HostBank(self.library,states);before = bank.states()
        out, errors = bank.integers(high,[True]*6+[False])
        self.assertEqual(errors.tolist(),[6]*5+[0,0])
        self.assertEqual(out.tolist(),[-1]*5+[0,0]);self.assertEqual(bank.states(),before)
        # A malformed device state also fails before committing a random draw.
        bank.fields[3][0] &= np.uint64(~np.uint64(1));bank.fields[4][1] = 2
        before = [a.copy() for a in bank.fields]
        out, errors = bank.integers([3]*len(high),[True,True]+[False]*5)
        self.assertEqual(errors.tolist(),[5,5]+[0]*5);self.assertEqual(out.tolist(),[-1,-1]+[0]*5)
        for a,b in zip(bank.fields,before):np.testing.assert_array_equal(a,b)

    def test_continuous_calls_interleaved_with_original_raw_and_uniform(self):
        states = [np.random.PCG64(seed).state for seed in (0,5,99,(1<<512)+27)]
        bank = HostBank(self.library,states);expected = [generator(s)for s in states]
        for step in range(60):
            active = [step%4 != row for row in range(4)]
            if step%7 in (1,2):
                name = 'raw' if step%7 == 1 else 'uniform'
                actual = bank.primitive(name,active,2)
                for row,g in enumerate(expected):
                    values = (g.bit_generator.random_raw(2)if name=='raw'else g.random(2)) if active[row]else np.zeros(2)
                    np.testing.assert_array_equal(actual[row],values)
            else:
                highs = [1,3,800,1431655766] if step%2 else [7,2**31-1,2,2**30+1]
                actual,errors = bank.integers(highs,active)
                np.testing.assert_array_equal(errors,0)
                np.testing.assert_array_equal(actual,[int(g.integers(high))if live else 0 for g,high,live in zip(expected,highs,active)])
            self.assertEqual(bank.states(),tuple(g.bit_generator.state for g in expected))


class BoundedIntegerApiTests(unittest.TestCase):
    def test_explicit_cpu_oracle_matches_actual_kernel_contract_and_rejects_bad_shapes(self):
        import torch
        states = [forced_zero_state(),np.random.PCG64(7).state,np.random.PCG64(8).state]
        bank = random.DevicePCG64(states,device='cpu',allow_cpu_reference=True,max_draws=4,max_choices=4)
        self.addCleanup(bank.close)
        active = torch.tensor([True,True,False])
        before = bank.export_numpy_states()
        failed = bank.integers_device(torch.tensor([3,0,-1]),active,max_retries=2)
        self.assertEqual(failed.indices.tolist(),[-1,-1,0]);self.assertEqual(failed.errors.tolist(),[7,6,0])
        self.assertEqual(bank.export_numpy_states(),before)
        highs = torch.tensor([3,800,2**31-1])
        actual = bank.integers_device(highs,active)
        expected = [generator(s)for s in states]
        self.assertEqual(actual.indices.tolist(),[int(g.integers(h))if live else 0 for g,h,live in zip(expected,[3,800,2**31-1],[True,True,False])])
        self.assertEqual(bank.export_numpy_states(),tuple(g.bit_generator.state for g in expected))
        before = bank.export_numpy_states()
        for retry in (0,-1,1025,2.,None):
            with self.assertRaises(ValueError):bank.integers_device(highs,active,max_retries=retry)
        with self.assertRaises(TypeError):bank.integers_device(highs,active,max_retries=True)
        for bad in (highs.float(),highs.bool(),highs[:2],torch.ones((3,2),dtype=torch.int64)[:,0]):
            with self.assertRaises(ValueError):bank.integers_device(bad,active)
        self.assertEqual(bank.export_numpy_states(),before)

    def test_cuda_branch_has_no_host_value_reads_or_distribution_substitution(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(random.DevicePCG64.integers_device)))
        branch = next(n for n in ast.walk(tree)if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)
            and isinstance(n.test.left,ast.Attribute)and n.test.left.attr=='type')
        calls = [n for statement in branch.orelse for n in ast.walk(statement)if isinstance(n,ast.Call)]
        self.assertTrue(any(isinstance(n.func,ast.Attribute)and n.func.attr=='_launch'for n in calls))
        self.assertFalse(any(isinstance(n.func,ast.Attribute)and n.func.attr in
            ('cpu','numpy','item','tolist','__bool__','categorical','uniform')for n in calls))
        self.assertEqual(random._KERNEL_SPECS['pcg64_bounded_integers'],
            ('uint64*',)*4+('uint32*',)*2+('int64*','uint8*','int64*','int32*','int32','int32'))
        self.assertFalse(random.DevicePCG64.supports_full_duel_rules)


if __name__=='__main__':unittest.main()
