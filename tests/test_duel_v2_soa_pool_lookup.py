"""Execute the emitted pool locator over every live ID and invalid boundaries."""
import ctypes
import shutil
import sys
import tempfile
import unittest

import numpy as np

from app.modules.card_game.rl.batched_duel.vm_soa_views import _emit_runtime_head


def locator_source(bases, capacities):
    head=_emit_runtime_head()
    start=head.index('__device__ inline int soa_locate(')
    end=head.index('\n__device__',start+1)
    tables=('const int SOA_POOL_COUNT='+str(len(bases))+';\n'
            '__device__ const int SOA_POOL_BASE[]={'+','.join(map(str,bases))+'};\n'
            '__device__ const int SOA_POOL_CAP[]={'+','.join(map(str,capacities))+'};\n')
    return tables+head[start:end]


def cases():
    from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
    schema=compile_execution_schema()
    yield [p.base for p in schema.pools],[p.capacity for p in schema.pools],schema.total_objects
    # Noncontiguous ranges make invalid holes visible, in addition to endpoints.
    yield [0,10,20],[3,3,4],24


def inputs_and_expected(bases,capacities,total):
    refs=np.concatenate((np.arange(-1,total+2,dtype=np.int32),np.array([-2**31,2**31-1],np.int32)))
    expected=np.tile(np.array([0,-1,-1],np.int32),(len(refs),1))
    for pool,(base,capacity) in enumerate(zip(bases,capacities)):
        take=(refs>base)&(refs<=base+capacity)
        expected[take,0]=1;expected[take,1]=pool;expected[take,2]=refs[take]-base-1
    return refs,expected


BODY='''
 int p=-1,s=-1,ok=soa_locate(refs[i],&p,&s);
 output[i*3]=ok;output[i*3+1]=p;output[i*3+2]=s;
'''


class PoolLookupTests(unittest.TestCase):
    @unittest.skipIf(sys.platform=='win32' or not any(shutil.which(c) for c in ('clang++','g++','c++')),
                     'POSIX C++ reference compiler required')
    def test_compiled_lookup_exhaustive_ids_and_invalid_holes(self):
        from app.modules.card_game.rl.batched_duel.native_array_cpu import compile_host_native_library
        for bases,capacities,total in cases():
            refs,expected=inputs_and_expected(bases,capacities,total)
            source=('#define __device__\n'+locator_source(bases,capacities)+
                    '\nextern "C" void probe(const int* refs,int* output,int n){for(int i=0;i<n;i++){'+BODY+'}}')
            with tempfile.TemporaryDirectory() as directory:
                library=compile_host_native_library(source,directory,timeout=30)
                fn=library.probe
                fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int];fn.restype=None
                actual=np.empty_like(expected)
                fn(refs.ctypes.data,actual.ctypes.data,len(refs))
                np.testing.assert_array_equal(actual,expected)

    def test_actual_cuda_lookup_exhaustive_ids_and_invalid_holes(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.cuda_runtime import CudaModule
        for bases,capacities,total in cases():
            refs,expected=inputs_and_expected(bases,capacities,total)
            source=locator_source(bases,capacities)+'''\nextern "C" __global__ void probe(const int* refs,int* output,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;'''+BODY+'}'
            values=torch.tensor(refs,device='cuda');actual=torch.empty((len(refs),3),dtype=torch.int32,device='cuda')
            with CudaModule(source,{'probe':['int32*','int32*','int32']}) as module:
                module.launch('probe',[values,actual,len(refs)],grid=(len(refs)+127)//128,block=128).synchronize()
                np.testing.assert_array_equal(actual.cpu().numpy(),expected)
