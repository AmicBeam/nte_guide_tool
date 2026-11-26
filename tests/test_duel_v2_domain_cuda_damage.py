"""Direct-kernel differential; skipped CUDA never implies device acceptance."""
import unittest
import ctypes
import shutil
import subprocess
import tempfile
from pathlib import Path
from dataclasses import fields
import torch
from app.modules.card_game.rl.batched_duel.domain_soa_cuda_damage import (
    DomainCudaDamage, SOURCE, supports_full_duel_rules,
)
from app.modules.card_game.rl.batched_duel.domain_soa_rules import damage_fields


class DirectDamageTest(unittest.TestCase):
    def test_scope_is_explicit(self):
        self.assertFalse(supports_full_duel_rules)
        self.assertNotIn('vm_',SOURCE)
        self.assertNotIn('opcode',SOURCE)

    def test_compiled_arithmetic_matches_tensor_reference(self):
        compiler=shutil.which('clang++') or shutil.which('g++')
        if not compiler:self.skipTest('C++ compiler unavailable; no compiled oracle')
        import numpy as np
        rng=np.random.default_rng(741)
        args=[]
        for i in range(9):
            if i in (3,4,5,8):value=rng.integers(0,2,400).astype(np.bool_)
            else:value=rng.integers(-2,100,400,dtype=np.int32)
            args.append(torch.from_numpy(value))
        args[0][:4]=torch.tensor([2147483647,1,0,7],dtype=torch.int32)
        args[2][:4]=2147483647
        expected=damage_fields(*args)
        arrays=[v.numpy()for v in args]
        output=[np.empty_like(getattr(expected,f.name).numpy())for f in fields(expected)]
        parameters='const int* h,const int* s,const int* a,const unsigned char* b,const unsigned char* i,const unsigned char* l,const int* f,const int* c,const unsigned char* on,int* oh,int* os,int* ab,int* lo,long long* hl,unsigned char* dead,int* err,int n'
        code='#define __global__\nstruct Dim {int x;}; static Dim blockIdx,blockDim,threadIdx;\n'+SOURCE
        code+='\nextern "C" void run('+parameters+'){blockDim.x=128;for(int row=0;row<n;row++){blockIdx.x=row/128;threadIdx.x=row%128;domain_damage(h,s,a,b,i,l,f,c,on,oh,os,ab,lo,hl,dead,err,n);}}'
        with tempfile.TemporaryDirectory()as directory:
            source=Path(directory)/'damage.cpp';library=Path(directory)/'damage.so'
            source.write_text(code)
            subprocess.run([compiler,'-shared','-fPIC','-O2',str(source),'-o',str(library)],check=True,capture_output=True)
            function=ctypes.CDLL(str(library)).run
            function.argtypes=[ctypes.c_void_p]*16+[ctypes.c_int];function.restype=None
            function(*(v.ctypes.data for v in arrays+output),400)
        for value,f in zip(output,fields(expected)):
            np.testing.assert_array_equal(value,getattr(expected,f.name).numpy(),err_msg=f.name)

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_mixed_rows_and_captured_replay_match(self):
        generator=torch.Generator().manual_seed(53)
        n=400
        def ints(low,high):
            return torch.randint(low,high,(n,),generator=generator,dtype=torch.int32).cuda()
        def bits():
            return torch.randint(0,2,(n,),generator=generator,dtype=torch.int32).bool().cuda()
        args=[ints(-1,50),ints(-1,50),ints(-5,100),bits(),bits(),bits(),
              ints(-1,4),ints(-2,10),bits()]
        args[0][:4]=torch.tensor([2147483647,1,0,7],dtype=torch.int32,device='cuda')
        args[2][:4]=2147483647
        before=[v.clone()for v in args]
        kernel=DomainCudaDamage()
        try:
            batch=kernel.prepare(*args)
            expected=damage_fields(*args)
            kernel.launch(batch).synchronize()
            for f in fields(expected):
                self.assertTrue(torch.equal(getattr(batch.result,f.name),getattr(expected,f.name)),f.name)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):kernel.launch(batch).synchronize()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):kernel.launch(batch)
            for _ in range(3):graph.replay()
            torch.cuda.synchronize()
            for f in fields(expected):
                self.assertTrue(torch.equal(getattr(batch.result,f.name),getattr(expected,f.name)),f.name)
            self.assertTrue(all(torch.equal(a,b)for a,b in zip(args,before)))
        finally:kernel.close()
