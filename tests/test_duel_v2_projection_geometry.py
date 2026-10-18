"""Small rawparse launch geometry; CUDA body and other paths stay independent."""
import ctypes
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from app.modules.card_game.rl.batched_duel.gpu_rule_projection import (
    GraphProjectionKernel, _masked_projection_source, _projection_call_boundaries,
)
from app.modules.card_game.rl.batched_duel.gpu_seven_projection import emit_seven_projection_source
from app.modules.card_game.rl.batched_duel.native_array_cpu import (
    HOST_CUDA_SHIMS, _cptr, compile_host_native_library,
)
from app.modules.card_game.rl.batched_duel.schema import GraphLayout


class ReachedLaunch(Exception):
    pass


class ProjectionGeometryTests(unittest.TestCase):
    def make_call(self, rows):
        import torch
        buffers=dict(nodes=torch.zeros((1,4),dtype=torch.int32),
                     edges=torch.zeros((1,2),dtype=torch.int32),payload=torch.zeros(1,dtype=torch.uint8))
        projector=GraphProjectionKernel.__new__(GraphProjectionKernel)
        projector.closed=False;projector.torch=torch;projector.device=torch.device('cpu')
        projector.runtime='seven';projector.max_candidates=800
        projector._intern_names=();projector._names=buffers['payload']
        projector._off=projector._len=torch.zeros(1,dtype=torch.int32)
        projector.module=SimpleNamespace(launch=Mock(side_effect=ReachedLaunch))
        arena=SimpleNamespace(rows=rows,layout=GraphLayout(rows,1,1,1),
            nodes=torch.zeros((rows,1,4),dtype=torch.int32),edges=torch.zeros((rows,1,2),dtype=torch.int32),
            payload=torch.zeros((rows,1),dtype=torch.uint8),counts=torch.zeros((rows,3),dtype=torch.int32))
        call=SimpleNamespace(arena=arena,executor=SimpleNamespace(_buffers=buffers,image=SimpleNamespace(**buffers)),
            outputs=torch.zeros(rows,dtype=torch.int32),statuses=torch.zeros((rows,4),dtype=torch.int32),check=lambda:None)
        return projector,call,torch.ones(rows,dtype=torch.uint8)

    def test_actual_rawparse_launch_small_boundary(self):
        for rows in (1,2,8,32,33,129):
            projector,call,active=self.make_call(rows)
            with self.assertRaises(ReachedLaunch):projector.project_tokens_device(call,active=active)
            args,kw=projector.module.launch.call_args
            block=1 if rows<=32 else 128
            self.assertEqual(args[0],'project_public_decision_device')
            self.assertEqual(args[1][-6],rows)
            self.assertEqual(kw,dict(grid=(rows+block-1)//block,block=block,capture_safe=True))

    def test_dense_host_and_observation_paths_keep128(self):
        for rows in (2,32,33):
            for path in ('device_dense','host','observation'):
                projector,call,active=self.make_call(rows)
                with self.assertRaises(ReachedLaunch):
                    if path=='observation':projector.project_observation(call,call.outputs,active=active)
                    else:projector.project_decision(call,active=None if path=='host'else active)
                _,kw=projector.module.launch.call_args
                self.assertEqual(kw['block'],128)
                self.assertEqual(kw['grid'],(rows+127)//128)

    def test_actual_cuda_body_row_coverage_errors_and_bounds(self):
        source=_projection_call_boundaries(_masked_projection_source(emit_seven_projection_source()))
        self.assertFalse(any(x in source for x in ('__shared__','__syncthreads','__shfl','__ballot','warpSize')))
        params=re.search(r'extern "C" __global__ void project_public_decision_device\((.*?)\)\s*\{',source,re.S)[1]
        parts=[p.strip()for p in params.split(',')]
        names=[re.search(r'(\w+)\s*$',p)[1]for p in parts]
        launcher='extern "C" void host_geometry('+params+',int threads){blockDim.x=threads;gridDim.x=(rows+threads-1)/threads;for(unsigned b=0;b<gridDim.x;b++){blockIdx.x=b;for(unsigned t=0;t<blockDim.x;t++){threadIdx.x=t;project_public_decision_device('+','.join(names)+');}}}'
        with tempfile.TemporaryDirectory(prefix='projection_geometry_')as directory:
            lib=compile_host_native_library(HOST_CUDA_SHIMS+source+launcher,Path(directory),timeout=30)
            fn=lib.host_geometry;fn.argtypes=[ctypes.c_void_p if '*'in p else ctypes.c_int for p in parts]+[ctypes.c_int];fn.restype=None
            for rows in (1,2,8,32,35,129):
                baseline=None
                for threads in (128,1,4):
                    arrays={n:np.zeros(rows*16+2,np.float32 if 'float*'in p else(np.uint8 if 'char*'in p else np.int32))for n,p in zip(names,parts)if '*'in p}
                    arrays['outputs'].fill(999);arrays['status'].fill(-777);arrays['action_roots'].fill(-777)
                    arrays['active'][:rows]=np.asarray([i%3!=1 for i in range(rows)],np.uint8)
                    scalar=dict(rows=rows,maxn=1,maxe=1,maxp=1,max_candidates=1,cnc=0,cec=0,cpc=0,n_names=0)
                    fn(*[_cptr(arrays[n])if '*'in p else scalar[n]for n,p in zip(names,parts)],threads)
                    self.assertEqual(arrays['status'][rows:].tolist(),[-777]*(len(arrays['status'])-rows))
                    for row in range(rows):self.assertEqual(arrays['status'][row],10 if row%3!=1 else 0)
                    if baseline is None:baseline=arrays
                    else:
                        for name in arrays:np.testing.assert_array_equal(arrays[name],baseline[name])


if __name__=='__main__':unittest.main()
