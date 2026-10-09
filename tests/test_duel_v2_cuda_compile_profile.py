"""Explicit CUDA short-test flags and driver option ABI; defaults stay unchanged."""
import ctypes
import os
import unittest
from unittest.mock import patch
from app.modules.card_game.rl.batched_duel.cuda_runtime import (
    cuda_compilation_profile, _load_driver_module, CudaDriverError, CudaModule,
)

class CompileProfileTest(unittest.TestCase):
    def test_defaults_and_all_explicit_levels(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(cuda_compilation_profile(),{'fast_compile':'0','jit_optimization_level':None})
        for level in range(5):
            with patch.dict(os.environ,{'NTE_CUDA_JIT_OPT_LEVEL':str(level),'NTE_CUDA_FAST_COMPILE':'max'},clear=True):
                self.assertEqual(cuda_compilation_profile(),{'fast_compile':'max','jit_optimization_level':level})
        for key in ('NTE_CUDA_JIT_OPT_LEVEL','NTE_CUDA_FAST_COMPILE'):
            with patch.dict(os.environ,{key:'unsupported'},clear=True):
                with self.assertRaises(ValueError):cuda_compilation_profile()

    def test_driver_default_and_cast_integer_optimization_abi(self):
        calls=[]
        class Driver:
            def cuModuleLoadData(self,module,ptx):
                calls.append(('default',ptx));return 0
            def cuModuleLoadDataEx(self,module,ptx,count,options,values):
                calls.append(('explicit',count,options[0],values[0]));return 0
        module=ctypes.c_void_p()
        _load_driver_module(Driver(),module,b'ptx',None)
        for level in (0,2,4):_load_driver_module(Driver(),module,b'ptx',level)
        self.assertEqual(calls,[('default',b'ptx'),('explicit',1,7,None),('explicit',1,7,2),('explicit',1,7,4)])

    def test_requested_profile_never_silently_falls_back(self):
        class Driver:
            def cuModuleLoadData(self,*args):raise AssertionError('Unexpected default fallback')
        with self.assertRaises(CudaDriverError):_load_driver_module(Driver(),ctypes.c_void_p(),b'ptx',0)


class StackLimitTest(unittest.TestCase):
    def module(self,initial,satisfied=True):
        from types import SimpleNamespace
        calls=[]
        class Driver:
            limit=initial
        driver=Driver()
        def get_limit(pointer,which):
            self.assertEqual(which,0)
            ctypes.cast(pointer,ctypes.POINTER(ctypes.c_size_t))[0]=driver.limit
            return 0
        def set_limit(which,value):
            calls.append(value)
            if satisfied:driver.limit=value
            return 0
        driver.cuCtxGetLimit=get_limit
        driver.cuCtxSetLimit=set_limit
        return SimpleNamespace(_drv=driver,_compilation_profile={}),calls
    def test_raise_small_stack_and_preserve_larger_existing_limit(self):
        for initial,expected_calls in ((1024,[65536]),(131072,[])):
            module,calls=self.module(initial)
            actual=CudaModule.ensure_stack.__wrapped__(module)
            self.assertEqual(actual,max(initial,65536))
            self.assertEqual(calls,expected_calls)
            self.assertEqual(module._compilation_profile['device_stack_bytes'],actual)
    def test_unsatisfied_or_unbounded_request_is_rejected(self):
        module,_=self.module(1024,satisfied=False)
        with self.assertRaises(CudaDriverError):CudaModule.ensure_stack.__wrapped__(module)
        for invalid in (0,True,1024,2**21):
            with self.assertRaises(ValueError):CudaModule.ensure_stack.__wrapped__(module,invalid)

if __name__=='__main__':unittest.main()
