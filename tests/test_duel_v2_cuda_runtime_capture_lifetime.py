"""Borrowed conditional stream lifetime contracts, without CUDA execution."""
from contextlib import contextmanager, nullcontext
import gc
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from app.modules.card_game.rl.batched_duel.cuda_runtime import CudaModule, KernelSpec


class CaptureLifetimeTests(unittest.TestCase):
    @contextmanager
    def fixture(self,capturing):
        calls=[];state=SimpleNamespace(capturing=capturing,sync_error=False)
        device=SimpleNamespace(type='cuda',index=0)
        class Tensor:
            dtype='float32'
            def __init__(self):self.device=device
            def is_contiguous(self):return True
            def data_ptr(self):return id(self)
            def record_stream(self,stream):calls.append(('record',id(self),stream.cuda_stream))
        class Stream:
            def __init__(self,handle):self.cuda_stream=handle
            def wait_stream(self,other):calls.append(('wait',self.cuda_stream,other.cuda_stream))
            def synchronize(self):calls.append(('stream_sync',self.cuda_stream))
        class Event:
            def record(self,stream):calls.append(('event',stream.cuda_stream))
        current=Stream(99);owned=Stream(11)
        def synchronize(target):
            calls.append(('device_sync',target))
            if state.sync_error:raise RuntimeError('pending device error')
        def launch(*args):calls.append(('launch',args[8].value));return 0
        def unload(module):calls.append(('unload',module));return 0
        fake=SimpleNamespace(Tensor=Tensor,cuda=SimpleNamespace(device=lambda target:nullcontext(),
            current_stream=lambda **kw:current,is_current_stream_capturing=lambda:state.capturing,
            synchronize=synchronize,Event=Event),int32='int32',uint32='uint32',
            int64='int64',uint64='uint64',uint8='uint8',float32='float32',float64='float64')
        module=object.__new__(CudaModule)
        module._device=device;module._device_idx=0;module._stream=owned;module._closed=False
        module._module=17;module._kernels={'borrow':23}
        module._kernel_specs={'borrow':KernelSpec.from_spec(['float32*','float32*'])}
        module._capture_tensor_refs={}
        module._drv=SimpleNamespace(cuLaunchKernel=launch,cuModuleUnload=unload)
        with patch.dict(sys.modules,{'torch':fake}):
            try:yield module,Tensor,calls,state
            finally:
                state.capturing=False;state.sync_error=False;module.close()

    def test_actual_capture_retains_all_borrowed_arguments_without_record_or_event(self):
        with self.fixture(True)as(module,Tensor,calls,state):
            a,b=Tensor(),Tensor();refs=(weakref.ref(a),weakref.ref(b))
            result=module.launch('borrow',[a,b],grid=1,block=1,capture_safe=True)
            self.assertIsNone(result)
            self.assertEqual(calls,[('launch',99)])
            self.assertEqual(set(module._capture_tensor_refs),{id(a),id(b)})
            # Repeat aliases do not duplicate ownership or copy allocations.
            module.launch('borrow',[a,a],grid=1,block=1,capture_safe=True)
            self.assertEqual(len(module._capture_tensor_refs),2)
            del a,b;gc.collect()
            self.assertTrue(all(ref()is not None for ref in refs))
            state.capturing=False;module.close();gc.collect()
            self.assertTrue(all(ref()is None for ref in refs))
            names=[c[0]for c in calls]
            self.assertLess(names.index('device_sync'),names.index('unload'))
            self.assertEqual(module._capture_tensor_refs,{})
            before=list(calls);module.close();self.assertEqual(calls,before)

    def test_capture_safe_but_eager_keeps_record_stream_without_retaining(self):
        with self.fixture(False)as(module,Tensor,calls,state):
            a,b=Tensor(),Tensor();refs=(weakref.ref(a),weakref.ref(b))
            result=module.launch('borrow',[a,b],grid=1,block=1,capture_safe=True)
            self.assertIsNone(result)
            self.assertEqual(calls,[('record',id(a),99),('record',id(b),99),('launch',99)])
            self.assertEqual(module._capture_tensor_refs,{})
            del a,b;gc.collect();self.assertTrue(all(ref()is None for ref in refs))
            module.close();self.assertNotIn('device_sync',[c[0]for c in calls])

    def test_ordinary_launch_preserves_wait_record_event_even_if_caller_capturing(self):
        for capturing in (False,True):
            with self.subTest(capturing=capturing),self.fixture(capturing)as(module,Tensor,calls,state):
                a,b=Tensor(),Tensor()
                self.assertIsNotNone(module.launch('borrow',[a,b],grid=1,block=1))
                self.assertEqual(calls,[('wait',11,99),('record',id(a),11),
                    ('record',id(b),11),('launch',11),('event',11)])
                self.assertEqual(module._capture_tensor_refs,{})

    def test_close_during_capture_or_failed_sync_keeps_references_and_module_for_retry(self):
        with self.fixture(True)as(module,Tensor,calls,state):
            a,b=Tensor(),Tensor()
            module.launch('borrow',[a,b],grid=1,block=1,capture_safe=True)
            refs=(weakref.ref(a),weakref.ref(b));del a,b;gc.collect()
            with self.assertRaisesRegex(RuntimeError,'during CUDA capture'):module.close()
            self.assertFalse(module.closed);self.assertEqual(module._module,17)
            self.assertTrue(all(ref()is not None for ref in refs))
            state.capturing=False;state.sync_error=True
            with self.assertRaisesRegex(RuntimeError,'pending device error'):module.close()
            self.assertFalse(module.closed);self.assertEqual(module._module,17)
            self.assertTrue(all(ref()is not None for ref in refs))
            self.assertNotIn('unload',[c[0]for c in calls])
            state.sync_error=False;module.close();gc.collect()
            self.assertTrue(all(ref()is None for ref in refs))

    def test_capture_safe_requires_explicit_bool_before_any_lifetime_change(self):
        with self.fixture(True)as(module,Tensor,calls,state):
            a,b=Tensor(),Tensor()
            for invalid in (1,None,'yes'):
                with self.assertRaises(TypeError):module.launch('borrow',[a,b],grid=1,block=1,capture_safe=invalid)
            self.assertEqual(calls,[]);self.assertEqual(module._capture_tensor_refs,{})


if __name__=='__main__':unittest.main()
