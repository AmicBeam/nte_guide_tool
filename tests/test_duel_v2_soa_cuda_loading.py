"""Machine-code loading and static capture paths; real CUDA is gated separately."""
import ctypes
import re
import shutil
from types import SimpleNamespace
import unittest

from app.modules.card_game.rl.batched_duel.cuda_runtime import CudaModule, CudaCompilationError, validate_cubin_profile
from app.modules.card_game.rl.batched_duel.native_soa_source import separate_soa_calls


class SoACallBoundaryTests(unittest.TestCase):
    def test_boundaries_change_only_attributes_not_rule_bodies(self):
        from app.modules.card_game.rl.batched_duel.native_array_codegen import emit_native_array_source
        image,_=scalar_fixture()
        source=emit_native_array_source(image)
        result=separate_soa_calls(source)
        self.assertTrue(result.replace('__noinline__','inline')==source,'Only call attributes may change')
        for name in ('native_pc_shard_0','native_dispatch_pc','native_step','execute'):
            self.assertTrue(re.search(r'__device__ __noinline__ void '+name+r'\(',result),name)
        for name in ('native_opcode','native_handlers','call','new_frame'):
            self.assertTrue(re.search(r'__device__ inline (?:void|int) '+name+r'\(',result),name)
        for name in ('stdlib_invoke','stdlib_resume','stdlib_call','object_attribute'):
            self.assertTrue(re.search(r'__device__ __noinline__ (?:void|int) '+name+r'\(',result),name)



class CubinBinaryTests(unittest.TestCase):
    def test_conflicting_assembler_options_rejected_before_compilation(self):
        with self.assertRaisesRegex(ValueError,'conflicts'):
            validate_cubin_profile('cubin',dict(fast_compile='mid',jit_optimization_level=2))
        for binary,fast,level in (('ptx','mid',2),('cubin','min',None),('cubin','max',0),('cubin','0',2)):
            validate_cubin_profile(binary,dict(fast_compile=fast,jit_optimization_level=level))

    def test_known_bad_mid_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'scalar comparison parity'):
            validate_cubin_profile('cubin',dict(fast_compile='mid',jit_optimization_level=None))

    def module(self,blob):
        calls=[]
        class NV:
            def nvrtcCreateProgram(self,out,*args):
                ctypes.cast(out,ctypes.POINTER(ctypes.c_void_p))[0]=1
                return 0
            def nvrtcCompileProgram(self,*args):return 0
            def nvrtcGetCUBINSize(self,prog,out):
                ctypes.cast(out,ctypes.POINTER(ctypes.c_size_t))[0]=len(blob)
                return 0
            def nvrtcGetCUBIN(self,prog,out):
                calls.append('cubin');ctypes.memmove(out,blob,len(blob));return 0
            def nvrtcDestroyProgram(self,*args):calls.append('destroy');return 0
        module=SimpleNamespace(_nv=NV())
        module._compile_source_to_binary=lambda source,flags,kind:CudaModule._compile_source_to_binary(module,source,flags,kind)
        return module,calls

    def test_binary_embedded_zero_bytes_are_preserved(self):
        module,calls=self.module(b'\x7fELF\x00\x01\x00payload')
        self.assertEqual(CudaModule._compile_source_to_cubin(module,'source',['--gpu-architecture=sm_120']),b'\x7fELF\x00\x01\x00payload')
        self.assertEqual(calls,['cubin','destroy'])

    def test_empty_binary_rejected_and_program_destroyed(self):
        module,calls=self.module(b'')
        with self.assertRaisesRegex(CudaCompilationError,'empty cubin'):
            CudaModule._compile_source_to_cubin(module,'source',[])
        self.assertEqual(calls,['destroy'])

    def test_missing_cubin_support_does_not_fall_back_to_ptx(self):
        with self.assertRaisesRegex(CudaCompilationError,'no PTX fallback'):
            CudaModule._compile_source_to_cubin(SimpleNamespace(_nv=object()),'source',[])


class DeviceStackBoundsTests(unittest.TestCase):
    def test_undersized_or_unbounded_stack_rejected_before_cuda_setup(self):
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import CudaSoARules
        with patch('torch.cuda.is_available',return_value=True):
            for invalid in (True,0,65536,2**21):
                with self.subTest(stack=invalid),self.assertRaisesRegex(ValueError,'at least 128 KiB'):
                    CudaSoARules(None,None,device='cuda:0',stack_bytes=invalid)


class StaticPacketPathTests(unittest.TestCase):
    def test_prepared_path_is_reused_during_capture(self):
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import CudaSoARules
        uploads=[];capturing=[False]
        torch=SimpleNamespace(int32='i32',cuda=SimpleNamespace(is_current_stream_capturing=lambda:capturing[0]),
            tensor=lambda data,**kwargs:uploads.append(tuple(data)) or object())
        module=SimpleNamespace(torch=torch,device='cuda',_result_paths={})
        first=CudaSoARules.prepare_result_path(module,(1,0,1))
        capturing[0]=True
        self.assertIs(CudaSoARules.prepare_result_path(module,[1,0,1]),first)
        self.assertEqual(uploads,[(1,0,1)])
        with self.assertRaisesRegex(RuntimeError,'before CUDA graph capture'):
            CudaSoARules.prepare_result_path(module,(1,1,1))
        self.assertEqual(uploads,[(1,0,1)])



# Capture these two functions into the small, version-independent test image.
def scalar_compare_values(a,b):
    return a==b,a!=b,a==a,b==b,a<b,b<a


def scalar_literal_branch(side):
    return ('A' if side=='a' else 'B'),(0 if side=='a' else 5)


def scalar_fixture():
    from pathlib import Path
    from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram
    from app.modules.card_game.rl.batched_duel.vm_image import lower_program
    from app.modules.card_game.rl.batched_duel.record_soa import compile_schema
    path=Path(__file__).parent/'fixtures'/'duel_v2_scalar_compare_program.json'
    image=lower_program(FrozenRuleProgram.from_json(path.read_text(encoding='utf8')))
    schema=compile_schema({'Game':{'kind':'record','capacity':8,'fields':{'hp':'int'}}},
                          max_strings=64,max_string_bytes=2048)
    return image,schema


def check_scalar_results(test,backend,schema):
    cases=[('scalar_compare_values',[['a','b'],['b','a']],
            [scalar_compare_values('a','b'),scalar_compare_values('b','a')]),
           ('scalar_compare_values',[[0,1],[1,2]],
            [scalar_compare_values(0,1),scalar_compare_values(1,2)]),
           ('scalar_literal_branch',[['a'],['b']],[('A',0),('B',5)])]
    for entry,args,expected in cases:
        with test.subTest(entry=entry,args=args):
            result=backend.call(entry,schema.allocate(2),args,state_positions=())
            test.assertEqual(result.decode(),expected)


@unittest.skipUnless(any(shutil.which(name) for name in ('clang++','g++','c++')),
                     'Host C++ compiler required for the CPU reference')
class ScalarComparisonCPU(unittest.TestCase):
    def test_mixed_mutable_and_frozen_operands(self):
        from app.modules.card_game.rl.batched_duel.native_soa_cpu import NativeSoACPU
        image,schema=scalar_fixture()
        with NativeSoACPU(image,schema,max_nodes=4096,max_edges=8192,max_payload_bytes=65536) as backend:
            check_scalar_results(self,backend,schema)


class ScalarComparisonCUDA(unittest.TestCase):
    def test_mixed_mutable_and_frozen_operands(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.native_soa_cuda_query import NativeSoACUDAQuery
        image,schema=scalar_fixture()
        backend=NativeSoACUDAQuery(image,schema,max_nodes=4096,max_edges=8192,max_payload_bytes=65536,seconds=30)
        try:
            check_scalar_results(self,backend,schema)
        finally:
            backend.close()

if __name__=='__main__':unittest.main()
