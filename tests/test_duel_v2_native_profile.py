import unittest
from types import SimpleNamespace
import numpy as np
from app.modules.card_game.rl.batched_duel.native_compile_units import NativeCompileUnit
from app.modules.card_game.rl.batched_duel.native_profile import instrument_runtime,decode_profile

SOURCE='''namespace duel_rule_vm {
template<int OP>
__device__ __noinline__ void native_step(Context* c,int* f,const int* fn,int arg,int value,int target){if(arg)return;}
__device__ __noinline__ void native_dispatch_pc(Context* c,int* f,const int* fn,int pc){if(!native_function_dispatch(c,f,fn,pc))c->error=BAD_PROGRAM;}
}
'''

class NativeProfileTests(unittest.TestCase):
    def test_explicit_runtime_only_no_abi_or_original_body_changes(self):
        unit=NativeCompileUnit('runtime',SOURCE);image=SimpleNamespace(function_ids=('f','g'),opcode_names=('A','B'))
        prof,layout=instrument_runtime(unit,image,helpers=())
        self.assertEqual(unit.source,SOURCE)
        self.assertIn('if(arg)return;',prof.source)
        self.assertEqual(prof.source.count('native_rule_profile::Scope native_profile_scope('),2)
        self.assertIn('~Scope()',prof.source)
        self.assertIn('OP>=0&&OP<2?',prof.source)
        self.assertEqual(layout.slots,8)
        with self.assertRaises(ValueError):instrument_runtime(NativeCompileUnit('rules_000',SOURCE),image)
        with self.assertRaises(ValueError):instrument_runtime(unit,image,rows=True)
        with self.assertRaises(ValueError):instrument_runtime(unit,image,helpers=('unknown',))
        with self.assertRaises(ValueError):instrument_runtime(NativeCompileUnit('runtime',SOURCE.replace('native_step','other')),image,helpers=())
    def test_binding_is_atomic_and_duplicate_registration_is_refused(self):
        import ctypes
        from app.modules.card_game.rl.batched_duel.native_profile import bind_profile_kernels
        from app.modules.card_game.rl.batched_duel.cuda_runtime import CudaDriverError
        for failure in (False, True):
            def lookup(output, module, name):
                if failure and name == b'native_profile_copy':
                    return 500
                ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 123
                return 0
            module = SimpleNamespace(_drv=SimpleNamespace(cuModuleGetFunction=lookup),
                                     _module=ctypes.c_void_p(9), _kernels={}, _kernel_specs={})
            if failure:
                with self.assertRaises(CudaDriverError):
                    bind_profile_kernels(module)
                self.assertFalse(module._kernels)
                self.assertFalse(module._kernel_specs)
            else:
                bind_profile_kernels(module)
                self.assertEqual(set(module._kernels), {'native_profile_reset','native_profile_copy'})
                with self.assertRaises(ValueError):
                    bind_profile_kernels(module)

    def test_per_row_decode_preserves_overlapping_categories(self):
        _,layout=instrument_runtime(NativeCompileUnit('runtime',SOURCE),SimpleNamespace(function_ids=('f',),opcode_names=('A',)),helpers=())
        rows=decode_profile(np.array([10,20,30,40]),np.array([1,2,3,4]),layout)
        self.assertEqual(rows[0],dict(category='function',name='f',nanoseconds=[10,30],calls=[1,3]))
        self.assertEqual(rows[1]['category'],'opcode')
        with self.assertRaises(ValueError):decode_profile([1],[1],layout)

class NativeProfileParityTests(unittest.TestCase):
    def test_instrumented_multifile_real_rules_match_without_abi_changes(self):
        import ctypes,subprocess
        from pathlib import Path
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_duel import native_soa_cpu
        from app.modules.card_game.rl.batched_duel.native_array_cpu import HOST_CUDA_SHIMS,_cxx_compiler
        from app.modules.card_game.rl.batched_duel.native_compile_units import split_native_source
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
        from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
        from tests.test_duel_v2_native_rule_stages import fixture_program,fixture_cases,assert_stage
        image=lower_program(fixture_program());captured={}
        def compile_units(source,directory):
            directory=Path(directory);paths=[]
            for unit in split_native_source(source):
                if unit.name=='runtime':
                    unit,layout=instrument_runtime(unit,image)
                    text=HOST_CUDA_SHIMS+unit.source.replace(HOST_CUDA_SHIMS,'',1)
                    for name in ('begin','end'):
                        text=text.replace('asm volatile("mov.u64 %0, %%globaltimer;" : "=l"('+name+'))',name+'=native_array_clock_ns()')
                    text+='\nextern "C" void native_profile_host_enable(int mask){for(int i=0;i<'+str(layout.slots)+';i++){native_rule_profile::elapsed[i]=0;native_rule_profile::calls[i]=0;}native_rule_profile::enabled=mask;}\n'
                    text+='extern "C" void native_profile_host_counts(unsigned long long* out){for(int i=0;i<'+str(layout.slots)+';i++)out[i]=native_rule_profile::calls[i];}\n'
                    captured['layout']=layout
                else:text=unit.source
                path=directory/(unit.name+'.cpp');path.write_text(text);paths.append(str(path))
            library=directory/'profile.dylib'
            run=subprocess.run([_cxx_compiler(),'-std=c++17','-shared','-fPIC','-O1','-fno-fast-math','-ffp-contract=off',*paths,'-o',str(library)],capture_output=True,text=True,timeout=30)
            if run.returncode:self.fail(run.stderr[-10000:])
            loaded=ctypes.CDLL(str(library));captured['library']=loaded;return loaded
        with patch.object(native_soa_cpu,'compile_host_native_library',compile_units):
            with native_soa_cpu.NativeSoACPU(image,compile_execution_schema(),native_functions=True,native_values=True,
                            limits=VmLimits(steps=100000),max_nodes=16384,max_edges=32768,max_payload_bytes=131072) as backend:
                library=captured['library'];library.native_profile_host_enable.argtypes=[ctypes.c_int]
                for mask in (0,7):
                    library.native_profile_host_enable(mask);assert_stage(backend,fixture_cases())
                    counts=np.zeros(captured['layout'].slots,dtype=np.uint64)
                    library.native_profile_host_counts.argtypes=[ctypes.c_void_p];library.native_profile_host_counts(ctypes.c_void_p(counts.ctypes.data))
                    if mask:self.assertTrue(counts.any())
                    else:self.assertFalse(counts.any())


if __name__=='__main__':unittest.main()
