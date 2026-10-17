"""Separate compilation retains rule bodies and uniquely owns shared symbols."""
import re
import unittest

from app.modules.card_game.rl.batched_duel.native_compile_units import split_native_source


SOURCE = '''namespace duel_rule_vm {
struct Context { int x; };
__device__ int errors[16];
__device__ const int table[]={1,2};
__device__ __noinline__ int shared(Context* c){return c->x;}
template<int OP>
__device__ __noinline__ void native_step(Context* c,int* f,const int* fn,int arg,int value,int target){c->x+=OP;}
__device__ __noinline__ int native_values_1(Context* c,int* f,const int* fn,int pc){return shared(c);}
__device__ __noinline__ void native_function_0(Context* c,int* f,const int* fn,int pc){native_step<2>(c,f,fn,0,0,0);}
__device__ __noinline__ void native_function_1(Context* c,int* f,const int* fn,int pc){const char* s="}";/* { */c->x++;}
}
extern "C" __global__ void vm_soa_execute(int* x){x[0]=1;}
'''


class NativeCompileUnitTests(unittest.TestCase):
    def test_full_projection_is_runtime_only_and_rule_objects_match_execute_prefix(self):
        from app.modules.card_game.rl.batched_duel.native_soa_source import emit_native_soa_source, execute_only_source
        from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
        from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from tests.test_duel_v2_native_rule_stages import fixture_program
        source, _ = emit_native_soa_source(lower_program(fixture_program()), compile_execution_schema(),
            VmLimits(), native_functions=True, native_values=True)
        full = split_native_source(source, functions_per_unit=2)
        execute = split_native_source(execute_only_source(source), functions_per_unit=2)
        self.assertEqual(full[1:], execute[1:])
        for kernel in ('soa_export_projection', 'soa_select_result', 'soa_decode_metadata',
                       'soa_decode_pair', 'soa_decode_float', 'vm_compact'):
            self.assertIn('void '+kernel+'(', full[0].source)
        for unit in full[1:]:
            self.assertNotIn('soa_export_index', unit.source)
            self.assertNotIn('RESULT_INVALID', unit.source)
        with self.assertRaisesRegex(ValueError, 'prefix references'):
            split_native_source(source.replace('int* f=frame(c,id);',
                'int* f=frame(c,id);soa_export_index(0,0);', 1))
        with self.assertRaisesRegex(ValueError, 'Unknown full-module'):
            split_native_source(source+'\n__device__ inline int surprise(){return 1;}\n')
        with self.assertRaisesRegex(ValueError, 'Unrecognized declarations'):
            split_native_source(source+'\n__device__ int extra_projection_data[1];\n')

    def test_unique_definitions_and_template_instantiation(self):
        units = split_native_source(SOURCE, functions_per_unit=1)
        self.assertEqual([u.name for u in units], ['runtime', 'rules_000', 'rules_001', 'values_000'])
        for name in ('shared', 'native_function_0', 'native_function_1', 'native_values_1', 'vm_soa_execute'):
            definitions = sum(len(re.findall(r'\b'+name+r'\([^;{}]*\)\s*\{', u.source)) for u in units)
            self.assertEqual(definitions, 1, name)
        self.assertIn('template __device__ __noinline__ void native_step<2>(', units[0].source)
        for unit in units[1:]:
            self.assertIn('extern __device__ const int table[];', unit.source)
            self.assertIn('extern __device__ int errors[16];', unit.source)
            self.assertNotIn('x[0]=1', unit.source)
        self.assertEqual(units, split_native_source(SOURCE, functions_per_unit=1))
        self.assertEqual(len(units[0].sha256), 64)

    def test_rejects_unknown_scope_and_invalid_chunk_sizes(self):
        for value in (True, 0, 257, 1.5):
            with self.assertRaises(ValueError):
                split_native_source(SOURCE, functions_per_unit=value)
        for value in ('', SOURCE+'native_pc_shard_0', SOURCE.replace('native_function_1(', 'native_function_0(')):
            with self.assertRaises(ValueError):
                split_native_source(value)

class NativeCompileUnitParityTests(unittest.TestCase):
    def test_real_rule_stage_separate_host_compilation(self):
        import ctypes
        from pathlib import Path
        import subprocess
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_duel import native_soa_cpu
        from app.modules.card_game.rl.batched_duel.native_array_cpu import _cxx_compiler
        from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
        from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from tests.test_duel_v2_native_rule_stages import fixture_program, fixture_cases, assert_stage

        def compile_units(source, directory):
            directory = Path(directory)
            paths = []
            for unit in split_native_source(source, functions_per_unit=2):
                path = directory/(unit.name+'.cpp')
                path.write_text(unit.source)
                paths.append(str(path))
            library = directory/'split-host.dylib'
            run = subprocess.run([_cxx_compiler(), '-std=c++17', '-shared', '-fPIC', '-O1',
                '-fno-fast-math', '-ffp-contract=off', '-fno-finite-math-only',
                *paths, '-o', str(library)], capture_output=True, text=True, timeout=30)
            if run.returncode:
                self.fail(run.stderr[-16000:])
            return ctypes.CDLL(str(library))

        with patch.object(native_soa_cpu, 'compile_host_native_library', compile_units):
            with native_soa_cpu.NativeSoACPU(lower_program(fixture_program()), compile_execution_schema(),
                    native_functions=True, native_values=True, limits=VmLimits(steps=100000),
                    max_nodes=16384, max_edges=32768, max_payload_bytes=131072) as backend:
                assert_stage(backend, fixture_cases())
            import copy
            from tests.test_duel_v2_native_functions import fixture_program as scalar_program, fixture_schema, states, CASES
            from app.modules.card_game.rl.batched_duel.record_soa import pack_game_states
            schema = fixture_schema()
            with native_soa_cpu.NativeSoACPU(lower_program(scalar_program()), schema,
                    native_functions=True, native_values=True, limits=VmLimits(steps=100000)) as backend:
                for entry, formal, extras in CASES:
                    result = backend.call(entry, pack_game_states(states(), schema),
                                          [[None, *extra] for extra in extras]).decode()
                    self.assertEqual(result, [formal(copy.deepcopy(state), *extra)
                                              for state, extra in zip(states(), extras)])


if __name__ == '__main__':
    unittest.main()
