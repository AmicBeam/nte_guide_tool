"""Runtime-only class lookup indexing: C3/descriptors/aliases and frozen bodies."""
import ctypes
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from app.modules.card_game.rl.batched_duel.native_compile_units import split_native_source
from app.modules.card_game.rl.batched_duel.native_runtime_index import index_runtime_attributes
from app.modules.card_game.rl.batched_duel import native_soa_cpu
from app.modules.card_game.rl.batched_duel.native_array_cpu import _cxx_compiler
from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from app.modules.card_game.rl.batched_duel.record_soa import pack_game_states


class NativeRuntimeIndexTests(unittest.TestCase):
    def compiler(self,image,source,directory):
        directory=Path(directory);units=split_native_source(source);paths=[]
        optimized=index_runtime_attributes(units[0],image)
        self.assertNotEqual(units[0].sha256,optimized.sha256)
        from app.modules.card_game.rl.batched_duel.native_runtime_index import _definition
        for name in ('lookup_own', 'lookup_class'):
            self.assertEqual(_definition(units[0].source,name),_definition(optimized.source,name))
        self.assertIn('native_indexed_lookup_class(c,',optimized.source)
        with self.assertRaises(ValueError):index_runtime_attributes(units[1],image)
        with self.assertRaises(ValueError):index_runtime_attributes(optimized,image)
        for unit in (optimized,*units[1:]):
            path=directory/(unit.name+'.cpp');path.write_text(unit.source);paths.append(str(path))
        library=directory/'indexed.dylib'
        result=subprocess.run([_cxx_compiler(),'-std=c++17','-shared','-fPIC','-O1','-fno-fast-math','-ffp-contract=off',*paths,'-o',str(library)],capture_output=True,text=True,timeout=30)
        if result.returncode:self.fail(result.stderr[-12000:])
        return ctypes.CDLL(str(library))

    def test_smaller_stack_is_an_explicit_execute_only_diagnostic(self):
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import _validate_soa_stack
        _validate_soa_stack(65536, True)
        for diagnostic in (False, True):
            _validate_soa_stack(131072, diagnostic)
            _validate_soa_stack(1024**2, diagnostic)
        with self.assertRaises(ValueError):
            _validate_soa_stack(65536, False)
        for value in (True, 0, 32768, 65535, 1024**2+1):
            with self.assertRaises(ValueError):
                _validate_soa_stack(value, True)
        with self.assertRaises(TypeError):
            _validate_soa_stack(65536, 1)

    def test_real_rule_helpers_state_and_entity_aliases(self):
        from tests.test_duel_v2_native_rule_stages import fixture_program,fixture_cases,assert_stage
        from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
        image=lower_program(fixture_program())
        with patch.object(native_soa_cpu,'compile_host_native_library',lambda source,directory:self.compiler(image,source,directory)):
            with native_soa_cpu.NativeSoACPU(image,compile_execution_schema(),native_functions=True,native_values=True,
                            limits=VmLimits(steps=100000),max_nodes=16384,max_edges=32768,max_payload_bytes=131072) as backend:
                assert_stage(backend,fixture_cases())

    def test_inherited_descriptors_shadowing_static_class_methods_and_dynamic_names(self):
        from tests.test_duel_v2_vm_frozen_attributes import program,attributes
        from tests.test_duel_v2_native_functions import fixture_schema
        image=lower_program(program());schema=fixture_schema()
        parts=[list('value'),list('_number'),['汉','字'],list('missing'),[],['汉']]
        states=[dict(hp=1,items=[]) for _ in parts]
        with patch.object(native_soa_cpu,'compile_host_native_library',lambda source,directory:self.compiler(image,source,directory)):
            with native_soa_cpu.NativeSoACPU(image,schema,native_functions=True,native_values=True,
                            limits=VmLimits(steps=100000),max_nodes=16384,max_edges=32768,max_payload_bytes=131072) as backend:
                call=backend.call('attributes',pack_game_states(states,schema),[[p] for p in parts],state_positions=())
                self.assertEqual(call.decode(),[attributes(p) for p in parts])
                bad=backend.call('attributes',pack_game_states(states[:1],schema),[[list('prop')]],state_positions=())
                self.assertEqual(int(bad.statuses[0,0]),106)


    def test_step_cut_errors_and_allocations_match_baseline(self):
        import numpy as np
        from tests.test_duel_v2_vm_frozen_attributes import program
        from tests.test_duel_v2_native_functions import fixture_schema
        image=lower_program(program());schema=fixture_schema()
        options=dict(native_functions=True,native_values=True,limits=VmLimits(steps=100000),max_nodes=16384,max_edges=32768,max_payload_bytes=131072)
        original=native_soa_cpu.compile_host_native_library
        with patch.object(native_soa_cpu,'compile_host_native_library',lambda source,directory:original(source,directory,timeout=30)):
            baseline=native_soa_cpu.NativeSoACPU(image,schema,**options)
        with baseline,patch.object(native_soa_cpu,'compile_host_native_library',lambda source,directory:self.compiler(image,source,directory)):
            with native_soa_cpu.NativeSoACPU(image,schema,**options) as indexed:
                for parts in (list('value'),list('prop')):
                    for steps in (1,2,20,100,500,100000):
                        calls=[backend.call('attributes',pack_game_states([dict(hp=1,items=[])],schema),[[parts]],state_positions=(),steps=steps) for backend in (baseline,indexed)]
                        np.testing.assert_array_equal(calls[0].statuses,calls[1].statuses)
                        np.testing.assert_array_equal(calls[0].outputs,calls[1].outputs)
                        for key in ('nodes','edges','payload','counts'):
                            np.testing.assert_array_equal(calls[0].heap[key],calls[1].heap[key])


if __name__=='__main__':unittest.main()
