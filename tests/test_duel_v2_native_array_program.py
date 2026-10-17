"""Compiler wiring and teacher-free adapter contracts; not CUDA qualification."""
import unittest
import sys
import numpy as np

from app.modules.card_game.rl.batched_duel.native_array_program import NativeArrayExporter


def original_value(x):
    return x + 1


def replacement_value(x):
    return x + 2


def imported_entry(x):
    return original_value(x)


class NativeArrayProgramTest(unittest.TestCase):
    def test_substitution_is_freeze_local_and_redirects_every_reference(self):
        exporter = NativeArrayExporter({original_value: replacement_value},
            allowed_module_prefixes=(__name__,), scope_character_ids=(), scope_card_ids=())
        program = exporter.run_capture({'entry': imported_entry})
        program.validate()
        names = [fn['name'] for fn in program.functions]
        self.assertIn('replacement_value', names)
        self.assertNotIn('original_value', names)
        self.assertEqual(imported_entry(4), 5)
        self.assertEqual(exporter.substituted, {original_value})

    def test_native_cuda_uploads_no_instruction_stream(self):
        from app.modules.card_game.rl.batched_duel.native_array_cuda import CudaNativeArray
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        exporter = NativeArrayExporter({}, allowed_module_prefixes=(__name__,),
                                       scope_character_ids=(), scope_card_ids=())
        image = lower_program(exporter.run_capture({'entry': original_value}))
        native = object.__new__(CudaNativeArray)
        old = object.__new__(CudaVm)
        placeholder = native._instruction_input(image)
        self.assertEqual(placeholder.shape, (1, 4))
        self.assertEqual(placeholder.dtype, np.int32)
        self.assertTrue((placeholder == -2147483648).all())
        self.assertTrue(np.array_equal(old._instruction_input(image), image.instructions))
        self.assertFalse(native.interpreter_fallback)
        self.assertFalse(native.runtime_instruction_fetch)

    def test_teacher_free_sites_have_no_teacher_dependency_or_capture(self):
        from app.modules.card_game.rl.batched_search.device_frozen_rules import (
            _positive_sites, DEFAULT_SITES, DeviceFrozenRules,
        )
        sites = _positive_sites(None, require_teacher=False)
        self.assertEqual(sites['rule_decision'], 0)
        self.assertEqual(sites['teacher_continue'], 0)
        self.assertEqual(_positive_sites(None), DEFAULT_SITES)
        self.assertGreater(DEFAULT_SITES['teacher_continue'], 0)
        with self.assertRaisesRegex(ValueError, 'Teacher-free'):
            _positive_sites({'rule_decision': 1}, require_teacher=False)
        with self.assertRaisesRegex(ValueError, 'Teacher-free'):
            _positive_sites({'teacher_continue': 1}, require_teacher=False)
        rules = object.__new__(DeviceFrozenRules)
        rules.require_teacher = False
        with self.assertRaisesRegex(RuntimeError, 'absent'):
            rules.teacher_continue_device(None, None, horizon=0)
        with self.assertRaisesRegex(RuntimeError, 'absent'):
            rules.rule_decision_device(None, None)

    def test_teacher_free_search_refuses_hidden_teacher_modes(self):
        from tests.test_duel_v2_device_search import DeviceSearchTest
        from app.modules.card_game.rl.batched_search.device_search import DeviceGumbelSearch
        search, rules = DeviceSearchTest().engine()
        rules.require_teacher = False
        args = (rules, search.inference, search.auxiliary, search.statistics, search.information_keys)
        options = dict(rows=2, max_actions=3, device='cpu', allow_cpu_reference=True, allow_synthetic=True)
        with self.assertRaisesRegex(ValueError, 'network_value'):
            DeviceGumbelSearch(*args, **options)
        with self.assertRaisesRegex(ValueError, 'policy opponents'):
            DeviceGumbelSearch(*args, **options, teacher_mode='network_value', terminal_horizon=0)

    @unittest.skipUnless(sys.version_info[:2] == (3, 10), 'Source capture requires CPython 3.10')
    def test_full_native_program_load_is_bound_to_protocol_and_source(self):
        from pathlib import Path
        import tempfile
        from app.modules.card_game.rl.batched_duel.native_array_program import (
            capture_native_array_program, load_native_array_program,
        )
        from app.modules.card_game.rl.rule_ir.portable_program import ValidationError, compute_identity
        program = capture_native_array_program()
        self.assertFalse(program.data['native_array']['teacher_entries'])
        self.assertFalse(program.data['native_array']['policy_query_cache'])
        self.assertNotIn('teacher_continue', program.entries)
        self.assertNotIn('rule_decision', program.entries)
        with tempfile.TemporaryDirectory(prefix='native_array_freeze_') as folder:
            path = Path(folder)/'program.json'
            path.write_text(program.to_json(), encoding='utf8')
            self.assertEqual(load_native_array_program(path).identity, program.identity)
            program.data['native_array']['interpreter_fallback'] = True
            program.data['identity'] = compute_identity(program.data)
            path.write_text(program.to_json(), encoding='utf8')
            with self.assertRaisesRegex(ValidationError, 'protocol'):
                load_native_array_program(path)
            program.data['native_array']['interpreter_fallback'] = False
            name = 'app/modules/card_game/rl/batched_duel/array_rule_sources.py'
            program.data['source_hashes'][name] = '0'*64
            program.data['identity'] = compute_identity(program.data)
            path.write_text(program.to_json(), encoding='utf8')
            with self.assertRaisesRegex(ValidationError, 'source hash mismatch'):
                load_native_array_program(path)

    @unittest.skipUnless(sys.version_info[:2]==(3,10),'Source capture requires CPython 3.10')
    def test_indexed_projection_is_explicit_and_bound_to_its_real_capture_source(self):
        from pathlib import Path
        import tempfile
        from app.modules.card_game.rl.batched_duel.native_array_program import capture_native_array_program,load_native_array_program
        from app.modules.card_game.rl.rule_ir.portable_program import ValidationError,compute_identity
        with self.assertRaises(TypeError):capture_native_array_program(policy_query_cache=1)
        program=capture_native_array_program(policy_query_cache=True)
        self.assertTrue(program.data['native_array']['policy_query_cache'])
        names={fn['name'] for fn in program.functions}
        self.assertIn('cached_policy_public_legal_decision',names);self.assertIn('_indexed_evaluate',names)
        self.assertIn('app/modules/card_game/rl/batched_duel/gpu_cached_policy_projection.py',program.data['source_hashes'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'program.json';path.write_text(program.to_json())
            self.assertEqual(load_native_array_program(path).identity,program.identity)
            program.data['native_array']['policy_query_cache']=1
            program.data['identity']=compute_identity(program.data);path.write_text(program.to_json())
            with self.assertRaises(ValidationError):load_native_array_program(path)

    def test_native_recipe_rejects_an_interpreter_with_a_native_label(self):
        from types import SimpleNamespace
        from app.modules.card_game.rl.batched_search.device_native_array_rules import (
            NativeArrayRuleCalls, DeviceNativeArrayRules, create_native_array_search,
        )
        fake = SimpleNamespace(execution_mode='native_array_aot', interpreter_fallback=False)
        with self.assertRaises(TypeError):
            NativeArrayRuleCalls(fake, None)
        with self.assertRaises(TypeError):
            DeviceNativeArrayRules(fake, None, action_store=None)
        with self.assertRaises(TypeError):
            create_native_array_search(fake, None, None, None, None, rows=8, random=None)

    def test_cuda_query_refuses_unbounded_budgets_before_device_setup(self):
        from app.modules.card_game.rl.batched_duel.native_array_cuda_query import NativeArrayCUDAQuery
        for seconds in (0, -1, True, float('inf'), float('nan'), 3601):
            with self.assertRaisesRegex(ValueError, 'budget'):
                NativeArrayCUDAQuery(None, seconds=seconds)


if __name__ == '__main__':
    unittest.main()
