"""Cross-module VM probes; source checks do not establish CUDA readiness."""
import json
import sys
import os
from pathlib import Path
from uuid import uuid4
import unittest
from collections import Counter
from enum import Enum

import numpy as np

from app.modules.card_game.rl.rule_ir.portable_program import (
    DEFAULT_ALLOWED_MODULE_PREFIXES, ExportError, FrozenRuleProgram, capture_program,
)

SELF_NUMBER = 418


def self_module_probe(value):
    return sys.modules[__name__].SELF_NUMBER + value


def dynamic_module_probe(value):
    return sys.modules[value]


def standalone_uuid_probe(value):
    return uuid4().hex


def numpy_json_probe(value):
    array = np.asarray(value, dtype=np.float32)
    array[0] += np.float32(0.5)
    return json.dumps({'values': array.tolist(), 'valid': bool(np.isfinite(array).all())}, sort_keys=True)


def seven_slice_probe(value):
    array = np.asarray(value, dtype=np.float32)
    legacy_columns = array[:, :3]
    legacy_columns[0, 0] += np.float32(0.5)
    return legacy_columns.tolist(), array.tolist()


def counter_zero_probe(value):
    counts = Counter(value)
    counts['explicit'] = 0
    counts.update({'explicit': 0, 'negative': -2})
    counts.update({'negative': 2})
    return dict(counts), list(counts.keys()), len(counts), bool(counts), counts['missing'], counts == Counter(value)


def signed_modulo_probe(value):
    divisor = float(value)
    result = 4.0 % divisor
    quotient = 4.0
    quotient /= divisor
    return str(result), quotient


class Word(str, Enum):
    A = 'a'


class OtherWord(str, Enum):
    A = 'a'


class Number(Enum):
    A = 1


def enum_keys_probe(value):
    mapping = {Word.A: value}
    return mapping['a'], 'a' in {Word.A}, Word.A == OtherWord.A, [Number.A] == [1]


def probe_program(entries, stem):
    if sys.version_info[:2] != (3, 10):
        directory = os.environ.get('NTE_VM_TEST_PROGRAM_DIR')
        if not directory:
            raise RuntimeError('Explicit CPython 3.10 frozen integration fixture required')
        program = FrozenRuleProgram.from_json((Path(directory) / (stem + '.json')).read_text())
        from app.modules.card_game.rl.batched_duel.gpu_rule_program import verify_source_hashes
        verify_source_hashes(program, repo_root=Path(__file__).resolve().parents[1])
        if set(program.entries) != set(entries):
            raise ValueError('Frozen integration entry mismatch')
        return program
    return capture_program(entries,
                           allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,),
                           scope_character_ids=None, scope_card_ids=None)


def integration_program():
    return probe_program(
        {'numpy_json': numpy_json_probe, 'counter_zero': counter_zero_probe,
         'signed_modulo': signed_modulo_probe, 'enum_keys': enum_keys_probe,
         'self_module': self_module_probe, 'seven_slice': seven_slice_probe},
        'integrations',
    )


class TestVmIntegrationSource(unittest.TestCase):
    def test_all_cross_module_dependencies_are_declared(self):
        from app.modules.card_game.rl.batched_duel.cuda_vm import SUPPORTED_INTRINSICS, SUPPORTED_OPS, emit_vm_source
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        image = lower_program(integration_program())
        for entry in image.entries:
            self.assertEqual(image.missing_capabilities(entry, opcodes=SUPPORTED_OPS, intrinsics=SUPPORTED_INTRINSICS),
                             {'opcodes': [], 'intrinsics': []})
        source = emit_vm_source(image)
        self.assertIn('numpy_call(c,callable', source)
        self.assertIn('json_call(c,callable', source)

    def test_explicit_zero_counts_survive_python_updates(self):
        self.assertEqual(counter_zero_probe([]),
                         ({'explicit': 0, 'negative': 0}, ['explicit', 'negative'], 2, True, 0, True))

    def test_self_module_lookup_excludes_ambient_modules(self):
        program = integration_program()
        self.assertEqual(set(program.modules['sys']), {'modules'})
        self.assertIn('SELF_NUMBER', program.modules[__name__])
        self.assertNotIn('torch', program.modules)
        if sys.version_info[:2] == (3, 10):
            with self.assertRaisesRegex(ExportError, 'Only static sys.modules'):
                capture_program({'bad': dynamic_module_probe},
                                allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,),
                                scope_character_ids=None, scope_card_ids=None)

    def test_standalone_guard_is_explicit_and_does_not_claim_uuid_support(self):
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm, SUPPORTED_INTRINSICS, emit_vm_source
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        program = probe_program({'uuid': standalone_uuid_probe}, 'integration_uuid')
        vm = CudaVm.__new__(CudaVm)
        vm.image = lower_program(program)
        vm.guard_standalone_entities = False
        self.assertFalse(vm.capability('uuid')['ready'])
        vm.guard_standalone_entities = True
        report = vm.capability('uuid')
        self.assertTrue(report['ready'])
        self.assertEqual(report['missing']['intrinsics'], ['uuid.uuid4'])
        self.assertEqual(report['runtime_denied_intrinsics'], ['uuid.uuid4'])
        self.assertFalse(report['full_duel_rules'])
        self.assertNotIn('uuid.uuid4', SUPPORTED_INTRINSICS)
        source = emit_vm_source(vm.image)
        self.assertIn('c->error=UNSUPPORTED_TYPE;return 1;', source)

    def test_device_cross_module_parity(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA unavailable; device integration parity not established')
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        with CudaVm(lower_program(integration_program()), device='cuda') as vm:
            for entry, function, values in (
                ('numpy_json', numpy_json_probe, [[1, 2], [-0.5, 4]]),
                ('counter_zero', counter_zero_probe, [[], ['a', 'a', 'b']]),
                ('signed_modulo', signed_modulo_probe, [-2, 2]),
                ('enum_keys', enum_keys_probe, [3, 7]),
                ('self_module', self_module_probe, [3, 7]),
                ('seven_slice', seven_slice_probe, [[[1, 2, 3, 4], [5, 6, 7, 8]]]),
            ):
                arena = vm.ingest(pack_states(values, GraphLayout(len(values), 8192, 16384, 131072)))
                self.assertEqual(vm.launch(arena, entry).decode(), [function(value) for value in values])

    def test_device_standalone_uuid_is_denied(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA unavailable; denied UUID dispatch not device-verified')
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm, VmExecutionError
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        program = probe_program({'uuid': standalone_uuid_probe}, 'integration_uuid')
        with CudaVm(lower_program(program), device='cuda', guard_standalone_entities=True) as vm:
            with self.assertRaises(VmExecutionError):
                vm.launch(vm.ingest(pack_states([0], GraphLayout(1, 1024, 2048, 8192))), 'uuid').check()
