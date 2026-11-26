"""Reachability of in-function frozen imports for GPU-rule VM images."""
from pathlib import Path
import hashlib
import os
import sys
import unittest

from app.modules.card_game.rl.batched_duel.cuda_vm import SUPPORTED_INTRINSICS, SUPPORTED_OPS
from app.modules.card_game.rl.batched_duel.vm_image import ImageError, lower_program
from app.modules.card_game.rl.rule_ir.portable_program import (
    FrozenRuleProgram,
    capture_program,
    compute_identity,
)


def power_callee(value):
    return value ** 2


def import_power(value):
    from tests.test_duel_v2_gpu_rule_imports import power_callee as callee
    return callee(value)


def test_program(stem, entry, function, *, own=False):
    if sys.version_info[:2] == (3, 10):
        kwargs = dict(scope_character_ids=None, scope_card_ids=None)
        if own:
            from app.modules.card_game.rl.rule_ir.portable_program import DEFAULT_ALLOWED_MODULE_PREFIXES
            kwargs['allowed_module_prefixes'] = DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,)
        return capture_program({entry: function}, **kwargs)
    directory = os.environ.get('NTE_VM_TEST_PROGRAM_DIR')
    if not directory:
        raise RuntimeError('CPython 3.13 consumes explicit 3.10 frozen fixtures; no native recapture')
    fixture = Path(directory) / (stem + '.json')
    program = FrozenRuleProgram.from_json(fixture.read_text())
    program.validate()
    root = Path(__file__).resolve().parents[1]
    from app.modules.card_game.rl.offline_sources import _allowed_source
    for name, digest in program.source_hashes.items():
        path = root / name
        if (
            not _allowed_source(name)
            or path.is_symlink()
            or root not in path.resolve().parents
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError('Frozen test source mismatch: ' + name)
    if entry not in program.entries:
        raise ValueError('Frozen test entry mismatch')
    return program


@unittest.skipUnless(
    sys.version_info[:2] == (3, 10) or os.environ.get('NTE_VM_TEST_PROGRAM_DIR'),
    'CPython 3.10 capture or an explicit frozen program directory is required',
)
class GpuRuleImportReachabilityTest(unittest.TestCase):
    def test_imported_callee_unsupported_opcode_blocks_entry(self):
        program = test_program('import_power', 'import_power', import_power, own=True)
        image = lower_program(program)
        missing = image.missing_capabilities(
            'import_power', opcodes=SUPPORTED_OPS, intrinsics=SUPPORTED_INTRINSICS,
        )
        self.assertIn('BINARY_POWER', missing['opcodes'])
        self.assertIn('BINARY_POWER', image.required['import_power']['opcodes'])

    def test_missing_frozen_import_symbol_is_not_skipped(self):
        program = test_program('import_power', 'import_power', import_power, own=True)
        payload = program.to_dict()
        table = payload['modules'].get(__name__)
        self.assertIsInstance(table, dict)
        self.assertIn('power_callee', table)
        del table['power_callee']
        payload['identity'] = compute_identity(payload)
        altered = FrozenRuleProgram(payload)
        with self.assertRaisesRegex(ImageError, 'Frozen import symbol missing'):
            lower_program(altered)

    def test_serving_root_metadata_includes_imported_callee_opcodes(self):
        from app.modules.card_game.rl.batched_duel import gpu_rule_entries as entries
        from app.modules.card_game.rl.batched_duel import gpu_rule_program as gp
        if sys.version_info[:2] != (3, 10):
            raise unittest.SkipTest('CPython 3.10 source capture required')
        program, status = gp.build_gpu_rule_program(entries.LIGHT_ENTRY_NAMES)
        self.assertTrue(status['image_ready'])
        image = gp.lower_gpu_program(program)
        opcodes = set(image.required['serving_root_metadata']['opcodes'])
        self.assertIn('GET_ITER', opcodes)
        self.assertIn('MAKE_FUNCTION', opcodes)


if __name__ == '__main__':
    unittest.main()
