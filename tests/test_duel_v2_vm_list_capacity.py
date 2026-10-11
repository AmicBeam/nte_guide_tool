"""Actual CUDA regression for fragmented lists and private reserved edges."""
from pathlib import Path
import os
import sys
import unittest

from app.modules.card_game.rl.batched_duel.codec import pack_states
from app.modules.card_game.rl.batched_duel.schema import GraphLayout, TAG_LIST
from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm, VmLimits, VmExecutionError
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from app.modules.card_game.rl.rule_ir.portable_program import capture_program, FrozenRuleProgram


def fragmented(count):
    values = []
    junk = []
    alias = values
    for i in range(count):
        junk.append((i, i + 1))
        values.append(float(i))
    return {'values': values, 'alias': alias, 'junk_count': len(junk)}


def mutations(count):
    from copy import deepcopy
    values = fragmented(count)['values']
    values.insert(0, -1.)
    del values[10]
    values = values[:21]
    values.append(777.)
    values.pop()
    values.append(888.)
    cloned = deepcopy([values, values])
    cloned[0].append(999.)
    cycle = []
    cycle.append(cycle)
    copied_cycle = deepcopy(cycle)
    return values, cloned, copied_cycle[0] is copied_cycle


def append_after_export(value):
    value['values'].append(1201.)
    return value


FROZEN_LIST = []


def constant_write(value):
    FROZEN_LIST.append(value)
    return FROZEN_LIST


def program():
    fixture = os.environ.get('NTE_VM_LIST_CAPACITY_PROGRAM')
    if fixture:
        return FrozenRuleProgram.from_json(Path(fixture).read_text())
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('Explicit CPython3.10 frozen list-capacity fixture required')
    return capture_program(dict(fragmented=fragmented, mutations=mutations, append=append_after_export,
                                constant_write=constant_write),
                           allowed_module_prefixes=(__name__,), scope_character_ids=None, scope_card_ids=None)


class ListCapacityCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest('Actual CUDA required')
        cls.vm = CudaVm(lower_program(program()), limits=VmLimits(steps=4000000))

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'vm'):
            cls.vm.close()

    def test_1200_fragmented_appends_fit_and_preserve_aliases(self):
        arena = self.vm.ingest(pack_states([1200], GraphLayout(1, 16384, 32768, 131072)))
        call = self.vm.launch(arena, 'fragmented')
        actual = call.decode()[0]
        self.assertEqual(actual, fragmented(1200))
        self.assertIs(actual['values'], actual['alias'])
        self.assertLess(int(call.arena.counts[0, 1].cpu()), 32768)

    def test_mutation_relocation_pop_and_deepcopy_alias_cycle(self):
        arena = self.vm.ingest(pack_states([120], GraphLayout(1, 16384, 32768, 131072)))
        actual = self.vm.launch(arena, 'mutations').decode()[0]
        self.assertEqual(actual, mutations(120))
        self.assertIs(actual[1][0], actual[1][1])
        self.assertIsNot(actual[0], actual[1][0])

    def test_compaction_resets_private_capacity_and_later_append_is_safe(self):
        arena = self.vm.ingest(pack_states([1200], GraphLayout(1, 16384, 32768, 131072)))
        packed = self.vm.launch(arena, 'fragmented').export()
        nodes = packed.nodes[0, :packed.counts[0, 0]]
        self.assertTrue((nodes[nodes[:, 0] == TAG_LIST, 3] == 0).all())
        actual = self.vm.launch(self.vm.ingest(packed), 'append').decode()[0]
        expected = append_after_export(fragmented(1200))
        self.assertEqual(actual, expected)
        self.assertIs(actual['values'], actual['alias'])

    def test_edge_capacity_still_fails_explicitly(self):
        arena = self.vm.ingest(pack_states([1200], GraphLayout(1, 16384, 64, 131072)))
        with self.assertRaises(VmExecutionError):
            self.vm.launch(arena, 'fragmented').check()

    def test_constant_write_guard_is_preserved(self):
        arena = self.vm.ingest(pack_states([3], GraphLayout(1, 128, 128, 1024)))
        call = self.vm.launch(arena, 'constant_write')
        with self.assertRaises(VmExecutionError):
            call.check()
        self.assertEqual(int(call.statuses[0, 0].cpu()), 8)


if __name__ == '__main__':
    unittest.main()
