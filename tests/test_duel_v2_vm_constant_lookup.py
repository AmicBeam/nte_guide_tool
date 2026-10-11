"""Frozen lookup equivalence across globals, attributes, misses and mutation."""
from pathlib import Path
import os
import sys
import unittest
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from app.modules.card_game.rl.batched_duel.vm_constant_lookup import emit_constant_lookup

VALUES = {'alpha': 11, 'omega': 29, 'zero': 0}
MIXED = {True: 'bool', 'name': 'str', 2.0: 'float'}


class Example:
    number = 17


def lookups(value):
    dynamic = ''.join(value)
    local = dict(VALUES)
    local['alpha'] = 101
    del local['omega']
    local['omega'] = 303
    return (VALUES['alpha'], VALUES.get(dynamic, -7), local['alpha'],
            local['omega'], MIXED[1], MIXED[2], Example.number)


def program():
    from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram, capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES
    path = os.environ.get('NTE_VM_CONSTANT_LOOKUP_PROGRAM')
    if path:
        return FrozenRuleProgram.from_json(Path(path).read_text())
    if sys.version_info[:2] != (3, 10):
        raise unittest.SkipTest('Same-source CPython3.10 fixture export required')
    return capture_program({'lookups': lookups}, allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,), scope_character_ids=None, scope_card_ids=None)


class ConstantLookupTest(unittest.TestCase):
    def test_stable_index_and_default_emission(self):
        from app.modules.card_game.rl.batched_duel.cuda_vm import emit_vm_source
        image = lower_program(program())
        plan = emit_constant_lookup(image)
        self.assertEqual(plan.identity, emit_constant_lookup(image).identity)
        self.assertGreater(plan.indexed_keys, 3)
        self.assertNotIn('frozen_dict_hit', emit_vm_source(image))
        source = emit_vm_source(image, constant_lookup=True)
        start = source.index('__device__ inline int dict_find(')
        hit = source.index('int hit = frozen_dict_hit', start)
        self.assertLess(source.index('node.tag != TAG_DICT', start), hit)
        self.assertLess(hit, source.index('key_equal(heap, k, key_ref)', hit))
        self.assertIn('if(dict_ref>=0||key_ref>=0)return -2', source)

    def test_real_cuda_plain_indexed_outputs_and_statuses(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        image = lower_program(program())
        inputs = [['a','l','p','h','a'], ['o','m','e','g','a'], ['missing']]
        observed = []
        for flag in (False, True):
            with CudaVm(image, constant_lookup=flag) as vm:
                call = vm.launch(vm.ingest(pack_states(inputs, GraphLayout(3,4096,8192,32768))), 'lookups')
                values = call.decode()
                observed.append((values, call.statuses.cpu().tolist()))
        self.assertEqual(observed[0], observed[1])
        self.assertEqual(observed[1][0], [lookups(v) for v in inputs])


if __name__ == '__main__':
    unittest.main()
