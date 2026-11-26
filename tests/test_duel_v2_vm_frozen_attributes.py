"""Frozen class indexes preserve descriptor categories, C3 and live instances."""
import os
from pathlib import Path
import sys
import unittest

from app.modules.card_game.rl.batched_duel.vm_image import lower_program


class Base:
    value = 11
    汉字 = 43

    def __init__(self, number):
        self._number = number

    @property
    def prop(self):
        return self._number + 5

    def method(self, amount):
        return self._number + amount

    @staticmethod
    def static(amount):
        return amount * 3

    @classmethod
    def clsvalue(cls):
        return cls.value


class Left(Base):
    value = 19

    def method(self, amount):
        return super().method(amount) + 2


class Right(Base):
    def method(self, amount):
        return super().method(amount) + 7


class Derived(Left, Right):
    pass


class Shadow(Base):
    prop = 101


def attributes(parts):
    name = ''.join(parts)
    obj = Derived(7)
    before = getattr(obj, name, -23)
    obj.value = 31
    obj._number = 13
    return (before, getattr(obj, name, -23), obj.value, Derived.value,
            obj.prop, obj.method(3), Derived.static(4), Derived.clsvalue(),
            Shadow(7).prop, hasattr(Derived, name))


def program():
    from app.modules.card_game.rl.rule_ir.portable_program import (
        FrozenRuleProgram, capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES)
    path = os.environ.get('NTE_VM_FROZEN_ATTRIBUTES_PROGRAM')
    if path:
        return FrozenRuleProgram.from_json(Path(path).read_text())
    if sys.version_info[:2] != (3, 10):
        raise unittest.SkipTest('Same-source CPython3.10 fixture export required')
    return capture_program({'attributes': attributes}, allowed_module_prefixes=
                           DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,),
                           scope_character_ids=None, scope_card_ids=None)


class FrozenAttributesTest(unittest.TestCase):
    def test_explicit_emission_and_identity(self):
        from app.modules.card_game.rl.batched_duel.cuda_vm import emit_vm_source
        image = lower_program(program())
        plain = emit_vm_source(image)
        indexed = emit_vm_source(image, frozen_class_lookup=True)
        self.assertNotIn('frozen_attr_lookup', plain)
        self.assertIn('frozen_attr_own_entries', indexed)
        self.assertIn('frozen_attr_class_entries', indexed)
        self.assertIn('query.tag!=TAG_STR)return 0', indexed)
        self.assertEqual(indexed, emit_vm_source(image, frozen_class_lookup=True))
        self.assertTrue(image.verify())
        with self.assertRaises(TypeError):
            emit_vm_source(image, frozen_class_lookup=1)

    def test_actual_cuda_exact_c3_descriptors_and_mutable_instances(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        image = lower_program(program())
        inputs = [['v', 'a', 'l', 'u', 'e'], ['_', 'number'],
                  ['汉', '字'], ['missing'], [], ['汉']]
        outputs = []
        refused = []
        for flag in (False, True):
            with CudaVm(image, constant_lookup=True, frozen_class_lookup=flag) as vm:
                call = vm.launch(vm.ingest(pack_states(inputs, GraphLayout(
                    len(inputs), 8192, 16384, 65536))), 'attributes')
                outputs.append((call.decode(), call.statuses.cpu().tolist()))
                # The old dynamic-getattr helper cannot execute a property
                # getter. Keep its explicit refusal instead of counting that
                # unsupported fixture as a successful indexed lookup.
                bad = vm.launch(vm.ingest(pack_states([list('prop')], GraphLayout(
                    1, 8192, 16384, 65536))), 'attributes')
                refused.append(bad.statuses.cpu().tolist())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1][0], [attributes(parts) for parts in inputs])
        self.assertEqual(refused[0], refused[1])
        self.assertEqual(refused[0][0][0], 106)


if __name__ == '__main__':
    unittest.main()
