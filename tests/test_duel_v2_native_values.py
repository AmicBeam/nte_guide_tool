"""Native reference locals/constant columns versus independent canonical builds."""
from collections import Counter
import copy
from dataclasses import replace
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
from app.modules.card_game.rl.batched_duel.native_array_codegen import native_array_plan
from app.modules.card_game.rl.batched_duel.native_values import compile_native_values, integrate_native_values
from app.modules.card_game.rl.batched_duel.native_soa_cpu import NativeSoACPU
from app.modules.card_game.rl.batched_duel.record_soa import compile_schema, pack_game_states, unpack_game_states
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from tests import test_duel_v2_native_functions as base


def alias_reads(state, n):
    left = state['left']
    value = left['hp']
    right = state['right']
    before = right['hp']
    left['hp'] = value + n
    return state, value, before, right['hp'], left is right


def counter_reads(state, n):
    counts = Counter({'hp': n})
    value = counts['hp']
    absent = counts['missing']
    return state, value, absent


def unbound_reads(state, n):
    if n:
        value = state['left']
    hp = value['hp']
    return state, hp


def fallback_reads(state, n):
    values = {'hp': n}
    value = values['hp']
    constant = {'hp': 7}['hp']
    return state, value, constant


def fixture_program():
    from app.modules.card_game.rl.rule_ir.portable_program import (
        FrozenRuleProgram, capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES)
    path = os.environ.get('NTE_NATIVE_VALUES_PROGRAM')
    if path:
        return FrozenRuleProgram.from_json(Path(path).read_text(encoding='utf8'))
    if sys.version_info[:2] != (3, 10):
        raise unittest.SkipTest('CPython3.10 frozen fixture required')
    functions = [base.branch_leaf, base.lookup_leaf, base.checked_lookup, base.indexed_leaf,
                 alias_reads, counter_reads, unbound_reads, fallback_reads]
    return capture_program({f.__name__: f for f in functions},
        scope_character_ids=None, scope_card_ids=None,
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__, base.__name__))


def fixture_schema():
    # Non-record pools precede Child to catch column-number mistakes.
    return compile_schema({
        'Game': dict(kind='record', capacity=8, fields={
            'hp': 'int', 'items': '@Items', 'mapping': '@Mapping',
            'left': '@Child', 'right': '@Child'}),
        'Items': dict(kind='list', capacity=32, value='int', max_items=8),
        'Mapping': dict(kind='map', capacity=8, key='str', value='int', max_items=8),
        'Child': dict(kind='record', capacity=16, fields={'hp': 'int'}),
    }, max_strings=64, max_string_bytes=2048)


def states():
    result = base.states()
    for i, state in enumerate(result):
        child = {'hp': i + 5}
        state.update(left=child, right=child, mapping={'hp': i})
    return result


class NativeValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image, cls.schema = lower_program(fixture_program()), fixture_schema()

    def test_static_regions_columns_and_fail_closed_boundaries(self):
        plan = native_array_plan(self.image)
        compiled = compile_native_values(self.image, self.schema, plan)
        compiled.verify()
        self.assertTrue(compiled.regions)
        self.assertGreater(compiled.constant_reads, 0)
        self.assertIn('int local_', compiled.source)
        self.assertIn('soa_box_cell(&c->h,', compiled.source)
        self.assertIn('native_value_subscript', compiled.source)
        self.assertNotIn('push(c,f', compiled.source)
        self.assertNotIn('pop(c,f', compiled.source)
        for _, _, begin, end in compiled.regions:
            self.assertLessEqual(end - begin, 16)
            for row in plan.coverage['instructions'][begin:end]:
                self.assertEqual(row['target'], -1)
                self.assertNotIn('CALL', row['op'])
        with self.assertRaises(ValueError):
            replace(compiled, constant_reads=-1).verify()
        with self.assertRaises(ValueError):
            compile_native_values(self.image, self.schema, plan, max_regions=True)
        with self.assertRaises(ValueError):
            integrate_native_values('changed PC source', self.image, self.schema, plan)

    def test_all_budget_cuts_aliases_errors_and_canonical_handlers(self):
        options = dict(limits=VmLimits(frames=64, locals=128, stack=128, cells=64, steps=100000),
                       max_nodes=8192, max_edges=16384, max_payload_bytes=65536)
        cases = (*base.CASES,
                 ('alias_reads', alias_reads, [[-2], [0], [3]]),
                 ('counter_reads', counter_reads, [[-2], [0], [3]]),
                 ('unbound_reads', unbound_reads, [[1], [2], [3]]),
                 ('fallback_reads', fallback_reads, [[-2], [0], [3]]))
        with NativeSoACPU(self.image, self.schema, **options) as baseline, \
             NativeSoACPU(self.image, self.schema, native_values=True, **options) as native:
            for entry, formal, extras in cases:
                args = [[None, *extra] for extra in extras]
                complete = baseline.call(entry, pack_game_states(states(), self.schema), args)
                complete.check()
                maximum = int(complete.statuses[:, 3].max())
                for steps in (*range(1, maximum + 2), 100000):
                    left = baseline.call(entry, pack_game_states(states(), self.schema), args, steps=steps)
                    right = native.call(entry, pack_game_states(states(), self.schema), args, steps=steps)
                    np.testing.assert_array_equal(left.statuses, right.statuses, err_msg=(entry, steps))
                    self.assertEqual(unpack_game_states(left.state), unpack_game_states(right.state))
                    if not left.statuses[:, 0].any():
                        actual = right.decode()
                        self.assertEqual(left.decode(), actual)
                        self.assertEqual(actual, [formal(copy.deepcopy(s), *extra)
                                                 for s, extra in zip(states(), extras)])
                        if entry == 'alias_reads':
                            self.assertTrue(all(value[0]['left'] is value[0]['right'] for value in actual))
            for entry, extras in (('unbound_reads', [[0]] * 3),
                                  ('indexed_leaf', [[3], [-4], [100]]),
                                  ('lookup_leaf', [['不存在', 1]] * 3),
                                  ('lookup_leaf', [['mapping', 2]] * 3)):
                args = [[None, *extra] for extra in extras]
                left = baseline.call(entry, pack_game_states(states(), self.schema), args)
                right = native.call(entry, pack_game_states(states(), self.schema), args)
                np.testing.assert_array_equal(left.statuses, right.statuses)
                self.assertTrue(right.statuses[:, 0].all())
                self.assertEqual(unpack_game_states(left.state), unpack_game_states(right.state))
            absent = states()
            del absent[0]['hp']
            args = [[None, 'hp', 1]] * 3
            left = baseline.call('checked_lookup', pack_game_states(absent, self.schema), args)
            right = native.call('checked_lookup', pack_game_states(absent, self.schema), args)
            self.assertEqual(left.decode(), right.decode())
            self.assertEqual(right.decode()[0], ('missing', ('hp',)))
            for capacity in (6, 8, 12):
                with patch.object(baseline, 'max_nodes', capacity), \
                     patch.object(native, 'max_nodes', capacity):
                    args = [[None, 3]] * 3
                    left = baseline.call('alias_reads', pack_game_states(states(), self.schema), args)
                    right = native.call('alias_reads', pack_game_states(states(), self.schema), args)
                    np.testing.assert_array_equal(left.statuses, right.statuses)
                    np.testing.assert_array_equal(left.heap['counts'], right.heap['counts'])
                    self.assertEqual(unpack_game_states(left.state), unpack_game_states(right.state))

    def test_actual_cuda_values_and_expired_deadline(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.native_soa_cuda_query import NativeSoACUDAQuery
        with NativeSoACUDAQuery(self.image, self.schema, native_values=True,
                limits=VmLimits(steps=100000), seconds=10,
                max_nodes=8192, max_edges=16384, max_payload_bytes=65536) as native:
            for entry, formal, extras in (*base.CASES,
                    ('alias_reads', alias_reads, [[-2], [0], [3]]),
                    ('counter_reads', counter_reads, [[-2], [0], [3]])):
                result = native.call(entry, pack_game_states(states(), self.schema),
                                     [[None, *extra] for extra in extras])
                self.assertEqual(result.decode(), [formal(copy.deepcopy(s), *extra)
                                                  for s, extra in zip(states(), extras)])
            with patch.object(native.executor, 'clock_ns', return_value=1), \
                 patch.object(native, 'seconds', 1e-9):
                expired = native.call('branch_leaf', pack_game_states(states(), self.schema), [[None, 512]] * 3)
                self.assertTrue((expired.statuses[:, 0] == 108).all())
                self.assertTrue((expired.statuses[:, 3] == 64).all())


if __name__ == '__main__':
    unittest.main()
