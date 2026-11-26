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


def scalar_values(state, n):
    before = state['hp']
    amount = (before + n) * 3
    positive = amount >= 0
    negative = -amount
    inverted = not positive
    state['hp'] = amount // 2
    return state, amount, negative, inverted, positive, (amount & 3), (amount % 3)


def closure_cell_reads(state, n):
    shared = state['left']
    def read():
        value = shared['hp']
        return shared, value
    def write():
        nonlocal shared
        before = shared
        shared = state['right']
        shared['hp'] = shared['hp'] + n
        return before, shared
    before = read()
    changed = write()
    after = read()
    return state, before, changed, after, shared


def closure_unbound_reads(state, n):
    def read():
        value = held['hp']
        return value + n
    if n:
        held = state['left']
    return read()


def closure_caught_missing(state, n):
    held = state['left']
    def read():
        nonlocal held
        held = state['right']
        value = held['missing']
        return value + n
    try:
        return read()
    except KeyError as error:
        return state, held, error.args


GLOBAL_REFERENCE = {'hp': 11}


def live_global_reads(state, n, namespace):
    before = GLOBAL_REFERENCE
    value = before['hp']
    def write():
        if namespace['delete']:
            namespace.pop('GLOBAL_REFERENCE')
        else:
            namespace['GLOBAL_REFERENCE'] = namespace['replacement']
    write()
    after = GLOBAL_REFERENCE
    state['hp'] = after['hp'] + n
    return state, value, before, after, namespace


def large_operand_fixture():
    # Real CPython extended-argument bytecode with a small local/stack footprint.
    lines = ['def large_operand_reads(state, n, namespace):', '    value = 0']
    for value in range(1, 262):
        lines.append('    value = ' + str(value))
    lines.extend(('    pass', "    state['hp'] = value + n", "    return state, value"))
    namespace = {'__name__': __name__}
    exec(compile('\n'.join(lines), __file__, 'exec'), namespace)
    return namespace['large_operand_reads']


def fixture_program():
    from app.modules.card_game.rl.rule_ir.portable_program import (
        FrozenRuleProgram, capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES)
    path = os.environ.get('NTE_NATIVE_VALUES_PROGRAM')
    if path:
        return FrozenRuleProgram.from_json(Path(path).read_text(encoding='utf8'))
    if sys.version_info[:2] != (3, 10):
        raise unittest.SkipTest('CPython3.10 frozen fixture required')
    functions = [base.branch_leaf, base.lookup_leaf, base.checked_lookup, base.indexed_leaf,
                 alias_reads, counter_reads, unbound_reads, fallback_reads, scalar_values,
                 closure_cell_reads, closure_unbound_reads, closure_caught_missing,
                 live_global_reads, large_operand_fixture()]
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


class NativeValueProjectionPlanningTests(unittest.TestCase):
    def test_frozen_policy_body_receives_bounded_hot_regions(self):
        from app.modules.card_game.rl.batched_duel.native_array_program import capture_native_array_program
        from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
        from app.modules.card_game.rl.batched_duel import native_values
        if sys.version_info[:2] != (3, 10):
            self.skipTest('Current rule bytecode capture requires CPython3.10')
        program = capture_native_array_program(policy_query_cache=True)
        packet = program.data
        functions = {f['id']: f for f in packet['functions']}
        entry = functions[packet['entries']['public_legal_decision']]
        # Follow the actual frozen binding rather than matching its tiny wrapper.
        helpers = [ref['id'] for name, ref in entry['globals'].items()
                   if name in ('_cached_observe', '_policy_observe') and ref['kind'] == 'function']
        self.assertEqual(len(helpers), 1)
        helper = functions[helpers[0]]
        self.assertIn(helper['name'], ('_cached_observe', '_policy_observe'))
        image = lower_program(program)
        plan, schema = native_array_plan(image), compile_execution_schema()
        old_hot = tuple(name for name in native_values._HOT_FUNCTIONS
                        if name not in ('_cached_observe', '_policy_observe'))
        with patch.object(native_values, '_HOT_FUNCTIONS', old_hot):
            before = compile_native_values(image, schema, plan)
        after = compile_native_values(image, schema, plan)
        body_regions = [region for region in after.regions if region[1] == helper['id']]
        self.assertTrue(body_regions)
        self.assertFalse(any(region[1] == helper['id'] for region in before.regions))
        self.assertEqual(len(after.regions), len(before.regions))
        self.assertLessEqual(len(after.regions), 256)
        removed = set(before.regions) - set(after.regions)
        self.assertTrue(removed)
        self.assertTrue(all(not any(':' + hot + ':' in region[1]
                                   for hot in native_values._HOT_FUNCTIONS) for region in removed))
        for _, _, begin, end in after.regions:
            self.assertLessEqual(end - begin, 16)
            for row in plan.coverage['instructions'][begin:end]:
                self.assertIn(row['op'], native_values.SUPPORTED)
                self.assertEqual(row['target'], -1)
        self.assertGreater(after.constant_reads, 0)


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
        cases = (*cases, ('scalar_values', scalar_values, [[-7], [0], [3]]))
        with NativeSoACPU(self.image, self.schema, **options) as baseline, \
             NativeSoACPU(self.image, self.schema, native_values=True, native_functions=True, **options) as native:
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

    def test_shared_cells_all_budget_cuts_error_heaps_and_aliases(self):
        from app.modules.card_game.rl.batched_duel import native_soa_cpu
        from app.modules.card_game.rl.batched_duel.native_array_cpu import compile_host_native_library
        from scripts.validate_duel_v2_native_soa import check_aliases
        options = dict(limits=VmLimits(frames=64, locals=128, stack=128, cells=64, steps=100000),
                       max_nodes=8192, max_edges=16384, max_payload_bytes=65536)
        # Every host compiler invocation in this test has its own short deadline.
        def bounded_compile(source, directory):
            return compile_host_native_library(source, directory, timeout=30)
        plan = native_array_plan(self.image)
        compiled = compile_native_values(self.image, self.schema, plan)
        covered = {row['op'] for _, _, begin, end in compiled.regions
                   for row in plan.coverage['instructions'][begin:end]}
        self.assertTrue({'LOAD_DEREF', 'STORE_DEREF'} <= covered)
        self.assertIn('native_value_cell(c,f,', compiled.source)
        self.assertNotIn('local_cell_', compiled.source)
        initial = states()
        initial[1]['right'] = {'hp': 23}
        def compare(left, right):
            np.testing.assert_array_equal(left.statuses, right.statuses)
            np.testing.assert_array_equal(left.outputs, right.outputs)
            np.testing.assert_array_equal(left.roots, right.roots)
            for key in left.heap:
                np.testing.assert_array_equal(left.heap[key], right.heap[key], err_msg=key)
            for key in left.state.buffers:
                np.testing.assert_array_equal(left.state.buffers[key], right.state.buffers[key], err_msg=key)
            self.assertEqual(unpack_game_states(left.state), unpack_game_states(right.state))
            if not left.statuses[:, 0].any():
                a, b = left.decode(), right.decode()
                self.assertEqual(a, b)
                check_aliases(a, b)
        with patch.object(native_soa_cpu, 'compile_host_native_library', bounded_compile), \
             NativeSoACPU(self.image, self.schema, **options) as baseline, \
             NativeSoACPU(self.image, self.schema, native_values=True, native_functions=True, **options) as native:
            for formal, extras in ((closure_cell_reads, [-2, 0, 3]),
                                   (closure_unbound_reads, [1, 2, 3]),
                                   (closure_unbound_reads, [0, 0, 0]),
                                   (closure_caught_missing, [1, 2, 3])):
                entry = formal.__name__
                args = [[None, n] for n in extras]
                complete = baseline.call(entry, pack_game_states(initial, self.schema), args)
                maximum = int(complete.statuses[:, 3].max())
                for steps in (*range(1, maximum + 2), 100000):
                    with self.subTest(entry=entry, steps=steps, extras=extras):
                        left = baseline.call(entry, pack_game_states(initial, self.schema), args, steps=steps)
                        right = native.call(entry, pack_game_states(initial, self.schema), args, steps=steps)
                        compare(left, right)
                if not complete.statuses[:, 0].any():
                    self.assertEqual(right.decode(), [formal(copy.deepcopy(s), n)
                                                     for s, n in zip(initial, extras)])
            for capacity in (6, 8, 12, 24):
                with patch.object(baseline, 'max_nodes', capacity), \
                     patch.object(native, 'max_nodes', capacity):
                    args = [[None, 3]] * len(initial)
                    compare(baseline.call('closure_cell_reads', pack_game_states(initial, self.schema), args),
                            native.call('closure_cell_reads', pack_game_states(initial, self.schema), args))
            zeros = np.zeros
            def expired_deadlines(shape, *args, **kwargs):
                result = zeros(shape, *args, **kwargs)
                if shape == len(initial) and kwargs.get('dtype') is np.int64:
                    result.fill(1)
                return result
            # The CPU executor has no public deadline argument. Override only its
            # exact deadline-array allocation, leaving all heap/GC arrays intact.
            with patch.object(native_soa_cpu.np, 'zeros', side_effect=expired_deadlines):
                args = [[None, 3]] * len(initial)
                left = baseline.call('closure_cell_reads', pack_game_states(initial, self.schema), args)
                right = native.call('closure_cell_reads', pack_game_states(initial, self.schema), args)
                compare(left, right)
                self.assertTrue((right.statuses[:, 0] == 108).all())
                self.assertTrue((right.statuses[:, 3] == 64).all())

    def test_globals_and_noops_all_budget_cuts(self):
        from app.modules.card_game.rl.batched_duel import native_soa_cpu
        from app.modules.card_game.rl.batched_duel.native_array_cpu import (
            compile_host_native_library, pack_call_arguments)
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_duel.vm_image import _image_identity
        from scripts.validate_duel_v2_native_soa import check_aliases
        options = dict(limits=VmLimits(frames=64, locals=128, stack=128, cells=64, steps=100000),
                       max_nodes=8192, max_edges=16384, max_payload_bytes=65536)
        namespace = {'GLOBAL_REFERENCE': {'hp': 11}, 'replacement': {'hp': 23}, 'delete': False}
        args = [[None, 3, copy.deepcopy(namespace)] for _ in states()]
        layout = GraphLayout(len(args), options['max_nodes'], options['max_edges'], options['max_payload_bytes'])
        _, refs = pack_call_arguments(args, layout)
        self.assertTrue((refs[:, 2] == refs[0, 2]).all())
        # Exercise the canonical dynamic-globals ABI with an ordinary mutable
        # argument dict. No STORE_GLOBAL opcode or CPU runtime callback is added.
        functions = self.image.functions.copy()
        functions[self.image.entries['live_global_reads'], 8] = refs[0, 2]
        image = replace(self.image, functions=functions)
        object.__setattr__(image, 'identity', _image_identity(image))
        image.verify()
        plan = native_array_plan(image)
        compiled = compile_native_values(image, self.schema, plan)
        covered = {r['op'] for _, _, b, e in compiled.regions for r in plan.coverage['instructions'][b:e]}
        self.assertTrue({'LOAD_GLOBAL', 'NOP', 'EXTENDED_ARG'} <= covered)
        self.assertTrue(any(r['op'] == 'LOAD_CONST' and r['arg'] > 255
                            for r in plan.coverage['instructions']))
        def bounded_compile(source, directory):
            return compile_host_native_library(source, directory, timeout=30)
        def compare(left, right):
            for key in ('statuses', 'outputs', 'roots'):
                np.testing.assert_array_equal(getattr(left, key), getattr(right, key), err_msg=key)
            for key in left.heap:
                np.testing.assert_array_equal(left.heap[key], right.heap[key], err_msg=key)
            for key in left.state.buffers:
                np.testing.assert_array_equal(left.state.buffers[key], right.state.buffers[key], err_msg=key)
            if not left.statuses[:, 0].any():
                a, b = left.decode(), right.decode()
                self.assertEqual(a, b); check_aliases(a, b)
        with patch.object(native_soa_cpu, 'compile_host_native_library', bounded_compile), \
             NativeSoACPU(image, self.schema, **options) as baseline, \
             NativeSoACPU(image, self.schema, native_values=True, native_functions=True, **options) as native:
            variants = [namespace, dict(namespace, delete=True),
                        {'replacement': {'hp': 23}, 'delete': False}]
            for entry, values in [('live_global_reads', v) for v in variants] + [('large_operand_reads', namespace)]:
                args = [[None, 3, copy.deepcopy(values)] for _ in states()]
                _, current_refs = pack_call_arguments(args, layout)
                self.assertTrue((current_refs[:, 2] == refs[0, 2]).all())
                complete = baseline.call(entry, pack_game_states(states(), self.schema), args)
                maximum = int(complete.statuses[:, 3].max())
                for steps in (*range(1, maximum + 2), 100000):
                    with self.subTest(entry=entry, steps=steps, delete=values.get('delete')):
                        left = baseline.call(entry, pack_game_states(states(), self.schema), args, steps=steps)
                        right = native.call(entry, pack_game_states(states(), self.schema), args, steps=steps)
                        compare(left, right)
                if entry == 'live_global_reads' and not complete.statuses[:, 0].any():
                    for state, value, before, after, actual_globals in right.decode():
                        self.assertEqual((value, before['hp'], after['hp'], state['hp']), (11, 11, 23, 26))
                        self.assertIs(after, actual_globals['GLOBAL_REFERENCE'])
                        self.assertIs(after, actual_globals['replacement'])
                        self.assertIsNot(before, after)
                elif entry == 'live_global_reads':
                    self.assertTrue((complete.statuses[:, 0] == 103).all())
                else:
                    self.assertEqual(right.decode(), [large_operand_fixture()(copy.deepcopy(s), 3, values)
                                                     for s in states()])
            packed, _ = pack_call_arguments(args, layout)
            minimum_nodes = int(packed.counts[:, 0].max())
            for capacity in (minimum_nodes, minimum_nodes + 2, minimum_nodes + 8):
                with patch.object(baseline, 'max_nodes', capacity), patch.object(native, 'max_nodes', capacity):
                    compare(baseline.call('live_global_reads', pack_game_states(states(), self.schema), args),
                            native.call('live_global_reads', pack_game_states(states(), self.schema), args))
            zeros = np.zeros
            def expired_deadlines(shape, *args, **kwargs):
                value = zeros(shape, *args, **kwargs)
                if shape == len(states()) and kwargs.get('dtype') is np.int64:
                    value.fill(1)
                return value
            with patch.object(native_soa_cpu.np, 'zeros', side_effect=expired_deadlines):
                left = baseline.call('large_operand_reads', pack_game_states(states(), self.schema), args)
                right = native.call('large_operand_reads', pack_game_states(states(), self.schema), args)
                compare(left, right)
                self.assertTrue((right.statuses[:, 0] == 108).all())
                self.assertTrue((right.statuses[:, 3] == 64).all())

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
