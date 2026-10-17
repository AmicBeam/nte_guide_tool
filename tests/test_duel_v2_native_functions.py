"""Whole-function native CFG parity, including every logical interruption."""
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
from app.modules.card_game.rl.batched_duel.native_functions import (
    compile_native_functions, integrate_native_functions,
)
from app.modules.card_game.rl.batched_duel.native_soa_cpu import NativeSoACPU
from app.modules.card_game.rl.batched_duel.record_soa import (
    compile_schema, pack_game_states, unpack_game_states,
)
from app.modules.card_game.rl.batched_duel.vm_image import lower_program


def branch_leaf(state, n):
    total = 0
    while n > 0:
        if n & 1:
            total += n * 3
        else:
            total -= n // 2
        state['hp'] += 1
        n -= 1
    state['hp'] += total
    return state, (total or n), state


def lookup_leaf(state, key, n):
    value = state[key]
    if value and n:
        return value / n
    return value if value is not None else -n


def checked_lookup(state, key, n):
    try:
        return lookup_leaf(state, key, n)
    except KeyError as error:
        return 'missing', error.args


def indexed_leaf(state, n):
    state['items'][n] += 1
    return state['items'], state


def leaf_generator(state):
    yield state


def fixture_program():
    from app.modules.card_game.rl.rule_ir.portable_program import (
        FrozenRuleProgram, capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES)
    path = os.environ.get('NTE_NATIVE_FUNCTION_PROGRAM')
    if path:
        return FrozenRuleProgram.from_json(Path(path).read_text(encoding='utf8'))
    if sys.version_info[:2] != (3, 10):
        raise unittest.SkipTest('CPython3.10 frozen fixture required')
    return capture_program({fn.__name__: fn for fn in (
        branch_leaf, lookup_leaf, checked_lookup, indexed_leaf, leaf_generator)},
        scope_character_ids=None, scope_card_ids=None,
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,))


def fixture_schema():
    return compile_schema({
        'Game': dict(kind='record', capacity=8, fields={'hp': 'int', 'items': '@Items'}),
        'Items': dict(kind='list', capacity=32, value='int', max_items=8),
    }, max_strings=64, max_string_bytes=2048)


def states():
    return [{'hp': i + 2, 'items': [1, 3, 5]} for i in range(3)]


CASES = (
    ('branch_leaf', branch_leaf, [[-2], [0], [5]]),
    ('lookup_leaf', lookup_leaf, [['hp', -2], ['hp', 1.5], ['hp', 0]]),
    ('checked_lookup', checked_lookup, [['不存在', 1], ['hp', 2], ['items', 0]]),
    ('indexed_leaf', indexed_leaf, [[-1], [0], [2]]),
)


class NativeFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image, cls.schema = lower_program(fixture_program()), fixture_schema()

    def test_compiles_complete_functions_and_keeps_explicit_barriers(self):
        plan = native_array_plan(self.image)
        compiled = compile_native_functions(plan)
        compiled.verify()
        ids = compiled.function_ids
        for name in ('branch_leaf', 'lookup_leaf', 'indexed_leaf'):
            self.assertTrue(any(':' + name + ':' in fid for fid in ids), name)
        self.assertFalse(any(':checked_lookup:' in fid or ':leaf_generator:' in fid for fid in ids))
        # One switch at entry per function, plus one at function dispatch. No
        # native_step/native_opcode or per-operation PC dispatch in the body.
        self.assertEqual(compiled.source.count('switch(pc)'), len(ids))
        self.assertNotIn('native_step', compiled.source)
        self.assertNotIn('native_opcode', compiled.source)
        self.assertNotIn('native_dispatch_pc', compiled.source)
        self.assertIn('goto admit_', compiled.source)
        self.assertEqual(compiled.instruction_count,
            sum(plan.coverage['functions'][i]['count'] for i in compiled.function_indexes))
        with self.assertRaises(ValueError):
            replace(compiled, instruction_count=0).verify()
        with self.assertRaises(ValueError):
            compile_native_functions(plan, max_function_ops=True)
        self.assertFalse(compile_native_functions(plan, max_function_ops=1).function_ids)
        with self.assertRaises(ValueError):
            integrate_native_functions('changed dispatch', plan)
        from app.modules.card_game.rl.batched_duel.native_soa_source import emit_native_soa_source
        with self.assertRaises(ValueError):
            emit_native_soa_source(self.image, self.schema, VmLimits(),
                                   stack_runs=True, native_functions=True)

    def test_complete_mode_covers_calls_exceptions_and_generators_without_shards(self):
        plan = native_array_plan(self.image)
        complete = compile_native_functions(plan, complete=True, max_function_ops=16384)
        self.assertEqual(complete.function_indexes, tuple(range(len(self.image.functions))))
        self.assertEqual(complete.instruction_count, len(self.image.instructions))
        self.assertEqual(complete.rejected, ())
        compact = compile_native_functions(plan, complete=True, max_function_ops=16384, resume_pcs=())
        self.assertEqual(compact.function_ids, complete.function_ids)
        self.assertLess(compact.source.count(':goto op_'), complete.source.count(':goto op_'))
        from app.modules.card_game.rl.batched_duel.native_soa_source import emit_native_soa_source
        source, _ = emit_native_soa_source(self.image, self.schema, VmLimits(), native_functions=True)
        self.assertNotIn('native_pc_shard_', source)
        self.assertIn('native_function_admit(', source)
        from app.modules.card_game.rl.batched_duel.native_soa_source import execute_only_source, _kernel
        diagnostic = execute_only_source(source)
        self.assertEqual(_kernel(diagnostic, 'vm_soa_execute'), _kernel(source, 'vm_soa_execute'))
        self.assertIn('native_function_dispatch', diagnostic)
        self.assertNotIn('void soa_export_projection(', diagnostic)
        self.assertNotIn('void vm_compact(', diagnostic)
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import CudaSoARules
        diagnostic_backend = object.__new__(CudaSoARules)
        diagnostic_backend.execute_only = True
        with self.assertRaises(RuntimeError):
            diagnostic_backend.metadata(None, None, None)
        with self.assertRaises(RuntimeError):
            diagnostic_backend.export_projection(None)
        with self.assertRaises(ValueError):
            compile_native_functions(plan, complete=True, max_function_ops=1)

    def test_results_mutations_aliases_and_every_step_cut(self):
        options = dict(limits=VmLimits(frames=64, locals=128, stack=128, cells=64, steps=100000),
                       max_nodes=8192, max_edges=16384, max_payload_bytes=65536)
        with NativeSoACPU(self.image, self.schema, **options) as baseline, \
             NativeSoACPU(self.image, self.schema, native_functions=True, **options) as native:
            for entry, formal, extras in CASES:
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
                        decoded = right.decode()
                        self.assertEqual(left.decode(), decoded)
                        self.assertEqual(decoded, [formal(copy.deepcopy(s), *extra)
                                                  for s, extra in zip(states(), extras)])
                        if entry == 'branch_leaf':
                            self.assertTrue(all(result[0] is result[2] for result in decoded))
            # Error sites after partially modifying a typed container must
            # retain identical error kind, PC, steps and state.
            for entry, extras in (('indexed_leaf', [[3], [-4], [100]]),
                                  ('lookup_leaf', [['缺失', 1]] * 3)):
                args = [[None, *extra] for extra in extras]
                left = baseline.call(entry, pack_game_states(states(), self.schema), args)
                right = native.call(entry, pack_game_states(states(), self.schema), args)
                np.testing.assert_array_equal(left.statuses, right.statuses)
                self.assertTrue(right.statuses[:, 0].all())
                self.assertEqual(unpack_game_states(left.state), unpack_game_states(right.state))

    def test_actual_cuda_semantics_and_deadline(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.native_soa_cuda_query import NativeSoACUDAQuery
        options = dict(limits=VmLimits(steps=100000), seconds=10,
                       max_nodes=8192, max_edges=16384, max_payload_bytes=65536)
        with NativeSoACUDAQuery(self.image, self.schema, **options) as baseline, \
             NativeSoACUDAQuery(self.image, self.schema, native_functions=True, **options) as native:
            for entry, formal, extras in CASES:
                args = [[None, *extra] for extra in extras]
                left = baseline.call(entry, pack_game_states(states(), self.schema), args)
                right = native.call(entry, pack_game_states(states(), self.schema), args)
                np.testing.assert_array_equal(left.statuses, right.statuses)
                self.assertEqual(left.decode(), right.decode())
                self.assertEqual(right.decode(), [formal(copy.deepcopy(s), *extra)
                                                 for s, extra in zip(states(), extras)])
            with patch.object(native.executor, 'clock_ns', return_value=1), \
                 patch.object(native, 'seconds', 1e-9):
                result = native.call('branch_leaf', pack_game_states(states(), self.schema), [[None, 512]] * 3)
                self.assertTrue((result.statuses[:, 0] == 108).all())
                self.assertTrue((result.statuses[:, 3] == 64).all())


if __name__ == '__main__':
    unittest.main()
