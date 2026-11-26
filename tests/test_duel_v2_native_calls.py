"""Canonical call binding versus direct rule-function and method routing."""
import copy
import os
from pathlib import Path
import sys
import unittest

import numpy as np

from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
from app.modules.card_game.rl.batched_duel.native_calls import integrate_native_calls
from app.modules.card_game.rl.batched_duel.native_soa_cpu import NativeSoACPU
from app.modules.card_game.rl.batched_duel.record_soa import pack_game_states, unpack_game_states
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from tests.test_duel_v2_native_values import fixture_schema, states


def rule_leaf(n, value=3, *, extra=1):
    return n + value + extra


def suspended(state, n):
    for i in range(n):
        yield rule_leaf(i, extra=2)


def routed_calls(state, n):
    extra = n
    def callback(x=2):
        return x + extra
    result = rule_leaf(n, extra=callback())
    state['items'].append(result)
    state['items'].insert(0, callback(3))
    state['items'].extend(suspended(state, 2))
    transient = {'hp': n}
    default = transient.setdefault('other', 5)
    transient.update(extra=n)
    values = transient.copy()
    popped = values.pop('hp')
    return state, state.get('hp'), state.get('missing', default), popped, values, list(transient.items())


def routed_errors(state, n):
    if n == 0:
        return state.get()
    if n == 1:
        return state.get('hp', 1, 2)
    if n == 2:
        return state.get(key='hp')
    if n == 3:
        return rule_leaf()
    if n == 4:
        return rule_leaf(1, unknown=3)
    state['items'].pop(100)
    return state


def fixture_program():
    from app.modules.card_game.rl.rule_ir.portable_program import (
        FrozenRuleProgram, capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES)
    path = os.environ.get('NTE_NATIVE_CALLS_PROGRAM')
    if path:
        return FrozenRuleProgram.from_json(Path(path).read_text(encoding='utf8'))
    if sys.version_info[:2] != (3, 10):
        raise unittest.SkipTest('CPython3.10 frozen fixture required')
    return capture_program({'calls': routed_calls, 'errors': routed_errors},
        scope_character_ids=None, scope_card_ids=None,
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,))


class NativeCallTests(unittest.TestCase):
    def test_changed_binding_source_rejected(self):
        with self.assertRaises(ValueError):
            integrate_native_calls('changed call binding source')

    def test_binding_methods_generators_and_every_budget_cut(self):
        image, schema = lower_program(fixture_program()), fixture_schema()
        options = dict(limits=VmLimits(frames=64, locals=128, stack=128, cells=64, steps=100000),
                       max_nodes=8192, max_edges=16384, max_payload_bytes=65536)
        baseline = NativeSoACPU(image, schema, **options)
        with baseline, NativeSoACPU(image, schema, native_values=True, native_functions=True, **options) as native:
            args = [[None, n] for n in (-2, 0, 3)]
            complete = baseline.call('calls', pack_game_states(states(), schema), args)
            complete.check()
            for steps in (*range(1, int(complete.statuses[:, 3].max()) + 2), 100000):
                left = baseline.call('calls', pack_game_states(states(), schema), args, steps=steps)
                right = native.call('calls', pack_game_states(states(), schema), args, steps=steps)
                np.testing.assert_array_equal(left.statuses, right.statuses, err_msg=steps)
                np.testing.assert_array_equal(left.heap['counts'], right.heap['counts'])
                self.assertEqual(unpack_game_states(left.state), unpack_game_states(right.state))
                if not left.statuses[:, 0].any():
                    self.assertEqual(left.decode(), right.decode())
                    self.assertEqual(right.decode(), [routed_calls(copy.deepcopy(s), n)
                        for s, n in zip(states(), (-2, 0, 3))])
            for n in range(6):
                args = [[None, n]] * 3
                left = baseline.call('errors', pack_game_states(states(), schema), args)
                right = native.call('errors', pack_game_states(states(), schema), args)
                np.testing.assert_array_equal(left.statuses, right.statuses, err_msg=n)
                np.testing.assert_array_equal(left.heap['counts'], right.heap['counts'])
                self.assertTrue(right.statuses[:, 0].all())


if __name__ == '__main__':
    unittest.main()
