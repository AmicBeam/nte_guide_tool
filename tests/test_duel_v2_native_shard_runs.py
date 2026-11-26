"""Native shard continuations preserve semantics and every logical budget cut."""
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
from app.modules.card_game.rl.batched_duel.native_soa_cpu import NativeSoACPU
from app.modules.card_game.rl.batched_duel.native_shard_runs import run_native_shards
from app.modules.card_game.rl.batched_duel.record_soa import compile_schema, pack_game_states, unpack_game_states
from app.modules.card_game.rl.batched_duel.vm_image import lower_program


def handler_mix(state, n):
    values = [1, 2, *range(max(0, n))]
    counts = Counter(values)
    counts[1] += 2
    indexed = [i + v for i, v in enumerate(values)]
    merged = {'x': 1, **{'y': 2}}
    try:
        if n < 0:
            raise ValueError('negative')
        result = n + counts[1]
    except ValueError:
        result = -n
    state['hp'] += result
    return state, indexed, f'{n:+d}', list(counts.items()), merged


def routing_loop(state, n):
    total = 0
    for i in range(n):
        total = total + (i * 3) // 2
        if i & 1:
            state['hp'] += 1
    return state, total


def fixture_program():
    from app.modules.card_game.rl.rule_ir.portable_program import (
        FrozenRuleProgram, capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES)
    path = os.environ.get('NTE_NATIVE_SHARD_PROGRAM')
    if path:
        return FrozenRuleProgram.from_json(Path(path).read_text(encoding='utf8'))
    if sys.version_info[:2] != (3, 10):
        raise unittest.SkipTest('CPython3.10 frozen fixture required')
    return capture_program({'handler_mix': handler_mix, 'routing_loop': routing_loop},
        scope_character_ids=None, scope_card_ids=None,
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,))


def fixture_schema():
    return compile_schema({'Game': dict(kind='record', capacity=8, fields={'hp': 'int'})},
                          max_strings=64, max_string_bytes=2048)


def baseline_routing():
    return patch('app.modules.card_game.rl.batched_duel.native_shard_runs.run_native_shards',
                 side_effect=lambda source, plan: source)


class NativeShardRunTests(unittest.TestCase):
    def test_source_fails_closed_on_changed_shards(self):
        from app.modules.card_game.rl.batched_duel.native_array_codegen import emit_native_array_source, native_array_plan
        image = lower_program(fixture_program())
        source = emit_native_array_source(image, direct_stack_dispatch=True)
        plan = native_array_plan(image)
        result = run_native_shards(source, plan)
        self.assertEqual(result.count('/* native shard continuation */'), plan.coverage['shard_count'])
        start = source.index('__device__ inline void native_pc_shard_0(')
        tail = source.index('__device__ inline void native_dispatch_pc(')
        self.assertEqual(result[:start], source[:start])
        self.assertTrue(result.endswith(source[tail:]))
        with self.assertRaises(ValueError):
            run_native_shards(source.replace(' switch(pc){', ' switch(pc) {', 1), plan)
        from app.modules.card_game.rl.batched_duel.native_shard_runs import _continued_case
        with self.assertRaises(ValueError):
            _continued_case('  case 1:{if(c->steps<0)return;push(c,f,1);return;}\n')

    def test_native_outputs_aliases_exceptions_and_all_step_cuts(self):
        image, schema = lower_program(fixture_program()), fixture_schema()
        options = dict(limits=VmLimits(frames=64, locals=128, stack=128, cells=64, steps=100000),
                       max_nodes=8192, max_edges=16384, max_payload_bytes=65536)
        with baseline_routing():
            plain = NativeSoACPU(image, schema, **options)
        with plain, NativeSoACPU(image, schema, **options) as routed:
            for entry, formal, numbers in (
                ('handler_mix', handler_mix, [-2, 0, 3]),
                ('routing_loop', routing_loop, [0, 3, 8]),
            ):
                states = [{'hp': i + 2} for i in range(3)]
                complete = plain.call(entry, pack_game_states(states, schema), [[None, n] for n in numbers])
                complete.check()
                maximum = int(complete.statuses[:, 3].max())
                for steps in (*range(1, maximum + 2), 100000):
                    left = plain.call(entry, pack_game_states(states, schema), [[None, n] for n in numbers], steps=steps)
                    right = routed.call(entry, pack_game_states(states, schema), [[None, n] for n in numbers], steps=steps)
                    np.testing.assert_array_equal(left.statuses, right.statuses, err_msg=(entry, steps))
                    self.assertEqual(unpack_game_states(left.state), unpack_game_states(right.state))
                    if not left.statuses[:, 0].any():
                        self.assertEqual(left.decode(), right.decode())
                        self.assertEqual(right.decode(), [formal(copy.deepcopy(s), n) for s, n in zip(states, numbers)])

    def test_actual_cuda_handler_semantics(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.native_soa_cuda_query import NativeSoACUDAQuery
        image, schema = lower_program(fixture_program()), fixture_schema()
        with NativeSoACUDAQuery(image, schema, limits=VmLimits(steps=100000), seconds=15,
                max_nodes=8192, max_edges=16384, max_payload_bytes=65536) as backend:
            for entry, formal, numbers in (
                ('handler_mix', handler_mix, [-2, 0, 3]),
                ('routing_loop', routing_loop, [0, 3, 8]),
            ):
                states = [{'hp': i + 2} for i in range(3)]
                actual = backend.call(entry, pack_game_states(states, schema), [[None, n] for n in numbers]).decode()
                self.assertEqual(actual, [formal(copy.deepcopy(s), n) for s, n in zip(states, numbers)])
            original = backend.limits
            for steps in (1, 63, 64, 127, 128, 129):
                limits = replace(original, steps=steps)
                with patch.object(backend, 'limits', limits), patch.object(backend.executor, 'limits', limits):
                    result = backend.call('routing_loop', pack_game_states([{'hp': 2}], schema), [[None, 512]])
                    self.assertEqual(result.statuses[0, 0], 107)
                    self.assertEqual(result.statuses[0, 3], steps + 1)
            with patch.object(backend.executor, 'clock_ns', return_value=1), patch.object(backend, 'seconds', 1e-9):
                expired = backend.call('routing_loop', pack_game_states([{'hp': 2}], schema), [[None, 512]])
                self.assertEqual(expired.statuses[0, 0], 108)
                self.assertEqual(expired.statuses[0, 3], 64)


if __name__ == '__main__':
    unittest.main()
