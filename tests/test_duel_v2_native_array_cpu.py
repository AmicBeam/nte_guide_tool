"""CPU native-array executor: same compiled CUDA semantic source, host clang++/g++."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
from app.modules.card_game.rl.batched_duel.native_array_cpu import (
    NativeArrayCPU,
    NativeArrayCapabilityError,
    NativeArrayExecutionError,
    pack_call_arguments,
    prepare_host_cuda_source,
)
from app.modules.card_game.rl.batched_duel.schema import GraphLayout
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from app.modules.card_game.rl.rule_ir.portable_program import (
    DEFAULT_ALLOWED_MODULE_PREFIXES,
    capture_program,
)


def arithmetic(value, offset=2):
    if value < 0:
        return -value
    return value + offset


def collections_probe(value):
    doubled = [item * 2 for item in value]
    a, b = (7, 8)
    mapping = {**{'a': 1}, 'a': a, 'b': b}
    if doubled:
        doubled[-1] = 33
        del doubled[0]
    members = {1, 1, 2}
    return {
        'list': doubled,
        'tuple': (a, b),
        'map': mapping,
        'members': (2 in members, 3 in members),
        'alias': [value, value],
    }


def entity_echo(value):
    value['unknown']['trace'][0] = 9
    return {'entities': [value, value], 'unknown': value['unknown']}


def exception_probe(mode):
    trace = []
    try:
        if mode == 0:
            {}['absent']
        if mode == 1:
            raise ValueError('bad')
        if mode == 2:
            try:
                raise ValueError('first')
            except ValueError as original:
                raise KeyError('second') from original
        if mode == 3:
            [][4]
        return 'ok', trace
    except KeyError as caught:
        return (
            'key',
            caught.args,
            caught.__cause__.args if caught.__cause__ else None,
            caught.__context__.args if caught.__context__ else None,
            trace,
        )
    except (ValueError, IndexError) as caught:
        return 'other', caught.args, trace
    finally:
        trace.append('finally')


def _own_program(entries):
    return capture_program(
        entries,
        scope_character_ids=None,
        scope_card_ids=None,
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,),
    )


class NativeArrayCPUImportTest(unittest.TestCase):
    def test_import_does_not_compile_or_load_torch(self):
        code = (
            "import sys\n"
            "from app.modules.card_game.rl.batched_duel import native_array_cpu as m\n"
            "assert 'torch' not in sys.modules\n"
            "assert m.NativeArrayCPU.interpreter_fallback is False\n"
            "assert m.NativeArrayCPU.runtime_instruction_fetch is False\n"
            "assert m.NativeArrayCPU.supports_full_duel_rules is False\n"
            "assert 'emit_vm_source' not in open(m.__file__, encoding='utf-8').read()\n"
        )
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)

    def test_limits_and_capacities_reject_bool_or_nonpositive(self):
        for field in ('frames', 'locals', 'stack', 'cells', 'steps'):
            with self.assertRaises(ValueError):
                VmLimits(**{field: True})
            with self.assertRaises(ValueError):
                VmLimits(**{field: 0})
        image = lower_program(_own_program({'add': arithmetic}))
        for name in ('max_nodes', 'max_edges', 'max_payload_bytes'):
            with self.assertRaises(ValueError):
                NativeArrayCPU(image, **{name: True})
            with self.assertRaises(ValueError):
                NativeArrayCPU(image, **{name: 0})


class NativeArrayCPUAbiTest(unittest.TestCase):
    def test_native_source_keeps_vm_execute_compact_abi_without_fetch(self):
        from app.modules.card_game.rl.batched_duel.cuda_vm import emit_vm_source
        from app.modules.card_game.rl.batched_duel.native_array_codegen import emit_native_array_source

        image = lower_program(_own_program({'add': arithmetic}))
        interpreter = emit_vm_source(image)
        native = emit_native_array_source(image)
        for source in (interpreter, native):
            self.assertIn('extern "C" __global__ void vm_execute', source)
            self.assertIn('extern "C" __global__ void vm_compact', source)
        self.assertIn('c->instructions+pc*4', interpreter)
        self.assertNotIn('c->instructions+pc*4', native)
        self.assertIn('native_dispatch_pc', native)
        host = prepare_host_cuda_source(native)
        self.assertIn('native_array_clock_ns', host)
        self.assertNotIn('globaltimer', host)
        self.assertIn('native_array_vm_execute', host)
        self.assertIn('native_array_vm_compact', host)

    def test_missing_generator_rejects_interpreter_fallback(self):
        image = lower_program(_own_program({'add': arithmetic}))
        import builtins
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.endswith('native_array_codegen') or name == 'app.modules.card_game.rl.batched_duel.native_array_codegen':
                raise ImportError('forced missing generator')
            return real_import(name, *args, **kwargs)

        with patch('builtins.__import__', blocked):
            with self.assertRaises(NativeArrayCapabilityError) as ctx:
                NativeArrayCPU(image, max_nodes=64, max_edges=64, max_payload_bytes=1024)
        self.assertIn('interpreter fallback', str(ctx.exception).lower())


class NativeArrayCPUExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        program = _own_program({
            'add': arithmetic,
            'collections': collections_probe,
            'echo': entity_echo,
            'exceptions': exception_probe,
        })
        cls.image = lower_program(program)
        cls.vm = NativeArrayCPU(
            cls.image,
            limits=VmLimits(steps=2_000_000),
            max_nodes=8192,
            max_edges=16384,
            max_payload_bytes=131072,
        )

    @classmethod
    def tearDownClass(cls):
        cls.vm.close()

    def test_scalar_call_compacts_constants_and_preserves_ingress(self):
        values = [0, 4, -7, 11]
        arguments = [[value] for value in values]
        packed, refs = pack_call_arguments(arguments, GraphLayout(len(values), 256, 256, 4096))
        before = packed.nodes.copy()
        result = self.vm.run_packed('add', packed, refs)
        self.assertTrue(np.array_equal(before, packed.nodes))
        self.assertEqual(result.statuses.shape, (len(values), 4))
        self.assertTrue(np.all(result.statuses[:, 0] == 0))
        self.assertTrue(np.all(result.compact_status == 0))
        decoded = result.decode()
        self.assertEqual(decoded, [arithmetic(value) for value in values])
        self.assertEqual(self.vm.call('add', arguments), decoded)
        self.assertTrue(self.vm.capability('add')['ready'])
        self.assertFalse(self.vm.capability('add')['gpu_qualified'])
        self.assertFalse(self.vm.capability('add')['interpreter_fallback'])

    def test_argument_refs_are_checked_without_casting_or_truncation(self):
        packed, refs = pack_call_arguments([[4]], GraphLayout(1, 256, 256, 4096))
        before = packed.nodes.copy()
        with self.assertRaisesRegex(ValueError, 'int32'):
            self.vm.run_packed('add', packed, refs.astype(np.float64)+.5)
        bad = refs.copy()
        bad[0, 0] = int(packed.counts[0, 0])
        with self.assertRaisesRegex(ValueError, 'reference'):
            self.vm.run_packed('add', packed, bad)
        bad[0, 0] = -2147483648
        with self.assertRaisesRegex(ValueError, 'reference'):
            self.vm.run_packed('add', packed, bad)
        self.assertTrue(np.array_equal(packed.nodes, before))

    def test_distinct_images_in_one_directory_do_not_reuse_loaded_code(self):
        import tempfile
        from pathlib import Path
        a = lower_program(_own_program({'entry': arithmetic}))
        b = lower_program(_own_program({'entry': collections_probe}))
        with tempfile.TemporaryDirectory(prefix='native_array_images_') as folder:
            with NativeArrayCPU(a, directory=Path(folder), max_nodes=1024, max_edges=2048,
                                max_payload_bytes=8192) as left:
                with NativeArrayCPU(b, directory=Path(folder), max_nodes=1024, max_edges=2048,
                                    max_payload_bytes=8192) as right:
                    self.assertNotEqual(left.source_path, right.source_path)
                    self.assertNotEqual(left.source_sha256, right.source_sha256)
                    self.assertEqual(left.call('entry', [[4]]), [arithmetic(4)])
                    self.assertEqual(right.call('entry', [[[1, 2]]]), [collections_probe([1, 2])])

    def test_list_aliases_and_mutations_are_native(self):
        values = [[], [1], [1, 2, 3, 4, 5], [-3, -2, 0, 7]]
        results = self.vm.call('collections', [[value] for value in values])
        expected = [collections_probe(value) for value in values]
        self.assertEqual(results, expected)
        for result in results:
            self.assertIs(result['alias'][0], result['alias'][1])

    def test_entity_aliases_unknown_fields_and_input_unchanged(self):
        from app.modules.card_game.engine.duel_v2.entities import CardEntity
        value = CardEntity(
            card_id='N01', entity_id='card_17', side='a',
            unknown={'trace': [1, 2], 'future': True},
        )
        result = self.vm.call('echo', [[value]])[0]
        self.assertIsInstance(result['entities'][0], CardEntity)
        self.assertIs(result['entities'][0], result['entities'][1])
        self.assertIs(result['unknown'], result['entities'][0]['unknown'])
        self.assertEqual(result['unknown']['trace'], [9, 2])
        self.assertEqual(value['unknown']['trace'], [1, 2])

    def test_exception_handlers_finally_cause_and_context(self):
        values = list(range(5))
        actual = self.vm.call('exceptions', [[mode] for mode in values])
        self.assertEqual(actual, [exception_probe(mode) for mode in values])

    def test_step_limit_error_includes_code_fid_and_pc(self):
        packed, refs = pack_call_arguments([[4]], GraphLayout(1, 64, 64, 1024))
        result = self.vm.run_packed('add', packed, refs, steps=1)
        self.assertNotEqual(int(result.statuses[0, 0]), 0)
        self.assertIsInstance(result.packed, type(packed))
        with self.assertRaises(NativeArrayExecutionError) as ctx:
            result.check()
        message = str(ctx.exception)
        self.assertIn('status=', message)
        self.assertIn('fid=', message)
        self.assertIn('pc=', message)

    def test_context_manager_closes_executor(self):
        image = lower_program(_own_program({'add': arithmetic}))
        with NativeArrayCPU(image, max_nodes=256, max_edges=256, max_payload_bytes=4096) as vm:
            self.assertEqual(vm.call('add', [[3]]), [5])
            self.assertFalse(vm.closed)
        self.assertTrue(vm.closed)
        with self.assertRaises(RuntimeError):
            vm.call('add', [[1]])


class NativeArrayCPUFormalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.modules.card_game.engine.duel_v2.state import alive, hero, other, team_order
        program = capture_program(
            {'other': other, 'hero': hero, 'alive': alive, 'order': team_order},
            scope_character_ids=None,
            scope_card_ids=None,
        )
        cls.image = lower_program(program)
        cls.vm = NativeArrayCPU(
            cls.image,
            limits=VmLimits(steps=2_000_000),
            max_nodes=8192,
            max_edges=16384,
            max_payload_bytes=131072,
        )

    @classmethod
    def tearDownClass(cls):
        cls.vm.close()

    def test_formal_other_is_native_without_python_callback(self):
        from app.modules.card_game.engine.duel_v2 import state
        with patch.object(state, 'other', side_effect=AssertionError('CPU rule fallback')):
            actual = self.vm.call('other', [['a'], ['b']])
        self.assertEqual(actual, ['b', 'a'])

    def test_formal_hero_alive_and_order_from_packed_game_rows(self):
        from app.modules.card_game.engine.duel_v2 import new_game, state
        rows = []
        expected_hero = []
        expected_alive = []
        expected_order = []
        for seed in (7201, 7202):
            game = new_game(seed=seed, skip_mulligan=True)
            for side in ('a', 'b'):
                cid = next(iter(game['sides'][side]['characters']))
                rows.append([game, side, cid])
                expected_hero.append(state.hero(game, side, cid))
                expected_alive.append(state.alive(game, side, cid))
                expected_order.append(state.team_order(game, side))
        with patch.object(state, 'hero', side_effect=AssertionError('CPU rule fallback')):
            heroes = self.vm.call('hero', rows)
        with patch.object(state, 'alive', side_effect=AssertionError('CPU rule fallback')):
            alives = self.vm.call('alive', rows)
        order_args = [[game, side] for game, side, _cid in rows]
        with patch.object(state, 'team_order', side_effect=AssertionError('CPU rule fallback')):
            orders = self.vm.call('order', order_args)
        self.assertEqual(heroes, expected_hero)
        self.assertEqual(alives, expected_alive)
        self.assertEqual(orders, expected_order)


if __name__ == '__main__':
    unittest.main()
