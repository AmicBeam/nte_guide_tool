"""Optional runtime primitive dispatch: same refs, errors and heap budgets."""
import ctypes
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.rl.batched_duel import native_soa_cpu
from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
from app.modules.card_game.rl.batched_duel.native_array_cpu import _cxx_compiler
from app.modules.card_game.rl.batched_duel.native_compile_units import split_native_source
from app.modules.card_game.rl.batched_duel.native_runtime_compare import specialize_runtime_compare
from app.modules.card_game.rl.batched_duel.record_soa import pack_game_states
from app.modules.card_game.rl.batched_duel.vm_image import lower_program


def comparisons(state, a, b):
    return state, (a < b, a <= b, a == b, a != b, a > b, a >= b), state


def equality(state, a, b):
    return state, a == b, a != b


def allocated_compare(state, a, b):
    return [a < b] * 16


class Marker:
    pass


def complex_pairs(state, kind):
    if kind == 0:
        a, b = Decimal('1.25'), Decimal('2.5')
        return a < b, a <= b, a == b, a != b, a > b, a >= b
    a, b = Marker(), Marker()
    return a == a, a == b, a != b


def fixture_program():
    from app.modules.card_game.rl.rule_ir.portable_program import capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES
    return capture_program({'comparisons': comparisons, 'equal': equality,
        'allocated': allocated_compare, 'complex': complex_pairs},
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,),
        scope_character_ids=(), scope_card_ids=())


class NativeRuntimeCompareTests(unittest.TestCase):
    def compiler(self, source, directory):
        units = split_native_source(source)
        candidate = specialize_runtime_compare(units[0])
        self.assertNotEqual(units[0].sha256, candidate.sha256)
        for before, after in zip(units[1:], (candidate, *units[1:])[1:]):
            self.assertEqual(before.source, after.source)
            self.assertEqual(before.sha256, after.sha256)
        with self.assertRaises(ValueError): specialize_runtime_compare(units[1])
        with self.assertRaises(ValueError): specialize_runtime_compare(candidate)
        with self.assertRaises(ValueError): specialize_runtime_compare(replace(units[0],
            source=units[0].source.replace('if(mode==2 || mode==3){int eq=vm_equal',
                'if(mode==2 || mode==4){int eq=vm_equal', 1)))
        paths = []
        for unit in (candidate, *units[1:]):
            path = Path(directory)/(unit.name+'.cpp')
            path.write_text(unit.source);paths.append(str(path))
        library = Path(directory)/'compare.dylib'
        result = subprocess.run([_cxx_compiler(), '-std=c++17', '-shared', '-fPIC', '-O1',
            '-fno-fast-math', '-ffp-contract=off', *paths, '-o', str(library)],
            capture_output=True, text=True, timeout=30)
        if result.returncode:self.fail(result.stderr[-12000:])
        return ctypes.CDLL(str(library))

    def assert_calls_equal(self, left, right):
        np.testing.assert_array_equal(left.statuses, right.statuses)
        np.testing.assert_array_equal(left.outputs, right.outputs)
        for key in ('nodes', 'edges', 'payload', 'counts'):
            np.testing.assert_array_equal(left.heap[key], right.heap[key], err_msg=key)
        for key in left.state.buffers:
            np.testing.assert_array_equal(left.state.buffers[key], right.state.buffers[key])
        if not left.statuses[:, 0].any():
            self.assertEqual(left.decode(), right.decode())

    def test_primitives_fallbacks_bad_refs_steps_and_allocations(self):
        from tests.test_duel_v2_native_functions import fixture_schema
        schema = fixture_schema();image = lower_program(fixture_program())
        options = dict(native_functions=True, native_values=False, limits=VmLimits(steps=100000),
            max_nodes=256, max_edges=256, max_payload_bytes=2048)
        compile_original = native_soa_cpu.compile_host_native_library
        with patch.object(native_soa_cpu, 'compile_host_native_library',
                lambda source, directory: compile_original(source, directory, timeout=30)):
            baseline = native_soa_cpu.NativeSoACPU(image, schema, **options)
        with baseline, patch.object(native_soa_cpu, 'compile_host_native_library', self.compiler):
            with native_soa_cpu.NativeSoACPU(image, schema, **options) as candidate:
                state = dict(hp=1, items=[])
                pairs = [(False, 0), (True, 1), (True, -1), (0, 0),
                    (2**53+1, 2**53), (2**63-1, -2**63), (-2**63, -2**63),
                    (2**80, 2**80), (-2**80, 2**80), (1.5, 2),
                    (2**53+1, float(2**53)),
                    ('', ''), ('汉字🙂', '汉字😃'), ('a\x00b', 'a\x00c')]
                for a, b in pairs:
                    for steps in (1, 4, 10, 20, 40, 80, 100000):
                        calls = [backend.call('comparisons', pack_game_states([state], schema),
                            [[None, a, b]], steps=steps) for backend in (baseline, candidate)]
                        self.assert_calls_equal(*calls)
                        if steps == 100000 and not calls[0].statuses[:, 0].any():
                            row = calls[1].decode()[0]
                            self.assertEqual(row, comparisons(state, a, b))
                            self.assertIs(row[0], row[2])
                for entry, args in [('equal', [None, None, None]), ('equal', [None, [1], [1]]),
                        ('equal', [None, 2**80, 2**80]), ('equal', [None, -2**80, 2**80]),
                        ('equal', [None, {'a': 1}, {'a': 1}]),
                        ('complex', [None, 0]), ('complex', [None, 1]), ('allocated', [None, 1, 2])]:
                    for steps in (1, 10, 40, 100000):
                        calls = [backend.call(entry, pack_game_states([state], schema), [args], steps=steps)
                            for backend in (baseline, candidate)]
                        self.assert_calls_equal(*calls)
                for bad_ref in (-2147483648, -9999999, 9999999):
                    calls = []
                    for backend in (baseline, candidate):
                        execute = backend._execute
                        def inject(*args):
                            ctypes.cast(args[4], ctypes.POINTER(ctypes.c_int32))[1] = bad_ref
                            execute(*args)
                        with patch.object(backend, '_execute', inject):
                            calls.append(backend.call('comparisons', pack_game_states([state], schema), [[None, 1, 2]]))
                    self.assert_calls_equal(*calls)
                    self.assertNotEqual(int(calls[0].statuses[0, 0]), 0,
                        (bad_ref,calls[0].statuses.tolist(),calls[0].heap['counts'].tolist()))
                # Canonical numeric reads have an explicit invalid-payload
                # error. String payload corruption has no safe canonical
                # error contract, so do not manufacture such native pointers.
                for entry, a, b in (('comparisons',1,2),):
                    calls=[]
                    for backend in (baseline,candidate):
                        execute=backend._execute
                        def broken_payload(*args):
                            refs=ctypes.cast(args[4],ctypes.POINTER(ctypes.c_int32))
                            nodes=ctypes.cast(args[0],ctypes.POINTER(ctypes.c_int32))
                            nodes[refs[1]*4+1]=backend.max_payload_bytes+1
                            execute(*args)
                        with patch.object(backend,'_execute',broken_payload):
                            calls.append(backend.call(entry,pack_game_states([state],schema),[[None,a,b]]))
                    self.assert_calls_equal(*calls)
                    self.assertNotEqual(int(calls[0].statuses[0,0]),0)
                # Identical compilation is reused; only runtime capacity bounds
                # shrink. The output list grows beyond the shared edge budget.
                baseline.max_edges = candidate.max_edges = 16
                calls = [backend.call('allocated', pack_game_states([state], schema), [[None, 1, 2]])
                    for backend in (baseline, candidate)]
                self.assert_calls_equal(*calls)
                self.assertNotEqual(int(calls[0].statuses[0, 0]), 0)


if __name__ == '__main__': unittest.main()
