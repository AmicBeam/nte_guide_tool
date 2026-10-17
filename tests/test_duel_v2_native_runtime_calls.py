"""Shared-only call routing: scalar/function results and full heap parity."""
import ctypes
import copy
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.rl.batched_duel import native_soa_cpu
from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
from app.modules.card_game.rl.batched_duel.native_array_cpu import _cxx_compiler
from app.modules.card_game.rl.batched_duel.native_compile_units import split_native_source
from app.modules.card_game.rl.batched_duel.native_runtime_calls import transform_runtime_calls
from app.modules.card_game.rl.batched_duel.native_runtime_index import index_runtime_attributes
from app.modules.card_game.rl.batched_duel.native_effect_queries import specialize_effect_queries
from app.modules.card_game.rl.batched_duel.record_soa import pack_game_states
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from app.modules.card_game.engine.duel_v2.entities import copy_state_graph


def scalar_calls(state, value):
    a = int(value)
    b = float(a)
    return state, len(state), bool(a), int(b), float(value), id(state) == id(state)


def scalar_errors(state, value):
    if value == 0:
        return len()
    if value == 1:
        return bool(1, 2)
    if value == 2:
        return int('not an int')
    if value == 3:
        return float('not a float')
    if value == 4:
        return id()
    if value == 5:
        return int(value=1)
    return len(1)


def copied_values(state, value):
    first = deepcopy(value)
    second = deepcopy(value, None)
    copied_state = deepcopy(state)
    graph = copy_state_graph(value)
    return state, first, second, copied_state, graph


def copy_errors(state, value):
    if value == 0:
        return deepcopy()
    if value == 1:
        return deepcopy(state, None, None)
    if value == 2:
        return deepcopy(state, {})
    if value == 3:
        return deepcopy(state, memo=None)
    return deepcopy(x=state)


class GetBase:
    def get(self, key, default=0):
        return key + default


class GetLeft(GetBase):
    def get(self, key, default=0):
        return super().get(key, default) + 10


class GetRight(GetBase):
    def get(self, key, default=0):
        return super().get(key, default) + 20


class GetDerived(GetLeft, GetRight):
    pass


def type_and_mapping_calls(state, value):
    obj = GetDerived()
    bound = state.get
    return (isinstance(obj, GetBase), isinstance(obj, (dict, GetDerived)),
            type(obj) is GetDerived, isinstance(True, int), isinstance(1, bool),
            type(state) is dict, obj.get(value, 2), bound('hp'),
            bound('missing'), bound('missing', value))


def type_and_mapping_errors(state, value):
    if value == 0:
        return type()
    if value == 1:
        return type(state, state)
    if value == 2:
        return isinstance(state)
    if value == 3:
        return isinstance(obj=state, class_or_tuple=dict)
    if value == 4:
        return state.get(key='hp')
    if value == 5:
        return state.get('hp', 1, 2)
    if value == 6:
        return type(object=state)
    return state.get()


class NativeRuntimeCallTests(unittest.TestCase):
    def compiler(self, image, source, directory, *, routes=None, effects=True):
        units = split_native_source(source)
        indexed = index_runtime_attributes(units[0], image)
        original_routes = transform_runtime_calls(indexed, image)
        routes = dict(type_builtins=True, mapping_get=True) if routes is None else routes
        optimized = transform_runtime_calls(indexed, image, **routes)
        self.assertNotEqual(original_routes.sha256, optimized.sha256)
        self.assertEqual('native_runtime_get_done' in optimized.source, routes.get('mapping_get', False))
        self.assertNotIn('native_runtime_get_done', original_routes.source)
        self.assertNotIn('native_runtime_copy_done', original_routes.source)
        self.assertEqual('native_runtime_copy_done' in optimized.source, routes.get('deepcopy_calls', False))
        with self.assertRaises(TypeError):
            transform_runtime_calls(indexed, image, type_builtins=1)
        if effects:
            optimized = specialize_effect_queries(optimized, image)
        self.assertNotEqual(units[0].sha256, optimized.sha256)
        with self.assertRaises(ValueError):
            transform_runtime_calls(units[1], image)
        with self.assertRaises(ValueError):
            transform_runtime_calls(optimized, image)
        with self.assertRaises(ValueError):
            transform_runtime_calls(replace(units[0], source=units[0].source.replace(
                'if(native_fast_call(c,callable,args,nargs,caller,kw_names,kw_values,nkw))return;',
                'if(native_fast_call(c,callable,args,nargs,caller,kw_names,kw_values,nkw)){}')), image)
        from app.modules.card_game.rl.batched_duel.native_runtime_index import _definition
        for name in ('lookup_own', 'lookup_class'):
            self.assertEqual(_definition(units[0].source, name), _definition(optimized.source, name))
        paths = []
        for unit in (optimized, *units[1:]):
            path = Path(directory) / (unit.name + '.cpp')
            path.write_text(unit.source)
            paths.append(str(path))
        for before, after in zip(units[1:], (optimized, *units[1:])[1:]):
            self.assertEqual(before.source, after.source)
            self.assertEqual(before.sha256, after.sha256)
        library = Path(directory) / 'calls.dylib'
        result = subprocess.run([_cxx_compiler(), '-std=c++17', '-shared', '-fPIC', '-O1',
            '-fno-fast-math', '-ffp-contract=off', *paths, '-o', str(library)],
            capture_output=True, text=True, timeout=30)
        if result.returncode:
            self.fail(result.stderr[-12000:])
        return ctypes.CDLL(str(library))

    def pair(self, program, schema, cases, *, golden=None, routes=None, effects=True, bounds=()):
        image = lower_program(program)
        options = dict(native_functions=True, native_values=True, limits=VmLimits(steps=100000),
            max_nodes=16384, max_edges=32768, max_payload_bytes=131072)
        original = native_soa_cpu.compile_host_native_library
        with patch.object(native_soa_cpu, 'compile_host_native_library',
                lambda source, directory: original(source, directory, timeout=30)):
            baseline = native_soa_cpu.NativeSoACPU(image, schema, **options)
        with baseline, patch.object(native_soa_cpu, 'compile_host_native_library',
                lambda source, directory: self.compiler(image, source, directory, routes=routes, effects=effects)):
            with native_soa_cpu.NativeSoACPU(image, schema, **options) as optimized:
                for entry, states, args, positions in cases:
                    for steps in (1, 2, 5, 10, 20, 40, 75, 100, 200, 500, 100000):
                        calls = [backend.call(entry, pack_game_states(states, schema), args,
                                 state_positions=positions, steps=steps) for backend in (baseline, optimized)]
                        np.testing.assert_array_equal(calls[0].statuses, calls[1].statuses)
                        np.testing.assert_array_equal(calls[0].outputs, calls[1].outputs)
                        for key in ('nodes', 'edges', 'payload', 'counts'):
                            np.testing.assert_array_equal(calls[0].heap[key], calls[1].heap[key],
                                                          err_msg=(entry, steps, key))
                        if not calls[0].statuses[:, 0].any():
                            self.assertEqual(calls[0].decode(), calls[1].decode())
                            if entry == 'copied':
                                for row in calls[1].decode():
                                    self.assertIsNot(row[0], row[3])
                                    if isinstance(row[1],dict) and 'a' in row[1]:
                                        self.assertIs(row[1]['a'],row[1]['b'])
                                        self.assertIsNot(row[1],row[2])
                                        self.assertIsNot(row[1],row[4])
                        if steps == 100000 and golden and entry in golden:
                            calls[1].check()
                            self.assertEqual(calls[1].decode(), [golden[entry](copy.deepcopy(s), *a[1:])
                                             for s, a in zip(states, args)])
                for entry, states, args, positions, capacities in bounds:
                    for backend in (baseline, optimized):
                        for name, value in capacities.items():setattr(backend, name, value)
                    calls = [backend.call(entry, pack_game_states(states, schema), args,
                        state_positions=positions) for backend in (baseline, optimized)]
                    np.testing.assert_array_equal(calls[0].statuses, calls[1].statuses)
                    np.testing.assert_array_equal(calls[0].outputs, calls[1].outputs)
                    for key in ('nodes', 'edges', 'payload', 'counts'):
                        np.testing.assert_array_equal(calls[0].heap[key], calls[1].heap[key])
                    self.assertTrue(calls[0].statuses[:, 0].any(), 'Capacity fixture did not hit a bound')

    def test_explicit_deepcopy_routes_scalars_aliases_memo_errors_and_capacity(self):
        from app.modules.card_game.rl.rule_ir.portable_program import capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES
        from tests.test_duel_v2_native_values import fixture_schema, states
        program = capture_program({'copied': copied_values, 'errors': copy_errors},
            scope_character_ids=(), scope_card_ids=(),
            allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,))
        shared=[1,2];payload={'a':shared,'b':shared}
        cases = [('copied', states(), [[None,v]]*3, (0,)) for v in (None,False,2**53+1,'汉字🙂',payload)]
        cases.extend(('errors',states(),[[None,code]]*3,(0,)) for code in range(5))
        self.pair(program,fixture_schema(),cases,
            routes=dict(function_calls=False,scalar_builtins=False,deepcopy_calls=True),effects=False,
            golden={'copied':copied_values},bounds=[
                ('copied',states(),[[None,payload]]*3,(0,),dict(max_nodes=16,max_edges=128,max_payload_bytes=256)),
                ('copied',states(),[[None,payload]]*3,(0,),dict(max_nodes=256,max_edges=16,max_payload_bytes=256))])

    def test_function_keywords_generators_bound_methods_and_errors(self):
        from tests.test_duel_v2_native_calls import fixture_program
        from tests.test_duel_v2_native_values import fixture_schema, states
        self.pair(fixture_program(), fixture_schema(), [
            ('calls', states(), [[None, n] for n in (-2, 0, 3)], (0,)),
            *[('errors', states(), [[None, n]] * 3, (0,)) for n in range(6)]])

    def test_scalar_results_conversion_errors_keywords_and_allocations(self):
        from app.modules.card_game.rl.rule_ir.portable_program import capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES
        from tests.test_duel_v2_native_values import fixture_schema, states
        program = capture_program({'scalars': scalar_calls, 'errors': scalar_errors},
            scope_character_ids=None, scope_card_ids=None,
            allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,))
        self.pair(program, fixture_schema(), [
            ('scalars', states(), [[None, n] for n in ('-3', '0', '42')], (0,)),
            *[('errors', states(), [[None, n]] * 3, (0,)) for n in range(7)]])

    def test_real_rule_helpers_and_entity_aliases(self):
        from tests.test_duel_v2_native_rule_stages import fixture_program, fixture_cases, assert_stage
        from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
        image = lower_program(fixture_program())
        with patch.object(native_soa_cpu, 'compile_host_native_library',
                lambda source, directory: self.compiler(image, source, directory)):
            with native_soa_cpu.NativeSoACPU(image, compile_execution_schema(),
                    native_functions=True, native_values=True, limits=VmLimits(steps=100000),
                    max_nodes=16384, max_edges=32768, max_payload_bytes=131072) as backend:
                assert_stage(backend, fixture_cases())

    def test_exact_type_matching_and_original_get_with_c3_overrides(self):
        from app.modules.card_game.rl.rule_ir.portable_program import capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES
        from tests.test_duel_v2_native_values import fixture_schema, states
        program = capture_program({'types': type_and_mapping_calls, 'errors': type_and_mapping_errors},
            scope_character_ids=None, scope_card_ids=None,
            allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,))
        self.pair(program, fixture_schema(), [
            ('types', states(), [[None, n] for n in (-3, 0, 42)], (0,)),
            *[('errors', states(), [[None, n]] * 3, (0,)) for n in range(8)]],
            golden={'types': type_and_mapping_calls},
            routes=dict(function_calls=False, scalar_builtins=False, type_builtins=True, mapping_get=True),
            effects=False)

    def test_independent_routes_options_and_repeated_processing_rejected(self):
        from itertools import product
        from tests.test_duel_v2_native_calls import fixture_program
        from tests.test_duel_v2_native_values import fixture_schema
        from app.modules.card_game.rl.batched_duel.native_soa_source import emit_native_soa_source
        image = lower_program(fixture_program())
        source, _ = emit_native_soa_source(image, fixture_schema(), VmLimits(),
            native_functions=True, native_values=True)
        indexed = index_runtime_attributes(split_native_source(source)[0], image)
        names = ('function_calls', 'scalar_builtins', 'type_builtins', 'mapping_get')
        for values in product((False, True), repeat=4):
            routes = dict(zip(names, values))
            if not any(values):
                with self.assertRaises(ValueError):
                    transform_runtime_calls(indexed, image, **routes)
                continue
            unit = transform_runtime_calls(indexed, image, **routes)
            for repeated in ({}, routes):
                with self.assertRaises(ValueError):
                    transform_runtime_calls(unit, image, **repeated)
        self.assertEqual(transform_runtime_calls(indexed,image).source,
            transform_runtime_calls(indexed,image,deepcopy_calls=False).source)
        only_copy=transform_runtime_calls(indexed,image,function_calls=False,
            scalar_builtins=False,deepcopy_calls=True)
        self.assertIn('native_runtime_copy_done',only_copy.source)
        with self.assertRaises(ValueError):
            transform_runtime_calls(only_copy,image,deepcopy_calls=True)
        for name in (*names,'deepcopy_calls'):
            for invalid in (0, 1, None, 'true'):
                with self.assertRaises(TypeError):
                    transform_runtime_calls(indexed, image, **{name: invalid})

    def test_class_c3_descriptors_static_methods_and_shadowing(self):
        from tests.test_duel_v2_vm_frozen_attributes import program
        from tests.test_duel_v2_native_functions import fixture_schema
        parts = [list('value'), list('_number'), ['汉', '字'], list('missing'), list('prop')]
        self.pair(program(), fixture_schema(), [('attributes',
            [dict(hp=1, items=[]) for _ in parts], [[p] for p in parts], ())])


if __name__ == '__main__':
    unittest.main()
