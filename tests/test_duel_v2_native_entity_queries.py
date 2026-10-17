"""Frozen tuple membership and live entity recovery retain exact VM behavior."""
import copy
import ctypes
from dataclasses import replace
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.content.duel_v2.registry import _ordered_ids
from app.modules.card_game.engine.duel_v2.entities import hydrate_entities
from app.modules.card_game.rl.batched_duel import native_soa_cpu
from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
from app.modules.card_game.rl.batched_duel.native_array_cpu import _cxx_compiler
from app.modules.card_game.rl.batched_duel.native_compile_units import split_native_source
from app.modules.card_game.rl.batched_duel.native_entity_queries import _membership_rows, specialize_entity_queries
from app.modules.card_game.rl.batched_duel.native_runtime_index import index_runtime_attributes
from app.modules.card_game.rl.batched_duel.record_soa import pack_game_states
from app.modules.card_game.rl.batched_duel.vm_image import lower_program


WORDS = ('alpha', 'z', '汉字', '', 'alpha', 'zero', 'nanali')
MIXED = ('alpha', 1, True, None)


def memberships(state, value):
    values = list(WORDS)
    before = value in values
    values.append(value)
    return (value in WORDS, value not in WORDS, value in MIXED,
            value in ('alpha', 'small'), before, value in values)


def dynamic_memberships(state, parts):
    return memberships(state, ''.join(parts))


def frozen_membership(state, value):
    return value in WORDS


def ordered_characters(state, ids):
    characters = {cid: {} for cid in ids}
    before = _ordered_ids(characters)
    del characters[ids[0]]
    characters['dynamic-other'] = {}
    return before, _ordered_ids(characters)


def recover_entities(state):
    original = state['sides']['a']
    hydrate_entities(state)
    return state, original is state['sides']['a']


def program(entries):
    from app.modules.card_game.rl.rule_ir.portable_program import capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES
    return capture_program(entries, scope_character_ids=None, scope_card_ids=None,
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,))


class NativeEntityQueryTests(unittest.TestCase):
    def compile(self, image, source, directory):
        units = split_native_source(source)
        indexed = index_runtime_attributes(units[0], image)
        optimized = specialize_entity_queries(indexed, image)
        self.assertNotEqual(indexed.sha256, optimized.sha256)
        with self.assertRaises(ValueError):
            specialize_entity_queries(units[1], image)
        with self.assertRaises(ValueError):
            specialize_entity_queries(optimized, image)
        with self.assertRaises(ValueError):
            specialize_entity_queries(replace(indexed, source=indexed.source.replace(
                'int n=length(c,owner);for(int i=0;i<n;i++)if(vm_equal(c,needle,edge(c,owner,i)))return 1;return 0;',
                'return 0;', 1)), image)
        paths = []
        for unit in (optimized, *units[1:]):
            path = Path(directory)/(unit.name+'.cpp')
            path.write_text(unit.source)
            paths.append(str(path))
        library = Path(directory)/'entities.dylib'
        result = subprocess.run([_cxx_compiler(), '-std=c++17', '-shared', '-fPIC', '-O1',
            '-fno-fast-math', '-ffp-contract=off', *paths, '-o', str(library)],
            capture_output=True, text=True, timeout=30)
        if result.returncode:
            self.fail(result.stderr[-12000:])
        return ctypes.CDLL(str(library))

    def equal(self, calls):
        np.testing.assert_array_equal(calls[0].statuses, calls[1].statuses)
        np.testing.assert_array_equal(calls[0].outputs, calls[1].outputs)
        for key in ('nodes', 'edges', 'payload', 'counts'):
            np.testing.assert_array_equal(calls[0].heap[key], calls[1].heap[key], err_msg=key)

    def test_dynamic_strings_registry_mutation_and_fallbacks(self):
        from tests.test_duel_v2_native_functions import fixture_schema
        from app.modules.card_game.content.duel_v2.catalog import CHARACTER_ORDER
        image = lower_program(program({'member': memberships, 'dynamic': dynamic_memberships, 'frozen': frozen_membership,
                                       'ordered': ordered_characters}))
        rows = _membership_rows(image)
        self.assertLessEqual(len(rows), 64)
        self.assertLessEqual(sum(len(values) for _, values in rows), 2048)
        def strings(refs):
            result = []
            for ref in refs:
                _, begin, size, _ = map(int, image.nodes[-1-ref])
                result.append(image.payload[begin:begin+size].tobytes().decode('utf8'))
            return set(result)
        self.assertIn(set(CHARACTER_ORDER), [strings(refs) for _, refs in rows])
        options = dict(native_functions=True, native_values=True, limits=VmLimits(steps=100000),
            max_nodes=16384, max_edges=32768, max_payload_bytes=131072)
        original = native_soa_cpu.compile_host_native_library
        with patch.object(native_soa_cpu, 'compile_host_native_library',
                lambda source, directory: original(source, directory, timeout=30)):
            baseline = native_soa_cpu.NativeSoACPU(image, fixture_schema(), **options)
        with baseline, patch.object(native_soa_cpu, 'compile_host_native_library',
                lambda source, directory: self.compile(image, source, directory)):
            with native_soa_cpu.NativeSoACPU(image, fixture_schema(), **options) as optimized:
                cases = [('member', value, memberships) for value in
                    ('alpha', '汉字', '', 'missing', 1, True, None, ['alpha'], {'alpha': 1})]
                cases += [('dynamic', list(value), dynamic_memberships) for value in
                    ('alpha', '汉字', '', 'missing')]
                cases += [('ordered', [CHARACTER_ORDER[-1], 'dynamic-extra', CHARACTER_ORDER[0]], ordered_characters)]
                for entry, value, formal in cases:
                    for steps in (1, 2, 5, 20, 60, 150, 500, 100000):
                        state = dict(hp=2, items=[])
                        calls = [backend.call(entry, pack_game_states([state], backend.schema),
                                 [[None, value]], steps=steps) for backend in (baseline, optimized)]
                        self.equal(calls)
                        if steps == 100000:
                            calls[1].check()
                            self.assertEqual(calls[1].decode(), [formal(state, value)])
                for capacity in (16, 32, 64):
                    with patch.object(baseline, 'max_nodes', capacity), patch.object(optimized, 'max_nodes', capacity):
                        calls = [backend.call('ordered', pack_game_states([dict(hp=2, items=[])], backend.schema),
                            [[None, [CHARACTER_ORDER[-1], 'extra', CHARACTER_ORDER[0]]]])
                            for backend in (baseline, optimized)]
                        self.equal(calls)
                        if capacity == 16:
                            self.assertNotEqual(int(calls[1].statuses[0, 0]), 0)
                original_pack = native_soa_cpu.pack_call_arguments
                for invalid_ref in (20000, -len(image.nodes)-17):
                    def corrupt_arguments(arguments, layout):
                        packed, refs = original_pack(arguments, layout)
                        refs = refs.copy()
                        refs[0, 1] = invalid_ref
                        return packed, refs
                    with patch.object(native_soa_cpu, 'pack_call_arguments', corrupt_arguments):
                        calls = [backend.call('frozen', pack_game_states([dict(hp=2, items=[])], backend.schema),
                            [[None, 'alpha']]) for backend in (baseline, optimized)]
                        self.equal(calls)
                        self.assertNotEqual(int(calls[1].statuses[0, 0]), 0)

    def test_real_hydration_preserves_aliases_live_updates_and_conflict_errors(self):
        from tests.test_duel_v2_native_rule_stages import fixture_cases
        from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
        image = lower_program(program({'recover': recover_entities}))
        schema = compile_execution_schema()
        options = dict(native_functions=True, native_values=True, limits=VmLimits(steps=100000),
            max_nodes=16384, max_edges=32768, max_payload_bytes=131072)
        original = native_soa_cpu.compile_host_native_library
        with patch.object(native_soa_cpu, 'compile_host_native_library',
                lambda source, directory: original(source, directory, timeout=30)):
            baseline = native_soa_cpu.NativeSoACPU(image, schema, **options)
        with baseline, patch.object(native_soa_cpu, 'compile_host_native_library',
                lambda source, directory: self.compile(image, source, directory)):
            with native_soa_cpu.NativeSoACPU(image, schema, **options) as optimized:
                states = [case['state'] for case in fixture_cases()]
                changed = copy.deepcopy(states[0])
                changed['sides']['a']['side'] = 'b'
                states.append(changed)
                duplicate = copy.deepcopy(states[0])
                duplicate['sides']['b']['entity_id'] = duplicate['sides']['a']['entity_id']
                states.append(duplicate)
                for index, state in enumerate(states):
                    for steps in (1, 20, 100, 500, 1500, 100000):
                        calls = [backend.call('recover', pack_game_states([state], schema),
                                 [[None]], steps=steps) for backend in (baseline, optimized)]
                        self.equal(calls)
                        if steps == 100000:
                            if index == 3:
                                self.assertNotEqual(int(calls[1].statuses[0, 0]), 0)
                            else:
                                calls[1].check()
                                self.assertEqual(calls[1].decode(), [recover_entities(copy.deepcopy(state))])


if __name__ == '__main__':
    unittest.main()
