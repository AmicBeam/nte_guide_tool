"""Frozen query-name dispatch retains allocations, mutations and VM budgets."""
import ctypes
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
from app.modules.card_game.rl.batched_duel.native_effect_queries import specialize_effect_queries
from app.modules.card_game.rl.batched_duel.record_soa import pack_game_states
from app.modules.card_game.rl.batched_duel.vm_image import lower_program


def query_methods(state, parts):
    # Exercise empty/present/missing effects, live nested mutation, dynamic
    # getattr fallback, and methods resolved after the first candidate.
    holder = {'effects': [], 'flags': {'atk_buff': state['hp']}}
    before = holder.get('effects', [])
    holder.setdefault('effects', []).append({'value': 3})
    dynamic = getattr(holder, ''.join(parts), None)
    values = list(holder.values())
    copied = holder.copy()
    flags = copied['flags']
    initial = flags.pop('atk_buff')
    flags.update({'next_attack': initial+1})
    items = list(flags.items())
    keys = list(flags.keys())
    sequence = [1, 2, 1]
    sequence.extend([4])
    sequence.insert(1, 3)
    sequence.remove(2)
    sequence.reverse()
    duplicated = sequence.copy()
    last = duplicated.pop()
    position = sequence.index(1)
    count = sequence.count(1)
    tpl = (4, 1, 4)
    return (holder, before, values, items, keys, last, position, count,
            tpl.index(1), tpl.count(4), dynamic('effects') if dynamic else None,
            before is holder['effects'], copied['flags'] is holder['flags'])


def bad_query_methods(state, mode):
    if mode == 0:
        return {}.get()
    if mode == 1:
        return [].append(1, 2)
    if mode == 2:
        return getattr([], 'get')()
    return (1,).index(7)


def program():
    from app.modules.card_game.engine.duel_v2.modifiers import all_effects
    from app.modules.card_game.rl.rule_ir.portable_program import capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES
    return capture_program({'query': query_methods, 'bad': bad_query_methods, 'effects': all_effects},
        scope_character_ids=None, scope_card_ids=None,
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,))


class NativeEffectQueryTests(unittest.TestCase):
    def compile(self, image, source, directory):
        units = split_native_source(source)
        optimized = specialize_effect_queries(units[0], image)
        self.assertNotEqual(units[0].sha256, optimized.sha256)
        self.assertIn('native_effect_query_names', optimized.source)
        with self.assertRaises(ValueError): specialize_effect_queries(units[1], image)
        with self.assertRaises(ValueError): specialize_effect_queries(optimized, image)
        with self.assertRaises(ValueError):
            specialize_effect_queries(replace(units[0], source=units[0].source.replace(
                'vm_str_equals(c, name, "get", 3)', 'vm_str_equals(c, name, "geX", 3)', 1)), image)
        paths = []
        for unit in (optimized, *units[1:]):
            path = Path(directory)/(unit.name+'.cpp')
            path.write_text(unit.source)
            paths.append(str(path))
        library = Path(directory)/'queries.dylib'
        compiled = subprocess.run([_cxx_compiler(), '-std=c++17', '-shared', '-fPIC', '-O1',
            '-fno-fast-math', '-ffp-contract=off', *paths, '-o', str(library)],
            capture_output=True, text=True, timeout=30)
        if compiled.returncode: self.fail(compiled.stderr[-12000:])
        return ctypes.CDLL(str(library))

    def test_queries_dynamic_fallback_budget_errors_and_allocations(self):
        from tests.test_duel_v2_native_functions import fixture_schema
        image = lower_program(program())
        schema = fixture_schema()
        options = dict(native_functions=True, native_values=True, limits=VmLimits(steps=100000),
            max_nodes=8192, max_edges=16384, max_payload_bytes=65536)
        original = native_soa_cpu.compile_host_native_library
        with patch.object(native_soa_cpu, 'compile_host_native_library',
                lambda source, directory: original(source, directory, timeout=30)):
            baseline = native_soa_cpu.NativeSoACPU(image, schema, **options)
        with baseline, patch.object(native_soa_cpu, 'compile_host_native_library',
                lambda source, directory: self.compile(image, source, directory)):
            with native_soa_cpu.NativeSoACPU(image, schema, **options) as optimized:
                state = dict(hp=2, items=[])
                for parts in (list('get'), list('absent'), []):
                    for steps in (1, 20, 60, 120, 300, 100000):
                        calls = [backend.call('query', pack_game_states([state], schema),
                            [[None, parts]], steps=steps) for backend in (baseline, optimized)]
                        self.assert_calls_equal(calls)
                        if steps == 100000:
                            self.assertEqual(calls[1].decode(), [query_methods(state, parts)])
                for mode in range(4):
                    calls = [backend.call('bad', pack_game_states([state], schema),
                        [[None, mode]]) for backend in (baseline, optimized)]
                    self.assert_calls_equal(calls)
                    self.assertNotEqual(int(calls[1].statuses[0, 0]), 0)
                from app.modules.card_game.engine.duel_v2.modifiers import all_effects
                # Real engine query: same entity identity with changed live
                # flags must never return a previous query's derived effects.
                for flags in ({}, {'atk_buff': 2}, {'atk_debuff': 3, 'next_attack': 2},
                        {'timed_attack': [{'left': 2, 'amount': -1}]},
                        {'damage_immunity_until': 3}):
                    holder = dict(entity_id='a:test', side='a', flags=flags,
                        effects=[dict(property='attack.panel', value=4)])
                    calls = [backend.call('effects', pack_game_states([state], schema),
                        [[holder]], state_positions=()) for backend in (baseline, optimized)]
                    self.assert_calls_equal(calls)
                    self.assertEqual(calls[1].decode(), [all_effects(holder)])

    def assert_calls_equal(self, calls):
        np.testing.assert_array_equal(calls[0].statuses, calls[1].statuses)
        np.testing.assert_array_equal(calls[0].outputs, calls[1].outputs)
        for key in ('nodes', 'edges', 'payload', 'counts'):
            np.testing.assert_array_equal(calls[0].heap[key], calls[1].heap[key])
        for key in calls[0].state.buffers:
            np.testing.assert_array_equal(calls[0].state.buffers[key], calls[1].state.buffers[key])


if __name__ == '__main__': unittest.main()
