"""Exact no-cache legacy-effect fast path, including mutable/error boundaries."""
from copy import deepcopy
import inspect
import sys
from unittest.mock import patch
import unittest

from app.modules.card_game.engine.duel_v2 import modifiers
from app.modules.card_game.engine.duel_v2.entities import PlayerEntity, CharacterEntity, CardEntity
from app.modules.card_game.rl.batched_duel.gpu_effect_queries import (
    derive_legacy_effects, query_legacy_effects, EffectQuerySpecializationError,
    derive_all_effects, query_all_effects,
)


def effect_query_packet(holder):
    return holder, modifiers.all_effects(holder), modifiers.compatibility_flags(holder)


class CustomFlags(dict):
    def get(self, key, default=None):
        return 4 if key == 'atk_buff' else super().get(key, default)


class CountingHolder(dict):
    def get(self, key, default=None):
        if key == 'entity_id':
            self['source_reads'] = self.get('source_reads', 0) + 1
        return super().get(key, default)


class RaisingHolder(dict):
    def get(self, key, default=None):
        if key == 'entity_id':
            raise ValueError('source lookup failed')
        return super().get(key, default)


class PolicyEffectQueryTests(unittest.TestCase):
    def test_all_effects_empty_results_fresh_and_live_mutation_no_cache(self):
        for holder_type in (dict, PlayerEntity, CharacterEntity, CardEntity):
            for stored in ([], ()):
                for flags in (None, {}, False, 0, 0.0, [], ()):
                    holder = holder_type(entity_id='a:hero', side='a', effects=stored, flags=flags)
                    expected = modifiers.all_effects(holder)
                    actual = query_all_effects(holder)
                    self.assertEqual(actual, expected)
                    self.assertIsNot(actual, stored)
                    self.assertIsNot(actual, query_all_effects(holder))
            holder = holder_type(entity_id='a:hero', side='a')
            self.assertEqual(query_all_effects(holder), [])
            holder['effects'] = [dict(property='attack.panel', value=3)]
            self.assertIs(query_all_effects(holder)[0], holder['effects'][0])
            holder['effects'][0]['value'] = 7
            self.assertEqual(query_all_effects(holder), modifiers.all_effects(holder))
            holder['effects'].clear()
            holder['flags'] = {'atk_buff': 4}
            self.assertEqual(query_all_effects(holder), modifiers.all_effects(holder))

    def test_all_effects_custom_holder_iterator_flags_order_and_errors(self):
        for holder_type in (CountingHolder, RaisingHolder):
            expected = holder_type(effects=[], flags={'unrelated': True})
            actual = holder_type(effects=[], flags={'unrelated': True})
            if holder_type is RaisingHolder:
                with self.assertRaisesRegex(ValueError, 'source lookup failed'):
                    modifiers.all_effects(expected)
                with self.assertRaisesRegex(ValueError, 'source lookup failed'):
                    query_all_effects(actual)
            else:
                self.assertEqual(query_all_effects(actual), modifiers.all_effects(expected))
                self.assertEqual(actual, expected)
        for function in (modifiers.all_effects, query_all_effects):
            events = []
            class OrderedHolder(dict):
                def get(self, key, default=None):
                    events.append(key)
                    return super().get(key, default)
            class IteratedList(list):
                def __iter__(self):
                    events.append('iterator')
                    return super().__iter__()
            class FlagTruth(dict):
                def __bool__(self):
                    events.append('flags')
                    return False
            self.assertEqual(function(dict(effects=IteratedList(), flags=FlagTruth())), [])
            self.assertEqual(events, ['iterator', 'flags'])
            events.clear()
            self.assertEqual(function(OrderedHolder(effects=IteratedList(), flags={'unrelated': True})), [])
            self.assertEqual(events, ['effects', 'iterator', 'flags', 'entity_id'])
            events.clear()
            with self.assertRaises(TypeError):
                function(dict(effects=None, flags=FlagTruth()))
            self.assertEqual(events, [])
        with patch.dict(query_all_effects.__globals__,
                {'legacy_effects': lambda holder: (_ for _ in ()).throw(AssertionError('fallback reached'))}):
            self.assertEqual(query_all_effects(dict(effects=[], flags={})), [])
            self.assertEqual(query_all_effects(dict(effects=(), flags=None)), [])
            with self.assertRaises(AssertionError): query_all_effects(dict(effects=[], flags=False))
            with self.assertRaises(AssertionError): query_all_effects(dict(effects=[], flags={'atk_buff': 0}))

    def test_all_effects_canonical_shape_change_fails_closed(self):
        lines, start = inspect.getsourcelines(modifiers.all_effects)
        changed = ''.join(lines).replace("holder.get('effects', [])", "holder.get('effects', [1])")
        with patch('inspect.getsourcelines', return_value=(changed.splitlines(True), start)):
            with self.assertRaises(EffectQuerySpecializationError): derive_all_effects()

    def compare(self, holder):
        before = deepcopy(holder)
        expected_holder, actual_holder = deepcopy(holder), deepcopy(holder)
        try:
            expected = modifiers.legacy_effects(expected_holder)
        except Exception as error:
            with self.assertRaises(type(error)) as caught:
                query_legacy_effects(actual_holder)
            self.assertEqual(str(caught.exception), str(error))
        else:
            actual = query_legacy_effects(actual_holder)
            self.assertEqual(actual, expected)
        self.assertEqual(holder, before)
        self.assertEqual(expected_holder, actual_holder)
        self.assertEqual(actual_holder, before)

    def test_no_legacy_inputs_and_unrelated_hidden_mimic_equipment_fields(self):
        for flags in ({}, None, False, [],
                {'mimic': {'id': 'zero'}, 'shape': 'N07', 'weapon': {'id': 'N08'},
                 'awakened': True, 'collapse': {'side': 'b', 'until': 8}},
                {'hidden_hand': ['FF01'], 'secret': {'private': [1, 2]}}):
            self.compare(dict(entity_id='a:hero', side='a', flags=flags))
        self.compare({'entity_id': 'a:hero'})

    def test_every_present_key_zero_nonzero_expiring_and_timed_list(self):
        for key in ('atk_buff', 'atk_debuff', 'form_attack', 'next_attack', 'damage_immunity_until'):
            for value in (0, False, 3, -2, '4', None, []):
                self.compare(dict(entity_id='a:hero', side='a', flags={key: value}))
        for flags in (
                {'atk_buff_expiring': {}}, {'atk_buff_expiring': {'amount': 0}},
                {'atk_buff': 5, 'atk_buff_expiring': {'amount': 2, 'until': 3}},
                {'atk_buff_expiring': {'amount': 2}},
                {'timed_attack': []}, {'timed_attack': None}, {'timed_attack': {}},
                {'timed_attack': [{'left': 3, 'amount': 2}, {'left': 0, 'amount': 8}]},
                {'timed_attack': [{'left': 3}]}, {'timed_attack': ['invalid']},
                {'timed_attack': [{'left': 'invalid', 'amount': 3}]},
                {'damage_immunity_until': 2, 'next_attack': 4}):
            self.compare(dict(entity_id='a:hero', side='a', flags=flags))

    def test_custom_mapping_fallback_and_live_edits_no_cache(self):
        self.compare(dict(entity_id='a:hero', side='a', flags=CustomFlags(other=True)))
        holder = dict(entity_id='a:hero', side='a', flags={'mimic': {'id': 'zero'}})
        self.assertEqual(query_legacy_effects(holder), [])
        holder['flags']['timed_attack'] = [{'left': 2, 'amount': 3}]
        self.compare(holder)
        holder['flags']['timed_attack'][0]['amount'] = 7
        self.compare(holder)
        holder['flags'].clear()
        self.assertEqual(query_legacy_effects(holder), [])
        holder['entity_id'] = 'b:other'
        holder['side'] = 'b'
        holder['flags']['atk_buff'] = 3
        self.compare(holder)

    def test_custom_holder_fallback_keeps_source_getter_side_effects_and_errors(self):
        for holder_type in (CountingHolder, RaisingHolder):
            expected = holder_type(flags={'mimic': {'id': 'zero'}})
            actual = holder_type(flags={'mimic': {'id': 'zero'}})
            if holder_type is RaisingHolder:
                with self.assertRaisesRegex(ValueError, 'source lookup failed'):
                    modifiers.legacy_effects(expected)
                with self.assertRaisesRegex(ValueError, 'source lookup failed'):
                    query_legacy_effects(actual)
            else:
                self.assertEqual(query_legacy_effects(actual), modifiers.legacy_effects(expected))
                self.assertEqual(actual['source_reads'], 1)
            self.assertEqual(actual, expected)

    def test_exact_entity_holders_fast_path_and_unknown_lookup_override_fail_closed(self):
        for holder_type in (dict, PlayerEntity, CharacterEntity, CardEntity):
            holder = holder_type(entity_id='a:hero', side='a', flags={'mimic': {'id': 'zero'}})
            self.assertEqual(query_legacy_effects(holder), modifiers.legacy_effects(holder))
            with patch.dict(query_legacy_effects.__globals__,
                    {'int': lambda value: (_ for _ in ()).throw(AssertionError('fallback reached'))}):
                self.assertEqual(query_legacy_effects(holder), [])
        with patch.object(CharacterEntity, 'get', RaisingHolder.get):
            with self.assertRaisesRegex(EffectQuerySpecializationError, 'holder lookup'):
                derive_legacy_effects()

    def test_missing_only_fast_path_reduces_work_without_calling_fallback(self):
        holder = dict(entity_id='a:hero', side='a', flags={'mimic': {'id': 'zero'}})
        canonical_globals = modifiers.legacy_effects.__globals__
        fast_globals = query_legacy_effects.__globals__
        with patch.dict(canonical_globals, {'int': lambda value: (_ for _ in ()).throw(AssertionError('canonical int reached'))}):
            with self.assertRaises(AssertionError): modifiers.legacy_effects(holder)
        with patch.dict(fast_globals, {'int': lambda value: (_ for _ in ()).throw(AssertionError('fallback reached'))}):
            self.assertEqual(query_legacy_effects(holder), [])
            with self.assertRaises(AssertionError): query_legacy_effects(dict(flags={'atk_buff': 0}))

    def test_ast_body_and_dependency_changes_fail_closed(self):
        lines, start = inspect.getsourcelines(modifiers.legacy_effects)
        for old, new in (("flags = holder.get('flags') or {}", "flags = holder.get('flags', {})"),
                ("flags.get('atk_buff', 0)", "flags.get('atk_buff', 1)"),
                ("flags.get('atk_buff', 0)", "flags.get('new_key', 0)"),
                ('result = []', 'result = [{"value": 1}]')):
            text = ''.join(lines)
            self.assertIn(old, text)
            with patch('inspect.getsourcelines', return_value=(text.replace(old, new).splitlines(True), start)):
                with self.assertRaises(EffectQuerySpecializationError): derive_legacy_effects()

    def test_native_cpu_exact_type_guard_and_canonical_fallback(self):
        if sys.version_info[:2] != (3, 10):
            self.skipTest('CPython3.10 capture required')
        from app.modules.card_game.rl.rule_ir.portable_program import DEFAULT_ALLOWED_MODULE_PREFIXES
        from app.modules.card_game.rl.batched_duel.native_array_program import NativeArrayExporter
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.native_soa_cpu import NativeSoACPU
        from app.modules.card_game.rl.batched_duel import native_soa_cpu
        from app.modules.card_game.rl.batched_duel.record_soa import pack_game_states
        from tests.test_duel_v2_native_functions import fixture_schema
        programs = []
        for replacements in ({}, {modifiers.legacy_effects: query_legacy_effects,
                                  modifiers.all_effects: query_all_effects}):
            exporter = NativeArrayExporter(replacements,
                allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,),
                scope_character_ids=(), scope_card_ids=())
            program = exporter.run_capture({'legacy': modifiers.legacy_effects,
                'all': modifiers.all_effects, 'packet': effect_query_packet})
            programs.append(program)
            if replacements:
                names = {fn['name'] for fn in program.functions}
                self.assertIn('query_legacy_effects', names)
                self.assertNotIn('legacy_effects', names)
                self.assertIn('query_all_effects', names)
                self.assertNotIn('all_effects', names)
                self.assertEqual(exporter.substituted, {modifiers.legacy_effects, modifiers.all_effects})
        schema = fixture_schema()
        original_compile = native_soa_cpu.compile_host_native_library
        with patch.object(native_soa_cpu, 'compile_host_native_library',
                lambda source, directory: original_compile(source, directory, timeout=30)):
            options = dict(native_functions=True, native_values=True,
                max_nodes=8192, max_edges=16384, max_payload_bytes=65536)
            accepted_packets = 0
            with NativeSoACPU(lower_program(programs[0]), schema, **options) as baseline, \
                    NativeSoACPU(lower_program(programs[1]), schema, **options) as candidate:
                for flags in ({}, {'mimic': {'id': 'zero'}, 'shape': 'N07'}, {'atk_buff': 0},
                        {'atk_buff': 3}, {'timed_attack': [{'left': 2, 'amount': -1}]},
                        {'timed_attack': None}, {'atk_buff_expiring': {'amount': 2}}):
                    for holder_type in (dict, PlayerEntity, CharacterEntity, CardEntity):
                        holder = holder_type(entity_id='a:hero', side='a', flags=flags,
                            effects=[modifiers.contribution('test.panel', 'attack.panel', 2,
                                source_entity_id='a:hero')])
                        for entry in ('legacy', 'packet'):
                            calls = [backend.call(entry, pack_game_states([dict(hp=1, items=[])], schema),
                                [[holder]], state_positions=()) for backend in (baseline, candidate)]
                            self.assertEqual(int(calls[0].statuses[0, 0]), int(calls[1].statuses[0, 0]))
                            for key in calls[0].state.buffers:
                                self.assertEqual(calls[0].state.buffers[key].tobytes(), calls[1].state.buffers[key].tobytes())
                            if int(calls[0].statuses[0, 0]) == 0:
                                self.assertEqual(calls[0].decode(), calls[1].decode())
                                if entry == 'packet':
                                    value = calls[1].decode()[0]
                                    self.assertIs(value[0]['effects'][0], value[1][0])
                                    accepted_packets += 1
                for holder_type in (dict, PlayerEntity, CharacterEntity, CardEntity):
                    for fields in ({}, {'effects': [], 'flags': {}}, {'effects': (), 'flags': None},
                            {'effects': None, 'flags': {}}, {'effects': [], 'flags': False},
                            {'effects': [], 'flags': {'atk_buff': 0}}):
                        holder = holder_type(entity_id='a:hero', side='a', **fields)
                        calls = [backend.call('all', pack_game_states([dict(hp=1, items=[])], schema),
                            [[holder]], state_positions=()) for backend in (baseline, candidate)]
                        self.assertEqual(int(calls[0].statuses[0, 0]), int(calls[1].statuses[0, 0]))
                        if int(calls[0].statuses[0, 0]) == 0:
                            self.assertEqual(calls[0].decode(), calls[1].decode())
                            self.assertEqual(calls[1].decode(), [modifiers.all_effects(holder)])
            self.assertGreaterEqual(accepted_packets, 20)


if __name__ == '__main__': unittest.main()
