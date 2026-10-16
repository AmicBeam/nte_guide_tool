"""Declarative public-17 RECORD-POOL schema: closed types, not a rule engine."""
import unittest

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, PUBLIC_CHARACTER_IDS, STARTER_DECKS
from app.modules.card_game.engine.duel_v2.flow import acting_side, apply_action, legal_actions, new_game
from app.modules.card_game.rl.batched_duel.domain_soa_records import (
    ROOT,
    compile_domain_schema,
    coverage,
    domain_types,
    scalar_container_escapes,
    supports_full_duel_rules,
    undeclared_fields,
    unresolved_references,
)
from app.modules.card_game.rl.batched_duel.vm_heap import (
    TAG_CARD_ENTITY,
    TAG_CHARACTER_ENTITY,
    TAG_DICT,
    TAG_LIST,
    TAG_PLAYER_ENTITY,
    TAG_TUPLE,
)


KEYS = tuple(item['id'] for item in STARTER_DECKS)
PUBLIC = frozenset(PUBLIC_CHARACTER_IDS)


def _decks():
    return {item['id']: item for item in STARTER_DECKS}


def _step(state, count=8):
    found = list(undeclared_fields(state))
    for _ in range(count):
        side = acting_side(state)
        if side is None:
            break
        actions = legal_actions(state, side)
        if not actions:
            break
        action = next((item for item in actions if item.get('type') != 'concede'), actions[0])
        state = apply_action(state, side, action)
        found.extend(undeclared_fields(state))
    return state, tuple(found)


class DomainSoARecordsTest(unittest.TestCase):
    def test_domain_types_are_closed_record_pools_rooted_at_game(self):
        types = domain_types()
        self.assertEqual(ROOT, 'Game')
        self.assertIn('Game', types)
        self.assertFalse(supports_full_duel_rules)
        report = coverage()
        self.assertFalse(report['supports_full_duel_rules'])
        self.assertFalse(report['complete_soa'])
        self.assertFalse(report['full_public17'])
        self.assertEqual(report['schema_root'], 'Game')
        self.assertEqual(report['type_count'], len(types))
        self.assertEqual(unresolved_references(types), ())
        self.assertEqual(scalar_container_escapes(types), ())
        for name, definition in types.items():
            kind = definition['kind']
            self.assertIn(kind, ('record', 'list', 'map', 'tuple'), name)
            self.assertIsInstance(definition['capacity'], int)
            self.assertGreater(definition['capacity'], 0, name)
            if kind == 'record':
                self.assertTrue(definition['tags'])
                allowed = (TAG_DICT, TAG_PLAYER_ENTITY, TAG_CHARACTER_ENTITY, TAG_CARD_ENTITY)
                self.assertTrue(all(tag in allowed for tag in definition['tags']), name)
                self.assertIsInstance(definition['fields'], dict)
            elif kind == 'list':
                self.assertIn(tuple(definition['tags']), ((TAG_LIST,), (TAG_LIST, TAG_TUPLE)))
                self.assertGreater(definition['max_items'], 0, name)
            elif kind == 'map':
                self.assertEqual(definition['key'], 'str')
                self.assertGreater(definition['max_items'], 0, name)
            else:
                self.assertTrue(definition['items'])
        self.assertEqual(tuple(types['Game']['tags']), (TAG_DICT,))
        self.assertEqual(tuple(types['Player']['tags']), (TAG_PLAYER_ENTITY, TAG_DICT))
        self.assertEqual(tuple(types['Character']['tags']), (TAG_CHARACTER_ENTITY, TAG_DICT))
        self.assertEqual(tuple(types['Card']['tags']), (TAG_CARD_ENTITY, TAG_DICT))

    def test_compile_schema_if_record_soa_exists(self):
        try:
            from app.modules.card_game.rl.batched_duel import record_soa as record_soa_module
        except ImportError:
            self.skipTest('record_soa missing')
        schema = compile_domain_schema()
        self.assertEqual(schema.names[schema.root], 'Game')
        self.assertEqual(len(schema.pools), len(domain_types()))
        self.assertTrue(callable(record_soa_module.compile_schema))

    def test_flag_choice_operation_deferred_and_presentation_types(self):
        types = domain_types()
        flags = types['Flags']['fields']
        self.assertEqual(flags['pact'], 'bool')
        self.assertEqual(flags['allied_hurt'], 'bool')
        self.assertEqual(flags['share_overflow'], 'bool')
        self.assertEqual(flags['mimic'], '@MimicStatus')
        self.assertEqual(flags['energy_next'], 'bool')
        self.assertEqual(types['Operation']['fields']['pending_sortie'], '@PendingSortie')
        self.assertEqual(types['Operation']['fields']['last_attack_target'], '@SideCid')
        self.assertEqual(tuple(types['SideCid']['tags']), (TAG_LIST, TAG_TUPLE))
        self.assertEqual(types['Player']['fields']['canhong_collapse_used'], 'bool')
        game = types['Game']['fields']
        for key in ('events', '_public_board', 'pending_choice', 'operation', 'deferred_actions',
                    '_pending_down_responses', '_deferred_card_marks', 'turn_snapshots', 'time_removed'):
            self.assertIn(key, game, key)
        self.assertEqual(game['events'], '@EventList')
        self.assertEqual(game['_public_board'], '@PublicBoard')
        self.assertIn('value', types['EffectConditionArgs']['fields'])
        self.assertIn('shield_removed', types['Event']['fields'])

    def test_catalog_source_mechanisms_ultimate_play_options_are_typed_records(self):
        types = domain_types()
        self.assertEqual(types['Card']['fields']['source'], '@CardSource')
        self.assertEqual(types['Card']['fields']['play_options'], '@PlayOptionList')
        self.assertEqual(types['Character']['fields']['mechanisms'], '@MechanismList')
        self.assertEqual(types['Character']['fields']['ultimate'], '@UltimateSpec')
        misses = []
        derived = []
        for card_id, card in CARDS.items():
            owner = card.get('character_id')
            if owner not in PUBLIC and not (card.get('derived') and owner in PUBLIC):
                continue
            if card.get('derived'):
                derived.append(card_id)
            misses.extend(undeclared_fields(card, 'Card'))
        for cid, hero in CHARACTERS.items():
            if cid in PUBLIC:
                misses.extend(undeclared_fields(hero, 'Character'))
        self.assertEqual(tuple(misses), ())
        self.assertEqual(sorted(derived), ['AF01', 'FF01', 'NF01', 'PF01', 'RF01'])
        self.assertEqual(KEYS, ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk', 'collapse', 'lingke'))

    def test_seven_team_canonical_states_have_no_undeclared_fields(self):
        decks = _decks()
        misses = []
        for index, key in enumerate(KEYS):
            first = 'a' if index % 2 == 0 else 'b'
            other = KEYS[(index + 1) % len(KEYS)]
            initial = new_game(seed=161000 + index, decks={'a': decks[key], 'b': decks[other]}, first_side=first)
            misses.extend(undeclared_fields(initial))
            playing = new_game(
                seed=162000 + index, decks={'a': decks[key], 'b': decks[other]},
                first_side=first, skip_mulligan=True,
            )
            _, found = _step(playing, 8)
            misses.extend(found)
        self.assertEqual(tuple(misses), ())

    def test_live_aliases_for_flags_choice_operation_and_attack_target(self):
        decks = _decks()
        state = new_game(seed=7, decks={'a': decks['starter'], 'b': decks['collapse']}, skip_mulligan=True)
        hero = next(iter(state['sides']['a']['characters'].values()))
        card = state['sides']['a']['hand'][0]
        hero['flags'].update(
            pact=True, allied_hurt=True, share_overflow=True, mimic={'id': 'zero'}, energy_next=True,
        )
        state['sides']['a']['canhong_collapse_used'] = True
        state['pending_choice'] = {'side': 'a', 'kind': 'discard', 'prompt': 'x', 'cards': [card]}
        state['operation'] = {
            'side': 'a', 'actor': hero['id'], 'card': card,
            'pending_sortie': {'spec': {'hits': 1}, 'stay_bench': False},
            'last_attack_target': ['b', 'player'], 'participants': [['a', hero['id']]],
        }
        state['deferred_actions'] = [
            {'side': 'a', 'actor': hero['id'], 'entity_id': hero['entity_id'], 'kind': 'extra'},
        ]
        hero['effects'] = [{
            'effect_id': 'e1', 'definition_id': 'adler.curse_attack', 'property': 'attack.panel',
            'op': 'add', 'value': 1,
            'conditions': [{'id': 'attribute_is', 'args': {'value': '\u5492'}}],
        }]
        self.assertEqual(undeclared_fields(state), ())
