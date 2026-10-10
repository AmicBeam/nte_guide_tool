"""Exact clone_state/preview copy helper. Not a 5x claim."""
import json
import unittest
from copy import deepcopy
from random import Random
from unittest.mock import patch

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, STARTER_DECKS, validate_deck
from app.modules.card_game.engine.duel_v2 import acting_side, apply_action, legal_actions, new_game, observe
from app.modules.card_game.engine.duel_v2.entities import (
    CardEntity, CharacterEntity, PlayerEntity, Entity, clone_state, copy_state_graph, hydrate_entities,
)
from app.modules.card_game.engine.duel_v2.flow import apply_owned_action
from app.modules.card_game.engine.duel_v2.projection import (
    _preview_owned_sandbox, prepared_preview_sandbox, preview,
)
from app.modules.card_game.engine.duel_v2.state import card_instance, hero
from app.modules.card_game.rl.league_rollout import clean


def reference_entity_copy(value, memo):
    """Original Entity copier; keep the reference independent of the new helper."""
    result = type(value).__new__(type(value))
    memo[id(value)] = result
    for key, nested in value.items():
        result[deepcopy(key, memo)] = deepcopy(nested, memo)
    return result


def reference_clone(state):
    with patch.object(Entity, '__deepcopy__', reference_entity_copy):
        return hydrate_entities(deepcopy(state))


def xun_deck():
    return dict(
        id='xun-copy', name='xun',
        character_ids=['xun', 'nanali', 'zero', 'jiuyuan'],
        card_ids=[prefix + str(index).zfill(2)
                  for prefix in ('X', 'N', 'Z', 'J') for index in range(1, 5) for _ in range(2)],
    )


def catalog_deck(start=0):
    ids = list(CHARACTERS)
    chars = [ids[(start + index) % len(ids)] for index in range(4)]
    cards = [key for cid in chars for key, card in CARDS.items()
             if card['character_id'] == cid and not card.get('derived')]
    return validate_deck(dict(id='parity', name='parity', character_ids=chars, card_ids=cards))


class Token:
    def __init__(self, n):
        self.n = n

    def __eq__(self, other):
        return type(other) is Token and other.n == self.n


class CopyHelperTest(unittest.TestCase):
    def test_entity_subclass_copy_hook_and_super_are_preserved(self):
        class CustomCard(CardEntity):
            def __deepcopy__(self, memo):
                result = super().__deepcopy__(memo)
                result['hooked'] = True
                return result
        card = CustomCard({'nested': [1]}, entity_id='e1', side='a')
        copied = copy_state_graph({'one': card, 'two': card})
        self.assertIs(copied['one'], copied['two'])
        self.assertTrue(copied['one']['hooked'])
        self.assertNotIn('hooked', card)
        copied['one']['nested'].append(2)
        self.assertEqual(card['nested'], [1])

    def test_generic_aliases_cycles_atomics_and_unknown_type_fallback(self):
        token = Token(7)
        cycle = {'n': 1, 'text': 'ok'}
        cycle['self'] = cycle
        inner = [True, 2, 3.5, None, 's']
        graph = {
            'atomic': inner,
            'alias': inner,
            'cycle': cycle,
            'token': token,
            'again': token,
            'tuple': (1, ['x']),
            'flags': {1, 2},
            3: 'int-key',
        }
        copied = copy_state_graph(graph)
        self.assertEqual(copied['atomic'], inner)
        self.assertIs(copied['atomic'], copied['alias'])
        self.assertIsNot(copied['atomic'], inner)
        self.assertIs(copied['cycle'], copied['cycle']['self'])
        self.assertIsNot(copied['cycle'], cycle)
        self.assertIs(copied['token'], copied['again'])
        self.assertIsNot(copied['token'], token)
        self.assertEqual(copied['token'], token)
        self.assertEqual(copied['tuple'], (1, ['x']))
        self.assertIsNot(copied['tuple'][1], graph['tuple'][1])
        copied['tuple'][1].append('mut')
        self.assertEqual(graph['tuple'][1], ['x'])
        self.assertEqual(copied['flags'], {1, 2})
        copied['flags'].add(9)
        self.assertEqual(graph['flags'], {1, 2})
        copied['atomic'].append('nope')
        cycle['n'] = 99
        self.assertEqual(graph['atomic'], [True, 2, 3.5, None, 's'])
        self.assertEqual(cycle['n'], 99)
        self.assertEqual(copied['cycle']['n'], 1)
        self.assertEqual(copied[3], 'int-key')

    def test_entity_aliases_use_shared_memo_without_sharing_mutables(self):
        card = CardEntity({'instance_id': 'i1', 'card_id': 'N01', 'hp': [4]}, entity_id='e9', side='a')
        graph = {'hand': [card], 'pending': [card], 'op': {'card': card}}
        copied = copy_state_graph(graph)
        self.assertIs(copied['hand'][0], copied['pending'][0])
        self.assertIs(copied['hand'][0], copied['op']['card'])
        self.assertIsInstance(copied['hand'][0], CardEntity)
        self.assertIsNot(copied['hand'][0], card)
        self.assertEqual(copied['hand'][0].entity_id, 'e9')
        via_deepcopy = deepcopy(graph)
        self.assertIs(via_deepcopy['hand'][0], via_deepcopy['pending'][0])
        self.assertEqual(via_deepcopy, copied)
        copied['hand'][0]['hp'].append(8)
        self.assertEqual(card['hp'], [4])


class StateCopyFastTest(unittest.TestCase):
    def game(self, seed=9, decks=None, skip_mulligan=True):
        selected = decks or STARTER_DECKS[0]
        return new_game(seed=seed, first_side='a', skip_mulligan=skip_mulligan,
                        decks={'a': selected, 'b': selected})

    def assert_clone_parity(self, state):
        original = deepcopy(state)
        cloned = clone_state(state)
        reference = reference_clone(state)
        self.assertEqual(cloned, reference)
        self.assertEqual(state, original)
        self.assertIsInstance(cloned['sides']['a'], PlayerEntity)
        self.assertIsInstance(next(iter(cloned['sides']['a']['characters'].values())), CharacterEntity)
        self.assertIsNot(cloned['sides']['a'], state['sides']['a'])
        self.assertIsNot(cloned['events'], state['events'])
        cloned['sides']['a']['hp'] = original['sides']['a']['hp'] - 17
        cloned['rng'] = 'mutated'
        cloned.setdefault('events', []).append({'synthetic': True})
        cloned['unrecognized'] = {'extra': [1]}
        self.assertEqual(state, original)
        return cloned

    def test_starter7_catalog_pending_operation_and_xun_snapshots(self):
        checked = 0
        games = []
        for index, deck in enumerate(STARTER_DECKS):
            other = STARTER_DECKS[(index + 1) % len(STARTER_DECKS)]
            games.append(new_game(seed=80 + index, first_side=('a', 'b')[index % 2], skip_mulligan=True,
                                  decks={'a': deck, 'b': other}))
        for start in range(0, len(CHARACTERS), 4):
            roster = catalog_deck(start)
            games.append(new_game(seed=91 + start, first_side='b', skip_mulligan=True,
                                  decks={'a': roster, 'b': roster}))
        xun = new_game(seed=21, first_side='a', skip_mulligan=True,
                       decks={'a': xun_deck(), 'b': xun_deck()})
        xun = apply_action(xun, 'a', {'type': 'end_turn'})
        xun = apply_action(xun, 'b', {'type': 'end_turn'})
        xun = apply_action(xun, 'a', {'type': 'end_turn'})
        xun = apply_action(xun, 'b', {'type': 'end_turn'})
        self.assertGreaterEqual(len(xun.get('turn_snapshots', {}).get('a', [])), 2)
        games.append(xun)

        inspect = new_game(seed=6, first_side='a', skip_mulligan=True)
        inspect_card = card_instance(inspect, 'J01', 'a')
        inspect['sides']['a']['hand'] = [inspect_card]
        inspect = apply_action(inspect, 'a', {'type': 'play_card', 'card_id': inspect_card['instance_id']})
        self.assertEqual(inspect['phase'], 'choice')
        games.append(inspect)

        aliased = self.game()
        live = aliased['sides']['a']['hand'][0]
        aliased['pending_choice'] = {'side': 'a', 'kind': 'inspect_top', 'prompt': 'choose', 'cards': [live]}
        aliased['operation'] = {'side': 'a', 'card': live, 'actor': live.get('character_id')}
        aliased['mystery'] = {'events': [{'n': 1}], 'resources': [aliased['sides']['a']['ap']]}
        games.append(aliased)

        for game_index, state in enumerate(games):
            rng = Random(400 + game_index)
            for step in range(16):
                self.assert_clone_parity(state)
                checked += 1
                if state['phase'] == 'finished':
                    break
                side = acting_side(state)
                actions = [item for item in legal_actions(state, side) if item['type'] != 'concede']
                if not actions:
                    break
                action = rng.choice(actions)
                expected = apply_owned_action(reference_clone(state), side, action)
                actual = apply_action(state, side, action)
                self.assertEqual(actual, expected)
                state = clean(actual)
        self.assertGreater(checked, 40)

        cloned = clone_state(aliased)
        self.assertIs(cloned['pending_choice']['cards'][0], cloned['sides']['a']['hand'][0])
        self.assertIs(cloned['operation']['card'], cloned['sides']['a']['hand'][0])
        self.assertEqual(cloned['mystery'], aliased['mystery'])
        cloned['mystery']['events'].append({'n': 2})
        self.assertEqual(aliased['mystery']['events'], [{'n': 1}])
        snap = clone_state(xun)
        self.assertEqual(snap['turn_snapshots'], xun['turn_snapshots'])
        self.assertIsNot(snap['turn_snapshots']['a'][0], xun['turn_snapshots']['a'][0])
        snap['turn_snapshots']['a'][0]['rng'] = 'mutated'
        self.assertNotEqual(xun['turn_snapshots']['a'][0]['rng'], 'mutated')

    def test_json_restore_conflict_validation_and_input_immutable(self):
        state = self.game()
        card = state['sides']['a']['hand'][0]
        state['pending_choice'] = {'side': 'a', 'cards': [card]}
        original = deepcopy(state)
        raw = json.loads(json.dumps(state))
        restored = clone_state(raw)
        self.assertEqual(restored, reference_clone(raw))
        self.assertEqual(restored, state)
        self.assertIsInstance(restored['pending_choice']['cards'][0], CardEntity)
        self.assertIs(restored['pending_choice']['cards'][0], restored['sides']['a']['hand'][0])
        self.assertIsInstance(raw['sides']['a'], dict)
        self.assertNotIsInstance(raw['sides']['a'], PlayerEntity)
        self.assertEqual(state, original)

        conflict = clone_state(self.game())
        hero(conflict, 'b', 'nanali')['entity_id'] = hero(conflict, 'a', 'nanali').entity_id
        with self.assertRaises(ValueError):
            clone_state(conflict)
        copy_state_graph(conflict)

    def test_public_apply_action_and_preview_match_reference_clone(self):
        state = self.game()
        state['sides']['b']['front'] = 'zero'
        action = {'type': 'attack', 'character_id': 'zero'}
        original = deepcopy(state)
        expected = apply_owned_action(reference_clone(state), 'a', action)
        actual = apply_action(state, 'a', action)
        self.assertEqual(actual, expected)
        self.assertEqual(state, original)
        self.assertEqual(preview(state, 'a', action), preview(reference_clone(state), 'a', action))
        view = observe(state, 'a', include_previews=True)
        self.assertEqual(state, original)
        combat = next(entry['preview'] for entry in view['legal_actions'] if entry['action'] == action)
        self.assertEqual(combat, preview(state, 'a', action))
        with prepared_preview_sandbox(state, 'a') as template:
            one = _preview_owned_sandbox(state, 'a')
            two = _preview_owned_sandbox(state, 'a')
            self.assertEqual(one, two)
            self.assertEqual(one, deepcopy(template))
            self.assertIsNot(one, two)
            one['sides']['a']['hp'] = -9
            self.assertNotEqual(two['sides']['a']['hp'], -9)
            self.assertNotEqual(template['sides']['a']['hp'], -9)
        self.assertEqual(state, original)


if __name__ == '__main__':
    unittest.main()
