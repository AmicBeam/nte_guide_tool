"""Automatic C07 releases its effect without entering the card-use lifecycle."""
import json
import unittest
from unittest.mock import patch

from tests.test_duel_v2_continuous_cast import _deck
from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.content.duel_v2.registry import KITS
from app.modules.card_game.engine.duel_v2 import apply_action, new_game, observe
from app.modules.card_game.engine.duel_v2.entities import clone_state
from app.modules.card_game.engine.duel_v2.modifiers import add_effect, all_effects
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.state import card_instance, front_debuff, hero, knockdowns


class AutoReleaseTest(unittest.TestCase):
    def game(self):
        deck = _deck(('canhong', 'lingke', 'zero', 'iloy'), ('C', 'K', 'Z', 'Y'))
        return new_game(seed=73, first_side='a', skip_mulligan=True,
                        decks={'a': deck, 'b': deck}, escalation=False)

    def test_auto_release_from_hand_or_deck_broadcasts_release_only_and_preserves_parent(self):
        for location in ('hand', 'deck'):
            with self.subTest(location=location):
                s = self.game()
                team = s['sides']['a']
                actor = hero(s, 'a', 'canhong')
                lingke = hero(s, 'a', 'lingke')
                lingke['lingke_chain'] = dict(entity_id=actor.entity_id, character_id='canhong', count=2)
                hero(s, 'a', 'zero')['awakened'] = True
                card = card_instance(s, 'C07', 'a')
                team.update(hand=[], deck=[], ap=0)
                team[location].append(card)
                # A use-time draw would end this game because the deck is empty after extraction.
                front_debuff(s, 'b')['burn'] = dict(left=2, by='a', stacks=20)
                parent = dict(side='b', actor='zero', card=None, participants=[], harmony=None,
                              first_bonus=0, redeem_bonus=0, target_id='a:canhong')
                s['operation'] = parent
                calls = []
                def used(c, actor_id, card, **_data):
                    calls.append(('use', actor_id, card['card_id'], hero(c.state, 'a', actor_id)['hp']))
                def released(c, actor_id, card, **_data):
                    calls.append(('release', actor_id, card['card_id'], hero(c.state, 'a', actor_id)['hp']))
                opening, seq = capture_public_board(s), s['event_seq']
                actor['hp'] = 0
                with patch.object(KITS['zero'], 'on_card_used', staticmethod(used), create=True), \
                     patch.object(KITS['zero'], 'on_card_released', staticmethod(released), create=True), \
                     patch.object(KITS['zero'], 'after_card', staticmethod(used)):
                    knockdowns(s)
                self.assertEqual(calls, [('release', 'canhong', 'C07', 0)])
                self.assertIs(s['operation'], parent)
                self.assertEqual((actor['hp'], actor['shape'], actor['down_turns']), (6, 'C07', 0))
                self.assertEqual((team['ap'], team['used']), (0, {}))
                self.assertEqual(lingke['lingke_chain']['count'], 2)
                self.assertEqual([c['entity_id'] for c in team['discard']], [card.entity_id])
                events = [e for e in s['events'] if e['seq'] > seq]
                notice, = [e for e in events if e['type'] == 'play']
                self.assertTrue(notice['release_only'])
                self.assertIn('自动释放「赤葵」', notice['text'])
                self.assertFalse(notice['response'])
                self.assertFalse(any(e['type'] == 'draw' for e in events))
                for side in ('a', 'b'):
                    projected = next(e for e in observe(s, side)['presentation']['events'] if e['seq'] == notice['seq'])
                    self.assertTrue(projected['release_only'])
                    self.assertIn('释放', projected['text'])
                self.assertEqual(replay_public_board(opening, events), capture_public_board(s))
                restored = clone_state(json.loads(json.dumps(s)))
                knockdowns(restored)
                self.assertEqual(len(restored['sides']['a']['discard']), 1)

    def test_manual_use_broadcasts_use_then_release_and_still_pays(self):
        s = self.game()
        team = s['sides']['a']
        team['ap'] = 1
        hero(s, 'a', 'canhong').update(hp=0, down_turns=3)
        card = card_instance(s, 'C07', 'a')
        team['hand'] = [card]
        calls = []
        def listener(kind):
            def collect(c, actor_id, card, **_data):
                calls.append((kind, actor_id, c.state['sides']['a']['ap'], hero(c.state, 'a', actor_id)['hp']))
            return staticmethod(collect)
        with patch.object(KITS['zero'], 'on_card_used', listener('use'), create=True), \
             patch.object(KITS['zero'], 'on_card_released', listener('release'), create=True):
            after = apply_action(s, 'a', {'type': 'play_card', 'card_id': card['instance_id']})
        self.assertEqual(calls, [('use', 'canhong', 0, 0), ('release', 'canhong', 0, 0)])
        self.assertEqual(hero(after, 'a', 'canhong')['shape'], 'C07')
        self.assertNotIn('不消耗行动力', CARDS['C07']['description'])
        self.assertIn('自动释放', CARDS['C07']['description'])

    def test_automatic_release_does_not_consume_next_card_instant_grant(self):
        s = self.game()
        actor = hero(s, 'a', 'canhong')
        add_effect(actor, 'test.next_instant', 'card.instant', True, op='allow',
                   clear_on=(), consume_on='card_use', source_entity_id=actor.entity_id)
        s['sides']['a']['hand'] = [card_instance(s, 'C07', 'a')]
        front_debuff(s, 'b')['burn'] = dict(left=2, by='a', stacks=20)
        # Down clears most effects; this explicit persistent grant survives it.
        actor['hp'] = 0
        knockdowns(s)
        self.assertTrue(any(e['definition_id'] == 'test.next_instant' for e in all_effects(actor)))

    def test_hand_copy_is_preferred_and_condition_is_required(self):
        for stacks in (19, 20):
            with self.subTest(stacks=stacks):
                s = self.game()
                team = s['sides']['a']
                held, deck_card = (card_instance(s, 'C07', 'a') for _ in range(2))
                team.update(hand=[held], deck=[deck_card])
                front_debuff(s, 'b')['burn'] = dict(left=2, by='a', stacks=stacks)
                hero(s, 'a', 'canhong')['hp'] = 0
                knockdowns(s)
                self.assertEqual([c['entity_id'] for c in team['deck']], [deck_card.entity_id])
                self.assertEqual([c['entity_id'] for c in team['discard']], [held.entity_id] if stacks == 20 else [])
                self.assertEqual(len(team['hand']), 0 if stacks == 20 else 1)
                self.assertEqual(hero(s, 'a', 'canhong')['hp'], 6 if stacks == 20 else 0)


if __name__ == '__main__':
    unittest.main()
