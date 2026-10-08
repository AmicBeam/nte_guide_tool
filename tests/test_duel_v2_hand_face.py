import unittest
from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe
from app.modules.card_game.engine.duel_v2.state import card_instance, hero, move_out, draw
from app.modules.card_game.engine.duel_v2.presentation import public_card, project_event


class HandFaceTest(unittest.TestCase):
    def game(self):
        team = ['zhenhong', 'yi', 'zero', 'jiuyuan']
        deck = dict(id='face', name='展示测试', character_ids=team,
                    card_ids=[key for cid in team for key, card in CARDS.items()
                              if card['character_id'] == cid and not card.get('derived')])
        s = new_game(seed=61, first_side='a', skip_mulligan=True, decks={'a':deck, 'b':deck}, escalation=False)
        for side in s['sides'].values():
            side.update(hand=[], shield=0)
        s['sides']['a']['hand'] = [card_instance(s, 'I04', 'a'), card_instance(s, 'Z08', 'a')]
        hero(s, 'a', 'zhenhong')['energy'] = 5
        return apply_action(s, 'a', {'type':'ultimate', 'character_id':'zhenhong'})

    def test_view_as_keeps_rule_identity_and_private_physical_face(self):
        s = self.game()
        hand = observe(s, 'a', include_previews=False)['sides']['a']['hand']
        self.assertEqual([c['hand_face']['card_id'] for c in hand], ['I04','Z08'])
        for card in hand:
            self.assertEqual(card['card_id'], 'RF01')
            self.assertEqual(card['character_id'], 'zhenhong')
            self.assertEqual(card['cost'], 1)
            self.assertFalse(card['action_point_free'])
            self.assertEqual(card['hand_face']['name'], CARDS[card['hand_face']['card_id']]['name'])
            self.assertNotIn('_physical_card', str(card))
            self.assertEqual(public_card(card)['hand_face'], card['hand_face'])
        enemy = observe(s, 'b', include_previews=False)['sides']['a']['hand']
        self.assertTrue(all(c == {'hidden':True} for c in enemy))
        move_out(s, 'a', 'zhenhong')
        restored = observe(s, 'a', include_previews=False)['sides']['a']['hand']
        self.assertTrue(all('hand_face' not in c for c in restored))
        self.assertEqual(restored[0]['card_id'], 'I04')

    def test_private_hand_and_new_draw_events_preserve_face(self):
        s = self.game()
        refresh = next(e for e in s['events'] if e.get('private_hand'))
        self.assertEqual(refresh['private_hand'][0]['hand_face']['card_id'], 'I04')
        s['sides']['a']['deck'].insert(0, card_instance(s, 'Z03', 'a'))
        draw(s, 'a')
        event = next(e for e in reversed(s['events']) if e['type']=='draw')
        self.assertEqual(event['private_card']['hand_face']['card_id'], 'Z03')
        self.assertNotIn('hand_face', str(project_event(event, 'b')))
