"""Current cumulative counts are projected from the same records as card effects."""
from copy import deepcopy
import json
import unittest

from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.engine.duel_v2 import apply_action, observe
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board
from app.modules.card_game.engine.duel_v2.state import card_instance, damage, hero
from tests import test_duel_v2_surplus_rework as fixtures


class CounterTextTest(unittest.TestCase):
    team = ('nanali', 'zhenhong', 'haniya', 'yi')
    counters = {'N07': 'nanali_family_played', 'R06': 'surplus_passive_triggers',
                'V04': 'dark_star_count', 'I06': 'delay_turns'}

    def game(self):
        return fixtures.SurplusReworkTest().game(team=self.team)

    def set_counts(self, state, side, count):
        state['sides'][side].update(nanali_family_played=count, dark_star_count=count, delay_turns=count)
        hero(state, side, 'zhenhong').update(surplus_passive_triggers=count, extra_attacks=99)

    def test_initial_values_and_live_counts_both_sides_json_and_no_mutation(self):
        state = self.game()
        for side in ('a', 'b'):
            for key in self.counters:
                card = card_instance(state, key, side)
                # Existing saved cards predate the zero placeholder.
                card['description'] = card['description'].removesuffix('（当前为0）')
                state['sides'][side]['hand'].append(card)
                self.assertTrue(CARDS[key]['description'].endswith('（当前为0）'))
        for a_count, b_count in ((0, 0), (2, 5)):
            self.set_counts(state, 'a', a_count)
            self.set_counts(state, 'b', b_count)
            state = json.loads(json.dumps(state))
            before = deepcopy(state)
            for side, count in (('a', a_count), ('b', b_count)):
                view = observe(state, side, include_previews=False)
                for card in view['sides'][side]['hand']:
                    self.assertTrue(card['description'].endswith(f'（当前为{count}）'), card)
                    self.assertEqual(card['description'].count('（当前为'), 1)
                self.assertTrue(all(card.get('hidden') for card in view['sides']['b' if side == 'a' else 'a']['hand']))
            self.assertEqual(state, before)

    def test_counts_remain_visible_while_owner_is_down_and_in_discard(self):
        state = self.game()
        self.set_counts(state, 'a', 3)
        for key in self.counters:
            card = card_instance(state, key, 'a')
            state['sides']['a']['discard'].append(card)
        for cid in self.team:
            damage(state, 'a', cid, 99)
        view = observe(state, 'b', include_previews=False)
        for card in view['sides']['a']['discard']:
            self.assertTrue(card['description'].endswith('（当前为3）'), card)

    def test_r06_play_event_preview_and_actual_frame_share_trigger_count(self):
        state = self.game()
        self.set_counts(state, 'a', 3)
        card = card_instance(state, 'R06', 'a')
        state['sides']['a']['hand'].append(card)
        action = dict(type='play_card', card_id=card['instance_id'])
        view = observe(state, 'a')
        entry = next(row for row in view['legal_actions'] if row['action'] == action)
        self.assertEqual(entry['preview']['attack'], 5)
        after = apply_action(state, 'a', action)
        self.assertEqual(hero(after, 'a', 'zhenhong')['shield'], 3)
        event = next(e for e in reversed(after['events']) if e['type'] == 'play')
        self.assertTrue(event['card']['description'].endswith('（当前为3）'))

    def test_equipment_visible_deck_and_transformed_hand_face_use_live_count(self):
        state = self.game()
        self.set_counts(state, 'a', 4)
        card = card_instance(state, 'N07', 'a')
        state['sides']['a']['hand'].append(card)
        state = apply_action(state, 'a', dict(type='play_card', card_id=card['instance_id']))
        view = observe(state, 'a', include_previews=False)
        char = next(h for h in view['sides']['a']['characters'] if h['id'] == 'nanali')
        self.assertTrue(char['shape_description'].endswith('（当前为4）'))
        board_char = next(h for h in capture_public_board(state)['sides']['a']['characters'] if h['id'] == 'nanali')
        self.assertEqual(board_char['shape_description'], char['shape_description'])
        state['sides']['a']['deck'] = [card_instance(state, 'R06', 'a')]
        state['sides']['a']['deck_visible_turn'] = state['turn']
        hero(state, 'a', 'zhenhong')['awakened'] = True
        state['sides']['a']['hand'].append(card_instance(state, 'R06', 'a'))
        view = observe(state, 'a', include_previews=False)
        self.assertTrue(view['sides']['a']['visible_deck'][0]['description'].endswith('（当前为4）'))
        self.assertTrue(view['sides']['a']['hand'][0]['hand_face']['description'].endswith('（当前为4）'))
        self.assertNotIn('visible_deck', observe(state, 'b', include_previews=False)['sides']['a'])
