"""Zero fills harmony on battle release; combat never grants a second refill."""
import unittest
from unittest.mock import patch

from app.modules.card_game.engine.duel_v2 import new_game, apply_action
from app.modules.card_game.engine.duel_v2 import combat, flow
from app.modules.card_game.engine.duel_v2.state import card_instance, force_collapse, hero
from app.modules.card_game.engine.events import GameEvent


class ZeroHarmonyTimingTest(unittest.TestCase):
    def game(self):
        state = new_game(seed=19, first_side='a', skip_mulligan=True, escalation=False)
        for team in state['sides'].values():
            team.update(hand=[], ap=10, shield=0)
        return state

    def play(self, state, card_id='Z03', *, free=False):
        card = card_instance(state, card_id, 'a')
        if free:
            card['instant'] = True
        state['sides']['a']['hand'].append(card)
        return apply_action(state, 'a', {'type': 'play_card', 'card_id': card['instance_id']})

    def test_paid_base_gain_precedes_use_and_release_fills_before_attack(self):
        state = self.game()
        observed = []
        original = flow.fire
        def used(context, event, **data):
            result = original(context, event, **data)
            if event in (GameEvent.V2_CARD_USED, GameEvent.V2_CARD_RELEASED):
                observed.append((event, context.character['harmony']))
            return result
        cursor = state['event_seq']
        with patch.object(flow, 'fire', side_effect=used):
            state = self.play(state)
        self.assertEqual(observed, [(GameEvent.V2_CARD_USED, 1), (GameEvent.V2_CARD_RELEASED, 3)])
        events = [event for event in state['events'] if event['seq'] > cursor]
        gains = [event for event in events if event.get('source') == 'zero'
                 and event.get('after', {}).get('harmony') != event.get('before', {}).get('harmony')]
        self.assertEqual(len(gains), 2)
        self.assertEqual(gains[-1]['after']['harmony'], 3)
        self.assertLess(gains[0]['seq'], next(event['seq'] for event in events if event['type'] == 'attack'))
        self.assertEqual(hero(state, 'a', 'zero')['harmony'], 3)

    def test_partial_resource_is_filled_without_extra_post_combat_gain(self):
        state = self.game()
        hero(state, 'a', 'zero')['harmony'] = 2
        state = self.play(state)
        self.assertEqual(hero(state, 'a', 'zero')['harmony'], 3)
        gains = [event for event in state['events'] if event.get('source') == 'zero'
                 and event.get('after', {}).get('harmony') == 3
                 and event.get('before', {}).get('harmony') == 2]
        self.assertEqual(len(gains), 1)

    def test_free_battle_use_still_applies_the_passive_override(self):
        state = self.game()
        ap = state['sides']['a']['ap']
        state = self.play(state, free=True)
        self.assertEqual(state['sides']['a']['ap'], ap)
        self.assertEqual(hero(state, 'a', 'zero')['harmony'], 3)

    def test_normal_sortie_and_tactic_only_gain_actual_payment(self):
        state = self.game()
        state = apply_action(state, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(hero(state, 'a', 'zero')['harmony'], 1)
        state = self.play(state, 'Z02')
        self.assertEqual(hero(state, 'a', 'zero')['harmony'], 2)

    def test_direct_release_refills_without_payment_or_source(self):
        state = self.game()
        flow.resolve_release(state, 'a', card_instance(state, 'Z03', 'a'))
        self.assertEqual(hero(state, 'a', 'zero')['harmony'], 3)
        self.assertIsNone(state['sides']['a']['last_front'])
        self.assertEqual(state['sides']['a']['ap'], 10)

    def test_lethal_counter_keeps_the_resource_gained_before_combat(self):
        state = self.game()
        hero(state, 'a', 'zero')['hp'] = 1
        state['sides']['b']['front'] = 'nanali'
        hero(state, 'b', 'nanali').update(base_attack=20, hp=50, max_hp=50)
        state = self.play(state)
        self.assertEqual(hero(state, 'a', 'zero')['hp'], 0)
        self.assertEqual(hero(state, 'a', 'zero')['harmony'], 3)

    def test_interrupted_battle_keeps_the_already_paid_refill(self):
        state = self.game()
        original = combat.fire
        def interrupt(context, event, **data):
            original(context, event, **data)
            if event == 'on_sortie_ready':
                force_collapse(context.state, 'a', 'zero', by='b')
        with patch.object(combat, 'fire', side_effect=interrupt):
            state = self.play(state)
        self.assertFalse(any(event['type'] == 'attack' for event in state['events']))
        self.assertEqual(hero(state, 'a', 'zero')['harmony'], 3)

    def test_copied_passive_changes_the_actual_users_gain_only(self):
        state = self.game()
        hero(state, 'a', 'nanali')['flags']['mimic'] = {'id': 'zero'}
        state = self.play(state, 'N03')
        self.assertEqual(hero(state, 'a', 'nanali')['harmony'], 3)
        self.assertEqual(hero(state, 'a', 'zero')['harmony'], 0)
