import unittest

from app.modules.card_game.engine.duel_v2 import new_game, apply_action
from app.modules.card_game.engine.duel_v2.state import hero


def scenario(attacker, mutual=False, self_down=False, weapon_side=None):
    defender = 'b' if attacker == 'a' else 'a'
    state = new_game(seed=9, first_side=attacker, skip_mulligan=True)
    for side in ('a', 'b'):
        state['sides'][side]['hand'] = []
        state['sides'][side]['shield'] = 0
        state['sides'][side]['ap'] = 2
    owner = weapon_side or attacker
    hero(state, owner, 'iloy')['shape'] = 'Y07'
    cid = 'iloy' if self_down else 'zero'
    hero(state, attacker, cid).update(hp=1 if mutual else 5, base_attack=2)
    hero(state, defender, 'zero').update(hp=1, base_attack=1)
    state['sides'][defender]['front'] = 'zero'
    return state, {'type': 'attack', 'character_id': cid}, defender


class IloyKnockdownTest(unittest.TestCase):
    def test_both_seats_and_self_exclusion(self):
        for attacker in ('a', 'b'):
            for mutual, self_down in ((False, False), (True, False), (True, True)):
                with self.subTest(attacker=attacker, mutual=mutual, self_down=self_down):
                    state, action, defender = scenario(attacker, mutual, self_down)
                    start = state['event_seq']
                    result = apply_action(state, attacker, action)
                    triggers = [e for e in result['events'] if e['seq'] > start
                                and '想做什么梦？' in e.get('text', '')]
                    fires = mutual and not self_down
                    self.assertEqual(len(triggers), int(fires))
                    self.assertEqual(result['sides'][defender]['hp'], 28 if fires else 30)
                    if fires:
                        downs = [e for e in result['events'] if e['seq'] > start and e['type'] == 'down']
                        self.assertEqual(len(downs), 2)
                        self.assertLess(max(e['seq'] for e in downs), triggers[0]['seq'])
                        self.assertEqual(triggers[0]['target'], defender + ':player')
                        for side in ('a', 'b'):
                            self.assertIsNone(result['sides'][side]['front'])

    def test_ally_down_still_hits_living_opposing_front(self):
        for attacker in ('a', 'b'):
            defender = 'b' if attacker == 'a' else 'a'
            state, action, _ = scenario(attacker, weapon_side=defender)
            result = apply_action(state, attacker, action)
            self.assertEqual(hero(result, attacker, 'zero')['hp'], 2)
            self.assertEqual(result['sides'][attacker]['hp'], 30)

    def test_iloy_downed_with_teammate_has_no_weapon_trigger(self):
        from app.modules.card_game.engine.duel_v2.state import knockdowns
        for side in ('a', 'b'):
            state, _, foe = scenario(side)
            hero(state, side, 'iloy')['hp'] = 0
            hero(state, side, 'zero')['hp'] = 0
            state['sides'][side]['front'] = 'iloy'
            start = state['event_seq']
            knockdowns(state)
            self.assertEqual(state['sides'][foe]['hp'], 30)
            self.assertFalse(any('想做什么梦？' in e.get('text', '')
                                 for e in state['events'] if e['seq'] > start))
