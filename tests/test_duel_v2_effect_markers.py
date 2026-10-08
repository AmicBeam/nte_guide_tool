import unittest

from app.modules.card_game.engine.duel_v2 import new_game, observe
from app.modules.card_game.engine.duel_v2.state import hero, knockdowns, move_out
from app.modules.card_game.engine.duel_v2.flow import begin_turn
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board


class EffectMarkersTest(unittest.TestCase):
    def test_pact_is_a_debuff_in_both_views_and_replay_and_clears(self):
        from app.modules.card_game.engine.duel_v2.context import EffectContext
        state = new_game(seed=9, first_side='a', skip_mulligan=True)
        c = EffectContext(state, 'a', 'jiuyuan', {'side': 'a'})
        c.attach_pact('b', 'zero')
        for board in (observe(state, 'a'), observe(state, 'b'), capture_public_board(state)):
            h = next(h for h in board['sides']['b']['characters'] if h['id'] == 'zero')
            marker = next(m for m in h['effect_markers'] if m['id'] == 'pact')
            self.assertEqual((marker['name'], marker['kind']), ('枚约', 'debuff'))
            self.assertTrue(marker['clears_on_down'])
        hero(state, 'b', 'zero')['flags'].pop('pact')
        h = next(h for h in capture_public_board(state)['sides']['b']['characters'] if h['id'] == 'zero')
        self.assertFalse(any(m['id'] == 'pact' for m in h['effect_markers']))
        c.attach_pact('b', 'zero')
        hero(state, 'b', 'zero')['hp'] = 0
        knockdowns(state)
        h = next(h for h in capture_public_board(state)['sides']['b']['characters'] if h['id'] == 'zero')
        self.assertFalse(any(m['id'] == 'pact' for m in h['effect_markers']))

    def test_character_pending_effect_survives_return_but_clears_on_down(self):
        state = new_game(seed=9, first_side='a', skip_mulligan=True)
        h = hero(state, 'a', 'iloy')
        h['flags']['energy_next'] = True
        state['sides']['a']['front'] = 'iloy'
        move_out(state, 'a', 'iloy')
        for board in (observe(state, 'a'), observe(state, 'b'), capture_public_board(state)):
            c = next(c for c in board['sides']['a']['characters'] if c['id'] == 'iloy')
            self.assertEqual(c['effect_markers'][0]['id'], 'energy_next')
            self.assertTrue(c['effect_markers'][0]['clears_on_down'])
        h['hp'] = 0
        knockdowns(state)
        c = next(c for c in capture_public_board(state)['sides']['a']['characters'] if c['id'] == 'iloy')
        self.assertEqual(c['effect_markers'], [])

    def test_effect_kind_is_public_for_both_views_and_replay(self):
        state = new_game(seed=9, first_side='a', skip_mulligan=True)
        h = hero(state, 'a', 'iloy')
        h['flags'].update(energy_next=True, timed_attack=[dict(amount=-2, left=2)])
        h['next_damage_resistance'] = {'光': -2, '灵': 1}
        for board in (observe(state, 'a'), observe(state, 'b'), capture_public_board(state)):
            c = next(c for c in board['sides']['a']['characters'] if c['id'] == 'iloy')
            kinds = {m['name']: m['kind'] for m in c['effect_markers']}
            self.assertEqual(kinds, {'光抗性': 'debuff', '灵抗性': 'buff',
                                     '攻击降低': 'debuff', '终结 · 回能': 'buff'})

    def test_door_is_team_effect_and_disappears_after_resolution(self):
        state = new_game(seed=9, first_side='a', skip_mulligan=True)
        team = state['sides']['a']
        team.update(extra_genesis_pending=True, extra_genesis_actor='zero')
        hero(state, 'a', 'iloy')['hp'] = 0
        knockdowns(state)
        for board in (observe(state, 'b'), capture_public_board(state)):
            marker = board['sides']['a']['effect_markers'][0]
            self.assertEqual(marker['id'], 'extra_genesis')
            self.assertFalse(marker['clears_on_down'])
            self.assertIn('零须存活', marker['description'])
        before_hp = state['sides']['b']['hp']
        state['sides']['b']['shield'] = 0
        begin_turn(state, 'a')
        self.assertLess(state['sides']['b']['hp'], before_hp)
        self.assertEqual(capture_public_board(state)['sides']['a']['effect_markers'], [])
