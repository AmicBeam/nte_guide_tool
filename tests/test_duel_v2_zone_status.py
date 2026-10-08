"""User-defined 2026-09-14 battle-zone timing and targeting rules."""
import unittest
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe
from app.modules.card_game.engine.duel_v2.flow import begin_turn, expire_turn_zone_states
from app.modules.card_game.engine.duel_v2.state import hero, front_debuff, move_out
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board


class BattleZoneStatusTest(unittest.TestCase):
    def game(self):
        s = new_game(seed=9, first_side='a', skip_mulligan=True)
        for side in ('a', 'b'):
            s['sides'][side]['hand'] = []
            s['sides'][side]['ap'] = 2
            s['sides'][side]['shield'] = 0
        return s

    def test_burn_ticks_on_applier_turn_end_not_enemy_turn_start(self):
        s = self.game()
        s['sides']['b']['front'] = 'zero'
        front_debuff(s, 'b')['burn'] = {'left': 2, 'by': 'a', 'stacks': 1, 'source': 'a:nanali'}
        hp = {cid: hero(s, 'b', cid)['hp'] for cid in s['sides']['b']['characters']}
        begin_turn(s, 'b')
        self.assertEqual({cid: hero(s, 'b', cid)['hp'] for cid in hp}, hp)
        self.assertEqual(front_debuff(s, 'b')['burn']['left'], 2)
        from app.modules.card_game.engine.duel_v2.flow import end_turn
        end_turn(s, 'a')
        lost = sum(hp[cid] - hero(s, 'b', cid)['hp'] for cid in hp)
        self.assertEqual(lost, 1)
        self.assertEqual(front_debuff(s, 'b')['burn']['left'], 1)

    def test_empty_zone_damage_and_star_zero(self):
        s = self.game()
        zone = front_debuff(s, 'b')
        zone.update(burn={'left': 2, 'by': 'a', 'stacks': 1}, star={'left': 2, 'by': 'a'})
        begin_turn(s, 'b')
        self.assertEqual(s['sides']['b']['hp'], 30)
        self.assertEqual(zone['star']['left'], 1)
        self.assertIn('burn', zone)
        begin_turn(s, 'b')
        self.assertEqual(s['sides']['b']['hp'], 28)
        self.assertIn('burn', zone)
        self.assertNotIn('star', zone)

    def test_returned_character_is_not_burn_target(self):
        s = self.game()
        s['sides']['b']['front'] = 'zero'
        front_debuff(s, 'b')['burn'] = {'left': 1, 'by': 'a', 'stacks': 1, 'source': 'a:nanali'}
        move_out(s, 'b', 'zero')
        from app.modules.card_game.engine.duel_v2.flow import end_turn
        end_turn(s, 'a')
        self.assertEqual(s['sides']['b']['hp'], 30)
        self.assertNotIn('burn', front_debuff(s, 'b'))

    def test_current_turn_status_expiration_and_weave_player_target(self):
        s = self.game()
        front_debuff(s, 'b').update(weave={'by': 'a', 'expires_side': 'a'},
                                  delay={'by': 'a', 'expires_side': 'a', 'slow': 1, 'left': 1})
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertEqual(s['sides']['b']['hp'], 26)
        overlays = [e for e in s['events'] if e['type'] == 'overlay']
        self.assertEqual(overlays[-1]['target'], 'b:player')
        expire_turn_zone_states(s, 'a')
        self.assertFalse(front_debuff(s, 'b'))

    def test_weave_does_not_retarget_after_front_dies(self):
        s = self.game()
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero')['hp'] = 1
        front_debuff(s, 'b')['weave'] = {'by': 'a', 'expires_side': 'a'}
        seq = s['event_seq']
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertEqual(s['sides']['b']['hp'], 30)
        self.assertFalse(any(e['type'] == 'overlay' and e['seq'] > seq for e in s['events']))

    def test_empty_zone_harmony_is_applied(self):
        s = self.game()
        s['sides']['a']['front'] = 'zero'
        hero(s, 'a', 'zero').update(attribute='暗', harmony=2)
        hero(s, 'a', 'nanali')['attribute'] = '魂'
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertEqual(front_debuff(s, 'b')['star']['left'], 2)
        self.assertEqual(hero(s, 'a', 'zero')['harmony'], 0)

    def test_lethal_zone_effect_stops_return_and_draw(self):
        s = self.game()
        s['sides']['b']['hp'] = 1
        front_debuff(s, 'b')['star'] = {'left': 1, 'by': 'a'}
        deck = len(s['sides']['b']['deck'])
        begin_turn(s, 'b')
        self.assertEqual(s['phase'], 'finished')
        self.assertEqual(len(s['sides']['b']['deck']), deck)
