"""User-authored Jiuyuan rush rules; numeric values are not Everness stats."""
import json
import unittest
from copy import deepcopy

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe
from app.modules.card_game.engine.duel_v2.combat import resolve_genesis
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.engine.duel_v2.flow import begin_turn
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.state import card_instance, damage, hero


class JiuyuanReworkTest(unittest.TestCase):
    def game(self):
        s = new_game(seed=52, first_side='a', skip_mulligan=True, escalation=False)
        for team in s['sides'].values():
            team.update(hand=[], ap=20, hp=100, shield=0)
        return s

    def hold(self, s, key, side='a'):
        card = card_instance(s, key, side)
        s['sides'][side]['hand'].append(card)
        return card

    def play(self, s, key):
        card = self.hold(s, key)
        return apply_action(s, 'a', {'type': 'play_card', 'card_id': card['instance_id']})

    def genesis(self, s, side='a', cid='zero'):
        resolve_genesis(EffectContext(s, side, cid, {'side': side}), arm_delay=False)

    def test_catalog_and_independent_card_proposals_unchanged(self):
        self.assertTrue(CARDS['J05']['instant'])
        self.assertIn('附着枚约', CHARACTERS['jiuyuan']['passive'])
        self.assertNotIn('枚约', CARDS['J08']['description'])
        self.assertIn('附着枚约', CARDS['J02']['description'])
        self.assertIn('倒地倒计时', CARDS['J03']['description'])

    def test_passive_attaches_without_equipment_and_when_shield_absorbs(self):
        s = self.game(); s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero')['shield'] = 9
        self.genesis(s)
        self.assertTrue(hero(s, 'b', 'zero')['flags']['pact'])
        self.assertEqual(hero(s, 'b', 'zero')['shield'], 7)
        self.assertEqual(s['sides']['a'].get('genesis_player_hits', 0), 0)

    def test_passive_stops_while_downed_and_does_not_mark_dead_target(self):
        s = self.game(); s['sides']['b']['front'] = 'zero'
        hero(s, 'a', 'jiuyuan').update(hp=0, down_turns=3)
        self.genesis(s)
        self.assertNotIn('pact', hero(s, 'b', 'zero')['flags'])
        s = self.game(); s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero')['hp'] = 1
        self.genesis(s)
        self.assertNotIn('pact', hero(s, 'b', 'zero')['flags'])
        self.assertEqual(s['sides']['a']['genesis_player_hits'], 1)

    def test_j08_damage_preserved_but_only_genesis_counts(self):
        s = self.game(); hero(s, 'a', 'jiuyuan')['shape'] = 'J08'
        self.genesis(s)
        self.assertEqual(s['sides']['b']['hp'], 96)
        self.assertEqual(s['sides']['a']['genesis_player_hits'], 2)
        self.assertEqual(s['sides']['a']['genesis_turn'], s['turn'])

    def test_count_hits_not_amount_and_exclude_other_damage(self):
        s = self.game(); s['sides']['a']['genesis_damage_bonus'] = 3
        self.genesis(s)
        self.assertEqual(s['sides']['b']['hp'], 92)
        self.assertEqual(s['sides']['a']['genesis_player_hits'], 2)
        for kind in ('flower', 'damage', 'combat', 'followup'):
            damage(s, 'b', 'player', 1, source='a:zero', kind=kind)
        damage(s, 'a', 'player', 1, source='a:zero', kind='genesis')
        self.assertEqual(s['sides']['a']['genesis_player_hits'], 2)
        self.genesis(s, 'b')
        self.assertEqual(s['sides']['b']['genesis_player_hits'], 2)

    def test_only_genesis_segments_that_reduce_player_life_count(self):
        # Two one-point genesis segments: shield may absorb both, one, or neither.
        for shield_value, expected in ((10, 0), (2, 0), (1, 1), (0, 2)):
            with self.subTest(shield=shield_value):
                s = self.game(); s['sides']['b']['shield'] = shield_value
                self.genesis(s)
                self.assertEqual(s['sides']['b']['hp'], 100 - expected)
                self.assertEqual(s['sides']['a'].get('genesis_player_hits', 0), expected)
                s = self.play(s, 'J06')
                self.assertEqual(s['sides']['b']['hp'], 100 - 2 * expected)

    def test_partial_absorption_counts_once_per_segment_not_damage_amount(self):
        s = self.game(); s['sides']['b']['shield'] = 2
        s['sides']['a']['genesis_damage_bonus'] = 3
        self.genesis(s)
        self.assertEqual(s['sides']['b']['hp'], 94)
        self.assertEqual(s['sides']['a']['genesis_player_hits'], 2)

    def test_life_lock_and_zero_damage_do_not_count_even_with_shield_loss(self):
        for shield_value in (0, 1, 10):
            with self.subTest(shield=shield_value):
                s = self.game(); s['sides']['b'].update(shield=shield_value, life_locked=True)
                self.genesis(s)
                damage(s, 'b', 'player', 0, source='a:zero', kind='genesis')
                self.assertEqual(s['sides']['b']['hp'], 100)
                self.assertEqual(s['sides']['a'].get('genesis_player_hits', 0), 0)

    def test_j05_targets_player_and_uses_shared_instant(self):
        s = self.game(); s['sides']['b']['front'] = 'zero'
        hp = hero(s, 'b', 'zero')['hp']; s['sides']['a']['ap'] = 0
        s = self.play(s, 'J05')
        self.assertEqual(s['sides']['b']['hp'], 98)
        self.assertEqual(hero(s, 'b', 'zero')['hp'], hp)
        self.assertTrue(s['sides']['a']['used']['instant'])
        s['sides']['a']['ap'] = 1
        self.genesis(s)
        s = self.play(s, 'J05')
        self.assertEqual(s['sides']['b']['hp'], 95)
        self.assertEqual(s['sides']['a']['ap'], 0)
        begin_turn(s, 'b'); begin_turn(s, 'a')
        s = self.play(s, 'J05')
        self.assertEqual(s['sides']['b']['hp'], 93)

    def test_j06_zero_is_zero_and_does_not_reveal(self):
        s = self.game(); hidden = self.hold(s, 'J01', 'b')
        s = self.play(s, 'J06')
        self.assertEqual(s['sides']['b']['hp'], 100)
        self.assertNotIn(hidden['instance_id'], s['sides']['b'].get('revealed_ids', []))
        self.assertEqual(s['sides']['a']['ap'], 19)

    def test_delayed_genesis_counts_for_j05_in_new_turn(self):
        s = self.game()
        s['sides']['a'].update(extra_genesis_pending=True, extra_genesis_actor='zero')
        begin_turn(s, 'b'); begin_turn(s, 'a')
        self.assertEqual(s['sides']['a']['genesis_player_hits'], 2)
        hp = s['sides']['b']['hp']
        s = self.play(s, 'J05')
        self.assertEqual(s['sides']['b']['hp'], hp - 3)

    def test_downed_jiuyuan_does_not_stop_team_history(self):
        s = self.game(); damage(s, 'a', 'jiuyuan', 99)
        self.genesis(s)
        self.assertEqual(s['sides']['a']['genesis_player_hits'], 1)
        self.assertEqual(s['sides']['a']['genesis_turn'], s['turn'])

    def test_j06_retargets_each_hit_and_does_not_charge_itself(self):
        s = self.game(); s['sides']['a']['genesis_player_hits'] = 5
        s['sides']['b']['front'] = 'zero'; hero(s, 'b', 'zero').update(hp=2, shield=1)
        seq = s['event_seq']; s = self.play(s, 'J06')
        hits = [e for e in s['events'] if e['seq'] > seq and e['type'] == 'damage']
        self.assertEqual([e['target'] for e in hits], ['b:zero'] * 3 + ['b:player'] * 2)
        self.assertTrue(all(e['amount'] == 1 for e in hits))
        self.assertEqual(s['sides']['b']['hp'], 98)
        self.assertEqual(s['sides']['a']['genesis_player_hits'], 5)
        self.assertTrue(any('（X为5）' in e.get('card', {}).get('description', '') for e in s['events'] if e['seq'] > seq))

    def test_j06_stops_at_victory_and_has_no_six_hit_cap(self):
        s = self.game(); s['sides']['a']['genesis_player_hits'] = 9
        seq=s['event_seq']; s = self.play(s, 'J06')
        self.assertEqual(s['sides']['b']['hp'], 91)
        s = self.game(); s['sides']['a']['genesis_player_hits'] = 9
        s['sides']['b']['hp'] = 2; seq = s['event_seq']
        s = self.play(s, 'J06')
        self.assertEqual(s['winner'], 'a')
        self.assertEqual(len([e for e in s['events'] if e['seq'] > seq and e['type'] == 'damage']), 2)

    def test_saved_counter_description_policy_and_privacy(self):
        s = self.game(); card = self.hold(s, 'J06'); baseline = deepcopy(card)
        self.genesis(s)
        damage(s, 'a', 'jiuyuan', 99)
        begin_turn(s, 'b'); begin_turn(s, 'a')
        s = json.loads(json.dumps(s))
        view = observe(s, 'a', include_previews=False)
        shown = next(c for c in view['sides']['a']['hand'] if c.get('card_id') == 'J06')
        self.assertTrue(shown['description'].endswith('（X为2）'))
        self.assertEqual(card['description'], baseline['description'])
        self.assertNotIn('（X为', CARDS['J06']['description'])
        self.assertEqual(view['policy_state']['sides']['a']['genesis_player_hits'], 2)
        hidden = observe(s, 'b', include_previews=False)['sides']['a']['hand']
        self.assertTrue(all(c.get('hidden') for c in hidden))

    def test_public_replay_patch_tracks_counter(self):
        s = self.game(); before = capture_public_board(s); s['_public_board'] = deepcopy(before)
        seq = s['event_seq']; self.genesis(s)
        after = replay_public_board(before, [e for e in s['events'] if e['seq'] > seq])
        self.assertEqual(after['sides']['a']['genesis_player_hits'], 2)
        self.assertEqual(after['sides']['a']['genesis_turn'], s['turn'])


if __name__ == '__main__':
    unittest.main()
