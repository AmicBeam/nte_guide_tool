"""Regression coverage for the user's 6/13/20 global-turn pacing rules."""
import unittest
from copy import deepcopy

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, legal_actions, observe
from app.modules.card_game.engine.duel_v2.flow import begin_turn
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.engine.duel_v2.escalation import public_progress
from app.modules.card_game.engine.duel_v2.combat import harmony_payer, apply_enter_harmony
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, diff_public
from app.modules.card_game.engine.duel_v2.state import energy, card_instance


class EscalationTest(unittest.TestCase):
    def game(self, turn=1, **kwargs):
        deck = next(d for d in STARTER_DECKS if d['id'] == 'weave-rush')
        s = new_game(seed=31, decks={'a': deck, 'b': deck}, first_side='a', skip_mulligan=True, **kwargs)
        s['turn'] = turn - 1
        begin_turn(s, 'a')
        return s

    def test_thresholds_and_countdown(self):
        for turn, stage, left in [(5, 0, 1), (6, 1, 7), (12, 1, 1), (13, 2, 7), (19, 2, 1), (20, 3, None)]:
            with self.subTest(turn=turn):
                s = self.game(turn)
                progress = public_progress(s)
                self.assertEqual(progress['stage'], stage)
                if left:
                    self.assertTrue(progress['hint'].startswith(f'{left}回合后，'))
                self.assertEqual(observe(s, 'a')['sides']['a']['ultimate_remaining'], 2 if turn >= 6 else 1)
                chars = observe(s, 'a')['sides']['a']['characters']
                for c in chars:
                    self.assertEqual(c['energy_max'], (6 if c['attribute'] in ('暗', '魂', '咒') else 5) - (turn >= 13))
                    self.assertEqual(c['harmony_max'], 1 if turn >= 13 else 2)
                self.assertEqual(s['sides']['a']['ap'], 3 if turn >= 20 else 2)

    def test_all_alive_energy_and_seed_replay(self):
        s = self.game(5)
        chars = s['sides']['a']['characters']
        ids = list(chars)
        for h in chars.values():
            h['energy'] = 2
        chars[ids[0]].update(hp=0, down_turns=3, energy=0)
        chars[ids[1]]['energy'] = 1
        original = deepcopy(s)
        begin_turn(s, 'a')
        begin_turn(original, 'a')
        self.assertEqual(s, original)
        self.assertEqual(chars[ids[0]]['energy'], 0)
        self.assertEqual(chars[ids[1]]['energy'], 2)
        self.assertEqual([chars[c]['energy'] for c in ids[2:]], [3, 3])
        self.assertTrue(all(h['energy'] == 0 for h in s['sides']['b']['characters'].values()))

    def test_all_alive_gain_without_consuming_randomness(self):
        for seed in range(12):
            s = self.game(5)
            s['rng'] = seed
            begin_turn(s, 'a')
            energized = [c['id'] for c in s['sides']['a']['characters'].values() if c['energy']]
            self.assertEqual(len(energized), 4)
            self.assertEqual(s['rng'], seed)

    def test_full_energy_and_newly_revived(self):
        s = self.game(5)
        chars = s['sides']['a']['characters']
        chars['baicang']['energy'] = 6
        chars['zero'].update(hp=0, down_turns=1, energy=2)
        begin_turn(s, 'a')
        self.assertEqual(chars['baicang']['energy'], 6)
        self.assertEqual(chars['zero']['energy'], 3)

    def test_zhenhong_bench_ultimate_and_xun_cannot_gain(self):
        deck = {'id': 'energy-gates', 'name': '回能限制', 'character_ids': ['zhenhong', 'xun', 'zero', 'jiuyuan'],
                'card_ids': [f'{prefix}{i:02}' for prefix in ('R', 'X', 'Z', 'J') for i in range(1, 9)]}
        s = new_game(seed=31, decks={'a': deck, 'b': deck}, first_side='a', skip_mulligan=True)
        s['turn'] = 5
        begin_turn(s, 'a')
        chars = s['sides']['a']['characters']
        self.assertEqual(chars['zhenhong']['energy'], 0)
        self.assertEqual(chars['xun']['energy'], 2)  # clock only, one per own turn
        s['sides']['a']['front'] = 'zhenhong'
        begin_turn(s, 'a')
        self.assertEqual(chars['zhenhong']['energy'], 1)
        chars['zhenhong']['awakened'] = True
        chars['zhenhong']['ultimate_turns'] = 2
        begin_turn(s, 'a')
        self.assertEqual(chars['zhenhong']['energy'], 1)

    def test_caps_clamp_both_teams_including_downed(self):
        s = self.game(12)
        for team in s['sides'].values():
            for h in team['characters'].values():
                h.update(energy=6, harmony=2)
        s['sides']['b']['characters']['baicang'].update(hp=0, down_turns=2)
        begin_turn(s, 'a')
        for team in s['sides'].values():
            for h in team['characters'].values():
                self.assertEqual(h['energy'], 5 if h['attribute'] == '咒' else 4)
                self.assertEqual(h['harmony'], 1)
        s['turn'] = 19
        before = capture_public_board(s)
        begin_turn(s, 'a')
        patch = diff_public(before, capture_public_board(s))
        self.assertEqual(patch['escalation']['stage'], 3)
        for team in s['sides'].values():
            self.assertTrue(all(h['harmony'] == 1 for h in team['characters'].values()))
        ctx = EffectContext(s, 'a', 'baicang', {'side': 'a'})
        ctx.grant_harmony('baicang', 2)
        energy(s, 'a', 'baicang', 10)
        self.assertEqual(ctx.character['harmony'], 1)
        self.assertEqual(ctx.character['energy'], 5)

    def test_two_ultimates_and_third_rejected_without_action_points(self):
        s = self.game(6)
        team = s['sides']['a']
        team['ap'] = 0
        for h in team['characters'].values():
            h['energy'] = 6 if h['attribute'] == '咒' else 5
        original = deepcopy(s)
        for index in range(2):
            action = next(a for a in legal_actions(s, 'a') if a['type'] == 'ultimate')
            s = apply_action(s, 'a', action)
            self.assertEqual(observe(s, 'a')['sides']['a']['ultimate_remaining'], 1 - index)
            self.assertEqual(s['sides']['a']['characters'][action['character_id']]['energy'], 0)
        self.assertEqual(team, original['sides']['a'])  # caller state stays unchanged
        self.assertEqual(team['ultimates_used'], 0)
        self.assertFalse(any(a['type'] == 'ultimate' for a in legal_actions(s, 'a')))
        with self.assertRaises(ValueError):
            apply_action(s, 'a', action)
        begin_turn(s, 'a')
        self.assertEqual(observe(s, 'a')['sides']['a']['ultimate_remaining'], 2)

    def test_one_harmony_payment_after_thirteen(self):
        s = self.game(13)
        team = s['sides']['a']
        team['last_front'] = 'baicang'
        team['characters']['baicang']['harmony'] = 1
        cid = next(cid for cid, h in team['characters'].items() if h['attribute'] == '灵')
        self.assertEqual(harmony_payer(s, 'a', cid), 'baicang')
        apply_enter_harmony(EffectContext(s, 'a', cid, {'side': 'a'}), 'baicang', '覆纹')
        self.assertEqual(team['characters']['baicang']['harmony'], 0)

    def test_turn_twenty_recovers_one_physical_buildable_card(self):
        s = self.game(19)
        team = s['sides']['a']
        ordinary = [card_instance(s, cid, 'a') for cid in ('B01', 'M02', 'Z03')]
        derived = card_instance(s, 'NF01', 'a')
        copy = card_instance(s, 'B01', 'a', copy=True)
        team['discard'] = [derived, *ordinary, copy]
        original = deepcopy(s)
        hand_n, deck_n = len(team['hand']), len(team['deck'])
        begin_turn(s, 'a')
        begin_turn(original, 'a')
        self.assertEqual(s, original)
        recovered = [c for c in ordinary if c in team['hand']]
        self.assertEqual(len(recovered), 1)
        self.assertIs(next(c for c in team['hand'] if c == recovered[0]), recovered[0])
        self.assertIn(derived, team['discard'])
        self.assertIn(copy, team['discard'])
        self.assertEqual(len(team['hand']), hand_n + 2)  # recovery plus usual one draw
        self.assertEqual(len(team['deck']), deck_n - 1)
        gain = next(e for e in s['events'] if e['type'] == 'gain' and '白热化回收' in e['text'])
        self.assertEqual(gain['card']['card_id'], recovered[0]['card_id'])
        self.assertEqual(team['ap'], 3)

    def test_recovery_threshold_empty_pool_and_full_hand(self):
        for turn, ordinary, full, expected in [(19, True, False, 0), (20, False, False, 0),
                                               (20, True, True, 0), (20, True, False, 1)]:
            with self.subTest(turn=turn, ordinary=ordinary, full=full):
                s = self.game(18)
                team = s['sides']['a']
                card = card_instance(s, 'B01' if ordinary else 'NF01', 'a')
                team['discard'] = [card]
                team['hand'] = [card_instance(s, 'B02', 'a') for _ in range(10 if full else 0)]
                s['turn'] = turn - 1
                begin_turn(s, 'a')
                self.assertEqual(sum(c is card for c in team['hand']), expected)
                self.assertEqual(card in team['discard'], not bool(expected))

    def test_recovery_is_random_and_does_not_replace_usual_draw(self):
        selected = set()
        for seed in range(12):
            s = self.game(19)
            team = s['sides']['a']
            team['discard'] = [card_instance(s, cid, 'a') for cid in ('B01', 'B02', 'M03')]
            s['rng'] = seed
            before = {c['instance_id'] for c in team['discard']}
            begin_turn(s, 'a')
            selected.update(c['card_id'] for c in team['hand'] if c['instance_id'] in before)
        self.assertGreater(len(selected), 1)
        s = self.game(19)
        s['sides']['a']['deck'] = []
        s['sides']['a']['discard'] = [card_instance(s, 'B01', 'a')]
        begin_turn(s, 'a')
        self.assertEqual(s['reason'], 'empty_draw')

    def test_recovery_does_not_require_a_living_character(self):
        s = self.game(19)
        team = s['sides']['a']
        for h in team['characters'].values():
            h.update(hp=0, down_turns=3)
        card = card_instance(s, 'B01', 'a')
        team['discard'] = [card]
        before = [h['energy'] for h in team['characters'].values()]
        begin_turn(s, 'a')
        self.assertIn(card, team['hand'])
        self.assertEqual([h['energy'] for h in team['characters'].values()], before)

    def test_action_points_refresh_and_response_carry_without_cap(self):
        s = self.game(19)
        team = s['sides']['a']
        team['extra_ap'] = 4
        begin_turn(s, 'a')
        self.assertEqual(team['ap'], 7)
        s = apply_action(s, 'a', {'type': 'end_turn'})
        self.assertEqual(s['sides']['a']['ap'], 7)
        s = apply_action(s, 'b', {'type': 'end_turn'})
        self.assertEqual(s['sides']['a']['ap'], 3)

    def test_disabled_baseline_and_legacy_snapshot(self):
        s = self.game(20, escalation=False)
        self.assertIsNone(public_progress(s))
        self.assertEqual(s['sides']['a']['ap'], 2)
        self.assertEqual(observe(s, 'a')['sides']['a']['ultimate_remaining'], 1)
        s.pop('escalation_enabled')
        self.assertIsNone(public_progress(s))


if __name__ == '__main__':
    unittest.main()
