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
from app.modules.card_game.engine.duel_v2.state import energy


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
                    self.assertEqual(c['harmony_max'], 1 if turn >= 20 else 2)

    def test_minimum_alive_energy_and_seed_replay(self):
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
        self.assertEqual([chars[c]['energy'] for c in ids[2:]], [2, 2])
        self.assertTrue(all(h['energy'] == 0 for h in s['sides']['b']['characters'].values()))

    def test_tie_grants_only_one_and_varies_across_seeds(self):
        selected = set()
        for seed in range(12):
            s = self.game(5)
            s['rng'] = seed
            begin_turn(s, 'a')
            energized = [c['id'] for c in s['sides']['a']['characters'].values() if c['energy']]
            self.assertEqual(len(energized), 1)
            selected.update(energized)
        self.assertGreater(len(selected), 1)

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

    def test_one_harmony_payment_after_twenty(self):
        s = self.game(20)
        team = s['sides']['a']
        team['last_front'] = 'baicang'
        team['characters']['baicang']['harmony'] = 1
        cid = next(cid for cid, h in team['characters'].items() if h['attribute'] == '灵')
        self.assertEqual(harmony_payer(s, 'a', cid), 'baicang')
        apply_enter_harmony(EffectContext(s, 'a', cid, {'side': 'a'}), 'baicang', '覆纹')
        self.assertEqual(team['characters']['baicang']['harmony'], 0)

    def test_disabled_baseline_and_legacy_snapshot(self):
        s = self.game(20, escalation=False)
        self.assertIsNone(public_progress(s))
        self.assertEqual(observe(s, 'a')['sides']['a']['ultimate_remaining'], 1)
        s.pop('escalation_enabled')
        self.assertIsNone(public_progress(s))


if __name__ == '__main__':
    unittest.main()
