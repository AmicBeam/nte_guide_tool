"""User-confirmed 2026-09-19 shield expiry and global-turn family instant rules."""
import tempfile
import unittest
from unittest.mock import patch

from app.modules.card_game.engine.duel_v2 import apply_action, legal_actions, new_game, observe
from app.modules.card_game.engine.duel_v2.state import card_instance


class FamilyGlobalTurnTest(unittest.TestCase):
    def test_boundary_both_sides_and_instant_allowance(self):
        for side, turn, own_turn, instant in [('a', 3, 2, False), ('b', 4, 2, False),
                                               ('a', 5, 3, True), ('b', 6, 3, True)]:
            with self.subTest(side=side, turn=turn):
                s = new_game(seed=12, first_side='a', skip_mulligan=True)
                s['turn'], s['active_side'] = turn, side
                team = s['sides'][side]
                team['turn_count'], team['ap'] = own_turn, 0
                card = card_instance(s, 'NF01', side)
                team['hand'] = [card, card_instance(s, 'NF01', side)]
                shown = observe(s, side)['sides'][side]['hand'][0]
                self.assertEqual(bool(shown.get('instant')), instant)
                action = {'type': 'play_card', 'card_id': card['instance_id']}
                can_play = any(a.get('card_id') == card['instance_id'] for a in legal_actions(s, side))
                self.assertEqual(can_play, instant)
                if instant:
                    s = apply_action(s, side, action)
                    self.assertEqual(s['sides'][side]['ap'], 0)
                    self.assertFalse(any(a['type'] == 'play_card' for a in legal_actions(s, side)))


class GeneratedTurnBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.modules.card_game.rl.rule_ir.compile_engine import compile_native_engine
        cls.tmp = tempfile.TemporaryDirectory()
        # Isolate these two rules from the existing unrelated Il oy content/IR
        # completeness gap. Production compilation retains the full gate.
        with patch('app.modules.card_game.rl.rule_ir.emit_engine.validate_compiled_deck'):
            cls.engine = compile_native_engine(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        cls.tmp.cleanup()

    def test_shield_expiry_and_family_legality(self):
        from app.modules.card_game.rl.rule_ir.pack_row import pack_python_row, legal_entries
        from app.modules.card_game.rl.rule_ir.compiled_oracle import map_python_action
        from app.modules.card_game.rl.rule_ir.layout import OFFSETS
        from app.modules.card_game.rl.gpu_duel.catalog import ACT_PLAY
        for side in ('a', 'b'):
            si = 0 if side == 'a' else 1
            s = new_game(seed=12, first_side=side, skip_mulligan=True)
            for team in s['sides'].values():
                team['hand'] = []
                team['shield'] = 3
            row = self.engine.launch_lists([pack_python_row(s)], [-1])[0]
            index = map_python_action(row, s, {'type': 'end_turn'})
            row = self.engine.launch_lists([row], [index])[0]
            self.assertEqual(row[OFFSETS['shield'] + si], 3)
            self.assertEqual(row[OFFSETS['shield'] + 1 - si], 0)
            for turn, expected in ((4, False), (5, True), (6, True)):
                s['turn'] = turn
                s['sides'][side]['turn_count'] = 3
                s['sides'][side]['ap'] = 0
                s['sides'][side]['hand'] = [card_instance(s, 'NF01', side)]
                row = self.engine.launch_lists([pack_python_row(s)], [-1])[0]
                self.assertEqual(any(e['type'] == ACT_PLAY for e in legal_entries(row)), expected)
