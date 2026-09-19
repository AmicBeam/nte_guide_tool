"""2026-09-20 user rule: eligibility is captured at departure, never by later bench gains."""
import json
import unittest
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe
from app.modules.card_game.engine.duel_v2.flow import begin_turn
from app.modules.card_game.engine.duel_v2.state import hero, move_out, card_instance, damage
from app.modules.card_game.engine.duel_v2.combat import harmony_marks
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board


class HarmonyResidueTest(unittest.TestCase):
    def game(self, harmony=1):
        s = new_game(seed=9, first_side='a', skip_mulligan=True)
        for t in s['sides'].values():
            t['hand'] = []
        s['sides']['a']['front'] = 'nanali'
        hero(s, 'a', 'nanali')['harmony'] = harmony
        return s

    def family(self, s):
        card = card_instance(s, 'NF01', 'a')
        s['sides']['a']['hand'] = [card]
        return apply_action(s, 'a', {'type': 'play_card', 'card_id': card['instance_id']})

    def test_benched_family_fill_does_not_create_residue_or_preview(self):
        s = self.game()
        begin_turn(s, 'a')
        self.assertIsNone(s['sides']['a']['last_front'])
        s = self.family(s)
        self.assertEqual(hero(s, 'a', 'nanali')['harmony'], 2)
        self.assertFalse(harmony_marks(s, 'a')['available'])
        self.assertFalse(capture_public_board(s)['sides']['a']['harmony_available'])
        v = observe(s, 'a')
        self.assertFalse(v['sides']['a']['harmony_available'])
        action = {'type': 'attack', 'character_id': 'zero'}
        row = next(r for r in v['legal_actions'] if r['action'] == action)
        self.assertEqual(row['preview']['harmony'], '不触发')
        s['events'] = []
        s = apply_action(json.loads(json.dumps(s)), 'a', action)
        self.assertFalse(any(e['type'] == 'harmony' for e in s['events']))
        self.assertEqual(hero(s, 'a', 'nanali')['harmony'], 2)

    def test_still_front_can_fill_and_switch_during_turn(self):
        s = self.game()
        s['sides']['a']['ap'] = 2
        s = self.family(s)
        self.assertTrue(harmony_marks(s, 'a')['available'])
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(hero(s, 'a', 'nanali')['harmony'], 0)
        self.assertTrue(s['sides']['a']['harmonized']['zero'])
        self.assertIsNone(s['sides']['a']['last_front'])

    def test_full_automatic_return_survives_json_and_pays_once(self):
        s = self.game(2)
        begin_turn(s, 'a')
        self.assertEqual(s['sides']['a']['last_front'], 'nanali')
        self.assertTrue(harmony_marks(s, 'a')['available'])
        s = apply_action(json.loads(json.dumps(s)), 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertIsNone(s['sides']['a']['last_front'])
        self.assertEqual(hero(s, 'a', 'nanali')['harmony'], 0)

    def test_manual_return_captures_fullness_at_departure(self):
        for amount in (1, 2):
            s = self.game(amount)
            move_out(s, 'a', 'nanali')
            hero(s, 'a', 'nanali')['harmony'] = 2
            self.assertEqual(s['sides']['a']['last_front'], 'nanali' if amount == 2 else None)
            self.assertEqual(harmony_marks(s, 'a')['available'], amount == 2)

    def test_twentieth_turn_lowers_cap_before_return(self):
        for turn, expected in ((19, None), (20, 'nanali')):
            s = self.game(1)
            s['turn'] = turn - 1
            begin_turn(s, 'a')
            self.assertEqual(s['sides']['a']['last_front'], expected)
            self.assertEqual(harmony_marks(s, 'a')['available'], turn == 20)
            if turn == 20:
                s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
                self.assertEqual(hero(s, 'a', 'nanali')['harmony'], 0)

    def test_collapsed_front_stays_live_without_capturing_residue(self):
        s = self.game(1)
        hero(s, 'a', 'nanali')['flags']['collapse'] = {'side': 'b', 'until': 99}
        begin_turn(s, 'a')
        self.assertEqual(s['sides']['a']['front'], 'nanali')
        self.assertIsNone(s['sides']['a']['last_front'])
        hero(s, 'a', 'nanali')['harmony'] = 2
        self.assertTrue(harmony_marks(s, 'a')['available'])

    def test_death_expires_residue(self):
        s = self.game(2)
        begin_turn(s, 'a')
        damage(s, 'a', 'nanali', 100)
        self.assertIsNone(s['sides']['a']['last_front'])
        self.assertFalse(harmony_marks(s, 'a')['available'])

    def test_haiyue_earns_harmony_before_return_and_leaves_it(self):
        from app.modules.card_game.content.duel_v2 import CARDS
        ids = ['haiyue', 'hathor', 'zero', 'nanali']
        build = dict(id='test', name='test', character_ids=ids,
                     card_ids=[key for cid in ids for key, card in CARDS.items()
                               if card['character_id'] == cid and not card.get('derived')
                               for _ in range(card['starter_copies'])])
        for amount in (0, 1):
            s = new_game(seed=9, decks={'a': build, 'b': build},
                         first_side='a', skip_mulligan=True, escalation=False)
            for team in s['sides'].values():
                team.update(hand=[], ap=10)
            hero(s, 'a', 'haiyue')['harmony'] = amount
            s['events'] = []
            s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'haiyue'})
            self.assertIsNone(s['sides']['a']['front'])
            self.assertEqual(hero(s, 'a', 'haiyue')['harmony'], amount + 1)
            self.assertEqual(s['sides']['a']['last_front'], 'haiyue' if amount else None)
            events = s['events']
            gain = next(i for i, e in enumerate(events) if e['type'] == 'resource' and e.get('source') == 'haiyue')
            leave = next(i for i, e in enumerate(events) if e['type'] == 'move' and e.get('source') == 'haiyue')
            self.assertLess(gain, leave)
            hero(s, 'a', 'haiyue')['harmony'] = 2
            s['sides']['a']['normal_attack_available'] = True
            s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'hathor'})
            self.assertEqual(bool(s['sides']['a']['harmonized'].get('hathor')), bool(amount))

    def test_legacy_snapshot_does_not_invent_turn_start_evidence(self):
        s = self.game(2)
        s.pop('harmony_residue_version')
        s['sides']['a'].update(front=None, last_front='nanali')
        self.assertFalse(harmony_marks(s, 'a')['available'])
        begin_turn(s, 'a')
        self.assertIsNone(s['sides']['a']['last_front'])
        self.assertEqual(s['harmony_residue_version'], 1)

class GeneratedHarmonyResidueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        from unittest.mock import patch
        from app.modules.card_game.rl.rule_ir.compile_engine import compile_native_engine
        cls.tmp = tempfile.TemporaryDirectory()
        # Isolate this rule from the existing unrelated Il oy content/IR gap.
        # Production compilation keeps its full content-completeness gate.
        with patch('app.modules.card_game.rl.rule_ir.emit_engine.validate_compiled_deck'):
            cls.engine = compile_native_engine(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        cls.tmp.cleanup()

    def test_return_snapshot_and_twentieth_turn(self):
        from app.modules.card_game.rl.rule_ir.pack_row import pack_python_row
        from app.modules.card_game.rl.rule_ir.compiled_oracle import map_python_action
        from app.modules.card_game.rl.rule_ir.layout import OFFSETS
        from app.modules.card_game.rl.gpu_duel.catalog import N_SEATS, SEAT_INDEX
        nana, zero = SEAT_INDEX['nanali'], SEAT_INDEX['zero']
        for turn, amount, expected in ((19, 1, -1), (19, 2, nana), (20, 1, nana)):
            s = new_game(seed=9, first_side='a', skip_mulligan=True)
            s['turn'] = turn - 1
            for team in s['sides'].values():
                team['hand'] = []
            s['sides']['b']['front'] = 'nanali'
            hero(s, 'b', 'nanali')['harmony'] = amount
            row = self.engine.launch_lists([pack_python_row(s)], [-1])[0]
            index = map_python_action(row, s, {'type': 'end_turn'})
            row = self.engine.launch_lists([row], [index])[0]
            self.assertEqual(row[OFFSETS['last_front'] + 1], expected)
            self.assertEqual(row[OFFSETS['front'] + 1], -1)
            h = OFFSETS['ch_harmony'] + N_SEATS + nana
            row[h] = 1 if turn == 20 else 2  # bench fill cannot invent departure evidence
            index = map_python_action(row, s, {'type': 'attack', 'character_id': 'zero'})
            row = self.engine.launch_lists([row], [index])[0]
            self.assertEqual(row[h], 0 if expected >= 0 else 2)
            self.assertEqual(row[OFFSETS['last_front'] + 1], -1)

    def test_legacy_packer_drops_unproven_residue(self):
        from app.modules.card_game.rl.rule_ir.pack_row import pack_python_row
        from app.modules.card_game.rl.rule_ir.layout import OFFSETS
        s = new_game(seed=9, first_side='a', skip_mulligan=True)
        s.pop('harmony_residue_version')
        s['sides']['a']['last_front'] = 'zero'
        self.assertEqual(pack_python_row(s)[OFFSETS['last_front']], -1)
