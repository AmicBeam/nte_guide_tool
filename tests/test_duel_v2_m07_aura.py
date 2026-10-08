"""M07 zeros the panel on enemy turns, preserving battle bonuses and stored stats."""
from app.modules.card_game.engine.duel_v2.modifiers import compatibility_flags
from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe
from app.modules.card_game.engine.duel_v2.state import attack_value, card_instance, hero
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board
from app.modules.card_game.engine.duel_v2.projection import preview


def fixture(owner='b', response=False):
    rush = next(d for d in STARTER_DECKS if d['id'] == 'weave-rush')
    foe = 'a' if owner == 'b' else 'b'
    s = new_game(seed=51, first_side=foe, skip_mulligan=True, decks={'a': rush, 'b': rush})
    for team in s['sides'].values():
        team['hand'] = []
    s['sides'][owner]['front'] = 'bohe'
    s['sides'][owner]['ap'] = 1
    hero(s, owner, 'bohe').update(shape='M07', hp=8, max_hp=8, base_attack=5)
    hero(s, owner, 'bohe')['flags']['atk_buff'] = 2
    if response:
        c = card_instance(s, 'M04', owner)
        s['sides'][owner]['hand'] = [c]
        s['sides'][owner]['revealed_ids'] = [c['instance_id']]
    return s


class M07AuraTest(unittest.TestCase):
    def test_panel_projection_replay_and_preview_share_aura(self):
        for owner in ('a', 'b'):
            s = fixture(owner)
            foe = 'a' if owner == 'b' else 'b'
            h = hero(s, owner, 'bohe')
            original = deepcopy(h)
            self.assertEqual(attack_value(h, s, owner), 0)
            view = observe(s, foe, include_previews=False)
            self.assertEqual(next(c['attack'] for c in view['sides'][owner]['characters'] if c['id'] == 'bohe'), 0)
            board = capture_public_board(s)
            self.assertEqual(next(c['attack'] for c in board['sides'][owner]['characters'] if c['id'] == 'bohe'), 0)
            self.assertEqual(preview(s, foe, {'type': 'attack', 'character_id': 'zero'})['counter'], 0)
            self.assertEqual(h, original)
            s['active_side'] = owner
            self.assertEqual(attack_value(h, s, owner), 9)  # form 5 + permanent 2 + buff 2
            s['active_side'] = foe
            h['shape'] = None
            self.assertEqual(attack_value(h, s, owner), 7)

    def test_response_bonus_survives_zero_panel(self):
        for owner in ('a', 'b'):
            s = fixture(owner, response=True)
            foe = 'a' if owner == 'b' else 'b'
            action = {'type': 'attack', 'character_id': 'zero'}
            self.assertEqual(preview(s, foe, action)['counter'], 2)
            after = apply_action(s, foe, action)
            battle = next(e for e in reversed(after['events']) if e['type'] == 'attack')
            self.assertEqual(battle['counter'], 2)
            self.assertEqual(attack_value(hero(after, owner, 'bohe'), after, owner), 0)
            self.assertEqual(hero(after, owner, 'bohe')['base_attack'], 5)
            self.assertEqual(compatibility_flags(hero(after, owner, 'bohe'))['atk_buff'], 2)

    def test_turn_switch_restores_panel_without_overwriting_growth(self):
        s = fixture('b')
        s = apply_action(s, 'a', {'type': 'end_turn'})
        self.assertEqual(attack_value(hero(s, 'b', 'bohe'), s, 'b'), 9)
        s = apply_action(s, 'b', {'type': 'end_turn'})
        self.assertEqual(attack_value(hero(s, 'b', 'bohe'), s, 'b'), 0)

    def test_generated_counter_matches_python(self):
        from app.modules.card_game.rl.rule_ir.compile_engine import compile_native_engine
        from app.modules.card_game.rl.rule_ir.pack_row import pack_python_row
        from app.modules.card_game.rl.rule_ir.compiled_oracle import map_python_action
        from app.modules.card_game.rl.rule_ir.layout import OFFSETS
        from app.modules.card_game.rl.gpu_duel.catalog import SEAT_INDEX, N_SEATS
        with tempfile.TemporaryDirectory() as path:
            # Isolated kernel test; unrelated old Il oy content gate stays strict in production.
            with patch('app.modules.card_game.rl.rule_ir.emit_engine.validate_compiled_deck'):
                engine = compile_native_engine(path)
            try:
                for owner in ('a', 'b'):
                    for response in (False, True):
                        s = fixture(owner, response)
                        foe = 'a' if owner == 'b' else 'b'
                        row = engine.launch_lists([pack_python_row(s)], [-1])[0]
                        action = {'type': 'attack', 'character_id': 'zero'}
                        index = map_python_action(row, s, action)
                        row = engine.launch_lists([row], [index])[0]
                        after = apply_action(s, foe, action)
                        off = OFFSETS['ch_hp'] + (0 if foe == 'a' else 1) * N_SEATS + SEAT_INDEX['zero']
                        self.assertEqual(row[off], hero(after, foe, 'zero')['hp'])
            finally:
                engine.close()
