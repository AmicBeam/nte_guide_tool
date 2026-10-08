"""User tabletop rule 2026-09-21: R07 no longer requires the front position."""
import unittest
from copy import deepcopy

from tests import test_duel_v2_surplus_rework as fixtures
from app.modules.card_game.engine.duel_v2 import observe
from app.modules.card_game.engine.duel_v2.combat import resolve_genesis
from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation
from app.modules.card_game.engine.duel_v2.state import hero
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board


class R07Test(unittest.TestCase):
    def test_bench_weapon_hits_all_opponents_once_before_deferred_entry(self):
        s = fixtures.SurplusReworkTest().game()
        fixtures.SurplusReworkTest().delay(s)
        hero(s, 'a', 'zhenhong')['shape'] = 'R07'
        s['sides']['a']['front'] = 'zero'
        s['sides']['b']['front'] = 'yi'
        hero(s, 'b', 'yi').update(hp=60, max_hp=60)
        before = capture_public_board(s)
        s['_public_board'] = deepcopy(before)
        seq = s['event_seq']
        resolve_genesis(operation(s, 'a', 'zero'))
        self.assertEqual(s['sides']['a']['front'], 'zero')
        # Jiuyuan repeats genesis, but the weapon fires once per surplus notification.
        events = [e for e in observe(s, 'a')['presentation']['events'] if e['seq'] > seq]
        hits = [e for e in events if '「梦的边缘」触发' in e.get('text', '')]
        self.assertEqual(len(hits), 4)
        self.assertEqual({e['target'] for e in hits}, {'b:' + cid for cid in s['sides']['b']['order']})
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 3)
        self.assertEqual(hero(s, 'b', 'yi')['hp'], 56)
        self.assertEqual(replay_public_board(before, events), capture_public_board(s))
        finish_operation(s)

    def test_no_weapon_or_knocked_down_does_not_trigger(self):
        for equipped, alive in ((False, True), (True, False)):
            with self.subTest(equipped=equipped, alive=alive):
                s = fixtures.SurplusReworkTest().game()
                fixtures.SurplusReworkTest().delay(s)
                h = hero(s, 'a', 'zhenhong')
                h['shape'] = 'R07' if equipped else None
                if not alive:
                    h.update(hp=0, down_turns=3)
                s['sides']['a']['front'] = 'zero'
                seq = s['event_seq']
                resolve_genesis(operation(s, 'a', 'zero'))
                self.assertFalse(any('「梦的边缘」触发' in e.get('text', '')
                                     for e in s['events'] if e['seq'] > seq))
