"""Public protection frames follow rules and survive replay serialization."""
import json
import unittest

from tests import test_duel_v2_surplus_rework as fixtures
from app.modules.card_game.engine.duel_v2 import observe
from app.modules.card_game.engine.duel_v2.entities import clone_state
from app.modules.card_game.engine.duel_v2.modifiers import add_effect
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.state import hero, protection_state, log


class ProtectionProjectionTest(unittest.TestCase):
    def test_public_traits_live_replay_and_down(self):
        s = fixtures.SurplusReworkTest().game(team=('bohe', 'baicang', 'canhong', 'heiyu'))
        opening = capture_public_board(s)
        seq = s['event_seq']
        s['_public_board'] = opening
        hero(s, 'a', 'bohe')['awakened'] = True
        hero(s, 'a', 'baicang')['flags']['hp_floor'] = 1
        h = hero(s, 'a', 'heiyu')
        add_effect(h, 'heiyu.combat_immune', 'rule.combat_immune', True, op='allow',
                   source_entity_id=h.entity_id, source_key='P01', clear_on=('battle', 'down'))
        log(s, '保护效果生效')
        s = clone_state(json.loads(json.dumps(s)))
        events = [e for e in observe(s, 'a')['presentation']['events'] if e['seq'] > seq]
        replay = replay_public_board(opening, events)
        for board in (observe(s, 'a', include_previews=False), observe(s, 'b', include_previews=False),
                      capture_public_board(s), replay):
            shown = {c['id']: c for c in board['sides']['a']['characters']}
            self.assertTrue(shown['bohe']['noncombat_immune'])
            self.assertEqual(shown['baicang']['hp_floor'], 1)
            self.assertTrue(shown['heiyu']['combat_immune'])
            self.assertFalse(shown['canhong']['combat_immune'])
        hero(s, 'a', 'bohe')['awakened'] = False
        self.assertFalse(protection_state(s, 'a', 'bohe')['noncombat_immune'])
        h = hero(s, 'a', 'baicang')
        h['hp'] = 0
        self.assertEqual(protection_state(s, 'a', 'baicang')['hp_floor'], 0)

    def test_battle_immunity_patch_is_temporary(self):
        for cid, card in [('canhong', 'C05'), ('heiyu', 'P01')]:
            with self.subTest(card=card):
                fixture = fixtures.SurplusReworkTest()
                s = fixture.game(team=('bohe', 'baicang', 'canhong', 'heiyu'))
                seq = s['event_seq']
                s['sides']['b']['front'] = 'baicang'
                s = fixture.play(s, card)
                events = observe(s, 'a')['presentation']['events']
                patches = [c for e in events if e['seq'] > seq
                           for c in e.get('patch', {}).get('sides', {}).get('a', {}).get('characters', [])
                           if c['id'] == cid]
                self.assertTrue(any(c.get('combat_immune') for c in patches))
                self.assertFalse(protection_state(s, 'a', cid)['combat_immune'])

    def test_canhong_life_floor_is_only_shown_during_eligible_battle(self):
        from app.modules.card_game.engine.duel_v2.continuous import apply_dot
        fixture = fixtures.SurplusReworkTest()
        s = fixture.game(team=('bohe', 'baicang', 'canhong', 'heiyu'))
        apply_dot(s, 'b', 'nightmare', by='a', source_cid='canhong', stacks=5, duration=2)
        seq = s['event_seq']
        s = fixture.play(s, 'C01')
        events = observe(s, 'a')['presentation']['events']
        patches = [c for e in events if e['seq'] > seq
                   for c in e.get('patch', {}).get('sides', {}).get('a', {}).get('characters', [])
                   if c['id'] == 'canhong']
        self.assertTrue(any(c.get('hp_floor') == 1 for c in patches))
        self.assertEqual(protection_state(s, 'a', 'canhong')['hp_floor'], 0)
