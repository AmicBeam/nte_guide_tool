"""One title per harmony trigger; silent status events still update replay boards."""
import unittest
from copy import deepcopy

from app.modules.card_game.engine.duel_v2 import new_game, observe
from app.modules.card_game.engine.duel_v2.combat import apply_enter_harmony
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.state import front_debuff, hero


class HarmonyPresentationTest(unittest.TestCase):
    def resolve(self, kind, existing=None):
        state = new_game(seed=9, first_side='a', skip_mulligan=True)
        hero(state, 'a', 'zero')['harmony'] = 2
        if existing:
            front_debuff(state, 'b')[existing] = {'left': 2, 'by': 'a'}
            state['sides']['b']['front'] = 'zero'
        before = capture_public_board(state)
        state['_public_board'] = deepcopy(before)
        seq = state['event_seq']
        apply_enter_harmony(EffectContext(state, 'a', 'nanali', {}), 'zero', kind)
        return state, before, seq

    def test_status_harmonies_show_one_title_and_keep_public_patches(self):
        for kind, status in [('延滞', 'delay'), ('浊燃', 'burn'), ('黯星', 'star'),
                             ('覆纹', 'weave'), ('浸染', 'stain')]:
            with self.subTest(kind=kind):
                state, before, seq = self.resolve(kind)
                self.assertEqual(hero(state, 'a', 'zero')['harmony'], 0)
                self.assertIn(status, front_debuff(state, 'b'))
                for viewer in ('a', 'b'):
                    events = [e for e in observe(state, viewer)['presentation']['events']
                              if e['seq'] > seq]
                    harmonies = [e for e in events if e['type'] == 'harmony']
                    self.assertEqual(len(harmonies), 2)
                    self.assertEqual([e['present'] for e in harmonies], [True, False])
                    self.assertIn('触发' + kind, harmonies[0]['text'])
                    self.assertIn('获得' + kind, harmonies[1]['text'])
                    self.assertIn(status, harmonies[1]['patch']['sides']['b']['front_debuff'])
                    self.assertEqual(replay_public_board(before, events), capture_public_board(state))

    def test_dissonance_keeps_its_own_title(self):
        for kind, existing in [('浊燃', 'star'), ('黯星', 'burn')]:
            with self.subTest(kind=kind):
                state, _, seq = self.resolve(kind, existing)
                events = [e for e in observe(state, 'a')['presentation']['events']
                          if e['seq'] > seq and e['type'] == 'harmony' and e['present']]
                self.assertEqual(len(events), 2)
                self.assertIn('触发' + kind, events[0]['text'])
                self.assertIn('触发失谐', events[1]['text'])

    def test_genesis_still_shows_one_title_with_jiuyuan_repeat(self):
        state, _, seq = self.resolve('创生')
        events = [e for e in observe(state, 'a')['presentation']['events'] if e['seq'] > seq]
        titles = [e for e in events if e['type'] == 'harmony' and e['present']]
        self.assertEqual(len(titles), 1)
        self.assertIn('触发创生', titles[0]['text'])
