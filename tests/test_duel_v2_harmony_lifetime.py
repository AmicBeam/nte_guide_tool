import importlib.util
import unittest
from copy import deepcopy

from app.modules.card_game.engine.duel_v2 import apply_action, new_game
from app.modules.card_game.engine.duel_v2.combat import harmony_marks
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.engine.duel_v2.flow import begin_turn
from app.modules.card_game.engine.duel_v2.state import card_instance, damage, hero, move_out
from app.modules.card_game.content.duel_v2.characters.common import revive_character


class HarmonyLifetimeTest(unittest.TestCase):
    def game(self):
        state = new_game(seed=9, first_side='a', skip_mulligan=True)
        for side in ('a', 'b'):
            state['sides'][side]['hand'] = []
        team = state['sides']['a']
        team.update(front='zero', last_front='nanali', ap=2)
        hero(state, 'a', 'zero').update(harmony=2, energy=3)
        return state

    def revive(self, state):
        revive_character(EffectContext(state, 'a', 'iloy', {'side': 'a'}), 'a', 'zero')

    def test_front_or_bench_source_death_expires_old_opportunity(self):
        for benched in (False, True):
            with self.subTest(benched=benched):
                state = self.game()
                if benched:
                    move_out(state, 'a', 'zero')
                self.assertTrue(harmony_marks(state, 'a')['available'])
                damage(state, 'a', 'zero', 100)
                self.assertIsNone(state['sides']['a']['last_front'])
                self.assertFalse(harmony_marks(state, 'a')['available'])
                self.revive(state)
                self.assertEqual((hero(state, 'a', 'zero')['harmony'], hero(state, 'a', 'zero')['energy']), (2, 3))
                self.assertFalse(harmony_marks(state, 'a')['available'])
                state['events'] = []
                state = apply_action(state, 'a', {'type': 'attack', 'character_id': 'nanali'})
                self.assertFalse(any(e['type'] == 'harmony' for e in state['events']))

    def test_normal_return_keeps_opportunity_and_unrelated_death_does_not_clear_it(self):
        state = self.game()
        move_out(state, 'a', 'zero')
        damage(state, 'a', 'jiuyuan', 100)
        self.assertEqual(state['sides']['a']['last_front'], 'zero')
        self.assertTrue(harmony_marks(state, 'a')['available'])

    def test_new_sortie_after_revive_can_create_a_new_opportunity(self):
        state = self.game()
        damage(state, 'a', 'zero', 100)
        self.revive(state)
        state = apply_action(state, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertTrue(harmony_marks(state, 'a')['available'])
        card = card_instance(state, 'N03', 'a')
        state['sides']['a']['hand'] = [card]
        state['events'] = []
        state = apply_action(state, 'a', {'type': 'play_card', 'card_id': card['instance_id']})
        self.assertTrue(any(e['type'] == 'harmony' for e in state['events']))

    def test_old_downed_snapshot_cannot_regain_opportunity_on_either_recovery(self):
        for natural in (False, True):
            with self.subTest(natural=natural):
                state = self.game()
                state['sides']['a'].update(front=None, last_front='zero')
                hero(state, 'a', 'zero').update(hp=0, down_turns=1)
                if natural:
                    begin_turn(state, 'a')
                else:
                    self.revive(state)
                self.assertEqual(hero(state, 'a', 'zero')['down_turns'], 0)
                self.assertIsNone(state['sides']['a']['last_front'])
                self.assertFalse(harmony_marks(state, 'a')['available'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is required')
    def test_gpu_knockdown_clears_front_and_benched_source(self):
        from app.modules.card_game.rl.gpu_duel.pack import pack_python
        from app.modules.card_game.rl.gpu_duel.rules import knockdown
        for benched in (False, True):
            state = self.game()
            if benched:
                move_out(state, 'a', 'zero')
            hero(state, 'a', 'zero')['hp'] = 0
            gpu = pack_python([deepcopy(state)], device='cpu')
            knockdown(gpu)
            self.assertEqual(int(gpu.last_front[0, 0]), -1)


if __name__ == '__main__':
    unittest.main()
