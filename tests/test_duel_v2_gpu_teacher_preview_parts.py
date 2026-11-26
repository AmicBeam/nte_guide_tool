"""Canonical composition equality and complete combat-preview coverage."""
from copy import deepcopy
import unittest
from app.modules.card_game.engine.duel_v2 import new_game,acting_side
from app.modules.card_game.engine.duel_v2.simulation import simulate_action
from app.modules.card_game.rl.covered_rollout import rule_decision
from app.modules.card_game.rl.league_rollout import clean
from app.modules.card_game.rl.batched_duel.gpu_teacher_preview_parts import teacher_prepare,teacher_preview,teacher_finish,compose_teacher_rule_decision


class TeacherPartsTest(unittest.TestCase):
    def test_full_canonical_choice_order_input_immutability_and_combat(self):
        combat_cases=0
        for seed in (17,29):
            state=new_game(seed=seed,first_side='a')
            for _ in range(32):
                if state['phase']=='finished':break
                actor=acting_side(state);before=deepcopy(state)
                expected=rule_decision(state,actor)
                view,legal,indices=teacher_prepare(state,actor)
                previews=[teacher_preview(state,actor,view['legal_actions'][i]['action']) for i in indices]
                actual=teacher_finish(view,legal,indices,previews)
                self.assertEqual(actual,expected)
                self.assertEqual(state,before)
                combat_cases+=bool(indices)
                state=clean(simulate_action(state,actor,expected[0][expected[1]]))
        self.assertGreater(combat_cases,0)

    def test_missing_reordered_and_foreign_preview_jobs_fail_closed(self):
        state=new_game(seed=37,skip_mulligan=True)
        for _ in range(24):
            actor=acting_side(state);view,legal,indices=teacher_prepare(state,actor)
            if indices:break
            actions,choice=rule_decision(state,actor)
            state=clean(simulate_action(state,actor,actions[choice]))
        self.assertTrue(indices)
        previews=[teacher_preview(state,actor,view['legal_actions'][i]['action']) for i in indices]
        for bad_i,bad_p in [(indices[:-1],previews[:-1]),(indices,previews[:-1]),([len(view['legal_actions'])],previews[:1])]:
            with self.assertRaises(ValueError):teacher_finish(deepcopy(view),deepcopy(legal),bad_i,bad_p)
        with self.assertRaises(ValueError):teacher_finish(deepcopy(view),legal[:-1],indices,previews)


if __name__=='__main__':unittest.main()
