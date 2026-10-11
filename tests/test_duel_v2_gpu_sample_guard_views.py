"""Canonical synthetic chance worlds with both exact guard obligations."""
from copy import deepcopy
import unittest
import numpy as np
from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.rl.batched_duel.gpu_sample_guard_views import seven_sample_guard_views_packet
from app.modules.card_game.rl.batched_duel.gpu_rule_entries import seven_sample_world
from app.modules.card_game.rl.seven_lineup import encode

class GuardViewsTest(unittest.TestCase):
    def test_both_pairs_and_world_match_with_input_immutable(self):
        for skip in (False,True):
            for first in ('a','b'):
                state=new_game(seed=19,first_side=first,skip_mulligan=skip);original=deepcopy(state)
                for viewer in ('a','b'):
                    for seed in (23,(1<<256)+71):
                        world,pairs=seven_sample_guard_views_packet(state,viewer,seed)
                        for left,right in pairs:
                            x,c=encode(left[1],left[0]);xx,cc=encode(right[1],right[0])
                            self.assertTrue(np.array_equal(x,xx));self.assertTrue(np.array_equal(c,cc))
                        self.assertEqual(world,seven_sample_world(state,viewer,seed));self.assertEqual(state,original)
                        if not skip:
                            self.assertEqual(pairs[0][0][1]['phase'],'playing')
                            self.assertEqual(pairs[1][0][1]['phase'],'mulligan')
    def test_unsupported_phase_fails(self):
        state=new_game(seed=1);state['phase']='finished'
        with self.assertRaises(ValueError):seven_sample_guard_views_packet(state,'a',2)

if __name__=='__main__':unittest.main()
