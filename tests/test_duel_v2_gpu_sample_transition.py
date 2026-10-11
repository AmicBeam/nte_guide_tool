"""CPU composition oracle for the internal packet; never sampler qualification."""
from copy import deepcopy
import unittest
import numpy as np
from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.rl.batched_duel.gpu_sample_transition import seven_sample_transition_packet
from app.modules.card_game.rl.batched_duel.gpu_rule_entries import seven_sample_world
from app.modules.card_game.rl.seven_runtime import decision


class SampleTransitionPacketTest(unittest.TestCase):
    def test_exact_canonical_world_after_complete_encoded_check(self):
        for first in ('a','b'):
            state=new_game(seed=19,first_side=first)
            original=deepcopy(state)
            for viewer in ('a','b'):
                for seed in (23,(1<<256)+71):
                    world,required=seven_sample_transition_packet(state,viewer,seed)
                    self.assertEqual(required,1)
                    _,(x,c)=decision(state,viewer)
                    _,(xx,cc)=decision(world,viewer)
                    self.assertTrue(np.array_equal(x,xx))
                    self.assertTrue(np.array_equal(c,cc))
                    self.assertEqual(world,seven_sample_world(state,viewer,seed))
                    self.assertEqual(state,original)

    def test_playing_and_invalid_phase_contract(self):
        state=new_game(seed=31,skip_mulligan=True)
        world,required=seven_sample_transition_packet(state,'a',59)
        self.assertEqual(required,0)
        self.assertEqual(world,seven_sample_world(state,'a',59))
        state['phase']='finished'
        with self.assertRaises(ValueError):seven_sample_transition_packet(state,'a',59)


if __name__=='__main__':
    unittest.main()
