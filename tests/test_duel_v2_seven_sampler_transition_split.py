"""Private generation is unqualified; public sampler keeps its exact guard."""
from copy import deepcopy
from unittest.mock import patch
import unittest
import numpy as np
from app.modules.card_game.engine.duel_v2 import new_game,observe
from app.modules.card_game.rl import seven_sampler as sampler

class TransitionSplitTest(unittest.TestCase):
    def test_world_before_and_input_are_identical(self):
        for first in ('a','b'):
            state=new_game(seed=19,first_side=first,skip_mulligan=True);original=deepcopy(state)
            for viewer in ('a','b'):
                for seed in (23,(1<<256)+71):
                    world,before=sampler._determinize_transition(state,viewer,seed)
                    self.assertEqual(world,sampler.determinize(state,viewer,seed))
                    self.assertEqual(before,observe(state,viewer,include_previews=False));self.assertEqual(state,original)
    def test_public_guard_is_still_mandatory(self):
        state=new_game(seed=1,skip_mulligan=True);real=sampler.encode;calls=[]
        def corrupted(view,actions):
            x,c=real(view,actions);calls.append(1)
            if len(calls)==2:x=x.copy();x[0]+=np.float32(1)
            return x,c
        with patch.object(sampler,'encode',side_effect=corrupted):
            with self.assertRaisesRegex(ValueError,'Particle changed learner information'):sampler.determinize(state,'a',31)
        for phase in ('mulligan','finished'):
            state['phase']=phase
            for function in (sampler.determinize,sampler._determinize_transition):
                with self.assertRaises(ValueError):function(state,'a',31)

if __name__=='__main__':unittest.main()
