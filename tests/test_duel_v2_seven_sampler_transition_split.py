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

class GenerationSupportTest(unittest.TestCase):
    def game(self,key='murk'):
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        deck=next(d for d in STARTER_DECKS if d['id']==key)
        return new_game(seed=7,first_side='b',skip_mulligan=True,decks={'a':deck,'b':deck})

    def test_white_heat_recovered_parent_does_not_erase_derived_card(self):
        from app.modules.card_game.engine.duel_v2.state import card_instance
        from app.modules.card_game.engine.duel_v2.escalation import begin_escalation
        state=self.game();team=state['sides']['b']
        for zone in('hand','deck'):
            card=next((c for c in team[zone]if c['card_id']=='A04'),None)
            if card is not None:team[zone].remove(card);break
        self.assertIsNotNone(card);team['discard']=[card,card_instance(state,'AF01','b')]
        state.update(turn=21,active_side='b');begin_escalation(state,'b')
        self.assertFalse(any(c['card_id']=='A04'for c in team['discard']))
        state.update(turn=22,active_side='a');original=deepcopy(state)
        for seed in range(8):
            out=sampler.determinize(state,'a',seed)
            self.assertEqual(len(out['sides']['b']['hand']),len(team['hand']))
            self.assertEqual(len(out['sides']['b']['deck']),len(team['deck']))
        self.assertEqual(state,original)

    def test_copies_generated_into_deck_need_not_fit_in_hidden_hand(self):
        from app.modules.card_game.engine.duel_v2.state import card_instance
        state=self.game('quick-rush');team=state['sides']['b'];team['discard']+=team['hand'];team['hand']=[]
        for _ in range(4):team['deck'].append(card_instance(state,'Q05','b'))
        for seed in range(8):
            out=sampler.determinize(state,'a',seed)
            self.assertEqual(out['sides']['b']['hand'],[])
            self.assertEqual(len(out['sides']['b']['deck']),len(team['deck']))

    def test_successful_prior_draws_and_rng_are_bit_exact(self):
        from random import Random
        team=self.game()['sides']['b']
        for seed in range(32):
            a,b=Random(seed),Random(seed);info={}
            old=sampler._allocation_prior(team,[],0,a,len(team['hand']),0,True)
            new=sampler._allocation(team,[],0,b,len(team['hand']),0,True,recovery_budget=5,continuation=info)
            self.assertEqual(old,new);self.assertEqual(a.getstate(),b.getstate());self.assertFalse(info)
