import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from app.modules.card_game.rl import covered_rollout as covered,recovery_runtime as rt
from app.modules.card_game.rl.recovery_search import search
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe, acting_side
from app.modules.card_game.engine.duel_v2 import projection
from app.modules.card_game.engine.duel_v2.simulation import collecting_patches
from app.modules.card_game.rl.league_rollout import clean
from scripts.train_duel_v2_current_round import preset_decks


class Uniform:
    def scores_only(self,x,c):return np.zeros(len(c),dtype=np.float32)
    def scores_value(self,x,c):return self.scores_only(x,c),0.


class CoveredRolloutTest(unittest.TestCase):
    def state(self):
        decks=preset_decks()
        return new_game(seed=3,first_side='a',skip_mulligan=True,
                        decks={'a':decks['zhenhong'],'b':decks['murk']})

    def test_opponent_gets_own_observation_not_full_hidden_state(self):
        public={'viewer_side':'b','legal_actions':[]};seen=[]
        def rule(view,legal):seen.append(view);return 1
        with patch.object(covered,'observe',return_value=public) as observe,\
             patch.object(covered,'VisibleEngineRulePolicy',return_value=rule):
            result=covered.opponent_action({'secret':'must not be passed'},'b',
                [{'type':'end_turn'},{'type':'attack'}],{},rt,np.random.default_rng(2))
        self.assertEqual(result,1);self.assertEqual(seen,[public])
        self.assertEqual(observe.call_args.args[1],'b')

    def test_selective_previews_match_full_preview_policy_without_patches(self):
        decks=preset_decks();keys=list(decks);checked=0
        original_preview=projection.preview
        for i,key in enumerate(keys):
            state=new_game(seed=31+i,first_side='a',skip_mulligan=True,
                           decks={'a':decks[key],'b':decks[keys[(i+1)%len(keys)]]})
            for _ in range(24):
                if state['phase']=='finished':break
                actor=acting_side(state);before=deepcopy(state)
                reference=observe(state,actor,include_previews=True)
                ref_legal=[e['action'] for e in reference['legal_actions'] if e['action']['type']!='concede']
                def preview(source,side,action):
                    nonlocal checked
                    self.assertFalse(collecting_patches());checked+=1
                    result=original_preview(source,side,action)
                    expected=next(e['preview'] for e in reference['legal_actions'] if e['action']==action)
                    self.assertEqual(result,expected)
                    return result
                with patch.object(projection,'preview',side_effect=preview):
                    legal,index=covered.rule_decision(state,actor)
                self.assertTrue(collecting_patches());self.assertEqual(state,before)
                self.assertEqual(legal,ref_legal)
                self.assertEqual(index,covered.VisibleEngineRulePolicy()(reference,tuple(ref_legal)))
                state=clean(apply_action(state,actor,legal[index]))
        self.assertGreater(checked,100)

    def test_preview_exception_restores_presentation_context(self):
        self.assertTrue(collecting_patches())
        with patch.object(projection,'preview',side_effect=RuntimeError('preview failure')):
            with self.assertRaisesRegex(RuntimeError,'preview failure'):
                covered.rule_decision(self.state(),'a')
        self.assertTrue(collecting_patches())

    def test_real_terminal_rollout_is_repeatable_and_preserves_root(self):
        state=self.state();original=deepcopy(state);policies={'a':Uniform(),'b':object()}
        first=covered.terminal_leaf(state,'a',policies,rt,np.random.default_rng(9),float('inf'))
        second=covered.terminal_leaf(state,'a',policies,rt,np.random.default_rng(9),float('inf'))
        self.assertEqual(first,second);self.assertTrue(first[1]);self.assertIn(first[0],(-1.,0.,1.))
        self.assertEqual(state,original)
        self.assertEqual(covered.terminal_leaf(state,'a',policies,rt,np.random.default_rng(9),0),(None,False))
        self.assertEqual(covered.terminal_leaf(state,'a',policies,rt,np.random.default_rng(9),float('inf'),max_actions=0),(None,False))

    def test_covered_search_uses_terminal_backup_and_rejects_partial_rollouts(self):
        state=self.state();policies={'a':Uniform(),'b':object()}
        with patch.object(covered,'terminal_leaf',return_value=(1.,True)) as leaf,\
             patch('app.modules.card_game.rl.recovery_search.core.short_terminal_value') as proof:
            result=search(state,'a',policies,runtime=rt,seed=8,simulations=2,max_depth=1,
                          teacher_mode='covered_terminal')
        self.assertTrue(result['complete']);self.assertGreater(leaf.call_count,0)
        self.assertFalse(proof.called)
        self.assertEqual(result['leaf_source'],'real_terminal')
        self.assertTrue(result['policy_source'].endswith(':covered_terminal'))
        self.assertTrue(np.all(result['mean_values'][result['visits']>0]==1.))
        with patch.object(covered,'terminal_leaf',return_value=(None,False)):
            partial=search(state,'a',policies,runtime=rt,seed=8,simulations=2,max_depth=1,
                           teacher_mode='covered_terminal')
        self.assertFalse(partial['complete']);self.assertEqual(partial['simulations'],0)
        with self.assertRaises(ValueError):
            search(state,'a',policies,runtime=rt,seed=8,teacher_mode='covered_terminal',q_scale='legacy')


if __name__=='__main__':unittest.main()
