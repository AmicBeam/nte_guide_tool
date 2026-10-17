"""Frozen capture adapter integrity only; synthetic rows never qualify CUDA."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np

from app.modules.card_game.rl.batched_search.soa_frozen_collection import (
    training_tape_capacity,completed_rows_after_batch,write_verified_episodes)


class FrozenCollectionTests(unittest.TestCase):
    def test_joint_existing_loss_computes_gradients_without_optimizer_or_weight_changes(self):
        import torch
        from app.modules.card_game.rl.seven_policy import network
        from app.modules.card_game.rl.seven_lineup import encode
        from app.modules.card_game.rl.outcome_runtime import update
        from tests.test_duel_v2_device_training_tape import _cases
        torch.set_num_threads(1);torch.manual_seed(173)
        view,legal=_cases()[:1][0];x,c=encode(view,legal)
        rows=[dict(x=x,c=c,pi=np.full(len(c),1/len(c),np.float32),z=1,value_actor=1),
              dict(x=x,z=-1,value_actor=0)]
        net=network(hidden=16);before={k:v.clone() for k,v in net.state_dict().items()}
        stats=update(net,None,rows,gradient_only=True)
        self.assertEqual(stats['optimizer_steps'],0)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in net.state_dict().items()))
        self.assertTrue(all(stats['policy_gradient_norms'][p]>0 for p in ('state_net','cand_net','score')))
        self.assertGreater(sum(float(p.grad.norm()) for n,p in net.named_parameters()
                               if n.startswith(('value_net','wdl')) and p.grad is not None),0)
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.grad is not None))
        with self.assertRaises(ValueError):update(net,None,rows)
        with self.assertRaises(ValueError):update(net,object(),rows,gradient_only=True)

    def test_budget_uses_full_observer_records_and_real_tape_allocation(self):
        from tests.test_duel_v2_device_training_tape import DeviceTrainingTapeTest,_cases
        from app.modules.card_game.rl.batched_search.device_training_tape import DeviceTrainingTape
        tape,compact,rules,obs,*_=DeviceTrainingTapeTest()._tape(_cases()[:2])
        dimensions=training_tape_capacity(compact,roots=2,decisions=3,
            observation_dim=obs.shape[1],max_bytes=1<<20)
        estimate=dimensions.pop('estimated_bytes')
        actual=DeviceTrainingTape(compact,rules,rows=2,observation_dim=obs.shape[1],device='cpu',
            allow_cpu_reference=True,allow_synthetic=True,**dimensions)
        self.assertEqual(actual.allocated_bytes,estimate)
        self.assertLessEqual(actual.allocated_bytes+compact.allocated_bytes,1<<20)
        self.assertEqual(actual.max_records,6)
        self.assertGreaterEqual(actual.max_legal_entries,compact.max_candidates)
        with self.assertRaises(MemoryError):
            training_tape_capacity(compact,roots=1600,decisions=400,
                observation_dim=obs.shape[1],max_bytes=512<<20)

    def test_exact_training_allocation_includes_turn_dtype_and_padding(self):
        from dataclasses import replace
        from tests.test_duel_v2_device_training_tape import DeviceTrainingTapeTest,_cases,_compact
        from app.modules.card_game.rl.batched_search.device_training_tape import DeviceTrainingTape
        for rows in (1,2):
            _,compact,rules,obs,*_=DeviceTrainingTapeTest()._tape(_cases()[:rows])
            for wide in (False,True):
                layout=_compact(replace(compact.tokens,turn_counts=compact.tokens.turn_counts.double())) if wide else compact
                actual=DeviceTrainingTape(layout,rules,rows=rows,max_records=3,max_legal_entries=67,
                    max_bytes=1<<24,observation_dim=obs.shape[1],device='cpu',allow_cpu_reference=True,allow_synthetic=True)
                self.assertEqual(actual.allocated_bytes,actual._estimate_bytes(layout,rows,4,68,obs.shape[1]))

    def test_requires_replay_complete_cohort_and_both_unique_viewers(self):
        policy=[dict(owner=0,step=0,learning_eligible=True,encode_error=0)]
        values=[dict(owner=0,step=0,actor=i) for i in (0,1)]
        tape=SimpleNamespace(numerical_rows_after_batch=lambda:policy,value_rows_after_batch=lambda:values)
        self.assertTrue(completed_rows_after_batch(tape,verified=True,natural_games=1,roots=1)['complete'])
        with self.assertRaises(ValueError):completed_rows_after_batch(tape,verified=False,natural_games=1,roots=1)
        with self.assertRaises(ValueError):completed_rows_after_batch(tape,verified=True,natural_games=1,roots=2)
        values[1]['actor']=0
        with self.assertRaises(ValueError):completed_rows_after_batch(tape,verified=True,natural_games=1,roots=1)
        policy.clear();values.clear()
        result=completed_rows_after_batch(tape,verified=True,natural_games=0,roots=1)
        self.assertFalse(result['complete']);self.assertEqual(result['optimizer_steps'],0)

    def test_existing_episode_format_and_rejects_changed_policy_or_opponent_view(self):
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        from app.modules.card_game.engine.duel_v2 import new_game,observe,apply_action
        from app.modules.card_game.rl.league_rollout import clean
        from app.modules.card_game.rl.seven_lineup import encode
        from app.modules.card_game.rl.episode_replay import load_episode
        deck=next(d for d in STARTER_DECKS if d['id']=='starter')
        state=new_game(seed=17,first_side='a',decks={'a':deck,'b':deck},skip_mulligan=True)
        state['sides']['b']['hp']=1
        state['sides']['b']['shield']=0
        view=observe(state,'a',include_previews=False)
        legal=[e['action'] for e in view['legal_actions'] if e['action']['type']!='concede']
        chosen=next(i for i,a in enumerate(legal) if a['type']=='attack')
        final=clean(apply_action(state,'a',legal[chosen]));self.assertEqual(final['phase'],'finished')
        x,c=encode(view,legal);other,_=encode(observe(state,'b',include_previews=False),[])
        pi=np.full(len(legal),1/len(legal),np.float32);visits=np.zeros(len(legal),np.int32);visits[chosen]=32
        row=dict(owner=0,step=0,actor=0,x=x,c=c,pi=pi,visit_pi=visits/32,
                 visits=visits,selected=chosen,z=1.,version=state['version'],job_seed=17,requested=32,completed=32)
        values=[dict(owner=0,step=0,actor=i,x=v,value_actor=1-i,z=1. if i==0 else -1.)
                for i,v in enumerate((x,other))]
        dataset=dict(complete=True,policy=[row],values=values)
        arguments=dict(initial=[state],final=[final],records=[dict(game=0,step=0,actor='a',
            raw_action=legal[chosen],chosen=chosen,requested=32,completed=32)],
            jobs=[dict(seed=17,first_side='a',keys={'a':'starter','b':'starter'})],versions={'starter':'frozen'},
            policy_reference={(0,0):dict(pi=pi.copy(),visits=visits.copy())})
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'valid'
            result=write_verified_episodes(dataset,output=out,**arguments)
            episode=load_episode(out/'soa-frozen-0.json.gz')
            self.assertTrue(result['complete']);self.assertFalse(result['formal_training_qualified'])
            self.assertEqual(len(episode['rows']),3)
            np.testing.assert_array_equal(episode['rows'][0]['pi'],pi)
            for field in ('pi','opponent'):
                broken=deepcopy(dataset)
                if field=='pi':broken['policy'][0]['pi'][chosen]+=.01
                else:broken['values'][1]['x'][0]+=.01
                failed=Path(directory)/field
                with self.assertRaises(AssertionError):write_verified_episodes(broken,output=failed,**arguments)
                self.assertFalse(failed.exists())


if __name__=='__main__':unittest.main()
