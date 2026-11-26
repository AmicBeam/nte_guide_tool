"""Explicit CPU oracle tests for the GPU training adapter; no CUDA readiness claim."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import io
import json
import runpy
import tempfile
import time
import unittest
import numpy as np

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.rl.batched_duel.process_backend import ResidentRuleBackend
from app.modules.card_game.rl.batched_search.inference import InferenceResult
from app.modules.card_game.rl.batched_search.training_rollout import (
    collect_training_games, source_alias, models_for_jobs, validate_training_jobs,
    bind_side_policies, rollout_from_config, verify_training_replay,
)
from app.modules.card_game.rl.batched_search.traversal import PolicyIdentity
from app.modules.card_game.rl import seven_runtime
from app.modules.card_game.rl.seven_lineup import ACTION_TYPES


class FrozenInference:
    def __init__(self):
        self.calls=[]
    def predict(self, requests):
        self.calls.append(requests)
        result=[]
        for req in requests:
            logits=np.full(len(req.candidates),-40,dtype=np.float32)
            if len(req.candidates):
                logits[np.asarray(req.candidates)[:,0]==ACTION_TYPES.index('attack')]=20
            result.append(InferenceResult(req.request_id,req.model_key,req.model_version,
                                         logits,np.asarray([.3,.4,.3],dtype=np.float32)))
        return result


def jobs(count=2):
    deck=deepcopy(next(d for d in STARTER_DECKS if d['id']=='starter'))
    return [dict(id='fixture-'+str(i),seed=899001+i,first='a' if i%2==0 else 'b',
                 pure=False,training=True,train_sides=['a','b'],simulations=32,
                 gumbel_candidates=16,terminal_horizon=0,max_depth=1,max_actions=2,
                 q_scale='natural_wdl',teacher_mode='network',deadline=time.time()+180,
                 decks={'a':deck,'b':deck},policies={s:PolicyIdentity('starter','fixture-v1') for s in ('a','b')})
            for i in range(count)]


def fixtures(plan):
    states=[]
    for job in plan:
        state=new_game(seed=job['seed'],decks=job['decks'],first_side=job['first'],skip_mulligan=True)
        for side in ('a','b'):
            state['sides'][side]['hp']=1; state['sides'][side]['shield']=0
        enemy='b' if job['first']=='a' else 'a'
        for character in state['sides'][enemy]['characters'].values():
            character['hp']=0; character['down_turns']=2
        states.append(state)
    return states


class CooperativeTrainingTest(unittest.TestCase):
    def collect(self, plan, states=fixtures):
        inference=FrozenInference()
        with ResidentRuleBackend(workers=1,max_games=len(plan),runtime=seven_runtime) as backend:
            result=collect_training_games(plan,inference=inference,backend=backend,
                models={'starter':SimpleNamespace(version='fixture-v1')},search_roots=len(plan),
                create_states=states,allow_cpu_oracle=True,runtime_name='seven')
            self.assertEqual(backend.stats()['live_states'],0)
        return result,inference

    def test_normal_terminal_only_full_pi_and_observer_wdl(self):
        plan=jobs(4);games,inference=self.collect(plan)
        self.assertTrue(all(game['complete'] and game['learning_eligible'] for game in games), [(g['reason'],g['roots']) for g in games])
        self.assertTrue(all(game['searched']==game['decisions']==len(game['rows'])==1 for game in games))
        for game,job in zip(games,plan):
            self.assertEqual(game['winner'],job['first'])
            self.assertEqual(len(game['value_rows']),2)
            for row in game['rows']+game['value_rows']:
                self.assertEqual(row['z'],1 if row['side']==job['first'] else -1)
                self.assertEqual(row['value_actor'],int(row['side']==job['first']))
            row=game['rows'][0]
            self.assertAlmostEqual(float(row['pi'].sum()),1)
            self.assertEqual(row['search_budget'],32)
            self.assertEqual(game['model_versions'],{'a':'fixture-v1','b':'fixture-v1'})
            self.assertTrue(verify_training_replay(job,game,initial_state=fixtures([job])[0])['passed'])
        self.assertGreater(max(len(batch) for batch in inference.calls),1)

    def test_deterministic_seeds_replay_actions_and_pi(self):
        plan=jobs(2);a,_=self.collect(plan);b,_=self.collect(plan)
        for left,right in zip(a,b):
            self.assertEqual(left['episode_actions'],right['episode_actions'])
            np.testing.assert_array_equal(left['rows'][0]['pi'],right['rows'][0]['pi'])
            np.testing.assert_array_equal(left['rows'][0]['visit_pi'],right['rows'][0]['visit_pi'])

    def test_action_limit_and_expired_deadline_cannot_create_labels(self):
        def durable(plan):
            states=fixtures(plan)
            for state in states:
                for side in ('a','b'):state['sides'][side]['hp']=30
            return states
        plan=jobs(1);plan[0]['max_actions']=1
        games,_=self.collect(plan,durable)
        self.assertFalse(games[0]['complete']);self.assertFalse(games[0]['learning_eligible'])
        self.assertEqual(games[0]['rows'],[]);self.assertEqual(games[0]['value_rows'],[])
        plan[0]['deadline']=time.time()-1
        games,_=self.collect(plan)
        self.assertFalse(games[0]['complete']);self.assertEqual(games[0]['rows'],[])
        self.assertEqual(games[0]['reason'],'deadline')

    def test_exact_historical_folder_is_bound_and_unknown_modes_rejected(self):
        job=jobs(1)[0];job['policies']={'a':('/current','starter'),'b':('/history','starter')}
        current=SimpleNamespace(version='current');history=SimpleNamespace(version='historical')
        models={source_alias(job['policies']['a']):current,source_alias(job['policies']['b']):history}
        identities=bind_side_policies(job,models)
        self.assertEqual(identities['b'].model_version,'historical')
        self.assertNotEqual(identities['a'].policy_key,identities['b'].policy_key)
        with patch('app.modules.card_game.rl.seven_runtime.model',side_effect=[current,history]) as load:
            self.assertEqual(models_for_jobs([job]),models)
            self.assertEqual(load.call_count,2)
        job['gumbel_candidates']=1.5
        with self.assertRaises(ValueError):validate_training_jobs([job])
        job['gumbel_candidates']=16
        job['teacher_mode']='invented'
        with self.assertRaises(ValueError):validate_training_jobs([job])

    def test_owned_constructor_failure_closes_backend(self):
        backend=SimpleNamespace(close=lambda:closed.append(True))
        closed=[]
        with patch('app.modules.card_game.rl.batched_search.training_rollout.bind_training_backend',return_value=backend), \
             patch('app.modules.card_game.rl.batched_search.training_rollout.bind_training_inference',side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):rollout_from_config({}, {})
        self.assertEqual(closed,[True])

    def test_job_flags_cannot_bypass_actual_model_value_precheck(self):
        plan=jobs(1);plan[0]['synthetic_training']=True
        with ResidentRuleBackend(workers=1,max_games=1,runtime=seven_runtime) as backend:
            with self.assertRaisesRegex(ValueError,'WDL precheck'):
                collect_training_games(plan,inference=FrozenInference(),backend=backend,
                    models={'starter':SimpleNamespace(version='fixture-v1')},allow_cpu_oracle=True)
            self.assertEqual(backend.stats()['live_states'],0)

    def test_buffer_overflow_releases_states_and_admits_no_partial_labels(self):
        plan=jobs(2);inference=FrozenInference()
        with ResidentRuleBackend(workers=1,max_games=2,runtime=seven_runtime) as backend:
            with self.assertRaisesRegex(MemoryError,'buffer budget'):
                collect_training_games(plan,inference=inference,backend=backend,
                    models={'starter':SimpleNamespace(version='fixture-v1')},allow_cpu_oracle=True,
                    create_states=fixtures,max_buffer_bytes=1)
            self.assertEqual(backend.stats()['live_states'],0)
            self.assertEqual(inference.calls,[])

    def test_collected_rows_joint_update_and_exact_round_restore(self):
        import torch
        from app.modules.card_game.rl import seven_round as rr
        from app.modules.card_game.rl.seven_schedule import ROSTER
        from app.modules.card_game.rl.next_training import plan as declared_plan
        from app.modules.card_game.rl.episode_replay import EpisodePool
        from app.modules.card_game.rl.batched_search.training_rollout import TrainingRollout
        torch.manual_seed(13);torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'initial';updated=root/'updated'
            decks=declared_plan()['builds']
            nets={key:seven_runtime.create_network(key,16) for key in ROSTER}
            opts={key:torch.optim.Adam(net.parameters(),lr=.0003) for key,net in nets.items()}
            before={key:{n:t.detach().clone() for n,t in net.state_dict().items()} for key,net in nets.items()}
            for key in ROSTER:seven_runtime.export(nets[key],source,key,decks[key],{'initialization':'synthetic_contract_test'})
            config=dict(learning_keys=['starter'],policy_passes=1,max_row_reuse=4,
                        train_until=time.time()+120,hard_deadline=time.time()+180,planned_batches=2)
            rng=np.random.default_rng(912)
            schedule=jobs(2)
            for job in schedule:
                job['policies']={side:(str(source),'starter') for side in ('a','b')}
            models=models_for_jobs(schedule)
            with ResidentRuleBackend(workers=1,max_games=2,runtime=seven_runtime) as backend:
                collector=TrainingRollout(FrozenInference(),backend,models,search_roots=2,
                    create_states=fixtures,allow_cpu_oracle=True)
                games=rr.collect(None,schedule,root/'sampling',learning=True,rollout=collector)
            pools={key:EpisodePool(limit=64,max_row_reuse=4) for key in ROSTER}
            health,_=rr.update_batch(nets,opts,pools,schedule,games,rng,config,config['train_until'],root/'updates')
            self.assertTrue(health['starter']['finite']);self.assertTrue(health['starter']['gradient'])
            self.assertEqual(health['starter']['coverage'],1)
            self.assertEqual(health['starter']['value_rows'],2)
            self.assertTrue(any(not torch.equal(t,before['starter'][n]) for n,t in nets['starter'].state_dict().items()))
            for key in ROSTER:
                if key!='starter':self.assertTrue(all(torch.equal(t,before[key][n]) for n,t in nets[key].state_dict().items()))
                seven_runtime.export(nets[key],updated,key,decks[key],{'initialization':'synthetic_contract_test'})
            rr.save_round(root,nets,opts,pools,updated,rng,config,1,source,False)
            restored,restored_opts,restored_pools,restored_rng,pointer,_=rr.restore_round(root,'cpu',config)
            self.assertEqual(pointer['cycle'],1);self.assertEqual(rng.bit_generator.state,restored_rng.bit_generator.state)
            self.assertEqual(pools['starter'].export_state()['row_uses'],restored_pools['starter'].export_state()['row_uses'])
            rows=games[0]['rows']
            seven_runtime.update(nets['starter'],opts['starter'],rows)
            seven_runtime.update(restored['starter'],restored_opts['starter'],rows)
            self.assertTrue(all(torch.equal(t,restored['starter'].state_dict()[n]) for n,t in nets['starter'].state_dict().items()))
            with self.assertRaises(ValueError):rr.restore_round(root,'cpu',dict(config,train_until=config['train_until']+1))

    def test_formal_launch_does_not_accept_smoke_as_readiness(self):
        from app.modules.card_game.rl.seven_round import validate_launch
        from app.modules.card_game.rl.seven_lineup import identity
        with self.assertRaisesRegex(ValueError,'preflight receipt'):
            validate_launch(dict(runtime='seven',rule_hash=identity(),teacher_mode='network'))

    def test_default_cli_is_seven_and_dry_creates_no_output(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'no-output'
            with patch('sys.argv',['train_duel_v2_current_round.py','--output',str(out)]),patch('sys.stdout',new_callable=io.StringIO) as stream:
                runpy.run_path('scripts/train_duel_v2_current_round.py',run_name='__main__')
            config=json.loads(stream.getvalue())
            self.assertEqual(config['runtime'],'seven');self.assertEqual(len(config['roster']),7)
            self.assertEqual(config['rule_backend'],'serial_cpu');self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
