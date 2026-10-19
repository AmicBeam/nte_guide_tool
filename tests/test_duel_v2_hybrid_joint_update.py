"""Controlled CPU optimizer integration; synthetic labels grant no train readiness."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from app.modules.card_game.rl import hybrid_joint_update as stage
from app.modules.card_game.rl import seven_runtime as rt
from app.modules.card_game.rl.seven_schedule import ROSTER
from app.modules.card_game.rl.next_training import plan
from app.modules.card_game.engine.duel_v2 import new_game

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None,'Torch unavailable')
class HybridJointUpdateTest(unittest.TestCase):
    def test_forced_root_uses_one_simulation_and_multiaction_roots_require_32(self):
        job = dict(policies={'a':('/frozen','starter')})
        row = dict(step=0,side='a',key='starter',c=np.zeros((1,2),dtype=np.float32),
                   search_budget=1,search_algorithm='gumbel',teacher_mode='covered_value',
                   q_scale='natural_wdl',z=1)
        stage._validate_policy_row(row,0,dict(side='a'),job)
        row['c'] = np.zeros((2,2),dtype=np.float32)
        with self.assertRaisesRegex(ValueError,'Gumbel label'):
            stage._validate_policy_row(row,0,dict(side='a'),job)
        row['search_budget'] = 32
        stage._validate_policy_row(row,0,dict(side='a'),job)

    def fixture(self, root):
        torch.set_num_threads(1)
        torch.manual_seed(3701)
        source = root/'source'
        builds = plan()['builds']
        jobs, games = [], []
        for index,key in enumerate(ROSTER):
            net = rt.create_network(key,16)
            rt.export(net,source,key,builds[key],{'initialization':'synthetic_optimizer_contract_test'})
            state = new_game(seed=7101+index,decks={s:builds[key] for s in ('a','b')},
                             first_side='a',skip_mulligan=True)
            actions,(x,c) = rt.decision(state,'a')
            pi = np.zeros(len(c),dtype=np.float32); pi[0] = 1
            row = dict(x=x,c=c,pi=pi,side='a',key=key,step=0,z=1,value_actor=1)
            values = [dict(x=x.copy(),side=s,key=key,step=0,z=1 if s=='a' else -1,
                           value_actor=int(s=='a')) for s in ('a','b')]
            job = dict(id='controlled-'+key,policies={s:(str(source),key) for s in ('a','b')},
                       train_sides=['a','b'])
            game = dict(complete=True,decisions=1,episode_actions=[dict(side='a',action=actions[0])],
                        rows=[row],value_rows=values,winner='a')
            jobs.append(job); games.append(game)
        config = dict(device='cpu',learning_keys=list(ROSTER),policy_passes=1,
            max_row_reuse=4,seed=7101,train_until=time.time()+120,hard_deadline=time.time()+180)
        return source,jobs,games,config

    def test_all_seven_update_restore_and_complete_is_once_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source,jobs,games,config = self.fixture(root)
            output = root/'stage'
            # Directly test the internal transaction using controlled rows.
            # The public admission gate correctly rejects these unqualified labels.
            with self.assertRaises(ValueError):
                stage.run_joint_update(source,jobs,games,root/'unqualified',config,
                                       deadline=time.time()+60)
            self.assertFalse((root/'unqualified').exists())
            receipt = stage._run_once(source,jobs,games,output,config,
                deadline=time.time()+60,contract='controlled-non-readiness-test')
            self.assertEqual(receipt['status'],'COMPLETE')
            self.assertFalse(receipt['quality_approved'])
            self.assertFalse(receipt['automatic_serving_approval'])
            for key in ROSTER:
                health = receipt['health'][key]
                self.assertTrue(health['finite'] and health['gradient'])
                self.assertEqual(health['coverage'],1)
                self.assertGreater(health['policy_rows'],0)
                self.assertNotEqual(receipt['before'][key],receipt['after'][key])
            nets,opts,pools,rng,restored = stage.load_joint_state(output,device='cpu')
            self.assertEqual(restored['files'],receipt['files'])
            self.assertTrue(all(opts[k].state_dict()['state'] for k in ROSTER))
            self.assertTrue(all(pools[k].coverage_report()['first_use_coverage']==1 for k in ROSTER))
            again = stage._run_once(source,jobs,games,output,config,
                deadline=time.time()-1,contract='controlled-non-readiness-test')
            self.assertEqual(receipt['files'],again['files'])
            self.assertEqual(receipt['after'],again['after'])
            # Safe restored Adam, pools and RNG yield the same next numeric step.
            other_nets,other_opts,other_pools,other_rng,_ = stage.load_joint_state(output,device='cpu')
            self.assertEqual(rng.bit_generator.state,other_rng.bit_generator.state)
            for key,game in zip(ROSTER,games):
                rt.update(nets[key],opts[key],game['rows'])
                rt.update(other_nets[key],other_opts[key],game['rows'])
                self.assertTrue(all(torch.equal(t,other_nets[key].state_dict()[n])
                                    for n,t in nets[key].state_dict().items()))
            with self.assertRaisesRegex(ValueError,'input contract'):
                stage._run_once(source,jobs,games,output,config,deadline=time.time()+60,contract='changed')
            (output/'models/starter.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'integrity'):
                stage.load_joint_state(output,device='cpu')

    def test_running_or_failed_marker_refuses_optimizer_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source,jobs,games,config = self.fixture(root)
            for status in ('RUNNING','FAILED'):
                output = root/status; output.mkdir()
                (output/'joint-status.json').write_text(json.dumps(dict(schema=stage.SCHEMA,status=status)))
                with self.assertRaisesRegex(ValueError,'cannot be replayed'):
                    stage._run_once(source,jobs,games,output,config,
                        deadline=time.time()+60,contract='controlled')
                self.assertEqual(list(output.iterdir()),[output/'joint-status.json'])

    def test_failed_gradient_gate_records_failed_and_refuses_rerun(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source,jobs,games,config = self.fixture(root)
            # A single candidate has a constant policy objective and zero policy
            # gradient. The real optimizer path must fail its learning gate.
            for game in games:
                game['rows'][0]['c'] = game['rows'][0]['c'][:1]
                game['rows'][0]['pi'] = np.ones(1,dtype=np.float32)
            output = root/'failed-gradient'
            with self.assertRaisesRegex(ValueError,'positive-gradient'):
                stage._run_once(source,jobs,games,output,config,
                    deadline=time.time()+60,contract='controlled-zero-gradient')

            self.assertEqual(json.loads((output/'joint-status.json').read_text())['status'],'FAILED')
            with self.assertRaisesRegex(ValueError,'cannot be replayed'):
                stage._run_once(source,jobs,games,output,config,
                    deadline=time.time()+60,contract='controlled-zero-gradient')

    def test_qualified_value_rebind_retains_policy_adam_and_continues_next_batch(self):
        from app.modules.card_game.rl.seven_lineup import identity
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source,jobs,games,config = self.fixture(root)
            first = root/'first'
            stage._run_once(source,jobs,games,first,config,deadline=time.time()+60,contract='first-fixture')
            nets,opts,pools,rng,previous = stage.load_joint_state(first,device='cpu')
            calibrated = root/'value-only'
            for key in ROSTER:
                policy = rt.model(previous['models'],key)
                with torch.no_grad():
                    nets[key].wdl.bias[0].add_(.01)
                rt.export(nets[key],calibrated,key,policy.serving_deck,policy.manifest['origin'])
            # Guard-schema fixture only: synthetic optimizer tests do not call
            # public collection admission or claim measured calibration/readiness.
            report = dict(schema='seven_value_precheck_v1',complete=True,passed=True,heldout=True,
                rule_hash=identity(),models={k:rt.model(calibrated,k).version for k in ROSTER},
                metrics={k:dict(normal_games=8,brier=.5,log_loss=.8) for k in ROSTER})
            with self.assertRaisesRegex(ValueError,'Qualified main value report'):
                stage._rebind_value_source(nets,opts,previous,calibrated,None)
            policy_states = {k:{n:deepcopy(opts[k].state[p]) for n,p in nets[k].named_parameters()
                                if not n.startswith(('wdl.','value_net.')) and p in opts[k].state}
                             for k in ROSTER}
            controls = {k:deepcopy(net) for k,net in nets.items()}
            control_opts = {k:torch.optim.Adam(net.parameters()) for k,net in controls.items()}
            for key in ROSTER:
                control_opts[key].load_state_dict(deepcopy(opts[key].state_dict()))
            rng_before = deepcopy(rng.bit_generator.state)
            torch_before = torch.get_rng_state().clone()
            transition = stage._rebind_value_source(nets,opts,previous,calibrated,report)
            self.assertTrue(transition['selective_value_adam_reset'])
            self.assertEqual(rng_before,rng.bit_generator.state)
            self.assertTrue(torch.equal(torch_before,torch.get_rng_state()))
            for key in ROSTER:
                for name,p in nets[key].named_parameters():
                    if name.startswith(('wdl.','value_net.')):
                        self.assertNotIn(p,opts[key].state)
                    elif name in policy_states[key]:
                        for field,value in policy_states[key][name].items():
                            current = opts[key].state[p][field]
                            self.assertTrue(torch.equal(value,current) if torch.is_tensor(value) else value==current)
            # The selective reset must leave the policy's next actual Adam step
            # identical to the uninterrupted optimizer (only value moments differ).
            for key,game in zip(ROSTER,games):
                rt.update(nets[key],opts[key],game['rows'])
                rt.update(controls[key],control_opts[key],game['rows'])
                for name,p in nets[key].named_parameters():
                    if not name.startswith(('wdl.','value_net.')):
                        control = dict(controls[key].named_parameters())[name]
                        self.assertTrue(torch.equal(p,control))
                        for field,value in opts[key].state[p].items():
                            other = control_opts[key].state[control][field]
                            self.assertTrue(torch.equal(value,other) if torch.is_tensor(value) else value==other)
            next_jobs = deepcopy(jobs)
            for job in next_jobs:
                job['id'] += '-next'
                job['policies'] = {s:(str(calibrated),job['policies'][s][1]) for s in ('a','b')}
            second = stage._run_once(calibrated,next_jobs,games,root/'second',config,
                deadline=time.time()+60,contract='second-fixture',prior_stage=first,main_value_report=report)
            self.assertTrue(second['source_transition']['value_recalibrated'])
            _,_,next_pools,_,_ = stage.load_joint_state(root/'second',device='cpu')
            self.assertTrue(all(len(pool.episodes)==2 for pool in next_pools.values()))
            drift = root/'policy-drift'
            for key in ROSTER:
                net,build,origin = rt.restore(calibrated,key)
                if key == ROSTER[0]:
                    with torch.no_grad():
                        net.score[0].bias.add_(.1)
                rt.export(net,drift,key,build,origin)
            with self.assertRaisesRegex(ValueError,'changed policy'):
                stage._rebind_value_source(nets,opts,previous,drift,report)


if __name__ == '__main__':
    unittest.main()
