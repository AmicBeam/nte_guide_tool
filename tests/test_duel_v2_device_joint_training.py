"""Actual CPU learner updates with explicit mocked CUDA ingress, not GPU evidence."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import time
import unittest
from unittest.mock import Mock,patch

from app.modules.card_game.rl.device_joint_training import run_joint_training_smoke
from app.modules.card_game.rl.batched_search.device_h3_collection import DeviceH3Collector
from app.modules.card_game.rl.batched_search.device_pure_collection import DevicePureCollector


class DeviceJointTrainingTest(unittest.TestCase):
    def test_reject_value_only_or_archived_collector_before_output(self):
        with TemporaryDirectory() as folder:
            out=Path(folder)/'out'
            with self.assertRaises(TypeError):
                run_joint_training_smoke(out,collector=NS(),pure_collector=NS(),source=folder,
                    train_jobs=[],evaluation_jobs=[],deadline=time.time()+600)
            self.assertFalse(out.exists())

    def test_incomplete_h3_batch_cannot_call_optimizer(self):
        collector=object.__new__(DeviceH3Collector)
        pure=object.__new__(DevicePureCollector)
        collector.executor=pure.executor=NS(device='cuda:0')
        collector.last_diagnostics={'complete':False,'completed_simulations':[17]}
        pure.last_diagnostics=None
        collector.collect=Mock(side_effect=RuntimeError('incomplete full-budget H3'))
        with TemporaryDirectory() as folder, \
             patch('app.modules.card_game.rl.batched_search.device_h3_collection.validate_h3_inputs'), \
             patch('app.modules.card_game.rl.batched_search.soa_pure_jobs.adapt_pure_jobs'), \
             patch('app.modules.card_game.rl.seven_runtime.model',return_value=NS()), \
             patch('app.modules.card_game.rl.seven_round.update_batch') as update:
            source=Path(folder)/'source';source.mkdir();out=Path(folder)/'out';deadline=time.time()+600
            from app.modules.card_game.rl.seven_schedule import ROSTER
            for key in ROSTER:
                for ext in ('.json','.npz'):(source/(key+ext)).write_bytes(b'fixture-only')
            train=[dict(id='joint',seed=1,deadline=deadline,training=True,pure=False,
                policies={s:(str(source),'starter')for s in ('a','b')})]
            evaluation=[dict(id='eval',seed=2,deadline=deadline,training=False,pure=True,
                policies={s:(str(source),'starter')for s in ('a','b')})]
            with self.assertRaisesRegex(RuntimeError,'incomplete'):
                run_joint_training_smoke(out,collector=collector,pure_collector=pure,source=source,
                    train_jobs=train,evaluation_jobs=tuple(evaluation),deadline=deadline)
            update.assert_not_called()
            import json
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['formal_policy_updates'],0)
            self.assertEqual(report['complete_gumbel_games'],0)

    def test_existing_joint_optimizer_changes_both_policy_and_value_and_freezes_eval(self):
        import numpy as np
        import torch
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.rl import seven_runtime as rt
        from app.modules.card_game.rl.seven_schedule import ROSTER
        torch.set_num_threads(1);torch.manual_seed(187)
        deadline=time.time()+600
        with TemporaryDirectory() as folder:
            source=Path(folder)/'source'
            decks={d['id']:d for d in STARTER_DECKS}
            for key in ROSTER:
                net=rt.create_network(key,16,'cpu');rt.export(net,source,key,decks[key],dict(fixture=True))
            state=new_game(seed=1,decks={'a':decks['starter'],'b':decks['starter']})
            actions,(x,c)=rt.decision(state,'a');n=len(actions)
            self.assertGreater(n,1)
            pi=np.zeros(n,dtype=np.float32);pi[-1]=1
            game=dict(id='joint',complete=True,learning_eligible=True,winner='a',decisions=1,searched=1,
                episode_actions=[dict(side='a',action=actions[-1])],rows=[dict(x=x,c=c,pi=pi,z=1.,side='a',
                    key='starter',step=0,value_actor=1,search_budget=32,search_algorithm='gumbel')],
                value_rows=[dict(x=x,z=-1.,side='b',key='starter',step=0,value_actor=0)])
            collector=object.__new__(DeviceH3Collector);pure=object.__new__(DevicePureCollector)
            collector.executor=pure.executor=NS(device=torch.device('cpu'))
            collector.last_diagnostics={'complete':True,'cpu_fixture':True};pure.last_diagnostics={'complete':True,'cpu_fixture':True}
            collector.collect=Mock(return_value=[game])
            def evaluate(jobs,bank):
                self.assertTrue(all(Path(m.directory).name=='models'for m in bank.values()))
                return [dict(game,id=jobs[0]['id'],rows=[],searched=0)]
            pure.collect=evaluate
            common=dict(first='a',left='starter',right='starter',deadline=deadline,train_sides=['a'],
                policies={s:(str(source),'starter')for s in ('a','b')})
            train=[dict(common,id='joint',seed=1,training=True,pure=False)]
            evaluation=[dict(common,id='eval',seed=2,training=False,pure=True)]
            with patch('app.modules.card_game.rl.batched_search.device_h3_collection.validate_h3_inputs'), \
                 patch('app.modules.card_game.rl.batched_search.soa_pure_jobs.adapt_pure_jobs'):
                result=run_joint_training_smoke(Path(folder)/'out',collector=collector,pure_collector=pure,source=source,
                    train_jobs=train,evaluation_jobs=evaluation,deadline=deadline,policy_passes=4)
            self.assertTrue(result['passed']);self.assertEqual(result['complete_gumbel_games'],1)
            self.assertEqual(result['formal_policy_updates'],4)
            self.assertTrue(result['initial_weights_unchanged']);self.assertTrue(result['frozen_weights_unchanged'])
            changed=result['parameter_changes']['starter']
            self.assertTrue(any(n.startswith('score.')for n in changed))
            self.assertTrue(any(n.startswith('wdl.')for n in changed))
            self.assertFalse(result['formal_training_complete'])

class DeviceJointPreparationTest(unittest.TestCase):
    """Orchestration fixtures only, never calibration or CUDA evidence."""
    def test_explicit_manual_builds_reach_main_initialization(self):
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        from app.modules.card_game.rl.seven_round import initialize
        from app.modules.card_game.rl.seven_schedule import ROSTER
        original={d['id']:d for d in STARTER_DECKS if d['id'] in ROSTER}
        manual=deepcopy(original)
        manual['collapse']['card_ids']=list(reversed(manual['collapse']['card_ids']))
        with patch('app.modules.card_game.rl.next_training.plan',side_effect=AssertionError('implicit plan')), \
             patch('app.modules.card_game.rl.seven_runtime.restore',
                   side_effect=lambda source,key,device:(object(),original[key],{})), \
             patch('app.modules.card_game.rl.seven_runtime.model',return_value=NS(version='fixture')), \
             patch('app.modules.card_game.rl.seven_runtime.export')as export:
            _,actual,_=initialize('fixture-output',16,'cpu',1,'fixture-source',starting_decks=manual)
        self.assertEqual(actual,manual)
        self.assertEqual(export.call_count,7)

    def test_exact_auxiliary_then_main_precheck_order(self):
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        from app.modules.card_game.rl.seven_schedule import ROSTER
        from app.modules.card_game.rl.build_acceptance import SEED_PHASES
        from app.modules.card_game.rl.device_joint_training import prepare_joint_training_inputs
        decks={d['id']:d for d in STARTER_DECKS if d['id'] in ROSTER}
        aux={p:1000000+i*10000 for i,p in enumerate(SEED_PHASES)}
        main={p:2000000+i*10000 for i,p in enumerate(SEED_PHASES)}
        until=time.time()+1000; order=[]
        phase=NS(collect=lambda jobs,**kw:order.append(('collect',len(jobs)))or ())
        def fit(*args,**kwargs):
            order.append(('fit',None));return {},dict(passed=True,models={'fixture':'only'})
        def calibrate(*args,**kwargs):
            order.append(('main',None));return dict(models='fixture-models',value_precheck={'passed':True})
        def model(source,key):return NS(serving_deck=decks[key],hidden=16)
        with TemporaryDirectory()as folder, \
             patch('app.modules.card_game.rl.seven_runtime.model',side_effect=model), \
             patch('app.modules.card_game.rl.batched_search.device_pure_collection.DevicePureCollector'), \
             patch('app.modules.card_game.rl.device_auxiliary_calibration.create_device_auxiliary_phase_collector',return_value=phase)as factory, \
             patch('app.modules.card_game.rl.device_auxiliary_calibration.fit_device_auxiliary_calibration',side_effect=fit), \
             patch('app.modules.card_game.rl.seven_round.run_device_value_calibration',side_effect=calibrate):
            result=prepare_joint_training_inputs(folder,Path(folder)/'out',executor=object(),starting_decks=decks,
                auxiliary_seeds=aux,main_seeds=main,auxiliary_collect_until=until,auxiliary_fit_until=until+100,
                value_until=until+200,hard_deadline=until+300,checkpoint_segment=lambda *args:None)
            self.assertTrue(result['ready']);self.assertEqual(result['formal_policy_updates'],0)
            self.assertEqual(order,[('collect',168),('collect',56),('collect',56),('fit',None),('main',None)])
            self.assertEqual(factory.call_args.kwargs['segment_decisions'],4)
            # Reject main train/select overlap before creating a second output.
            with self.assertRaisesRegex(ValueError,'partitions overlap'):
                prepare_joint_training_inputs(folder,Path(folder)/'bad',executor=object(),starting_decks=decks,
                    auxiliary_seeds=aux,main_seeds=dict(main,selection=main['adaptation']),
                    auxiliary_collect_until=until,auxiliary_fit_until=until+100,value_until=until+200,
                    hard_deadline=until+300,checkpoint_segment=lambda *args:None)
            self.assertFalse((Path(folder)/'bad').exists())

if __name__=='__main__':unittest.main()
