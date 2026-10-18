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



class DeviceJointPersistentStagesTest(unittest.TestCase):
    """Actual tiny CPU optimizer with explicitly mocked CUDA collection ingress."""
    def setUp(self):
        import numpy as np
        import torch
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.rl import seven_runtime as rt
        from app.modules.card_game.rl.seven_schedule import ROSTER
        torch.set_num_threads(1);torch.manual_seed(967)
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.source=Path(self.temp.name)/'source';self.out=Path(self.temp.name)/'out';self.deadline=time.time()+600
        decks={d['id']:d for d in STARTER_DECKS}
        for key in ROSTER:rt.export(rt.create_network(key,16,'cpu'),self.source,key,decks[key],dict(fixture=True))
        state=new_game(seed=1,decks={'a':decks['starter'],'b':decks['starter']})
        actions,(x,c)=rt.decision(state,'a');pi=np.zeros(len(actions),np.float32);pi[-1]=1
        self.game=dict(id='joint',complete=True,learning_eligible=True,winner='a',decisions=1,searched=1,
            episode_actions=[dict(side='a',action=actions[-1])],
            rows=[dict(x=x,c=c,pi=pi,z=1.,side='a',key='starter',step=0,value_actor=1,search_budget=32,search_algorithm='gumbel')],
            value_rows=[dict(x=x,z=-1.,side='b',key='starter',step=0,value_actor=0)])
        common=dict(first='a',left='starter',right='starter',deadline=self.deadline,train_sides=['a'],
            policies={s:(str(self.source),'starter')for s in('a','b')})
        self.train=[dict(common,id='joint',seed=1,training=True,pure=False)]
        self.evaluation=[dict(common,id='eval',seed=2,training=False,pure=True)]
        self.h3validate=patch('app.modules.card_game.rl.batched_search.device_h3_collection.validate_h3_inputs').start()
        self.purevalidate=patch('app.modules.card_game.rl.batched_search.soa_pure_jobs.adapt_pure_jobs').start()
        self.addCleanup(patch.stopall)

    def collectors(self,sample=None,evaluation=None):
        import torch
        h3=object.__new__(DeviceH3Collector);pure=object.__new__(DevicePureCollector)
        h3.executor=pure.executor=NS(device=torch.device('cpu'),rule_source_sha256='a'*64,
            image=NS(identity='b'*64),schema=NS(identity='c'*64))
        h3.last_diagnostics={'complete':True,'cpu_fixture':True,'all_full_budget_roots':True}
        pure.last_diagnostics={'complete':True,'cpu_fixture':True}
        h3.collect=Mock(return_value=[deepcopy(self.game)])if sample is None else Mock(side_effect=sample)
        def evaluated(jobs,bank,**kwargs):
            self.assertTrue(all(Path(model.directory).resolve()==(self.out/'models').resolve()for model in bank.values()))
            return[dict(deepcopy(self.game),id=jobs[0]['id'],rows=[],searched=0)]
        pure.collect=Mock(side_effect=evaluated if evaluation is None else evaluation)
        return h3,pure

    def run_stage(self,h3,pure,resume=False,**kwargs):
        args=dict(collector=h3,pure_collector=pure,source=self.source,train_jobs=self.train,
            evaluation_jobs=self.evaluation,deadline=self.deadline,policy_passes=4,resume=resume,
            max_chunks=1,evaluation_max_segments=1)
        args.update(kwargs)
        return run_joint_training_smoke(self.out,**args)

    def test_new_process_sampling_update_once_evaluation_pause_cache_complete(self):
        from app.modules.card_game.rl.batched_search.device_h3_segments import H3SegmentPaused
        from app.modules.card_game.rl.batched_search.device_pure_segments import PureSegmentPaused
        from app.modules.card_game.rl.seven_round import update_batch
        import json
        with patch('app.modules.card_game.rl.seven_round.update_batch',wraps=update_batch)as update:
            h3,pure=self.collectors(sample=H3SegmentPaused('actual-synthetic-checkpoint'))
            report=self.run_stage(h3,pure)
            self.assertEqual(report['stage'],'gumbel_h3_sampling_paused');self.assertEqual(report['formal_policy_updates'],0)
            h3,pure=self.collectors();report=self.run_stage(h3,pure,resume=True)
            self.assertEqual(report['stage'],'joint_training_samples_cached');self.assertEqual(report['formal_policy_updates'],0)
            self.assertTrue((self.out/'training-episodes/arrays.npz').is_file());pure.collect.assert_not_called()
            h3,pure=self.collectors(sample=AssertionError('must reuse complete sampling cache'))
            report=self.run_stage(h3,pure,resume=True)
            self.assertEqual(report['stage'],'joint_update_committed');self.assertEqual(report['formal_policy_updates'],4)
            h3.collect.assert_not_called();pure.collect.assert_not_called();self.assertEqual(update.call_count,1)
            committed=json.loads((self.out/'stage.json').read_text())
            self.assertEqual(len(committed['updated_models']),14);self.assertEqual(len(committed['optimizer_files']),1)
            h3,pure=self.collectors(sample=AssertionError('no sample'),evaluation=PureSegmentPaused('eval-synthetic-checkpoint'))
            report=self.run_stage(h3,pure,resume=True)
            self.assertEqual(report['stage'],'frozen_evaluation_paused');self.assertEqual(report['formal_policy_updates'],4)
            h3,pure=self.collectors(sample=AssertionError('no sample'));report=self.run_stage(h3,pure,resume=True)
            self.assertEqual(report['stage'],'frozen_evaluation_cached');self.assertEqual(update.call_count,1)
            h3,pure=self.collectors(sample=AssertionError('no sample'),evaluation=AssertionError('no eval'))
            report=self.run_stage(h3,pure,resume=True)
            self.assertTrue(report['passed']);self.assertEqual(report['stage'],'complete')
            import gzip
            with gzip.open(self.out/'evaluation/eval.json.gz','rt')as stream:replay=json.load(stream)
            self.assertTrue(all(Path(v[0]).resolve()==(self.out/'models').resolve()for v in replay['job']['policies'].values()))
            again=self.run_stage(h3,pure,resume=True)
            self.assertEqual(again,report);h3.collect.assert_not_called();pure.collect.assert_not_called()
            self.assertEqual(update.call_count,1)

    def test_source_or_cached_episode_tamper_cannot_reach_update(self):
        h3,pure=self.collectors();self.run_stage(h3,pure)
        path=self.out/'training-episodes/arrays.npz';path.write_bytes(path.read_bytes()+b'tampered')
        with patch('app.modules.card_game.rl.seven_round.update_batch')as update,self.assertRaises(ValueError):
            self.run_stage(*self.collectors(),resume=True)
        update.assert_not_called()
        source=self.source/'starter.json';source.write_text(source.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'source/model/profile'):
            self.run_stage(*self.collectors(),resume=True)

    def test_optimizer_uncertainty_is_sticky_and_updated_bank_tamper_rejected(self):
        h3,pure=self.collectors();self.run_stage(h3,pure)
        with patch('app.modules.card_game.rl.seven_round.update_batch',side_effect=RuntimeError('crash during actual update')):
            with self.assertRaisesRegex(RuntimeError,'crash during'):
                self.run_stage(*self.collectors(),resume=True)
        with patch('app.modules.card_game.rl.seven_round.update_batch')as update,self.assertRaisesRegex(RuntimeError,'cannot be repeated'):
            self.run_stage(*self.collectors(),resume=True)
        update.assert_not_called()

    def test_new_deadline_requires_receipt_authorization_and_keeps_original_jobs(self):
        from app.modules.card_game.rl.batched_search.device_h3_segments import H3SegmentPaused
        import json
        old=self.deadline
        self.run_stage(*self.collectors(sample=H3SegmentPaused('old-checkpoint')))
        self.deadline+=600
        self.train[0]['deadline']=self.evaluation[0]['deadline']=self.deadline
        with self.assertRaisesRegex(ValueError,'continuation receipt'):
            self.run_stage(*self.collectors(),resume=True)
        h3,pure=self.collectors()
        result=self.run_stage(h3,pure,resume=True,continuation={'training':{'explicit':'CPU_fixture_only'}},
                              resume_authorization='new explicit deadline-only budget')
        self.assertEqual(result['stage'],'joint_training_samples_cached')
        self.assertEqual(h3.collect.call_args.kwargs['continuation'],{'explicit':'CPU_fixture_only'})
        binding=json.loads((self.out/'input-binding.json').read_text())
        self.assertEqual(binding['original_deadline'],old)
        self.assertEqual(binding['original_train_jobs'][0]['deadline'],old)
        self.train[0]['seed']+=9
        with self.assertRaisesRegex(ValueError,'source/model/profile'):
            self.run_stage(*self.collectors(),resume=True)

    def test_committed_updated_bank_and_report_are_sha_frozen(self):
        import json
        self.run_stage(*self.collectors())
        self.run_stage(*self.collectors(),resume=True)
        marker=json.loads((self.out/'stage.json').read_text())
        self.assertEqual(marker['stage'],'evaluation')
        path=self.out/'models/starter.npz';path.write_bytes(path.read_bytes()+b'changed')
        h3,pure=self.collectors()
        with patch('app.modules.card_game.rl.seven_round.update_batch')as update,self.assertRaisesRegex(ValueError,'numerical models changed'):
            self.run_stage(h3,pure,resume=True)
        update.assert_not_called();h3.collect.assert_not_called();pure.collect.assert_not_called()

    def test_short_root_budget_cannot_become_cached_learning_rows(self):
        bad=deepcopy(self.game);bad['rows'][0]['search_budget']=17
        h3,pure=self.collectors();h3.collect.return_value=[bad]
        with patch('app.modules.card_game.rl.seven_round.update_batch')as update,self.assertRaisesRegex(ValueError,'root budget'):
            self.run_stage(h3,pure)
        update.assert_not_called();self.assertFalse((self.out/'training-episodes').exists())


if __name__=='__main__':unittest.main()
