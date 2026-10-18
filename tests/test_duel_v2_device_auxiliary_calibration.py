"""CPU ingress/plan fixtures only; mocked exports are not CUDA qualification."""
from copy import deepcopy
from dataclasses import replace
import time
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
import unittest
from unittest.mock import patch

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.rl.build_acceptance import canonical_build, SEED_PHASES
from app.modules.card_game.rl.seven_schedule import ROSTER
from app.modules.card_game.rl.seven_auxiliary_jobs import adapt_auxiliary_jobs
from app.modules.card_game.rl.device_auxiliary_collection import AuxiliarySegment
from app.modules.card_game.rl.device_auxiliary_calibration import (
    auxiliary_phase_plans, validate_auxiliary_phases, DeviceAuxiliaryPhaseCollector)


class AuxiliaryCalibrationIngressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        all_decks={d['id']:d for d in STARTER_DECKS}
        cls.decks={key:canonical_build(all_decks[key]) for key in ROSTER}
        cls.seeds={key:10000+i*10000 for i,key in enumerate(SEED_PHASES)}
        cls.deadline=time.time()+10000
        cls.plans=auxiliary_phase_plans(starting_decks=cls.decks,
            seed_reservation=cls.seeds, deadline=cls.deadline)

    def phases(self):
        phases={}
        for phase,jobs in self.plans.items():
            segment=AuxiliarySegment(adapt_auxiliary_jobs(jobs,starting_decks=self.decks),
                (),(),(),(),(),(),(),(),(),(),(), 'fixture',
                source_binding=dict(rule_source='a'*64,image='b'*64,schema='c'*64),
                replay_verified=True)
            phases[phase]=(segment,)
        return phases

    @staticmethod
    def exported(segment,**kwargs):
        # Deliberately a synthetic export fixture: no rule/GPU run is claimed.
        return [dict(job=deepcopy(job),complete=True,value_rows=[dict(key=job['keys']['a'])],
            rule_backend='cuda_soa', source_binding=segment.source_binding)
            for job in segment.metadata.jobs]

    def test_counts_and_paired_seeds_unchanged(self):
        self.assertEqual({p:len(j) for p,j in self.plans.items()},
                         dict(training=168,selection=56,heldout=56))
        for jobs in self.plans.values():
            self.assertEqual(jobs[0]['seed'],jobs[1]['seed'])
            self.assertNotEqual(jobs[0]['first'],jobs[1]['first'])
        for bad in (dict(training=56,selection=56,heldout=56),
                    dict(training=168,selection=56,heldout=112)):
            with self.assertRaises(ValueError):
                auxiliary_phase_plans(starting_decks=self.decks,seed_reservation=self.seeds,
                    deadline=self.deadline,phase_counts=bad)

    def test_overlapping_phase_seeds_rejected(self):
        seeds=dict(self.seeds,confirmation=self.seeds['foundation'])
        with self.assertRaisesRegex(ValueError,'overlap'):
            auxiliary_phase_plans(starting_decks=self.decks,seed_reservation=seeds,deadline=self.deadline)

    def test_complete_seed_ledger_and_live_deadline_required(self):
        with self.assertRaises(ValueError):
            auxiliary_phase_plans(starting_decks=self.decks,seed_reservation={},deadline=self.deadline)
        with self.assertRaises(TimeoutError):
            auxiliary_phase_plans(starting_decks=self.decks,seed_reservation=self.seeds,deadline=1)

    def test_full_fixture_ingress_exact_order_and_reverify(self):
        with patch('app.modules.card_game.rl.device_auxiliary_collection.verified_auxiliary_games',
                   side_effect=self.exported) as verify:
            games,binding=validate_auxiliary_phases(self.phases(),plans=self.plans,deadline=self.deadline)
        self.assertEqual(verify.call_count,3)
        self.assertEqual([g['job'] for g in games['training']],self.plans['training'])
        self.assertEqual(binding['image'],'b'*64)

    def test_duplicate_missing_build_and_source_mismatches_rejected(self):
        cases=[]
        phases=self.phases();phases['selection']=();cases.append(phases)
        phases=self.phases();phases['selection']*=2;cases.append(phases)
        phases=self.phases();s=phases['selection'][0]
        phases['selection']=(replace(s,source_binding=dict(s.source_binding,image='d'*64)),);cases.append(phases)
        phases=self.phases();s=phases['selection'][0];jobs=deepcopy(list(s.metadata.jobs))
        jobs[0]['decks']['a']['name']='different frozen build'
        phases['selection']=(replace(s,metadata=adapt_auxiliary_jobs(jobs,starting_decks=self.decks)),)
        cases.append(phases)
        for phase_set in cases:
            with self.subTest(case=len(phase_set['selection'])):
                with patch('app.modules.card_game.rl.device_auxiliary_collection.verified_auxiliary_games',
                           side_effect=self.exported), self.assertRaises(ValueError):
                    validate_auxiliary_phases(phase_set,plans=self.plans,deadline=self.deadline)

    def test_actual_verifier_error_not_suppressed(self):
        with patch('app.modules.card_game.rl.device_auxiliary_collection.verified_auxiliary_games',
                   side_effect=ValueError('non-natural terminal')), self.assertRaisesRegex(ValueError,'non-natural'):
            validate_auxiliary_phases(self.phases(),plans=self.plans,deadline=self.deadline)

    def test_shrunken_phase_and_untyped_evidence_rejected(self):
        plans=dict(self.plans,training=self.plans['training'][:56])
        with self.assertRaises(ValueError):
            validate_auxiliary_phases(self.phases(),plans=plans,deadline=self.deadline)
        phases=self.phases();phases['training']=(dict(complete=True),)
        with self.assertRaises(TypeError):
            validate_auxiliary_phases(phases,plans=self.plans,deadline=self.deadline)

    def test_fit_candidate_protocol_with_cpu_mocked_gpu_boundary(self):
        # This exercises orchestration/real AdamW on tiny CPU tensors only.
        # It explicitly mocks CUDA admission and archive verification, never
        # claims that synthetic games or this checkpoint qualify for GPU use.
        import torch
        from app.modules.card_game.rl.device_auxiliary_calibration import fit_device_auxiliary_calibration
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.policy=torch.nn.Linear(1,1)
                self.value_net=torch.nn.Linear(1,1)
                self.wdl=torch.nn.Linear(1,3)
        torch.manual_seed(3)
        originals={key:deepcopy(Tiny().state_dict()) for key in ROSTER}
        snapshots={}; exported=[]; metrics=[]; update_count=[]
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';out=Path(temp)/'new';source.mkdir()
            def restore(folder,key,device):
                self.assertEqual(device,'cuda')
                net=Tiny();net.load_state_dict(originals[key] if Path(folder)==source else snapshots[str(folder),key])
                return net,self.decks[key],dict(test_fixture=True)
            def export(net,folder,key,build,origin):
                snapshots[str(folder),key]=deepcopy(net.state_dict())
                exported.append((str(folder),key,deepcopy(origin)))
                self.assertTrue(torch.equal(net.policy.weight,originals[key]['policy.weight']))
            def model(folder,key):
                return SimpleNamespace(serving_deck=self.decks[key],directory=str(folder),key=key)
            def update(net,opt,rows,value_only):
                self.assertTrue(value_only);update_count.append(1)
                opt.zero_grad();loss=net.wdl(net.value_net(torch.ones(1,1))).square().sum()
                loss.backward();opt.step()
            def value_metrics(games,models,backend,device):
                self.assertEqual((backend,device),('cuda_batched','cuda'))
                metrics.append(tuple(models))
                return {key:dict(normal_games=8,brier=.4,log_loss=.8) for key in models}
            def scope(folder):
                report=json.loads((Path(folder)/'scope.json').read_text())
                self.assertTrue(report['passed'])
                self.assertEqual(report['real_learner_updates'],0)
                self.assertEqual(len(report['choices']),7)
                self.assertEqual(len(list(Path(folder).glob('*-private.json.gz'))),280)
                self.assertEqual(len(list(Path(folder).glob('*-segment-*-raw.json.gz'))),3)
                return {},report
            with ExitStack() as stack:
                for target,value in (
                    ('torch.cuda.is_available',lambda:True),
                    ('app.modules.card_game.rl.device_auxiliary_collection.verified_auxiliary_games',self.exported),
                    ('app.modules.card_game.rl.seven_runtime.restore',restore),
                    ('app.modules.card_game.rl.seven_runtime.export',export),
                    ('app.modules.card_game.rl.seven_runtime.model',model),
                    ('app.modules.card_game.rl.seven_runtime.update',update),
                    ('app.modules.card_game.rl.seven_round.value_metrics',value_metrics),
                    ('app.modules.card_game.rl.seven_round.versions',lambda folder:{key:'fixture' for key in ROSTER}),
                    ('app.modules.card_game.rl.seven_covered_value.load_scope',scope)):
                    stack.enter_context(patch(target,side_effect=value))
                _,report=fit_device_auxiliary_calibration(source,out,starting_decks=self.decks,
                    seed_reservation=self.seeds,phase_segments=self.phases(),deadline=self.deadline)
            self.assertEqual(len(metrics),7*4*3+1)
            self.assertEqual(len(update_count),7*8)
            self.assertEqual(len(exported),7*(4+12+1))
            self.assertTrue(all(c['epoch']==1 and c['temperature']==1 for c in report['choices'].values()))

    def test_default_factory_fresh_arenas_borrowed_executor_and_failure_cleanup(self):
        from app.modules.card_game.rl.device_auxiliary_calibration import create_device_auxiliary_phase_collector
        stores=[]; collectors=[]
        class Executor:
            execute_only=False
            device=SimpleNamespace(type='cuda')
            schema=SimpleNamespace(lengths=dict(i64=10,f64=10,i32=10,u8=10),
                                   total_objects=10,max_strings=10)
            image=SimpleNamespace(nodes=[0,1,2])
            torch=SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda device:(15<<30,16<<30)))
            close_count=0
            def close(self):self.close_count+=1
        class Store:
            def __init__(self,*args,**kwargs):
                self.args,self.kwargs,self.closes=args,kwargs,0;stores.append(self)
            def close(self):self.closes+=1
        class Collector:
            def __init__(self,executor,*arenas,**kwargs):
                self.executor,self.arenas,self.kwargs=executor,arenas,kwargs;collectors.append(self)
            def _memory(self):return 1<<30
            def close(self):
                for arena in self.arenas:arena.close()
        metadata=adapt_auxiliary_jobs(self.plans['selection'][:3],starting_decks=self.decks)
        with ExitStack() as stack:
            stack.enter_context(patch('app.modules.card_game.rl.batched_duel.native_soa_cuda.CudaSoARules',Executor))
            stack.enter_context(patch('app.modules.card_game.rl.batched_duel.device_sparse_soa_storage.DeviceSparseSoAStore',Store))
            stack.enter_context(patch('app.modules.card_game.rl.batched_duel.device_graph_storage.CompactDeviceGraphStore',Store))
            stack.enter_context(patch('app.modules.card_game.rl.device_auxiliary_collection.DeviceAuxiliaryCollector',Collector))
            cache_extent=stack.enter_context(patch(
                'app.modules.card_game.rl.batched_duel.vm_soa_views.value_cache_extent',return_value=30))
            workspace_bound=stack.enter_context(patch(
                'app.modules.card_game.rl.batched_search.soa_capacity.rule_workspace_minimum_bytes',return_value=1<<30))
            executor=Executor()
            phases=create_device_auxiliary_phase_collector(executor,starting_decks=self.decks,rows=8)
            a,b=phases.factory(metadata),phases.factory(metadata)
            self.assertEqual(len(stores),6)
            workspace_bound.assert_called_with(3,schema_lengths=executor.schema.lengths,
                total_objects=10,value_cache_entries=30,constant_nodes=3,max_strings=10)
            cache_extent.assert_called_with(executor.schema)
            self.assertEqual(len({id(s) for s in stores}),6)
            self.assertEqual(a.kwargs['rows'],3)
            self.assertEqual(a.kwargs['max_decisions'],512)
            self.assertEqual(a.kwargs['segment_decisions'],16)
            self.assertEqual(a.arenas[2].kwargs['slots'],3*512)
            self.assertEqual(a.kwargs['max_device_bytes'],14336<<20)
            self.assertFalse(any(s.closes for s in stores))
            a.close();b.close()
            self.assertTrue(all(s.closes==1 for s in stores))
            self.assertEqual(executor.close_count,0)
            before=len(stores)
            with patch('app.modules.card_game.rl.device_auxiliary_collection.DeviceAuxiliaryCollector',
                       side_effect=RuntimeError('collector allocation failed')):
                # Recreate closure to capture the currently injected constructor.
                phases=create_device_auxiliary_phase_collector(executor,starting_decks=self.decks)
                with self.assertRaisesRegex(RuntimeError,'allocation failed'):
                    phases.factory(metadata)
            self.assertTrue(all(s.closes==1 for s in stores[before:]))
            self.assertEqual(executor.close_count,0)
            before=len(stores);graph_calls=[]
            def graph_fail(**kwargs):
                graph_calls.append(1)
                if len(graph_calls)==2:raise MemoryError('audit arena failed')
                return Store(**kwargs)
            with patch('app.modules.card_game.rl.batched_duel.device_graph_storage.CompactDeviceGraphStore',
                       side_effect=graph_fail):
                phases=create_device_auxiliary_phase_collector(executor,starting_decks=self.decks)
                with self.assertRaisesRegex(MemoryError,'audit arena failed'):
                    phases.factory(metadata)
            self.assertEqual(len(stores)-before,2)
            self.assertTrue(all(s.closes==1 for s in stores[before:]))
            self.assertEqual(executor.close_count,0)
            before=len(stores);before_collectors=len(collectors)
            workspace_bound.return_value=14<<30
            phases=create_device_auxiliary_phase_collector(executor,starting_decks=self.decks)
            with self.assertRaisesRegex(MemoryError,'workspace plus resident'):
                phases.factory(metadata)
            self.assertEqual(len(stores)-before,3)
            self.assertTrue(all(s.closes==1 for s in stores[before:]))
            self.assertEqual(len(collectors),before_collectors)
            self.assertEqual(executor.close_count,0)
            for kwargs in (dict(rows=65),dict(segment_decisions=17),dict(max_device_bytes=14337<<20)):
                with self.assertRaises(ValueError):
                    create_device_auxiliary_phase_collector(executor,starting_decks=self.decks,**kwargs)

    def test_factory_and_cohort_guard(self):
        for rows in (0,65,True):
            with self.assertRaises(ValueError):
                DeviceAuxiliaryPhaseCollector(lambda m:None,starting_decks=self.decks,rows=rows)
        collector=DeviceAuxiliaryPhaseCollector(lambda m:object(),starting_decks=self.decks)
        with self.assertRaises(TypeError):
            collector.collect(self.plans['selection'][:2],deadline=self.deadline)


if __name__=='__main__':unittest.main()
