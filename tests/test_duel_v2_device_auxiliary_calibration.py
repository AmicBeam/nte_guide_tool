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




class AuxiliaryCalibrationLedgerTest(unittest.TestCase):
    """Real safe-checkpoint IO plus tiny CPU oracle; no CUDA qualification."""
    def setUp(self):
        import time
        from app.modules.card_game.rl.device_auxiliary_calibration import DeviceAuxiliaryCalibrationLedger
        from app.modules.card_game.rl.build_acceptance import SEED_PHASES
        from tests import test_duel_v2_device_auxiliary_collection as fixtures
        self.fixtures=fixtures;self.oracle=fixtures.Oracle()
        meta=fixtures.metadata();self.starts=meta.starting_decks
        self.seeds={name:71+i*10000 for i,name in enumerate(SEED_PHASES)}
        self.deadline=time.time()+2000
        self.binding=dict(rule_source='0'*64,image='1'*64,schema='2'*64)
        self.models={k:'f'*64 for k in ROSTER}
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)/'ledger'
        self.cls=DeviceAuxiliaryCalibrationLedger
        self.ledger=self.reopen()

    def reopen(self,**changes):
        args=dict(starting_decks=self.starts,seed_reservation=self.seeds,deadline=self.deadline,
                  source_model_versions=self.models,source_binding=self.binding)
        args.update(changes);return self.cls(self.directory,**args)

    def metadata(self,phase,offset,count,deadline=None):
        from app.modules.card_game.rl.seven_auxiliary_jobs import adapt_auxiliary_jobs
        jobs=[dict(deepcopy(j),deadline=self.deadline if deadline is None else deadline)
              for j in self.ledger.contract['physical_plans'][phase][offset:offset+count]]
        return adapt_auxiliary_jobs(jobs,starting_decks=self.starts)

    def segment(self,metadata,steps=8,prior=None,receipt=None,errors=False):
        from app.modules.card_game.rl.seven_auxiliary_jobs import adapt_auxiliary_jobs
        from app.modules.card_game.rl.device_auxiliary_collection import AuxiliarySegment
        parts=[];raw=[];audit=[];values=[]
        for owner,job in enumerate(metadata.jobs):
            one=adapt_auxiliary_jobs([job],starting_decks=self.starts)
            part=self.fixtures.segment(one,self.oracle,steps=steps)
            parts.append(part)
            for source,dest in ((part.raw_records,raw),(part.audit_records,audit),(part.value_records,values)):
                for row in deepcopy(source):
                    row['owner']=owner
                    if 'game'in row:row['game']=owner
                    dest.append(row)
        if prior is not None:
            raw=list(deepcopy(prior.raw_records))+[r for r in raw if r['step']>=prior.steps[r['owner']]]
            audit=list(deepcopy(prior.audit_records))+[r for r in audit if r['step']>=prior.steps[r['owner']]]
        n=len(parts)
        return AuxiliarySegment(metadata,tuple(p.initial[0]for p in parts),tuple(p.final[0]for p in parts),
            (steps,)*n,(steps==8,)*n,(False,)*n,((256 if errors else 0),)*n,
            tuple(p.pcg_states[0]for p in parts),tuple(raw),tuple(audit),tuple(values),
            ()if prior is None else deepcopy(prior.attempts),'CPU_fixture',decision_limit=512,
            source_binding=self.binding,replay_verified=True,
            budget_continuations=(()if prior is None else deepcopy(prior.budget_continuations))+
                (()if receipt is None else(receipt,)))

    def admit(self,phase='training',offset=0,count=2,steps=7):
        actual=self.segment(self.metadata(phase,offset,count,deadline=time.time()-100),steps)
        with patch('app.modules.card_game.rl.device_auxiliary_collection._oracle',self.oracle.api):
            entry=self.ledger.admit_phase_prefix(phase,offset=offset,segment=actual,
                parent_report_sha256='a'*64,parent_checkpoint_sha256='b'*64,authorization='explicit continue existing prefix')
        return actual,entry

    def factory(self,calls,*,errors=False,steps=8):
        from app.modules.card_game.rl.device_auxiliary_collection import DeviceAuxiliaryCollector
        outer=self
        class CPUFixtureCollector(DeviceAuxiliaryCollector):
            def __init__(self,metadata):self.metadata=metadata
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def run_segment(self,metadata,**kwargs):
                calls.append((metadata,kwargs))
                return outer.segment(metadata,steps,prior=kwargs['resume'],receipt=kwargs['continuation_budget'],errors=errors)
        return CPUFixtureCollector

    def test_admit_existing_two_rows_roundtrip_continue_and_next_offset(self):
        from app.modules.card_game.rl.device_auxiliary_continuation import auxiliary_content_sha256
        old,entry=self.admit();before=auxiliary_content_sha256(old)
        self.ledger=self.reopen();calls=[]
        with patch('app.modules.card_game.rl.device_auxiliary_collection._oracle',self.oracle.api):
            result=self.ledger.run_window(self.factory(calls),deadline=self.deadline,max_segments=1,
                rows=8,authorization='new explicit bounded continuation')
        self.assertEqual(result['status'],'planned_pause');self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][1]['resume'].steps,(7,7))
        self.assertIsNotNone(calls[0][1]['continuation_budget'])
        self.assertEqual(auxiliary_content_sha256(old),before)
        self.assertEqual(self.ledger.manifest['cohorts'][0]['status'],'COMPLETE')
        self.assertEqual(len(self.ledger.manifest['cohorts'][0]['checkpoints']),2)
        calls=[]
        with patch('app.modules.card_game.rl.device_auxiliary_collection._oracle',self.oracle.api):
            self.ledger.run_window(self.factory(calls),deadline=self.deadline,max_segments=1,rows=8,authorization='same authorized run')
        self.assertEqual(self.ledger.manifest['cohorts'][1]['offset'],2)
        self.assertEqual(len(calls[0][0].jobs),8)
        self.assertIsNone(calls[0][1]['resume'])
        with patch('app.modules.card_game.rl.device_auxiliary_collection._oracle',self.oracle.api),self.assertRaises(ValueError):
            self.ledger.admit_phase_prefix('training',offset=0,segment=old,parent_report_sha256='a'*64,
                parent_checkpoint_sha256='b'*64,authorization='cannot count again')

    def test_guard_rejection_is_durable_and_earlier_clean_cannot_wash(self):
        self.admit();calls=[]
        with patch('app.modules.card_game.rl.device_auxiliary_collection._oracle',self.oracle.api):
            result=self.ledger.run_window(self.factory(calls,errors=True,steps=7),deadline=self.deadline,
                authorization='authorized short segment')
        self.assertEqual(result['status'],'rejected')
        self.ledger=self.reopen();again=[]
        result=self.ledger.run_window(self.factory(again),deadline=self.deadline,authorization='no rewind')
        self.assertEqual(result['status'],'rejected');self.assertEqual(again,[])
        self.assertEqual(len(self.ledger.manifest['cohorts'][0]['checkpoints']),2)

    def test_source_seed_build_binding_and_manifest_rollback_rejected(self):
        self.admit()
        with self.assertRaises(ValueError):self.reopen(source_model_versions=dict(self.models,starter='e'*64))
        seeds=dict(self.seeds,foundation=999)
        with self.assertRaises(ValueError):self.reopen(seed_reservation=seeds)
        starts=deepcopy(self.starts);starts['starter']['character_ids'].reverse()
        with self.assertRaises(ValueError):self.reopen(starting_decks=starts)
        oldest=sorted(self.directory.glob('manifest-revision-*.json'))[0]
        (self.directory/'manifest.json').write_bytes(oldest.read_bytes())
        with self.assertRaisesRegex(ValueError,'rollback'):self.reopen()

    def test_clean_pause_persists_latest_steps_and_uncertain_running_cannot_rewind(self):
        calls=[]
        with patch('app.modules.card_game.rl.device_auxiliary_collection._oracle',self.oracle.api):
            first=self.ledger.run_window(self.factory(calls,steps=4),deadline=self.deadline,max_segments=1,
                rows=2,seconds=20,authorization='bounded clean sampling')
            self.assertEqual(first['status'],'planned_pause')
            self.ledger=self.reopen()
            second=self.ledger.run_window(self.factory(calls,steps=5),deadline=self.deadline,max_segments=1,
                rows=8,seconds=20,authorization='continue latest actual prefix')
        self.assertEqual(second['status'],'planned_pause')
        self.assertEqual(calls[1][1]['resume'].steps,(4,4))
        self.assertEqual(len(calls[1][0].jobs),2)
        entry=self.ledger.manifest['cohorts'][0]
        self.assertEqual(entry['checkpoints'][-1]['steps'],[5,5])
        entry['status']='RUNNING';self.ledger._persist()  # Simulated process loss after durable start.
        self.ledger=self.reopen();never=[]
        stopped=self.ledger.run_window(self.factory(never),deadline=self.deadline,authorization='cannot rewind unknown run')
        self.assertEqual(stopped['status'],'rejected');self.assertEqual(never,[])

    def test_checkpoint_corruption_and_mutable_contract_tampering_rejected(self):
        self.admit();entry=self.ledger.manifest['cohorts'][0]
        checkpoint=self.directory/entry['checkpoints'][-1]['path']
        checkpoint.write_bytes(checkpoint.read_bytes()+b'corruption')
        never=[]
        with self.assertRaises(ValueError):
            self.ledger.run_window(self.factory(never),deadline=self.deadline,authorization='must load latest only')
        self.assertEqual(never,[])
        self.ledger.contract['physical_plans']['training'][0]['seed']+=1
        with self.assertRaises(ValueError):self.ledger.completed_phase_segments(deadline=self.deadline)

    def test_full280_actual_fixture_cache_reuses_without_factory_and_only_explicit_ledger_ignores_deadline(self):
        from app.modules.card_game.rl.device_auxiliary_calibration import auxiliary_phase_plans,validate_auxiliary_phases
        with patch('app.modules.card_game.rl.device_auxiliary_collection._oracle',self.oracle.api):
            for phase,count in dict(training=168,selection=56,heldout=56).items():
                for offset in range(0,count,8):self.admit(phase,offset,min(8,count-offset),steps=8)
            phases=self.ledger.completed_phase_segments(deadline=self.deadline)
            self.assertEqual(sum(len(s.metadata.jobs)for batch in phases.values()for s in batch),280)
            never=[];result=self.ledger.run_window(self.factory(never),deadline=self.deadline,authorization='reuse complete actual cache')
            self.assertEqual(result['status'],'complete');self.assertEqual(never,[])
            plans=auxiliary_phase_plans(starting_decks=self.starts,seed_reservation=self.seeds,deadline=self.deadline+1)
            with self.assertRaises(ValueError):validate_auxiliary_phases(phases,plans=plans,deadline=self.deadline)
            games,binding=validate_auxiliary_phases(phases,plans=plans,deadline=self.deadline,ledger=self.ledger)
            self.assertEqual({p:len(v)for p,v in games.items()},dict(training=168,selection=56,heldout=56))
            self.assertEqual(binding,self.binding)
        self.assertTrue(all(not name.startswith('value_')for name in self.ledger.manifest['contract']))


    def external_commit(self,entry,segment,**changes):
        args=dict(offset=entry['offset'],segment=segment,
            expected_parent_checkpoint_sha256=entry['checkpoints'][-1]['sha256'],
            producer_report_sha256='c'*64,producer_checkpoint_sha256='d'*64,
            authorization='independent full CPU reproof of actual GPU prefix')
        args.update(changes)
        return self.ledger.commit_phase_segment(entry['phase'],**args)

    def continued_segment(self,old,steps):
        from app.modules.card_game.rl.device_auxiliary_continuation import create_auxiliary_continuation
        receipt=create_auxiliary_continuation(old,deadline=self.deadline,authorization='explicit continuation',
            parent_report_sha256='e'*64,parent_checkpoint_sha256='f'*64)
        return self.segment(receipt.new_metadata,steps,prior=old,receipt=receipt)

    def test_external_commit_full_proof_terminal_targets_and_persistence(self):
        old,entry=self.admit()
        current=self.continued_segment(old,8)
        from app.modules.card_game.rl import device_auxiliary_collection as aux
        original=aux.verify_auxiliary_segment
        with patch.object(aux,'_oracle',self.oracle.api),patch.object(aux,'verify_auxiliary_segment',wraps=original)as verify:
            committed=self.external_commit(entry,current)
        self.assertGreaterEqual(verify.call_count,1)
        self.assertEqual(committed['status'],'COMPLETE')
        self.assertEqual(len(committed['checkpoints']),2)
        self.assertEqual(self.reopen().manifest,self.ledger.manifest)
        event=self.ledger.manifest['events'][-1]
        self.assertEqual(event['producer_report_sha256'],'c'*64)
        self.assertEqual(event['producer_checkpoint_sha256'],'d'*64)
        # A completed cohort cannot be used as a fresh external update entry.
        with self.assertRaises(ValueError):self.external_commit(committed,current)

    def test_external_commit_failure_never_changes_manifest_or_head(self):
        old,entry=self.admit(steps=5);current=self.continued_segment(old,6)
        from app.modules.card_game.rl import device_auxiliary_collection as aux
        cases=[]
        bad=deepcopy(current);bad.raw_records[0]['raw_action']['index']+=1;cases.append(bad)
        bad=deepcopy(current);bad.audit_records[0]['chosen']+=1;cases.append(bad)
        bad=deepcopy(current);bad.value_records[0]['context'][0]+=1;cases.append(bad)
        bad=deepcopy(current);bad.value_records[0]['z'][0]=1;cases.append(bad)
        cases.append(replace(current,steps=(4,6)))
        cases.append(replace(current,replay_verified=False))
        cases.append(replace(current,source_binding=dict(current.source_binding,image='f'*64)))
        before=deepcopy(self.ledger.manifest);head=self.ledger._manifest_path.read_bytes()
        # Exercise the boundary checks independently of an oracle which already
        # rejects these corruptions. The actual success case above runs full proof.
        with patch.object(aux,'verify_auxiliary_segment',return_value=True):
            for invalid in cases:
                with self.assertRaises(ValueError):self.external_commit(entry,invalid)
                self.assertEqual(self.ledger.manifest,before);self.assertEqual(self.ledger._manifest_path.read_bytes(),head)
            with self.assertRaises(ValueError):self.external_commit(entry,current,expected_parent_checkpoint_sha256='0'*64)
        self.assertEqual(len(list((self.directory/'checkpoints').glob('*.gz'))),1)

    def test_external_commit_unadvanced_state_pcg_and_sticky_status(self):
        old,entry=self.admit(steps=5);same=self.continued_segment(old,5)
        from app.modules.card_game.rl import device_auxiliary_collection as aux
        for field in('final','pcg_states'):
            bad=deepcopy(same);getattr(bad,field)[0]['changed']=True
            with patch.object(aux,'verify_auxiliary_segment',return_value=True),self.assertRaises(ValueError):
                self.external_commit(entry,bad)
        for status in('RUNNING','REJECTED'):
            self.ledger.manifest['cohorts'][0]['status']=status;self.ledger._persist()
            before=deepcopy(self.ledger.manifest)
            with self.assertRaises(ValueError):self.external_commit(entry,same)
            self.assertEqual(self.ledger.manifest,before)
            self.assertEqual(self.reopen().manifest['cohorts'][0]['status'],status)

    def test_external_commit_full_proof_failure_blocks_save(self):
        old,entry=self.admit(steps=5);current=self.continued_segment(old,6)
        from app.modules.card_game.rl import device_auxiliary_collection as aux
        before=deepcopy(self.ledger.manifest)
        with patch.object(aux,'verify_auxiliary_segment',side_effect=ValueError('independent proof failed')):
            with self.assertRaisesRegex(ValueError,'independent proof failed'):self.external_commit(entry,current)
        self.assertEqual(self.ledger.manifest,before)


    def test_external_commit_mixed_cohort_never_advances_completed_row(self):
        metadata=self.metadata('training',0,2,deadline=time.time()-100)
        partial=self.segment(metadata,7);terminal=self.segment(metadata,8)
        def combine(name):
            return tuple(r for r in getattr(terminal,name)if r['owner']==0)+tuple(r for r in getattr(partial,name)if r['owner']==1)
        old=replace(partial,final=(terminal.final[0],partial.final[1]),steps=(8,7),complete=(True,False),
            pcg_states=(terminal.pcg_states[0],partial.pcg_states[1]),
            raw_records=combine('raw_records'),audit_records=combine('audit_records'),value_records=combine('value_records'))
        from app.modules.card_game.rl import device_auxiliary_collection as aux
        with patch.object(aux,'_oracle',self.oracle.api):
            entry=self.ledger.admit_phase_prefix('training',offset=0,segment=old,parent_report_sha256='a'*64,
                parent_checkpoint_sha256='b'*64,authorization='preserve actual mixed completed/paused cohort')
        current=self.continued_segment(old,8)
        before=deepcopy(self.ledger.manifest)
        with patch.object(aux,'verify_auxiliary_segment',return_value=True):
            with self.assertRaisesRegex(ValueError,'Previously complete'):
                self.external_commit(entry,replace(current,steps=(9,8)))
            changed=deepcopy(current);changed.value_records[0]['z'][0]=0
            with self.assertRaises(ValueError):self.external_commit(entry,changed)
        self.assertEqual(self.ledger.manifest,before)
        with patch.object(aux,'_oracle',self.oracle.api):
            self.assertEqual(self.external_commit(entry,current)['status'],'COMPLETE')


    def test_pending_candidate_full_proof_once_preserves_original_and_marks_event(self):
        old,entry=self.admit(steps=5);current=self.continued_segment(old,6)
        pending=replace(current,replay_verified=False,reason='incomplete; pending_independent_cpu_proof')
        from app.modules.card_game.rl import device_auxiliary_collection as aux
        from app.modules.card_game.rl.device_auxiliary_continuation import auxiliary_content_sha256
        before=auxiliary_content_sha256(pending);original=aux.verify_auxiliary_segment
        with self.assertRaises(ValueError):self.external_commit(entry,pending)
        with patch.object(aux,'_oracle',self.oracle.api),patch.object(aux,'verify_auxiliary_segment',wraps=original)as verify:
            committed=self.external_commit(entry,pending,pending_candidate=True)
        self.assertEqual(verify.call_count,1)
        self.assertFalse(pending.replay_verified);self.assertEqual(auxiliary_content_sha256(pending),before)
        restored=self.ledger._load_segment(committed,deadline=self.deadline,verify=False)
        self.assertTrue(restored.replay_verified)
        event=self.ledger.manifest['events'][-1]
        self.assertTrue(event['pending_candidate']);self.assertTrue(event['full_replay_proof_applied'])
        self.assertFalse(event['producer_replay_verified']);self.assertEqual(event['producer_reason'],pending.reason)

    def test_pending_proof_error_and_actual_guard_never_change_head(self):
        old,entry=self.admit(steps=5);current=self.continued_segment(old,6)
        pending=replace(current,replay_verified=False,reason='incomplete; pending_independent_cpu_proof')
        from app.modules.card_game.rl import device_auxiliary_collection as aux
        before=deepcopy(self.ledger.manifest);head=self.ledger._manifest_path.read_bytes()
        with patch.object(aux,'verify_auxiliary_segment',side_effect=TimeoutError('full independent proof exhausted')):
            with self.assertRaises(TimeoutError):self.external_commit(entry,pending,pending_candidate=True)
        for invalid in (replace(pending,errors=(256,0)),replace(pending,reason='incomplete; proof_failed:old'),
                replace(pending,attempts=(dict(memory_limit_exceeded=True),)),replace(pending,replay_verified=True)):
            with self.assertRaises(ValueError):self.external_commit(entry,invalid,pending_candidate=True)
            self.assertEqual(self.ledger.manifest,before);self.assertEqual(self.ledger._manifest_path.read_bytes(),head)
        self.assertEqual(pending.errors,(0,0));self.assertFalse(pending.replay_verified)


if __name__=='__main__':unittest.main()
