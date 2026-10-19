"""Synthetic CPU trust/induction fixtures; no CUDA or training qualification."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from app.modules.card_game.rl import device_auxiliary_collection as aux
from app.modules.card_game.rl import device_auxiliary_continuation as wire
from app.modules.card_game.rl import device_auxiliary_proof as proof
from tests import test_duel_v2_device_auxiliary_calibration as ledger_fixtures


class AuxiliaryInductiveProofTests(unittest.TestCase):
    def setUp(self):
        self.f=ledger_fixtures.AuxiliaryCalibrationLedgerTest('test_external_commit_full_proof_failure_blocks_save')
        self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.root=self.f.directory.parent;self.counter=0
        self.worker=self.root/'cpu-fixture.py';self.worker.write_text('# Explicit synthetic CPU fixture, never CUDA evidence\n')
        self.worker_sha=sha256(self.worker.read_bytes()).hexdigest()
        self.old,self.entry=self.f.admit(steps=5)

    def descriptor(self,path):
        return dict(path=path.relative_to(self.root).as_posix(),sha256=sha256(path.read_bytes()).hexdigest())

    def write_json(self,name,value):
        path=self.root/name;path.write_text(json.dumps(value,sort_keys=True,indent=2));return self.descriptor(path)

    def seal(self,claim):
        value=json.loads((self.root/claim['path']).read_text());n=self.counter;self.counter+=1
        contract=json.loads((self.root/value['revision']['path']).read_text())['contract']
        # This is an explicit fake supervisor fixture. Production callers must
        # pin these hashes from their actual process supervisor, never a report.
        report=self.write_json(f'cpu-report-{n}.json',dict(schema='auxiliary_prefix_cpu_proof_v1',passed=True,
            cuda_initialized=False,cleanup_complete=True,producer_inputs_unchanged=True,
            claim_sha256=claim['sha256'],checkpoint_sha256=value['checkpoint']['sha256'],
            committed_revision_sha256=value['revision']['sha256'],worker_sha256=self.worker_sha,
            contract_sha256=value['contract_sha256'],source_binding=value['source_binding'],
            source_model_versions=contract['source_model_versions'],proof_mode=value['proof_mode'],learning_updates=0,qualified_games=0))
        process=self.write_json(f'cpu-process-{n}.json',dict(schema='auxiliary_prefix_cpu_process_v1',returncode=0,
            reason=None,cpu_only=True,outer_seconds=180,child_pid=os.getpid(),seconds=1.,worker_sha256=self.worker_sha,
            cpu_report_sha256=report['sha256']))
        witness=self.write_json(f'witness-{n}.json',dict(schema='auxiliary_prefix_supervisor_witness_v1',
            claim=claim,cpu_report=report,process_receipt=process,worker=self.descriptor(self.worker)))
        return proof.AuxiliaryProofAnchor(self.root,witness['path'],witness['sha256'],self.worker_sha)

    def claim(self,entry,parent=None):
        checkpoint=self.descriptor(self.f.directory/entry['checkpoints'][-1]['path'])
        revision=self.descriptor(self.f.directory/f"manifest-revision-{self.f.ledger.manifest['revision']:08d}.json")
        with patch.object(aux,'_oracle',self.f.oracle.api):
            return proof.write_auxiliary_proof_claim(self.root,f'claim-{self.counter}.json',checkpoint=checkpoint,
                revision=revision,phase='training',offset=0,proof_deadline=time.monotonic()+20,parent_proof=parent)

    def bootstrap(self):
        claim=self.claim(self.entry);anchor=self.seal(claim)
        return proof.load_auxiliary_parent_proof(anchor),anchor

    def test_bootstrap_requires_actual_full_verification_and_explicit_anchor(self):
        with patch.object(aux,'verify_auxiliary_segment',side_effect=ValueError('full oracle failed')):
            with self.assertRaisesRegex(ValueError,'full oracle failed'):self.claim(self.entry)
        cap,anchor=self.bootstrap()
        self.assertTrue(proof.verify_auxiliary_parent_checkpoint(self.old,cap))
        with self.assertRaises(TypeError):proof.AuxiliaryParentProof()
        for invalid in ({'passed':True},json.loads((self.root/anchor.witness).read_text()),True):
            with self.assertRaises(TypeError):proof.load_auxiliary_parent_proof(invalid)
            with patch.object(aux,'_oracle',self.f.oracle.api),self.assertRaises(TypeError):
                aux.verify_auxiliary_segment(self.old,parent_proof=invalid)
        with patch.object(proof,'implementation_binding',return_value={'changed':'verifier'}),self.assertRaises(ValueError):
            proof.verify_auxiliary_parent_checkpoint(self.old,cap)

    def test_induction_runs_only_delta_same_oracle_across_exploration_boundary(self):
        cap,_=self.bootstrap();current=self.f.continued_segment(self.old,7)
        counts=dict(new=0,apply=0)
        def oracle():
            api=list(self.f.oracle.api());new,apply=api[0],api[3]
            def observed_new(**kw):counts['new']+=1;return new(**kw)
            def observed_apply(*args):counts['apply']+=1;return apply(*args)
            api[0]=observed_new;api[3]=observed_apply;return tuple(api)
        with patch.object(aux,'_oracle',oracle):self.assertTrue(aux.verify_auxiliary_segment(current))
        self.assertEqual(counts,dict(new=2,apply=14));counts.update(new=0,apply=0)
        with patch.object(aux,'_oracle',oracle):self.assertTrue(aux.verify_auxiliary_segment(current,parent_proof=cap))
        self.assertEqual(counts,dict(new=0,apply=4))
        terminal=self.f.continued_segment(self.old,8)
        with patch.object(aux,'_oracle',oracle):self.assertTrue(aux.verify_auxiliary_segment(terminal,parent_proof=cap))
        with patch.object(aux,'_oracle',side_effect=AssertionError('resume must not replay rules')):
            self.assertTrue(proof.verify_auxiliary_parent_checkpoint(self.old,cap))

    def test_incremental_mutations_guards_and_unadvanced_state_are_rejected(self):
        cap,_=self.bootstrap();current=self.f.continued_segment(self.old,7);cases=[]
        for name,field in(('raw_records','raw_action'),('audit_records','chosen'),('value_records','context'),('value_records','x'),('value_records','z')):
            bad=deepcopy(current)
            if field=='raw_action':bad.raw_records[0][field]['index']+=1
            elif field=='chosen':bad.audit_records[0][field]+=1
            else:getattr(bad,name)[0][field].flat[0]+=1
            cases.append(bad)
        cases.extend((replace(current,errors=(256,0)),replace(current,truncated=(True,False)),
            replace(current,source_binding=dict(current.source_binding,image='f'*64)),
            replace(current,attempts=(dict(errors=(0,0),raw_errors=(0,0),value_errors=(0,0),memory_limit_exceeded=True),)),
            replace(current,steps=(4,7))))
        for bad in cases:
            with patch.object(aux,'_oracle',self.f.oracle.api),self.assertRaises(ValueError):
                aux.verify_auxiliary_segment(bad,parent_proof=cap)
        unchanged=self.f.continued_segment(self.old,5)
        for field in('final','pcg_states'):
            bad=deepcopy(unchanged);getattr(bad,field)[0]['tampered']=1
            with self.assertRaises(ValueError):aux.verify_auxiliary_segment(bad,parent_proof=cap)
        changed=replace(self.old,reason='changed')
        with self.assertRaises(ValueError):proof.verify_auxiliary_parent_checkpoint(changed,cap)

    def test_loaded_capability_rejects_later_checkpoint_or_receipt_mutation(self):
        cap,anchor=self.bootstrap()
        w=json.loads((self.root/anchor.witness).read_text())
        report=self.root/w['cpu_report']['path'];report.write_text('changed normal-exit proof report')
        with self.assertRaises(ValueError):proof.verify_auxiliary_parent_checkpoint(self.old,cap)
        with self.assertRaises(TypeError):
            proof.require_auxiliary_parent_proof(object.__new__(proof.AuxiliaryParentProof))

    def test_transitive_rl_tree_binding_tracks_helpers_and_refuses_external_symlinks(self):
        package=self.root/'isolated-rl';package.mkdir()
        engine=self.root/'engine';engine.mkdir();(engine/'events.py').write_text('# Direct shared event vocabulary fixture')
        module=package/'device_auxiliary_proof.py';module.write_text('# CPU tree fixture')
        nested=package/'nested';nested.mkdir();helper=nested/'helper.py';helper.write_text('VALUE=1')
        cache=package/'__pycache__';cache.mkdir();(cache/'ignored.py').write_text('not source')
        with patch.object(proof,'__file__',str(module)):
            before=proof.implementation_binding()
            self.assertEqual(list(before['files']),['device_auxiliary_proof.py','nested/helper.py'])
            helper.write_text('VALUE=2')
            self.assertNotEqual(before,proof.implementation_binding())
            changed=proof.implementation_binding();(engine/'events.py').write_text('# Changed shared rule event vocabulary')
            self.assertNotEqual(changed,proof.implementation_binding())
            alias=package/'alias';alias.symlink_to(nested,target_is_directory=True)
            linked=proof.implementation_binding()
            self.assertEqual(linked['symlinks'],{'alias':'nested'})
            other=package/'other';other.mkdir();alias.unlink();alias.symlink_to(other,target_is_directory=True)
            self.assertNotEqual(linked,proof.implementation_binding())
            external=self.root/'external.py';external.write_text('# Outside RL tree; must never be hashed')
            (package/'outside.py').symlink_to(external)
            original=Path.read_bytes
            def contained_read(path):
                if path.resolve()==external.resolve():raise AssertionError('external source read')
                return original(path)
            with patch.object(Path,'read_bytes',contained_read),self.assertRaises(ValueError):
                proof.implementation_binding()

    def test_gpu_ingress_reuses_only_the_exact_certified_checkpoint_before_allocations(self):
        cap,_=self.bootstrap();current=self.f.continued_segment(self.old,7)
        receipt=current.budget_continuations[-1]
        class BeforeGPUAllocation(RuntimeError):pass
        class Gate(aux.DeviceAuxiliaryCollector):
            @property
            def _objects(self):raise BeforeGPUAllocation('validated parent; no GPU objects created')
        gate=object.__new__(Gate);gate._source_binding=self.old.source_binding
        gate._used=False;gate.rows=2;gate.max_decisions=512;gate._memory=lambda:0
        with patch.object(aux,'_oracle',side_effect=AssertionError('old rules must not run')):
            with self.assertRaises(BeforeGPUAllocation):
                gate._run_segment(receipt.new_metadata,deadline=self.f.deadline,resume=self.old,
                    continuation_budget=receipt,parent_proof=cap)
        self.assertTrue(gate._used)
        gate._used=False
        with self.assertRaises(TypeError):
            gate._run_segment(receipt.new_metadata,deadline=self.f.deadline,resume=self.old,
                continuation_budget=receipt,parent_proof={'replay_verified':True})
        self.assertFalse(gate._used)

    def test_normal_process_worker_and_immutable_revision_are_not_boolean_claims(self):
        cap,anchor=self.bootstrap();w=json.loads((self.root/anchor.witness).read_text())
        process=json.loads((self.root/w['process_receipt']['path']).read_text());process['returncode']=1
        w['process_receipt']=self.write_json('failed-process.json',process)
        bad=self.write_json('bad-witness.json',w)
        with self.assertRaises(ValueError):proof.load_auxiliary_parent_proof(replace(anchor,witness=bad['path'],witness_sha256=bad['sha256']))
        with self.assertRaises(ValueError):proof.load_auxiliary_parent_proof(replace(anchor,worker_sha256='a'*64))
        self.worker.write_text('changed worker')
        with self.assertRaises(ValueError):proof.load_auxiliary_parent_proof(anchor)

    def test_pending_commit_then_load_real_inductive_witness_chain(self):
        cap,_=self.bootstrap();current=self.f.continued_segment(self.old,7)
        pending=replace(current,replay_verified=False,reason='incomplete; pending_independent_cpu_proof')
        with patch.object(aux,'_oracle',self.f.oracle.api):
            entry=self.f.external_commit(self.entry,pending,pending_candidate=True,parent_proof=cap)
        event=self.f.ledger.manifest['events'][-1]
        self.assertEqual(event['proof_mode'],'inductive');self.assertTrue(event['full_prefix_proof_applied'])
        self.assertFalse(event['full_replay_proof_applied']);self.assertFalse(pending.replay_verified)
        next_cap=proof.load_auxiliary_parent_proof(self.seal(self.claim(entry,cap)))
        parent=self.f.ledger._load_segment(entry,deadline=self.f.deadline,verify=False)
        terminal=self.f.segment(parent.metadata,8,prior=parent)
        with patch.object(aux,'_oracle',self.f.oracle.api):
            self.assertTrue(aux.verify_auxiliary_segment(terminal,parent_proof=next_cap))
        with self.assertRaises(ValueError):
            proof.require_auxiliary_parent_proof(cap,checkpoint_sha256=entry['checkpoints'][-1]['sha256'])
        altered=deepcopy(terminal);altered.value_records[0]['x'][0,0]+=1
        with self.assertRaises(ValueError):aux.verify_auxiliary_segment(altered,parent_proof=next_cap)


if __name__=='__main__':unittest.main()
