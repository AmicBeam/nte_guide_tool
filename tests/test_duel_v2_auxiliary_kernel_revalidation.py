"""Mocked GPU-ingress, real CPU prefix/codec/ledger tests; no CUDA evidence."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from app.modules.card_game.rl import device_auxiliary_collection as aux
from app.modules.card_game.rl import device_auxiliary_continuation as continuation
from app.modules.card_game.rl.batched_duel import native_preview_arena as arena
from tests import test_duel_v2_device_auxiliary_collection as fixtures

ACTUAL_REPORT_VERIFIER=continuation.verify_auxiliary_kernel_fix_report


class KernelRevalidationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name);self.oracle=fixtures.Oracle()
        binding=dict(backend=arena.BACKEND_ABI,program=arena.PROGRAM_IDENTITY,image=arena.IMAGE_IDENTITY,
            schema=arena.SCHEMA_IDENTITY,canonical_units=dict(arena.CANONICAL_UNIT_SHA256),
            derived_units=dict(arena.CANONICAL_UNIT_SHA256),
            reclaim='private typed slots only; heap and strings retained',cuda_qualified=False)
        binding['derived_units'][arena.TARGET_UNIT]='d'*64
        self.new=dict(rule_source=arena._digest(binding),image=arena.IMAGE_IDENTITY,schema=arena.SCHEMA_IDENTITY)
        self.old_binding=dict(self.new,rule_source=continuation._ORIGINAL_PREVIEW_SOURCE)
        self.old=replace(fixtures.segment(fixtures.metadata(),self.oracle,steps=7),source_binding=self.old_binding)
        budget=continuation.create_auxiliary_continuation(self.old,deadline=self.old.metadata.jobs[0]['deadline']+100,
            authorization='fixture explicit extra scheduling budget',parent_report_sha256='a'*64,parent_checkpoint_sha256='b'*64)
        self.old=replace(self.old,metadata=budget.new_metadata,budget_continuations=(budget,))
        self.original_checkpoint='c'*64
        physical=[{k:deepcopy(v)for k,v in j.items()if k!='deadline'}for j in self.old.metadata.jobs]
        fix_jobs=[dict(physical[0],seed=1000+i)for i in range(8)]
        self.ledger=dict(contract=dict(source_binding=self.old_binding,physical_plans=dict(training=physical+fix_jobs),
            starting_decks=self.old.metadata.starting_decks,source_model_versions=dict(fixture='e'*64),
            seed_reservation=dict(fixture=123),decision_limit=512,rule_hash='4'*64),
            cohorts=[dict(phase='training',offset=0,count=1,status='PAUSED',checkpoints=[
                dict(sha256=self.original_checkpoint,content_sha256=continuation.auxiliary_content_sha256(self.old))]),
                dict(phase='training',offset=1,count=8,status='REJECTED',checkpoints=[
                    dict(sha256='8'*64,content_sha256='9'*64)])])
        self.ledger['contract_sha256']=sha256(json.dumps(self.ledger['contract'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self.ledger_path=self.directory/'old-ledger.json';self.ledger_sha=self.write(self.ledger_path,self.ledger)
        self.report=dict(source_binding=self.old_binding,ledger_manifest_sha256=self.ledger_sha,
            ledger_contract_sha256=self.ledger['contract_sha256'],
            windows=[dict(status='rejected',manifest_sha256=self.ledger_sha)])
        self.report_path=self.directory/'old-failed-report.json';self.report_sha=self.write(self.report_path,self.report)
        # Explicit MOCK snapshot. No test attempts to pass it to the real GPU
        # evidence verifier or count it as actual GPU qualification.
        self.evidence=dict(adapter_sha256='f'*64,report=dict(original_source_binding=self.old_binding,
            derived_binding=binding,derived_identity=self.new['rule_source'],diagnostic=dict(
                parent_checkpoint=dict(sha256='8'*64),original_job=fix_jobs[3])),numeric=dict(cpu_input={}))
        self.addCleanup(patch.stopall)
        patch.object(aux,'_oracle',self.oracle.api).start()
        self.report_verifier=patch.object(continuation,'verify_auxiliary_kernel_fix_report',return_value=self.evidence).start()
        self.snapshot_verifier=patch.object(continuation,'verify_auxiliary_kernel_fix_snapshot',return_value=self.new).start()

    def write(self,path,value):
        raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode();path.write_bytes(raw)
        return sha256(raw).hexdigest()

    def migrate(self,segment=None,**changes):
        args=dict(proof_deadline=time.monotonic()+30,original_checkpoint_sha256=self.original_checkpoint,
            old_ledger_path=self.ledger_path,old_ledger_sha256=self.ledger_sha,
            old_failed_report_path=self.report_path,old_failed_report_sha256=self.report_sha,
            fix_report_path=self.directory/'mock-never-a-real-gpu-proof.json',fix_report_sha256='f'*64,
            authorization='Explicit fixture correction; preserve rejected ledger, resume physical prefix')
        args.update(changes)
        return continuation.create_auxiliary_kernel_revalidation(self.old if segment is None else segment,**args)

    def test_rebind_clean_prefix_preserves_full_old_budget_and_files_roundtrip(self):
        before=continuation.auxiliary_content_sha256(self.old);ledger_bytes=self.ledger_path.read_bytes()
        migrated,lineage=self.migrate()
        self.assertEqual(continuation.auxiliary_content_sha256(self.old),before)
        self.assertEqual(self.ledger_path.read_bytes(),ledger_bytes)
        self.assertEqual(migrated.source_binding,self.new);self.assertEqual(migrated.budget_continuations,())
        self.assertFalse(lineage['gpu_replay_executed']);self.assertEqual(lineage['original_steps'],[7])
        event=migrated.attempts[-1]
        self.assertEqual(continuation.auxiliary_content_sha256(event['old_segment']),before)
        self.assertEqual(event['old_budget_continuations'][0].receipt_sha256,self.old.budget_continuations[0].receipt_sha256)
        self.assertTrue(aux.verify_auxiliary_segment(migrated))
        path=self.directory/'new-safe-checkpoint.json.gz';digest=continuation.dump_auxiliary_checkpoint(migrated,path)
        restored=continuation.load_auxiliary_checkpoint(path,expected_sha256=digest)
        self.assertEqual(continuation.auxiliary_content_sha256(restored),continuation.auxiliary_content_sha256(migrated))
        self.assertIs(restored.initial[0]['left'],restored.initial[0]['right'])
        self.assertTrue(aux.verify_auxiliary_segment(restored))

    def test_future_budget_natural_terminal_keeps_lineage_and_labels_only_then(self):
        migrated,_=self.migrate()
        budget=continuation.create_auxiliary_continuation(migrated,
            deadline=migrated.metadata.jobs[0]['deadline']+100,authorization='Explicit future scheduling budget',
            parent_report_sha256='a'*64,parent_checkpoint_sha256='b'*64)
        finished=fixtures.segment(budget.new_metadata,self.oracle,steps=8)
        finished=replace(finished,source_binding=self.new,budget_continuations=(budget,),attempts=migrated.attempts)
        self.assertTrue(aux.verify_auxiliary_segment(finished))
        games=aux.verified_auxiliary_games(finished)
        self.assertEqual(len(games),1);self.assertFalse(games[0]['source_lineage'][0]['gpu_replay_executed'])
        self.assertEqual(games[0]['source_lineage'][0]['old_source_binding'],self.old_binding)
        self.assertTrue(all(r['z']==1 if r['side']=='a'else r['z']==-1 for r in games[0]['value_rows']))

    def test_already_natural_games_are_revalidated_without_replaying_or_relabelling(self):
        complete=fixtures.segment(self.old.metadata,self.oracle,steps=8)
        self.old=replace(complete,source_binding=self.old_binding,budget_continuations=self.old.budget_continuations)
        self.ledger['cohorts'][0]['status']='COMPLETE'
        self.ledger['cohorts'][0]['checkpoints'][0]['content_sha256']=continuation.auxiliary_content_sha256(self.old)
        self.ledger_sha=self.write(self.ledger_path,self.ledger)
        self.report['ledger_manifest_sha256']=self.ledger_sha;self.report['windows'][0]['manifest_sha256']=self.ledger_sha
        self.report_sha=self.write(self.report_path,self.report)
        before=continuation.auxiliary_content_sha256(self.old)
        migrated,_=self.migrate()
        self.assertEqual(migrated.steps,(8,));self.assertEqual(migrated.complete,(True,))
        self.assertEqual(continuation.auxiliary_content_sha256(self.old),before)
        self.assertTrue(aux.verified_auxiliary_games(migrated)[0]['complete'])
        for old,new in zip(self.old.value_records,migrated.value_records):
            self.assertTrue((old['z']==new['z']).all());self.assertTrue((old['eligible']==new['eligible']).all())

    def test_failed_input_and_wrong_reports_sources_data_or_hidden_event_rejected(self):
        for bad in(replace(self.old,errors=(197,)),replace(self.old,errors=(256,)),
                   replace(self.old,truncated=(True,)),replace(self.old,export_errors=('bad',))):
            with self.assertRaises(ValueError):self.migrate(bad)
        for changes in(dict(authorization=' '),dict(old_ledger_sha256='0'*64),dict(original_checkpoint_sha256='e'*64)):
            with self.assertRaises(ValueError):self.migrate(**changes)
        migrated,_=self.migrate()
        with self.assertRaises(ValueError):aux.verify_auxiliary_segment(replace(migrated,source_binding=self.old_binding))
        corrupt=deepcopy(migrated);corrupt.value_records[0]['x'][0,0]+=1
        with self.assertRaises(ValueError):aux.verify_auxiliary_segment(corrupt)
        corrupt=deepcopy(migrated);corrupt.attempts[-1]['old_segment'].pcg_states[0]['state']['state']+=1
        with self.assertRaises(ValueError):aux.verify_auxiliary_segment(corrupt)
        corrupt=deepcopy(migrated);corrupt.attempts[-1]['kind']='ordinary_attempt'
        with self.assertRaises(ValueError):aux.verify_auxiliary_segment(corrupt)
        with self.assertRaises(ValueError):self.migrate(migrated)

    def test_old_final_boundary_is_actually_checked_in_current_single_cpu_replay(self):
        migrated,_=self.migrate();corrupt=deepcopy(migrated);event=corrupt.attempts[-1]
        # Simulate an otherwise self-consistent corrupted receipt: content and
        # ledger metadata are recomputed, so only actual prefix replay rejects.
        event['old_segment'].final[0]['rng']+=1
        event['old_segment_sha256']=continuation.auxiliary_content_sha256(event['old_segment'])
        event['old_ledger']['cohorts'][0]['checkpoints'][0]['content_sha256']=event['old_segment_sha256']
        event['event_sha256']=continuation._kernel_event_hash(event)
        finished=fixtures.segment(migrated.metadata,self.oracle,steps=8)
        finished=replace(finished,source_binding=self.new,attempts=corrupt.attempts)
        with self.assertRaises(ValueError):aux.verify_auxiliary_segment(finished)

    def test_production_derivation_gate_rejects_second_changed_unit(self):
        binding=deepcopy(self.evidence['report']['derived_binding'])
        self.assertEqual(continuation._preview_derived_binding(binding,arena._digest(binding)),self.new)
        binding['derived_units']['rules_008']='1'*64
        with self.assertRaises(ValueError):continuation._preview_derived_binding(binding,arena._digest(binding))

    def test_actual_report_reader_checks_files_and_rejects_ready_json_and_path_escape(self):
        path=self.directory/'ready.json';digest=self.write(path,dict(passed=True))
        with self.assertRaises(ValueError):ACTUAL_REPORT_VERIFIER(path,expected_sha256=digest,proof_deadline=time.monotonic()+10)
        roles={'frozen_program','v85_raw','v85_decoded','cpu_input','cpu_clean','cpu_expected_rule','original_parent_after'}
        report=dict(schema='auxiliary_preview_kernel_fix_v1',files={k:dict(path='unused',sha256='0'*64)for k in roles},
            implementation_files={k:dict(path='unused',sha256='0'*64)for k in continuation._kernel_implementation_hashes()},
            additional_raw_files=[dict(path='../outside.npz',sha256='0'*64)])
        digest=self.write(path,report)
        with self.assertRaises(ValueError):ACTUAL_REPORT_VERIFIER(path,expected_sha256=digest,proof_deadline=time.monotonic()+10)
        (self.directory/'corrupt.npz').write_bytes(b'actual-corrupt-file')
        report['additional_raw_files']=[dict(path='corrupt.npz',sha256='0'*64)]
        digest=self.write(path,report)
        with self.assertRaises(ValueError):ACTUAL_REPORT_VERIFIER(path,expected_sha256=digest,proof_deadline=time.monotonic()+10)
        self.assertEqual(continuation._diagnostic_raw_files(dict(stages=[dict(raw_npz=dict(path='actual.npz',sha256='a'*64))])),{'actual.npz':'a'*64})

    def test_corrective_ledger_records_lineage_without_touching_rejected_parent(self):
        from app.modules.card_game.rl.device_auxiliary_calibration import DeviceAuxiliaryCalibrationLedger
        migrated,_=self.migrate();ledger=DeviceAuxiliaryCalibrationLedger.__new__(DeviceAuxiliaryCalibrationLedger)
        ledger.directory=self.directory/'corrective';ledger.directory.mkdir();ledger._manifest_path=ledger.directory/'manifest.json'
        ledger.deadline=time.time()+60;ledger.contract=deepcopy(self.ledger['contract']);ledger.contract['source_binding']=self.new
        ledger.manifest=dict(cohorts=[],events=[]);ledger._assert_current=lambda:None;ledger._persist=lambda:None
        ledger._expected=lambda phase,offset,count:[{k:v for k,v in j.items()if k!='deadline'}for j in migrated.metadata.jobs]
        before=self.ledger_path.read_bytes()
        admitted=ledger.admit_phase_prefix('training',segment=migrated,parent_report_sha256=self.report_sha,
            parent_checkpoint_sha256=self.original_checkpoint,authorization='Explicit new corrective ledger')
        self.assertEqual(admitted['status'],'PAUSED');self.assertTrue(admitted['kernel_lineage'])
        self.assertEqual(self.ledger_path.read_bytes(),before)
        # New process reads ordinary manifest JSON, while checkpoint keeps its
        # lossless tuples/arrays. The public lineage must compare identically.
        admitted=json.loads(json.dumps(admitted))
        loaded=ledger._load_segment(admitted,deadline=ledger.deadline)
        self.assertEqual(loaded.source_binding,self.new)
        with self.assertRaises(ValueError):ledger.admit_phase_prefix('training',segment=migrated,parent_report_sha256=self.report_sha,
            parent_checkpoint_sha256=self.original_checkpoint,authorization='Cannot count original prefix twice')


if __name__=='__main__':unittest.main()
