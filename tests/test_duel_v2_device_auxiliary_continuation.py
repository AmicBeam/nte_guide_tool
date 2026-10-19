"""CPU continuation/checkpoint fixtures, never fresh CUDA execution evidence."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.rl import device_auxiliary_collection as aux
from app.modules.card_game.rl import device_auxiliary_continuation as budget
from app.modules.card_game.rl.seven_auxiliary_jobs import adapt_auxiliary_jobs
from tests import test_duel_v2_device_auxiliary_collection as fixtures


class AuxiliaryContinuationTests(unittest.TestCase):
    def setUp(self):
        self.oracle=fixtures.Oracle();meta=fixtures.metadata();jobs=deepcopy(list(meta.jobs))
        for job in jobs:job['deadline']=time.time()-100
        self.meta=adapt_auxiliary_jobs(jobs,starting_decks=meta.starting_decks)
        self.old=fixtures.segment(self.meta,self.oracle,steps=7)
        self.deadline=time.time()+600

    def receipt(self,old=None,**kwargs):
        args=dict(deadline=self.deadline,authorization='用户明确授权从现有32步继续至自然终局，使用新有界预算',
                  parent_report_sha256='a'*64,parent_checkpoint_sha256='b'*64)
        args.update(kwargs)
        return budget.create_auxiliary_continuation(self.old if old is None else old,**args)

    def validate(self,receipt,**kwargs):
        args=dict(resume=self.old,metadata=self.meta,deadline=self.deadline,source_binding=self.old.source_binding)
        args.update(kwargs);return budget.validate_auxiliary_continuation(receipt,**args)

    def test_expired_prefix_only_explicit_budget_and_no_mutation(self):
        before=budget.auxiliary_content_sha256(self.old);receipt=self.receipt()
        new=self.validate(receipt)
        self.assertEqual(new.jobs[0]['deadline'],self.deadline)
        self.assertEqual(budget.auxiliary_content_sha256(self.old),before)
        self.assertEqual(receipt.original_steps,(7,))
        collector=aux.DeviceAuxiliaryCollector.__new__(aux.DeviceAuxiliaryCollector)
        collector._used=False;collector.rows=1;collector._source_binding=self.old.source_binding
        collector._memory=lambda:(_ for _ in()).throw(MemoryError('passed budget preflight'))
        with self.assertRaises(TimeoutError):
            collector._run_segment(self.meta,deadline=self.deadline,resume=self.old)
        with self.assertRaisesRegex(MemoryError,'passed budget preflight'):
            collector._run_segment(self.meta,deadline=self.deadline,resume=self.old,continuation_budget=receipt)
        self.assertEqual(budget.auxiliary_content_sha256(self.old),before)

    def test_non_deadline_changes_source_and_receipt_tampering_rejected(self):
        receipt=self.receipt()
        changed=deepcopy(receipt.new_metadata);changed.jobs[0]['seed']+=1
        cases=[replace(receipt,new_metadata=changed),replace(receipt,authorization='changed'),
               replace(receipt,old_segment_sha256='f'*64),replace(receipt,segment_seconds=91)]
        for invalid in cases:
            with self.assertRaises(ValueError):self.validate(invalid)
        with self.assertRaises(ValueError):self.validate(receipt,source_binding=dict(self.old.source_binding,image='f'*64))
        with self.assertRaises(ValueError):self.validate(receipt,seconds=91)
        with self.assertRaises(ValueError):self.validate(receipt,soft_seconds=151)
        with self.assertRaises(ValueError):self.validate(receipt,deadline=self.deadline+1)
        changed=deepcopy(self.old);changed.raw_records[0]['raw_action']['index']+=1
        with self.assertRaises(ValueError):self.validate(receipt,resume=changed)

    def test_guard_error_truncation_and_previous_failures_never_receive_budget(self):
        for old in (replace(self.old,errors=(256,)),replace(self.old,truncated=(True,)),
                    replace(self.old,attempts=(dict(errors=(256,)),)),
                    replace(self.old,attempts=(dict(errors=(0,),memory_limit_exceeded=True),)),
                    replace(self.old,replay_verified=False)):
            with self.assertRaises(ValueError):self.receipt(old)
        for kwargs in (dict(authorization=' '),dict(parent_checkpoint_sha256='missing'),dict(deadline=1)):
            with self.assertRaises(ValueError):self.receipt(**kwargs)

    def test_actual_terminal_can_label_prior_prefix_but_cannot_change_observations(self):
        receipt=self.receipt();current=fixtures.segment(receipt.new_metadata,self.oracle,steps=8)
        current=replace(current,budget_continuations=(receipt,))
        with patch.object(aux,'_oracle',self.oracle.api):self.assertTrue(aux.verify_auxiliary_segment(current))
        corrupted=deepcopy(current);corrupted.value_records[0]['x'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'Historical x/context'):
            budget.verify_auxiliary_budget_history(corrupted)
        corrupted=deepcopy(current);corrupted.audit_records[0]['chosen']+=1
        with self.assertRaises(ValueError):budget.verify_auxiliary_budget_history(corrupted)
        disconnected=replace(current,metadata=self.meta)
        with self.assertRaises(ValueError):budget.verify_auxiliary_budget_history(disconnected)

    def test_two_budget_chain_and_checkpoint_roundtrip_exact_bits_alias_entity(self):
        from app.modules.card_game.engine.duel_v2.entities import CharacterEntity
        first=self.receipt()
        paused=replace(self.old,metadata=first.new_metadata,budget_continuations=(first,))
        second=self.receipt(paused,deadline=self.deadline+600)
        twice=replace(paused,metadata=second.new_metadata,budget_continuations=(first,second))
        self.assertTrue(budget.verify_auxiliary_budget_history(twice))
        with tempfile.TemporaryDirectory()as temp:
            path=Path(temp)/'checkpoint.json.gz';digest=budget.dump_auxiliary_checkpoint(twice,path)
            restored=budget.load_auxiliary_checkpoint(path,expected_sha256=digest)
            self.assertEqual(budget.auxiliary_content_sha256(twice),budget.auxiliary_content_sha256(restored))
            self.assertIs(restored.initial[0]['left'],restored.initial[0]['right'])
            with self.assertRaises(ValueError):budget.load_auxiliary_checkpoint(path,expected_sha256='0'*64)
        entity=CharacterEntity(id='fixture',entity_id='e1',side='a');array=np.array([0x80000000,0x7fc00123],np.uint32).view(np.float32)
        payload=dict(entity=entity,alias=entity,array=array,alias_array=array,tuple=(np.int64(3),))
        restored=budget.decode_auxiliary_payload(budget.encode_auxiliary_payload(payload))
        self.assertIs(restored['entity'],restored['alias']);self.assertIsInstance(restored['entity'],CharacterEntity)
        self.assertIs(restored['array'],restored['alias_array'])
        self.assertEqual(restored['array'].dtype,array.dtype)
        self.assertEqual(restored['array'].tobytes(),array.tobytes())
        self.assertIsInstance(restored['tuple'],tuple)
        self.assertEqual(budget.auxiliary_content_sha256(payload),budget.auxiliary_content_sha256(restored))
        distinct=dict(payload,alias=deepcopy(entity))
        self.assertNotEqual(budget.auxiliary_content_sha256(payload),budget.auxiliary_content_sha256(distinct))

    def test_safe_codec_rejects_unknown_types_corrupt_array_and_allocation_bombs(self):
        with self.assertRaises(TypeError):budget.encode_auxiliary_payload(object())
        payload=budget.encode_auxiliary_payload(np.zeros(2,np.float32));payload['nodes'][0]['shape']=[2**50]
        with self.assertRaises(ValueError):budget.decode_auxiliary_payload(payload)

    def test_flat_value_field_order_reproof_preserves_original_failed_archive(self):
        receipt=self.receipt()
        current=fixtures.segment(receipt.new_metadata,self.oracle,steps=8)
        records=tuple({k:row[k]for k in reversed(list(row))}for row in current.value_records)
        current=replace(current,value_records=records,budget_continuations=(receipt,),
                        replay_verified=False,reason='old verifier field-order failure')
        original=budget.auxiliary_content_sha256(current)
        with patch.object(aux,'_oracle',self.oracle.api):
            corrected,proof=budget.reverify_auxiliary_evidence(current,proof_deadline=time.monotonic()+20,
                original_checkpoint_sha256='e'*64,authorization='reverify actual archived GPU fields')
        self.assertTrue(corrected.replay_verified);self.assertTrue(proof['passed'])
        self.assertFalse(current.replay_verified)
        self.assertEqual(budget.auxiliary_content_sha256(current),original)
        changed=deepcopy(current);changed.value_records[0]['context'][0]+=1
        with self.assertRaises(ValueError):budget.verify_auxiliary_budget_history(changed)
        with self.assertRaises(ValueError):
            budget.reverify_auxiliary_evidence(replace(current,errors=(256,)),proof_deadline=time.monotonic()+20,
                original_checkpoint_sha256='e'*64,authorization='cannot wash out a real guard')
        payload=budget.encode_auxiliary_payload(np.zeros(2,np.float32));payload['nodes'][0]['dtype']='O'
        with self.assertRaises(ValueError):budget.decode_auxiliary_payload(payload)
        payload=budget.encode_auxiliary_payload(self.old);payload['nodes'][0]['type']='arbitrary.dynamic.Import'
        with self.assertRaises(ValueError):budget.decode_auxiliary_payload(payload)


    def test_single_encode_checkpoint_file_and_exact_content_identity(self):
        import gzip,json,hashlib
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'once.json.gz'
            original=budget.encode_auxiliary_payload
            with patch.object(budget,'encode_auxiliary_payload',wraps=original)as encode:
                file_sha,content_sha=budget.dump_auxiliary_checkpoint_with_content(self.old,path)
            self.assertEqual(encode.call_count,1)
            self.assertEqual(file_sha,hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(content_sha,budget.auxiliary_content_sha256(self.old))
            with gzip.open(path,'rb')as stream:
                self.assertEqual(content_sha,hashlib.sha256(stream.read()).hexdigest())
            restored=budget.load_auxiliary_checkpoint(path,expected_sha256=file_sha)
            self.assertEqual(budget.auxiliary_content_sha256(restored),content_sha)
            self.assertIsInstance(budget.dump_auxiliary_checkpoint(self.old,Path(directory)/'legacy.json.gz'),str)


if __name__=='__main__':unittest.main()
