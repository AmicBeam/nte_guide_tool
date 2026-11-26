"""Full ordered encodings and checked narrowing of candidate storage."""
from dataclasses import replace
import unittest
from tests.test_duel_v2_compact_device_candidates import _cases,_tokens
from app.modules.card_game.rl.batched_duel import gpu_seven_candidates as encoder
from app.modules.card_game.rl.batched_search.compact_device_candidates import CompactDeviceCandidates
from app.modules.card_game.rl.batched_search.device_int32_presentation import INDEX_FIELDS,narrow_candidate_indices

class Int32StorageTest(unittest.TestCase):
    def run_encoding_parity(self,device):
        import torch as t
        tokens=_tokens(_cases(),device=device);small,errors=narrow_candidate_indices(tokens,allow_cpu_reference=device=='cpu')
        self.assertFalse(errors.any().item())
        table=encoder.get_card_template_table(tokens.int_tokens.device)
        original=CompactDeviceCandidates(tokens,device=device,allow_cpu_reference=device=='cpu',template_table=table)
        narrowed=CompactDeviceCandidates(small,device=device,allow_cpu_reference=device=='cpu',template_table=table)
        for start in range(0,original.shape[0],3):
            stop=min(start+3,original.shape[0]);active=t.ones(stop-start,dtype=t.bool,device=device)
            left,le=original.encode_slice_device(start,stop,active);right,re=narrowed.encode_slice_device(start,stop,active)
            self.assertTrue(t.equal(le,re));self.assertTrue(t.equal(left.candidates,right.candidates));self.assertTrue(t.equal(left.mask,right.mask));self.assertTrue(t.equal(left.counts,right.counts))
        saved=sum(getattr(tokens,n).numel()*4 for n in INDEX_FIELDS if getattr(tokens,n).dtype==t.int64)
        self.assertEqual(original.allocated_bytes-narrowed.allocated_bytes,saved)
        self.assertEqual(narrowed.clone().allocated_bytes,narrowed.allocated_bytes)
        for n in INDEX_FIELDS:self.assertEqual(getattr(small,n).dtype,t.int32)
    def test_cpu_exact_encodings_and_saved_storage(self):self.run_encoding_parity('cpu')
    def test_actual_cuda_exact_encodings(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        self.run_encoding_parity('cuda')
    def test_overflow_is_rejected_before_use(self):
        import torch as t
        tokens=_tokens(_cases()[:2]);bad=tokens.int_tokens.to(t.int64);bad[0,0,0]=1<<40
        _,errors=narrow_candidate_indices(replace(tokens,int_tokens=bad),allow_cpu_reference=True)
        self.assertNotEqual(errors[0].item(),0);self.assertEqual(errors[1].item(),0)
        with self.assertRaises(ValueError):narrow_candidate_indices(tokens)

if __name__=='__main__':unittest.main()
