"""Checked candidate index widths; complete float encodings remain identical."""
from dataclasses import fields,replace
import unittest
import numpy as np
import torch

from app.modules.card_game.rl.batched_search.device_tiled_soa_rules import _candidate_indices,_empty_tokens
from tests.test_duel_v2_compact_device_candidates import _cases,_tokens,_compact


class CandidateIndexStorageTests(unittest.TestCase):
    def test_seven_teams_both_opening_phases_full_prefix_encoding_and_byte_savings(self):
        cases=_cases();tokens=_tokens(cases,capacity=40)
        narrow=replace(tokens,**{name:getattr(tokens,name).to(torch.int32) for name in
            ('int_tokens','mulligan_cards','order_cards','select_cards')})
        wide,small=_compact(tokens,scratch_rows=3),_compact(narrow,scratch_rows=3)
        self.assertEqual(small.tokens.counts.dtype,torch.int64)
        self.assertEqual(wide.allocated_bytes-small.allocated_bytes,len(cases)*40*47*4)
        for start,stop in wide._slices():
            active=torch.ones(stop-start,dtype=torch.bool)
            left,le=wide.encode_slice_device(start,stop,active)
            right,re=small.encode_slice_device(start,stop,active)
            self.assertEqual(le.tolist(),re.tolist());self.assertEqual(le.tolist(),[0]*(stop-start))
            np.testing.assert_array_equal(left.candidates.numpy(),right.candidates.numpy())
        prototype=_empty_tokens(torch,len(cases),40,'cpu',integer_dtype=torch.int32)
        for field in fields(tokens):
            if field.name!='request_ids':self.assertEqual(getattr(prototype,field.name).dtype,getattr(narrow,field.name).dtype)

    def test_narrowing_overflow_is_a_sticky_row_error_not_a_valid_wrapped_index(self):
        source=torch.tensor([[-1,0,2**31-1],[-2**31,3,17],[2**40,0,1]],dtype=torch.int64)
        converted,errors=_candidate_indices(torch,source,torch.int32,torch.tensor([True,True,True]))
        self.assertEqual(errors.tolist(),[0,0,156])
        self.assertTrue(torch.equal(converted[:2].long(),source[:2]))
        _,inactive=_candidate_indices(torch,source,torch.int32,torch.tensor([True,True,False]))
        self.assertEqual(inactive.tolist(),[0,0,0])
        with self.assertRaises(ValueError):_candidate_indices(torch,source.float(),torch.int32,torch.ones(3,dtype=torch.bool))


if __name__=='__main__':unittest.main()
