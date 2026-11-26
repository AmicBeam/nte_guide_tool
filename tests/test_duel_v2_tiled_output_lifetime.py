"""CPU ownership/reset contracts; CUDA snapshot parity is checked separately."""
from dataclasses import fields
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from app.modules.card_game.rl.batched_search import device_tiled_soa_rules as tiled
from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch


class TiledPersistentOutputTests(unittest.TestCase):
    def make_rules(self):
        store=object()
        from app.modules.card_game.rl.batched_duel.gpu_seven_candidates import get_card_template_table
        from app.modules.card_game.rl.batched_search import compact_device_candidates as compact
        actual=compact.CompactDeviceCandidates
        def cpu_compact(*args,**kwargs):
            return actual(*args,allow_cpu_reference=True,**kwargs)
        inner=SimpleNamespace(t=torch,device=torch.device('cpu'),templates=get_card_template_table('cpu'),layout=None,
                              workspace_bytes=0,close=lambda:None)
        with patch.object(compact,'CompactDeviceCandidates',side_effect=cpu_compact),patch.object(tiled,'DeviceSoARules',return_value=inner),patch.object(tiled.DeviceTiledSoARules,'_new_queue',return_value=None),patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
            rules=tiled.DeviceTiledSoARules(object(),store,object(),rows=3,work_rows=2,require_teacher=True)
        roots=DeviceStateBatch(store,torch.tensor([1,2,3]),torch.tensor([5,6,7]),torch.tensor([11,12,13]),
                              torch.tensor([True,False,True]),torch.tensor([0,8,0],dtype=torch.int32))
        return rules,roots

    def test_constructor_owns_unique_initialized_sites_before_any_capture(self):
        rules,_=self.make_rules();pointers=[]
        for kind,outputs in rules._output_sites.items():
            self.assertEqual(len(outputs),len(rules._sites[kind]))
            for output in outputs:
                for field in fields(output):
                    value=getattr(output,field.name)
                    if not isinstance(value,torch.Tensor)or not value.numel():continue
                    pointers.append(value.untyped_storage().data_ptr())
                    expected=-1 if field.name in('slots','action_slots','action_refs')or (kind=='rule_decision'and field.name=='action_types')else 0
                    self.assertTrue(torch.equal(value,torch.full_like(value,expected)))
        self.assertEqual(len(pointers),len(set(pointers)))
        self.assertGreater(rules.workspace_bytes,0)

    def test_capture_reset_borrows_existing_buffers_without_factories(self):
        rules,roots=self.make_rules()
        for kind in('metadata','value','sample','step','clean','teacher'):
            expected=rules._output_sites[kind][0]
            addresses={f.name:getattr(expected,f.name).data_ptr()for f in fields(expected)
                       if isinstance(getattr(expected,f.name),torch.Tensor)}
            for f in fields(expected):
                value=getattr(expected,f.name)
                if isinstance(value,torch.Tensor):value.fill_(True if value.dtype==torch.bool else 123)
            with patch.object(torch.cuda,'is_current_stream_capturing',return_value=True),patch.object(torch,'zeros',side_effect=AssertionError('capture allocation')),patch.object(torch,'full',side_effect=AssertionError('capture allocation')):
                _,output=rules._prepared_output(kind,roots)
            self.assertIs(output,expected)
            for f in fields(output):
                value=getattr(output,f.name)
                if not isinstance(value,torch.Tensor):continue
                self.assertEqual(value.data_ptr(),addresses[f.name])
                want=roots.errors if f.name=='errors'else roots.owners if f.name=='owners'else torch.full_like(value,-1 if f.name=='slots'else 0)
                torch.testing.assert_close(value,want,rtol=0,atol=0)
            # A second capture site is physically independent of the first.
            if len(rules._output_sites[kind])>1:
                with patch.object(torch.cuda,'is_current_stream_capturing',return_value=True):
                    _,second=rules._prepared_output(kind,roots)
                self.assertIsNot(second,output)

    def test_decisions_preserve_masks_counts_templates_and_dynamic_state_binding(self):
        rules,roots=self.make_rules()
        from app.modules.card_game.rl.batched_search.compact_device_candidates import CompactDeviceCandidates
        for kind in('decision','rule_decision'):
            prototype=rules._output_sites[kind][0]
            for field in fields(prototype):
                value=getattr(prototype,field.name)
                if isinstance(value,torch.Tensor):value.fill_(True if value.dtype==torch.bool else 27)
            if kind=='decision':
                for field in fields(prototype.candidates.tokens):
                    if field.name!='request_ids':getattr(prototype.candidates.tokens,field.name).fill_(1)
            with patch.object(torch.cuda,'is_current_stream_capturing',return_value=True),patch.object(torch,'zeros',side_effect=AssertionError('capture allocation')),patch.object(torch,'full',side_effect=AssertionError('capture allocation')):
                _,result=rules._prepared_output(kind,roots)
            self.assertIs(result.states,roots);self.assertIsNone(prototype.states)
            self.assertIs(result.candidates,prototype.candidates)
            self.assertIs(result.counts,prototype.counts);self.assertFalse(result.valid.any())
            self.assertEqual(result.counts.tolist(),[0,0,0]);self.assertEqual(result.action_slots.tolist(),[-1]*3)
            if kind=='decision':
                self.assertIsInstance(result.candidates,CompactDeviceCandidates)
                self.assertIs(result.counts,result.candidates.tokens.counts)
                self.assertIs(result.valid,result.candidates.tokens.mask)
                self.assertIs(result.candidates.template_table,rules.templates)
                self.assertEqual(result.candidates.request_ids,())
                for field in fields(result.candidates.tokens):
                    if field.name=='request_ids':continue
                    value=getattr(result.candidates.tokens,field.name)
                    expected=-1 if field.name in('int_tokens','mulligan_cards','order_cards','select_cards')else 0
                    torch.testing.assert_close(value,torch.full_like(value,expected),rtol=0,atol=0)
            else:
                self.assertEqual(rules._choice_sites[kind][0].tolist(),[-1]*3)
                self.assertEqual(result.candidates.shape,(3,800,0))

    def test_late_fused_site_preparation_and_capture_allocation_refusal(self):
        rules,_=self.make_rules();rules._sites['step_clean']=[None];rules._site_index['step_clean']=0
        with patch.object(torch.cuda,'is_current_stream_capturing',return_value=True):
            with self.assertRaises(RuntimeError):rules._prepare_output_sites('step_clean')
        self.assertNotIn('step_clean',rules._output_sites)
        with patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
            rules._prepare_output_sites('step_clean')
        self.assertEqual(len(rules._output_sites['step_clean']),1)
        with patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
            with self.assertRaises(RuntimeError):rules._prepare_output_sites('step_clean')


if __name__=='__main__':unittest.main()
