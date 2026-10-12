"""Byte-exact masked updates retain untouched rows and replay masks."""
import unittest
from app.modules.card_game.rl.batched_search.device_masked_row_copy import prepare_masked_row_copy,copy_masked_rows_device

class MaskedRowCopyTest(unittest.TestCase):
    def test_actual_cuda_bytes_and_graph_replay(self):
        import torch as t
        if not t.cuda.is_available():self.skipTest('Actual CUDA required')
        module=prepare_masked_row_copy('cuda')
        for dtype in (t.int32,t.int64,t.float32,t.float64,t.bool):
            source=t.arange(3*7*5,device='cuda').reshape(3,7,5).to(dtype)
            initial=t.zeros_like(source);target=initial.clone();mask=t.tensor([True,False,True],device='cuda')
            if dtype in (t.float32,t.float64):source[0,0,0]=float('nan');source[2,0,0]=-0.
            graph=t.cuda.CUDAGraph()
            with t.cuda.graph(graph):
                target.copy_(initial);copy_masked_rows_device(source,target,mask,module)
            for pattern in ([True,False,True],[False,True,False],[False,False,False]):
                mask.copy_(t.tensor(pattern,device='cuda'));graph.replay();t.cuda.synchronize()
                expected=t.where(mask[:,None,None],source,initial)
                self.assertTrue(t.equal(target.reshape(-1).view(t.uint8),expected.reshape(-1).view(t.uint8)))
            copy_masked_rows_device(target,target,mask,module)
    def test_cpu_and_invalid_device_reject(self):
        import torch as t
        with self.assertRaises(ValueError):prepare_masked_row_copy('cpu')
        with self.assertRaises(ValueError):copy_masked_rows_device(t.zeros(2),t.zeros(2),t.ones(2,dtype=t.bool),None)

if __name__=='__main__':unittest.main()

class CompactMaskedCopyTest(unittest.TestCase):
    def test_actual_cuda_all_token_fields_masked_snapshot(self):
        import torch as t
        from dataclasses import fields
        from tests import test_duel_v2_compact_device_candidates as fixture
        from app.modules.card_game.rl.batched_search.compact_device_candidates import CompactDeviceCandidates
        if not t.cuda.is_available():self.skipTest('Actual CUDA required')
        raw=fixture._tokens(fixture._cases(),device='cuda');source=CompactDeviceCandidates(raw,device='cuda')
        target=source.clone();mask=t.arange(source.shape[0],device='cuda')%3==0
        before={f.name:getattr(target.tokens,f.name).clone() for f in fields(raw) if f.name!='request_ids'}
        for name in before:
            tensor=getattr(target.tokens,name)
            if tensor.dtype==t.bool:tensor.logical_not_()
            else:tensor.add_(3)
            before[name]=tensor.clone()
        graph=t.cuda.CUDAGraph()
        with t.cuda.graph(graph):errors=target.copy_masked_device(source,mask)
        graph.replay();t.cuda.synchronize();self.assertFalse(errors.any().item())
        for name,original in before.items():
            src=getattr(source.tokens,name);view=mask.view((-1,)+(1,)*(src.ndim-1))
            self.assertTrue(t.equal(getattr(target.tokens,name),t.where(view,src,original)))
