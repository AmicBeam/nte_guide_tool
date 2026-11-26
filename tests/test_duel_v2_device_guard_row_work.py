"""CPU routing contracts use an explicit mock queue; production remains CUDA-only."""
from dataclasses import fields
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from app.modules.card_game.rl.batched_duel.gpu_seven_candidates import RawCandidateTokens
from app.modules.card_game.rl.batched_search import device_guard_row_work as module


class _CpuTestQueue:
    def __init__(self,*,rows,scratch_rows,seconds,device):
        self.rows=rows;self.tile=scratch_rows;self.closed=False;self.calls=0;self.fail=False
    def capture_device(self,active,operation):
        import torch as t
        for start in range(0,self.rows,self.tile):
            if self.fail and start:self.calls+=1;break
            raw=t.arange(start,start+self.tile);inside=raw<self.rows
            ids=raw.clamp(0,self.rows-1)
            operation(ids,inside&active.index_select(0,ids),t.where(inside,raw,self.rows))
            self.calls+=1
        return t.tensor([int(self.fail)],dtype=t.int32)
    def close(self):self.closed=True


def _owner(rows,comparison):
    # Deliberately bypass the CUDA-only public constructor for logical CPU tests.
    import torch as t
    owner=object.__new__(module.DeviceGuardRowWork)
    owner.t=t;owner.device=t.device('cpu');owner.rows=rows;owner.tile=8;owner.seconds=10
    owner.comparison=comparison;owner.sites=[];owner.index=0;owner.closed=False
    with patch.object(module,'DeviceTensorRowQueue',_CpuTestQueue):
        for _ in range(2):owner.sites.append(module._GuardSite(owner))
    return owner


def _view(rows):
    import torch as t
    data={}
    for f in fields(RawCandidateTokens):
        if f.name!='request_ids':data[f.name]=t.arange(rows,dtype=t.int64)[:,None].expand(rows,800).clone()
    data['counts']=t.arange(rows,dtype=t.int64)
    data['turn_counts']=data['counts'].clone();data['mask']=data['mask'].bool()
    return SimpleNamespace(candidates=RawCandidateTokens(**data),errors=t.zeros(rows,dtype=t.int32),
        observations=SimpleNamespace(observations=t.arange(rows,dtype=t.float32)[:,None]))


class GuardRowWorkTest(unittest.TestCase):
    def test_cuda_only_constructor_and_dimensions(self):
        fn=lambda *args:None
        for kwargs in (dict(rows=0),dict(rows=3,tile=True),dict(rows=3,seconds=float('nan')),
                       dict(rows=3,seconds=0),dict(rows=3,compare=None)):
            values=dict(device='cpu',compare=fn);values.update(kwargs)
            with self.assertRaises((ValueError,TypeError)):module.DeviceGuardRowWork(**values)
        with self.assertRaises(ValueError):module.DeviceGuardRowWork(rows=3,device='cpu',compare=fn)

    def test_all_fields_tail_two_sites_and_error_guard(self):
        import torch as t
        for rows in (64,65):
            left=_view(rows);right=_view(rows);seen=[]
            def compare(a,b,on,table):
                seen.append(int(on.sum()))
                self.assertEqual(a.candidates.request_ids,())
                expected=a.observations.observations[:,0].to(t.int64)
                for f in fields(RawCandidateTokens):
                    if f.name=='request_ids':continue
                    v=getattr(a.candidates,f.name)
                    reference=left.candidates.__dict__[f.name].index_select(0,expected)
                    self.assertTrue(t.equal(v,reference),f.name)
                self.assertTrue(t.equal(a.errors,left.errors.index_select(0,expected)))
                return t.where(on,b.errors,0).to(t.int32)
            owner=_owner(rows,compare);active=t.arange(rows)%3==0
            right.errors.copy_(t.arange(rows,dtype=t.int32)+1)
            with patch('torch.cuda.is_current_stream_capturing',return_value=True):
                first=owner.compare(left,right,active,None)
                second=owner.compare(left,left,active,None)
                self.assertTrue(t.equal(first,t.where(active,right.errors,0)))
                self.assertFalse(second.any());self.assertLessEqual(max(seen),8)
                with self.assertRaises(RuntimeError):owner.compare(left,right,active,None)
            queues=[s.queue for s in owner.sites];owner.close();owner.close()
            self.assertTrue(all(q.closed for q in queues))
            with self.assertRaises(RuntimeError):owner.compare(left,right,active,None)
            # Simulate an interrupted queue: all unfinished active rows fail closed,
            # while the callback's earlier error remains intact.
            owner=_owner(rows,compare);owner.sites[0].queue.fail=True;active.fill_(True)
            with patch('torch.cuda.is_current_stream_capturing',return_value=True):
                out=owner.compare(left,right,active,None)
            self.assertTrue(t.equal(out[:8],right.errors[:8]))
            self.assertTrue(t.equal(out[8:],t.full((rows-8,),158,dtype=t.int32)))
            owner.close()

    def test_invalid_cohort_does_not_consume_sites(self):
        import torch as t
        owner=_owner(3,lambda *args:t.zeros(8,dtype=t.int32));view=_view(3)
        with self.assertRaises(ValueError):owner.compare(view,view,t.zeros(2,dtype=t.bool),None)
        view.candidates.extra_ap=t.zeros((2,800))
        with self.assertRaises(ValueError):owner.compare(view,view,t.ones(3,dtype=t.bool),None)
        self.assertEqual(owner.index,0);owner.close()

    def test_partial_site_allocation_closes_native_queue(self):
        import torch as t
        queues=[]
        def factory(**kwargs):
            q=_CpuTestQueue(**kwargs);queues.append(q);return q
        real_zeros=t.zeros
        def allocate(*args,**kwargs):
            if len(queues)==2:raise RuntimeError('allocation failure')
            kwargs['device']='cpu';return real_zeros(*args,**kwargs)
        with patch.object(module,'DeviceTensorRowQueue',side_effect=factory),\
             patch('torch.cuda.is_available',return_value=True),\
             patch('torch.cuda.is_current_stream_capturing',return_value=False),\
             patch('torch.zeros',side_effect=allocate):
            with self.assertRaisesRegex(RuntimeError,'allocation failure'):
                module.DeviceGuardRowWork(rows=3,device='cuda:0',compare=lambda *args:None)
        self.assertEqual(len(queues),2);self.assertTrue(all(q.closed for q in queues))


if __name__=='__main__':unittest.main()
