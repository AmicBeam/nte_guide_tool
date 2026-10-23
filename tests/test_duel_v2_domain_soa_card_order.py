"""Card registration order restores strict export for accepted numeric samples."""
from copy import deepcopy
import shutil
import unittest
import torch
import numpy as np
from tests.test_duel_v2_domain_soa_character_queries import state,pack
from tests import test_duel_v2_domain_soa_actions as action_fixtures
from app.modules.card_game.rl.batched_duel.domain_soa_card_order import DomainCardOrder,UNSUPPORTED_VIEW,DUPLICATE_ZONE
from app.modules.card_game.rl.batched_duel.domain_soa_sampler import cpu_oracle as sampler_oracle
from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch
from app.modules.card_game.rl.batched_duel.domain_soa_codec import decode_domain_states,encode_domain_states
from app.modules.card_game.rl.seven_sampler import determinize


def backend(L,device='cpu'):
    if device=='cpu'and not(shutil.which('clang++')or shutil.which('g++')):raise unittest.SkipTest('Explicit compiler oracle unavailable')
    return DomainCardOrder(L,device=device,allow_cpu_oracle=device=='cpu')


def permute(fields,perm):
    B,K=perm.shape;inverse=torch.empty_like(perm);inverse.scatter_(1,perm,torch.arange(K,device=perm.device)[None,:].expand(B,-1))
    out={k:v.clone()for k,v in fields.items()}
    for k,v in fields.items():
        if k.startswith('card_'):out[k]=v.gather(1,perm.reshape((B,K)+(1,)*(v.ndim-2)).expand_as(v))
    for k in ('zone_card_index','choice_card_index','op_card_index','op_choice_card_index'):
        v=fields[k];out[k]=inverse.gather(1,v.long().clamp(0,K-1).reshape(B,-1)).reshape_as(v).to(v.dtype)
        out[k]=torch.where(v>=0,out[k],v)
    out['card_physical_index']=torch.where(out['card_is_view'],inverse.gather(1,out['card_physical_index'].long().clamp(0,K-1)).int(),0)
    return out


class CardOrderTest(unittest.TestCase):
    def assert_columns_equal(self,left,right):
        for k in left:self.assertTrue(torch.equal(left[k],right[k]),k)

    def test_arbitrary_permutations_all_references_and_aliases(self):
        states=action_fixtures.ActionsTest().fixtures();L,fields=pack(states);B=len(states);K=L.max_cards
        perm=torch.arange(K).flip(0)[None,:].repeat(B,1);shuffled=permute(fields,perm)
        with backend(L)as order:
            p=order.prepare(shuffled,torch.ones(B,dtype=torch.bool));result=order.launch(p)
            self.assertFalse(result['errors'].any());self.assert_columns_equal(result['fields'],fields)
            self.assertFalse(result['full_canonical_codec_qualified'])

    def test_numeric_samples_strict_decode_and_full_formal_state(self):
        if not(shutil.which('clang++')or shutil.which('g++')):self.skipTest('Explicit compiler oracle unavailable')
        states=[state(20+i,a=k,b=k)for i,k in enumerate(('starter','weave-rush','quick-rush','zhenhong','murk','collapse','lingke'))]
        L,fields=pack(states);B=len(states);viewers=torch.arange(B)%2;seeds=torch.arange(B)+39
        with sampler_oracle(L,guard=False)as sampler,backend(L)as order:
            sampled=sampler.determinize(fields,viewers,seeds,torch.ones(B,dtype=torch.int32))
            result=order.launch(order.prepare(sampled,torch.ones(B,dtype=torch.bool)))
            self.assertFalse(result['errors'].any())
            batch=DomainSoABatch(layout=L,**{k:v.numpy()for k,v in result['fields'].items()});worlds=decode_domain_states(batch)
            for i,world in enumerate(worlds):
                gold=determinize(states[i],('a','b')[int(viewers[i])],int(seeds[i]));gold.pop('events',None)
                self.assertEqual(world,gold,str(i))
                expected=encode_domain_states([gold],L,presentation='drop')
                for k,v in result['fields'].items():self.assertTrue(np.array_equal(v[i].numpy(),getattr(expected,k)[0]),str((i,k)))

    def test_refusal_and_exact_rollback(self):
        L,f=pack([state(1),state(2),state(3)]);bad={k:v.clone()for k,v in f.items()}
        bad['card_is_view'][0,0]=True;bad['zone_card_index'][1,0,0,1]=bad['zone_card_index'][1,0,0,0]
        bad['row_error'][2]=64
        with backend(L)as order:
            ac=torch.ones(3,dtype=torch.bool);p=order.prepare(bad,ac);out=order.launch(p)
            self.assertTrue(int(out['errors'][0])&UNSUPPORTED_VIEW);self.assertTrue(int(out['errors'][1])&DUPLICATE_ZONE);self.assertEqual(int(out['errors'][2]),64)
            for k in bad:
                if k!='row_error':self.assertTrue(torch.equal(out['fields'][k],bad[k]),k)
            ac.zero_();out=order.launch(p);self.assertFalse(out['errors'].any());self.assert_columns_equal(out['fields'],bad)

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_graph_reorders_real_sampling_and_references(self):
        from app.modules.card_game.rl.batched_duel.domain_soa_sampler import DomainSoASampler
        base=action_fixtures.ActionsTest().fixtures();states=[base[i%len(base)]for i in range(400)];L,fields=pack(states,device='cuda');B=400;K=L.max_cards
        perm=torch.arange(K,device='cuda').flip(0)[None,:].repeat(B,1);shuffled=permute(fields,perm)
        with backend(L,'cuda')as order:
            ac=torch.ones(B,dtype=torch.bool,device='cuda');p=order.prepare(shuffled,ac);order.launch(p);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=order.launch(p)
            g.replay();torch.cuda.synchronize();self.assertFalse(out['errors'].any());self.assert_columns_equal(out['fields'],fields)
            shuffled['row_error'][0]=64;g.replay();torch.cuda.synchronize();self.assertEqual(int(out['errors'][0]),64)
            for k in fields:shuffled[k].copy_(fields[k])
            g.replay();torch.cuda.synchronize();self.assertFalse(out['errors'].any());self.assert_columns_equal(out['fields'],fields)
        states=[state(20+i%7)for i in range(B)];L,fields=pack(states,device='cuda');vi=torch.arange(B,device='cuda')%2;seeds=torch.arange(B,device='cuda')+39
        with DomainSoASampler(L,device='cuda',guard=False)as sampler,backend(L,'cuda')as order:
            sample=sampler.prepare(fields,vi,seeds,ac.int());sampler.launch(sample);p=order.prepare(sample.outputs,ac);order.launch(p);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):sampler.launch(sample);out=order.launch(p)
            g.replay();torch.cuda.synchronize();self.assertFalse(out['errors'].any())
            selected=list(range(14));batch=DomainSoABatch(layout=L,**{k:v[selected].cpu().numpy()for k,v in out['fields'].items()});worlds=decode_domain_states(batch)
            for i,world in enumerate(worlds):
                gold=determinize(states[i],('a','b')[i%2],i+39);gold.pop('events',None);self.assertEqual(world,gold)


class EffectiveViewOrderTest(unittest.TestCase):
    def fixtures(self):
        from app.modules.card_game.engine.duel_v2.state import effective_card
        states=[]
        for detached in (False,True):
            st=state(8,a='zhenhong',b='starter');team=st['sides']['a'];team['characters']['zhenhong']['awakened']=True
            raw=team['hand'][0];view=effective_card(st,'a',raw)
            st['operation']={'side':'a','actor':'zhenhong','card':view,'participants':[],
                'harmony':None,'first_bonus':0,'redeem_bonus':0,'record_candidate':False,'choice_cards':[view,raw]}
            st['phase']='choice';st['pending_choice']={'kind':'kit','side':'a','prompt':'选择','cards':[view]}
            if detached:team['hand'].remove(raw)
            states.append(st)
        return states

    def check(self,states,L,fields,device):
        B=len(states);K=L.max_cards;perm=torch.arange(K,device=device).flip(0)[None,:].repeat(B,1);shuffled=permute(fields,perm)
        with backend(L,device)as order:
            ac=torch.ones(B,dtype=torch.bool,device=device);p=order.prepare(shuffled,ac);out=order.launch(p)
            if device=='cuda':
                torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):out=order.launch(p)
                g.replay();torch.cuda.synchronize()
            self.assertFalse(out['errors'].any())
            for k in fields:self.assertTrue(torch.equal(out['fields'][k],fields[k]),k)
            chosen=list(range(min(B,2)));worlds=decode_domain_states(DomainSoABatch(layout=L,**{k:v[chosen].cpu().numpy()for k,v in out['fields'].items()}))
            for world in worlds:
                view=world['pending_choice']['cards'][0];self.assertIs(view,world['operation']['card']);self.assertIs(view,world['operation']['choice_cards'][0]);self.assertIs(view['_physical_card'],world['operation']['choice_cards'][1])
            # Corrupt only the physical link; source columns are preserved on failure.
            row=0;ref=int(shuffled['choice_card_index'][row,0]);shuffled['card_physical_index'][row,ref]=ref
            if device=='cuda':g.replay();torch.cuda.synchronize()
            else:out=order.launch(p)
            self.assertTrue(int(out['errors'][row])&UNSUPPORTED_VIEW)
            for k in shuffled:
                if k!='row_error':self.assertTrue(torch.equal(out['fields'][k][row],shuffled[k][row]),k)

    def test_views_register_before_detached_physical_and_preserve_shared_aliases(self):
        states=self.fixtures();L,fields=pack(states);self.check(states,L,fields,'cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_recursive_physical_registration_graph(self):
        base=self.fixtures();states=[base[i%2]for i in range(400)];L,fields=pack(states,device='cuda');self.check(states,L,fields,'cuda')
