"""Physical draw compared to formal state.draw, not prepared host-rule masks."""
from copy import deepcopy
import shutil
import unittest
import torch
from app.modules.card_game.engine.duel_v2.state import draw,card_instance
from app.modules.card_game.rl.batched_duel.domain_soa_codec import _decode_row
from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch
from app.modules.card_game.rl.batched_duel.domain_soa_physical import DomainPhysicalCore,TYPES,CAPACITY,INVALID
from tests.test_duel_v2_domain_soa_character_queries import state,pack


def backend(L,device='cpu'):
    if device=='cpu'and not(shutil.which('clang++')or shutil.which('g++')):raise unittest.SkipTest('Explicit compiler oracle unavailable')
    return DomainPhysicalCore(L,device=device,allow_cpu_oracle=device=='cpu')


class PhysicalDrawTest(unittest.TestCase):
    def fixtures(self):
        cases=[]
        for key in ('starter','weave-rush','quick-rush','zhenhong','murk','collapse','lingke'):
            for side in (0,1):
                for filter in (-1,0,1,2):cases.append((state(70,a=key,b=key),side,3,filter))
        for mode in ('awake','dead','mimic','empty_mimic','foreign_mimic','full','nine','copy_overflow'):
            st=state(71,a='quick-rush');team=st['sides']['a'];h=team['characters']['xiaozhi'];h['awakened']=True
            if mode=='dead':h['hp']=0
            if mode=='mimic':h['flags']['mimic']={'id':'zero'}
            if mode=='empty_mimic':h['flags']['mimic']={'id':''}
            if mode=='foreign_mimic':h['awakened']=False;team['characters']['zero']['flags']['mimic']={'id':'xiaozhi'};team['characters']['zero']['awakened']=True
            if mode in ('full','nine','copy_overflow'):
                while len(team['hand'])<(9 if mode=='nine'else 10):team['hand'].append(team['deck'].pop(0))
            if mode=='copy_overflow':
                card=card_instance(st,'Q01','a');card.update(copy=True,expires_turn=9,jingu_mark={'delta':1,'source_entity_id':h['entity_id']});team['deck'].insert(0,card)
            cases.append((st,0,3,-1))
        for wins in (False,True):
            for filter in(-1,0):
                st=state(72);team=st['sides']['a'];team['discard'].extend(team['deck']);team['deck']=[]
                if wins:team['empty_draw_wins']=True
                cases.append((st,0,3,filter))
        st=state(73);st.update(phase='finished',winner='b',reason='player_hp');cases.append((st,0,2,-1));cases.append((state(74),0,0,-1))
        return cases

    def compare(self,cases,L,out):
        self.assertFalse(out['errors'].any(),out['errors'].cpu().tolist())
        batch=DomainSoABatch(layout=L,**{k:v.cpu().numpy()for k,v in out['fields'].items()})
        for b,(st,side,count,filter)in enumerate(cases):
            gold=deepcopy(st);draw(gold,('a','b')[side],count,card_type=None if filter<0 else TYPES[filter]);gold.pop('events',None);gold.pop('_public_board',None)
            got=_decode_row(batch,b);self.assertEqual(got,gold,str((b,side,count,filter)))
        self.assertFalse(out['full_duel_qualified']);self.assertFalse(out['full_canonical_codec_qualified'])

    def test_all_seven_hooks_overflow_filtered_draw_and_terminal(self):
        cases=self.fixtures();L,f=pack([c[0]for c in cases]);B=len(cases)
        with backend(L)as core:
            p=core.prepare_draw(f,torch.tensor([c[1]for c in cases],dtype=torch.int32),torch.tensor([c[2]for c in cases],dtype=torch.int32),torch.tensor([c[3]for c in cases],dtype=torch.int32),torch.ones(B,dtype=torch.bool))
            self.compare(cases,L,core.launch_draw(p))

    def test_errors_roll_back_all_columns_and_replay_refreshes(self):
        cases=[(state(91),0,2,-1),(state(92),0,1,-1),(state(93),1,1,-1)];L,f=pack([c[0]for c in cases]);si=torch.tensor([0,0,1],dtype=torch.int32);co=torch.tensor([2,1,1],dtype=torch.int32);ty=torch.full((3,),-1,dtype=torch.int32);ac=torch.ones(3,dtype=torch.bool)
        with backend(L)as core:
            p=core.prepare_draw(f,si,co,ty,ac);f['row_error'][0]=64;si[1]=8;ac[2]=False
            out=core.launch_draw(p);self.assertEqual(out['errors'].tolist(),[64,INVALID,0]);self.assertFalse(out['drawn'].any())
            for k in f:
                if k!='row_error':self.assertTrue(torch.equal(out['fields'][k],f[k]),k)
            f['row_error'].zero_();si[1]=0;ac[2]=True;self.compare(cases,L,core.launch_draw(p))
            f['catalog_digest'][0,0]^=1;out=core.launch_draw(p);self.assertEqual(int(out['errors'][0]),INVALID)
            self.assertTrue(torch.equal(out['fields']['zone_card_index'][0],f['zone_card_index'][0]))

    def test_late_capacity_failure_restores_mark_rng_and_event_counter(self):
        st=state(94,a='quick-rush');team=st['sides']['a'];h=team['characters']['xiaozhi'];h['awakened']=True
        while len(team['hand'])<9:team['hand'].append(team['deck'].pop(0))
        while len(team['discard'])<96:team['discard'].append(card_instance(st,'Q01','a'))
        L,f=pack([st])
        with backend(L)as core:
            p=core.prepare_draw(f,torch.zeros(1,dtype=torch.int32),torch.tensor([2],dtype=torch.int32),torch.tensor([-1],dtype=torch.int32),torch.ones(1,dtype=torch.bool));out=core.launch_draw(p)
            self.assertEqual(int(out['errors'][0]),CAPACITY);self.assertEqual(int(out['drawn'][0]),0)
            for k in f:
                if k!='row_error':self.assertTrue(torch.equal(out['fields'][k],f[k]),k)

    def test_duplicate_zone_and_invalid_reference_are_closed_errors(self):
        L,f=pack([state(95),state(96)]);f['zone_card_index'][0,0,1,0]=f['zone_card_index'][0,0,0,0];f['zone_card_index'][1,0,1,0]=L.max_cards
        with backend(L)as core:
            p=core.prepare_draw(f,torch.zeros(2,dtype=torch.int32),torch.ones(2,dtype=torch.int32),torch.full((2,),-1,dtype=torch.int32),torch.ones(2,dtype=torch.bool));out=core.launch_draw(p)
            self.assertEqual(out['errors'].tolist(),[INVALID,INVALID])
            for k in f:
                if k!='row_error':self.assertTrue(torch.equal(out['fields'][k],f[k]),k)

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required; host oracle is not GPU qualification')
    def test_cuda_400_draw_capture_and_query_refresh(self):
        base=self.fixtures();cases=[base[i%len(base)]for i in range(400)];L,f=pack([c[0]for c in cases],device='cuda');si=torch.tensor([c[1]for c in cases],dtype=torch.int32,device='cuda');co=torch.tensor([c[2]for c in cases],dtype=torch.int32,device='cuda');ty=torch.tensor([c[3]for c in cases],dtype=torch.int32,device='cuda');ac=torch.ones(400,dtype=torch.bool,device='cuda')
        with backend(L,'cuda')as core:
            p=core.prepare_draw(f,si,co,ty,ac);core.launch_draw(p);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=core.launch_draw(p)
            g.replay();torch.cuda.synchronize();self.compare(cases[:len(base)],L,{**out,'fields':{k:v[:len(base)]for k,v in out['fields'].items()},'errors':out['errors'][:len(base)]})
            f['row_error'][0]=64;co[1]=0;ac[2]=False;g.replay();torch.cuda.synchronize();self.assertEqual(int(out['errors'][0]),64)
            for b in(0,1,2):
                for k in f:
                    if k!='row_error':self.assertTrue(torch.equal(out['fields'][k][b],f[k][b]),str((b,k)))
