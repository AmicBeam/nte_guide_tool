"""Actual input equality, including late candidate windows and captured reuse."""
from copy import deepcopy
import shutil
import unittest
import torch
from app.modules.card_game.engine.duel_v2.flow import legal_actions
from app.modules.card_game.engine.duel_v2.projection import observe
from app.modules.card_game.rl import seven_lineup as seven
from app.modules.card_game.rl.batched_duel.domain_soa_information_guard import DomainInformationGuard, INFORMATION_CHANGED
from app.modules.card_game.rl.batched_duel.domain_soa_sampler import cpu_oracle as sampler_oracle
from app.modules.card_game.rl.batched_duel.domain_soa_schema import CARD_INT_FIELDS
from tests.test_duel_v2_domain_soa_character_queries import state,pack


def backend(layout,device='cpu',**kw):
    if device=='cpu'and not(shutil.which('clang++')or shutil.which('g++')):raise unittest.SkipTest('Explicit compiler oracle unavailable')
    return DomainInformationGuard(layout,device=device,allow_cpu_oracle=device=='cpu',**kw)


class InformationGuardTest(unittest.TestCase):
    def fixtures(self):
        states=[state(20+i,a=key,b=key)for i,key in enumerate(('starter','weave-rush','quick-rush','zhenhong','murk','collapse','lingke'))]
        for st in states:st['active_side']='a';st['sides']['a']['ap']=100
        return states

    def test_numeric_samples_all_seven_and_mutation_rollback_boundary(self):
        states=self.fixtures();layout,fields=pack(states);B=len(states);viewers=torch.zeros(B,dtype=torch.int64);active=torch.ones(B,dtype=torch.bool)
        if not(shutil.which('clang++')or shutil.which('g++')):self.skipTest('Explicit compiler oracle unavailable')
        with sampler_oracle(layout,guard=False)as sampler,backend(layout,max_actions=64)as guard:
            sampled=sampler.determinize(fields,viewers,torch.arange(B,dtype=torch.int64)+39,active.int())
            self.assertFalse(sampled['row_error'].any())
            p=guard.prepare(fields,sampled,viewers,active);out=guard.launch(p)
            self.assertEqual(out['errors'].tolist(),[0]*B);self.assertTrue(out['accepted'].all())
            sampled['player_hp'][0,0]-=1
            out=guard.launch(p);self.assertEqual(int(out['errors'][0]),INFORMATION_CHANGED)
            self.assertFalse(out['accepted'][0]);self.assertTrue(out['accepted'][1:].all())
            active[0]=False;out=guard.launch(p);self.assertEqual(int(out['errors'][0]),0);self.assertFalse(out['accepted'][0])
            self.assertFalse(out['sampler_qualified']);self.assertFalse(out['full_duel_qualified'])
            self.assertFalse(guard.gpu_qualified)

    def test_full_800_capacity_and_candidate_only_late_window(self):
        st=state(110);st['active_side']='a';st['sides']['a'].update(ap=100,normal_attack_available=True)
        # Own-hand surcharge is encoded ONLY in candidates, not the 3479 observation.
        original=deepcopy(st);changed=deepcopy(st)
        plays=[a for a in legal_actions(original,'a')if a['type']=='play_card']
        self.assertTrue(plays)
        card=next(c for c in changed['sides']['a']['hand']if c['instance_id']==plays[-1]['card_id']);card['additional_ap_cost']=1
        actions=[a for a in legal_actions(original,'a')if a['type']!='concede']
        x,c=seven.encode(observe(original,'a',include_previews=False),actions)
        xx,cc=seven.encode(observe(changed,'a',include_previews=False),actions)
        self.assertTrue(torch.equal(torch.tensor(x),torch.tensor(xx)))
        differing=(torch.tensor(c)!=torch.tensor(cc)).any(-1).nonzero().flatten().tolist()
        self.assertTrue(differing);self.assertGreaterEqual(max(differing),4)
        layout,before=pack([original]);_,after=pack([changed],layout=layout)
        with backend(layout,max_actions=800,window=4)as guard:
            out=guard.launch(guard.prepare(before,after,torch.zeros(1,dtype=torch.int64),torch.ones(1,dtype=torch.bool)))
            self.assertEqual(int(out['errors'][0]),INFORMATION_CHANGED);self.assertFalse(out['accepted'][0])

    def test_errors_replay_and_no_pruning_on_capacity_failure(self):
        states=self.fixtures()[:2];layout,before=pack(states);_,after=pack(states,layout=layout)
        with backend(layout,max_actions=1,window=1)as guard:
            p=guard.prepare(before,after,torch.zeros(2,dtype=torch.int64),torch.ones(2,dtype=torch.bool));out=guard.launch(p)
            self.assertTrue((out['errors']!=0).all());self.assertFalse(out['accepted'].any())
        with backend(layout,max_actions=64)as guard:
            active=torch.ones(2,dtype=torch.bool);p=guard.prepare(before,after,torch.zeros(2,dtype=torch.int64),active)
            after['row_error'][0]=64;after['catalog_digest'][1,0]^=1;out=guard.launch(p)
            self.assertEqual(out['errors'].tolist(),[64,1]);self.assertFalse(out['accepted'].any())
            after['row_error'].zero_();after['catalog_digest'][1,0]^=1;out=guard.launch(p)
            self.assertTrue(out['accepted'].all())

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_captured_complete_guard(self):
        states=[self.fixtures()[i%7]for i in range(400)];layout,before=pack(states,device='cuda');_,after=pack(states,layout=layout,device='cuda')
        vi=torch.zeros(400,dtype=torch.int64,device='cuda');active=torch.ones(400,dtype=torch.bool,device='cuda')
        with backend(layout,'cuda',max_actions=64)as guard:
            p=guard.prepare(before,after,vi,active);guard.launch(p);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=guard.launch(p)
            g.replay();torch.cuda.synchronize();self.assertTrue(out['accepted'].all())
            after['player_hp'][0,0]-=1;after['row_error'][1]=64;after['catalog_digest'][2,0]^=1
            plays=[a for a in legal_actions(states[3],'a')if a['type']=='play_card'];self.assertTrue(plays)
            instance=int(plays[-1]['card_id'].split('-')[-1]);ref=int(((after['card_instance_slot'][3]==instance)&(after['card_side'][3]==0)).nonzero()[0]);key=CARD_INT_FIELDS.index('additional_ap_cost')
            after['card_int_presence'][3,ref,key]=2;after['card_int_value'][3,ref,key]=1
            g.replay();torch.cuda.synchronize();self.assertEqual(out['errors'][:4].cpu().tolist(),[INFORMATION_CHANGED,64,1,INFORMATION_CHANGED]);self.assertFalse(out['accepted'][:4].any());self.assertTrue(out['accepted'][4:].all())
            for k in after:after[k].copy_(before[k])
            vi.fill_(1);g.replay();torch.cuda.synchronize();self.assertTrue(out['accepted'].all())

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_numeric_sampler_then_guard_in_same_graph(self):
        from app.modules.card_game.rl.batched_duel.domain_soa_sampler import DomainSoASampler
        from app.modules.card_game.rl.seven_sampler import determinize as formal_sample
        from app.modules.card_game.rl.batched_duel.domain_soa_schema import CARD_VOCAB
        base=self.fixtures();states=[base[i%7]for i in range(400)]
        layout,before=pack(states,device='cuda');vi=torch.arange(400,device='cuda',dtype=torch.int64)%2
        seeds=torch.arange(400,device='cuda',dtype=torch.int64)+91;active=torch.ones(400,dtype=torch.bool,device='cuda')
        with DomainSoASampler(layout,device='cuda',guard=False)as sampler,backend(layout,'cuda',max_actions=64)as guard:
            sample=sampler.prepare(before,vi,seeds,active.int());sampler.launch(sample)
            check=guard.prepare(before,sample.outputs,vi,active);guard.launch(check);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                sampled=sampler.launch(sample);out=guard.launch(check)
            g.replay();torch.cuda.synchronize()
            self.assertFalse(sampled['row_error'].any());self.assertTrue(out['accepted'].all());self.assertFalse(out['sampler_qualified'])
            inverse={v:k for k,v in CARD_VOCAB.items()}
            def verify():
                for b in range(14):
                    v=int(vi[b]);gold=formal_sample(states[b],('a','b')[v],int(seeds[b]))
                    self.assertEqual(int(sampled['rng'][b]),gold['rng'])
                    self.assertEqual(int(sampled['next_entity'][b]),gold['next_entity'])
                    self.assertEqual(int(sampled['next_instance'][b]),gold['next_instance'])
                    for side,zone in ((v,1),(1-v,0),(1-v,1)):
                        n=int(sampled['zone_count'][b,side,zone]);refs=sampled['zone_card_index'][b,side,zone,:n].long()
                        ids=[inverse[int(k)]for k in sampled['card_template'][b,refs].cpu()]
                        self.assertEqual(ids,[c['card_id']for c in gold['sides'][('a','b')[side]][('hand','deck')[zone]]])
            verify();seeds.add_(1000);vi.copy_(1-vi);g.replay();torch.cuda.synchronize()
            self.assertTrue(out['accepted'].all());self.assertFalse(sampled['row_error'].any());verify()
