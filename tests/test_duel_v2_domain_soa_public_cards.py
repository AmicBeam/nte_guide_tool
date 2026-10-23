"""Raw named state to public numeric candidates; no host runtime rule calls."""
from copy import deepcopy
import shutil
import unittest
import torch
from app.modules.card_game.engine.duel_v2.projection import observe
from app.modules.card_game.engine.duel_v2.flow import legal_actions
from app.modules.card_game.rl.batched_duel.domain_soa_public_cards import DomainPublicCards
from app.modules.card_game.rl.batched_duel.domain_soa_candidates import DomainSoACandidates
from tests.test_duel_v2_domain_soa_candidates import numeric
from tests.test_duel_v2_domain_soa_character_queries import state,pack
from app.modules.card_game.rl import seven_lineup as seven


def backend(layout,device='cpu'):
    if device=='cpu'and not(shutil.which('clang++')or shutil.which('g++')):raise unittest.SkipTest('Explicit compiler oracle unavailable')
    return DomainPublicCards(layout,device=device,allow_cpu_oracle=device=='cpu')


class PublicCardsTest(unittest.TestCase):
    def compare(self,states,viewers,out):
        self.assertEqual(out['errors'].cpu().tolist(),[0]*len(states))
        for b,(st,v)in enumerate(zip(states,viewers)):
            view=observe(st,('a','b')[v],include_previews=False)
            expected=numeric([view],[[]])['cards'][0]
            n=len(view['sides'][('a','b')[v]]['hand']);m=len((view.get('pending_choice')or{}).get('choices',[]))
            # Network presentation sorts a hand; table references preserve raw
            # zone order. Compare through physical instance identity instead.
            cards={c['instance_id']:i for i,c in enumerate(view['sides'][('a','b')[v]]['hand'])}
            actual=[];gold=[]
            for j,c in enumerate(st['sides'][('a','b')[v]]['hand']):
                actual.append(out['cards'][b,int(out['hand_refs'][b,j])].cpu());gold.append(expected[cards[c['instance_id']]])
            for j in range(m):
                actual.append(out['cards'][b,int(out['choice_refs'][b,j])].cpu());gold.append(expected[n+j])
            if actual:self.assertTrue(torch.equal(torch.stack(actual),torch.stack(gold)),str((b,[(i,seven.CARD_IDS[int(a[0])],[(k,float(x),float(y))for k,x,y in zip(out['card_fields'],a,g)if x!=y])for i,(a,g)in enumerate(zip(actual,gold))if not torch.equal(a,g)])))
            self.assertEqual(int(out['valid'][b].sum()),n+m)
            self.assertFalse(out['valid'][b][out['cards'][b,:,0]<0].any())
        self.assertFalse(out['full_duel_qualified']);self.assertFalse(out['legal_actions_available']);self.assertNotIn('_hand_rules',out)

    def test_initial_natural_and_both_views(self):
        from tests.test_duel_v2_domain_soa_codec import _natural_states,_fourteen_initial_states
        states=[st for st in _fourteen_initial_states()+_natural_states()[::11]for _ in range(2)]
        viewers=[i%2 for i in range(len(states))];layout,fields=pack(states)
        with backend(layout)as enc:self.compare(states,viewers,enc.project(fields,torch.tensor(viewers),torch.ones(len(states),dtype=torch.bool)))

    def test_substitution_instant_scopes_and_raw_choices(self):
        from app.modules.card_game.engine.duel_v2.modifiers import add_effect
        states=[]
        st=state(30,a='zhenhong');st['sides']['a']['characters']['zhenhong']['awakened']=True;states.append(st)
        st=state(31,a='lingke');st['sides']['a']['characters']['lingke']['shape']='K03';st['sides']['a']['ap']=0
        from app.modules.card_game.engine.duel_v2.state import card_instance
        st['sides']['a']['hand']=[card_instance(st,c,'a')for c in ('K01','K02','K03')];states.append(st)
        st=state(32);team=st['sides']['a'];team['characters']['zero']['awakened']=True;states.append(st)
        st=state(33,a='lingke');team=st['sides']['a'];h=team['characters']['lingke']
        add_effect(team,'lingke.next_instant','card.instant',True,op='allow',scope='allies',source_entity_id=h['entity_id'],source_key='K02',clear_on=(),conditions=[{'id':'card_owner_other'}]);states.append(st)
        for st in states:
            st['sides']['a']['hand'][0].update(copy=True,expires_turn=5,additional_ap_cost=2,jingu_mark={'delta':-1,'source_entity_id':st['sides']['a']['characters'][st['sides']['a']['order'][0]]['entity_id']})
            team=st['sides']['a'];team['revealed_ids']=[team['hand'][0]['instance_id']]
            st['pending_choice']={'kind':'hand_redraw','side':'a','prompt':'选择','cards':team['hand'][:2]};st['phase']='choice'
        states=[s for s in states for _ in range(2)];viewers=[i%2 for i in range(len(states))];layout,fields=pack(states)
        with backend(layout)as enc:self.compare(states,viewers,enc.project(fields,torch.tensor(viewers),torch.ones(len(states),dtype=torch.bool)))

    def test_dynamic_instant_kit_sources_and_effect_forbid(self):
        from app.modules.card_game.engine.duel_v2.state import card_instance
        from app.modules.card_game.engine.duel_v2.modifiers import add_effect
        states=[]
        for seed,deck,cids in ((70,'starter',('NF01','Z05')),(71,'quick-rush',('U01','U02','U07')),(72,'zhenhong',('I08',))):
            for turn in (1,5):
                st=state(seed,a=deck)
                if deck=='quick-rush':
                    from app.modules.card_game.content.duel_v2 import STARTER_DECKS,CARDS
                    from app.modules.card_game.engine.duel_v2 import new_game
                    custom=deepcopy(next(d for d in STARTER_DECKS if d['id']=='starter'))
                    custom['character_ids']=['haiyue' if c=='iloy'else c for c in custom['character_ids']]
                    custom['card_ids']=['U'+c[-2:] if CARDS[c]['character_id']=='iloy'else c for c in custom['card_ids']]
                    st=new_game(seed=seed,decks={'a':custom,'b':custom},skip_mulligan=True)
                st['turn']=turn;team=st['sides']['a'];team['hand']=[card_instance(st,cid,'a')for cid in cids]
                if deck=='starter':team['characters']['zero']['shape']='Z05'
                if deck=='quick-rush':team['attacked_this_turn']=[team['characters'][team['order'][0]]['entity_id']]
                if deck=='zhenhong':st['sides']['b']['front_debuff']['delay']={'left':1,'by':'a'}
                states.extend((deepcopy(st),deepcopy(st)))
                # A forbidden instant query does not clear an already printed
                # instant flag in the existing public presentation.
                for h in states[-1]['sides']['a']['characters'].values():
                    add_effect(h,'test.forbid','card.instant',True,op='forbid',clear_on=())
        viewers=[0]*len(states);layout,fields=pack(states)
        with backend(layout)as enc:self.compare(states,viewers,enc.project(fields,torch.tensor(viewers),torch.ones(len(states),dtype=torch.bool)))

    def test_terminal_and_hidden_deck_do_not_leak(self):
        st=state(41,a='zhenhong');st['sides']['a']['characters']['zhenhong']['awakened']=True
        st.update(phase='finished',winner='a',reason='player_hp');layout,fields=pack([st])
        with backend(layout)as enc:
            p=enc.prepare(fields,torch.tensor([0]),torch.ones(1,dtype=torch.bool));out=enc.launch(p);self.compare([st],[0],out);baseline=out['cards'].clone()
            fields['rng'].add_(1);hidden=int(fields['zone_card_index'][0,1,0,0]);fields['card_template'][0,hidden]=fields['card_template'][0,int(fields['zone_card_index'][0,1,1,0])]
            self.assertTrue(torch.equal(enc.launch(p)['cards'],baseline))
            fields['row_error'][0]=64;out=enc.launch(p);self.assertEqual(int(out['errors'][0]),64);self.assertFalse(out['valid'].any());self.assertTrue((out['cards'][...,0]==-1).all())

    def test_candidate_encoder_uses_projected_hand_references(self):
        states=[state(50+i,a=k)for i,k in enumerate(seven.KEYS)];layout,fields=pack(states)
        views=[observe(st,'a',include_previews=False)for st in states]
        acts=[[{'type':'play_card','card_id':c['instance_id']}for c in view['sides']['a']['hand']]for view in views]
        args=numeric(views,acts)
        with backend(layout)as enc,DomainSoACandidates(device='cpu',allow_cpu_oracle=True)as candidate:
            out=enc.project(fields,torch.zeros(len(states),dtype=torch.int64),torch.ones(len(states),dtype=torch.bool))
            for b,(st,view)in enumerate(zip(states,views)):
                refs={c['instance_id']:int(out['hand_refs'][b,j])for j,c in enumerate(st['sides']['a']['hand'])}
                for j,a in enumerate(acts[b]):args['actions'][b,j,2]=refs[a['card_id']]
            args['cards']=out['cards'];args['prior_errors']=out['errors']
            actual=candidate.encode(**args)
            self.assertFalse(actual['errors'].any())
            for b,(view,a)in enumerate(zip(views,acts)):
                _,gold=seven.encode(view,a);self.assertTrue(torch.equal(actual['features'][b,:len(a)],torch.tensor(gold)))

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_graph_live_projection(self):
        states=[state(100+i%7,a=seven.KEYS[i%7])for i in range(400)]
        from app.modules.card_game.engine.duel_v2.modifiers import add_effect
        for i,st in enumerate(states):
            team=st['sides']['a']
            if i%7==3:team['characters']['zhenhong']['awakened']=True
            if i%7==6:
                h=team['characters']['lingke'];h['shape']='K03';team['ap']=0
                add_effect(team,'lingke.next_instant','card.instant',True,op='allow',scope='allies',source_entity_id=h['entity_id'],source_key='K02',clear_on=(),conditions=[{'id':'card_owner_other'}])
            if i%5==0:
                st['phase']='choice';st['pending_choice']={'kind':'hand_redraw','side':'a','prompt':'选择','cards':team['hand'][:2]}
        viewers=[i%2 for i in range(400)];layout,fields=pack(states,device='cuda')
        with backend(layout,'cuda')as enc:
            vi=torch.tensor(viewers,device='cuda');ac=torch.ones(400,dtype=torch.bool,device='cuda');p=enc.prepare(fields,vi,ac)
            enc.launch(p);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=enc.launch(p)
            graph.replay();torch.cuda.synchronize();self.compare(states,viewers,out)
            vi[0]=1;fields['row_error'][1]=32;fields['catalog_digest'][2,0]^=1
            graph.replay();torch.cuda.synchronize();self.assertEqual(out['errors'][:3].cpu().tolist(),[0,32,1]);self.assertFalse(out['valid'][1:3].any())
            first={k:v[:1]for k,v in out.items()if isinstance(v,torch.Tensor)};first.update(full_duel_qualified=False,legal_actions_available=False,card_fields=out['card_fields'])
            self.compare(states[:1],[1],first)
            # Reset, then bind public table and error buffers directly into the
            # candidate encoder. Host actions are diagnostics, not GPU legality.
            vi.copy_(torch.tensor(viewers,device='cuda'));fields['row_error'].zero_();fields['catalog_digest'][2,0]^=1
            enc.launch(p)
            views=[observe(st,('a','b')[v],include_previews=False)for st,v in zip(states,viewers)]
            actions=[[{'type':'end_turn'},{'type':'play_card','card_id':st['sides'][('a','b')[v]]['hand'][0]['instance_id']}]for st,v in zip(states,viewers)]
            args=numeric(views,actions,device='cuda');args['cards']=out['cards'];args['prior_errors']=out['errors']
            args['actions'][:,1,2].copy_(out['hand_refs'][:,0])
            with DomainSoACandidates(device='cuda')as candidate:
                cp=candidate.prepare(**args);candidate.launch(cp);torch.cuda.synchronize();combined=torch.cuda.CUDAGraph()
                with torch.cuda.graph(combined):
                    enc.launch(p);encoded=candidate.launch(cp)
                combined.replay();torch.cuda.synchronize()
                from tests.test_duel_v2_domain_soa_candidates import CandidatesTest
                CandidatesTest().compare(views,actions,encoded)
                fields['row_error'][1]=32;fields['catalog_digest'][2,0]^=1
                combined.replay();torch.cuda.synchronize()
                self.assertEqual(encoded['errors'][:3].cpu().tolist(),[0,32,1]);self.assertFalse(encoded['valid'][1:3].any())

