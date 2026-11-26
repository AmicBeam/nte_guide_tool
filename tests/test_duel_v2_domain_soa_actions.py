"""All phase action lists independently compared to formal legal_actions."""
from copy import deepcopy
import shutil
import unittest
import torch
from app.modules.card_game.engine.duel_v2.flow import legal_actions,apply_action
from app.modules.card_game.engine.duel_v2.state import card_instance
from app.modules.card_game.engine.duel_v2.projection import observe
from app.modules.card_game.rl import seven_lineup as seven
from app.modules.card_game.rl.league_schema import OPTION_IDS
from app.modules.card_game.rl.batched_duel.domain_soa_actions import DomainSoAActions
from app.modules.card_game.rl.batched_duel.domain_soa_candidates import DomainSoACandidates
from tests.test_duel_v2_domain_soa_character_queries import state,pack
from tests.test_duel_v2_domain_soa_codec import _fourteen_initial_states,_natural_states


def backend(layout,device='cpu',max_actions=800):
    if device=='cpu'and not(shutil.which('clang++')or shutil.which('g++')):raise unittest.SkipTest('Explicit compiler oracle unavailable')
    return DomainSoAActions(layout,device=device,allow_cpu_oracle=device=='cpu',max_actions=max_actions)


class ActionsTest(unittest.TestCase):
    def compare(self,states,viewers,fields,out):
        self.assertEqual(out['errors'].cpu().tolist(),[0]*len(states))
        actions=out['actions'].cpu();payload=out['payloads'].cpu();counts=out['counts'].cpu()
        selected=out['selected'].cpu();top=out['top'].cpu();bottom=out['bottom'].cpu();plural=out['bottom_plural'].cpu();K=fields['card_valid'].shape[1]
        def identity(b,ref):
            return ('a','b')[int(fields['card_side'][b,ref])]+ '-'+str(int(fields['card_instance_slot'][b,ref]))
        for b,(st,v)in enumerate(zip(states,viewers)):
            side=('a','b')[v];expected=[a for a in legal_actions(st,side)if a['type']!='concede'];actual=[]
            for i,(code,actor,ref,ts,target,opt)in enumerate(actions[b,:int(counts[b])].tolist()):
                entry={'type':seven.ACTION_TYPES[code]}
                if code in (1,2,7):entry['character_id']=list(st['sides'][side]['characters'])[int(payload[b,i,1])]
                if code==3:entry['card_id']=identity(b,ref)
                if code==4:entry['choice_id']=identity(b,ref-K)
                if code in (5,8):entry['card_ids']=[identity(b,x if code==5 else x-K)for x in selected[b,i].tolist()if x>=0]
                if code==6:
                    entry['card_ids']=[identity(b,x-K)for x in top[b,i].tolist()if x>=0]
                    bottoms=[identity(b,x-K)for x in bottom[b,i].tolist()if x>=0]
                    if bool(plural[b,i]):entry['bottom_card_ids']=bottoms
                    else:entry['bottom_card_id']=bottoms[0]if bottoms else None
                if ts>=0:
                    tk=int(payload[b,i,3]);slot=int(payload[b,i,5]);targetid='player'if tk==1 else list(st['sides'][('a','b')[ts]]['characters'])[slot]
                    entry['target_id']=('a','b')[ts]+':'+targetid
                if opt>=0:entry['option_id']=OPTION_IDS[opt]
                actual.append(entry)
            self.assertEqual(actual,expected,(b,side,st['phase'],(st.get('pending_choice')or{}).get('kind')))
        self.assertFalse(out['full_duel_qualified']);self.assertFalse(out['physical_rules_available'])

    def fixtures(self):
        states=_fourteen_initial_states()[:2]
        for kind in ('kit','hand_redraw','kit_multi','inspect_top','deck_order','hand_deck_placement','heiyu_divination'):
            st=state(100,a='collapse');team=st['sides']['a'];cards=team['deck'][:3];team['deck']=team['deck'][3:]
            st['phase']='choice';pending={'kind':kind,'side':'a','prompt':'选择','cards':cards}
            if kind=='kit_multi':pending['max_select']=2
            if kind in ('inspect_top','hand_deck_placement'):pending['deck_order']=True
            if kind=='hand_deck_placement':pending.update(hand_placement=True,max_bottom=None)
            if kind=='heiyu_divination':pending['actor']='heiyu'
            st['pending_choice']=pending;states.append(st)
        for actor,card in (('heiyu','PF01'),('anhunqu','PF01'),('heiyu',None)):
            st=state(105,a='collapse');team=st['sides']['a'];cards=team['deck'][:4];team['deck']=team['deck'][4:]
            st['phase']='choice';st['pending_choice']={'kind':'heiyu_divination','side':'a','prompt':'检视','cards':cards,'actor':actor}
            if actor=='anhunqu':team['characters'][actor]['flags']['mimic']={'id':'heiyu'}
            st['operation']={'side':'a','actor':actor,'card':card_instance(st,card,'a')if card else None,'first_bonus':0,'redeem_bonus':0,'record_candidate':False,'participants':[]};states.append(st)
        done=apply_action(_fourteen_initial_states()[0],'a',{'type':'mulligan','card_ids':[]});states.append(done)
        terminal=state(110);terminal.update(phase='finished',winner='a',reason='player_hp');states.append(terminal)
        return states

    def test_all_initial_natural_and_both_viewers(self):
        states=[st for st in _fourteen_initial_states()+_natural_states()[::13]for _ in range(2)]
        viewers=[i%2 for i in range(len(states))];layout,fields=pack(states)
        with backend(layout)as enc:self.compare(states,viewers,fields,enc.project(fields,torch.tensor(viewers),torch.ones(len(states),dtype=torch.bool)))

    def test_resolved_deck_orders_mulligans_multiselect_and_nonactors(self):
        states=[st for st in self.fixtures()for _ in range(2)];viewers=[i%2 for i in range(len(states))];layout,fields=pack(states)
        with backend(layout)as enc:self.compare(states,viewers,fields,enc.project(fields,torch.tensor(viewers),torch.ones(len(states),dtype=torch.bool)))

    def test_candidate_encoding_from_all_phase_outputs(self):
        states=self.fixtures()[:10];viewers=[0]*len(states);layout,fields=pack(states)
        with backend(layout)as enc,DomainSoACandidates(device='cpu',allow_cpu_oracle=True)as cand:
            out=enc.project(fields,torch.zeros(len(states),dtype=torch.int64),torch.ones(len(states),dtype=torch.bool))
            from app.modules.card_game.rl.batched_duel.domain_soa_observation import PI
            turns=fields['player_int_value'][:,0,PI['turn_count']].long()
            result=cand.encode(cards=out['public_cards'],actions=out['actions'],selected=out['selected'],top=out['top'],bottom=out['bottom'],
                counts=out['counts'],viewers=torch.zeros(len(states),dtype=torch.int64),turn_counts=turns,
                active=torch.ones(len(states),dtype=torch.bool),prior_errors=out['errors'])
            self.assertFalse(result['errors'].any())
            for b,st in enumerate(states):
                actions=[a for a in legal_actions(st,'a')if a['type']!='concede'];_,gold=seven.encode(observe(st,'a',include_previews=False),actions)
                self.assertTrue(torch.equal(result['features'][b,:len(actions)],torch.tensor(gold)),str((b,st['phase'])))

    def test_capacity_counts_exclude_concede_and_inactive_rows(self):
        st=state(20);st['phase']='choice';st['pending_choice']={'kind':'kit','side':'a','prompt':'选择','cards':st['sides']['a']['hand'][:1]}
        layout,fields=pack([st])
        with backend(layout,max_actions=1)as enc:
            p=enc.prepare(fields,torch.zeros(1,dtype=torch.int64),torch.ones(1,dtype=torch.bool));out=enc.launch(p)
            self.compare([st],[0],fields,out);self.assertEqual(int(out['counts'][0]),1)
            p.playing.public.observation.active_source.zero_();fields['pending_choice_present'].zero_()
            out=enc.launch(p);self.assertEqual(int(out['errors'][0]),0);self.assertEqual(int(out['counts'][0]),0)

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_all_phases_and_graph(self):
        base=self.fixtures()+_natural_states()[::23];states=[base[(i//2)%len(base)]for i in range(400)];viewers=[i%2 for i in range(400)]
        layout,fields=pack(states,device='cuda')
        with backend(layout,'cuda')as enc:
            vi=torch.tensor(viewers,device='cuda');ac=torch.ones(400,dtype=torch.bool,device='cuda');p=enc.prepare(fields,vi,ac)
            enc.launch(p);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=enc.launch(p)
            g.replay();torch.cuda.synchronize();chosen=list(range(2*len(self.fixtures())+14))
            small={k:v[chosen]for k,v in out.items()if isinstance(v,torch.Tensor)};small.update(full_duel_qualified=False,physical_rules_available=False)
            self.compare([states[i]for i in chosen],[viewers[i]for i in chosen],{k:v[chosen]for k,v in fields.items()},small)
            vi[0]=1-viewers[0];fields['row_error'][1]=64;fields['catalog_digest'][2,0]^=1
            g.replay();torch.cuda.synchronize();self.assertEqual(out['errors'][:3].cpu().tolist(),[0,64,1]);self.assertFalse(out['counts'][1:3].any())
            small={k:v[:1]for k,v in out.items()if isinstance(v,torch.Tensor)};small.update(full_duel_qualified=False,physical_rules_available=False)
            self.compare(states[:1],[1-viewers[0]],{k:v[:1]for k,v in fields.items()},small)
            # Reuse the same captured graph across actual phase/buffer changes.
            replacement=state(999,a='quick-rush');replacement['active_side']='a';replacement['sides']['a'].update(ap=3,normal_attack_available=True)
            for replacement in (replacement,self.fixtures()[9]):
                _,new=pack([replacement],device='cuda',layout=layout)
                for name,value in fields.items():value[0].copy_(new[name][0])
                vi[0]=0;g.replay();torch.cuda.synchronize()
                small={k:v[:1]for k,v in out.items()if isinstance(v,torch.Tensor)};small.update(full_duel_qualified=False,physical_rules_available=False)
                self.compare([replacement],[0],{k:v[:1]for k,v in fields.items()},small)

