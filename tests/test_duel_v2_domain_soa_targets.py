"""Exact target ordering versus formal targets, separate from full legality."""
from copy import deepcopy
from unittest.mock import patch
import unittest
import torch
from app.modules.card_game.rl import seven_lineup as seven
from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.content.duel_v2.effects import TARGET_POLICIES
from app.modules.card_game.engine.duel_v2.flow import targets
from app.modules.card_game.engine.duel_v2.state import card_instance
from app.modules.card_game.engine.duel_v2.summons import summon_front,dismiss
from app.modules.card_game.rl.batched_duel.domain_soa_targets import DomainSoATargets,POLICIES,NONE,PLAYER,CHARACTER,DISCARD
from tests.test_duel_v2_domain_soa_character_queries import state,pack


def requests(states,cards=seven.CARD_IDS,device='cpu'):
    specs=[(s,c)for s in (0,1)for c in cards];B=len(states)
    side=torch.tensor([[s for s,c in specs]]*B,dtype=torch.int64,device=device)
    kinds=torch.tensor([[seven.CARD_INDEX[c]for s,c in specs]]*B,dtype=torch.int64,device=device)
    owners=torch.tensor([[seven.SEAT_INDEX[CARDS[c]['character_id']]for s,c in specs]]*B,dtype=torch.int64,device=device)
    battle=torch.tensor([[CARDS[c]['type']=='battle'for s,c in specs]]*B,dtype=torch.bool,device=device)
    return side,kinds,owners,battle,torch.ones_like(side,dtype=torch.bool),torch.zeros_like(side,dtype=torch.int32)


class TargetsTest(unittest.TestCase):
    def compare(self,states,fields,args,out):
        self.assertFalse(out['errors'].any(),str(out['errors'].cpu().tolist()))
        desc=out['targets'].cpu();counts=out['counts'].cpu();sides=args[0].cpu();kinds=args[1].cpu();owners=args[2].cpu();battle=args[3].cpu()
        for b,st in enumerate(states):
            for n in range(sides.shape[1]):
                side=('a','b')[int(sides[b,n])];cid=seven.CARD_IDS[int(kinds[b,n])]
                card=dict(CARDS[cid],card_id=cid,character_id=seven.SEATS[int(owners[b,n])],type='battle'if bool(battle[b,n])else CARDS[cid]['type'])
                if len(args)>6 and bool(args[6][b,n]):
                    from app.modules.card_game.content.duel_v2.registry import decide
                    from app.modules.card_game.engine.duel_v2.context import EffectContext
                    aimed=decide(EffectContext(st,side,card['character_id'],{'side':side}),'normal_attack_targets',None,actor_only=True)
                    expected=[{'target_id':x}for x in aimed]if isinstance(aimed,list)and aimed else [{}]
                else:expected=targets(st,side,card)
                actual=[]
                for row in desc[b,n,:int(counts[b,n])].tolist():
                    kind,which,slot=row
                    if kind==NONE:actual.append({});continue
                    ps=('a','b')[which]
                    if kind==PLAYER:target=ps+':player'
                    elif kind==CHARACTER:target=ps+':'+list(st['sides'][ps]['characters'])[slot]
                    elif kind==DISCARD:target=('a','b')[int(fields['card_side'][b,slot])]+ '-' +str(int(fields['card_instance_slot'][b,slot]))
                    else:self.fail('Unknown target descriptor')
                    actual.append({'target_id':target})
                self.assertEqual(actual,expected,(b,n,cid,side))
        self.assertFalse(out['full_duel_qualified']);self.assertFalse(out['legal_actions_available'])

    def test_all_public_cards_initial_and_natural_states(self):
        from tests.test_duel_v2_domain_soa_codec import _fourteen_initial_states,_natural_states
        states=_fourteen_initial_states()+_natural_states()[::31];layout,fields=pack(states)
        args=requests(states)
        with DomainSoATargets(layout,device='cpu',allow_cpu_oracle=True)as enc:self.compare(states,fields,args,enc.project(fields,*args))

    def fixture(self):
        st=state(9,a='starter',b='murk')
        for side in ('a','b'):
            team=st['sides'][side];team['hp']=20
            hs=list(team['characters'].values());hs[0]['hp']=0;hs[0]['down_turns']=2
            hs[1]['hp']=2;hs[2]['flags']['collapse']={'side':'a','until':3};team['front']=team['order'][1]
            # Original discard order and copy exclusion must survive filtering.
            for c in ('N01','N02','N03','N04'):
                team['discard'].append(card_instance(st,c,side,copy=c=='N03'))
        summon_front(st,'a',name='鬼郎丸',attack=1,hp=4)
        old=summon_front(st,'b',name='旧召唤',attack=1,hp=4);dismiss(st,'b',old)
        summon_front(st,'b',name='新召唤',attack=1,hp=4)
        return st

    def test_every_policy_downs_collapse_fronts_summons_and_discards(self):
        st=self.fixture();layout,fields=pack([st]);cards=seven.CARD_IDS[:len(POLICIES)]
        overrides=dict(zip(cards,POLICIES))
        with patch.dict(TARGET_POLICIES,overrides),DomainSoATargets(layout,device='cpu',allow_cpu_oracle=True)as enc:
            args=requests([st],cards);self.compare([st],fields,args,enc.project(fields,*args))

    def test_canhong_override_empty_list_mimic_and_finished(self):
        states=[]
        for variant in ('ready','empty','unready','other_mimic','self_mimic','down','finished'):
            st=state(31,a='murk');team=st['sides']['a'];h=team['characters']['canhong'];h['shape']='C08'
            st['sides']['b']['front_debuff'].update(burn={'by':'a','stacks':0})
            from app.modules.card_game.engine.duel_v2.continuous import apply_dot
            apply_dot(st,'b','etch',by='a',source_cid='canhong',stacks=4)
            if variant=='empty':
                for other in st['sides']['b']['characters'].values():other['hp']=0
            if variant=='unready':h['shape']=None
            if variant=='other_mimic':h['flags']['mimic']={'id':'adler'}
            if variant=='self_mimic':h['flags']['mimic']={'id':'canhong'}
            if variant=='down':h['hp']=0
            if variant=='finished':st.update(phase='finished',winner='a',reason='player_hp')
            states.append(st)
        layout,fields=pack(states);args=requests(states,('C01','C02','N01'))
        with DomainSoATargets(layout,device='cpu',allow_cpu_oracle=True)as enc:
            out=enc.project(fields,*args);self.compare(states,fields,args,out)
            self.assertEqual(int(out['counts'][1,0]),0);self.assertEqual(int(out['counts'][2,0]),1)

    def test_normal_attack_empty_override_fallback_and_mimicked_canhong(self):
        states=[]
        for variant in ('ready','empty','mimicked','no_callback'):
            st=state(70,a='murk');team=st['sides']['a'];h=team['characters']['canhong'];h['shape']='C08'
            st['sides']['b']['front_debuff']['burn']={'by':'a','stacks':5}
            if variant=='empty':
                for e in st['sides']['b']['characters'].values():e['hp']=0
            if variant=='mimicked':
                team['characters']['anhunqu']['shape']='C08';team['characters']['anhunqu']['flags']['mimic']={'id':'canhong'}
            if variant=='no_callback':h['flags']['mimic']={'id':'adler'}
            states.append(st)
        layout,fields=pack(states);args=requests(states,('C01',));normal=torch.ones_like(args[4])
        args[2][2,0]=seven.SEAT_INDEX['anhunqu'];args=(*args,normal)
        with DomainSoATargets(layout,device='cpu',allow_cpu_oracle=True)as enc:
            out=enc.project(fields,*args);self.compare(states,fields,args,out);self.assertEqual(int(out['counts'][1,0]),1)

    def test_live_noncontiguous_errors_unsupported_and_inactive(self):
        states=[self.fixture(),state(12)];layout,raw=pack(states);fields={}
        for k,v in raw.items():
            storage=torch.empty((4,)+tuple(v.shape[1:]),dtype=v.dtype);fields[k]=storage[::2];fields[k].copy_(v)
        args=requests(states,('N05','C01'))
        with DomainSoATargets(layout,device='cpu',allow_cpu_oracle=True)as enc:
            p=enc.prepare(fields,*args);self.compare(states,fields,args,enc.launch(p))
            args[0][0,0]=-1;args[2][0,1]=seven.SEAT_INDEX['summon'];fields['row_error'][1]=64
            out=enc.launch(p);self.assertEqual(out['errors'][0,:2].tolist(),[1,2]);self.assertTrue((out['errors'][1]==64).all());self.assertFalse(out['counts'][1].any())
            args[4][0,:2]=False;out=enc.launch(p);self.assertTrue((out['errors'][0,:2]==0).all());self.assertFalse(out['counts'][0,:2].any())
            fields['catalog_digest'][0,0]^=1;self.assertTrue((enc.launch(p)['errors'][0,2:]==1).all())

    def test_empty_fronts_hands_and_unknown_policy_refuse_partial_success(self):
        st=state(42);st['sides']['b']['hand']=[]
        cards=('N01','N02','N03')
        with patch.dict(TARGET_POLICIES,dict(zip(cards,('enemy_front','ally_front','enemy_hand')))):
            layout,fields=pack([st]);args=requests([st],cards)
            with DomainSoATargets(layout,device='cpu',allow_cpu_oracle=True)as enc:
                out=enc.project(fields,*args);self.compare([st],fields,args,out)
                self.assertTrue((out['counts'][0,:3]==0).all())
                enc.policies[seven.CARD_INDEX['N01']]=-1
                out=enc.project(fields,*args);self.assertEqual(int(out['errors'][0,0]),2);self.assertEqual(int(out['counts'][0,0]),0)

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_graph_ordered_targets_and_live_mutations(self):
        st=self.fixture();st['sides']['b']['characters']['canhong']['shape']='C08'
        st['sides']['a']['front_debuff']['burn']={'by':'b','stacks':5}
        states=[st]*400;layout,fields=pack(states,device='cuda');args=requests(states,('N05','C01','K01','N06'),'cuda')
        with DomainSoATargets(layout,device='cuda')as enc:
            normal=torch.zeros_like(args[4]);normal[:,5]=True;args=(*args,normal)
            p=enc.prepare(fields,*args);enc.launch(p);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=enc.launch(p)
            g.replay();torch.cuda.synchronize()
            subset={k:v[:3]for k,v in out.items()if isinstance(v,torch.Tensor)};subset.update(full_duel_qualified=False,legal_actions_available=False)
            self.compare(states[:3],{k:v[:3]for k,v in fields.items()},tuple(a[:3]for a in args),subset)
            args[0][0,0]=1;fields['zone_status_i32'][0,0,1,0]=0;fields['row_error'][1]=32;fields['catalog_digest'][2,0]^=1
            g.replay();torch.cuda.synchronize();self.assertTrue((out['errors'][1]==32).all());self.assertTrue((out['errors'][2]==1).all())
            subset={k:v[:1]for k,v in out.items()if isinstance(v,torch.Tensor)};subset.update(full_duel_qualified=False,legal_actions_available=False)
            after=deepcopy(st);after['sides']['a']['front_debuff']['burn']['stacks']=0
            self.compare([after],{k:v[:1]for k,v in fields.items()},tuple(a[:1]for a in args),subset)
