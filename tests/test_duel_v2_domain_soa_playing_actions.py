"""Whole playing-phase lists versus formal legality; no physical execution."""
from copy import deepcopy
import shutil
import unittest
import torch
from app.modules.card_game.engine.duel_v2.flow import legal_actions
from app.modules.card_game.engine.duel_v2.projection import observe
from app.modules.card_game.rl import seven_lineup as seven
from app.modules.card_game.rl.league_schema import OPTION_IDS
from app.modules.card_game.rl.batched_duel.domain_soa_playing_actions import DomainPlayingActions
from app.modules.card_game.rl.batched_duel.domain_soa_candidates import DomainSoACandidates
from tests.test_duel_v2_domain_soa_character_queries import state,pack
from tests.test_duel_v2_domain_soa_codec import _natural_states


def backend(layout,device='cpu',max_actions=800):
    if device=='cpu'and not(shutil.which('clang++')or shutil.which('g++')):raise unittest.SkipTest('Explicit compiler oracle unavailable')
    return DomainPlayingActions(layout,device=device,allow_cpu_oracle=device=='cpu',max_actions=max_actions)


class PlayingActionsTest(unittest.TestCase):
    def compare(self,states,viewers,fields,out):
        self.assertEqual(out['errors'].cpu().tolist(),[0]*len(states))
        actions=out['actions'].cpu();payloads=out['payloads'].cpu();counts=out['counts'].cpu()
        for b,(st,v)in enumerate(zip(states,viewers)):
            side=('a','b')[v];expected=[a for a in legal_actions(st,side)if a['type']!='concede'];actual=[]
            for index,(code,actor,ref,ts,target,opt)in enumerate(actions[b,:int(counts[b])].tolist()):
                pc,pa,pr,pt,ps,px,po=payloads[b,index].tolist();self.assertEqual((pc,pr,po),(code,ref,opt))
                entry={'type':seven.ACTION_TYPES[code]}
                if code in (1,2,7):
                    entry['character_id']=list(st['sides'][side]['characters'])[pa];self.assertEqual(entry['character_id'],seven.SEATS[actor])
                if code==3:entry['card_id']=('a','b')[int(fields['card_side'][b,ref])]+ '-'+str(int(fields['card_instance_slot'][b,ref]))
                if ts>=0:
                    self.assertEqual(ps,ts)
                    cid='player'if pt==1 else list(st['sides'][('a','b')[ps]]['characters'])[px]
                    entry['target_id']=('a','b')[ts]+':'+cid
                if opt>=0:entry['option_id']=OPTION_IDS[opt]
                actual.append(entry)
            self.assertEqual(actual,expected,(b,side,st['turn'],st['phase']))
        self.assertFalse(out['full_duel_qualified']);self.assertFalse(out['physical_rules_available'])

    def test_seven_natural_playing_and_both_viewers(self):
        base=[st for st in _natural_states()if st['phase']=='playing'][::11]
        states=[st for st in base for _ in range(2)];viewers=[i%2 for i in range(len(states))];layout,fields=pack(states)
        with backend(layout)as enc:self.compare(states,viewers,fields,enc.project(fields,torch.tensor(viewers),torch.ones(len(states),dtype=torch.bool)))

    def test_payment_mimic_normal_instant_seal_ultimate_and_options(self):
        from app.modules.card_game.engine.duel_v2.state import card_instance
        from app.modules.card_game.engine.duel_v2.modifiers import add_effect
        states=[]
        for seed,key in enumerate(seven.KEYS):
            for variant in range(4):
                st=state(100+seed,a=key);st['active_side']='a';team=st['sides']['a'];team['ap']=variant
                team['normal_attack_available']=variant%2==0;team['ultimate_available']=True
                for h in team['characters'].values():h['energy']=10;h['jingu']=6
                if key=='quick-rush':
                    team['characters']['xiaozhi']['shape']='Q07';team['hand']=[card_instance(st,c,'a')for c in ('Q01','Q02','Q03')]
                if key=='zhenhong'and variant==0:team['characters']['zhenhong']['awakened']=True
                if key=='lingke'and variant==0:team['characters']['lingke']['shape']='K03'
                if key=='collapse'and variant==0:team['characters']['heiyu']['awakened']=True
                if variant==1:team['seal_normal_attack']='summon_e999'
                if variant==2:
                    h=team['characters'][team['order'][0]]
                    add_effect(h,'test.instant','rule.normal_attack_instant',True,op='allow',clear_on=())
                    team['used']['instant']=False;h['flags']['free_normal_attack_turn']=st['turn']
                if variant==3:
                    team['used']['instant']=True
                    for h in team['characters'].values():add_effect(h,'test.discount','card.cost',-.5,clear_on=())
                states.append(st)
        layout,fields=pack(states);vi=torch.zeros(len(states),dtype=torch.int64)
        with backend(layout)as enc:self.compare(states,[0]*len(states),fields,enc.project(fields,vi,torch.ones(len(states),dtype=torch.bool)))

    def special_states(self):
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS,CARDS
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.engine.duel_v2.state import card_instance
        from app.modules.card_game.engine.duel_v2.summons import summon_front
        from app.modules.card_game.engine.duel_v2.modifiers import add_effect
        result=[]
        custom=deepcopy(next(d for d in STARTER_DECKS if d['id']=='starter'))
        custom['character_ids']=['haiyue'if c=='iloy'else c for c in custom['character_ids']]
        custom['card_ids']=['U'+c[-2:]if CARDS[c]['character_id']=='iloy'else c for c in custom['card_ids']]
        st=new_game(seed=5,decks={'a':custom,'b':custom},first_side='a',skip_mulligan=True)
        team=st['sides']['a'];team['ap']=0;team['normal_attack_available']=False;team['characters']['haiyue']['shape']='U06'
        team['attacked_this_turn']=[team['characters']['nanali']['entity_id']];result.append(st)
        st=state(6,a='murk');st['active_side']='a';team=st['sides']['a'];team['ap']=0;team['normal_attack_available']=False
        h=team['characters']['anhunqu'];h['shape']='U06';h['flags']['mimic']={'id':'haiyue'}
        team['attacked_this_turn']=[team['characters']['canhong']['entity_id']];result.append(st)
        for ghost in (False,True):
            st=state(9,a='murk');st['active_side']='a';team=st['sides']['a'];team['ap']=3
            team['hand']=[card_instance(st,c,'a')for c in ('S06','S04','C01')]
            if ghost:summon_front(st,'a',name='鬼郎丸',attack=1,hp=4)
            team['characters']['canhong']['shape']='C08';st['sides']['b']['front_debuff']['burn']={'by':'a','stacks':5};result.append(st)
        st=state(20);st['active_side']='a';team=st['sides']['a'];team['ap']=0
        cid=next(c for c in CARDS if CARDS[c]['character_id']in team['characters']and CARDS[c].get('instant'))
        team['hand']=[card_instance(st,cid,'a')]
        for h in team['characters'].values():add_effect(h,'test.forbid','card.instant',True,op='forbid',clear_on=())
        result.append(st)
        return result

    def test_haiyue_mimic_ghost_canhong_and_forbidden_printed_instant(self):
        states=self.special_states();layout,fields=pack(states)
        with backend(layout)as enc:self.compare(states,[0]*len(states),fields,enc.project(fields,torch.zeros(len(states),dtype=torch.int64),torch.ones(len(states),dtype=torch.bool)))

    def test_source_errors_noncontiguous_and_capacity_do_not_truncate(self):
        states=[state(31),state(32)];layout,raw=pack(states);fields={}
        for k,v in raw.items():
            storage=torch.empty((4,)+tuple(v.shape[1:]),dtype=v.dtype);fields[k]=storage[::2];fields[k].copy_(v)
        with backend(layout)as enc:
            p=enc.prepare(fields,torch.zeros(2,dtype=torch.int64),torch.ones(2,dtype=torch.bool));enc.launch(p)
            fields['row_error'][0]=64;fields['catalog_digest'][1,0]^=1;out=enc.launch(p)
            self.assertEqual(out['errors'].tolist(),[64,1]);self.assertFalse(out['counts'].any());self.assertTrue((out['payloads']==-1).all())
        st=state(40);st['active_side']='a';st['sides']['a'].update(ap=3,normal_attack_available=True);layout,fields=pack([st])
        with backend(layout,max_actions=1)as enc:
            out=enc.project(fields,torch.zeros(1,dtype=torch.int64),torch.ones(1,dtype=torch.bool))
            self.assertEqual(int(out['errors'][0]),4);self.assertEqual(int(out['counts'][0]),0)

    def test_nonactor_finished_and_unsupported_choice(self):
        states=[state(3),state(4),state(5)];states[0]['active_side']='b';states[1].update(phase='finished',winner='a',reason='player_hp')
        states[2].update(phase='choice',pending_choice={'kind':'hand_redraw','side':'a','prompt':'选择','cards':states[2]['sides']['a']['hand'][:2]})
        layout,fields=pack(states)
        with backend(layout)as enc:
            out=enc.project(fields,torch.zeros(3,dtype=torch.int64),torch.ones(3,dtype=torch.bool))
            self.assertEqual(out['errors'].tolist(),[0,0,2]);self.assertFalse(out['counts'].any())

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_complete_playing_lists_and_graph_errors(self):
        base=[st for st in _natural_states()if st['phase']=='playing'][::9]+self.special_states();states=[base[i%len(base)]for i in range(400)];viewers=[('a','b').index(st['active_side'])for st in states]
        layout,fields=pack(states,device='cuda')
        with backend(layout,'cuda')as enc:
            vi=torch.tensor(viewers,device='cuda');ac=torch.ones(400,dtype=torch.bool,device='cuda');p=enc.prepare(fields,vi,ac)
            enc.launch(p);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=enc.launch(p)
            graph.replay();torch.cuda.synchronize()
            chosen=list(range(7))+[10,20,30]+list(range(len(base)-5,len(base)))
            small={k:v[chosen]for k,v in out.items()if isinstance(v,torch.Tensor)};small.update(full_duel_qualified=False,physical_rules_available=False)
            self.compare([states[i]for i in chosen],[viewers[i]for i in chosen],{k:v[chosen]for k,v in fields.items()},small)
            fields['row_error'][0]=64;fields['catalog_digest'][1,0]^=1
            graph.replay();torch.cuda.synchronize();self.assertEqual(out['errors'][:2].cpu().tolist(),[64,1]);self.assertFalse(out['counts'][:2].any())
