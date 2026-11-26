"""Independent public encoder parity, distinct from legal-rule qualification."""
import unittest
from copy import deepcopy
import torch
from app.modules.card_game.rl import seven_lineup as seven
from app.modules.card_game.rl.league_schema import OPTION_IDS
from app.modules.card_game.engine.duel_v2.projection import observe
from app.modules.card_game.engine.duel_v2.flow import legal_actions
from app.modules.card_game.rl.batched_duel.domain_soa_candidates import DomainSoACandidates,CARD_FIELDS
from tests.test_duel_v2_domain_soa_character_queries import state


def numeric(views,actions,device='cpu',capacity=None):
    B=len(views);A=capacity or max(1,max(map(len,actions)));K=64
    f={'cards':torch.zeros(B,K,len(CARD_FIELDS),dtype=torch.float64,device=device),
       'actions':torch.full((B,A,6),-1,dtype=torch.int64,device=device),
       'selected':torch.full((B,A,10),-1,dtype=torch.int64,device=device),
       'top':torch.full((B,A,4),-1,dtype=torch.int64,device=device),
       'bottom':torch.full((B,A,4),-1,dtype=torch.int64,device=device),
       'counts':torch.tensor(list(map(len,actions)),device=device),
       'viewers':torch.tensor([('a','b').index(v['viewer_side']) for v in views],device=device),
       'turn_counts':torch.tensor([v['sides'][v['viewer_side']]['turn_count']for v in views],device=device),
       'active':torch.ones(B,dtype=torch.bool,device=device),'prior_errors':torch.zeros(B,dtype=torch.int32,device=device)}
    def seat(cid):return seven.SEAT_INDEX['summon' if cid.startswith('summon_') else cid]
    for b,(view,acts)in enumerate(zip(views,actions)):
        own=view['sides'][view['viewer_side']]
        # Public hand and choices may share physical identity but differ in face.
        hand={};choices={};n=0
        for rows,mapping in ((own['hand'],hand),([c['card']for c in (view.get('pending_choice')or{}).get('choices',[])],choices)):
            for card in rows:
                mark=card.get('jingu_mark');face=card.get('hand_face')or{}
                f['cards'][b,n]=torch.tensor([seven.CARD_INDEX[card['card_id']],seat(card['character_id']),bool(card.get('revealed')),
                    mark is not None,0 if mark is None else mark['delta'],card.get('attack_mode')=='set',bool(card.get('instant')),
                    bool(card.get('copy')),card.get('expires_turn')or 0,bool(card.get('derived')),card.get('antique_investment')or 0,
                    seven.CARD_INDEX[face['card_id']]if face.get('card_id')else -1,card.get('additional_ap_cost')or 0],dtype=torch.float64,device=device)
                mapping[card['instance_id']]=n;n+=1
        assert n<=K
        for i,a in enumerate(acts):
            target=a.get('target_id');ts=-1;targetseat=-1
            if target:
                side,cid=target.split(':',1);ts=('a','b').index(side);targetseat=-1 if cid=='player'else seat(cid)
            f['actions'][b,i]=torch.tensor([seven.ACTION_TYPES.index(a['type']),seat(a['character_id'])if a.get('character_id')else -1,
                hand[a['card_id']]if a['type']=='play_card'else choices[a['choice_id']]if a['type']=='choose'else -1,
                ts,targetseat,OPTION_IDS.index(a['option_id'])if a.get('option_id')is not None else -1],device=device)
            if a['type']in ('mulligan','choose_cards'):
                refs=[(hand[x]if a['type']=='mulligan'else choices.get(x,hand.get(x)))for x in a['card_ids']]
                f['selected'][b,i,:len(refs)]=torch.tensor(refs,dtype=torch.int64,device=device)
            if a['type']=='choose_deck_order':
                for name,ids in (('top',a['card_ids']),('bottom',a.get('bottom_card_ids',[a['bottom_card_id']]if a.get('bottom_card_id')else []))):
                    f[name][b,i,:len(ids)]=torch.tensor([choices[x]for x in ids],dtype=torch.int64,device=device)
    return f


class CandidatesTest(unittest.TestCase):
    def compare(self,views,acts,out):
        self.assertEqual(out['errors'].cpu().tolist(),[0]*len(views))
        for b,(view,actions)in enumerate(zip(views,acts)):
            _,gold=seven.encode(view,actions)
            actual=out['features'][b,:len(actions)].cpu();expected=torch.tensor(gold)
            differences=[(i,k,float(actual[i,j]),float(expected[i,j]))for i in range(len(actions))for j,k in enumerate(seven.candidate_names())if actual[i,j]!=expected[i,j]]
            self.assertEqual(differences,[],str((b,differences[:10])))
        self.assertFalse(out['full_duel_qualified']);self.assertFalse(out['legal_actions_available'])

    def test_all_seven_natural_legal_lists(self):
        from tests.test_duel_v2_domain_soa_codec import _natural_states,_fourteen_initial_states
        views=[];acts=[]
        for st in _fourteen_initial_states()+_natural_states()[::11]:
            for side in ('a','b'):
                view=observe(st,side,include_previews=False)
                actions=[a for a in legal_actions(st,side)if a['type']!='concede']
                views.append(view);acts.append(actions)
        with DomainSoACandidates(device='cpu',allow_cpu_oracle=True)as enc:self.compare(views,acts,enc.encode(**numeric(views,acts)))

    def test_all_categories_order_context_and_substitution(self):
        views=[];acts=[]
        for seed,deck in ((81,'zhenhong'),(82,'lingke'),(83,'murk')):
            st=state(seed,a=deck);team=st['sides']['a']
            if deck=='zhenhong':team['characters']['zhenhong']['awakened']=True
            if deck=='lingke':team['characters']['lingke']['shape']='K03';team['ap']=0
            team['hand'][0].update(copy=True,expires_turn=3,additional_ap_cost=2,antique_investment=3,jingu_mark={'delta':-1})
            view=observe(st,'a',include_previews=False);hand=view['sides']['a']['hand']
            view['pending_choice']={'kind':'kit','side':'a','choices':[{'id':c['instance_id'],'card':deepcopy(c)}for c in hand[:4]]}
            ids=[x['instance_id']for x in hand[:4]]
            actions=[{'type':k,'character_id':team['order'][0]}for k in ('attack','ultimate','exit_ultimate')]
            actions += [{'type':'end_turn'},{'type':'play_card','card_id':ids[0],'option_id':'boost','target_id':'b:player'},
                        {'type':'choose','choice_id':ids[1]},{'type':'mulligan','card_ids':ids[:3]},
                        {'type':'choose_cards','card_ids':ids[1:3]},
                        {'type':'choose_deck_order','card_ids':ids[:2],'bottom_card_id':ids[2]},
                        {'type':'choose_deck_order','card_ids':ids[2:],'bottom_card_ids':ids[:2]}]
            views.append(view);acts.append(actions)
        with DomainSoACandidates(device='cpu',allow_cpu_oracle=True)as enc:self.compare(views,acts,enc.encode(**numeric(views,acts)))

    def test_live_noncontiguous_errors_and_private_unreferenced_rows(self):
        st=state(10);view=observe(st,'a',include_previews=False);actions=[a for a in legal_actions(st,'a')if a['type']!='concede']
        f=numeric([view],[actions]);noncontig={}
        for k,v in f.items():
            storage=torch.empty((2,)+tuple(v.shape[1:]),dtype=v.dtype);noncontig[k]=storage[::2];noncontig[k].copy_(v)
        with DomainSoACandidates(device='cpu',allow_cpu_oracle=True)as enc:
            p=enc.prepare(**noncontig);self.compare([view],[actions],enc.launch(p))
            noncontig['cards'][0,-1]=float('nan');self.compare([view],[actions],enc.launch(p))
            noncontig['prior_errors'][0]=32;out=enc.launch(p);self.assertEqual(int(out['errors'][0]),32);self.assertFalse(out['valid'].any());self.assertFalse(out['features'].any())
            noncontig['prior_errors'].zero_();noncontig['actions'][0,0,0]=9
            self.assertEqual(int(enc.launch(p)['errors'][0]),1)

    def test_windows_preserve_complete_candidate_order(self):
        from tests.test_duel_v2_domain_soa_codec import _fourteen_initial_states
        views=[observe(st,'a',include_previews=False)for st in _fourteen_initial_states()]
        acts=[[a for a in legal_actions(st,'a')if a['type']!='concede']for st in _fourteen_initial_states()]
        f=numeric(views,acts);size=f['actions'].shape[1]
        with DomainSoACandidates(device='cpu',allow_cpu_oracle=True)as enc:
            windows=[];masks=[]
            for start in range(0,size,3):
                out=enc.encode(**f,action_start=start,action_width=min(3,size-start))
                self.assertEqual(out['action_start'],start);self.assertEqual(out['descriptor_capacity'],size)
                self.assertFalse(out['errors'].any());windows.append(out['features'].clone());masks.append(out['valid'].clone())
            joined={'features':torch.cat(windows,1),'valid':torch.cat(masks,1),'errors':out['errors'],
                    'full_duel_qualified':False,'legal_actions_available':False}
            self.compare(views,acts,joined)
            for start,width in ((-1,1),(size,1),(0,size+1)):
                with self.assertRaises(ValueError):enc.prepare(**f,action_start=start,action_width=width)

    def test_reject_hidden_references_capacity_holes_and_duplicates(self):
        st=state(11);view=observe(st,'a',include_previews=False);ids=[c['instance_id']for c in view['sides']['a']['hand'][:2]]
        actions=[{'type':'mulligan','card_ids':ids}];f=numeric([view],[actions])
        with DomainSoACandidates(device='cpu',allow_cpu_oracle=True)as enc:
            self.compare([view],[actions],enc.encode(**f))
            for mutation in ('hidden','overflow','hole','duplicate'):
                g={k:v.clone()for k,v in f.items()}
                if mutation=='hidden':g['cards'][0,0,0]=-1
                if mutation=='overflow':g['counts'][0]=2
                if mutation=='hole':g['selected'][0,0,0]=-1
                if mutation=='duplicate':g['selected'][0,0,1]=g['selected'][0,0,0]
                out=enc.encode(**g);self.assertEqual(int(out['errors'][0]),1,mutation);self.assertFalse(out['features'].any())

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_graph_live_tables_and_errors(self):
        views=[];acts=[]
        for i in range(400):
            view=observe(state(i%7+1,a=seven.KEYS[i%7]),'a',include_previews=False)
            views.append(view);acts.append([{'type':'end_turn'},{'type':'play_card','card_id':view['sides']['a']['hand'][0]['instance_id']}])
        f=numeric(views,acts,device='cuda',capacity=800)
        with DomainSoACandidates(device='cuda')as enc:
            full=enc.encode(**f);self.assertEqual(tuple(full['features'].shape),(400,800,1576));self.compare(views,acts,full)
            del full
            p=enc.prepare(**f,action_width=16);enc.launch(p);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=enc.launch(p)
            graph.replay();torch.cuda.synchronize();self.compare(views,acts,out)
            f['cards'][:,0,12]=3;f['prior_errors'][1]=64;f['actions'][2,0,0]=9
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(out['errors'][:3].cpu().tolist(),[0,64,1]);self.assertAlmostEqual(float(out['features'][0,1,-1]),.3,places=6)
            self.assertFalse(out['features'][1:3].any())
            self.assertEqual(tuple(out['features'].shape),(400,16,1576))
            self.assertFalse(out['valid'][:,2:].any())
