"""Full seven-column packing parity; never complete-duel qualification."""
from copy import deepcopy
import shutil
import unittest
import torch
from app.modules.card_game.engine.duel_v2.projection import observe
from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.rl import seven_lineup
from app.modules.card_game.rl.batched_duel.domain_soa_observation import DomainSoAObservation,FEATURE_NAMES
from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoALayout
from tests.test_duel_v2_domain_soa_character_queries import state,pack


def cpu(layout):
    if not(shutil.which('clang++')or shutil.which('g++')):raise unittest.SkipTest('Explicit C++ oracle compiler unavailable')
    return DomainSoAObservation(layout,device='cpu',allow_cpu_oracle=True)


class ObservationTest(unittest.TestCase):
    def compare(self,states,viewers,result):
        self.assertEqual(result['errors'].cpu().tolist(),[0]*len(states))
        for row,(st,viewer)in enumerate(zip(states,viewers)):
            gold,_=seven_lineup.encode(observe(st,('a','b')[viewer],include_previews=False),[])
            actual=result['features'][row].cpu();gold=torch.tensor(gold,dtype=torch.float32)
            bad=[(k,float(a),float(b))for k,a,b in zip(FEATURE_NAMES,actual,gold)if a!=b]
            self.assertEqual(bad,[],msg=str((row,st['phase'],st['turn'],bad[:15])))

    def test_all_initial_seven_and_both_observers(self):
        from tests.test_duel_v2_domain_soa_codec import _fourteen_initial_states
        states=[s for s in _fourteen_initial_states()for _ in range(2)]
        viewers=[i%2 for i in range(len(states))];layout,fields=pack(states)
        with cpu(layout)as enc:
            out=enc.observe(fields,torch.tensor(viewers),torch.ones(len(states),dtype=torch.bool))
            self.compare(states,viewers,out)
            self.assertEqual(out['features'].shape[1],3479)
            self.assertFalse(out['full_duel_qualified']);self.assertFalse(out['action_features_available'])
            self.assertNotIn('_private_card_fields',out)

    def test_natural_playing_choices_effects_and_summons(self):
        from tests.test_duel_v2_domain_soa_codec import _natural_states
        states=_natural_states()[::7];viewers=[i%2 for i in range(len(states))]
        layout,fields=pack(states)
        with cpu(layout)as enc:self.compare(states,viewers,enc.observe(fields,torch.tensor(viewers),torch.ones(len(states),dtype=torch.bool)))

    def test_transformed_cards_mimic_visible_hands_and_integer_clocks(self):
        states=[]
        st=state(25,a='zhenhong');st['sides']['a']['characters']['zhenhong']['awakened']=True;states.append(st)
        st=state(26,a='lingke');st['sides']['a']['characters']['lingke']['shape']='K03';st['sides']['a']['ap']=0;states.append(st)
        st=state(27,a='murk');team=st['sides']['a'];h=team['characters']['anhunqu'];h['flags']['mimic']={'id':'adler'};h['shield']=4;team['ap']=0
        eid=team['characters']['canhong']['entity_id'];team['used']['canhong_dot_attack_turn:'+eid]=st['turn'];states.append(st)
        for st in states:
            for side in ('a','b'):
                team=st['sides'][side];team['revealed_ids']=[team['hand'][0]['instance_id']]
                team['hand'][1]['copy']=True
        states=[s for s in states for _ in range(2)];viewers=[i%2 for i in range(len(states))]
        layout,fields=pack(states)
        with cpu(layout)as enc:self.compare(states,viewers,enc.observe(fields,torch.tensor(viewers),torch.ones(len(states),dtype=torch.bool)))

    def test_finished_states_stop_passive_substitution_and_panel_override(self):
        states=[]
        st=state(25,a='zhenhong');st['sides']['a']['characters']['zhenhong']['awakened']=True
        st.update(phase='finished',winner='a',reason='player_hp');states.extend((deepcopy(st),deepcopy(st)))
        st=state(26,a='murk');st['sides']['a']['characters']['adler']['shield']=7
        st.update(phase='finished',winner='a',reason='player_hp');states.extend((deepcopy(st),deepcopy(st)))
        layout,fields=pack(states);viewers=[0,1,0,1]
        with cpu(layout)as enc:self.compare(states,viewers,enc.observe(fields,torch.tensor(viewers),torch.ones(4,dtype=torch.bool)))

    def test_late_clocks_and_inspection_choices_match_both_views(self):
        states=[]
        for turn in (0,6,13,20):
            st=state(60+turn,a='murk',b='lingke');st['turn']=turn
            st['sides']['a']['turn_count']=turn//2;st['sides']['b']['turn_count']=max(0,turn//2-1)
            team=st['sides']['a'];cards=team['deck'][:2];team['deck']=team['deck'][2:]
            st['phase']='choice';st['pending_choice']={'kind':'inspect_top','side':'a','prompt':'检视','cards':cards}
            states.extend((deepcopy(st),deepcopy(st)))
        viewers=[i%2 for i in range(len(states))];layout,fields=pack(states)
        with cpu(layout)as enc:self.compare(states,viewers,enc.observe(fields,torch.tensor(viewers),torch.ones(len(states),dtype=torch.bool)))

    def test_hidden_cards_deck_order_and_rng_do_not_change_public_vector(self):
        st=state(64);layout,fields=pack([st])
        with cpu(layout)as enc:
            vi=torch.tensor([0]);active=torch.ones(1,dtype=torch.bool)
            baseline=enc.observe(fields,vi,active)['features'].clone()
            hidden=int(fields['zone_card_index'][0,1,0,0]);fields['card_template'][0,hidden]=fields['card_template'][0,int(fields['zone_card_index'][0,1,1,0])]
            fields['rng'].add_(1000)
            n=int(fields['zone_count'][0,1,1])
            fields['zone_card_index'][0,1,1,:n].copy_(fields['zone_card_index'][0,1,1,:n].flip(0))
            out=enc.observe(fields,vi,active)
            self.assertEqual(int(out['errors'][0]),0)
            self.assertTrue(torch.equal(out['features'],baseline))

    def test_live_noncontiguous_input_and_failed_rows(self):
        states=[state(31),state(32)];layout,raw=pack(states);fields={}
        for name,value in raw.items():
            storage=torch.empty((4,)+tuple(value.shape[1:]),dtype=value.dtype);fields[name]=storage[::2];fields[name].copy_(value)
        with cpu(layout)as enc:
            prepared=enc.prepare(fields,torch.tensor([0,1]),torch.ones(2,dtype=torch.bool))
            self.compare(states,[0,1],enc.launch(prepared))
            fields['row_error'][0]=64;fields['catalog_digest'][1,0]^=1
            out=enc.launch(prepared)
            self.assertEqual(int(out['errors'][0]),64)
            self.assertNotEqual(int(out['errors'][1]),0)
            self.assertFalse(bool(out['features'].any()))

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA full observation required')
    def test_cuda_400_graph_matches_formal_encoder(self):
        from tests.test_duel_v2_domain_soa_codec import _natural_states
        base=_natural_states()[::7]
        extra=state(25,a='zhenhong');extra['sides']['a']['characters']['zhenhong']['awakened']=True
        base.extend((deepcopy(extra),deepcopy(extra)))
        extra=state(26,a='lingke');extra['sides']['a']['characters']['lingke']['shape']='K03';extra['sides']['a']['ap']=0
        base.extend((deepcopy(extra),deepcopy(extra)))
        extra=state(27,a='murk');team=extra['sides']['a'];team['characters']['anhunqu']['flags']['mimic']={'id':'adler'};team['characters']['anhunqu']['shield']=4
        team['used']['canhong_dot_attack_turn:'+team['characters']['canhong']['entity_id']]=extra['turn']
        team['revealed_ids']=[team['hand'][0]['instance_id']]
        base.extend((deepcopy(extra),deepcopy(extra)))
        terminal=state(71,a='zhenhong');terminal['sides']['a']['characters']['zhenhong']['awakened']=True;terminal.update(phase='finished',winner='a',reason='player_hp')
        base.extend((deepcopy(terminal),deepcopy(terminal)))
        terminal=state(72,a='murk');terminal['sides']['a']['characters']['adler']['shield']=7;terminal.update(phase='finished',winner='a',reason='player_hp')
        base.extend((deepcopy(terminal),deepcopy(terminal)))
        states=[base[i%len(base)]for i in range(400)]
        viewers=[i%2 for i in range(400)];layout,fields=pack(states,'cuda')
        with DomainSoAObservation(layout,device='cuda')as enc:
            prepared=enc.prepare(fields,torch.tensor(viewers,dtype=torch.int64,device='cuda'),torch.ones(400,dtype=torch.bool,device='cuda'))
            enc.launch(prepared);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=enc.launch(prepared)
            graph.replay();torch.cuda.synchronize()
            chosen=[i*len(base)//14 for i in range(14)]+list(range(len(base)-10,len(base)))
            self.compare([states[i]for i in chosen],[viewers[i]for i in chosen],{'features':out['features'][chosen],'errors':out['errors'][chosen]})
            fields['row_error'][0]=73;fields['catalog_digest'][1,0]^=1
            fields['turn'][2]=13
            prepared.viewer_source[3]=1-viewers[3]
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(int(out['errors'][0]),73);self.assertNotEqual(int(out['errors'][1]),0)
            self.assertFalse(bool(out['features'][:2].any()))
            later=deepcopy(states[2]);later['turn']=13
            self.compare([later,states[3]],[viewers[2],1-viewers[3]],{'features':out['features'][2:4],'errors':out['errors'][2:4]})


if __name__=='__main__':unittest.main()
