"""Direct immutable character queries against current formal rule functions."""
from copy import deepcopy
import shutil
import unittest
import torch

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.engine.duel_v2.state import attack_value, collapse_max, damage_immune, damage_limit
from app.modules.card_game.engine.duel_v2.equipment import permanent_attack, unarmed_max_hp, equip_weapon
from app.modules.card_game.engine.duel_v2.escalation import energy_cap, harmony_cap
from app.modules.card_game.engine.duel_v2.modifiers import compatibility_flags, resistance_values, contribution
from app.modules.card_game.rl.batched_duel.domain_soa_codec import encode_domain_states
from app.modules.card_game.rl.batched_duel.domain_soa_schema import ARRAY_NAMES, DomainSoALayout
from app.modules.card_game.rl.batched_duel.domain_soa_character_queries import (
    DomainCharacterQueries, OUTPUT_NAMES, ATTRIBUTES, INVALID,
)


def state(seed=1, a='starter', b='weave-rush'):
    decks={d['id']:d for d in STARTER_DECKS}
    return new_game(seed=seed,decks={'a':decks[a],'b':decks[b]},skip_mulligan=True)


def pack(states, device='cpu', layout=None):
    layout=layout or DomainSoALayout(game_capacity=len(states))
    arrays=encode_domain_states(states,layout,presentation='drop')
    return layout,{name:torch.as_tensor(getattr(arrays,name).copy(),device=device)for name in ARRAY_NAMES}


def expected(st, side, cid):
    h=st['sides'][side]['characters'][cid]
    flags=compatibility_flags(h,st,side);resistance=resistance_values(h)
    limit=damage_limit(st,side,cid)
    values=[attack_value(h,st,side),permanent_attack(h),unarmed_max_hp(h),energy_cap(st,h),harmony_cap(st),
            collapse_max(h),damage_immune(st,side,cid),0 if limit is None else limit,limit is not None,
            flags.get('atk_buff')or 0,flags.get('atk_debuff')or 0,flags.get('form_attack')or 0,
            flags.get('next_attack')or 0]
    values += [sum(e['amount']for e in flags.get('timed_attack')or[]if e['left']==left)for left in (1,2)]
    values += [resistance.get(a)or 0 for a in ATTRIBUTES]
    return torch.tensor(values,dtype=torch.float64)


def backend(layout,device='cpu'):
    if device=='cpu'and not(shutil.which('clang++')or shutil.which('g++')):
        raise unittest.SkipTest('Explicit C++ oracle compiler unavailable')
    return DomainCharacterQueries(layout,device=device,allow_cpu_oracle=device=='cpu')


class CharacterQueryTest(unittest.TestCase):
    def compare(self,states,out):
        self.assertEqual(out['errors'].cpu().tolist(),[0]*len(states))
        if 'player_resistances' in out:
            gold=torch.tensor([[[resistance_values(st['sides'][side]).get(a)or 0 for a in ATTRIBUTES]for side in ('a','b')]for st in states],dtype=torch.float64)
            self.assertTrue(torch.equal(out['player_resistances'].cpu(),gold))
        for row,st in enumerate(states):
            for s,side in enumerate(('a','b')):
                for c,cid in enumerate(st['sides'][side]['characters']):
                    gold=expected(st,side,cid);got=out['values'][row,s,c].cpu()
                    self.assertTrue(torch.equal(got,gold),msg=str((row,side,cid,[(k,float(a),float(b))for k,a,b in zip(OUTPUT_NAMES,got,gold)if a!=b])))

    def test_seven_initial_and_escalation_states(self):
        states=[]
        for deck in STARTER_DECKS:
            for turn in (1,13,20):
                st=state(turn,deck['id'],deck['id']);st['turn']=turn;states.append(st)
        layout,fields=pack(states)
        with backend(layout)as q:
            self.compare(states,q.query(fields,torch.ones(len(states),dtype=torch.bool)))
            self.assertFalse(q.supports_full_duel_rules)

    def test_natural_trajectories_include_effects_and_nested_choices(self):
        from tests.test_duel_v2_domain_soa_codec import _natural_states
        states = _natural_states()[::7]
        layout,fields = pack(states)
        self.assertGreater(len(states),40)
        with backend(layout)as q:self.compare(states,q.query(fields,torch.ones(len(states),dtype=torch.bool)))

    def test_modifiers_scopes_layers_conditions_and_compatibility(self):
        st=state(a='murk',b='zhenhong');st['turn']=15;st['sides']['a']['turn_count']=4
        team=st['sides']['a'];cid=team['order'][0];h=team['characters'][cid]
        h['awakened']=True;h['energy']=3;team['front']=cid
        h['flags'].update(atk_buff=3,atk_buff_expiring={'amount':1,'until':5},atk_debuff=1,
                          form_attack=2,next_attack=2,timed_attack=[{'amount':-1,'left':1},{'amount':2,'left':2}],
                          damage_immunity_until=5)
        for i,(condition,args)in enumerate((('source_front',{}),('source_alive',{}),('source_awakened',{}),
                  ('enemy_turn',{}),('resource_at_least',{'resource':'energy','amount':2}),
                  ('card_owner_other',{}),('damage_attribute',{'attribute':'光'}))):
            effect=contribution('case'+str(i),'attack.panel',.5+i,source_entity_id=h['entity_id'],conditions=[{'id':condition,'args':args}])
            h.setdefault('effects',[]).append(effect)
        h['effects'] += [contribution('limit','damage.limit',2,op='set',source_entity_id=h['entity_id']),
                        contribution('set','attack.panel',2,op='set',layer=-1,source_entity_id=h['entity_id']),
                        contribution('mul','attack.panel',1.5,op='multiply',source_entity_id=h['entity_id']),
                        contribution('resist','damage.resistance',2,source_entity_id=h['entity_id'],conditions=[{'id':'damage_attribute','args':{'attribute':'光'}}]),
                        contribution('timed','attack.panel',-2,source_entity_id=h['entity_id'],expiry={'side':'a','phase':'turn_start','turn':6})]
        states=[st]
        later=deepcopy(st);later['sides']['a']['turn_count']=5;states.append(later)
        later=deepcopy(st);later['sides']['a']['characters'][cid]['hp']=0;states.append(later)
        layout,fields=pack(states)
        with backend(layout)as q:self.compare(states,q.query(fields,torch.ones(len(states),dtype=torch.bool)))

    def test_all_scopes_stable_layers_and_player_holders(self):
        states=[]
        for scope in ('self','allies','other_allies','enemies','all'):
            st=state(31,a='murk',b='quick-rush')
            team=st['sides']['a'];cid=team['order'][0];h=team['characters'][cid]
            team['effects']=[contribution('team.set','attack.panel',2,op='set',scope=scope,
                                         source_entity_id=team['entity_id'],layer=-1),
                            contribution('team.resistance','damage.resistance',2,source_entity_id=team['entity_id'],
                                         conditions=[{'id':'damage_attribute','args':{'attribute':'光'}}])]
            h['effects']=[contribution('hero.add','attack.panel',.5,scope=scope,source_entity_id=h['entity_id']),
                          contribution('hero.mul','attack.panel',1.5,op='multiply',scope=scope,source_entity_id=h['entity_id']),
                          contribution('hero.resist','damage.resistance',.25,scope=scope,source_entity_id=h['entity_id'],
                                       conditions=[{'id':'damage_attribute','args':{'attribute':'咒'}}])]
            states.append(st)
        layout,fields=pack(states)
        with backend(layout)as q:self.compare(states,q.query(fields,torch.ones(len(states),dtype=torch.bool)))

    def test_weapon_auras_panel_override_and_delay(self):
        states=[]
        for a in ('murk','weave-rush','quick-rush'):
            st=state(a=a,b='starter');team=st['sides']['a']
            cid='adler'if a=='murk'else'bohe'
            h=team['characters'][cid];h['shape']='D07'if cid=='adler'else'B08'
            h['shield']=3;st['active_side']='b';team['front']=cid
            team['front_debuff']['delay']={'left':2,'slow':2,'slow_turn':st['turn'],'by':'b'}
            states.append(st)
        layout,fields=pack(states)
        with backend(layout)as q:self.compare(states,q.query(fields,torch.ones(len(states),dtype=torch.bool)))

    def test_live_inputs_noncontiguous_prior_error_and_catalog(self):
        states=[state(12),state(13)]
        layout,raw=pack(states);fields={}
        for name,value in raw.items():
            allocation=torch.empty((4,)+tuple(value.shape[1:]),dtype=value.dtype)
            fields[name]=allocation[::2];fields[name].copy_(value)
        with backend(layout)as q:
            prepared=q.prepare(fields,torch.ones(2,dtype=torch.bool))
            out=q.launch(prepared);self.compare(states,out)
            fields['row_error'][0]=71;fields['catalog_digest'][1,0]^=1
            q.launch(prepared)
            self.assertEqual(out['errors'].tolist(),[71,INVALID])
            self.assertFalse(bool(out['values'].any()))

    def test_corrupt_counts_strings_and_numeric_bounds_fail_closed(self):
        states=[state(i)for i in range(6)]
        layout,fields=pack(states)
        fields['holder_effect_count'][0,0]=layout.max_effects+1
        from app.modules.card_game.rl.batched_duel.domain_soa_schema import CHARACTER_STR_FIELDS
        shape=CHARACTER_STR_FIELDS.index('shape')
        fields['character_str_presence'][1,0,0,shape]=2
        fields['character_str_slot'][1,0,0,shape]=2147483647
        from app.modules.card_game.rl.batched_duel.domain_soa_schema import FLAG_INT_FIELDS
        at=FLAG_INT_FIELDS.index('atk_buff')
        fields['flag_int_presence'][2,1,at]=2
        fields['flag_int_value'][2,1,at]=2147483647
        fields['row_error'][3]=61
        fields['string_bytes_used'][4]=layout.max_string_bytes+1
        fields['player_character_count'][5,0]=-1
        active=torch.ones(6,dtype=torch.bool);active[3]=False
        with backend(layout)as q:
            out=q.query(fields,active)
            self.assertTrue(bool((out['errors']!=0).all()))
            self.assertEqual(int(out['errors'][3]),61)
            self.assertFalse(bool(out['values'].any()))

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA character queries required')
    def test_cuda_400_natural_roots_and_graph(self):
        from tests.test_duel_v2_domain_soa_codec import _natural_states
        base=_natural_states()[::7]
        states=[base[i%len(base)]for i in range(400)]
        layout,fields=pack(states,'cuda')
        with backend(layout,'cuda')as q:
            prepared=q.prepare(fields,torch.ones(400,dtype=torch.bool,device='cuda'))
            q.launch(prepared);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=q.launch(prepared)
            graph.replay();torch.cuda.synchronize()
            chosen=[(i*len(base))//14 for i in range(14)]
            self.compare([states[i]for i in chosen],{'errors':out['errors'][chosen],'values':out['values'][chosen],'player_resistances':out['player_resistances'][chosen]})
            fields['row_error'][0]=79;fields['catalog_digest'][1,0]^=1
            fields['turn'][2]=13
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(out['errors'][:2].cpu().tolist(),[79,INVALID])
            self.assertFalse(bool(out['values'][:2].any()))
            self.assertTrue(bool((out['errors'][2:]==0).all()))
            later=deepcopy(states[2]);later['turn']=13
            self.compare([later],{'errors':out['errors'][2:3],'values':out['values'][2:3]})


if __name__=='__main__':unittest.main()
