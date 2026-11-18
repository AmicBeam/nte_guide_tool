"""Exe's resource model, summon buffs, arc refinements and test-only access."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import unittest

from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import normalize_axis_payload
from tests.test_solo_room_flow import RoomFlowTestCase

ROOT=Path(__file__).resolve().parents[1]
CID='char_exe'
ARC='arc_exe_dotted_horizon'


class ExeTest(unittest.TestCase):
    def axis(self, steps, nodes=(), initial=None, others=(), arc='', refinement=1):
        return {'team':[dict(slot=0,character_id=CID,arc_id=arc,arc_refinement=refinement,cartridge_id='',awakening_nodes=list(nodes),
                            skill_levels=dict(basic=1,skill=1,ultimate=1,support=1)),*others],
                'steps':[dict(id=str(i),slot=slot,action_id=action,start_tick=tick) for i,(slot,action,tick)in enumerate(steps)],
                'options':{'switch_loss_ticks':0}}

    def sim(self, axis, initial=None):
        catalog=deepcopy(load_shaft_catalog())
        if initial is not None:next(c for c in catalog['characters']if c['id']==CID)['initial_personal_resources'].update(initial)
        result=subprocess.run(['node',str(ROOT/'scripts/shaft_compute.js')],input=json.dumps(dict(axis=axis,catalog=catalog)),text=True,capture_output=True,check=True)
        payload=json.loads(result.stdout);self.assertTrue(payload['ok']);return payload['result']

    def own(self, pairs, **kwargs):
        initial=kwargs.pop('initial',None)
        return self.sim(self.axis([(0,'action_exe_'+name,tick)for name,tick in pairs],**kwargs),initial)

    def buffs(self, detail):return {b['rule_id']:b for b in detail['applied_buffs']}

    def test_catalog_exact_levels_durations_and_no_bond(self):
        c=load_shaft_catalog();char=next(x for x in c['characters']if x['id']==CID)
        self.assertEqual(char['base_stats'],dict(atk=676,hp=15271,**{'def':884}))
        self.assertEqual((char['adaptation'],char['element'],char['energy_capacity']),('等离子','暗',120))
        self.assertNotIn('bond_bonus',char)
        actions={a['name']:a for a in c['actions_by_character'][CID]}
        for name,atk,hits,ticks in [('a1',.537,1,6),('a2',.916,2,6),('a3',1.047,3,7),('a4',2.228,7,13),
                                    ('a5',.786,6,8),('z',1.032,1,7),('e',3.849,6,12),('q',7.7,19,0),('援护',3.08,3,15),('数字残像',.924,1,0)]:
            with self.subTest(name=name):
                a=actions[name];self.assertAlmostEqual(a['multipliers']['atk'],atk);self.assertEqual(a['hit_count'],hits);self.assertEqual(a['duration_ticks'],ticks)
                self.assertEqual(len(a['atk_multipliers_by_level']),13)
        self.assertEqual(actions['q']['energy_cost'],120)
        self.assertTrue(actions['q']['is_time_stop_zero'])
        self.assertEqual(actions['数字残像']['personal_resource_cost'],{'残像额度':1})
        self.assertEqual(next(a for a in c['arcs']if a['adaptation']=='等离子')['id'],ARC)

    def test_frame_gain_score_progress_and_export(self):
        r=self.own([('e',0),('a5',20),('clone',35)])
        first=r['details'][0]['personal_resources_after']
        self.assertAlmostEqual(first['帧数'],1.356)
        self.assertEqual(first['残像额度'],1)
        self.assertEqual(r['details'][-1]['personal_resources_after']['残像额度'],0)
        charges=[e for e in r['personal_resource_state_events']if e['resource']=='残像额度']
        self.assertEqual(charges[-1]['value'],0)
        self.assertGreater(r['details'][-1]['personal_resources_after']['帧数'],first['帧数'])
        self.assertTrue(r['personal_resource_state_events'])

    def test_interrupt_keeps_exact_levels_and_requires_finisher(self):
        axis=self.axis([(0,'action_exe_a4',0)])
        axis['team'][0]['skill_levels']['basic']=10
        axis['steps'][0].update(interrupted=True,interrupt_duration_ticks=3,interrupt_hit_count=3)
        detail=self.sim(axis)['details'][0]
        expected=(.492*2+.68)
        self.assertAlmostEqual(detail['formula_parts']['base'] / detail['panel']['atk'],expected)
        for hits,expected_charges in [(0,0),(5,0),(6,1)]:
            axis=self.axis([(0,'action_exe_a5',0)])
            axis['steps'][0].update(interrupted=True,interrupt_duration_ticks=5,interrupt_hit_count=hits)
            result=self.sim(axis,{'帧数':50})
            self.assertEqual(result['details'][0]['personal_resources_after']['残像额度'],expected_charges)

    def test_score_decay_and_independent_three_resonance(self):
        for nodes,resonances,decays in [((),[],True),((),[3],False),([1,2,3],[],True)]:
            axis=self.axis([(0,'action_none_exe',100)],nodes=nodes)
            axis['team'][0]['awakening_resonances']=resonances
            r=self.sim(axis,{'评分':600})
            score=r['resources_by_slot'][0]['personal_resources']['评分']
            self.assertEqual(score<600,decays)

    def test_a_b_and_omega_bonuses(self):
        for score,nodes,expected_atk,expected_dark in [(0,[2],.1,0),(500,[1,2],.2,.2),(3000,[2],.3,0),(5000,[1,2,3,4,5,6],.5,.2)]:
            d=self.own([('a1',0)],nodes=nodes,initial={'评分':score})['details'][0]
            buffs=self.buffs(d)
            self.assertAlmostEqual(buffs['character_exe_b']['effects']['atk_pct']+buffs.get('character_exe_b_omega',{}).get('effects',{}).get('atk_pct',0),expected_atk)
            self.assertAlmostEqual(buffs.get('character_exe_a',{}).get('effects',{}).get('element_dmg_暗',0),expected_dark)

    def test_summon_per_character_cap_and_equipped_arc_seeds(self):
        c=load_shaft_catalog();characters={x['name']:x['id']for x in c['characters']};arcs={x['name']:x['id']for x in c['arcs']}
        for gear_names,expected in [([],0),(['茶花会'],2),(['海特洛的安宁'],2),(['茶花会','海特洛的安宁'],4)]:
            others=[dict(slot=i+1,character_id=characters[name],arc_id=arcs[gear],cartridge_id='')for i,(name,gear)in enumerate(zip(['阿德勒','浔'],gear_names))]
            d=self.own([('a1',0)],others=others)['details'][0]
            self.assertEqual(self.buffs(d).get('character_exe_overclock',{}).get('stack_count',0),expected)
        # The same wearer cannot add two more layers after their seeded arc layers.
        teammate=dict(slot=1,character_id=characters['阿德勒'],arc_id=arcs['茶花会'],cartridge_id='')
        adler_action=next(a['id']for a in c['actions_by_character'][teammate['character_id']]if a['action_type']=='E')
        r=self.sim(self.axis([(1,adler_action,0),(1,adler_action,30),(0,'action_exe_a1',60)],others=[teammate]))
        self.assertEqual(self.buffs(r['details'][-1])['character_exe_overclock']['stack_count'],2)

    def test_d_summon_bonus_and_c_resistance_expiry(self):
        r=self.own([('e',0),('a1',20),('a1',150)],nodes=[3,4])
        self.assertIn('character_exe_c',self.buffs(r['details'][1]))
        self.assertNotIn('character_exe_c',self.buffs(r['details'][2]))
        self.assertAlmostEqual(self.buffs(r['details'][1])['character_exe_d_team']['effects']['action_dmg'],.1)
        self.assertNotIn('character_exe_d_clone',self.buffs(r['details'][1]))

    def test_clone_cooldown_floor_and_awakened_burst(self):
        r=self.own([('e',0),('clone',50),('e',70)],initial={'残像额度':30})
        self.assertTrue(any('CD' in w for w in r['details'][-1]['warnings']))
        axis=self.axis([(0,'action_exe_e',0),(0,'action_exe_clone',50),(0,'action_exe_e',70)])
        axis['steps'][1]['repeat']=20
        r=self.sim(axis,{'残像额度':30})
        self.assertFalse(any('CD' in w for w in r['details'][-1]['warnings']))
        r=self.own([('q',0),('clone',10),('a1',200)],nodes=[5],initial={'残像额度':2})
        hits=[e for e in r['periodic_damage_events']if e.get('action_id')=='action_exe_clone_boom']
        self.assertEqual(len(hits),1)
        self.assertGreater(hits[0]['damage'],0)
        r=self.own([('clone',0)],nodes=[5],initial={'残像额度':2})
        self.assertFalse(any(e.get('action_id')=='action_exe_clone_boom'for e in r['periodic_damage_events']))

    def test_f_healing_and_empty_hit(self):
        r=self.own([('a1',0)],nodes=[6])
        self.assertTrue(any(e['source']=='Exe·帧数回复'for e in r['healing_events']))
        axis=self.axis([(0,'action_exe_a1',0)],nodes=[6]);axis['steps'][0].update(interrupted=True,interrupt_duration_ticks=1,interrupt_hit_count=0)
        r=self.sim(axis);self.assertEqual(r['details'][0]['personal_resources_after']['帧数'],0)
        self.assertFalse(r['healing_events'])

    def test_display_only_reaction_tail_does_not_advance_score_past_axis(self):
        c=load_shaft_catalog()
        soul=next(x for x in c['characters'] if x['element']=='魂')
        support=next(a['id']for a in c['actions_by_character'][soul['id']]if a['action_type']=='援护')
        axis=self.axis([(0,'action_none_exe',0),(1,support,2)],others=[dict(slot=1,character_id=soul['id'],arc_id='',cartridge_id='')])
        r=self.sim(axis,{'评分':600})
        self.assertTrue(r['reaction_effects'])
        self.assertTrue(all(e['tick']<=r['summary']['duration_ticks']for e in r['personal_resource_state_events']))

    def test_five_refinements_and_loop_resource_normalization(self):
        refs=load_shaft_catalog()['arc_refinements']['arcs'][ARC]['levels']
        for i in range(1,6):
            level=refs[str(i)];self.assertAlmostEqual(level['panel_modifiers']['atk_pct'],.28+.07*(i-1));self.assertAlmostEqual(level['panel_modifiers']['crit_dmg'],.76+.03*(i-1))
            self.assertAlmostEqual(level['buff_effects']['arc_exe_summon']['action_dmg'],.02+.005*(i-1))
        axis=self.axis([(0,'action_exe_a1',0)])
        axis['options'].update(loop_enabled=True,loop_initial_resources={CID:{'personal_resources':{'帧数':999,'评分':99999,'残像额度':4}}})
        self.assertEqual(normalize_axis_payload(axis)['options']['loop_initial_resources'][CID]['personal_resources'],{'帧数':100,'评分':5000,'残像额度':4})


class ExeAccessTest(RoomFlowTestCase):
    def test_catalog_save_market_and_detail_require_test_access(self):
        tokens={name:self._issue_login_and_get_token('exe-'+name)for name in ['formal','invited','test']}
        models=importlib.import_module('app.models')
        models.Player.update(shaft_invited=True).where(models.Player.player_uid=='exe-invited').execute()
        models.Player.update(shaft_test_whitelisted=True).where(models.Player.player_uid=='exe-test').execute()
        for name,token in tokens.items():
            character=next(c for c in self._get('/api/shaft/catalog',token=token)['characters'] if c['id']==CID)
            self.assertEqual(character['selection_disabled'],name!='test')
        axis={'team':[dict(slot=0,character_id=CID)],'steps':[dict(id='a',slot=0,action_id='action_exe_e',start_tick=0)]}
        for name in ['formal','invited']:
            response=self.client.post('/api/shaft/axes',json=dict(title='Exe denied',axis=axis,result={}),headers={'Authorization':'Bearer '+tokens[name]})
            self.assertEqual(response.status_code,400)
        saved=self._post('/api/shaft/axes',dict(title='Exe test',axis=axis,result=self._shaft_client_result(axis)),token=tokens['test'])
        published=self._post(f"/api/shaft/axes/{saved['id']}/publish",{},token=tokens['test'])
        for name,token in tokens.items():
            visible={x['id']for x in self._get('/api/shaft/market',token=token)['items']}
            self.assertEqual(published['id']in visible,name=='test')
            response=self.client.get(f"/api/shaft/axes/{published['id']}",headers={'Authorization':'Bearer '+token})
            self.assertEqual(response.status_code,200 if name=='test'else 400)
