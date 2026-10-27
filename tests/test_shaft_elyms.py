"""Test-only framework and deliberately incomplete character data."""
import importlib
import json
import subprocess
import unittest
from pathlib import Path

from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import normalize_axis_payload, simulate_shaft_axis
from tests.test_solo_room_flow import RoomFlowTestCase

CID = 'char_elyms'
ROOT = Path(__file__).resolve().parents[1]


class ElymsCatalogTest(unittest.TestCase):
    def axis(self, steps, nodes=(), arc='', others=(), **options):
        return {'team': [dict(slot=0, character_id=CID, arc_id=arc, cartridge_id='',
                             awakening_nodes=list(nodes), skill_levels=dict(basic=1, skill=1, ultimate=1, support=1)), *others],
                'steps': [dict(id=str(i), slot=slot, action_id=action, start_tick=tick)
                          for i, (slot, action, tick) in enumerate(steps)], 'options': options}

    def sim(self, steps, **kwargs):
        return simulate_shaft_axis(self.axis(steps, **kwargs))['result']

    def own(self, names, nodes=(), **kwargs):
        return self.sim([(0, 'action_elyms_'+name, tick) for name, tick in names], nodes=nodes, **kwargs)

    def effects(self, detail):
        return {b['rule_id']: b.get('effects', {}) for b in detail['applied_buffs']}

    def test_confirmed_catalog_and_missing_fields(self):
        c = load_shaft_catalog()
        character = next(x for x in c['characters'] if x['id'] == CID)
        self.assertEqual(character['adaptation'], '固态')
        self.assertEqual(character['base_stats'], {'atk':660,'hp':15392,'def':909})
        actions = {a['name']:a for a in c['actions_by_character'][CID]}
        self.assertEqual(len(actions),22)
        expected = {'a1':(.224,2),'a2':(.314,2),'a3':(.834,3),'a4':(.698,5),'a5':(.915,1),
                    '强a1':(.259,2),'强a2':(2.236,24),'强a3':(.788,4),'重击':(.449,4),
                    '瞄准':(.12,1),'下落':(.296,1),'闪反':(1,1),'e1':(1.999,6),'e2':(.998,6),
                    'e3':(1.502,6),'q群':(5.683,2),'q单':(4.01,5),'追忆群':(1.576,4),'追忆单':(1.752,4)}
        for name,(mult,hits) in expected.items():
            a=actions[name]
            self.assertAlmostEqual(a['multipliers']['atk'],mult)
            self.assertEqual(a['hit_count'],hits)
            self.assertEqual((a['energy_gain'],a['harmony'],a['stagger']),(0,0,0))
            if not name.startswith('追忆'):
                self.assertTrue(a['duration_pending'])
                self.assertFalse(a.get('is_time_stop_zero'))
        self.assertEqual(actions['e1']['harmony_on_start'],100)
        self.assertEqual(actions['e1']['cooldown_ticks'],160)
        self.assertEqual(actions['q群']['cooldown_ticks'],200)
        self.assertEqual(actions['援护']['multipliers']['atk'],0)
        self.assertEqual(c['formula_constants']['default_build_options'][CID]['curtain_bonus'],
                         {'value':8,'stat':'暴击','passive_type':'type3','fixed':True})
        self.assertEqual(next(a['id'] for a in c['arcs'] if a['adaptation']=='固态'),'arc_elyms_waiting_ringtone')
        ref=c['arc_refinements']['arcs']['arc_elyms_waiting_ringtone']
        self.assertEqual(ref['available_levels'],[1,5])
        self.assertEqual(set(ref['levels']),{'1','5'})

    def test_signal_multiplier_snapshot_reset_and_weak_action_checks(self):
        r=self.own([('skill2',0),('goodbye',10),('melee1',20),('ultraskilla1',30),('melee1',40),('xietongskilla1',50)])
        details=r['details']
        self.assertAlmostEqual(details[1]['formula_parts']['base_multiplier_factor'],3.5)
        self.assertAlmostEqual(details[3]['formula_parts']['base_multiplier_factor'],2)
        self.assertAlmostEqual(details[5]['formula_parts']['base_multiplier_factor'],2)
        repeated=self.own([('skill2',0),('goodbye',10),('ultraskilla1',20),('ultraskilla1',30),('xietongskilla1',40)])
        self.assertEqual(repeated['details'][-1]['formula_parts']['base_multiplier_factor'],1)
        self.assertTrue(any('当前形态' in w for w in details[2]['warnings']))
        self.assertFalse(any('当前形态' in w for w in details[4]['warnings']))
        self.assertTrue(any('后台' in w for w in details[5]['warnings']))
        expired=self.own([('skill2',0),('goodbye',170)])['details'][1]
        self.assertAlmostEqual(expired['formula_parts']['base_multiplier_factor'],1)

    def test_awakening_consumption_replacement_and_six_extra(self):
        r=self.own([('skill2',0),('goodbye',10),('ultraskilla1',20),('melee1',30),('ultraskilla1',40),('melee1',50)],nodes=range(1,7))
        d=r['details']
        self.assertAlmostEqual(d[1]['formula_parts']['base_multiplier_factor'],6)
        fx=self.effects(d[3]);self.assertAlmostEqual(fx['character_elyms_a']['crit_dmg'],.2)
        self.assertAlmostEqual(fx['character_elyms_d']['crit_dmg'],.2)
        # F remains active for all follow damage, even when this probe has no follow tag.
        fx=self.effects(d[5]);self.assertEqual(fx['character_elyms_a'],{})
        extras=[e for e in r['periodic_damage_events'] if e.get('action_id')=='action_elyms_six_extra']
        self.assertEqual(len(extras),1)
        self.assertEqual(extras[0]['atk_multiplier'],6)
        self.assertGreater(extras[0]['damage'],0)
        no_consume=self.own([('goodbye',0)],nodes=range(1,7))
        self.assertFalse(any(e.get('action_id')=='action_elyms_six_extra' for e in no_consume['periodic_damage_events']))

    def test_passive_defense_window_team_attack_and_initial_harmony(self):
        r=self.own([('skill1',0),('ultraskilla1',10),('melee1',20),('melee1',230)],nodes=[4,5],arc='arc_elyms_waiting_ringtone')
        self.assertAlmostEqual(r['details'][0]['formula_parts']['base_multiplier_factor'],2)
        self.assertEqual(self.own([('melee1',0)],nodes=[5])['resources_by_slot'][0]['initial_harmony'],100)
        self.assertEqual(self.own([('melee1',0)],nodes=[5],loop_enabled=True)['resources_by_slot'][0]['initial_harmony'],0)
        self.assertEqual(self.own([('melee1',0)])['resources_by_slot'][0]['initial_harmony'],0)
        fx=self.effects(r['details'][2])
        self.assertAlmostEqual(fx['character_elyms_q_atk']['flat_atk'],246)
        self.assertAlmostEqual(fx['character_elyms_talent2']['crit_dmg'],.1)
        self.assertNotIn('character_elyms_q_atk',self.effects(r['details'][3]))

    def test_follow_shape_shared_cooldown_quota_and_weapon_threshold(self):
        teammate=dict(slot=1,character_id='char_dd034941ef',arc_id='',cartridge_id='')
        steps=[(0,'action_elyms_ultraskilla1',0),(1,'action_none_dd034941ef',10),
               (0,'action_elyms_xietongskilla1',20),(0,'action_elyms_xietongskillb1',25),
               (0,'action_elyms_xietongskilla1',60),
               *[(0,'action_elyms_xietongskilla1',90+i*30) for i in range(5)]]
        r=self.sim(steps,others=[teammate],arc='arc_elyms_waiting_ringtone');d=r['details']
        self.assertTrue(any('额外激活' in w for w in d[3]['warnings']))
        self.assertTrue(any('CD' in w for w in d[3]['warnings']))
        self.assertFalse(any('后台' in w for w in d[2]['warnings']))
        self.assertTrue({'arc_elyms_team_def','arc_elyms_self_def'} <= {b['rule_id'] for b in d[2]['triggered_buffs']})
        self.assertAlmostEqual(self.effects(d[4])['arc_elyms_team_def']['def_ignore'],.12)
        self.assertTrue(any('最多使用6次' in w for w in d[-1]['warnings']))

    def test_energy_gain_talent_and_echo_do_not_recurse(self):
        catalog=load_shaft_catalog()
        chaos=next(a['id'] for a in catalog['actions_by_character']['char_d38b672525'] if a['name']=='援护')
        others=[dict(slot=1,character_id='char_dd034941ef',arc_id='',cartridge_id=''),
                dict(slot=2,character_id='char_d38b672525',arc_id='',cartridge_id='')]
        axis=self.axis([(1,'action_982c67944f',0),(2,chaos,30),(1,'action_none_dd034941ef',200)],others=others)
        axis['initial_energy']=0
        r=simulate_shaft_axis(axis)['result']
        light=[b for d in r['details'] for b in d['applied_buffs'] if b['rule_id']=='character_elyms_energy_light']
        self.assertTrue(light)
        self.assertTrue(any(b['rule_id']=='character_elyms_energy_light' for d in r['details'] for b in d['triggered_buffs']))
        axis['initial_energy']=1000
        capped=simulate_shaft_axis(axis)['result']
        self.assertFalse(any(b['rule_id']=='character_elyms_energy_light' for d in capped['details'] for b in d['applied_buffs']))
        echoes=[e for e in r['reaction_effects'] if e.get('is_echo')]
        self.assertEqual(len(echoes),1)
        original=next(e for e in r['reaction_effects'] if e['reaction']=='延滞' and not e.get('is_echo'))
        self.assertEqual(echoes[0]['start_tick']-original['start_tick'],75)

    def test_old_partial_entries_are_removed_and_self_check_keeps_missing_duration(self):
        axis=self.axis([(0,'action_elyms_melee3_1',0),(0,'action_elyms_melee1',10)])
        self.assertEqual([s['action_id'] for s in normalize_axis_payload(axis)['steps']],['action_elyms_melee1'])
        catalog=load_shaft_catalog()
        script="const s=require(process.argv[1]);const p=JSON.parse(require('fs').readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(s.inspectAxis(p.axis,p.catalog)));"
        result=subprocess.run(['node','-e',script,str(ROOT/'app/modules/shaft/static/js/shaft_self_check.js')],input=json.dumps(dict(axis=axis,catalog=catalog)),text=True,capture_output=True,check=True)
        self.assertTrue(any('时长未实装' in w for w in json.loads(result.stdout)))


class ElymsAccessTest(RoomFlowTestCase):
    def test_test_only_access_save_publish_and_market(self):
        models = importlib.import_module('app.models')
        regular = self._issue_login_and_get_token('elyms-regular')
        invited = self._issue_login_and_get_token('elyms-invited')
        tester = self._issue_login_and_get_token('elyms-tester')
        models.Player.update(shaft_invited=True).where(models.Player.player_uid == 'elyms-invited').execute()
        models.Player.update(shaft_test_whitelisted=True).where(models.Player.player_uid == 'elyms-tester').execute()
        for token in (None, regular, invited, tester):
            catalog = self._get('/api/shaft/catalog', token=token)
            character = next(c for c in catalog['characters'] if c['id'] == CID)
            self.assertEqual(character['selection_disabled'], token != tester)
        axis = {'team': [dict(slot=0, character_id=CID, arc_id='', cartridge_id='')],
                'steps': [dict(id='a', slot=0, action_id='action_elyms_melee1', start_tick=0)]}
        payload = dict(title='曳尔沐基础框架', axis=axis, result=self._shaft_client_result(axis))
        for token in (regular, invited):
            response = self.client.post('/api/shaft/axes', json=payload, headers={'Authorization': f'Bearer {token}'})
            self.assertEqual(response.status_code, 400)
        saved = self._post('/api/shaft/axes', payload, token=tester)
        public = self._post(f"/api/shaft/axes/{saved['id']}/publish", {}, token=tester)
        self.assertEqual(self._get(f"/api/shaft/axes/{saved['id']}", token=tester)['axis']['team'][0]['character_id'], CID)
        for token in (None, regular, invited, tester):
            ids = [a['id'] for a in self._get('/api/shaft/market', token=token)['items']]
            self.assertEqual(public['id'] in ids, token == tester)
            response = self.client.get(f"/api/shaft/axes/{public['id']}", headers={'Authorization': f'Bearer {token}'} if token else {})
            self.assertEqual(response.status_code, 200 if token == tester else 400)


class ElymsActivationTest(unittest.TestCase):
    axis = ElymsCatalogTest.axis
    sim = ElymsCatalogTest.sim
    own = ElymsCatalogTest.own
    def check(self, names, nodes=(), result=None, loop=False):
        axis=self.axis([(0,'action_elyms_'+name,tick) for name,tick in names],nodes=nodes)
        axis['options']['loop_enabled']=loop
        result=result if result is not None else simulate_shaft_axis(axis)['result']
        script="const s=require(process.argv[1]);const p=JSON.parse(require('fs').readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(s.inspectAxis(p.axis,p.catalog,p.result)));"
        response=subprocess.run(['node','-e',script,str(ROOT/'app/modules/shaft/static/js/shaft_self_check.js')],
                                input=json.dumps(dict(axis=axis,catalog=load_shaft_catalog(),result=result)),text=True,capture_output=True,check=True)
        return [w for w in json.loads(response.stdout) if '尚未激活或激活已过期' in w]

    def test_weak_finisher_and_e1_activate_e2_for_twelve_seconds(self):
        self.assertTrue(self.check([('skill2',0)]))
        self.assertEqual(self.check([('melee5',0),('skill2',124)]),[])
        self.assertTrue(self.check([('melee5',0),('skill2',125)]))
        self.assertEqual(self.check([('skill1',0),('skill2',10)]),[])
        self.assertTrue(self.check([('melee4',0),('skill2',10)]))
        self.assertEqual(self.check([('melee5',0),('ultraskilla1',10),('skill2',20)]),[])
        self.assertTrue(self.check([('melee5',0),('skill2',10),('skill2',20)]))

    def test_strong_finisher_activates_e3_once_for_six_seconds(self):
        prefix=[('skill1',0),('skill2',10),('meleeb3',20)]
        self.assertEqual(self.check(prefix+[('goodbye',84)]),[])
        self.assertTrue(self.check(prefix+[('goodbye',85)]))
        self.assertTrue(self.check([('skill1',0),('skill2',10),('meleeb2',20),('goodbye',30)]))
        self.assertTrue(self.check(prefix+[('goodbye',30),('goodbye',40)]))
        # An attack in the wrong signal form cannot grant the other form's activation.
        self.assertTrue(self.check([('meleeb3',0),('goodbye',10)]))
        self.assertTrue(self.check([('skill1',0),('skill2',10),('melee5',20),('skill2',30)]))

    def test_interruption_requires_actual_last_hit(self):
        names=[('melee5',0),('skill2',10)]
        result=self.own(names)
        result['details'][0].update(hit_count=0,original_hit_count=1)
        self.assertTrue(self.check(names,result=result))
        names=[('skill1',0),('skill2',10),('meleeb3',20),('goodbye',30)]
        result=self.own(names)
        result['details'][2].update(hit_count=3,original_hit_count=4)
        self.assertTrue(self.check(names,result=result))

    def test_resonance_and_current_signal_are_only_validation(self):
        names=[('skill1',0),('skill2',10),('goodbye',20),('goodbye',30)]
        # Three active nodes enable the one immediate E3 when B jumps to strong signal.
        self.assertEqual(len(self.check(names,nodes=[1,2,3])),1)
        names=[('skill1',0),('skill2',10),('meleeb3',20),('ultraskilla1',30),('goodbye',40)]
        self.assertTrue(self.check(names))
        result=self.own([('skill2',0)])
        self.assertGreater(result['details'][0]['direct_damage'],0)
        self.assertFalse(any('elyms_e2' in b.get('definition_id','') or 'elyms_e3' in b.get('definition_id','')
                             for detail in result['details'] for b in detail['triggered_buffs']+detail['applied_buffs']))

    def test_loop_finisher_completion_carries_to_next_cycle(self):
        names=[('skill2',0),('melee5',180)]
        result={'summary':{'duration_ticks':200},'details':[
            {'step_id':'0','start_tick':0,'end_tick':5,'hit_count':6,'applied_buffs':[]},
            {'step_id':'1','start_tick':180,'end_tick':200,'hit_count':1,'original_hit_count':1,'applied_buffs':[]},
        ]}
        self.assertTrue(self.check(names,result=result))
        self.assertEqual(self.check(names,result=result,loop=True),[])
