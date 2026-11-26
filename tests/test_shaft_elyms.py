"""Test-only framework and deliberately incomplete character data."""
import importlib
import json
import subprocess
import unittest
from pathlib import Path

from tests.shaft_duration_fixtures import with_scenario_ticks

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

    def test_bond_bonus_adds_five_percent_attack_only_when_enabled(self):
        character=next(c for c in load_shaft_catalog()['characters'] if c['id']==CID)
        self.assertEqual(character['bond_bonus'],{'label':'5攻击','modifiers':{'atk_pct':.05}})
        panels=[]
        for enabled in (False,True):
            axis=self.axis([(0,'action_elyms_melee1',0)])
            axis['team'][0]['bond_full']=enabled
            normalized=normalize_axis_payload(axis)
            self.assertEqual(normalized['team'][0]['bond_full'],enabled)
            self.assertEqual(normalized['character_builds'][CID]['bond_full'],enabled)
            panels.append(simulate_shaft_axis(axis)['result']['details'][0]['panel'])
        self.assertAlmostEqual(panels[1]['atk']-panels[0]['atk'],character['base_stats']['atk']*.05)
        for key in ('crit_rate','hp','def','crit_dmg'):
            self.assertEqual(panels[1][key],panels[0][key])

    def test_confirmed_catalog_and_action_resources(self):
        c = load_shaft_catalog()
        character = next(x for x in c['characters'] if x['id'] == CID)
        self.assertEqual(character['adaptation'], '固态')
        self.assertEqual(character['base_stats'], {'atk':660,'hp':15392,'def':909})
        self.assertEqual(character['energy_capacity'],100)
        self.assertEqual(character['avatar'],'/static/images/characters/avatar/曳尔沐-square-v2.webp')
        self.assertEqual(character['portrait'],'/static/images/characters/portrait/曳尔沐.webp')
        self.assertNotIn('fullbody',character)
        actions = {a['name']:a for a in c['actions_by_character'][CID]}
        self.assertEqual(len(actions),21)
        self.assertNotIn('瞄准',actions)
        expected = {'a1': (0.224, 2, 1.408, 2.34, 0.278, 5),
         'a2': (0.314, 2, 1.964, 3.266, 0.388, 5),
         'a3': (0.834, 3, 5.227, 8.692, 1.034, 12),
         'a4': (0.698, 5, 4.371, 7.266, 0.865, 6),
         'a5': (0.915, 1, 5.733, 9.534, 1.134, 11),
         '强a1': (0.259, 2, 1.62, 2.695, 0.328, 5),
         '强a2': (0.376, 4, 2.352, 3.912, 0.476, 4),
         '强a3': (0.788, 4, 4.928, 8.194, 0.996, 10),
         '重击': (0.449, 4, 2.801, 4.7, 0.6, 7),
         '下落': (0.296, 1, 1.9, 3.1, 0.4, 8),
         '闪反': (1.0, 1, 2.5, 4.2, 2.5, 8),
         'e1': (1.701, 6, 5.1, 8.399, 1.001, 11),
         'e2': (0.998, 6, 8.3, 0.0, 0.0, 0),
         'e3': (1.277, 6, 8.3, 0.0, 0.0, 0),
         'q群': (4.05, 4, 0.0, 0.0, 1.501, 0),
         'q单': (4.5, 8, 0.0, 0.0, 1.502, 0),
         '追忆群': (1.08, 4, 1.7, 2.8, 0.3, 0),
         '追忆单': (1.2, 4, 1.7, 2.8, 0.3, 0),
         '援护': (2.0, 1, 5.2, 0.0, 2.5, 11)}
        for name,(mult,hits,energy,harmony,stagger,ticks) in expected.items():
            with self.subTest(action=name):
                a=actions[name]
                self.assertAlmostEqual(a['multipliers']['atk'],mult)
                self.assertEqual(a['hit_count'],hits)
                self.assertEqual((a['energy_gain'],a['harmony'],a['stagger']),(energy,harmony,stagger))
                self.assertEqual(a['duration_ticks'],ticks)
                self.assertFalse(a['duration_pending'])
                self.assertTrue(a['duration_implemented'])
                if ticks > 0 and a['action_type'] != '援护':
                    tail=a['hit_profile']['cumulative'][-1]
                    self.assertEqual(tail['hit_count'],hits)
                    for key,value in [('damage',mult),('energy',energy),('harmony',harmony),('stagger',stagger)]:
                        self.assertAlmostEqual(tail[key],value)
                self.assertEqual(bool(a.get('is_time_stop_zero')),name in ('e2','e3','q群','q单'))
        self.assertEqual(actions['e1']['harmony_on_start'],100)
        self.assertEqual(actions['e1']['cooldown_ticks'],160)
        for name in ('q群','q单'):
            self.assertEqual(actions[name]['energy_cost'],100)
            self.assertEqual(actions[name]['cooldown_ticks'],200)
        for name in ('追忆群','追忆单'):
            self.assertEqual(actions[name]['action_type'],'被动')
            self.assertEqual(actions[name]['damage_type'],'Q')
            self.assertEqual(actions[name]['skill_level_category'],'ultimate')
            self.assertTrue(actions[name]['resonance_skill_level_bonus'])
        self.assertEqual(c['formula_constants']['default_build_options'][CID]['curtain_bonus'],
                         {'value':8,'stat':'暴击','passive_type':'type3','fixed':True})
        self.assertEqual(next(a['id'] for a in c['arcs'] if a['adaptation']=='固态'),'arc_elyms_waiting_ringtone')
        ref=c['arc_refinements']['arcs']['arc_elyms_waiting_ringtone']
        self.assertEqual(ref['available_levels'],[1,2,3,4,5])
        for level in range(1,6):
            values=ref['levels'][str(level)]
            self.assertNotIn('harmony_strength',values['panel_modifiers'])
            self.assertAlmostEqual(values['panel_modifiers']['atk_pct'],.15+.0375*(level-1))
            self.assertAlmostEqual(values['buff_effects']['arc_elyms_light_pen']['res_down_光'],.08+.03*(level-1))
            self.assertAlmostEqual(values['buff_effects']['arc_elyms_team_def']['def_ignore'],.12+.02*(level-1))
            self.assertAlmostEqual(values['buff_effects']['arc_elyms_self_def']['def_ignore'],.1+.025*(level-1))

    def test_b_awakening_only_grants_consumption_defense_ignore_to_e3(self):
        result=self.own([('skill2',0),('goodbye',10),('melee1',20)],nodes=[2,3])
        self.assertAlmostEqual(self.effects(result['details'][1])['character_elyms_b_ignore']['def_ignore'],.05)
        self.assertNotIn('character_elyms_b_ignore',self.effects(result['details'][2]))
        empty=self.own([('goodbye',0)],nodes=[2])
        self.assertEqual(self.effects(empty['details'][0]).get('character_elyms_b_ignore',{}),{})
        no_b=self.own([('skill2',0),('goodbye',10)])
        self.assertNotIn('character_elyms_b_ignore',self.effects(no_b['details'][1]))

    def test_partial_hits_only_grant_their_actual_resources(self):
        axis=self.axis([(0,'action_elyms_melee3',0)])
        axis['steps'][0].update(interrupted=True,interrupt_duration_ticks=4,interrupt_hit_count=2)
        d=simulate_shaft_axis(axis)['result']['details'][0]
        self.assertEqual(d['hit_count'],2)
        self.assertAlmostEqual(d['base_energy_gain'],4.188)
        self.assertAlmostEqual(d['harmony'],6.964)
        self.assertAlmostEqual(d['stagger_amount'],.828)

    def test_estimated_durations_do_not_add_trial_warnings(self):
        catalog=load_shaft_catalog()
        own_actions=catalog['actions_by_character'][CID]
        self.assertTrue(any(a.get('duration_estimated') for a in own_actions))
        self.assertFalse(any('试算' in a.get('validation_note','') for a in own_actions))
        result=self.own([('melee1',0),('ultraskilla1',10)])
        self.assertFalse(any('试算' in warning for d in result['details'] for warning in d['warnings']))

    def test_signal_multiplier_snapshot_reset_and_weak_action_checks(self):
        r=self.own([('skill2',0),('goodbye',10),('melee1',20),('ultraskilla1',30),('melee1',40),('xietongskilla1',50)])
        details=r['details']
        self.assertAlmostEqual(details[1]['formula_parts']['base_multiplier_factor'],2)
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
        self.assertAlmostEqual(d[1]['formula_parts']['base_multiplier_factor'],5)
        fx=self.effects(d[3]);self.assertAlmostEqual(fx['character_elyms_a']['crit_dmg'],.4)
        self.assertAlmostEqual(fx['character_elyms_d']['crit_dmg'],.2)
        # F remains active for all follow damage, even when this probe has no follow tag.
        fx=self.effects(d[5]);self.assertEqual(fx['character_elyms_a'],{})
        extras=[e for e in r['periodic_damage_events'] if e.get('action_id')=='action_elyms_six_extra']
        self.assertEqual(len(extras),1)
        self.assertAlmostEqual(extras[0]['atk_multiplier'],6.72)
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

    def test_signature_window_clears_on_threshold_without_refreshing_defense(self):
        teammate=dict(slot=1,character_id='char_dd034941ef',arc_id='',cartridge_id='')
        result=self.sim([
            (0,'action_elyms_ultraskilla1',0),(1,'action_none_dd034941ef',10),
            (0,'action_elyms_xietongskilla1',20),(0,'action_elyms_xietongskilla1',60),
            (1,'action_none_dd034941ef',230),(0,'action_elyms_ultraskilla1',260),
            (1,'action_none_dd034941ef',270),(0,'action_elyms_xietongskilla1',280),
        ],others=[teammate],arc='arc_elyms_waiting_ringtone')
        details=result['details']
        window=next(b for b in details[0]['triggered_buffs'] if b['rule_id']=='arc_elyms_q_window')
        self.assertEqual(window['end_tick'],details[2]['start_tick'])
        self.assertEqual(window['visual_end_tick'],details[2]['visual_start_tick'])
        for key in ('arc_elyms_q_window','arc_elyms_counter'):
            self.assertNotIn(key,self.effects(details[3]))
        for key in ('arc_elyms_team_def','arc_elyms_self_def'):
            buff=next(b for b in details[2]['triggered_buffs'] if b['rule_id']==key)
            self.assertEqual(buff['end_tick'],details[2]['start_tick']+200)
            self.assertIn(key,self.effects(details[3]))
            self.assertFalse(any(b['rule_id']==key for b in details[3]['triggered_buffs']))
            self.assertNotIn(key,self.effects(details[4]))
            self.assertTrue(any(b['rule_id']==key for b in details[-1]['triggered_buffs']))
        new_window=next(b for b in details[5]['triggered_buffs'] if b['rule_id']=='arc_elyms_q_window')
        self.assertEqual(new_window['end_tick'],details[-1]['start_tick'])

    def test_energy_gain_talent_and_echo_do_not_recurse(self):
        catalog=load_shaft_catalog()
        chaos=next(a['id'] for a in catalog['actions_by_character']['char_d38b672525'] if a['name']=='援护')
        others=[dict(slot=1,character_id='char_dd034941ef',arc_id='',cartridge_id=''),
                dict(slot=2,character_id='char_d38b672525',arc_id='',cartridge_id='')]
        axis=self.axis([(1,'action_982c67944f',0),(2,chaos,30),(1,'action_none_dd034941ef',200)],others=others)
        axis['initial_energy']=0
        r=simulate_shaft_axis(axis)['result']
        light=[b for d in r['details'] for b in d['applied_buffs'] if b['rule_id']=='character_elyms_energy_light']
        # A delay without creation never produces 盈蓄, even when the front gains energy.
        self.assertFalse(light)
        self.assertFalse(any(b['rule_id']=='character_elyms_energy_light' for d in r['details'] for b in d['triggered_buffs']))
        axis['initial_energy']=1000
        capped=simulate_shaft_axis(axis)['result']
        self.assertFalse(any(b['rule_id']=='character_elyms_energy_light' for d in capped['details'] for b in d['applied_buffs']))
        echoes=[e for e in r['reaction_effects'] if e.get('is_echo')]
        self.assertEqual(len(echoes),1)
        original=next(e for e in r['reaction_effects'] if e['reaction']=='延滞' and not e.get('is_echo'))
        self.assertEqual(echoes[0]['start_tick']-original['start_tick'],75)

    def test_delay_echo_waiting_line_lasts_seven_and_half_seconds_and_carries(self):
        catalog=load_shaft_catalog()
        support=next(a['id'] for a in catalog['actions_by_character']['char_d38b672525'] if a['name']=='援护')
        others=[dict(slot=1,character_id='char_dd034941ef',arc_id='',cartridge_id=''),
                dict(slot=2,character_id='char_d38b672525',arc_id='',cartridge_id='')]
        steps=[(1,'action_982c67944f',0),(2,support,30),(1,'action_none_dd034941ef',100)]
        for loop in (False,True):
            with self.subTest(loop=loop):
                axis=self.axis(steps,others=others,loop_enabled=loop)
                result=simulate_shaft_axis(axis)['result']
                detail=next(d for d in result['details'] if d['step_id']=='1')
                waiting,=[b for b in detail['triggered_buffs'] if b['rule_id']=='character_elyms_delay_echo']
                self.assertTrue(waiting['display_as_line'])
                self.assertEqual(waiting['end_tick']-waiting['trigger_tick'],75)
                self.assertEqual(waiting['owner_slot'],0)
                echoes=[e for e in result['reaction_effects'] if e.get('is_echo')]
                if loop:
                    duration=result['summary']['duration_ticks']
                    self.assertIn(waiting['end_tick']-duration,[e['start_tick'] for e in echoes])
                    source=(ROOT/'app/modules/shaft/static/js/shaft.js').read_text()
                    functions=source[source.index('  function wrappedTick('):source.index('  function mergedRenderedBuffSegments(')]
                    script='''const calculationTickFromVisual=v=>v, visualTickFromCalculation=v=>v;
                      const shouldDrawBuffLine=b=>b.display_as_line, buffRuleDisplayName=b=>b.name;
                      const formatNumber=String, ticksToSeconds=v=>v/10, BUFF_DURATION_LABEL_LIMIT_TICKS=9000;'''+functions
                    script+='const detail='+json.dumps(dict(detail,triggered_buffs=[waiting]))+';'
                    script+=f'const lines=mergedBuffLineSegments([detail],{duration},true,0);'
                    script+='console.log(JSON.stringify(lines));'
                    lines=json.loads(subprocess.check_output(['node','-e',script],text=True))
                    carry=next(line for line in lines if line['startTick']==0)
                    self.assertEqual(carry['endTick'],waiting['end_tick']-duration)
                else:
                    self.assertIn(waiting['end_tick'],[e['start_tick'] for e in echoes])
                    self.assertEqual(len(echoes),1)

    def test_delay_echo_advances_to_first_mask_and_shortens_waiting_line(self):
        catalog=load_shaft_catalog()
        support=next(a['id'] for a in catalog['actions_by_character']['char_d38b672525'] if a['name']=='援护')
        others=[dict(slot=1,character_id='char_dd034941ef',arc_id='',cartridge_id=''),
                dict(slot=2,character_id='char_d38b672525',arc_id='',cartridge_id=''),
                dict(slot=3,character_id='char_0846d632e0',arc_id='',cartridge_id='')]
        for slot,action,advances in [(0,'action_elyms_skill2',True),
                (0,'action_elyms_goodbye',True),(0,'action_elyms_ultraskilla1',True),
                (1,'action_b356b46da2',True),(3,'action_lingke_time_stop_select',True),
                (0,'action_elyms_melee1',False),(0,'action_elyms_xietongskilla1',False)]:
            with self.subTest(action=action):
                result=simulate_shaft_axis(self.axis([(1,'action_982c67944f',0),(2,support,30),
                    (slot,action,65),(0,'action_elyms_ultraskillb1',170),
                    (1,'action_none_dd034941ef',240)],others=others))['result']
                waiting=next(b for d in result['details'] for b in d['triggered_buffs']
                             if b['rule_id']=='character_elyms_delay_echo')
                expected=result['details'][2]['start_tick'] if advances else waiting['trigger_tick']+75
                echoes=[e for e in result['reaction_effects'] if e.get('is_echo')]
                self.assertEqual([e['start_tick'] for e in echoes],[expected])
                self.assertEqual(waiting['end_tick'],expected)
                self.assertTrue(waiting['display_as_line'])

    def test_echo_cooldown_advances_on_mask_and_otherwise_lasts_seven_seconds(self):
        catalog=load_shaft_catalog()
        support=next(a['id'] for a in catalog['actions_by_character']['char_d38b672525'] if a['name']=='援护')
        others=[dict(slot=1,character_id='char_dd034941ef',arc_id='',cartridge_id=''),
                dict(slot=2,character_id='char_d38b672525',arc_id='',cartridge_id='')]
        for masked,next_start,expected_count in [(True,99,2),(True,100,2),(False,99,1),(False,100,2)]:
            with self.subTest(masked=masked,next_start=next_start):
                action='action_elyms_skill2' if masked else 'action_elyms_melee1'
                # A mask inserts five display ticks without advancing the real clock.
                shift=5 if masked else 0
                result=simulate_shaft_axis(self.axis([(1,'action_982c67944f',0),(2,support,30),
                    (0,action,65),(1,'action_none_dd034941ef',90),
                    (2,support,next_start+shift),(1,'action_none_dd034941ef',240)],others=others))['result']
                waits=[b for d in result['details'] for b in d['triggered_buffs']
                       if b['rule_id']=='character_elyms_delay_echo']
                self.assertEqual(len(waits),expected_count)
                self.assertEqual(waits[0]['end_tick'],65 if masked else waits[0]['trigger_tick']+75)
                if expected_count==2:
                    self.assertEqual(waits[1]['trigger_tick']-waits[0]['trigger_tick'],next_start-30)

    def test_mask_ends_carried_cooldown_in_a_cycle_shorter_than_seven_seconds(self):
        catalog=load_shaft_catalog()
        support=next(a['id'] for a in catalog['actions_by_character']['char_d38b672525'] if a['name']=='援护')
        others=[dict(slot=1,character_id='char_dd034941ef',arc_id='',cartridge_id=''),
                dict(slot=2,character_id='char_d38b672525',arc_id='',cartridge_id='')]
        for masked in (False,True):
            with self.subTest(masked=masked):
                action='action_elyms_skill2' if masked else 'action_elyms_melee1'
                result=simulate_shaft_axis(self.axis([(1,'action_982c67944f',0),(0,action,20),
                    (1,'action_none_dd034941ef',27),(2,support,35),
                    (1,'action_none_dd034941ef',65 if masked else 60)],
                    others=others,loop_enabled=True))['result']
                self.assertEqual(result['summary']['duration_ticks'],60)
                waits=[b for d in result['details'] for b in d['triggered_buffs']
                       if b['rule_id']=='character_elyms_delay_echo']
                self.assertEqual(len(waits),1 if masked else 0)

    def test_carried_echo_advances_to_next_cycles_first_mask(self):
        catalog=load_shaft_catalog()
        support=next(a['id'] for a in catalog['actions_by_character']['char_d38b672525'] if a['name']=='援护')
        others=[dict(slot=1,character_id='char_dd034941ef',arc_id='',cartridge_id=''),
                dict(slot=2,character_id='char_d38b672525',arc_id='',cartridge_id='')]
        result=simulate_shaft_axis(self.axis([(0,'action_elyms_skill2',20),
            (1,'action_982c67944f',50),(2,support,100),(1,'action_none_dd034941ef',160)],
            others=others,loop_enabled=True))['result']
        mask_tick=result['details'][0]['start_tick']
        self.assertEqual([e['start_tick'] for e in result['reaction_effects'] if e.get('is_echo')],[mask_tick])
        waiting=next(b for d in result['details'] for b in d['triggered_buffs']
                     if b['rule_id']=='character_elyms_delay_echo')
        self.assertEqual(waiting['end_tick'],result['summary']['duration_ticks']+mask_tick)
        self.assertLess(waiting['end_tick']-waiting['trigger_tick'],75)

    def test_regular_energy_without_yingxu_never_grants_light_stacks(self):
        axis=self.axis([(0,'action_elyms_melee1',0),(0,'action_elyms_melee2',10),
                        (0,'action_elyms_skill2',20),(0,'action_elyms_melee1',30)])
        axis['initial_energy']=0
        result=simulate_shaft_axis(axis)['result']
        self.assertTrue(any(e['amount']>0 for e in result['energy_events']))
        for detail in result['details']:
            self.assertNotIn('character_elyms_energy_light',self.effects(detail))
            self.assertFalse(any(b['rule_id']=='character_elyms_energy_light' for b in detail['triggered_buffs']))

    def yingxu_axis(self, initial_energy=0):
        others=[dict(slot=1,character_id='char_b2e3b2bf7a',arc_id='',cartridge_id=''),
                dict(slot=2,character_id='char_d38b672525',arc_id='',cartridge_id='')]
        axis=self.axis([(0,'action_none_elyms',0),(1,'action_8a8000d4a4',2),
                        (0,'action_none_elyms',18),(2,'action_17efab5dbd',20),
                        *[(0,'action_none_elyms',t) for t in (40,55,80,105,120,145,314,315)]],
                       others=others,switch_loss_ticks=0)
        axis['initial_energy']=initial_energy
        return axis

    @with_scenario_ticks
    def test_only_actual_yingxu_energy_stacks_and_refreshes(self):
        r=simulate_shaft_axis(self.yingxu_axis())['result']
        events=[e for e in r['energy_events'] if e['kind']=='yingxu_gain']
        # Creation hits at 35/55/75 overlap delay; 95 misses, 115 hits the delayed echo and refreshes the 20s buff.
        self.assertEqual([(e['tick'],e['slot']) for e in events],[(35,2),(55,0),(75,0),(115,0)])
        for e in events:
            self.assertAlmostEqual(e['amount'],10)
            self.assertEqual(e['source_reaction'],'盈蓄')
        summaries=[b for d in r['details'] for b in d['triggered_buffs']
                   if b['rule_id']=='character_elyms_energy_light']
        self.assertEqual(sorted((b['trigger_tick'], b['visual_trigger_tick'], b['stack_count'])
                                for b in summaries), [(35,35,1),(55,55,2),(75,75,3),(115,115,4)])
        by_tick={d['start_tick']:d for d in r['details']}
        for tick,stacks in ((40,1),(55,2),(80,3),(105,3),(120,4),(145,4),(314,4)):
            buff=next(b for b in by_tick[tick]['applied_buffs'] if b['rule_id']=='character_elyms_energy_light')
            self.assertEqual(buff['stack_count'],stacks)
            self.assertAlmostEqual(buff['effects']['element_dmg_光'],.0375*stacks)
        for tick in (315,):
            self.assertNotIn('character_elyms_energy_light',self.effects(by_tick[tick]))

    @with_scenario_ticks
    def test_full_energy_still_stacks_but_creation_without_delay_does_not(self):
        capped=simulate_shaft_axis(self.yingxu_axis(1000))['result']
        events=[e for e in capped['energy_events'] if e['kind']=='yingxu_gain']
        self.assertEqual([e['tick'] for e in events],[35,55,75,115])
        self.assertTrue(all(e['amount']==0 for e in events))
        for tick,stacks in ((40,1),(55,2),(80,3),(105,3),(120,4),(145,4),(314,4)):
            detail=next(d for d in capped['details'] if d['start_tick']==tick)
            buff=next(b for b in detail['applied_buffs'] if b['rule_id']=='character_elyms_energy_light')
            self.assertEqual(buff['stack_count'],stacks)
        self.assertTrue(all(e['energy_after']<=100 for e in events))
        axis=self.yingxu_axis()
        axis['steps']=[s for s in axis['steps'] if s['action_id']!='action_17efab5dbd']
        r=simulate_shaft_axis(axis)['result']
        self.assertTrue(any(e['reaction']=='创生' for e in r['reaction_effects']))
        self.assertFalse(any(e['kind']=='yingxu_gain' for e in r['energy_events']))
        self.assertFalse(any('character_elyms_energy_light' in self.effects(d) for d in r['details']))

    def test_yingxu_stacks_cap_at_eight_across_front_switches(self):
        axis=self.yingxu_axis()
        nanali=next(c['id'] for c in load_shaft_catalog()['characters'] if c['name']=='娜娜莉')
        axis['team'].append(dict(slot=3,character_id=nanali,arc_id='',cartridge_id=''))
        axis['steps']=[dict(id=str(i),slot=slot,action_id=action,start_tick=tick)
                       for i,(slot,action,tick) in enumerate([
                           (0,'action_none_elyms',0),(1,'action_8a8000d4a4',2),
                           (0,'action_none_elyms',18),(1,'action_8a8000d4a4',20),
                           (0,'action_none_elyms',36),(2,'action_17efab5dbd',38),
                           (0,'action_none_elyms',58),(0,'action_none_elyms',100),
                           (0,'action_none_elyms',400)])]
        r=simulate_shaft_axis(axis)['result']
        self.assertEqual(len([e for e in r['energy_events'] if e['kind']=='yingxu_gain']),11)
        full=next(d for d in r['details'] if d['start_tick']==100)
        buff=next(b for b in full['applied_buffs'] if b['rule_id']=='character_elyms_energy_light')
        self.assertEqual(buff['stack_count'],8)
        self.assertAlmostEqual(buff['effects']['element_dmg_光'],.3)
        self.assertNotIn('character_elyms_energy_light',self.effects(r['details'][-1]))

    def test_old_partial_entries_are_removed_without_missing_duration_for_known_actions(self):
        axis=self.axis([(0,'action_elyms_melee3_1',0),(0,'action_elyms_melee1',10)])
        self.assertEqual([s['action_id'] for s in normalize_axis_payload(axis)['steps']],['action_elyms_melee1'])
        catalog=load_shaft_catalog()
        script="const s=require(process.argv[1]);const p=JSON.parse(require('fs').readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(s.inspectAxis(p.axis,p.catalog)));"
        result=subprocess.run(['node','-e',script,str(ROOT/'app/modules/shaft/static/js/shaft_self_check.js')],input=json.dumps(dict(axis=axis,catalog=catalog)),text=True,capture_output=True,check=True)
        self.assertFalse(any('a1」：该动作的时长未实装' in w for w in json.loads(result.stdout)))


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

    @with_scenario_ticks
    def test_weak_finisher_and_e1_activate_e2_for_twelve_seconds(self):
        self.assertTrue(self.check([('skill2',0)]))
        self.assertEqual(self.check([('melee5',0),('skill2',129)]),[])
        self.assertTrue(self.check([('melee5',0),('skill2',130)]))
        self.assertEqual(self.check([('skill1',0),('skill2',10)]),[])
        self.assertTrue(self.check([('melee4',0),('skill2',10)]))
        self.assertEqual(self.check([('melee5',0),('ultraskilla1',10),('skill2',20)]),[])
        self.assertTrue(self.check([('melee5',0),('skill2',10),('skill2',20)]))

    @with_scenario_ticks
    def test_strong_finisher_activates_e3_once_for_six_seconds(self):
        prefix=[('skill1',0),('skill2',10),('meleeb3',20)]
        self.assertEqual(self.check(prefix+[('goodbye',85)]),[])
        self.assertTrue(self.check(prefix+[('goodbye',86)]))
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
        axis=self.axis([(0,'action_elyms_'+name,tick) for name,tick in names])
        axis['steps'][2].update(interrupted=True,interrupt_duration_ticks=3,interrupt_hit_count=3)
        result=simulate_shaft_axis(axis)['result']
        self.assertEqual(result['details'][2]['hit_count'],3)
        self.assertTrue(self.check(names,result=result))

    @with_scenario_ticks
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


class ElymsSpecialEZeroTimeTest(unittest.TestCase):
    axis = ElymsCatalogTest.axis
    sim = ElymsCatalogTest.sim

    @with_scenario_ticks
    def test_special_es_use_public_zero_time_mask_and_foreground_lock(self):
        teammate=dict(slot=1,character_id='char_dd034941ef',arc_id='',cartridge_id='')
        for name in ('skill2','goodbye'):
            with self.subTest(skill=name):
                r=self.sim([(0,'action_elyms_'+name,0),(1,'action_982c67944f',2)],others=[teammate],switch_loss_ticks=0)
                special,intruder=r['details']
                self.assertEqual(special['duration_ticks'],0)
                self.assertEqual(special['start_tick'],special['end_tick'])
                self.assertEqual(special['action_type'],'E')
                self.assertEqual(intruder['visual_start_tick'],special['visual_end_tick'])
                self.assertEqual(intruder['foreground_lock_ticks'],3)
                self.assertEqual(r['summary']['frozen_ticks'],5)
                r=self.sim([(1,'action_982c67944f',0),(0,'action_elyms_'+name,2)],others=[teammate],switch_loss_ticks=0)
                covered,mask=r['details']
                self.assertEqual(covered['end_tick'],mask['start_tick'])
                self.assertEqual(covered['duration_ticks'],2)
                self.assertEqual(mask['duration_ticks'],0)
                self.assertEqual(r['summary']['frozen_ticks'],13)

    def test_special_es_no_longer_report_unimplemented_time(self):
        catalog=load_shaft_catalog()
        actions=[a for a in catalog['actions_by_character'][CID] if a['name'] in ('e2','e3')]
        script="const s=require(process.argv[1]);const a=JSON.parse(require('fs').readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(a.map(x=>s.hasUnimplementedForegroundDuration({},x))));"
        result=subprocess.run(['node','-e',script,str(ROOT/'app/modules/shaft/static/js/shaft_self_check.js')],input=json.dumps(actions),text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(result.stdout),[False,False])
