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

    def test_catalog_exact_levels_durations_and_bond(self):
        c=load_shaft_catalog();char=next(x for x in c['characters']if x['id']==CID)
        self.assertEqual(char['base_stats'],dict(atk=676,hp=15271,**{'def':884}))
        self.assertEqual((char['adaptation'],char['element'],char['energy_capacity']),('等离子','暗',120))
        self.assertEqual(char['bond_bonus'],{'label':'8暴伤','modifiers':{'crit_dmg':.08}})
        actions={a['name']:a for a in c['actions_by_character'][CID]}
        for name,atk,hits,ticks in [('a1',.406,1,5),('a2',.692,2,7),('a3',.792,3,7),('a4',1.684,7,13),
                                    ('a5',.594,6,7),('z',.780,1,7),('e',2.91,6,16),('q',5.822,19,0),('援护',2,3,18)]:
            with self.subTest(name=name):
                a=actions[name];self.assertAlmostEqual(a['multipliers']['atk'],atk);self.assertEqual(a['hit_count'],hits);self.assertEqual(a['duration_ticks'],ticks)
                self.assertEqual(len(a['atk_multipliers_by_level']),13)
        self.assertEqual(actions['q']['energy_cost'],120)
        self.assertTrue(actions['q']['is_time_stop_zero'])
        self.assertNotIn('数字残像',actions)
        self.assertEqual(char['initial_personal_resources'],{'帧数':0,'评分':0})
        self.assertEqual(char['timeline_personal_resources'],['帧数','评分'])
        clone=char['ranked_resource_system']['export_action']
        self.assertAlmostEqual(clone['multipliers']['atk'],.776)
        self.assertEqual(len(clone['atk_multipliers_by_level']),13)
        self.assertEqual(clone['personal_resource_cost'],{})
        self.assertEqual(next(a for a in c['arcs']if a['adaptation']=='等离子')['id'],ARC)

    def test_bond_adds_eight_percent_critical_damage_only_when_enabled(self):
        panels=[]
        for enabled in [False,True]:
            axis=self.axis([(0,'action_exe_a1',0)])
            axis['team'][0]['bond_full']=enabled
            normalized=normalize_axis_payload(axis)
            self.assertEqual(normalized['team'][0]['bond_full'],enabled)
            self.assertEqual(normalized['character_builds'][CID]['bond_full'],enabled)
            panels.append(self.sim(axis)['details'][0]['panel'])
        self.assertAlmostEqual(panels[1]['crit_dmg']-panels[0]['crit_dmg'],.08)
        for key in ['atk','crit_rate','hp','def']:self.assertEqual(panels[0][key],panels[1][key])

    def test_frame_gain_score_progress_and_export(self):
        r=self.own([('e',0),('a5',20)])
        hits=[e for e in r['periodic_damage_events']if e.get('action_id')=='action_exe_clone']
        self.assertEqual(len(hits),1)
        self.assertEqual((hits[0]['kind'],hits[0]['tick']),('triggered_damage',0))
        self.assertGreater(hits[0]['damage'],0)
        self.assertTrue(all(not b['display_as_line'] for b in hits[0]['triggered_buffs'] if b['rule_id']=='character_exe_cooldown'))
        first=r['details'][0]['personal_resources_after']
        self.assertAlmostEqual(first['帧数'],5.856)
        self.assertAlmostEqual(r['details'][-1]['personal_resources_after']['帧数'],5.856)
        self.assertFalse(any(e['resource']=='残像额度'for e in r['personal_resource_state_events']))
        self.assertTrue(all('残像额度'not in d['personal_resources_after']for d in r['details']))

    def test_interrupt_keeps_exact_levels_and_requires_finisher(self):
        axis=self.axis([(0,'action_exe_a4',0)])
        axis['team'][0]['skill_levels']['basic']=10
        axis['steps'][0].update(interrupted=True,interrupt_duration_ticks=3,interrupt_hit_count=3)
        detail=self.sim(axis)['details'][0]
        expected=(.372*2+.514)
        self.assertAlmostEqual(detail['formula_parts']['base'] / detail['panel']['atk'],expected)
        for hits,expected_charges in [(0,0),(5,0),(6,1)]:
            axis=self.axis([(0,'action_exe_a5',0)])
            axis['steps'][0].update(interrupted=True,interrupt_duration_ticks=5,interrupt_hit_count=hits)
            result=self.sim(axis,{'帧数':50})
            self.assertEqual(len([e for e in result['periodic_damage_events']if e.get('action_id')=='action_exe_clone']),expected_charges)

    def test_counter_repeats_damage_and_resources_for_each_retained_hit(self):
        counter = next(
            action for action in load_shaft_catalog()['actions_by_character'][CID]
            if action['id'] == 'action_exe_counter'
        )
        hits = counter['ranked_resource_hits']
        self.assertEqual(counter['hit_count'], 2)
        self.assertEqual([(hit['frames'], hit['score']) for hit in hits], [(5, 150)] * 2)
        self.assertEqual(hits[0]['atk_levels'], hits[1]['atk_levels'])
        self.assertEqual(counter['atk_multipliers_by_level'], [value * 2 for value in hits[0]['atk_levels']])
        self.assertEqual((counter['energy_gain'], counter['harmony'], counter['stagger']), (3.5, 5.8, 2.5))

        for count, expected_frames, exports in [(0, 0, 0), (1, 5, 0), (2, 5.5, 1)]:
            with self.subTest(count=count):
                axis = self.axis([(0, 'action_exe_counter', 0)])
                axis['team'][0]['skill_levels']['basic'] = 10
                if count < 2:
                    axis['steps'][0].update(interrupted=True, interrupt_duration_ticks=7, interrupt_hit_count=count)
                result = self.sim(axis, {'帧数': 0, '评分': 0})
                detail = result['details'][0]
                self.assertAlmostEqual(detail['formula_parts']['base'] / detail['panel']['atk'], 1.221 * count)
                self.assertAlmostEqual(detail['stagger_amount'], 1.25 * count)
                self.assertAlmostEqual(detail['personal_resources_after']['帧数'], expected_frames)
                self.assertEqual(sum(event.get('action_id') == 'action_exe_clone'
                                     for event in result['periodic_damage_events']), exports)

    def test_score_decay_and_independent_three_resonance(self):
        for nodes,resonances,decays in [((),[],True),((),[3],False),([1,2,3],[],True)]:
            axis=self.axis([(0,'action_none_exe',100)],nodes=nodes)
            axis['team'][0]['awakening_resonances']=resonances
            r=self.sim(axis,{'评分':600})
            score=r['resources_by_slot'][0]['personal_resources']['评分']
            self.assertEqual(score<600,decays)

    def test_teammates_dark_star_raises_exe_rating(self):
        others = [
            {'slot': 1, 'character_id': 'char_c78f7a08d5', 'arc_id': '', 'cartridge_id': ''},
            {'slot': 2, 'character_id': 'char_heiyu', 'arc_id': '', 'cartridge_id': ''},
        ]
        axis = self.axis([(1, 'action_none_c78f7a08d5', 0),
                          (2, 'action_heiyu_support', 2)], others=others)
        result = self.sim(axis)
        self.assertTrue(any(reaction['reaction'] == '黯星' for reaction in result['reaction_effects']))
        self.assertTrue(any(event['slot'] == 0 and event['resource'] == '评分' and event['value'] == 700
                            for event in result['personal_resource_state_events']))

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
            if expected:self.assertAlmostEqual(self.buffs(d)['character_exe_overclock']['effects']['all_dmg'],expected*.025)
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
        # E's own export immediately lowers its cooldown from 15 to 14 seconds.
        r=self.own([('e',0),('e',140)])
        self.assertFalse(any('CD' in w for w in r['details'][-1]['warnings']))
        r=self.own([('e',0),('e',130)])
        self.assertTrue(any('CD' in w for w in r['details'][-1]['warnings']))
        # E exports five, Q exports ten: shared floor remains two seconds after Q.
        for tick,warning in [(39,True),(45,False)]:
            r=self.own([('e',0),('q',20),('e',tick)],initial={'帧数':100})
            self.assertEqual(any('CD' in w for w in r['details'][-1]['warnings']),warning)
        r=self.own([('q',0),('a1',200)],nodes=[5],initial={'帧数':100})
        hits=[e for e in r['periodic_damage_events']if e.get('action_id')=='action_exe_clone_boom']
        self.assertEqual(len(hits),10)
        self.assertTrue(all(e['damage']>0 for e in hits))
        r=self.own([('a5',0)],nodes=[5],initial={'帧数':100})
        self.assertFalse(any(e.get('action_id')=='action_exe_clone_boom'for e in r['periodic_damage_events']))

    def test_six_resonance_interruption_resistance_and_explicit_off(self):
        for resonances,expected in [([],0),([3],0),([6],.5),([3,6],.5)]:
            axis=self.axis([(0,'action_exe_a1',0)],nodes=[1,2,3,4,5,6])
            axis['team'][0]['awakening_resonances']=resonances
            detail=self.sim(axis)['details'][0]
            self.assertAlmostEqual(detail['panel']['interrupt_resistance'],expected)
            self.assertEqual('character_exe_resonance6_interrupt'in self.buffs(detail),6 in resonances)

    def test_cooldown_reduction_clamps_two_point_five_seconds_and_never_extends(self):
        # E exports five. a5 exports once at 2.5s or 1.5s remaining.
        for trigger,release,warning in [(75,94,True),(75,95,False),(85,99,True),(85,100,False)]:
            result=self.own([('e',0),('a5',trigger),('e',release)],initial={'帧数':100})
            self.assertEqual(any('CD'in w for w in result['details'][-1]['warnings']),warning)

    def test_loop_warmup_afterimages_reduce_cooldown_without_counting_previous_cycle_hits(self):
        for frames, length, warning, exports in [
            (0, 100, '动作 CD 尚未结束，需等到 4.0s。', 1),
            (25, 99, '动作 CD 尚未结束，需等到 0.1s。', 5),
            (25, 100, None, 5),
        ]:
            with self.subTest(frames=frames, length=length):
                axis = self.axis([(0, 'action_exe_e', 0), (0, 'action_none_exe', length)])
                initial = {'帧数': frames, '评分': 0}
                axis['options'].update(loop_enabled=True, loop_initial_resources={CID: {'personal_resources': initial}})
                result = self.sim(axis)
                self.assertEqual(result['details'][0]['warnings'], [warning] if warning else [])
                clone_ticks = [event['tick'] for event in result['periodic_damage_events']
                               if event.get('action_id') == 'action_exe_clone']
                self.assertEqual(clone_ticks, [0] * exports)
                axis['options']['loop_enabled'] = False
                single = self.sim(axis, initial)
                self.assertEqual(result['details'][0]['personal_resources_after'],
                                 single['details'][0]['personal_resources_after'])

    def test_q_clone_frames_feed_back_immediately_with_one_ten_export_budget(self):
        for frames,count,remaining,score in [(0,4,6.48,705),(10,6,6.6,805),(100,10,72.96,1005)]:
            result=self.own([('q',0)],initial={'帧数':frames,'评分':0})
            events=[e for e in result['periodic_damage_events']if e.get('action_id')=='action_exe_clone']
            self.assertEqual(len(events),count)
            self.assertTrue(all(e['tick']==0 and e['damage']>0 for e in events))
            self.assertAlmostEqual(result['details'][0]['personal_resources_after']['帧数'],remaining)
            self.assertAlmostEqual(result['resources_by_slot'][0]['personal_resources']['帧数'],remaining)
            self.assertEqual(result['resources_by_slot'][0]['personal_resources']['评分'],score)
        # At Ω, feedback must stop at ten exports even though each hit returns nine frames.
        axis=self.axis([(0,'action_exe_q',0)])
        axis['team'][0]['awakening_resonances']=[6]
        result=self.sim(axis,{'帧数':0,'评分':5000})
        self.assertEqual(len([e for e in result['periodic_damage_events']if e.get('action_id')=='action_exe_clone']),10)
        self.assertAlmostEqual(result['resources_by_slot'][0]['personal_resources']['帧数'],50.6)

    def test_q_feedback_keeps_bursts_healing_and_loop_initial_resources_single(self):
        result=self.own([('q',0)],nodes=[5],initial={'帧数':0,'评分':0})
        self.assertEqual(len([e for e in result['periodic_damage_events']if e.get('action_id')=='action_exe_clone']),4)
        self.assertEqual(len([e for e in result['periodic_damage_events']if e.get('action_id')=='action_exe_clone_boom']),4)
        self.assertAlmostEqual(result['resources_by_slot'][0]['personal_resources']['帧数'],6.84)
        result=self.own([('q',0)],nodes=[6],initial={'帧数':0,'评分':0})
        self.assertEqual(len([e for e in result['healing_events']if e.get('source')=='Exe·帧数回复']),23)
        axis=self.axis([(0,'action_exe_q',0),(0,'action_none_exe',25)])
        axis['options'].update(loop_enabled=True,loop_initial_resources={CID:{'personal_resources':{'帧数':0,'评分':0}}})
        result=self.sim(axis)
        self.assertEqual(len([e for e in result['periodic_damage_events']if e.get('action_id')=='action_exe_clone']),4)
        self.assertAlmostEqual(result['resources_by_slot'][0]['personal_resources']['帧数'],6.48)
        self.assertFalse(any(e['tick']<0 for e in result['personal_resource_state_events']))
        result=self.own([('q',0),('q',205)],initial={'帧数':100,'评分':0})
        for source in ['0','1']:
            self.assertEqual(len([e for e in result['periodic_damage_events']if e.get('action_id')=='action_exe_clone'and e.get('source_step_id')==source]),10)

    def test_e_counter_and_support_reuse_frames_and_keep_their_own_export_limits(self):
        result = self.own([('e', 0)], initial={'帧数': 25, '评分': 0})
        self.assertEqual(sum(e.get('action_id') == 'action_exe_clone' for e in result['periodic_damage_events']), 5)
        self.assertAlmostEqual(result['resources_by_slot'][0]['personal_resources']['帧数'], 7.894)
        for name, count, remaining in [('e', 4, 6.116), ('counter', 3, 10.68), ('support', 3, 12.48)]:
            with self.subTest(name=name):
                result = self.own([('q', 0), (name, 5)], initial={'帧数': 0, '评分': 0})
                exports = [event for event in result['periodic_damage_events']
                           if event.get('action_id') == 'action_exe_clone']
                self.assertEqual([event['source_step_id'] for event in exports], ['0'] * 4 + ['1'] * count)
                self.assertAlmostEqual(result['details'][-1]['personal_resources_after']['帧数'], remaining)

    def test_afterimage_markers_follow_each_action_at_the_same_calculation_tick(self):
        result = self.own([('q', 0), ('e', 5)], nodes=[5], initial={'帧数': 30, '评分': 0})
        q, e = result['details']
        self.assertEqual((q['start_tick'], e['start_tick']), (0, 0))
        self.assertEqual((q['visual_start_tick'], e['visual_start_tick']), (0, 5))
        events = result['periodic_damage_events']
        for action_id in ('action_exe_clone', 'action_exe_clone_boom'):
            marks = [(event['tick'], event['visual_tick']) for event in events if event['action_id'] == action_id]
            self.assertEqual(marks, [(0, 0)] * 10 + [(0, 5)] * 4)
        self.assertEqual(result['summary']['duration_ticks'], 16)

    def test_triangle_merges_equal_damage_without_hiding_rating_changes(self):
        source = (ROOT / 'app/modules/shaft/static/js/shaft.js').read_text()

        def function(name):
            start = source.index('  function ' + name + '(')
            return source[start:source.index('\n  function ', start + 1)]

        functions = '\n'.join(function(name) for name in [
            'escapeHtml', 'displayText', 'isDigitalAfterimageDamage',
            'summarizeTimelineDamageEvents', 'groupedTimelineDamageEvents',
            'damageMarkerTooltip', 'damageEventSummaryRows',
        ])
        for nodes, frames in [([5], 0), ([2, 5], 100)]:
            with self.subTest(nodes=nodes):
                result = self.own([('q', 0)], nodes=nodes, initial={'帧数': frames, '评分': 0})
                events = result['periodic_damage_events']
                script = functions + """
                    const formatNumber = (value, digits = 0) => Number(value).toFixed(digits);
                    const ticksToSeconds = (ticks) => ticks / 10;
                    const events = JSON.parse(require('fs').readFileSync(0, 'utf8'));
                    const marker = groupedTimelineDamageEvents(events)[0];
                    console.log(JSON.stringify({
                      entries: marker.events,
                      tooltip: damageMarkerTooltip(marker),
                      rows: marker.events.map(damageEventSummaryRows),
                      rawEvents: events,
                    }));
                """
                rendered = json.loads(subprocess.check_output(
                    ['node', '-e', script], input=json.dumps(events), text=True,
                ))
                self.assertEqual(rendered['rawEvents'], events)
                expected = {}
                for event in events:
                    key = event['reaction'], event['damage']
                    expected[key] = expected.get(key, 0) + 1
                entries = rendered['entries']
                self.assertEqual(
                    {(entry['reaction'], entry['damage']): entry['display_count'] for entry in entries},
                    expected,
                )
                self.assertEqual(len(entries), len(expected))
                self.assertEqual(
                    rendered['tooltip'].count(' 伤害'), len(entries),
                )
                if nodes == [5]:
                    self.assertEqual(list(expected.values()), [4, 4])
                    self.assertEqual(rendered['tooltip'].count('× 4次'), 2)
                    self.assertTrue(all('× 4次' in row for row in rendered['rows']))
                else:
                    self.assertGreater(len(entries), 2)

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
        self.assertEqual(normalize_axis_payload(axis)['options']['loop_initial_resources'][CID]['personal_resources'],{'帧数':100,'评分':5000})


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
