"""Consumption buffs must see stack sources carried from an earlier cycle."""
import unittest

from app.modules.shaft.service import simulate_shaft_axis


class ShaftLoopStackMemoryTest(unittest.TestCase):
    def result(self, nodes=(1, 3), loop=True, end=222, resonances=(), opening=False):
        return simulate_shaft_axis({
            'team': [dict(slot=0, character_id='char_elyms', arc_id='', cartridge_id='',
                          awakening_nodes=list(nodes), awakening_resonances=list(resonances))],
            'steps': ([dict(id='opening',slot=0,action_id='action_elyms_meleeb1',start_tick=0)] if opening else [])
                     + [dict(id=str(i), slot=0, action_id='action_elyms_'+name, start_tick=tick)
                      for i, (name, tick) in enumerate([
                          ('ultraskillb1',17), ('skill2',27), ('goodbye',98)])]
                     + [dict(id='end', slot=0, action_id='action_none_elyms', start_tick=end)],
            'options': {'loop_enabled':loop, 'loop_initial_resources':{
                'char_elyms': {'energy':100, 'harmony':0,
                               'personal_resources':{'信号等级':2}}}},
        })['result']

    def buff(self, detail, key):
        return next(b for b in detail['triggered_buffs'] if b['rule_id']==key)

    def test_a_reads_four_consumed_layers_from_previous_preconditioned_cycle(self):
        result=self.result()
        self.assertEqual(result['summary']['duration_ticks'],207)
        q, e2, e3=result['details'][:3]
        self.assertEqual([d['start_tick'] for d in (q,e2,e3)],[17,22,88])
        self.assertEqual(self.buff(e2,'character_elyms_illusion_normal')['stack_count'],4)
        self.assertEqual(self.buff(e3,'character_elyms_accumulate')['stack_count'],4)
        self.assertEqual(self.buff(q,'character_elyms_a')['effects'],{'crit_dmg':.4})
        resources=result['resources_by_slot'][0]
        self.assertEqual(resources['initial_energy'],100)
        self.assertEqual(resources['initial_harmony'],0)
        self.assertEqual(resources['initial_personal_resources']['信号等级'],2)
        self.assertEqual(len(result['details']),4)
        for key in ['energy_events','harmony_events','personal_resource_state_events']:
            self.assertTrue(all(e['tick']>=0 for e in result[key]))

    def test_f_uses_the_same_four_consumed_layers(self):
        result=self.result(nodes=(3,6))
        self.assertEqual(self.buff(result['details'][0],'character_elyms_f')['effects'],{'follow_dmg':1})

    def test_opening_inherits_the_previous_q_bonus_before_this_q(self):
        result=self.result(opening=True)
        opening=result['details'][0]
        buff=next(b for b in opening['applied_buffs'] if b['rule_id']=='character_elyms_a')
        self.assertEqual(buff['effects'],{'crit_dmg':.4})
        self.assertEqual(buff['end_tick'],10)

    def test_expired_illusion_does_not_gain_phantom_layers(self):
        result=self.result(end=255)
        self.assertEqual(self.buff(result['details'][2],'character_elyms_accumulate')['stack_count'],1)
        self.assertEqual(self.buff(result['details'][0],'character_elyms_a')['effects'],{'crit_dmg':.1})

    def test_non_loop_opening_q_does_not_receive_future_consumption(self):
        result=self.result(loop=False)
        self.assertEqual(self.buff(result['details'][0],'character_elyms_a')['effects'],{})

    def test_three_resonance_already_gives_four_layers_without_extra_stacking(self):
        result=self.result(resonances=(3,))
        self.assertEqual(self.buff(result['details'][2],'character_elyms_accumulate')['stack_count'],4)
        self.assertEqual(self.buff(result['details'][0],'character_elyms_a')['effects'],{'crit_dmg':.4})
