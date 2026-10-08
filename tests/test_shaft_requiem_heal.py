"""User 2026-09-20: E-awakened nightmare ticks count as healing, without a 3s timer."""
import unittest

from app.modules.shaft.service import simulate_shaft_axis


class RequiemHealTest(unittest.TestCase):
    team_rule = 'arc_wrong_door_heal_team_damage'
    self_rule = 'arc_wrong_door_heal_spirit'

    def simulate(self, nodes=(5,), refinement=1, equipped=True, settlement=False):
        steps = [dict(id='gain', slot=0, action_id='action_00edea34a8', start_tick=0),
                 dict(id='before', slot=1, action_id='action_982c67944f', start_tick=2),
                 dict(id='inside', slot=1, action_id='action_982c67944f', start_tick=40),
                 dict(id='outside', slot=1, action_id='action_982c67944f', start_tick=250)]
        if settlement:
            steps.insert(2, dict(id='settle', slot=0, action_id='action_a7443f657f', start_tick=5))
        return simulate_shaft_axis({
            'team': [dict(slot=0, character_id='char_c78f7a08d5',
                          arc_id='arc_bab179ec33' if equipped else '', arc_refinement=refinement,
                          cartridge_id='', awakening_nodes=list(nodes)),
                     dict(slot=1, character_id='char_dd034941ef', arc_id='', cartridge_id='')],
            'steps': steps,
        })['result']

    def test_every_nightmare_tick_triggers_and_refreshes_other_team_damage(self):
        for refinement, bonus in ((1, .15), (5, .30)):
            with self.subTest(refinement=refinement):
                result = self.simulate(refinement=refinement)
                events = [e for e in result['periodic_damage_events'] if e['reaction']=='噩梦' and e['damage']>0]
                self.assertEqual([e['tick'] for e in events], [10, 20, 30])
                for event in events:
                    buffs = {b['rule_id']: b for b in event['triggered_buffs']}
                    self.assertEqual(buffs[self.team_rule]['effects'], {'all_dmg': bonus})
                    self.assertEqual(buffs[self.team_rule]['end_tick'], event['tick']+200)
                    self.assertIn(self.self_rule, buffs)
                    self.assertNotIn(self.team_rule, {b['rule_id'] for b in event['applied_buffs']})
                details = {d['step_id']: d for d in result['details']}
                for key in ('before', 'outside'):
                    self.assertNotIn(self.team_rule, {b['rule_id'] for b in details[key]['applied_buffs']})
                buff, = [b for b in details['inside']['applied_buffs'] if b['rule_id']==self.team_rule]
                self.assertEqual(buff['stack_count'], 1)
                self.assertAlmostEqual(details['inside']['panel']['all_dmg']-details['outside']['panel']['all_dmg'], bonus)

    def test_without_e_or_equipment_does_not_trigger(self):
        for kwargs in ({'nodes': ()}, {'nodes': (1, 2, 3, 4, 6)}, {'equipped': False}):
            with self.subTest(kwargs=kwargs):
                result = self.simulate(**kwargs)
                self.assertFalse(any(b['rule_id']==self.team_rule for e in result['periodic_damage_events']
                                     for b in e.get('triggered_buffs', [])))

    def test_c_awakening_nightmare_settlement_also_triggers(self):
        result = self.simulate(nodes=(3, 5), settlement=True)
        event = next(e for e in result['periodic_damage_events'] if e['kind']=='buff_periodic_settlement')
        self.assertGreater(event['damage'], 0)
        self.assertIn(self.team_rule, {b['rule_id'] for b in event['triggered_buffs']})
