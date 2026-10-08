"""Cartridge integration tests must trigger real reactions, not synthetic debuff tags."""
import unittest

from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import simulate_shaft_axis


class StreetBoxerReactionTest(unittest.TestCase):
    rule = 'cartridge_street_boxer_delay_reaction'

    def payload(self, element='光', equipped=True):
        catalog = load_shaft_catalog()
        previous = next(c for c in catalog['characters'] if c['element'] == element)
        support = next(a for a in catalog['actions_by_character']['char_d38b672525'] if a['name'] == '援护')
        # The wearer is off-field and does not participate in the reaction.
        ids = [previous['id'], 'char_d38b672525', 'char_912dbfe17c']
        return {'team': [dict(slot=i, character_id=id, arc_id='',
                             cartridge_id='cartridge_9229a5376c' if i == 2 and equipped else '')
                         for i, id in enumerate(ids)],
                'steps': [dict(id='start', slot=0, action_id=f"action_none_{previous['id'][5:]}", start_tick=0),
                          dict(id='support', slot=1, action_id=support['id'], start_tick=20),
                          dict(id='inside', slot=2, action_id='action_61c61302f2', start_tick=100),
                          dict(id='outside', slot=2, action_id='action_61c61302f2', start_tick=250)]}

    def test_real_delay_and_stain_draw_twenty_second_wearer_line(self):
        for element, reaction in (('光', '延滞'), ('魂', '浸染')):
            with self.subTest(reaction=reaction):
                result = simulate_shaft_axis(self.payload(element))['result']
                details = {d['step_id']: d for d in result['details']}
                support = details['support']
                self.assertEqual(support['triggered_reaction']['reaction'], reaction)
                buff, = [b for b in support['triggered_buffs'] if b['rule_id'] == self.rule]
                self.assertEqual(buff['owner_slot'], 2)
                self.assertEqual(buff['end_tick'] - buff['start_tick'], 200)
                self.assertEqual(buff['start_tick'], support['triggered_reaction']['start_tick'])
                self.assertTrue(buff['display_as_line'])
                self.assertAlmostEqual(buff['effects']['crit_rate'], .14)
                self.assertIn(self.rule, {b['rule_id'] for b in details['inside']['applied_buffs']})
                self.assertNotIn(self.rule, {b['rule_id'] for b in details['outside']['applied_buffs']})
                self.assertAlmostEqual(details['inside']['panel']['crit_rate'] - details['outside']['panel']['crit_rate'], .24 if reaction == '延滞' else .14)

    def test_no_equipment_or_unrelated_pair_does_not_trigger(self):
        for payload in (self.payload(equipped=False), self.payload('灵')):
            result = simulate_shaft_axis(payload)['result']
            self.assertFalse(any(b['rule_id'] == self.rule for d in result['details'] for b in d['triggered_buffs']))

    def test_refresh_does_not_stack_and_loop_carries_remaining_buff(self):
        payload = self.payload('魂')
        payload['steps'][2:2] = [dict(id='switch', **{k:v for k,v in payload['steps'][0].items() if k!='id'},),
                                dict(id='again', **{k:v for k,v in payload['steps'][1].items() if k!='id'})]
        payload['steps'][2]['start_tick'] = 60
        payload['steps'][3]['start_tick'] = 80
        result = simulate_shaft_axis(payload)['result']
        inside = next(d for d in result['details'] if d['step_id']=='inside')
        buff, = [b for b in inside['applied_buffs'] if b['rule_id']==self.rule]
        self.assertEqual(buff['stack_count'], 1)
        self.assertAlmostEqual(buff['effects']['crit_rate'], .14)
        payload = self.payload('魂')
        payload['steps'] = payload['steps'][:2] + [dict(id='tail', slot=2, action_id='action_none_912dbfe17c', start_tick=100)]
        previous_id = payload['team'][0]['character_id']
        payload['options'] = {'loop_enabled': True, 'loop_initial_resources': {previous_id: {'harmony': 100}}}
        # Inspect wearer at tick zero without changing foreground reaction participants.
        probe_action = next(a['id'] for a in load_shaft_catalog()['actions_by_character']['char_912dbfe17c'] if a['name'] == '优评锁定触发')
        payload['steps'].insert(0, dict(id='probe', slot=2, action_id=probe_action, start_tick=0, placement='background'))
        result = simulate_shaft_axis(payload)['result']
        probe = next(d for d in result['details'] if d['step_id']=='probe')
        self.assertIn(self.rule, {b['rule_id'] for b in probe['applied_buffs']})
