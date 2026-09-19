"""Everness 延时预警, verified 2026-09-19: team delay lasts 12 seconds."""
import unittest

from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import simulate_shaft_axis


class HathorDelayTest(unittest.TestCase):
    def team(self, hathor):
        ids = ['char_dd034941ef', 'char_d38b672525']
        if hathor:
            ids.append('char_912dbfe17c')
        return [dict(slot=i, character_id=id, arc_id='', cartridge_id='') for i, id in enumerate(ids)]

    def test_teammate_reaction_and_crit_window(self):
        catalog = load_shaft_catalog()
        support = next(a['id'] for a in catalog['actions_by_character']['char_d38b672525']
                       if a['name'] == '援护')
        for hathor, duration in ((False, 50), (True, 120)):
            with self.subTest(hathor=hathor):
                result = simulate_shaft_axis({
                    'team': self.team(hathor),
                    'steps': [
                        {'id': 'gain', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                        {'id': 'support', 'slot': 1, 'action_id': support, 'start_tick': 30},
                        *[{'id': f'probe{tick}', 'slot': 0, 'action_id': 'action_982c67944f',
                           'start_tick': tick} for tick in (90, 149, 170)],
                    ],
                })['result']
                effect = next(e for e in result['reaction_effects'] if e['reaction'] == '延滞')
                self.assertEqual(effect['end_tick'] - effect['start_tick'], duration)
                details = {d['step_id']: d for d in result['details']}
                for tick in (90, 149, 170):
                    buffs = {b['rule_id'] for b in details[f'probe{tick}']['applied_buffs']}
                    self.assertEqual('character_hathor_delay_crit' in buffs, hathor and tick <= effect['end_tick'])

    def test_loop_initial_inherits_remaining_duration(self):
        for hathor, remaining in ((False, 20), (True, 90)):
            with self.subTest(hathor=hathor):
                result = simulate_shaft_axis({
                    'team': self.team(hathor),
                    'steps': [{'id': 'end', 'slot': 0, 'action_id': 'action_none_dd034941ef', 'start_tick': 30}],
                    'options': {'loop_enabled': True, 'loop_initial_resources': {
                        'char_dd034941ef': {'reaction': '延滞'},
                    }},
                })['result']
                effect = next(e for e in result['reaction_effects'] if e.get('loop_initial'))
                self.assertEqual(effect['start_tick'], 0)
                self.assertEqual(effect['end_tick'], remaining)

    def test_retired_settlement_does_not_reapply_delay(self):
        result = simulate_shaft_axis({
            'team': self.team(True),
            'steps': [
                {'id': 'old', 'slot': 1, 'action_id': 'action_be26cd6a7c', 'start_tick': 0},
                {'id': 'probe', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 60},
            ],
        })['result']
        self.assertEqual(len(result['details']), 1)
        self.assertNotIn('character_hathor_delay_crit',
                         {b['rule_id'] for b in result['details'][0]['applied_buffs']})
