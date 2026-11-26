import json
import subprocess
import unittest
from copy import deepcopy
from pathlib import Path

from app.modules.shaft.domain.catalog import load_shaft_catalog

ROOT = Path(__file__).resolve().parents[1]
CID = 'char_elyms'


class ShaftDamageZonesTest(unittest.TestCase):
    def simulate(self, effects, reaction='', kind='普攻', mind=False, periodic=False):
        catalog = deepcopy(load_shaft_catalog())
        if reaction == '覆纹':
            next(c for c in catalog['characters'] if c['id'] == CID)['element'] = '灵'
        catalog['buffs'] = [{
            'id': 'probe', 'name': 'probe', 'providers': [{'kind': 'character', 'id': CID}],
            'trigger': {'event': 'passive'}, 'target': {'scope': 'team'},
            'duration': {'type': 'permanent'}, 'effects': effects,
        }]
        action = deepcopy(next(a for a in catalog['actions'] if a['id'] == 'action_elyms_melee1'))
        action.update(id='probe-action', action_type=kind, damage_type=kind, stagger=5,
                      duration_ticks=10, duration_seconds=1, duration_pending=False,
                      self_modifiers={}, tags=['心灵'] if mind else [])
        if periodic:
            action['periodic_damage'] = {'interval_ticks': 10, 'tick_count': 3, 'multipliers': {'atk': .5}}
        catalog['actions'].append(action)
        axis = {'team': [{'slot': 0, 'character_id': CID, 'arc_id': '', 'cartridge_id': ''}],
                'steps': [{'id': 'hit', 'slot': 0, 'action_id': action['id'], 'start_tick': 0},
                          {'id': 'tail', 'slot': 0, 'action_id': 'action_none_elyms', 'start_tick': 40}],
                'options': {'loop_enabled': True,
                            'loop_initial_resources': {CID: {'reaction': reaction, 'energy': 100}}}}
        raw = subprocess.check_output(['node', str(ROOT / 'scripts/shaft_compute.js')],
            input=json.dumps({'axis': axis, 'catalog': catalog}), text=True)
        result = json.loads(raw)
        self.assertTrue(result['ok'], result)
        return result['result']

    def test_character_periodic_damage_uses_general_resistance_reduction(self):
        base = self.simulate({}, periodic=True)
        pen = self.simulate({'res_down_光': .2}, periodic=True)
        self.assertTrue(base['periodic_damage_events'])
        self.assertEqual(len(base['periodic_damage_events']), len(pen['periodic_damage_events']))
        for old, new in zip(base['periodic_damage_events'], pen['periodic_damage_events']):
            self.assertGreater(new['damage'], old['damage'])
            self.assertAlmostEqual(new['formula_parts']['res_down'], .2)

    def test_arc_refinements_use_general_resistance_reduction(self):
        c = load_shaft_catalog()
        rule = next(b for b in c['buffs'] if b['id'] == 'arc_elyms_light_pen')
        self.assertEqual(rule['effects'], {'res_down_光': .08})
        levels = c['arc_refinements']['arcs']['arc_elyms_waiting_ringtone']['levels']
        for level, value in [('1', .08), ('5', .2)]:
            self.assertEqual(levels[level]['buff_effects']['arc_elyms_light_pen'], {'res_down_光': value})

    def test_three_groups_are_additive_and_mind_is_seventh_element(self):
        effects = {'all_dmg': .2, 'basic_dmg': .3, 'follow_dmg': .4, 'other_dmg': .5,
                   'action_dmg': .1, 'mind_dmg': .6, 'element_dmg_光': .7}
        detail = self.simulate(effects)['details'][0]
        self.assertAlmostEqual(detail['panel']['all_dmg'], .2)
        self.assertAlmostEqual(detail['panel']['action_dmg'], .4)
        self.assertAlmostEqual(detail['panel']['other_dmg'], .5)
        self.assertAlmostEqual(detail['formula_parts']['dmg_bonus'], 1.1 + detail['panel']['element_dmg_光'])
        mind = self.simulate(effects, mind=True)['details'][0]
        self.assertAlmostEqual(mind['panel']['mind_dmg'], .6)
        self.assertAlmostEqual(mind['panel']['action_dmg'], .4)
        self.assertAlmostEqual(mind['formula_parts']['dmg_bonus'], 1.7)

    def test_general_resistance_reduction_includes_reaction_fuwen_and_stagger(self):
        for reaction in ('浊燃', '覆纹'):
            with self.subTest(reaction=reaction):
                base = self.simulate({}, reaction)
                pen = self.simulate({'res_down': .2}, reaction)
                self.assertGreater(pen['details'][0]['direct_damage'], base['details'][0]['direct_damage'])
                self.assertGreater(pen['summary']['stagger_damage'], base['summary']['stagger_damage'])
                if reaction == '覆纹':
                    self.assertGreater(base['details'][0]['fuwen_damage'], 0)
                    self.assertGreater(pen['details'][0]['fuwen_damage'], base['details'][0]['fuwen_damage'])
                else:
                    self.assertTrue(base['reaction_damage_events'])
                    for old, new in zip(base['reaction_damage_events'], pen['reaction_damage_events']):
                        self.assertGreater(new['damage'], old['damage'])
        for kind in ('环合', '倾陷'):
            self.assertGreater(self.simulate({'res_down_光': .2}, kind=kind)['details'][0]['direct_damage'],
                               self.simulate({}, kind=kind)['details'][0]['direct_damage'])
        base = self.simulate({}, mind=True)['details'][0]
        light = self.simulate({'res_down_光': .2}, mind=True)['details'][0]
        self.assertEqual(base['direct_damage'], light['direct_damage'])
