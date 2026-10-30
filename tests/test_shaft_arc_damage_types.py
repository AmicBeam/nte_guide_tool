"""Arc damage bonuses cover passive damage without turning it into a skill cast."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import unittest

from app.modules.shaft.domain.buffs import active_buff_applies, event_matches_rule
from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import simulate_shaft_axis

ROOT = Path(__file__).resolve().parents[1]
TIME = 'arc_7d0ca08e3d'
RECORD = 'arc_614aec6ed1'
READY = 'arc_ce53905d70'
ORA = 'arc_4d19ef5194'
PLANE = 'arc_edbcee2b65'


class ArcDamageTypesTest(unittest.TestCase):
    def simulate(self, arc_id, specs, disabled=False, extra_effects=None):
        catalog = deepcopy(load_shaft_catalog())
        arc = next(a for a in catalog['arcs'] if a['id'] == arc_id)
        character = next(c for c in catalog['characters'] if c['adaptation'] == arc['adaptation'])
        steps = []
        for i, spec in enumerate(specs):
            action_type, damage_type, tick, element = spec[:4]
            action = dict(id=f'probe-{i}', name=f'probe-{i}', character_id=character['id'],
                          character_name=character['name'], action_type=action_type,
                          damage_type=damage_type, damage_element=element,
                          is_background_damage=action_type == '被动', duration_ticks=0,
                          duration_seconds=0, duration_implemented=True, hit_count=1,
                          tags=spec[4] if len(spec) > 4 else [],
                          multipliers={'atk': 1, 'hp': 0, 'def': 0, 'flat': 0})
            catalog['actions'].append(action)
            steps.append(dict(id=str(i), slot=0, action_id=action['id'], start_tick=tick))
        if disabled:
            catalog['buffs'] = [b for b in catalog['buffs'] if not any(
                p.get('id') == arc_id for p in b.get('providers', []))]
        if extra_effects:
            catalog['buffs'].append(dict(id='probe-effects', name='probe-effects',
                providers=[dict(kind='character', id=character['id'])],
                trigger=dict(event='passive'), target=dict(scope='registrar'),
                duration=dict(type='permanent'), effects=extra_effects))
        axis = dict(team=[dict(slot=0, character_id=character['id'], arc_id=arc_id,
                               arc_refinement=1, cartridge_id='', awakening_nodes=[],
                               awakening_resonances=[])], steps=steps, options={})
        raw = subprocess.check_output(['node', str(ROOT / 'scripts/shaft_compute.js')],
                                      input=json.dumps(dict(axis=axis, catalog=catalog)), text=True)
        result = json.loads(raw)
        self.assertTrue(result['ok'], result)
        return result['result']['details']

    def effects(self, detail, rule_id):
        return next((b['effects'] for b in detail['applied_buffs'] if b['rule_id'] == rule_id), None)

    def test_time_outside_applies_to_both_real_memories_at_all_refinements_and_stacks(self):
        for refinement in range(1, 6):
            for stacks in (0, 1, 2, 3):
                for suffix in ('a', 'b'):
                    with self.subTest(refinement=refinement, stacks=stacks, suffix=suffix):
                        actions = [(0, 'action_elyms_skill1', 0)]
                        actions += [(1, 'action_982c67944f', 20 + i * 20) for i in range(stacks)]
                        actions += [(0, f'action_elyms_ultraskill{suffix}1', 100),
                                    (1, 'action_none_dd034941ef', 110),
                                    (0, f'action_elyms_xietongskill{suffix}1', 120)]
                        result = simulate_shaft_axis(dict(
                            team=[dict(slot=0, character_id='char_elyms', arc_id=TIME,
                                       arc_refinement=refinement, cartridge_id='', awakening_nodes=[]),
                                  dict(slot=1, character_id='char_dd034941ef', arc_id='', cartridge_id='')],
                            steps=[dict(id=str(i), slot=s, action_id=a, start_tick=t)
                                   for i, (s, a, t) in enumerate(actions)]))['result']
                        q, memory = result['details'][-3], result['details'][-1]
                        expected = (.24 + .06 * (refinement - 1)) + stacks * (.08 + .02 * (refinement - 1))
                        for detail in (q, memory):
                            effects = self.effects(detail, 'arc_time_outside_recorded_crit_def')
                            self.assertIsNotNone(effects)
                            self.assertAlmostEqual(effects['crit_dmg'], expected)
                            self.assertAlmostEqual(effects.get('def_ignore', 0),
                                                   .12 + .03 * (refinement - 1) if stacks == 3 else 0)
                        self.assertAlmostEqual(memory['panel']['crit_dmg'] - result['details'][0]['panel']['crit_dmg'], expected)
                        self.assertFalse(any(b['rule_id'].startswith('arc_time_outside_')
                                             for b in memory['triggered_buffs']))

    def test_time_reward_survives_new_maze_and_expires_without_passive_q_consuming_it(self):
        specs = [('E', 'E', 0, '光'), ('Q', 'Q', 10, '光'),
                 ('E', 'E', 20, '光'), ('被动', 'Q', 30, '光'),
                 ('被动', 'Q', 710, '光'), ('Q', 'Q', 720, '光')]
        details = self.simulate(TIME, specs)
        reward = 'arc_time_outside_recorded_crit_def'
        self.assertIsNotNone(self.effects(details[3], reward))
        self.assertIsNone(self.effects(details[4], reward))
        self.assertIsNone(self.effects(details[2], reward))
        self.assertTrue(any(b['rule_id'] == reward for b in details[5]['triggered_buffs']))

    def test_ready_bonuses_cover_passive_basic_and_counter_damage_but_need_a_real_cast(self):
        specs = [('被动', 'E', 0, '灵'), ('被动', '普攻', 10, '灵'),
                 ('E', 'E', 20, '灵'), ('被动', '普攻', 30, '灵'),
                 ('被动', '闪反', 40, '灵'), ('被动', 'Q', 50, '灵')]
        active, control = self.simulate(READY, specs), self.simulate(READY, specs, disabled=True)
        for i in range(len(specs)):
            with self.subTest(step=i):
                expected = .15 if i in (3, 4) else 0
                self.assertAlmostEqual(active[i]['formula_parts']['dmg_bonus'] - control[i]['formula_parts']['dmg_bonus'], expected)

    def test_ora_stacks_from_passive_basic_hits_and_only_buffs_basic_damage(self):
        specs = [('被动', '普攻', 0, '相'), ('被动', '普攻', 10, '相'),
                 ('被动', 'E', 20, '相'), ('被动', '普攻', 30, '相')]
        active, control = self.simulate(ORA, specs), self.simulate(ORA, specs, disabled=True)
        for i, expected in enumerate((0, .02, 0, .04)):
            with self.subTest(step=i):
                self.assertAlmostEqual(active[i]['formula_parts']['dmg_bonus'] - control[i]['formula_parts']['dmg_bonus'], expected)

    def test_gold_record_support_and_ultimate_damage_include_passive_and_legacy_followups(self):
        specs = [('被动', '援护', 0, '灵'), ('被动', 'Q', 10, '灵'),
                 ('Q', '追击', 20, '灵'), ('被动', '追击', 25, '灵'),
                 ('被动', 'Q', 30, '灵')]
        active, control = self.simulate(RECORD, specs), self.simulate(RECORD, specs, disabled=True)
        for i, expected in enumerate((.66, .22, .22, 0, 0)):
            with self.subTest(step=i):
                self.assertAlmostEqual(active[i]['panel']['crit_dmg'] - control[i]['panel']['crit_dmg'], expected)

    def test_paperplane_checks_passive_damage_category_and_actual_element(self):
        specs = [('被动', 'E', 0, '灵'), ('被动', 'Q', 10, '灵'),
                 ('被动', 'E', 20, '相'), ('被动', '普攻', 30, '灵'),
                 ('被动', '援护', 40, '灵')]
        active, control = self.simulate(PLANE, specs), self.simulate(PLANE, specs, disabled=True)
        for i in range(len(specs)):
            with self.subTest(step=i):
                self.assertAlmostEqual(active[i]['formula_parts']['dmg_bonus'] - control[i]['formula_parts']['dmg_bonus'], .2 if i < 2 else 0)

    def test_python_target_and_trigger_classification_matches_damage_categories(self):
        step = dict(id='probe', slot=0)
        snapshot = dict(character=dict(element='灵'))
        cases = [('被动', 'Q', [], 'Q'), ('被动', '援护', [], '援护'),
                 ('Q', '追击', [], 'Q'), ('被动', 'Q', ['追击'], '追击'),
                 ('被动', 'Q', ['附着'], '附着')]
        for action_type, damage_type, tags, category in cases:
            with self.subTest(action_type=action_type, damage_type=damage_type, category=category):
                action = dict(action_type=action_type, damage_type=damage_type, tags=tags)
                rule = dict(owner_slot=0, trigger=dict(event='action_hit', source=dict(damage_types=[category])),
                            target=dict(damage_types=[category]))
                self.assertTrue(active_buff_applies(dict(owner_slot=0, rule=rule), step, action, snapshot, True))
                self.assertTrue(event_matches_rule(rule, 'action_hit', step, action, snapshot, True))
                rule['target'] = dict(damage_types=['闪反'])
                self.assertFalse(active_buff_applies(dict(owner_slot=0, rule=rule), step, action, snapshot, True))

    def test_category_modifiers_add_once_for_passive_and_multitype_damage(self):
        specs = [('被动', 'Q', 0, '灵', ['追击']), ('Q', '追击', 10, '灵'),
                 ('被动', '普攻', 20, '灵', ['附着']),
                 ('被动', 'E', 30, '灵', ['追加攻击']), ('Q', 'Q', 40, '灵'),
                 ('被动', '援护', 50, '灵')]
        effects = dict(ultimate_dmg=.2, follow_dmg=.3, attach_dmg=.4,
                       basic_dmg=.1, skill_dmg=.6)
        active = self.simulate(PLANE, specs, disabled=True, extra_effects=effects)
        control = self.simulate(PLANE, specs, disabled=True)
        for i, expected in enumerate((.5, .5, .5, .9, .2, 0)):
            with self.subTest(step=i):
                self.assertAlmostEqual(active[i]['formula_parts']['dmg_bonus'] - control[i]['formula_parts']['dmg_bonus'], expected)
                self.assertAlmostEqual(active[i]['panel']['action_dmg'] - control[i]['panel']['action_dmg'], expected)
