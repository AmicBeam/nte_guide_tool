"""Manual resonance overrides are user-requested simulator controls (2026-09-22)."""
import unittest
from copy import deepcopy

from app.modules.shaft.domain.buffs import registered_buff_rules, event_matches_rule
from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import normalize_axis_payload, normalize_axis_for_hash, simulate_shaft_axis


class ShaftResonanceTest(unittest.TestCase):
    def payload(self, nodes, resonances=None):
        member = dict(slot=0, character_id='char_akane', arc_id='', cartridge_id='',
                      awakening_nodes=nodes, skill_levels=dict(basic=1, skill=1, ultimate=1, support=1))
        if resonances is not None:
            member['awakening_resonances'] = resonances
        return dict(team=[member], steps=[dict(id='e', slot=0, action_id='action_akane_e', start_tick=0)])

    def test_three_resonance_controls_skill_without_attack_bonus_or_changing_nodes(self):
        # User Oct 11 notice removes the 10% attack buff; level and resource remain separate.
        for nodes in ([], [1, 2, 3], list(range(1, 7))):
            off = simulate_shaft_axis(self.payload(nodes, []))['result']['details'][0]
            on = simulate_shaft_axis(self.payload(nodes, [3]))['result']['details'][0]
            self.assertEqual(off['formula_parts']['skill_level_multiplier'], 1)
            self.assertGreater(on['formula_parts']['skill_level_multiplier'], 1)
            self.assertEqual(on['panel']['atk'], off['panel']['atk'])
            self.assertNotIn('character_akane_three', {b['rule_id'] for b in on['applied_buffs']})

    def test_three_resonance_controls_initial_resources_independently_of_six(self):
        for nodes in ([], [1, 2, 3], list(range(1, 7))):
            for resonances in ([], [3], [6], [3, 6]):
                with self.subTest(nodes=nodes, resonances=resonances):
                    result = simulate_shaft_axis(self.payload(nodes, resonances))['result']
                    expected = 30 if 3 in resonances else 0
                    self.assertEqual(result['resources_by_slot'][0]['initial_personal_resources']['弦能'], expected)
                    self.assertEqual(result['details'][0]['personal_resources_after']['弦能'], 30 if expected else 6)
                    self.assertAlmostEqual(result['details'][0]['formula_parts']['skill_level_multiplier'],
                                           1.08 if 3 in resonances else 1)

    def test_three_resonance_does_not_refill_loop_start(self):
        p = self.payload(list(range(1, 7)), [3, 6])
        p['options'] = {'loop_enabled': True, 'loop_initial_resources': {
            'char_akane': {'personal_resources': {'弦能': 7}}}}
        result = simulate_shaft_axis(p)['result']
        self.assertEqual(result['resources_by_slot'][0]['initial_personal_resources']['弦能'], 7)
        self.assertEqual(result['details'][0]['personal_resources_after']['弦能'], 13)

    def test_disabled_six_resonance_uses_base_field_even_with_six_nodes(self):
        for nodes, resonances, suffix in ((list(range(1, 7)), [], 'base'), ([], [6], 'full')):
            p = self.payload(nodes, resonances)
            p['team'][0]['character_id'] = 'char_0846d632e0'
            p['steps'][0]['action_id'] = 'action_lingke_q'
            detail = simulate_shaft_axis(p)['result']['details'][0]
            field_rules = {b['rule_id'] for b in detail['triggered_buffs']
                           if b['rule_id'].startswith('character_lingke_resonance_field_')}
            self.assertEqual(field_rules, {'character_lingke_resonance_field_' + suffix})

    def test_legacy_payload_matches_automatic_thresholds(self):
        for nodes in ([], [2, 4, 6], list(range(1, 7))):
            legacy = simulate_shaft_axis(self.payload(nodes))['result']
            explicit = simulate_shaft_axis(self.payload(nodes, [n for n in (3, 6) if len(nodes) >= n]))['result']
            self.assertEqual(legacy['summary'], explicit['summary'])

    def test_build_roundtrip_and_hash(self):
        p = self.payload([1, 2, 3], [6, 6, 3, 9, 'bad'])
        normalized = normalize_axis_payload(p)
        self.assertEqual(normalized['team'][0]['awakening_resonances'], [3, 6])
        self.assertEqual(normalized['character_builds']['char_akane']['awakening_resonances'], [3, 6])
        p['character_builds'] = {'char_akane': dict(p['team'][0], awakening_resonances=[])}
        normalized = normalize_axis_payload(p)
        self.assertEqual(normalized['team'][0]['awakening_resonances'], [])
        self.assertEqual(normalize_axis_payload(normalized)['team'][0]['awakening_resonances'], [])
        changed = deepcopy(normalized)
        changed['team'][0]['awakening_resonances'] = [6]
        self.assertEqual(normalize_axis_for_hash(normalized), normalize_axis_for_hash(changed))

    def test_python_buff_conditions_follow_override(self):
        catalog = load_shaft_catalog()
        for nodes, resonances, expected in (([], [6], True), (list(range(1, 7)), [], False)):
            rules = registered_buff_rules(self.payload(nodes, resonances)['team'], catalog)
            rule = next(r for r in rules if r.get('id') == 'character_akane_six')
            self.assertEqual(event_matches_rule(rule, 'action_hit', {'slot': 0}, {'id': 'action_akane_short_enhanced_e'}, {}, False), expected)
