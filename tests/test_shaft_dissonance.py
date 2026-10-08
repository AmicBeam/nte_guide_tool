import unittest

from app.modules.shaft.service import simulate_shaft_axis


class DissonanceTest(unittest.TestCase):
    def simulate(self, *, level=80, outside=False, daphne=False, repeat=False, late=False, loop=False, reverse=False, no_burn=False, rose=False):
        characters = ['char_c78f7a08d5', 'char_1895e259be', 'char_heiyu']
        if daphne:
            characters.append('char_e8ad982185')
        steps = [(0, 'action_none_c78f7a08d5', 0),
                 (1, 'action_dee1ceb970', 2),
                 (0, 'action_none_c78f7a08d5', 20),
                 (2, 'action_heiyu_support', 22)]
        if no_burn:
            steps[1] = (1, 'action_none_1895e259be', 2)
        if reverse:
            steps[1], steps[3] = (2, 'action_heiyu_support', 2), (1, 'action_dee1ceb970', 22)
        if repeat:
            steps += [(0, 'action_none_c78f7a08d5', 42), (2, 'action_heiyu_support', 44)]
        if late:
            steps += [(0, 'action_none_c78f7a08d5', 450), (1, 'action_dee1ceb970', 452),
                      (0, 'action_none_c78f7a08d5', 470), (2, 'action_heiyu_support', 472)]
        steps.append((0, 'action_none_c78f7a08d5', 600 if late else 200))
        return simulate_shaft_axis({
            'team': [dict(slot=i, character_id=c, arc_id='arc_dcd5900afc' if rose and i == 0 else '', cartridge_id='') for i, c in enumerate(characters)],
            'steps': [dict(id=str(i), slot=slot, action_id=action, start_tick=tick)
                      for i, (slot, action, tick) in enumerate(steps)],
            'enemy': {'level': level, 'track_outside': outside},
            'options': {'loop_enabled': loop},
        })['result']

    def test_fifteen_percent_level_gap_uses_enemy_gauge_limit(self):
        # 用户2026-09-24：基础为当时韧性上限的15%。敌人更高且等级差≤5时减半；
        # 等级差>5时只有基础效果的20%。敌人不更高时保持15%。普通70、轨外50。
        for outside, limit in [(False, 70), (True, 50)]:
            for level, ratio in [(70, .15), (80, .15), (85, .075), (86, .03), (90, .03)]:
                with self.subTest(outside=outside, level=level):
                    r = self.simulate(level=level, outside=outside)
                    self.assertEqual(r['summary']['stagger_limit'], limit)
                    self.assertEqual(len(r['dissonance_events']), 2)
                    for event in r['dissonance_events']:
                        self.assertAlmostEqual(event['stagger_amount'], limit * ratio)
                        self.assertEqual(event['daphne_extra_stagger'], 0)
                    self.assertAlmostEqual(r['summary']['total_stagger'],
                        sum(d['stagger_amount'] for d in r['details']) + limit * ratio * 2)
                    self.assertAlmostEqual(r['summary']['stagger_frequency'], 1 / (
                        limit / r['summary']['effective_stagger'] + 10 / r['summary']['duration_seconds']))

    def test_recast_triggers_once_and_requires_active_burn(self):
        r = self.simulate()
        self.assertEqual([e['heiyu_recast'] for e in r['dissonance_events']], [False, True])
        self.assertEqual(len({e['effect_id'] for e in r['dissonance_events']}), 2)
        r = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_c78f7a08d5'},
                     {'slot': 1, 'character_id': 'char_heiyu'}],
            'steps': [{'id': 'front', 'slot': 0, 'action_id': 'action_none_c78f7a08d5', 'start_tick': 0},
                      {'id': 'support', 'slot': 1, 'action_id': 'action_heiyu_support', 'start_tick': 2},
                      {'id': 'end', 'slot': 0, 'action_id': 'action_none_c78f7a08d5', 'start_tick': 200}],
        })['result']
        self.assertEqual(r['dissonance_events'], [])

    def test_burn_added_to_dark_star_also_triggers(self):
        r = self.simulate(reverse=True)
        self.assertEqual(r['dissonance_events'][0]['trigger_reaction'], '浊燃')
        self.assertTrue(r['dissonance_events'][1]['heiyu_recast'])

    def test_daphne_extra_removal_ignores_level_gap_at_max_stacks(self):
        for outside, limit in [(False, 70), (True, 50)]:
            r = self.simulate(level=90, outside=outside, daphne=True, repeat=True)
            events = r['dissonance_events']
            self.assertEqual([e['daphne_stacks'] for e in events], [1, 2, 2, 2])
            for event in events:
                self.assertAlmostEqual(event['daphne_extra_stagger'], limit * .1)
                self.assertAlmostEqual(event['dissonance_stagger'], event['stagger_limit_before'] * .15 * .2)
            self.assertAlmostEqual(events[-1]['stagger_limit_after'], limit * .8)
            self.assertGreater(r['summary']['effective_stagger'], r['summary']['total_stagger'])

    def test_daphne_cap_reduction_expires_and_restarts(self):
        r = self.simulate(daphne=True, repeat=True, late=True)
        event = next(e for e in r['dissonance_events'] if e['tick'] > 450)
        self.assertEqual(event['stagger_limit_before'], 70)
        self.assertEqual(event['daphne_stacks'], 1)
        self.assertEqual(event['daphne_expires_at'], event['tick'] + 300)

    def test_daphne_reduced_cap_does_not_reduce_damage_per_stagger(self):
        # 用户2026-09-18：破鞘改变削条频率，不能降低单次倾陷伤害。
        for outside, limit in [(False, 70), (True, 50)]:
            base = self.simulate(daphne=True, outside=outside, no_burn=True)
            repeated = self.simulate(daphne=True, outside=outside, repeat=True)
            self.assertEqual(repeated['dissonance_events'][-1]['stagger_limit_after'], limit * .8)
            self.assertAlmostEqual(repeated['summary']['stagger_damage_per_trigger'],
                                   base['summary']['stagger_damage_per_trigger'])
            self.assertGreater(repeated['summary']['stagger_frequency'], base['summary']['stagger_frequency'])
            self.assertGreater(repeated['summary']['stagger_damage'], base['summary']['stagger_damage'])

    def test_daphne_recovery_time_scales_with_cap_and_real_time_coverage(self):
        for rose, recovery in [(False, 10), (True, 13)]:
            with self.subTest(rose=rose):
                r = self.simulate(daphne=True, rose=rose)
                events = r['dissonance_events']
                self.assertEqual([e['recovery_scale_after'] for e in events], [.9, .8])
                duration = r['summary']['duration_ticks']
                first, second = [e['tick'] for e in events]
                expected_scale = (first + (second - first) * .9 + (duration - second) * .8) / duration
                self.assertAlmostEqual(r['summary']['stagger_recovery_seconds'], 10 * expected_scale + (recovery - 10))
                self.assertEqual(r['summary']['base_stagger_recovery_seconds'], recovery)
                self.assertAlmostEqual(r['summary']['stagger_frequency'], 1 / (
                    r['summary']['stagger_limit'] / r['summary']['effective_stagger']
                    + (10 * expected_scale + (recovery - 10)) / r['summary']['duration_seconds']))
                untouched = self.simulate(daphne=True, no_burn=True, rose=rose)
                self.assertEqual(untouched['summary']['stagger_recovery_seconds'], recovery)

    def test_daphne_recovery_returns_to_base_after_expiry(self):
        r = self.simulate(daphne=True, late=True)
        first, second, third, fourth = r['dissonance_events']
        duration = r['summary']['duration_ticks']
        # First two layers expire before the late pair; the uncovered gap uses full recovery.
        expected_ticks = (first['tick'] + (second['tick'] - first['tick']) * .9
                          + 300 * .8 + third['tick'] - second['daphne_expires_at']
                          + (fourth['tick'] - third['tick']) * .9
                          + (duration - fourth['tick']) * .8)
        self.assertAlmostEqual(r['summary']['stagger_recovery_seconds'], 10 * expected_ticks / duration)

    def test_loop_counts_only_current_loop_removal(self):
        r = self.simulate(daphne=True, repeat=True, loop=True)
        self.assertTrue(r['dissonance_events'])
        self.assertTrue(all(0 <= e['tick'] <= r['summary']['duration_ticks'] for e in r['dissonance_events']))
        self.assertAlmostEqual(r['summary']['dissonance_stagger'],
                               sum(e['stagger_amount'] for e in r['dissonance_events']))


if __name__ == '__main__':
    unittest.main()
