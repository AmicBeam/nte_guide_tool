import unittest

from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import simulate_shaft_axis


class HarmonyTimingTest(unittest.TestCase):
    # 2026-09-19用户口径；本体逐段值与释放时100分开，非Nanoka数值。
    CASES = [('char_heiyu', 'action_heiyu_e', 'char_c78f7a08d5', [2.6] * 4, 12),
             ('char_dd034941ef', 'action_982c67944f', 'char_bdc43f82c6', [.973, .973, 6.11, 6.543], 15)]

    def simulate(self, cid, steps, other=None, options=None):
        chars = [cid] + ([other] if other else [])
        return simulate_shaft_axis({
            'team': [dict(slot=i, character_id=c, arc_id='', cartridge_id='') for i, c in enumerate(chars)],
            'steps': steps, 'initial_energy': 0, 'options': options or {},
        })['result']

    def support(self, cid, tick):
        action = next(a for a in load_shaft_catalog()['actions']
                      if a['character_id'] == cid and a['action_type'] == '援护')
        return dict(id='support', slot=1, action_id=action['id'], start_tick=tick)

    def test_normal_action_accumulates_uniformly_and_only_for_owner(self):
        r = self.simulate('char_heiyu', [dict(id='a', slot=0, action_id='action_heiyu_a2', start_tick=0)], 'char_c78f7a08d5')
        self.assertEqual(r['details'][0]['harmony_after'], 0)
        events = r['harmony_events']
        self.assertEqual([e['tick'] for e in events], list(range(1, 7)))
        self.assertTrue(all(e['slot'] == 0 for e in events))
        for e in events:
            self.assertAlmostEqual(e['amount'], 4.782 / 6)
        self.assertAlmostEqual(events[2]['harmony_after'], 4.782 / 2)
        self.assertAlmostEqual(r['resources_by_slot'][0]['harmony'], 4.782)
        self.assertEqual(r['resources_by_slot'][1]['harmony'], 0)

    def test_cast_bonus_and_capped_gain_resume_after_mid_action_consumption(self):
        for cid, aid, other, hits, duration in self.CASES:
            for loop in (False, True):
                with self.subTest(cid=cid, loop=loop):
                    r = self.simulate(cid, [dict(id='e', slot=0, action_id=aid, start_tick=0), self.support(other, 2)],
                                      other, {'loop_enabled': loop})
                    detail = r['details'][0]
                    self.assertEqual(detail['harmony_on_start'], 100)
                    self.assertAlmostEqual(detail['harmony'], sum(hits))
                    self.assertEqual(detail['harmony_after'], 100)
                    events = [e for e in r['harmony_events'] if e['slot'] == 0]
                    self.assertEqual((events[0]['tick'], events[0]['amount'], events[0]['kind']), (0, 100, 'action_start_gain'))
                    self.assertEqual((events[1]['tick'], events[1]['amount'], events[1]['kind']), (2, -100, 'reaction_cost'))
                    self.assertEqual([e['tick'] for e in events[2:]], list(range(3, duration + 1)))
                    self.assertAlmostEqual(events[-1]['harmony_after'], sum(hits) * (duration - 2) / duration)
                    self.assertFalse(any('环合值不足' in w for d in r['details'] for w in d['warnings']))

    def test_interrupt_scales_only_base_harmony_by_retained_hits(self):
        for cid, aid, other, hits, _ in self.CASES:
            for count in (0, 1, 2, 3, 4):
                with self.subTest(cid=cid, count=count):
                    r = self.simulate(cid, [dict(id='e', slot=0, action_id=aid, start_tick=0,
                                                interrupted=True, interrupt_duration_ticks=4, interrupt_hit_count=count),
                                            self.support(other, 2)], other)
                    d = r['details'][0]
                    self.assertEqual(d['harmony_on_start'], 100)
                    self.assertAlmostEqual(d['harmony'], sum(hits[:count]))
                    self.assertAlmostEqual(r['resources_by_slot'][0]['harmony'], sum(hits[:count]) / 2)
                    self.assertEqual(r['harmony_events'][0]['amount'], 100)

    def test_underfunded_support_uses_elapsed_fraction_then_remaining_gain_continues(self):
        r = self.simulate('char_heiyu', [dict(id='a', slot=0, action_id='action_heiyu_a2', start_tick=0),
                                       self.support('char_c78f7a08d5', 2)], 'char_c78f7a08d5')
        cost = next(e for e in r['harmony_events'] if e['kind'] == 'reaction_cost')
        self.assertAlmostEqual(cost['amount'], -4.782 * 2 / 6)
        self.assertAlmostEqual(r['resources_by_slot'][0]['harmony'], 4.782 * 4 / 6)
        self.assertTrue(any('当前 1.6' in w for d in r['details'] for w in d['warnings']))

    def test_zero_duration_background_gain_is_instant_and_multiplied(self):
        r = self.simulate('char_heiyu', [dict(id='assist', slot=0, action_id='action_heiyu_assist', start_tick=0,
                                            repeat=2)])
        self.assertEqual(len(r['harmony_events']), 1)
        self.assertEqual(r['harmony_events'][0]['tick'], 0)
        self.assertAlmostEqual(r['harmony_events'][0]['amount'], r['details'][0]['harmony'])
        self.assertEqual(r['details'][0]['harmony_gain_timing'], 'instant')
        self.assertAlmostEqual(r['details'][0]['harmony'], 6.68)

    def test_overlapping_background_streams_finish_at_axis_tail(self):
        r = self.simulate('char_heiyu', [dict(id='a', slot=0, action_id='action_heiyu_a2', start_tick=0, placement='background'),
                                       dict(id='b', slot=0, action_id='action_heiyu_a2', start_tick=2, placement='background'),
                                       dict(id='tail', slot=0, action_id='action_none_heiyu', start_tick=8)])
        self.assertAlmostEqual(r['resources_by_slot'][0]['harmony'], 4.782 * 2)
        self.assertEqual(max(e['tick'] for e in r['harmony_events']), 8)

    def test_full_cap_drops_overflow(self):
        for cid, aid, _, _, _ in self.CASES:
            r = self.simulate(cid, [dict(id='e', slot=0, action_id=aid, start_tick=0)])
            self.assertEqual(len(r['harmony_events']), 1)
            self.assertEqual(r['resources_by_slot'][0]['harmony'], 100)

    def test_q_mask_distributes_full_harmony_over_compressed_real_time(self):
        r = self.simulate('char_bdc43f82c6', [
            dict(id='e', slot=0, action_id='action_5870d8ba67', start_tick=0),
            dict(id='q', slot=1, action_id='action_b356b46da2', start_tick=5),
        ], 'char_dd034941ef')
        detail = r['details'][0]
        self.assertTrue(detail['q_instant_release'])
        self.assertEqual(detail['duration_ticks'], 5)
        events = [e for e in r['harmony_events'] if e['source_step_id'] == 'e']
        self.assertEqual([e['tick'] for e in events], list(range(1, 6)))
        for event in events:
            self.assertAlmostEqual(event['amount'], detail['harmony'] / 5)
        self.assertAlmostEqual(r['resources_by_slot'][0]['harmony'], detail['harmony'])
