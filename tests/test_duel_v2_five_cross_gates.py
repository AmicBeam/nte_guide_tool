import subprocess
import sys
import tempfile
import unittest
from scripts.calibrate_duel_v2_policy_fit import choose_variant
from app.modules.card_game.rl.five_cross_gates import (
    first_gate, late_gate, launch_allowed, mid_gate, schedule_gates)


def _ok(play=0.4, reference=0.4):
    return dict(finite=True, policy_gradient=True, play_legal=30, play_top1=play,
                reference_play_top1=reference)


class FiveCrossGateTest(unittest.TestCase):
    def test_three_gates_sit_inside_a_frozen_batch_count(self):
        gates = schedule_gates(9)
        self.assertEqual(gates, dict(first=1, mid=3, late=6))
        self.assertLess(gates['first'], gates['mid'])
        self.assertLess(gates['mid'], gates['late'])
        with self.assertRaises(ValueError):
            schedule_gates(2)

    def test_short_window_cannot_claim_a_complete_run(self):
        self.assertFalse(launch_allowed(seconds_available=3600, batch_seconds=2000,
            eval_seconds=4000, gate_seconds=600, reserve_seconds=900))
        self.assertTrue(launch_allowed(seconds_available=20000, batch_seconds=2000,
            eval_seconds=4000, gate_seconds=600, reserve_seconds=900))

    def test_first_gate_stops_when_the_network_never_plays(self):
        decision = first_gate(dict(zhenhong=_ok(play=0, reference=0)))
        self.assertEqual(decision['decision'], 'p0')
        self.assertEqual(decision['reasons'][0]['reason'], 'no_play')

    def test_first_gate_stops_on_regression_and_a_dead_policy_gradient(self):
        broken = _ok(play=0.1, reference=0.4)
        broken['policy_gradient'] = False
        decision = first_gate(dict(starter=broken))
        self.assertEqual(decision['decision'], 'p0')
        self.assertIn('no_policy_gradient', {row['reason'] for row in decision['reasons']})
        self.assertIn('regression', {row['reason'] for row in decision['reasons']})

    def test_mid_gate_promotes_only_after_a_second_search_seed(self):
        screen = dict(complete=True, candidate=6, best=3)
        confirm = dict(complete=True, candidate=5, best=4)
        self.assertEqual(mid_gate('pass', screen, confirm), 'promote')
        self.assertEqual(mid_gate('pass', screen, dict(complete=True, candidate=4, best=4)), 'keep')
        self.assertEqual(mid_gate('p0', screen, confirm), 'p0')
        self.assertEqual(mid_gate('pass', screen, None), 'keep')

    def test_formal_run_is_refused_without_a_scheduled_window(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = subprocess.run([sys.executable, 'scripts/train_duel_v2_current_round.py',
                                   '--output', folder, '--run', '--device', 'cpu'],
                                  capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('refused', (proc.stderr + proc.stdout).lower())

    def test_calibration_rejects_a_play_rate_above_the_target_band(self):
        target = 0.373
        rows = [dict(name='low', model_play_top1=0.11, target_play_top1=target,
                     cross_entropy_drop=0.09, model_end_top1=0.2),
                dict(name='over', model_play_top1=0.62, target_play_top1=target,
                     cross_entropy_drop=0.12, model_end_top1=0.1)]
        self.assertIsNone(choose_variant(rows, target))
        rows.append(dict(name='inside', model_play_top1=0.40, target_play_top1=target,
                         cross_entropy_drop=0.11, model_end_top1=0.1))
        self.assertEqual(choose_variant(rows, target)['name'], 'inside')

    def test_late_gate_rolls_back_or_stops_after_two_flat_checks(self):
        self.assertEqual(late_gate('pass', False, True, False), 'plateau')
        self.assertEqual(late_gate('improved', False, False, True), 'rollback')
        self.assertEqual(late_gate('improved', True, False, False), 'promote')
        self.assertEqual(late_gate('p0', True, False, False), 'p0')



    def test_missing_policy_gradient_or_nonfinite_is_still_p0(self):
        dead = _ok()
        dead['finite'] = False
        dead['policy_gradient'] = False
        decision = first_gate(dict(zhenhong=dead))
        self.assertEqual(decision['decision'], 'p0')
        self.assertEqual({row['reason'] for row in decision['reasons']}, {'nonfinite', 'no_policy_gradient'})


if __name__ == '__main__':
    unittest.main()
