import json
from pathlib import Path
import subprocess
import sys
import unittest

from app.modules.card_game.rl.batched_search.soa_capacity import SoACapacityConfig, legacy_dense_state_bytes, archive_capacities


class SoACapacityTests(unittest.TestCase):
    def test_plan_import_is_cuda_and_file_free(self):
        command = "from app.modules.card_game.rl.batched_search.soa_capacity import SoACapacityConfig; import sys,json; assert 'torch' not in sys.modules; print(json.dumps(SoACapacityConfig(roots=1600).plan()))"
        result = subprocess.run([sys.executable, '-c', command], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1], timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual((plan['roots'], plan['effective_rule_lanes'], plan['effective_inference_rows']), (1600, 8, 2))
        self.assertFalse(plan['gpu_qualified']); self.assertFalse(plan['capacity_qualified'])
        self.assertTrue(plan['frozen_weights']); self.assertEqual(plan['optimizer_steps'], 0)

    def test_ladder_keeps_roots_separate_from_work_rows(self):
        for roots in (8, 32, 128, 400, 1600):
            plan = SoACapacityConfig(roots=roots, rule_lanes=32, inference_rows=8).plan()
            self.assertEqual(plan['roots'], roots)
            self.assertEqual(plan['effective_rule_lanes'], min(roots, 32))
            self.assertEqual(plan['effective_inference_rows'], min(roots, 8))
            self.assertEqual((plan['simulations'], plan['root_cap'], plan['terminal_horizon']), (32, 16, 0))
            self.assertEqual(plan['world_sampling'], 'per_decision')

    def test_invalid_or_nonfinite_resource_contract_rejected(self):
        for options in ({'roots':True}, {'roots':0}, {'roots':1601}, {'rule_lanes':33},
                        {'inference_rows':False}, {'max_actions':0}, {'max_decisions':0},
                        {'execution_seconds':float('nan')}, {'proof_seconds':float('inf')},
                        {'gpu_budget_mib':False}, {'state_budget_mib':20000}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SoACapacityConfig(**options)

    def test_dense_formula_includes_slot_metadata_and_types(self):
        lengths = {'i64':10, 'f64':3, 'i32':7, 'u8':11}
        self.assertEqual(legacy_dense_state_bytes(lengths, 8), 56*(80+24+28+11+29))
        with self.assertRaises(ValueError):
            legacy_dense_state_bytes(dict(lengths, extra=4), 8)
        with self.assertRaises(ValueError):
            legacy_dense_state_bytes(lengths, True)

    def test_ragged_archive_plan_matches_actual_byte_inventory(self):
        from app.modules.card_game.rl.batched_search.device_raw_action_tape import DeviceRawActionTape
        from app.modules.card_game.rl.batched_search.device_replay_policy_tape import DeviceReplayPolicyTape
        cfg=SoACapacityConfig(roots=8,max_decisions=100,raw_tape_budget_mib=1,policy_tape_budget_mib=1)
        layout=archive_capacities(cfg,100)
        self.assertLess(layout['max_legal_entries'],8*100*800)
        raw=DeviceRawActionTape(8,layout['max_records'],layout['node_capacity'],layout['edge_capacity'],
            layout['payload_capacity'],device='cpu',allow_cpu_reference=True)
        self.addCleanup(raw.close)
        policy=DeviceReplayPolicyTape(8,layout['max_records'],layout['max_legal_entries'],800,1<<20,
            device='cpu',allow_cpu_reference=True,allow_synthetic=True)
        self.assertEqual(raw.allocated_bytes,layout['raw_bytes'])
        self.assertEqual(policy.allocated_bytes,layout['policy_bytes'])
        self.assertLessEqual(raw.allocated_bytes,1<<20)
        self.assertLessEqual(policy.allocated_bytes,1<<20)


if __name__ == '__main__':
    unittest.main()
