"""Synthetic no-CUDA tests for GPU rule capacity planning and slot lifetimes."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

from app.modules.card_game.rl.batched_duel.gpu_rule_capacity import (
    DEFAULT_MAX_STATES,
    GpuRuleCapacityError,
    GpuRuleSlotLedger,
    dumps_capacity_plan,
    plan_gpu_rule_capacity,
    resolve_gpu_rule_backend_capacity,
    split_query_batches,
    trace_terminal_probe_liveness,
    validate_capacity_plan,
    DEFAULT_QUERY_BATCH_ROWS,
    DEFAULT_TERMINAL_HORIZON,
    KNOWN_DEVICE_BYTES,
    KNOWN_DEVICE_NAME,
)


REPO = Path(__file__).resolve().parents[1]


class GpuRuleCapacityPlannerTest(unittest.TestCase):
    def test_1600_roots_exceed_historical_2048_and_stay_jsonable(self):
        plan = plan_gpu_rule_capacity(1600, 1600)
        self.assertTrue(plan['valid'])
        self.assertFalse(plan['rejected'])
        self.assertEqual(plan['runtime'], 'recovery')
        self.assertEqual(plan['max_roots'], 1600)
        self.assertEqual(plan['max_states'], 12800)
        self.assertGreater(plan['max_states'], DEFAULT_MAX_STATES)
        self.assertFalse(plan['default_2048_states_sufficient'])
        self.assertFalse(plan['estimated_peak_bytes_are_device_observed'])
        self.assertEqual(plan['lifetime']['real'], 1600)
        self.assertEqual(plan['lifetime']['root_clean'], 1600)
        self.assertEqual(plan['lifetime']['sampled'], 1600)
        self.assertEqual(plan['lifetime']['simulated'], 1600)
        self.assertEqual(plan['lifetime']['temporary_live'], 1600)
        self.assertEqual(plan['lifetime']['probe_parents'], 1600 * DEFAULT_TERMINAL_HORIZON)
        self.assertEqual(plan['query_batch_rows'], DEFAULT_QUERY_BATCH_ROWS)
        self.assertLess(plan['query_batch_rows'], plan['max_roots'])
        self.assertEqual(plan['encoder_dim_source'], 'gpu_schema_contract')
        self.assertFalse(plan['device_coverage_claimed'])
        self.assertEqual(plan['known_device_name'], KNOWN_DEVICE_NAME)
        self.assertEqual(plan['known_device_bytes'], KNOWN_DEVICE_BYTES)
        self.assertIsNone(plan['observed_peak_bytes'])
        payload = dumps_capacity_plan(plan)
        self.assertEqual(json.loads(payload)['max_states'], 12800)

    def test_small_windows_are_not_inflated_to_2048(self):
        plan = plan_gpu_rule_capacity(2, 2)
        self.assertTrue(plan['valid'])
        self.assertEqual(plan['max_states'], 16)
        self.assertLess(plan['max_states'], DEFAULT_MAX_STATES)
        self.assertTrue(plan['default_2048_states_sufficient'])
        self.assertEqual(plan['lifetime']['query_batch_rows'], DEFAULT_QUERY_BATCH_ROWS)

    def test_memory_budget_rejects_instead_of_truncating(self):
        plan = plan_gpu_rule_capacity(1600, 1600, memory_budget_bytes=1024)
        self.assertTrue(plan['rejected'])
        self.assertFalse(plan['valid'])
        self.assertIn('refusing to truncate', plan['rejection_reason'])
        with self.assertRaises(GpuRuleCapacityError):
            validate_capacity_plan(plan)

    def test_invalid_runtime_and_bounds_are_explicit(self):
        bad = plan_gpu_rule_capacity(1600, 1600, runtime='five')
        self.assertTrue(bad['rejected'])
        with self.assertRaises(TypeError):
            plan_gpu_rule_capacity(True, 8)
        over = plan_gpu_rule_capacity(1601, 8)
        self.assertTrue(over['rejected'])
        empty = plan_gpu_rule_capacity(8, 0)
        self.assertTrue(empty['rejected'])

    def test_seven_runtime_does_not_reuse_recovery_widths(self):
        recovery = plan_gpu_rule_capacity(8, 8, runtime='recovery')
        seven = plan_gpu_rule_capacity(8, 8, runtime='seven')
        self.assertTrue(recovery['valid'])
        self.assertTrue(seven['valid'], seven.get('rejection_reason'))
        self.assertEqual(recovery['obs_dim'], 2420)
        self.assertEqual(recovery['cand_dim'], 254)
        self.assertNotEqual((seven['obs_dim'], seven['cand_dim']), (recovery['obs_dim'], recovery['cand_dim']))
        self.assertEqual(seven['obs_dim'], 3479)
        self.assertEqual(seven['cand_dim'], 1576)
        self.assertEqual(seven['encoder_dim_source'], 'gpu_schema_contract')
        self.assertEqual(recovery['encoder_dim_source'], 'gpu_schema_contract')
        self.assertEqual(seven['max_payload_bytes'], recovery['max_payload_bytes'])
        self.assertGreater(seven['vm_max_payload_bytes'], seven['max_payload_bytes'])
        self.assertGreaterEqual(seven['vm_max_payload_bytes'], 2 * 120 * 1576 * 4)
        self.assertGreater(seven['vm_max_nodes'], seven['max_nodes'])
        self.assertEqual(seven['max_states'], recovery['max_states'])

    def test_planner_does_not_import_torch(self):
        code = (
            'import sys\n'
            'from app.modules.card_game.rl.batched_duel.gpu_rule_capacity import plan_gpu_rule_capacity\n'
            'plan = plan_gpu_rule_capacity(4, 4)\n'
            'assert plan["valid"] is True\n'
            'assert plan["estimated_peak_bytes_are_device_observed"] is False\n'
            'assert "torch" not in sys.modules\n'
        )
        subprocess.run([sys.executable, '-c', code], check=True, cwd=REPO, capture_output=True)

    def test_resolve_keeps_legacy_2048_without_claiming_1600_proof(self):
        resolved = resolve_gpu_rule_backend_capacity()
        self.assertFalse(resolved['derived_from_plan'])
        self.assertEqual(resolved['max_states'], 2048)
        self.assertIsNone(resolved['default_2048_states_sufficient'])
        self.assertFalse(resolved['estimated_peak_bytes_are_device_observed'])
        plan = plan_gpu_rule_capacity(1600, 1600)
        adopted = resolve_gpu_rule_backend_capacity(capacity_plan=plan)
        self.assertTrue(adopted['derived_from_plan'])
        self.assertEqual(adopted['max_states'], 12800)
        self.assertEqual(adopted['query_batch_rows'], DEFAULT_QUERY_BATCH_ROWS)
        self.assertEqual(adopted['vm_max_nodes'], plan['vm_max_nodes'])
        self.assertFalse(adopted['default_2048_states_sufficient'])
        with self.assertRaises(GpuRuleCapacityError):
            resolve_gpu_rule_backend_capacity(capacity_plan=plan, max_states=4096)

    def test_query_workspace_is_independent_of_admitted_roots(self):
        small = plan_gpu_rule_capacity(32, 32, runtime='seven')
        large = plan_gpu_rule_capacity(1600, 1600, runtime='seven')
        self.assertEqual(small['query_batch_rows'], large['query_batch_rows'])
        self.assertEqual(small['vm_max_payload_bytes'], large['vm_max_payload_bytes'])
        self.assertEqual(small['components']['batch_workspace_bytes'], large['components']['batch_workspace_bytes'])
        self.assertGreater(large['lifetime']['admitted_roots'], large['query_batch_rows'])
        self.assertEqual(sum(split_query_batches(1600, large['query_batch_rows'])), 1600)
        self.assertTrue(all(size == large['query_batch_rows'] for size in split_query_batches(1600, large['query_batch_rows'])))

    def test_vm_workspace_refuses_silent_shrink(self):
        plan = plan_gpu_rule_capacity(8, 8, runtime='seven', vm_max_payload_bytes=131072)
        self.assertTrue(plan['rejected'])
        self.assertIn('below the conservative encode bound', plan['rejection_reason'])
        with self.assertRaises(GpuRuleCapacityError):
            validate_capacity_plan(plan)

    def test_known_device_overflow_rejects_budget_without_claiming_coverage(self):
        plan = plan_gpu_rule_capacity(1600, 1600, runtime='seven', query_batch_rows=1600, memory_budget_bytes=KNOWN_DEVICE_BYTES)
        self.assertTrue(plan['rejected'])
        self.assertIn('refusing to truncate', plan['rejection_reason'])
        self.assertFalse(plan.get('device_coverage_claimed', False))
        self.assertFalse(plan['fits_known_device'])

    def test_argument_bind_exposes_destination_layout(self):
        import inspect
        from app.modules.card_game.rl.batched_duel.gpu_rule_arguments import GraphArgumentKernel, emit_argument_source
        params = inspect.signature(GraphArgumentKernel.bind).parameters
        self.assertIn('destination_layout', params)
        source = emit_argument_source()
        self.assertIn('src_maxn', source)
        self.assertIn('dst_maxn', source)
        self.assertIn('src_maxp', source)
        self.assertIn('dst_maxp', source)

    def test_argument_kernel_host_stub_compiles_distinct_from_cuda(self):
        from app.modules.card_game.rl.batched_duel.gpu_rule_arguments import emit_argument_source
        stub = (
            '#define __device__\n'
            '#define __global__\n'
            'struct Dim { int x; };\n'
            'static Dim threadIdx{0};\n'
            'static Dim blockIdx{0};\n'
            'static Dim blockDim{1};\n'
        )
        source = stub + emit_argument_source()
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / 'graph_append_arguments_host_stub.cpp'
            path.write_text(source)
            compiled = Path(raw) / 'graph_append_arguments_host_stub.o'
            proc = subprocess.run(
                ['c++', '-std=c++17', '-fsyntax-only', '-x', 'c++', str(path)],
                cwd=REPO, capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse(compiled.exists())
            self.assertNotIn('nvcc', proc.stdout + proc.stderr)


class GpuRuleSlotLedgerTest(unittest.TestCase):
    def test_1600_root_kinds_and_overflow_poison(self):
        ledger = GpuRuleSlotLedger(6400, 1600)
        ledger.admit_real_games(['g-%s' % i for i in range(1600)])
        real = ledger.allocate(1600, 'real')
        for slot in real:
            ledger.register(slot, 'real')
        sampled = ledger.allocate(1600, 'sampled')
        for slot in sampled:
            ledger.register(slot, 'sampled')
        simulated = ledger.allocate(1600, 'simulated')
        for slot in simulated:
            ledger.register(slot, 'simulated')
        temporary = ledger.allocate(1600, 'temporary')
        keys = [ledger.register(slot, 'temporary') for slot in temporary]
        metrics = ledger.metrics()
        self.assertEqual(metrics['live_states'], 6400)
        self.assertEqual(metrics['peak_states'], 6400)
        self.assertEqual(metrics['live_by_kind']['temporary'], 1600)
        self.assertEqual(metrics['free_slots'], 0)
        with self.assertRaises(OverflowError):
            ledger.allocate(1, 'hypothetical')
        self.assertTrue(ledger.poisoned)
        self.assertTrue(ledger.closed)
        with self.assertRaisesRegex(RuntimeError, 'poisoned'):
            ledger.validate_key(keys[0])

    def test_stale_repeated_release_and_closed_backend(self):
        ledger = GpuRuleSlotLedger(8, 2)
        slots = ledger.allocate(2, 'real')
        keys = [ledger.register(slot, 'real') for slot in slots]
        ledger.release_keys(keys)
        with self.assertRaisesRegex(ValueError, 'Stale or released'):
            ledger.release_key(keys[0])
        with self.assertRaisesRegex(ValueError, 'Duplicate state release'):
            ledger.release_keys([keys[0], keys[0]])
        ledger.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            ledger.allocate(1, 'sampled')

    def test_nested_terminal_probe_trace_reaches_horizon(self):
        plan = plan_gpu_rule_capacity(4, 4, terminal_horizon=DEFAULT_TERMINAL_HORIZON)
        ledger = GpuRuleSlotLedger(plan['max_states'], plan['max_games'])
        metrics = trace_terminal_probe_liveness(
            ledger, games=4, search_roots=4, terminal_horizon=DEFAULT_TERMINAL_HORIZON,
        )
        self.assertEqual(metrics['trace_depth'], DEFAULT_TERMINAL_HORIZON)
        self.assertEqual(metrics['configured_horizon'], DEFAULT_TERMINAL_HORIZON)
        self.assertLessEqual(metrics['peak_states'], plan['max_states'])
        self.assertEqual(metrics['formula_max_states'], plan['max_states'])
        self.assertGreater(metrics['peak_states'], 4 + 3 * 4)
        self.assertFalse(plan['device_coverage_claimed'])


if __name__ == '__main__':
    unittest.main()
