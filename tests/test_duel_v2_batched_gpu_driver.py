"""Config, constructor, matrix, and synthetic inference tests for the GPU rule driver.

These tests never claim real CUDA, never load trained weights, and never mark
device_verified or concurrency_1600_ready true from source alone. Fake GPU
constructors are explicitly labeled.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from app.modules.card_game.rl.batched_search.evaluator import (
    COMPLETED_DENOMINATOR,
    MATRIX_ORDER,
    RECOVERY_MATRIX_KEYS,
    RULE_BACKEND_CUDA_VM,
    RULE_BACKEND_RESIDENT_CPU,
    RUNTIME_RECOVERY,
    RUNTIME_SEVEN,
    compute_matrix_summary,
    generate_matrix_jobs,
    generate_report_markdown,
    make_source_snapshot,
    runtime_matrix_keys,
    select_rule_backend,
    write_result_json,
)
from app.modules.card_game.rl.batched_search.inference import (
    CANDIDATE_DIM,
    FEATURE_DIM,
    BatchedInference,
    InferenceRequest,
    named_observation_contract,
)
from scripts.evaluate_duel_v2_batched import main, parse_args


REPO = Path(__file__).resolve().parents[1]


class FakeProgram:
    """FAKE frozen program object. Not a CUDA image and not device-verified."""

    fake_label = 'FAKE_GPU_RULE_PROGRAM'
    identity = 'fake-program-identity'
    device_verified = False


class FakeProgramApi:
    """FAKE A-side capture/build/load surface with explicit runtime=."""

    fake_label = 'FAKE_GPU_PROGRAM_API'

    def load_program(self, path, runtime=None):
        return FakeProgram()

    def build_gpu_rule_program(self, runtime=None):
        return FakeProgram(), {'device_verified': False, 'fake_label': self.fake_label}

    def capture_gpu_rule_program(self, runtime=None):
        return FakeProgram()

    def source_capture_available(self):
        return True


def fake_plan_gpu_rule_capacity(
    games,
    search_roots,
    memory_budget_bytes=None,
    max_nodes=None,
    max_edges=None,
    max_payload_bytes=None,
    max_candidates=None,
    runtime=None,
):
    """FAKE B-side capacity planner. Pure dict, not a CUDA reservation."""
    return {
        'fake_label': 'FAKE_CAPACITY_PLAN',
        'games': games,
        'search_roots': search_roots,
        'memory_budget_bytes': memory_budget_bytes,
        'max_nodes': max_nodes,
        'max_edges': max_edges,
        'max_payload_bytes': max_payload_bytes,
        'max_candidates': max_candidates,
        'runtime': getattr(runtime, '__name__', runtime),
        'device_verified': False,
    }


class FakeLabeledGpuRuleBackend:
    """FAKE GPU rule backend constructor spy. Not CUDA. device_verified stays false."""

    backend_kind = 'cuda_vm'
    supports_gpu = True
    supports_full_duel_rules = False
    device_verified = False
    fake_label = 'FAKE_LABELED_GPU_RULE_BACKEND'

    def __init__(
        self,
        program=None,
        *,
        runtime=None,
        capacity_plan=None,
        guard_standalone_entities=False,
        record_output=None,
        max_games=None,
        **kwargs,
    ):
        self.program = program
        self.runtime = runtime
        self.runtime_name = runtime if runtime in ('recovery', 'seven') else getattr(runtime, '__name__', runtime)
        self.capacity_plan = capacity_plan
        self.guard_standalone_entities = guard_standalone_entities
        self.record_output = record_output
        self.max_games = max_games
        self.workers = 0
        self.init_kwargs = dict(kwargs)


class SyntheticSevenPolicy:
    """SYNTHETIC in-memory seven-team policy. Not a trained checkpoint."""

    def __init__(self, key: str, hidden: int = 16):
        import torch
        from app.modules.card_game.rl import seven_lineup, seven_policy
        net = seven_policy.network(hidden=hidden, device='cpu')
        expected = seven_policy.tensor_shapes(hidden)
        state = net.state_dict()
        self.weights = {name: state[name].detach().cpu().numpy().copy() for name in expected}
        for weight in self.weights.values():
            weight.flags.writeable = False
        self.hidden = hidden
        self.schema = seven_policy.SCHEMA
        self.version = f'synthetic_seven_{key}'
        self.teacher = None
        self.manifest = {
            'schema': seven_policy.SCHEMA,
            'deck': key,
            'hidden': hidden,
            'sha256': self.version,
            'features': list(seven_lineup.all_features()),
            'candidates': list(seven_lineup.candidate_names()),
            'wdl_order': ['loss', 'draw', 'win'],
            'value_context': ['viewer_is_actor'],
            'reward_mode': 'terminal_wdl_only',
            'automatic_serving_approval': False,
        }

    def scores(self, x, candidates):
        from app.modules.card_game.rl import seven_policy
        return seven_policy.scores_numpy(x, candidates, self.weights)

    def wdl(self, x, is_actor):
        from app.modules.card_game.rl import seven_policy
        return seven_policy.wdl_numpy(x, is_actor, self.weights)


def make_seven_observation(seed: int = 0):
    from app.modules.card_game.rl import seven_lineup
    rng = np.random.RandomState(seed)
    names = seven_lineup.all_features()
    x = rng.randn(len(names)).astype(np.float32) * 0.05
    x[names.index('front:1')] = 0.0
    return x


def make_seven_candidates(count: int, seed: int = 0):
    from app.modules.card_game.rl import seven_lineup
    names = seven_lineup.candidate_names()
    seats = len(seven_lineup.SEATS)
    candidates = np.zeros((count, len(names)), dtype=np.float32)
    for index in range(count):
        candidates[index, 0] = float((seed + index) % 9)
        candidates[index, 1] = float((seed + index) % seats)
        candidates[index, 2] = -1.0
        candidates[index, 3] = 1.0
        candidates[index, 4] = float((seed + index) % seats)
        candidates[index, 5:] = float(index + 1) * 0.01
    return candidates


class TestDuelV2BatchedGpuDriver(unittest.TestCase):
    def test_default_cli_uses_unified_seven_and_does_not_import_torch(self):
        # Check a fresh process: other device tests may already have loaded Torch
        # into the unittest process before this test runs.
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root / 'scripts/evaluate_duel_v2_batched.py')],
                                cwd=root, text=True, capture_output=True, check=True)
        payload = json.loads(result.stdout)
        self.assertEqual(payload['action'], 'config_check_only')
        self.assertEqual(payload['rule_backend'], RULE_BACKEND_RESIDENT_CPU)
        self.assertEqual(payload['runtime'], RUNTIME_SEVEN)
        self.assertEqual(payload['matrix_keys'], 7)
        self.assertEqual(payload['matrix_order'], MATRIX_ORDER)
        self.assertEqual(payload['denominator'], COMPLETED_DENOMINATOR)
        self.assertFalse(payload['device_verified'])
        self.assertFalse(payload['concurrency_1600_ready'])
        self.assertFalse(payload['gpu_rule_execution'])
        self.assertFalse(payload['torch_imported'])
        self.assertFalse(payload['processes_started'])
        self.assertFalse(payload['weights_loaded'])
        self.assertIsNone(payload['seed_base'])

    def test_cli_rejects_gpu_flags_on_resident_cpu_and_invalid_cuda_vm_combinations(self):
        with self.assertRaises(SystemExit):
            parse_args(['--gpu-rule-program', 'frozen.json'])
        with self.assertRaises(SystemExit):
            parse_args(['--guard-standalone-entities'])
        with self.assertRaises(SystemExit):
            parse_args(['--gpu-memory-budget-mib', '1024'])
        with self.assertRaises(SystemExit):
            parse_args(['--gpu-max-nodes', '8'])
        with self.assertRaises(SystemExit):
            parse_args(['--rule-backend', 'cuda_vm', '--gpu-memory-budget-mib', '0'])
        if sys.version_info[:2] != (3, 10):
            with self.assertRaises(SystemExit):
                parse_args(['--rule-backend', 'cuda_vm'])
        args = parse_args([
            '--rule-backend', 'cuda_vm',
            '--gpu-rule-program', 'frozen.json',
            '--device', 'cpu',
            '--runtime', 'seven',
            '--guard-standalone-entities',
            '--gpu-memory-budget-mib', '8192',
            '--gpu-max-nodes', '4096',
            '--gpu-max-edges', '8192',
            '--gpu-max-payload-bytes', '65536',
            '--gpu-max-candidates', '256',
            '--search-roots', '1600',
            '--games', '1600',
        ])
        self.assertEqual(args.rule_backend, RULE_BACKEND_CUDA_VM)
        self.assertEqual(args.device, 'cpu')
        self.assertEqual(args.runtime, RUNTIME_SEVEN)
        self.assertTrue(args.guard_standalone_entities)
        self.assertEqual(args.search_roots, 1600)

    def test_config_check_subprocess_does_not_import_torch_or_write_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'would_be_output'
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO / 'scripts' / 'evaluate_duel_v2_batched.py'),
                    '--rule-backend', 'resident_cpu',
                    '--runtime', 'seven',
                    '--output', str(output),
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(output.exists())
            payload = json.loads(completed.stdout)
            self.assertEqual(payload['runtime'], RUNTIME_SEVEN)
            self.assertEqual(payload['matrix_keys'], 7)
            self.assertFalse(payload['torch_imported'])
            self.assertFalse(payload['device_verified'])

    def test_legacy_five_team_matrix_order_is_unchanged(self):
        keys = RECOVERY_MATRIX_KEYS
        decks = {key: {'id': key} for key in keys}
        jobs = generate_matrix_jobs(keys, decks, seed_base=10000, total_games=1600)
        self.assertEqual(len(jobs), 1600)
        self.assertEqual(jobs[0]['matrix_row'], 'starter')
        self.assertEqual(jobs[0]['column'], 'starter')
        self.assertEqual(jobs[0]['first'], 'a')
        self.assertEqual(jobs[1]['cell'], 1)
        self.assertEqual(jobs[1]['matrix_row'], 'starter')
        self.assertEqual(jobs[1]['column'], 'weave-rush')
        self.assertEqual(jobs[25]['cell'], 0)
        self.assertEqual(jobs[25]['first'], 'b')
        self.assertEqual(jobs[25]['seed'], jobs[0]['seed'])

    def test_seven_team_matrix_is_7x7_with_same_first_row_order(self):
        keys = runtime_matrix_keys(RUNTIME_SEVEN)
        self.assertEqual(len(keys), 7)
        self.assertEqual(keys[:5], RECOVERY_MATRIX_KEYS)
        decks = {key: {'id': key} for key in keys}
        jobs = generate_matrix_jobs(keys, decks, seed_base=7, total_games=98)
        self.assertEqual(len(jobs), 98)
        self.assertEqual(len({job['cell'] for job in jobs}), 49)
        self.assertEqual(jobs[0]['matrix_row'], keys[0])
        self.assertEqual(jobs[0]['column'], keys[0])
        self.assertEqual(jobs[1]['matrix_row'], keys[0])
        self.assertEqual(jobs[1]['column'], keys[1])
        self.assertEqual(jobs[7]['matrix_row'], keys[1])
        self.assertEqual(jobs[7]['column'], keys[0])
        self.assertEqual(jobs[49]['cell'], 0)
        self.assertEqual(jobs[49]['first'], 'b')
        self.assertEqual(jobs[49]['seed'], jobs[0]['seed'])
        summary = compute_matrix_summary(keys, [
            {'matrix_row': keys[0], 'column': keys[6], 'winner': 'a', 'first': 'a', 'complete': True},
            {'matrix_row': keys[0], 'column': keys[6], 'complete': False},
        ])
        self.assertEqual(summary['team_count'], 7)
        self.assertEqual(summary['matrix_order'], MATRIX_ORDER)
        self.assertEqual(summary['denominator'], COMPLETED_DENOMINATOR)
        cell = next(item for item in summary['cells'] if item['first_model'] == keys[0] and item['second_model'] == keys[6])
        self.assertEqual(cell['wins'], 1)
        self.assertEqual(cell['uncompleted'], 1)
        self.assertEqual(cell['completed'], 1)

    def test_constructor_selection_with_labeled_fake_gpu_backend(self):
        backend = select_rule_backend(
            rule_backend=RULE_BACKEND_CUDA_VM,
            runtime=RUNTIME_SEVEN,
            workers=2,
            max_games=49,
            search_roots=16,
            gpu_rule_program='FAKE_FROZEN_PROGRAM.json',
            guard_standalone_entities=True,
            gpu_memory_budget_mib=1024,
            max_nodes=128,
            max_edges=256,
            max_payload_bytes=4096,
            max_candidates=32,
            gpu_backend_cls=FakeLabeledGpuRuleBackend,
            capacity_planner=fake_plan_gpu_rule_capacity,
            program_api=FakeProgramApi(),
            fake_label='FAKE_LABELED_GPU_RULE_BACKEND',
        )
        self.assertIsInstance(backend, FakeLabeledGpuRuleBackend)
        self.assertEqual(backend.fake_label, 'FAKE_LABELED_GPU_RULE_BACKEND')
        self.assertFalse(backend.device_verified)
        self.assertTrue(backend.guard_standalone_entities)
        self.assertEqual(backend.capacity_plan['fake_label'], 'FAKE_CAPACITY_PLAN')
        self.assertEqual(backend.capacity_plan['games'], 49)
        self.assertEqual(backend.capacity_plan['search_roots'], 16)
        self.assertEqual(backend.capacity_plan['memory_budget_bytes'], 1024 * 1024 * 1024)
        self.assertEqual(backend.capacity_plan['runtime'], RUNTIME_SEVEN)
        self.assertEqual(backend.runtime_name, RUNTIME_SEVEN)
        self.assertTrue(str(backend.fake_label).startswith('FAKE'))

    def test_unlabeled_fake_gpu_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'labeled fake or synthetic'):
            select_rule_backend(
                rule_backend=RULE_BACKEND_CUDA_VM,
                runtime=RUNTIME_RECOVERY,
                workers=1,
                max_games=1,
                search_roots=1,
                gpu_rule_program='unused.json',
                gpu_backend_cls=FakeLabeledGpuRuleBackend,
                capacity_planner=fake_plan_gpu_rule_capacity,
                program_api=FakeProgramApi(),
                fake_label='real_cuda_backend',
            )

    def test_real_cuda_vm_constructor_is_pending_without_cpu_fallback(self):

        with self.assertRaises((FileNotFoundError, RuntimeError, TypeError, ValueError)) as ctx:
            select_rule_backend(
                rule_backend=RULE_BACKEND_CUDA_VM,
                runtime=RUNTIME_RECOVERY,
                workers=1,
                max_games=1,
                search_roots=1,
                gpu_rule_program='unused.json',
            )
        self.assertNotIn('ResidentRuleBackend', type(ctx.exception).__name__)
        message = str(ctx.exception)
        self.assertFalse('fell back to CPU' in message.lower())

    def test_resident_cpu_constructor_still_binds_explicit_runtime(self):
        class FakeResident:
            backend_kind = 'python_resident_process'
            supports_gpu = False
            device_verified = False

            def __init__(self, *, workers, max_games, runtime, record_output=None):
                self.workers = workers
                self.max_games = max_games
                self.runtime = runtime
                name = getattr(runtime, '__name__', '')
                self.runtime_name = 'seven' if name.endswith('seven_runtime') else 'recovery'
                self.record_output = record_output

        backend = select_rule_backend(
            rule_backend=RULE_BACKEND_RESIDENT_CPU,
            runtime=RUNTIME_SEVEN,
            workers=3,
            max_games=8,
            search_roots=4,
            resident_backend_cls=FakeResident,
        )
        self.assertEqual(backend.runtime_name, RUNTIME_SEVEN)
        self.assertEqual(backend.workers, 3)
        self.assertFalse(backend.supports_gpu)


    def test_resident_runtime_bind_accepts_recovery_and_seven_only(self):
        from app.modules.card_game.rl import recovery_runtime, seven_runtime
        from app.modules.card_game.rl.batched_duel.process_backend import bind_resident_runtime
        _, name = bind_resident_runtime(None)
        self.assertEqual(name, RUNTIME_RECOVERY)
        _, name = bind_resident_runtime(recovery_runtime)
        self.assertEqual(name, RUNTIME_RECOVERY)
        _, name = bind_resident_runtime(seven_runtime)
        self.assertEqual(name, RUNTIME_SEVEN)
        with self.assertRaisesRegex(ValueError, 'No implicit fallback'):
            bind_resident_runtime(sys)

    def test_snapshot_and_report_identify_backend_runtime_and_keep_device_unverified(self):
        class TinyModel:
            version = 'synthetic'
            manifest = {'sha256': 'synthetic'}
            serving_build_sha256 = 'build'

        snapshot = make_source_snapshot(
            {'starter': TinyModel()},
            'simulations',
            None,
            rule_backend=RULE_BACKEND_CUDA_VM,
            runtime=RUNTIME_SEVEN,
            search_roots=16,
            games=49,
            capacity_plan={'fake_label': 'FAKE_CAPACITY_PLAN', 'max_nodes': 8},
            gpu_program_identity='fake-program-identity',
            gpu_memory_budget_mib=512,
            guard_standalone_entities=True,
            device_verified=False,
        )
        self.assertEqual(snapshot['rule_backend'], RULE_BACKEND_CUDA_VM)
        self.assertEqual(snapshot['runtime'], RUNTIME_SEVEN)
        self.assertEqual(snapshot['requested_search_roots'], 16)
        self.assertFalse(snapshot['device_verified'])
        self.assertFalse(snapshot['concurrency_1600_ready'])
        eval_result = {
            'game_execution_complete': False,
            'concurrency_1600_full_games_complete': False,
            'evaluation_accepted': False,
            'gpu_rule_execution': True,
            'rule_backend': RULE_BACKEND_CUDA_VM,
            'runtime': RUNTIME_SEVEN,
            'device_verified': False,
            'requested_search_roots': 16,
            'observed_search_root_peak': 4,
            'capacity_plan': snapshot['capacity_plan'],
            'gpu_memory_budget_mib': 512,
            'denominator': COMPLETED_DENOMINATOR,
            'total_games': 4,
            'completed_games': 1,
            'uncompleted_games': 3,
            'peak_outstanding_roots': 4,
            'search_root_limit': 16,
            'stats': {},
            'behavior': {'behavior_approved': False},
        }
        keys = runtime_matrix_keys(RUNTIME_SEVEN)
        summary = compute_matrix_summary(keys, [
            {'matrix_row': keys[0], 'column': keys[1], 'winner': 'a', 'first': 'a', 'complete': True},
            {'matrix_row': keys[0], 'column': keys[1], 'complete': False},
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            write_result_json(path, eval_result, summary, {'rule_backend': RULE_BACKEND_CUDA_VM, 'runtime': RUNTIME_SEVEN}, snapshot)
            payload = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(payload['rule_backend'], RULE_BACKEND_CUDA_VM)
        self.assertEqual(payload['runtime'], RUNTIME_SEVEN)
        self.assertTrue(payload['gpu_rule_execution'])
        self.assertFalse(payload['device_verified'])
        self.assertEqual(payload['requested_search_roots'], 16)
        self.assertEqual(payload['observed_search_root_peak'], 4)
        self.assertEqual(payload['denominator'], COMPLETED_DENOMINATOR)
        report = generate_report_markdown(summary, eval_result, {'rule_backend': RULE_BACKEND_CUDA_VM})
        self.assertIn('device_verified', report)
        self.assertIn('cuda_vm', report)
        self.assertIn('7×7', report)

    def test_seven_inference_binds_named_widths_and_matches_synthetic_fp32(self):
        from app.modules.card_game.rl import seven_lineup, seven_policy
        model = SyntheticSevenPolicy('starter', hidden=16)
        contract = named_observation_contract(model.schema, model.manifest)
        self.assertEqual(contract['family'], 'seven')
        self.assertEqual(contract['feature_dim'], len(seven_lineup.all_features()))
        self.assertEqual(contract['candidate_dim'], len(seven_lineup.candidate_names()))
        self.assertNotEqual(contract['feature_dim'], FEATURE_DIM)
        self.assertNotEqual(contract['candidate_dim'], CANDIDATE_DIM)
        runner = BatchedInference({'starter': model}, device='cpu')
        x = make_seven_observation(11)
        candidates = make_seven_candidates(3, 11)
        result = runner.predict([
            InferenceRequest('s1', 'starter', model.version, x, candidates, True),
        ])[0]
        np.testing.assert_allclose(result.logits, model.scores(x, candidates), atol=1e-5)
        np.testing.assert_allclose(result.wdl, model.wdl(x, True), atol=1e-5)
        self.assertEqual(result.logits.shape, (3,))
        stats = runner.stats()
        self.assertEqual(stats['feature_dims']['starter'], contract['feature_dim'])
        self.assertEqual(stats['candidate_dims']['starter'], contract['candidate_dim'])
        self.assertEqual(stats['schemas']['starter'], seven_policy.SCHEMA)
        with self.assertRaises(ValueError):
            runner.predict([
                InferenceRequest(
                    'old-width',
                    'starter',
                    model.version,
                    np.zeros(FEATURE_DIM, dtype=np.float32),
                    np.zeros((1, CANDIDATE_DIM), dtype=np.float32),
                    True,
                ),
            ])

    def test_mixed_recovery_and_seven_policies_are_rejected(self):
        from tests.test_duel_v2_batch_inference import SyntheticNumericModel
        five = SyntheticNumericModel('starter', hidden=16)
        seven = SyntheticSevenPolicy('collapse', hidden=16)
        with self.assertRaisesRegex(ValueError, 'cannot be mixed'):
            BatchedInference({'starter': five, 'collapse': seven}, device='cpu')


class StatisticsCapacityBindingTest(unittest.TestCase):
    def test_actual_gpu_statistics_sizes_match_capacity_plan(self):
        from app.modules.card_game.rl.batched_search.evaluator import statistics_capacity
        from app.modules.card_game.rl.batched_duel.gpu_rule_capacity import plan_gpu_rule_capacity
        plan = plan_gpu_rule_capacity(1600, 1600, runtime='seven', max_candidates=128)
        self.assertTrue(plan['valid'])
        self.assertEqual(statistics_capacity('cuda_vm', 1600, plan),
                         (plan['node_capacity'], plan['candidate_capacity']))
        self.assertEqual(statistics_capacity('resident_cpu', 1600), (52800, 2048))


if __name__ == '__main__':
    unittest.main()
