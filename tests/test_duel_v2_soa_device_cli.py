"""CLI guards for the bounded SoA CUDA acceptance script. No real device run."""
import ast
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CLI = REPO / 'scripts' / 'benchmark_duel_v2_soa_device.py'


def _main(argv):
    from scripts.benchmark_duel_v2_soa_device import main
    return main(argv)


def _stderr(argv):
    buf = io.StringIO()
    try:
        with redirect_stderr(buf):
            _main(argv)
    except SystemExit as exc:
        if exc.code in (0, None):
            raise AssertionError(argv)
        return buf.getvalue()
    raise AssertionError('expected SystemExit for %r' % (argv,))


class SoADeviceCliTests(unittest.TestCase):
    def test_freeze_check_detects_resident_tensor_changes_and_gradient_inputs(self):
        import torch
        from types import SimpleNamespace
        from scripts.benchmark_duel_v2_soa_device import inference_weight_digest
        bank=SimpleNamespace(weights={'a':torch.tensor([1.,2.])},legacy_weights={'b':torch.tensor([3.])})
        before=inference_weight_digest(bank)
        self.assertEqual(before,inference_weight_digest(bank))
        bank.legacy_weights['b'][0]+=1
        self.assertNotEqual(before,inference_weight_digest(bank))
        bank.weights['a'].requires_grad_(True)
        with self.assertRaises(ValueError):inference_weight_digest(bank)

    def test_memory_budget_rejects_driver_use_and_allocator_cache_separately(self):
        from scripts.benchmark_duel_v2_soa_device import check_memory_budget
        from app.modules.card_game.rl.batched_search.soa_capacity import SoACapacityConfig
        cfg=SoACapacityConfig(gpu_budget_mib=4096,state_budget_mib=2048)
        receipt=dict(device_used=3<<30,reserved=2<<30,process_peak_reserved=2<<30,
                     action_store=100<<20)
        check_memory_budget(receipt,cfg)
        for field in ('device_used','reserved','process_peak_reserved'):
            with self.assertRaises(MemoryError):
                check_memory_budget(dict(receipt,**{field:(4096<<20)+1}),cfg)
        with self.assertRaises(MemoryError):
            check_memory_budget(dict(receipt,action_store=(512<<20)+1),cfg)
        with self.assertRaises(MemoryError):
            check_memory_budget(dict(receipt,training_tape=(512<<20)+1),cfg)

    def test_lane_receipt_checks_tail_coverage_without_counting_work_slots_as_roots(self):
        import torch
        from types import SimpleNamespace as NS
        from scripts.benchmark_duel_v2_soa_device import work_lane_receipt
        q=NS(total_rows=torch.tensor([3]),total_dispatches=torch.tensor([2]),
             peak_selected=torch.tensor([2]),root_visits=torch.tensor([1,1,1]))
        rules=NS(rows=3,work_rows=2,_queues=[q])
        result=work_lane_receipt(rules)
        self.assertEqual((result['roots'],result['work_rows']),(3,2))
        q.root_visits=torch.tensor([2,1,0])
        with self.assertRaisesRegex(AssertionError,'never reached'):
            work_lane_receipt(rules)
        q.root_visits=torch.tensor([1,1,1]);q.peak_selected[0]=3
        with self.assertRaisesRegex(AssertionError,'coverage'):
            work_lane_receipt(rules)

    def test_direct_script_large_capacity_plan_without_cuda(self):
        result=subprocess.run([sys.executable,str(CLI),'--rows','1600'],
            cwd=REPO,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads(result.stdout)
        self.assertEqual(plan['rows'],1600)
        self.assertEqual(plan['capacity']['effective_rule_lanes'],8)
        self.assertEqual(plan['capacity']['effective_inference_rows'],2)
        self.assertFalse(plan['cuda_imported'])
        self.assertFalse(plan['passed'])

    def test_cohort_only_requires_multiple_physical_decisions(self):
        self.assertIn('cohort-only',_stderr(['--cohort-only']))
        buf=io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(_main(['--cohort-only','--max-decisions','4']),0)
        plan=json.loads(buf.getvalue())
        self.assertTrue(plan['cohort_only'])
        self.assertFalse(plan['passed'])

    def test_training_capture_is_one_cohort_not_repeated_timing_exports(self):
        self.assertIn('--cohort-only',_stderr(['--capture-training']))
        buf=io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(_main(['--capture-training','--cohort-only','--max-decisions','4']),0)
        plan=json.loads(buf.getvalue());self.assertTrue(plan['capture_training'])
        self.assertFalse(plan['passed'])

    def test_numerical_receipt_with_numpy_reference_q_serializes(self):
        import time
        import numpy as np
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.benchmark_duel_v2_soa_device import verify_last_searches, save_report
        # Import aliases before mocking the public entry; first import inside
        # the patch would permanently bind information_search.acting_side to it.
        from app.modules.card_game.rl import recovery_search, information_search
        from app.modules.card_game.engine.duel_v2 import acting_side
        tensor = lambda x: torch.tensor(x, dtype=torch.float64)
        cohort = SimpleNamespace(
            diagnostics=dict(q=tensor([[.2, .3]]), prior=tensor([[.5, .5]]),
                             mean=tensor([[.2, .3]]), value=tensor([.25]),
                             gumbel=tensor([[0., 0.]]), errors=torch.tensor([False])),
            light=SimpleNamespace(_rec_count=torch.tensor([2]), _rec_offset=torch.tensor([0]),
                                  _packed_visits=torch.tensor([16, 16]),
                                  _packed_pi=tensor([.5, .5]), _rec_chosen=torch.tensor([0])))
        result=dict(complete=True, visits=np.array([16, 16]), pi=np.array([.5, .5]),
                    root_diagnostics={'completed_q':np.array([.2, .3])},
                    root_prior=np.array([.5, .5]), mean_values=np.array([.2, .3]),
                    value=.25, root_gumbel=np.array([0., 0.]), search_choice=0)
        shared=dict(inference=None, models={'m':object()},
                    jobs=[dict(seed=1, keys={'a':'m', 'b':'m'})])
        with patch('app.modules.card_game.engine.duel_v2.acting_side', return_value='a'), \
             patch('app.modules.card_game.rl.recovery_search.search', return_value=result):
            rows=verify_last_searches(cohort, shared, [{}], [1], {(0, 0):0}, time.monotonic()+10)
        self.assertIs(information_search.acting_side, acting_side)
        with tempfile.TemporaryDirectory() as directory:
            save_report(Path(directory), {'postbatch':rows}, 'complete', time.monotonic())
            saved=json.loads((Path(directory)/'report.json').read_text())
        self.assertEqual(saved['postbatch'][0]['reference_completed_q'], [.2, .3])
        self.assertTrue(all(saved['postbatch'][0]['checks'].values()))

    def test_postbatch_value_oracle_recomputes_and_rejects_changed_forward(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from scripts.benchmark_duel_v2_soa_device import PostbatchValueOracle
        expected = np.array([.2, .3, .5], dtype=np.float32)
        class Bank:
            device = 'cpu'
            max_rows = 2
            candidate_dim = 3
            keys = ('frozen',)
            calls = 0
            changed = False
            def predict_device(self, x, c, valid, ids, context, *, value_only):
                self.calls += 1
                self.context = context.clone()
                self.features = x.clone()
                self_empty = c.shape == (2, 0, 3) and valid.shape == (2, 0)
                assert self_empty and value_only.all()
                p = torch.as_tensor(expected).repeat(2, 1)
                if self.changed:
                    p[:, 0] += .01
                return SimpleNamespace(wdl=p, invalid_rows=torch.zeros(2, dtype=torch.bool))
        bank = Bank()
        model = SimpleNamespace(wdl=lambda x, actor: expected,
                                scores=lambda x, c: np.arange(len(c)))
        oracle = PostbatchValueOracle(bank, {'frozen': model})
        policy = oracle.policy('frozen')
        x = np.array([1., 2., 3.], dtype=np.float32)
        np.testing.assert_array_equal(policy.wdl(x, False), expected)
        np.testing.assert_array_equal(bank.features.numpy(), np.stack([x, x]))
        self.assertFalse(bank.context.any())
        policy.wdl(x.copy(), False)
        self.assertEqual(bank.calls, 1)
        _, value = policy.scores_value(x, [0, 1])
        self.assertAlmostEqual(value, .3)
        self.assertTrue(bank.context.all())
        bank.changed = True
        with self.assertRaisesRegex(AssertionError, 'WDL component parity'):
            policy.wdl(x + 1, True)

    def test_module_import_does_not_load_cuda(self):
        result = subprocess.run(
            [sys.executable, '-c',
             "import sys; import scripts.benchmark_duel_v2_soa_device as m; "
             "assert 'torch' not in sys.modules; assert m.SEARCH['simulations']==32"],
            cwd=REPO, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_plan_has_no_cuda_models_or_files(self):
        before = sys.modules.get('torch')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'unused-out'
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(_main(['--output', str(target)]), 0)
            self.assertIs(sys.modules.get('torch'), before)
            self.assertFalse(target.exists())
            plan = json.loads(buf.getvalue())
        self.assertEqual(plan['mode'], 'plan')
        self.assertFalse(plan['cuda_imported'])
        self.assertFalse(plan['models_loaded'])
        self.assertFalse(plan['files_created'])
        self.assertFalse(plan['passed'])
        self.assertFalse(plan['gpu_qualified'])
        self.assertEqual(plan['learning_updates'], 0)
        self.assertEqual(plan['rows'], 8)
        self.assertEqual(plan['max_decisions'], 1)
        self.assertEqual(plan['warmup'], 2)
        self.assertEqual(plan['repeats'], 3)
        self.assertFalse(plan['require_natural'])
        self.assertEqual(plan['search'], {
            'simulations': 32, 'gumbel_candidates': 16, 'terminal_horizon': 0,
            'teacher_mode': 'network_value', 'opponent_mode': 'policy',
            'prior': 'array_zhenhong', 'extra_guidance_simulations': 0,
            'cpu_fallback': False,
            'world_sampling': 'per_decision',
        })
        self.assertEqual(plan['presets'], [
            'starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk', 'collapse', 'lingke'])
        self.assertTrue(plan['mixed_first_sides'])

    def test_plan_rejects_cpu_and_out_of_range_schema(self):
        self.assertIn('CUDA-only', _stderr(['--device', 'cpu']))
        self.assertIn('rows', _stderr(['--rows', '0']))
        self.assertIn('rows', _stderr(['--rows', '1601']))
        self.assertIn('Direct SoA', _stderr(['--rows','128','--rule-dispatch','direct']))
        self.assertIn('max-decisions', _stderr(['--max-decisions', '0']))
        self.assertIn('max-decisions', _stderr(['--max-decisions', '401']))
        self.assertIn('warmup', _stderr(['--warmup', '-1']))
        self.assertIn('repeats', _stderr(['--repeats', '0']))
        self.assertIn('setup-seconds', _stderr(['--setup-seconds', '0']))
        self.assertIn('execution-seconds', _stderr(['--execution-seconds', '86401']))
        self.assertIn('seed',_stderr(['--seed','-1']))

    def test_run_requires_frozen_inputs_and_new_evals_directory(self):
        self.assertIn('--program --models --teachers --output', _stderr(['--run']))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = root / 'frozen.json'
            program.write_text('{}', encoding='utf8')
            models = root / 'models'
            teachers = root / 'teachers'
            models.mkdir()
            teachers.mkdir()
            self.assertIn('artifacts/rl-evals', _stderr([
                '--run', '--program', str(program), '--models', str(models),
                '--teachers', str(teachers), '--output', str(root / 'out')]))
            existing = REPO / 'artifacts' / 'rl-evals'
            self.assertTrue(existing.is_dir())
            self.assertIn('Use a new output directory', _stderr([
                '--run', '--program', str(program), '--models', str(models),
                '--teachers', str(teachers), '--output', str(existing)]))

    def test_fixed_search_semantics_are_wired_in_source(self):
        source = CLI.read_text(encoding='utf8')
        tree = ast.parse(source)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertIn('create_soa_search', names)
        self.assertIn('DeviceGameLoop', names)
        self.assertIn('DeviceSoARules', names)
        self.assertIn('RawActionPolicySink', names)
        self.assertIn('DeviceValueOnlyInference', names)
        self.assertIn('bind_game_loop', attrs)
        self.assertNotIn('DeviceZhenhongGuide', names)
        self.assertNotIn('DeviceBatchTeacherActions', names)
        self.assertNotIn('DeviceFrozenRules', names)
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'create_soa_search']
        self.assertTrue(calls)
        reserved = {'teacher_mode', 'terminal_horizon', 'opponent_mode', 'prior_guide',
                    'simulations', 'gumbel_candidates', 'device'}
        for call in calls:
            found = {kw.arg for kw in call.keywords}
            self.assertFalse(found & reserved)
            self.assertIn('zhenhong_prior', found)
        rules_calls = [node for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'DeviceSoARules']
        self.assertTrue(any('raw_tape' in {kw.arg for kw in call.keywords} for call in rules_calls))

    def test_plan_does_not_create_output_even_under_evals_name(self):
        target = REPO / 'artifacts' / 'rl-evals' / 'soa-device-cli-plan-should-not-exist'
        self.assertFalse(target.exists())
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(_main(['--output', str(target), '--rows', '8']), 0)
        self.assertFalse(target.exists())
        plan = json.loads(buf.getvalue())
        self.assertEqual(plan['mode'], 'plan')
        self.assertFalse(plan['files_created'])

    def test_require_natural_is_plan_only_until_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(_main(['--require-natural', '--max-decisions', '16']), 0)
        plan = json.loads(buf.getvalue())
        self.assertTrue(plan['require_natural'])
        self.assertEqual(plan['max_decisions'], 16)
        self.assertFalse(plan['passed'])
        self.assertFalse(plan['cuda_imported'])

    def test_root_diagnostic_capture_uses_real_statistics_and_survives_release(self):
        import torch
        from types import SimpleNamespace
        from scripts.benchmark_duel_v2_soa_device import install_root_diagnostics
        from app.modules.card_game.rl.batched_search.device_statistics import DeviceStatisticsPool
        stats=DeviceStatisticsPool(2,3,device='cpu',allow_cpu_reference=True,max_batch=2,max_path_len=2)
        slots=torch.tensor([0,1],dtype=torch.int64)
        generations=torch.ones(2,dtype=torch.int64)
        active=torch.tensor([True,False])
        written=stats.write_nodes(slots,torch.tensor([1,2]),torch.tensor([[.6,.4,0.],[.2,.3,.5]],dtype=torch.float64),
            torch.tensor([.2,-.2],dtype=torch.float64),torch.tensor([[True,True,False],[True,True,True]]),
            expected_generations=generations)
        self.assertFalse(written.error.any())
        stats._visits[0,:2]=torch.tensor([2,1],dtype=torch.int32)
        stats._total[0,:2]=torch.tensor([.5,-.2],dtype=torch.float64)
        noise=torch.zeros((2,3),dtype=torch.float64)
        cohort=SimpleNamespace(stats=stats,search=SimpleNamespace(_noise=noise))
        expected=stats.completed_q(slots,written.generations,active=active).completed_q.clone()
        install_root_diagnostics(cohort,torch,2,3)
        stats.snapshot(slots,written.generations,active=active)
        self.assertTrue(torch.equal(cohort.diagnostics['q'],expected))
        self.assertFalse(cohort.diagnostics['errors'].any())
        stats.release_nodes(slots,written.generations)
        self.assertTrue(torch.equal(cohort.diagnostics['q'],expected))


if __name__ == '__main__':
    unittest.main()
