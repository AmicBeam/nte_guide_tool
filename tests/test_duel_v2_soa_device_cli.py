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
        self.assertIn('rows', _stderr(['--rows', '129']))
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
