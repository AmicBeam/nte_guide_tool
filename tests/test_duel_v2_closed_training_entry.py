"""Current entrypoints refuse host rules before directories, seeds or models."""
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch


class ClosedTrainingEntryTest(unittest.TestCase):
    def test_cli_run_worker_and_resume_reject_before_any_output(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            for flag in ('--run','--worker','--resume'):
                out=Path(folder)/flag[2:]
                result=subprocess.run([sys.executable,str(root/'scripts/train_duel_v2_current_round.py'),
                    '--output',str(out),'--runtime','seven','--rule-backend','cuda_vm',flag],
                    cwd=root,capture_output=True,text=True)
                self.assertEqual(result.returncode,2)
                self.assertIn('qualified closed GPU Gumbel driver',result.stderr)
                self.assertFalse(out.exists())

    def test_direct_training_functions_cannot_bypass_gate_with_ready_json(self):
        from scripts.train_duel_v2_current_round import learn
        from app.modules.card_game.rl import seven_round
        from app.modules.card_game.rl.batched_search.device_execution_contract import DeviceExecutionNotReady
        config=dict(runtime='seven',smoke=True,preflight=True,rule_backend='cuda_vm',
                    ready=True,throughput_passed=True,formal_launch_approved=True)
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'absent'
            with patch.object(seven_round,'_historical_cooperative_run',side_effect=AssertionError('CPU rules fallback')):
                for function in (learn,seven_round.run):
                    with self.assertRaises(DeviceExecutionNotReady):function(out,config)
                    self.assertFalse(out.exists())
