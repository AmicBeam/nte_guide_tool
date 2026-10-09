"""CPU contract tests for the CUDA-only bulk Duel V2 rule backend."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from app.modules.card_game.rl.batched_duel import gpu_rule_backend as gb
from app.modules.card_game.rl.batched_duel import gpu_rule_entries as entries
from app.modules.card_game.rl.batched_duel import gpu_rule_program as gp
from app.modules.card_game.rl.batched_search.cooperative import _PROTOCOL_METHODS
from app.modules.card_game.rl.rule_ir.portable_program import ValidationError, compute_identity


REPO = Path(__file__).resolve().parents[1]
CLI = REPO / 'scripts' / 'check_duel_v2_gpu_rule_backend.py'


def _can_capture():
    return sys.version_info[:2] == (3, 10)


def _cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


class GpuRuleBackendContractTest(unittest.TestCase):
    def test_standalone_guard_requires_an_explicit_boolean(self):
        with self.assertRaisesRegex(TypeError, 'Boolean standalone'):
            gb.GpuRuleBackend(None, guard_standalone_entities=1)
    def test_protocol_and_readiness_flags(self):
        self.assertEqual(gb.PROTOCOL_METHODS, _PROTOCOL_METHODS)
        self.assertFalse(gb.GpuRuleBackend.supports_full_duel_rules)
        self.assertFalse(gb.GpuRuleBackend.device_verified)
        self.assertTrue(gb.GpuRuleBackend.supports_gpu)
        self.assertEqual(gb.GpuRuleBackend.backend_kind, 'cuda_vm')
        self.assertEqual(gb.METHOD_ENTRY['decision'], 'public_legal_decision')
        self.assertEqual(gb.METHOD_ENTRY['value_observation'], 'public_legal_decision')
        self.assertEqual(gb.METHOD_ENTRY['sample_world'], 'sample_world')
        self.assertEqual(gb.METHOD_ENTRY['commit'], 'apply_action')

    def test_cpu_device_is_rejected_before_program_or_torch(self):
        code = (
            'from app.modules.card_game.rl.batched_duel.gpu_rule_backend import '
            'GpuRuleBackend, GpuRuleBackendError\n'
            'try:\n'
            '    GpuRuleBackend(None, device="cpu")\n'
            'except GpuRuleBackendError as exc:\n'
            '    assert "CPU" in str(exc)\n'
            'else:\n'
            '    raise SystemExit("cpu device was accepted")\n'
            'import sys\n'
            'assert "torch" not in sys.modules\n'
        )
        subprocess.run([sys.executable, '-c', code], check=True, cwd=REPO, capture_output=True)

    def test_production_path_does_not_import_host_extractors_or_numpy_encoder(self):
        owned = [
            REPO / 'app/modules/card_game/rl/batched_duel/gpu_rule_backend.py',
            REPO / 'app/modules/card_game/rl/batched_duel/gpu_rule_arguments.py',
            REPO / 'app/modules/card_game/rl/batched_duel/gpu_rule_projection.py',
        ]
        text = '\n'.join(path.read_text() for path in owned)
        self.assertNotIn('extract_raw_tokens', text)
        self.assertNotIn('encode_league', text)
        self.assertNotIn('from app.modules.card_game.rl.cross_lineup import encode', text)
        self.assertNotIn('import ResidentRuleBackend', text)
        self.assertNotIn('process_backend import ResidentRuleBackend', text)
        self.assertIn('encode_observation_tokens', text)
        self.assertIn('encode_tokens', text)

    def test_inspect_check_does_not_import_torch(self):
        code = (
            'import sys\n'
            'from app.modules.card_game.rl.batched_duel.gpu_rule_backend import inspect_backend_configuration\n'
            'status = inspect_backend_configuration(("acting_side", "finished"))\n'
            'assert status["mode"] == "check"\n'
            'assert status["supports_full_duel_rules"] is False\n'
            'assert status["device_verified"] is False\n'
            'assert status["hidden_cpu_fallback"] is False\n'
            'assert status["actual_backend"] == "cuda"\n'
            'assert "torch" not in sys.modules\n'
            'assert status["torch_imported"] is False\n'
        )
        subprocess.run([sys.executable, '-c', code], check=True, cwd=REPO, capture_output=True)


@unittest.skipUnless(_can_capture(), 'CPython 3.10 source capture required')
class GpuRuleBackendCapabilityTest(unittest.TestCase):
    def test_full_entries_fail_closed_before_cuda(self):
        program, status = gp.build_gpu_rule_program()
        self.assertFalse(status['entries']['public_legal_decision']['declared_capabilities_ready'])
        with self.assertRaises(gb.GpuRuleCapabilityError) as caught:
            gb.GpuRuleBackend(program)
        message = str(caught.exception)
        self.assertIn('public_legal_decision', message)
        self.assertIn('sample_world', message)

    def test_stale_source_hash_is_rejected(self):
        program, _ = gp.build_gpu_rule_program(('acting_side', 'finished'), lower=False)
        payload = program.to_dict()
        victim = next(iter(payload['source_hashes']))
        payload['source_hashes'] = dict(payload['source_hashes'])
        payload['source_hashes'][victim] = 'ab' * 32
        payload['identity'] = compute_identity(payload)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / 'stale.json'
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValidationError, 'source hash mismatch'):
                gb.GpuRuleBackend(path, entries=('acting_side', 'finished'))

    def test_light_entries_pass_capability_then_require_cuda(self):
        program, status = gp.build_gpu_rule_program(entries.LIGHT_ENTRY_NAMES)
        for name in entries.LIGHT_ENTRY_NAMES:
            self.assertTrue(status['entries'][name]['declared_capabilities_ready'], name)
            self.assertFalse(status['entries'][name]['device_verified'])
        import torch
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(gb.GpuRuleBackendError, 'Actual CUDA required'):
                gb.GpuRuleBackend(program, entries=entries.LIGHT_ENTRY_NAMES)

    def test_full_entries_allow_only_explicit_denied_uuid_guard(self):
        import torch
        program, status = gp.build_gpu_rule_program()
        self.assertEqual(status['program_missing']['intrinsics'], ['uuid.uuid4'])
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(gb.GpuRuleBackendError, 'Actual CUDA required'):
                gb.GpuRuleBackend(program, guard_standalone_entities=True)

    def test_inspect_reports_real_entry_blockers(self):
        status = gb.inspect_backend_configuration()
        self.assertEqual(status['mode'], 'check')
        self.assertFalse(status['supports_full_duel_rules'])
        self.assertFalse(status['device_verified'])
        self.assertIn('public_legal_decision', status['blocked_entries'])
        self.assertIn('sample_world', status['blocked_entries'])
        self.assertIn('acting_side', status['capability_ready_entries'])
        missing = status['blocked_entries']['public_legal_decision']
        self.assertTrue(missing['opcodes'] or missing['intrinsics'])


class GpuRuleBackendCliTest(unittest.TestCase):
    def test_help_documents_supported_flags(self):
        out = subprocess.check_output([sys.executable, str(CLI), '--help'], cwd=REPO, text=True)
        for flag in ('--check', '--run', '--output', '--entries', '--max-cases', '--max-seconds'):
            self.assertIn(flag, out)

    def test_default_check_does_not_write_or_import_torch(self):
        with tempfile.TemporaryDirectory() as raw:
            proc = subprocess.run(
                [sys.executable, str(CLI)],
                cwd=REPO, text=True, capture_output=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload['mode'], 'check')
            self.assertFalse(payload['supports_full_duel_rules'])
            self.assertFalse(payload['device_verified'])
            self.assertFalse(payload['torch_imported'])
            self.assertFalse(payload['hidden_cpu_fallback'])
            self.assertEqual(list(Path(raw).iterdir()), [])


@unittest.skipUnless(_cuda_available(), 'Actual CUDA required')
class GpuRuleBackendCudaParityTest(unittest.TestCase):
    def test_light_entry_parity_and_decision_fail_closed(self):
        from app.modules.card_game.content.duel_v2.catalog import STARTER_DECKS
        from app.modules.card_game.engine.duel_v2 import new_game

        program, status = gp.build_gpu_rule_program(entries.LIGHT_ENTRY_NAMES)
        self.assertFalse(status['full_duel_rules'])
        collapse = next(item for item in STARTER_DECKS if item['id'] == 'collapse')
        lingke = next(item for item in STARTER_DECKS if item['id'] == 'lingke')
        state = new_game(seed=20261009, first_side='a', skip_mulligan=True,
                         decks={'a': collapse, 'b': lingke})
        with gb.GpuRuleBackend(program, entries=entries.LIGHT_ENTRY_NAMES, max_games=2, max_states=8) as backend:
            self.assertFalse(backend.supports_full_duel_rules)
            self.assertFalse(backend.device_verified)
            refs = backend.import_states([state], ['game-light'], [0])
            ref = refs[0]
            self.assertEqual(backend.acting_side(ref), entries.acting_side(state))
            self.assertEqual(backend.finished(ref), entries.finished(state))
            self.assertEqual(backend.terminal_value(ref, 'a'), entries.terminal_value(state, 'a'))
            self.assertEqual(backend.terminal_value(ref, 'b'), entries.terminal_value(state, 'b'))
            cleaned = backend.clean(ref)
            exported = backend.export_states([cleaned])[0]
            self.assertEqual(exported['events'], [])
            self.assertNotIn('_public_board', exported)
            with self.assertRaises(gb.GpuRuleCapabilityError):
                backend.decision(ref, 'a')
            stats = backend.stats()
            self.assertEqual(stats['allocated_slots'], 2)
            self.assertGreaterEqual(stats['free_slots'], 1)
            backend.release([ref, cleaned])
            self.assertEqual(backend.stats()['live_states'], 0)


if __name__ == '__main__':
    unittest.main()
