"""Unit tests for Duel V2 batch inference verification CLI.

Verifies:
1. Default CLI behavior does NOT launch execution, does NOT import Torch,
   does NOT read model weights, and does NOT create output files.
2. Rejection of illegal roots, seconds, devices, and non-existent models.
3. Strict non-overwriting behavior for existing output directories.
4. Monotonic timeout tracking and helper functions.
5. Absolute zero Torch import during CLI configuration checks.
"""

import io
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_duel_v2_batch_inference import (
    DEFAULT_MODELS_PATH,
    DEFAULT_ROOTS,
    DEFAULT_SECONDS,
    MAX_SECONDS,
    MonotonicTimeout,
    build_parser,
    main,
    resolve_models_dir,
    validate_config,
)


class TestBatchInferenceCheckCLI(unittest.TestCase):
    """Test CLI parameter validation, timeout tracker, and default non-execution."""

    def test_torch_not_imported_on_module_load(self):
        """Verify that importing check script does not import torch."""
        result=subprocess.run([sys.executable,'-c',
            "import sys;import scripts.check_duel_v2_batch_inference;assert 'torch' not in sys.modules"],
            cwd=REPO_ROOT,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_default_cli_check_only_does_not_run_nor_import_torch(self):
        """Default CLI invocation must output check json without importing torch or running."""
        stdout_capture = io.StringIO()
        before=sys.modules.get('torch')
        with patch('sys.stdout', stdout_capture):
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertIs(sys.modules.get('torch'),before)

        output = stdout_capture.getvalue()
        self.assertIn('"mode": "check_only"', output)
        self.assertIn('"run_requested": false', output)
        self.assertIn('"status": "config_valid_ready_for_run"', output)

    def test_output_directory_not_created_when_run_is_false(self):
        """Even if --output is specified, it must NOT be created when --run is False."""
        with tempfile.TemporaryDirectory() as temp_parent:
            non_existent_output = Path(temp_parent) / "new_out_dir"
            self.assertFalse(non_existent_output.exists())

            stdout_capture = io.StringIO()
            with patch('sys.stdout', stdout_capture):
                exit_code = main(["--output", str(non_existent_output)])

            self.assertEqual(exit_code, 0)
            self.assertFalse(
                non_existent_output.exists(),
                "Output directory must NOT be created when --run is not passed"
            )

    def test_existing_output_directory_rejected(self):
        """Rejects existing output directory to prevent accidental overwrite."""
        with tempfile.TemporaryDirectory() as existing_dir:
            parser = build_parser()
            args = parser.parse_args(["--output", existing_dir, "--run"])
            with self.assertRaises(ValueError) as ctx:
                validate_config(args, repo_root=REPO_ROOT)
            self.assertIn("already exists", str(ctx.exception))

            # Also check main() returns non-zero
            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                exit_code = main(["--output", existing_dir])
            self.assertNotEqual(exit_code, 0)

    def test_run_requires_output_directory(self):
        """When --run is passed, --output must be explicitly provided."""
        parser = build_parser()
        args = parser.parse_args(["--run"])
        with self.assertRaises(ValueError) as ctx:
            validate_config(args, repo_root=REPO_ROOT)
        self.assertIn("--output is required when --run is specified", str(ctx.exception))

    def test_illegal_roots_rejected(self):
        """Rejects non-positive roots."""
        parser = build_parser()

        for invalid_roots in [0, -1, -100, 1601]:
            args = parser.parse_args(["--roots", str(invalid_roots)])
            with self.assertRaises(ValueError) as ctx:
                validate_config(args, repo_root=REPO_ROOT)
            self.assertIn("roots must be an integer in [1,1600]", str(ctx.exception))

            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                exit_code = main(["--roots", str(invalid_roots)])
            self.assertNotEqual(exit_code, 0)

    def test_illegal_seconds_rejected(self):
        """Rejects seconds <= 0 or > MAX_SECONDS."""
        parser = build_parser()

        for invalid_sec in [0, -5, MAX_SECONDS + 1, 1000]:
            args = parser.parse_args(["--seconds", str(invalid_sec)])
            with self.assertRaises(ValueError) as ctx:
                validate_config(args, repo_root=REPO_ROOT)
            self.assertIn("seconds must be a positive integer", str(ctx.exception))

            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                exit_code = main(["--seconds", str(invalid_sec)])
            self.assertNotEqual(exit_code, 0)

    def test_valid_boundary_seconds(self):
        """Accepts valid boundary seconds like 1 and MAX_SECONDS."""
        parser = build_parser()
        for valid_sec in [1, 120, MAX_SECONDS]:
            args = parser.parse_args(["--seconds", str(valid_sec)])
            cfg = validate_config(args, repo_root=REPO_ROOT)
            self.assertEqual(cfg['seconds'], valid_sec)

    def test_illegal_device_rejected(self):
        """Rejects devices other than cpu and cuda."""
        parser = build_parser()
        # Direct argparse parse failure for unrecognized choice
        with patch('sys.stderr', io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--device", "tpu"])

        # Manual namespace test against validate_config
        args = parser.parse_args([])
        args.device = "vulkan"
        with self.assertRaises(ValueError) as ctx:
            validate_config(args, repo_root=REPO_ROOT)
        self.assertIn("Invalid device", str(ctx.exception))

    def test_nonexistent_models_dir_rejected(self):
        """Rejects non-existent models directory."""
        parser = build_parser()
        args = parser.parse_args(["--models", "completely_non_existent_models_path_xyz"])
        with self.assertRaises(ValueError) as ctx:
            validate_config(args, repo_root=REPO_ROOT)
        self.assertIn("Models directory not found", str(ctx.exception))

    def test_resolve_default_models_dir(self):
        """Resolves default models path successfully."""
        resolved = resolve_models_dir(DEFAULT_MODELS_PATH, repo_root=REPO_ROOT)
        self.assertTrue(resolved.is_dir())
        self.assertTrue((resolved / "serving.json").is_file())

    def test_monotonic_timeout_helper(self):
        """MonotonicTimeout detects expiration and raises TimeoutError with stage."""
        timeout = MonotonicTimeout(seconds=0.02)
        self.assertFalse(timeout.is_expired())
        self.assertGreater(timeout.remaining(), 0.0)

        # Allow timeout to elapse
        time.sleep(0.03)

        self.assertTrue(timeout.is_expired())
        self.assertEqual(timeout.remaining(), 0.0)
        with self.assertRaises(TimeoutError) as ctx:
            timeout.check("unit_test_stage")
        self.assertIn("unit_test_stage", str(ctx.exception))
        self.assertIn("Monotonic timeout", str(ctx.exception))

    def test_no_torch_imported_throughout_test_suite(self):
        """A fresh configuration check must neither load weights nor import Torch."""
        code="""
import sys
from pathlib import Path
original=Path.read_bytes
def guarded(path):
    assert path.suffix not in ('.npz','.pt'), 'Configuration read numeric weights'
    return original(path)
Path.read_bytes=guarded
from scripts.check_duel_v2_batch_inference import main
assert main([])==0
assert 'torch' not in sys.modules
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=REPO_ROOT,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__':
    unittest.main()
