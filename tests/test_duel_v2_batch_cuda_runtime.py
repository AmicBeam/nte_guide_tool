"""Unit tests for batched duel CUDA NVRTC runtime and argument validation.

Includes:
- Pure parameter spec, type alias, and scalar bounds tests (runs in any environment without CUDA).
- Device and shape/dtype validation checks.
- Real CUDA compilation and launch tests (strictly skipped if CUDA GPU/driver is unavailable).
- Real MT19937 random source compilation and exact bit alignment against CPython random.Random.
"""

from __future__ import annotations

import math
import os
import random
import unittest
from typing import Any

from app.modules.card_game.rl.batched_duel.cuda_runtime import (
    CudaCompilationError,
    CudaDriverError,
    CudaError,
    CudaModule,
    KernelSpec,
    ParamType,
    convert_scalar,
    normalize_dim3,
    parse_param_type,
    validate_shared_bytes,
)
from app.modules.card_game.rl.batched_duel.random_source import emit_random_source, seed_words


def is_cuda_available() -> bool:
    """Check if PyTorch and CUDA driver/device are genuinely available for testing."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        # Verify device capability can be queried
        _ = torch.cuda.get_device_capability(0)
        return True
    except Exception:
        return False


class TestCudaModuleParamValidation(unittest.TestCase):
    """Tests for parameter type parsing, normalization, and scalar conversion (no CUDA required)."""

    def test_parse_pointer_types(self) -> None:
        for p_str, expected_dtype in [
            ("int32*", "int32"),
            ("int64*", "int64"),
            ("uint8*", "uint8"),
            ("float32*", "float32"),
            ("float64*", "float64"),
        ]:
            pt = parse_param_type(p_str)
            self.assertTrue(pt.is_pointer)
            self.assertFalse(pt.is_scalar)
            self.assertEqual(pt.dtype_name, expected_dtype)
            self.assertEqual(pt.normalized, p_str)

    def test_parse_pointer_aliases_and_const(self) -> None:
        aliases = {
            "const int32*": "int32*",
            "int*": "int32*",
            "const int*": "int32*",
            "long*": "int64*",
            "int64_t*": "int64*",
            "uint8_t*": "uint8*",
            "uchar*": "uint8*",
            "float*": "float32*",
            "double*": "float64*",
            "const double*": "float64*",
        }
        for alias, norm in aliases.items():
            pt = parse_param_type(alias)
            self.assertTrue(pt.is_pointer)
            self.assertEqual(pt.normalized, norm)

    def test_parse_scalar_types(self) -> None:
        for s_str in ["int32", "uint32", "int64", "uint64", "float32", "float64"]:
            pt = parse_param_type(s_str)
            self.assertFalse(pt.is_pointer)
            self.assertTrue(pt.is_scalar)
            self.assertEqual(pt.dtype_name, s_str)
            self.assertEqual(pt.normalized, s_str)

    def test_parse_scalar_aliases(self) -> None:
        aliases = {
            "int": "int32",
            "uint": "uint32",
            "long": "int64",
            "ulong": "uint64",
            "float": "float32",
            "double": "float64",
        }
        for alias, norm in aliases.items():
            pt = parse_param_type(alias)
            self.assertTrue(pt.is_scalar)
            self.assertEqual(pt.normalized, norm)

    def test_parse_struct_types(self) -> None:
        for value in ("struct MyState*", "struct:CustomHeader"):
            with self.assertRaisesRegex(ValueError, "validated ABI"):
                parse_param_type(value)

    def test_cuda_uint32_launch_dimensions_do_not_wrap(self):
        for value in (2**32, (1, 2**32)):
            with self.assertRaises(ValueError):
                normalize_dim3(value, 'grid')
        with self.assertRaises(ValueError):
            validate_shared_bytes(2**32)

    def test_parse_unsupported_types_rejected(self) -> None:
        for bad in ["int16*", "complex64", "", "   ", "void", "char**"]:
            with self.assertRaises((ValueError, TypeError)):
                parse_param_type(bad)

    def test_kernel_spec_construction(self) -> None:
        spec = KernelSpec.from_spec(["int32*", "uint32", "float32*"], struct_kind="ArenaState")
        self.assertEqual(len(spec.params), 3)
        self.assertEqual(spec.struct_kind, "ArenaState")
        self.assertEqual(spec.params[0].normalized, "int32*")
        self.assertEqual(spec.params[1].normalized, "uint32")
        self.assertEqual(spec.params[2].normalized, "float32*")

    def test_convert_scalar_int32(self) -> None:
        pt = parse_param_type("int32")
        # Boundary values
        c_min = convert_scalar(pt, -2_147_483_648)
        self.assertEqual(c_min.value, -2_147_483_648)
        c_max = convert_scalar(pt, 2_147_483_647)
        self.assertEqual(c_max.value, 2_147_483_647)

        # Strict rejection of boolean values
        with self.assertRaises(TypeError):
            convert_scalar(pt, True)
        with self.assertRaises(TypeError):
            convert_scalar(pt, False)

        # Out-of-bounds integers
        with self.assertRaises(ValueError):
            convert_scalar(pt, 2_147_483_648)
        with self.assertRaises(ValueError):
            convert_scalar(pt, -2_147_483_649)

        # Rejects floats and strings
        with self.assertRaises(TypeError):
            convert_scalar(pt, 42.0)
        with self.assertRaises(TypeError):
            convert_scalar(pt, "42")

    def test_convert_scalar_uint32(self) -> None:
        pt = parse_param_type("uint32")
        c_zero = convert_scalar(pt, 0)
        self.assertEqual(c_zero.value, 0)
        c_max = convert_scalar(pt, 4_294_967_295)
        self.assertEqual(c_max.value, 4_294_967_295)

        with self.assertRaises(TypeError):
            convert_scalar(pt, True)
        with self.assertRaises(ValueError):
            convert_scalar(pt, -1)
        with self.assertRaises(ValueError):
            convert_scalar(pt, 4_294_967_296)

    def test_convert_scalar_int64(self) -> None:
        pt = parse_param_type("int64")
        c_min = convert_scalar(pt, -(2**63))
        self.assertEqual(c_min.value, -(2**63))
        c_max = convert_scalar(pt, 2**63 - 1)
        self.assertEqual(c_max.value, 2**63 - 1)

        with self.assertRaises(TypeError):
            convert_scalar(pt, True)
        with self.assertRaises(ValueError):
            convert_scalar(pt, 2**63)
        with self.assertRaises(ValueError):
            convert_scalar(pt, -(2**63) - 1)

    def test_convert_scalar_uint64(self) -> None:
        pt = parse_param_type("uint64")
        c_zero = convert_scalar(pt, 0)
        self.assertEqual(c_zero.value, 0)
        c_max = convert_scalar(pt, 2**64 - 1)
        self.assertEqual(c_max.value, 2**64 - 1)

        with self.assertRaises(TypeError):
            convert_scalar(pt, False)
        with self.assertRaises(ValueError):
            convert_scalar(pt, -1)
        with self.assertRaises(ValueError):
            convert_scalar(pt, 2**64)

    def test_convert_scalar_float32(self) -> None:
        pt = parse_param_type("float32")
        c_val = convert_scalar(pt, 3.14159)
        self.assertAlmostEqual(c_val.value, 3.14159, places=5)

        # Int converted to float
        c_int = convert_scalar(pt, 42)
        self.assertEqual(c_int.value, 42.0)

        # Rejects bool
        with self.assertRaises(TypeError):
            convert_scalar(pt, True)

        # Rejects non-finite floats
        with self.assertRaises(ValueError):
            convert_scalar(pt, float("nan"))
        with self.assertRaises(ValueError):
            convert_scalar(pt, float("inf"))
        with self.assertRaises(ValueError):
            convert_scalar(pt, float("-inf"))

        # Rejects overflow
        with self.assertRaises(ValueError):
            convert_scalar(pt, 1e39)

    def test_convert_scalar_float64(self) -> None:
        pt = parse_param_type("float64")
        c_val = convert_scalar(pt, 1.23456789012345)
        self.assertAlmostEqual(c_val.value, 1.23456789012345, places=12)

        with self.assertRaises(TypeError):
            convert_scalar(pt, False)
        with self.assertRaises(ValueError):
            convert_scalar(pt, float("nan"))
        with self.assertRaises(ValueError):
            convert_scalar(pt, float("inf"))

    def test_normalize_dim3(self) -> None:
        self.assertEqual(normalize_dim3(16, "grid"), (16, 1, 1))
        self.assertEqual(normalize_dim3((16, 2), "grid"), (16, 2, 1))
        self.assertEqual(normalize_dim3((16, 2, 4), "block"), (16, 2, 4))

        # Rejects bool
        with self.assertRaises(TypeError):
            normalize_dim3(True, "grid")
        with self.assertRaises(TypeError):
            normalize_dim3((16, False, 1), "block")

        # Rejects non-positive
        with self.assertRaises(ValueError):
            normalize_dim3(0, "grid")
        with self.assertRaises(ValueError):
            normalize_dim3(-5, "grid")
        with self.assertRaises(ValueError):
            normalize_dim3((16, -1, 1), "grid")

        # Rejects > 3 dims or empty
        with self.assertRaises(ValueError):
            normalize_dim3((1, 2, 3, 4), "grid")
        with self.assertRaises(ValueError):
            normalize_dim3((), "grid")

    def test_validate_shared_bytes(self) -> None:
        self.assertEqual(validate_shared_bytes(0), 0)
        self.assertEqual(validate_shared_bytes(1024), 1024)

        with self.assertRaises(TypeError):
            validate_shared_bytes(True)
        with self.assertRaises(TypeError):
            validate_shared_bytes(128.5)
        with self.assertRaises(ValueError):
            validate_shared_bytes(-1)


class TestCudaModuleRuntime(unittest.TestCase):
    """Tests executing real CUDA compilation and kernels.

    Explicitly skipped when no functional CUDA GPU or driver is available.
    """

    def setUp(self) -> None:
        if not is_cuda_available():
            self.skipTest("CUDA GPU or driver is not available in the current environment")

    def test_cuda_add_kernel_compilation_and_execution(self) -> None:
        import torch

        source = """
        extern "C" __global__ void add_kernel(const float* a, const float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        """
        kernel_specs = {
            "add_kernel": ("float32*", "float32*", "float32*", "int32"),
        }

        with CudaModule(source, kernel_specs) as mod:
            self.assertEqual(mod.backend, "cuda")
            self.assertEqual(mod.actual_backend, "cuda")
            self.assertFalse(mod.closed)
            self.assertTrue(len(mod.source_sha) == 64)
            self.assertIsInstance(mod.gpu_capability, tuple)
            self.assertIn("--std=c++17", mod.compile_options)
            self.assertIn("--fmad=false", mod.compile_options)

            n = 1024
            a = torch.randn(n, device="cuda", dtype=torch.float32)
            b = torch.randn(n, device="cuda", dtype=torch.float32)
            c = torch.empty(n, device="cuda", dtype=torch.float32)

            block = 128
            grid = (n + block - 1) // block

            event = mod.launch("add_kernel", [a, b, c, n], grid=grid, block=block)
            self.assertIsInstance(event, torch.cuda.Event)

            # Wait for completion
            event.synchronize()

            # Verify alignment against CPU calculation
            expected = a.cpu() + b.cpu()
            self.assertTrue(torch.allclose(c.cpu(), expected, atol=1e-6))

    def test_cuda_argument_validation_and_errors(self) -> None:
        import torch

        source = """
        extern "C" __global__ void test_arg_kernel(float* out, int count) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < count) {
                out[idx] = (float)idx;
            }
        }
        """
        with CudaModule(source, {"test_arg_kernel": ("float32*", "int32")}) as mod:
            out = torch.empty(64, device="cuda", dtype=torch.float32)

            # Arity mismatch (too few / too many)
            with self.assertRaises(ValueError):
                mod.launch("test_arg_kernel", [out], grid=1, block=64)
            with self.assertRaises(ValueError):
                mod.launch("test_arg_kernel", [out, 64, 128], grid=1, block=64)

            # Dtype mismatch (int32 tensor for float32*)
            bad_dtype_tensor = torch.empty(64, device="cuda", dtype=torch.int32)
            with self.assertRaises(TypeError):
                mod.launch("test_arg_kernel", [bad_dtype_tensor, 64], grid=1, block=64)

            # Wrong device (CPU tensor)
            cpu_tensor = torch.empty(64, device="cpu", dtype=torch.float32)
            with self.assertRaises(ValueError):
                mod.launch("test_arg_kernel", [cpu_tensor, 64], grid=1, block=64)

            # Non-contiguous tensor
            matrix = torch.empty(16, 16, device="cuda", dtype=torch.float32)
            non_contiguous = matrix.t()
            with self.assertRaises(ValueError):
                mod.launch("test_arg_kernel", [non_contiguous, 64], grid=1, block=64)

            # Scalar bool rejected
            with self.assertRaises(TypeError):
                mod.launch("test_arg_kernel", [out, True], grid=1, block=64)

            # Scalar overflow
            with self.assertRaises(ValueError):
                mod.launch("test_arg_kernel", [out, 2**32], grid=1, block=64)

            # Unknown kernel name
            with self.assertRaises(KeyError):
                mod.launch("nonexistent_kernel", [out, 64], grid=1, block=64)

    def test_cuda_compilation_error_includes_compiler_log(self) -> None:
        bad_source = """
        extern "C" __global__ void bad_syntax_kernel(float* x) {
            this_is_a_syntax_error_here();;;
        }
        """
        with self.assertRaises(CudaCompilationError) as ctx:
            CudaModule(bad_source, {"bad_syntax_kernel": ("float32*",)})

        error_msg = str(ctx.exception)
        self.assertIn("NVRTC compilation failed", error_msg)
        self.assertTrue(
            "this_is_a_syntax_error_here" in error_msg or "error:" in error_msg.lower()
        )

    def test_cuda_module_lifecycle_and_close(self) -> None:
        import torch

        source = """
        extern "C" __global__ void dummy_kernel(int* x) {
            x[0] = 42;
        }
        """
        mod = CudaModule(source, {"dummy_kernel": ("int32*",)})
        t = torch.zeros(1, device="cuda", dtype=torch.int32)
        event = mod.launch("dummy_kernel", [t], grid=1, block=1)
        event.synchronize()
        self.assertEqual(t.item(), 42)

        # Close module
        mod.close()
        self.assertTrue(mod.closed)

        # Duplicate close is safe
        mod.close()

        # Cannot launch after close
        with self.assertRaises(RuntimeError):
            mod.launch("dummy_kernel", [t], grid=1, block=1)

    def test_cuda_random_source_alignment(self) -> None:
        """Verify MT19937 kernel output strictly aligns with Python random.Random."""
        import torch

        source = f"""
        #define BMT_DEVICE __device__
        {emit_random_source()}

        extern "C" __global__ void sample_random_bits_kernel(
            const uint32_t* seeds,
            int seed_count,
            uint64_t* out_bits,
            int k_bits,
            int n
        ) {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {{
                bmt_state_t state;
                bmt_seed_words(&state, seeds, seed_count);
                uint64_t val = 0;
                bmt_getbits64(&state, k_bits, &val);
                out_bits[idx] = val;
            }}
        }}
        """
        kernel_specs = {
            "sample_random_bits_kernel": ("uint32*", "int32", "uint64*", "int32", "int32"),
        }

        with CudaModule(source, kernel_specs) as mod:
            test_cases = [
                (42, 20),
                (123456789, 32),
                (9876543210123456789, 48),
                (112233445566778899, 64),
            ]

            for seed_val, k_bits in test_cases:
                words = seed_words(seed_val)
                py_rng = random.Random(seed_val)
                expected_bits = py_rng.getrandbits(k_bits)

                seeds_t = torch.tensor(words, device="cuda", dtype=torch.uint32)
                out_t = torch.zeros(1, device="cuda", dtype=torch.int64)

                event = mod.launch(
                    "sample_random_bits_kernel",
                    [seeds_t, len(words), out_t, k_bits, 1],
                    grid=1,
                    block=1,
                )
                event.synchronize()

                gpu_val = int(out_t.cpu().item())
                if gpu_val < 0:
                    gpu_val += 2**64

                self.assertEqual(
                    gpu_val,
                    expected_bits,
                    f"Mismatch for seed={seed_val}, k={k_bits}: gpu={gpu_val} != py={expected_bits}",
                )


if __name__ == "__main__":
    unittest.main()
