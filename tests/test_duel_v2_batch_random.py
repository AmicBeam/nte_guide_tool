"""Unit tests verifying C99 MT19937 random primitives against CPython Random.

Tests compile the header into a temporary C shared library using standard library tools,
load it via ctypes, and verify bit-for-bit equivalence with CPython's `random.Random`
across seeds, getrandbits, _randbelow, shuffle, and game commit orders.
"""

import ctypes
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

from app.modules.card_game.rl.batched_duel.random_source import (
    BMT_ERR_EMPTY_RANGE,
    BMT_ERR_INVALID_ARG,
    BMT_ERR_INVALID_BITS,
    BMT_ERR_NULL_POINTER,
    BMT_OK,
    emit_random_source,
    seed_words,
)


class TestPythonSeedWords(unittest.TestCase):
    """Verify seed_words normalization and error handling in Python."""

    def test_seed_zero(self):
        self.assertEqual(seed_words(0), [0])

    def test_seed_one(self):
        self.assertEqual(seed_words(1), [1])

    def test_negative_seeds(self):
        self.assertEqual(seed_words(-1), [1])
        self.assertEqual(seed_words(-42), [42])
        self.assertEqual(seed_words(-0x12345678), [0x12345678])

    def test_64bit_boundary(self):
        seed_63 = (1 << 63) - 1
        expected_63 = [0xFFFFFFFF, 0x7FFFFFFF]
        self.assertEqual(seed_words(seed_63), expected_63)
        self.assertEqual(seed_words(-seed_63), expected_63)

        seed_64 = 1 << 64
        self.assertEqual(seed_words(seed_64), [0, 0, 1])

    def test_256bit_seed(self):
        seed_256 = (1 << 256) - 1
        words = seed_words(seed_256)
        self.assertEqual(len(words), 8)
        self.assertTrue(all(w == 0xFFFFFFFF for w in words))

    def test_normalize_trailing_zeros(self):
        # High zero words must not be appended
        val = 0x5
        self.assertEqual(seed_words(val), [5])

    def test_strict_reject_bool(self):
        with self.assertRaises(TypeError):
            seed_words(True)
        with self.assertRaises(TypeError):
            seed_words(False)

    def test_strict_reject_non_int(self):
        with self.assertRaises(TypeError):
            seed_words("1234")  # type: ignore
        with self.assertRaises(TypeError):
            seed_words(3.14)  # type: ignore
        with self.assertRaises(TypeError):
            seed_words(None)  # type: ignore


class CBmtState(ctypes.Structure):
    _fields_ = [
        ("mt", ctypes.c_uint32 * 624),
        ("index", ctypes.c_int),
    ]


class TestCRandomSharedLibrary(unittest.TestCase):
    """Compile C99 header and compare against CPython Random via ctypes."""

    c_lib = None
    temp_dir = None

    @classmethod
    def setUpClass(cls):
        # Discover C compiler
        compiler = None
        for candidate in ["cc", "gcc", "clang"]:
            p = shutil.which(candidate)
            if p:
                compiler = p
                break

        if not compiler:
            raise unittest.SkipTest(
                "No C compiler (cc/gcc/clang) found in PATH; "
                "skipping C shared library tests (CUDA status not evaluated)."
            )

        cls.temp_dir = tempfile.TemporaryDirectory()
        header_path = os.path.join(cls.temp_dir.name, "random_source.h")
        wrapper_c_path = os.path.join(cls.temp_dir.name, "wrapper.c")
        lib_ext = ".dylib" if sys.platform == "darwin" else ".so"
        so_path = os.path.join(cls.temp_dir.name, f"librandom_test{lib_ext}")

        # Write header
        with open(header_path, "w", encoding="utf-8") as f:
            f.write(emit_random_source())

        # Write C wrapper to expose symbols
        wrapper_c = """
#include "random_source.h"

#if defined(_WIN32) || defined(__CYGWIN__)
#define BMT_EXPORT __declspec(dllexport)
#else
#define BMT_EXPORT __attribute__((visibility("default")))
#endif

BMT_EXPORT bmt_status_t test_bmt_seed_words(bmt_state_t *st, const uint32_t *words, size_t n) {
    return bmt_seed_words(st, words, n);
}

BMT_EXPORT bmt_status_t test_bmt_u32(bmt_state_t *st, uint32_t *out) {
    return bmt_u32(st, out);
}

BMT_EXPORT bmt_status_t test_bmt_getbits64(bmt_state_t *st, int k, uint64_t *out) {
    return bmt_getbits64(st, k, out);
}

BMT_EXPORT bmt_status_t test_bmt_below(bmt_state_t *st, uint64_t n, uint64_t *out) {
    return bmt_below(st, n, out);
}

BMT_EXPORT bmt_status_t test_bmt_shuffle_i32(bmt_state_t *st, int32_t *array, size_t n) {
    return bmt_shuffle_i32(st, array, n);
}

BMT_EXPORT size_t test_bmt_state_size(void) {
    return sizeof(bmt_state_t);
}
"""
        with open(wrapper_c_path, "w", encoding="utf-8") as f:
            f.write(wrapper_c)

        cmd = [
            compiler,
            "-O3",
            "-shared",
            "-fPIC",
            "-std=c99",
            f"-I{cls.temp_dir.name}",
            wrapper_c_path,
            "-o",
            so_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(
                f"C compilation failed with exit code {result.returncode}:\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        cls.c_lib = ctypes.CDLL(so_path)

        # Declare ctypes signatures
        cls.c_lib.test_bmt_seed_words.argtypes = [
            ctypes.POINTER(CBmtState),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
        ]
        cls.c_lib.test_bmt_seed_words.restype = ctypes.c_int

        cls.c_lib.test_bmt_u32.argtypes = [
            ctypes.POINTER(CBmtState),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        cls.c_lib.test_bmt_u32.restype = ctypes.c_int

        cls.c_lib.test_bmt_getbits64.argtypes = [
            ctypes.POINTER(CBmtState),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        cls.c_lib.test_bmt_getbits64.restype = ctypes.c_int

        cls.c_lib.test_bmt_below.argtypes = [
            ctypes.POINTER(CBmtState),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        cls.c_lib.test_bmt_below.restype = ctypes.c_int

        cls.c_lib.test_bmt_shuffle_i32.argtypes = [
            ctypes.POINTER(CBmtState),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
        ]
        cls.c_lib.test_bmt_shuffle_i32.restype = ctypes.c_int

        cls.c_lib.test_bmt_state_size.argtypes = []
        cls.c_lib.test_bmt_state_size.restype = ctypes.c_size_t

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir:
            cls.temp_dir.cleanup()

    def _init_c_state(self, seed: int) -> CBmtState:
        words = seed_words(seed)
        c_words = (ctypes.c_uint32 * len(words))(*words)
        st = CBmtState()
        ret = self.c_lib.test_bmt_seed_words(
            ctypes.byref(st), c_words, len(words)
        )
        self.assertEqual(ret, BMT_OK)
        return st

    def test_state_struct_size(self):
        c_size = self.c_lib.test_bmt_state_size()
        py_size = ctypes.sizeof(CBmtState)
        self.assertEqual(c_size, py_size)

    def test_seeds_and_u32_raw_draws(self):
        """Test diverse seeds: 0, 1, 2^63-1, negative integers, 256-bit integer."""
        seeds = [
            0,
            1,
            (1 << 63) - 1,
            -1,
            -42,
            -9223372036854775808,
            (1 << 256) - 1,
            (1 << 200) + 987654321,
        ]
        for seed in seeds:
            with self.subTest(seed=seed):
                py_rng = random.Random(seed)
                c_st = self._init_c_state(seed)
                for _ in range(50):
                    py_u32 = py_rng.getrandbits(32)
                    c_out = ctypes.c_uint32(0)
                    ret = self.c_lib.test_bmt_u32(
                        ctypes.byref(c_st), ctypes.byref(c_out)
                    )
                    self.assertEqual(ret, BMT_OK)
                    self.assertEqual(c_out.value, py_u32)

    def test_getrandbits_sequences(self):
        """Test sequence of bit lengths 0, 1, 5, 31, 32, 33, 63, 64."""
        test_seeds = [
            0,
            1,
            (1 << 63) - 1,
            -1,
            -12345,
            (1 << 256) - 1,
        ]
        bit_pattern = [0, 1, 5, 31, 32, 33, 63, 64] * 5

        for seed in test_seeds:
            with self.subTest(seed=seed):
                py_rng = random.Random(seed)
                c_st = self._init_c_state(seed)
                for k in bit_pattern:
                    py_val = py_rng.getrandbits(k)
                    c_val = ctypes.c_uint64(0)
                    ret = self.c_lib.test_bmt_getbits64(
                        ctypes.byref(c_st), k, ctypes.byref(c_val)
                    )
                    self.assertEqual(ret, BMT_OK)
                    self.assertEqual(
                        c_val.value,
                        py_val,
                        f"Mismatch for seed={seed}, k={k}: C={c_val.value} != Py={py_val}",
                    )

    def test_randbelow_exact_rejection(self):
        """Test _randbelow on 1, 2, 3, 16, 17, 2^32+1, and other boundaries."""
        test_seeds = [0, 1, (1 << 63) - 1, -7, (1 << 256) - 1]
        ranges = [1, 2, 3, 16, 17, (1 << 32) + 1, 10, 100, (1 << 63) - 1]

        for seed in test_seeds:
            with self.subTest(seed=seed):
                py_rng = random.Random(seed)
                c_st = self._init_c_state(seed)
                for n in ranges:
                    for _ in range(15):
                        py_val = py_rng._randbelow(n)
                        c_val = ctypes.c_uint64(0)
                        ret = self.c_lib.test_bmt_below(
                            ctypes.byref(c_st), n, ctypes.byref(c_val)
                        )
                        self.assertEqual(ret, BMT_OK)
                        self.assertEqual(
                            c_val.value,
                            py_val,
                            f"Mismatch for seed={seed}, n={n}: C={c_val.value} != Py={py_val}",
                        )

    def test_choice_length_1_consumes_random_bits(self):
        """Ensure choice of length 1 calls getrandbits(1) and consumes stream bits."""
        seed = 42
        py_rng = random.Random(seed)
        c_st = self._init_c_state(seed)

        # CPython choice of 1-element list
        py_val = py_rng.choice([999])
        self.assertEqual(py_val, 999)

        c_idx = ctypes.c_uint64(0)
        ret = self.c_lib.test_bmt_below(
            ctypes.byref(c_st), 1, ctypes.byref(c_idx)
        )
        self.assertEqual(ret, BMT_OK)
        self.assertEqual(c_idx.value, 0)

        # After consuming bits for choice(1), subsequent draws must still match exactly
        self.assertEqual(
            self._get_c_bits(c_st, 32), py_rng.getrandbits(32)
        )

    def test_shuffle_arrays(self):
        """Verify Fisher-Yates shuffle produces identical permutations."""
        test_seeds = [0, 1, 42, (1 << 63) - 1, -999]
        sizes = [0, 1, 2, 3, 10, 32, 64]

        for seed in test_seeds:
            for size in sizes:
                with self.subTest(seed=seed, size=size):
                    py_rng = random.Random(seed)
                    c_st = self._init_c_state(seed)

                    py_arr = list(range(size))
                    py_rng.shuffle(py_arr)

                    if size > 0:
                        c_arr = (ctypes.c_int32 * size)(*range(size))
                        ret = self.c_lib.test_bmt_shuffle_i32(
                            ctypes.byref(c_st), c_arr, size
                        )
                        self.assertEqual(ret, BMT_OK)
                        self.assertEqual(list(c_arr), py_arr)
                    else:
                        ret = self.c_lib.test_bmt_shuffle_i32(
                            ctypes.byref(c_st), None, 0
                        )
                        self.assertEqual(ret, BMT_OK)

    def test_xiaozhi_mark_card_commit_order(self):
        """Xiaozhi mark_card order:
        1. choice((-1, 0, 1))
        2. state['rng'] = rng.getrandbits(63)
        """
        current_state_rng = 123456789
        deltas = (-1, 0, 1)

        for step in range(30):
            # Python side
            py_rng = random.Random(current_state_rng)
            py_chosen = py_rng.choice(deltas)
            py_next_rng = py_rng.getrandbits(63)

            # C side
            c_st = self._init_c_state(current_state_rng)
            c_idx = ctypes.c_uint64(0)
            ret1 = self.c_lib.test_bmt_below(
                ctypes.byref(c_st), len(deltas), ctypes.byref(c_idx)
            )
            self.assertEqual(ret1, BMT_OK)
            c_chosen = deltas[c_idx.value]

            c_next_rng = ctypes.c_uint64(0)
            ret2 = self.c_lib.test_bmt_getbits64(
                ctypes.byref(c_st), 63, ctypes.byref(c_next_rng)
            )
            self.assertEqual(ret2, BMT_OK)

            self.assertEqual(
                c_chosen,
                py_chosen,
                f"Xiaozhi choice mismatch at step {step}",
            )
            self.assertEqual(
                c_next_rng.value,
                py_next_rng,
                f"Xiaozhi next_rng mismatch at step {step}",
            )

            # Advance state seed for chained iterations
            current_state_rng = py_next_rng

    def test_zaowu_adler_rng_commit_order(self):
        """Zaowu / Adler _rng order:
        1. state['rng'] = rng.getrandbits(63) committed FIRST.
        2. Same rng instance draws 1 or more choices.
        """
        current_state_rng = 987654321
        items = ("vinyl", "collar", "liquid")
        targets = [101, 102, 103, 104]

        for step in range(30):
            # Python side
            py_rng = random.Random(current_state_rng)
            py_next_rng = py_rng.getrandbits(63)
            py_item = py_rng.choice(items)
            py_target = py_rng.choice(targets)

            # C side
            c_st = self._init_c_state(current_state_rng)
            c_next_rng = ctypes.c_uint64(0)
            ret1 = self.c_lib.test_bmt_getbits64(
                ctypes.byref(c_st), 63, ctypes.byref(c_next_rng)
            )
            self.assertEqual(ret1, BMT_OK)

            c_item_idx = ctypes.c_uint64(0)
            ret2 = self.c_lib.test_bmt_below(
                ctypes.byref(c_st), len(items), ctypes.byref(c_item_idx)
            )
            self.assertEqual(ret2, BMT_OK)
            c_item = items[c_item_idx.value]

            c_target_idx = ctypes.c_uint64(0)
            ret3 = self.c_lib.test_bmt_below(
                ctypes.byref(c_st), len(targets), ctypes.byref(c_target_idx)
            )
            self.assertEqual(ret3, BMT_OK)
            c_target = targets[c_target_idx.value]

            self.assertEqual(
                c_next_rng.value,
                py_next_rng,
                f"Zaowu next_rng mismatch at step {step}",
            )
            self.assertEqual(
                c_item, py_item, f"Zaowu item choice mismatch at step {step}"
            )
            self.assertEqual(
                c_target,
                py_target,
                f"Zaowu target choice mismatch at step {step}",
            )

            current_state_rng = py_next_rng

    def test_c_error_protocol(self):
        """Verify explicit error status returns for invalid arguments."""
        st = self._init_c_state(12345)
        out64 = ctypes.c_uint64(0)

        # n == 0 in bmt_below must fail with BMT_ERR_EMPTY_RANGE (not infinite loop)
        ret = self.c_lib.test_bmt_below(
            ctypes.byref(st), 0, ctypes.byref(out64)
        )
        self.assertEqual(ret, BMT_ERR_EMPTY_RANGE)

        # k < 0 in bmt_getbits64 must fail with BMT_ERR_INVALID_BITS
        ret = self.c_lib.test_bmt_getbits64(
            ctypes.byref(st), -1, ctypes.byref(out64)
        )
        self.assertEqual(ret, BMT_ERR_INVALID_BITS)

        # k > 64 in bmt_getbits64 must fail with BMT_ERR_INVALID_BITS
        ret = self.c_lib.test_bmt_getbits64(
            ctypes.byref(st), 65, ctypes.byref(out64)
        )
        self.assertEqual(ret, BMT_ERR_INVALID_BITS)

        # n == 0 in bmt_seed_words must fail with BMT_ERR_INVALID_ARG
        words = (ctypes.c_uint32 * 1)(1)
        ret = self.c_lib.test_bmt_seed_words(ctypes.byref(st), words, 0)
        self.assertEqual(ret, BMT_ERR_INVALID_ARG)

        # NULL pointer checks
        ret = self.c_lib.test_bmt_below(None, 10, ctypes.byref(out64))
        self.assertEqual(ret, BMT_ERR_NULL_POINTER)

        ret = self.c_lib.test_bmt_below(ctypes.byref(st), 10, None)
        self.assertEqual(ret, BMT_ERR_NULL_POINTER)

        ret = self.c_lib.test_bmt_getbits64(None, 10, ctypes.byref(out64))
        self.assertEqual(ret, BMT_ERR_NULL_POINTER)

        ret = self.c_lib.test_bmt_getbits64(ctypes.byref(st), 10, None)
        self.assertEqual(ret, BMT_ERR_NULL_POINTER)

    def _get_c_bits(self, st: CBmtState, k: int) -> int:
        out = ctypes.c_uint64(0)
        ret = self.c_lib.test_bmt_getbits64(
            ctypes.byref(st), k, ctypes.byref(out)
        )
        self.assertEqual(ret, BMT_OK)
        return out.value


if __name__ == "__main__":
    unittest.main()
