"""Object cache identity, linked image cache and partial-failure restoration."""
from hashlib import sha256
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.modules.card_game.rl.batched_duel.cuda_compile_units import (
    compile_linked_binary, validate_compilation_units,
)


class CudaCompileUnitTests(unittest.TestCase):
    def test_invalid_inputs_before_cuda(self):
        for value in ([], {}, [('a', '')], [('a', 'x'), ('a', 'y')], [('a\0', 'x')], [True]):
            with self.assertRaises(ValueError):
                validate_compilation_units(value)
        self.assertIsNone(validate_compilation_units(None))
        self.assertEqual(validate_compilation_units([('x', 'code')]), (('x', 'code'),))

    def module(self):
        module = SimpleNamespace(_source_sha='original', _compilation_profile={
            'jit_optimization_level': None, 'fast_compile': 'min'}, _drv=object())
        module._cache_inputs = lambda flags: dict(source=module._source_sha, flags=list(flags))
        calls = []
        def compiled(source, flags, fmt, *, source_sha, profile):
            calls.append((source, flags, fmt, source_sha))
            profile.update(cache_status='hit', compile_seconds=0)
            return b'\x7fELF'+source.encode()
        module._compiled_binary = compiled
        return module, calls

    def test_link_cache_and_order_and_options_invalidation(self):
        module, calls = self.module()
        units, flags = [('first', 'x'), ('second', 'y')], ['--fmad=false']
        profile = module._compilation_profile
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, NTE_CUDA_BINARY_CACHE_DIR=directory):
            with patch('app.modules.card_game.rl.batched_duel.cuda_linker.link_cuda_objects', return_value=b'\x7fELFlinked') as link:
                self.assertEqual(compile_linked_binary(module, units, flags), b'\x7fELFlinked')
                self.assertEqual(profile['cache_status'], 'miss')
                compile_linked_binary(module, units, flags)
                self.assertEqual(profile['cache_status'], 'hit')
                self.assertEqual(link.call_count, 1)
                compile_linked_binary(module, list(reversed(units)), flags)
                self.assertEqual(link.call_count, 2)
                compile_linked_binary(module, units, flags+['-DA=1'])
                self.assertEqual(link.call_count, 3)
        self.assertIs(module._compilation_profile, profile)
        self.assertEqual(module._source_sha, 'original')
        self.assertTrue(all('--device-c' in c[1] and c[2] == 'cubin' for c in calls))
        self.assertEqual(calls[0][3], sha256(b'x').hexdigest())

    def test_runtime_object_profiles_and_cache_keys_are_isolated(self):
        from app.modules.card_game.rl.batched_duel.cuda_runtime import CudaModule
        calls = []
        original = {'parent': True}
        module = SimpleNamespace(_compilation_profile=original,
            _cache_inputs=lambda flags: dict(source_sha256='parent'),
            _compile_source_to_cubin=lambda source, flags: calls.append(source) or b'\x7fELF'+source.encode())
        first, second, repeat = {}, {}, {}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, NTE_CUDA_BINARY_CACHE_DIR=directory):
            a = CudaModule._compiled_binary(module, 'x', ['--device-c'], 'cubin', source_sha='x', profile=first)
            b = CudaModule._compiled_binary(module, 'y', ['--device-c'], 'cubin', source_sha='y', profile=second)
            c = CudaModule._compiled_binary(module, 'x', ['--device-c'], 'cubin', source_sha='x', profile=repeat)
        self.assertEqual(calls, ['x', 'y'])
        self.assertEqual(a, c)
        self.assertNotEqual(a, b)
        self.assertNotEqual(first['cache_key'], second['cache_key'])
        self.assertEqual(first['cache_key'], repeat['cache_key'])
        self.assertEqual(repeat['cache_status'], 'hit')
        self.assertEqual(original, {'parent': True})

    def test_constructor_rejects_units_before_cuda_initialization(self):
        from app.modules.card_game.rl.batched_duel.cuda_runtime import CudaModule
        with self.assertRaises(ValueError):
            CudaModule('source', {'kernel': []}, compilation_units=[])
        with self.assertRaisesRegex(ValueError, 'requires CUBIN'):
            CudaModule('source', {'kernel': []}, compilation_units=[('a', 'code')], binary_format='ptx')

    def test_independent_concurrent_compiles_keep_identity_and_link_order(self):
        import threading
        module, _ = self.module()
        rendezvous = threading.Barrier(2, timeout=5)
        def compile_one(source, flags, fmt, *, source_sha, profile):
            self.assertEqual(module._source_sha, 'original')
            self.assertEqual(source_sha, sha256(source.encode()).hexdigest())
            rendezvous.wait()
            profile['compile_seconds'] = 1
            return b'\x7fELF'+source.encode()
        module._compiled_binary = compile_one
        with patch.dict(os.environ, NTE_CUDA_BINARY_CACHE_DIR=''), patch(
                'app.modules.card_game.rl.batched_duel.cuda_linker.link_cuda_objects', return_value=b'\x7fELFlinked') as link:
            compile_linked_binary(module, [('a', 'x'), ('b', 'y')], [], workers=2)
            self.assertEqual([pair[0] for pair in link.call_args.args[1]], ['a', 'b'])
        self.assertEqual(module._compilation_profile['compile_seconds'], 2)

    def test_compiler_failure_restores_module(self):
        module, _ = self.module()
        original = module._compilation_profile
        def failed(*args, **kwargs):
            raise RuntimeError('compile failed')
        module._compiled_binary = failed
        with self.assertRaisesRegex(RuntimeError, 'compile failed'):
            compile_linked_binary(module, [('x', 'code')], [])
        self.assertEqual(module._source_sha, 'original')
        self.assertIs(module._compilation_profile, original)


if __name__ == '__main__':
    unittest.main()
