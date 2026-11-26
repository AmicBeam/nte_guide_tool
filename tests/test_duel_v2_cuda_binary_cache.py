"""Cached machine code must match its compiler inputs and content hash."""
from pathlib import Path
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from app.modules.card_game.rl.batched_duel.cuda_binary_cache import CudaBinaryCache
from app.modules.card_game.rl.batched_duel.cuda_runtime import CudaModule


class CudaBinaryCacheTests(unittest.TestCase):
    def identity(self,**changes):
        data=dict(source_sha256='source-A',compile_options=['--fmad=false'],gpu_capability=[12,0],
                  nvrtc_version=[13,0],driver_api_version=13000,kernel_specs={'step':['int32*']})
        return dict(data,**changes)

    def test_round_trip_and_every_compatibility_field_invalidates(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=CudaBinaryCache(directory);identity=self.identity();binary=b'\x7fELF\x00kernel\x00payload'
            self.assertEqual(cache.read(identity),(None,'miss'))
            cache.write(identity,binary)
            self.assertEqual(cache.read(identity),(binary,'hit'))
            for changes in (dict(source_sha256='source-B'),dict(compile_options=['--fmad=true']),
                            dict(gpu_capability=[9,0]),dict(nvrtc_version=[13,1]),
                            dict(driver_api_version=14000),dict(kernel_specs={'step':['float32*']})):
                with self.subTest(changes=changes):
                    self.assertEqual(cache.read(self.identity(**changes)),(None,'miss'))
            self.assertFalse(list(Path(directory).glob('*.tmp')))

    def test_corrupt_receipt_or_binary_never_becomes_a_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=CudaBinaryCache(directory);identity=self.identity();key=cache.key(identity)
            binary=b'\x7fELFvalid';cache.write(identity,binary)
            path=Path(directory)/(key+'.cubin');receipt=Path(directory)/(key+'.json')
            path.write_bytes(b'\x7fELFcorrupt')
            self.assertEqual(cache.read(identity),(None,'rejected'))
            cache.write(identity,binary);receipt.write_text('{bad json')
            self.assertEqual(cache.read(identity),(None,'rejected'))
            cache.write(identity,binary)
            data=json.loads(receipt.read_text());data['inputs']['source_sha256']='different'
            receipt.write_text(json.dumps(data))
            self.assertEqual(cache.read(identity),(None,'rejected'))
            cache.write(identity,binary)
            self.assertEqual(cache.read(identity),(binary,'hit'))

    def test_invalid_output_and_unfinished_writes_are_not_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=CudaBinaryCache(directory);identity=self.identity()
            with self.assertRaises(ValueError):cache.write(identity,b'PTX text')
            (Path(directory)/(cache.key(identity)+'.cubin.tmp')).write_bytes(b'\x7fELFunfinished')
            self.assertEqual(cache.read(identity),(None,'miss'))

    def test_runtime_hit_does_not_invoke_compiler(self):
        calls=[];binary=b'\x7fELFcompiled'
        module=SimpleNamespace(_compilation_profile={},_cache_inputs=lambda flags:self.identity(),
            _compile_source_to_cubin=lambda source,flags:calls.append(source) or binary)
        with tempfile.TemporaryDirectory() as directory,patch.dict('os.environ',{'NTE_CUDA_BINARY_CACHE_DIR':directory}):
            self.assertEqual(CudaModule._compiled_binary(module,'source',[],'cubin'),binary)
            self.assertEqual(module._compilation_profile['cache_status'],'miss')
            def forbidden(*args):raise AssertionError('Compiler invoked on cache hit')
            module._compile_source_to_cubin=forbidden
            self.assertEqual(CudaModule._compiled_binary(module,'source',[],'cubin'),binary)
            self.assertEqual(module._compilation_profile['cache_status'],'hit')
            self.assertEqual(module._compilation_profile['compile_seconds'],0.0)
        self.assertEqual(calls,['source'])

    def test_unwritable_cache_does_not_replace_compiled_result(self):
        binary=b'\x7fELFcompiled'
        module=SimpleNamespace(_compilation_profile={},_cache_inputs=lambda flags:self.identity(),
            _compile_source_to_cubin=lambda source,flags:binary)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'file';path.write_text('occupied')
            with patch.dict('os.environ',{'NTE_CUDA_BINARY_CACHE_DIR':str(path)}):
                self.assertEqual(CudaModule._compiled_binary(module,'source',[],'cubin'),binary)
            self.assertIn('cache_write_error',module._compilation_profile)


if __name__=='__main__':unittest.main()
