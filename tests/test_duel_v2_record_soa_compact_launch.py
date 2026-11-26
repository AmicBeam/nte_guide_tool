"""Launch geometry only: identical emitted kernel, per-row outputs and failures."""
import ctypes
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from app.modules.card_game.rl.batched_duel.native_array_cpu import HOST_CUDA_SHIMS, _cptr, compile_host_native_library
from app.modules.card_game.rl.batched_duel.record_soa import RecordSoABatch, RecordSoAError, compile_schema, pack_game_states
from app.modules.card_game.rl.batched_duel.record_soa_compact import CudaRecordSoACompact, emit_compact_source

THREADS = (1, 2, 4, 8, 16, 32, 64, 128)


def schema():
    return compile_schema({'Game': {'kind': 'record', 'capacity': 4, 'fields': {
        'left': '@Item', 'right': '@Item', 'text': 'str'}},
        'Item': {'kind': 'record', 'capacity': 8, 'fields': {'n': 'int'}}},
        max_strings=16, max_string_bytes=128)


LAUNCHER = r'''
extern "C" void host_geometry(
 const long long* si,const double* sf,const int* sw,const unsigned char* sb,const int* sr,
 long long* di,double* df,int* dw,unsigned char* db,int* dr,
 int* map,int* stack,unsigned char* color,int* strings,int* status,int rows,int threads){
 blockDim.x=threads;gridDim.x=(rows+threads-1)/threads;
 for(unsigned int b=0;b<gridDim.x;++b){blockIdx.x=b;
  for(unsigned int t=0;t<blockDim.x;++t){threadIdx.x=t;
   record_soa_compact(si,sf,sw,sb,sr,di,df,dw,db,dr,map,stack,color,strings,status,rows);
  }
 }
}
'''


class CompactGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = schema()
        cls.temp = tempfile.TemporaryDirectory(prefix='compact_geometry_')
        cls.library = compile_host_native_library(HOST_CUDA_SHIMS + emit_compact_source(cls.schema) + LAUNCHER,
                                                  Path(cls.temp.name), timeout=30)
        cls.fn = cls.library.host_geometry
        cls.fn.argtypes = (ctypes.c_void_p,) * 15 + (ctypes.c_int,) * 2
        cls.fn.restype = None

    @classmethod
    def tearDownClass(cls):
        cls.fn = cls.library = None
        cls.temp.cleanup()

    def run_geometry(self, packed, threads):
        s = self.schema
        dest = RecordSoABatch(s, packed.rows)
        arrays = [np.zeros((s.total_objects, packed.rows), np.int32),
                  np.zeros((s.total_objects, packed.rows), np.int32),
                  np.zeros((s.total_objects, packed.rows), np.uint8),
                  np.zeros((s.max_strings, packed.rows), np.int32),
                  np.full(packed.rows, -999, np.int32)]
        ordered = [packed.buffers[k] for k in ('i64', 'f64', 'i32', 'u8')] + [packed.roots]
        ordered += [dest.buffers[k] for k in ('i64', 'f64', 'i32', 'u8')] + [dest.roots] + arrays
        self.fn(*[_cptr(x) for x in ordered], packed.rows, threads)
        return dest, arrays

    def test_all_geometries_exact_all_outputs_and_source_pure(self):
        for rows in (0, 1, 3, 32, 35, 129):
            item = {'n': 7}
            packed = pack_game_states([{'right': item, 'left': item, 'text': '真红'} for _ in range(rows)], self.schema)
            # Both successful rows and multiple invalid roots retain original status and partial writes.
            if rows > 2:
                packed.roots[1] = 0
                packed.roots[2] = self.schema.total_objects + 1
            if rows > 4:
                game = self.schema.pools[self.schema.root]
                packed.buffers['u8'][game.tag, 3] = 99
                packed.buffers['i32'][game.count, 4] = game.max_items + 1
            before = {k: a.copy() for k, a in packed.buffers.items()}
            expected, expected_aux = self.run_geometry(packed, 128)
            for threads in THREADS:
                actual, aux = self.run_geometry(packed, threads)
                for k in actual.buffers:
                    np.testing.assert_array_equal(actual.buffers[k], expected.buffers[k])
                    np.testing.assert_array_equal(packed.buffers[k], before[k])
                np.testing.assert_array_equal(actual.roots, expected.roots)
                for a, b in zip(aux, expected_aux):
                    np.testing.assert_array_equal(a, b)
                if rows:
                    self.assertNotIn(-999, aux[-1])

    def test_parameter_validation_and_default(self):
        self.assertEqual(CudaRecordSoACompact(self.schema).threads_per_block, 128)
        for value in THREADS:
            runtime = CudaRecordSoACompact(self.schema, threads_per_block=value)
            self.assertEqual(runtime.threads_per_block, value)
        for value in (True, False, 0, -1, 3, 127, 129, 256, 1.0, '1', None):
            with self.subTest(value=value), self.assertRaises(RecordSoAError):
                CudaRecordSoACompact(self.schema, threads_per_block=value)
        runtime = CudaRecordSoACompact(self.schema)
        with self.assertRaises(RecordSoAError):
            runtime.threads_per_block = 3
        self.assertEqual(runtime.threads_per_block, 128)

    def test_actual_device_launch_grid_and_unchanged_kernel(self):
        import torch
        source = emit_compact_source(self.schema)
        self.assertNotIn('__syncthreads', source)
        self.assertNotIn('__shared__', source)
        for rows in (0, 1, 32, 35, 129):
            batch = pack_game_states([{} for _ in range(rows)], self.schema).to_torch('cpu')
            for threads in THREADS:
                runtime = CudaRecordSoACompact(self.schema, threads_per_block=threads)
                module = SimpleNamespace(device='cpu', launch=Mock(return_value='event'))
                runtime._module = module
                workspace = runtime.prepare_workspace(rows)
                _, _, event = runtime.compact_into_device(batch, workspace)
                self.assertEqual(emit_compact_source(self.schema), source)
                if not rows:
                    module.launch.assert_not_called()
                    self.assertIsNone(event)
                else:
                    self.assertEqual(module.launch.call_args.kwargs,
                                     dict(grid=(rows + threads - 1) // threads, block=threads, capture_safe=True))
                    self.assertEqual(event, 'event')


if __name__ == '__main__':
    unittest.main()
