"""Execute the actual memo-epoch helpers, including wrap and interleaved rows."""
import ctypes
import sys
import tempfile
import unittest

import numpy as np

from app.modules.card_game.rl.batched_duel.vm_soa_views import _emit_runtime_head


def source():
    head = _emit_runtime_head()
    reset = head[head.index('__device__ inline void soa_reset_range('):
                 head.index('__device__ inline void soa_reset_caches(')]
    memo = head[head.index('__device__ inline void soa_reset_clone_memo('):
                head.index('__device__ inline int soa_alloc_header(')]
    return '''
struct DomainSoAContext { int rows,row;int* clone_map;int* clone_queue;int clone_epoch; };
const int SOA_MISSING=(-2147483647-1), SOA_TOTAL_OBJECTS=5;
''' + reset + memo + '''
extern "C" __global__ void probe(int* map,int* stamps,int* output){
 int row=ROW;
 DomainSoAContext s;s.rows=3;s.row=row;s.clone_map=map;s.clone_queue=stamps;s.clone_epoch=0;
 soa_reset_clone_memo(&s);
 soa_store_clone(&s,2,10+row);
 output[row*4]=map[2*3+row];
 soa_reset_clone_memo(&s);
 output[row*4+1]=stamps[2*3+row]==s.clone_epoch;
 soa_store_clone(&s,3,20+row);
 s.clone_epoch=2147483647;
 soa_reset_clone_memo(&s);
 output[row*4+2]=s.clone_epoch;
 output[row*4+3]=stamps[3*3+row];
 soa_store_clone(&s,4,30+row);
}
'''


def expected():
    stamps = np.full((5, 3), -2**31, np.int32)
    stamps[4] = 1
    output = np.array([[10+i, 0, 1, -2**31] for i in range(3)], np.int32)
    return stamps, output


class CloneEpochTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == 'win32', 'POSIX host compiler')
    def test_native_epochs_reject_old_memos_wrap_and_isolate_rows(self):
        from app.modules.card_game.rl.batched_duel.native_array_cpu import compile_host_native_library
        text = ('#define __device__\n#define __global__\n' + source().replace('int row=ROW;', 'for(int row=0;row<3;row++){')
                + '}')
        with tempfile.TemporaryDirectory() as directory:
            library = compile_host_native_library(text, directory, timeout=30)
            fn = library.probe
            fn.argtypes = [ctypes.c_void_p]*3; fn.restype = None
            maps = np.zeros((5, 3), np.int32)
            stamps = np.zeros_like(maps); output = np.zeros((3, 4), np.int32)
            fn(maps.ctypes.data, stamps.ctypes.data, output.ctypes.data)
            want_stamps, want_output = expected()
            np.testing.assert_array_equal(stamps, want_stamps)
            np.testing.assert_array_equal(output, want_output)
            np.testing.assert_array_equal(maps[4], [30, 31, 32])

    def test_actual_cuda_epoch_helpers(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.cuda_runtime import CudaModule
        text = source().replace('int row=ROW;', 'int row=blockIdx.x;')
        maps = torch.zeros((5, 3), dtype=torch.int32, device='cuda')
        stamps = torch.zeros_like(maps); output = torch.zeros((3, 4), dtype=torch.int32, device='cuda')
        with CudaModule(text, {'probe': ['int32*']*3}, binary_format='cubin') as module:
            module.launch('probe', [maps, stamps, output], grid=3, block=1).synchronize()
        want_stamps, want_output = expected()
        np.testing.assert_array_equal(stamps.cpu().numpy(), want_stamps)
        np.testing.assert_array_equal(output.cpu().numpy(), want_output)


if __name__ == '__main__':
    unittest.main()
