"""Actual-device scalar parity using the full canonical heap/library helpers."""
import unittest

from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
from app.modules.card_game.rl.batched_duel.native_soa_source import emit_native_soa_source
from app.modules.card_game.rl.batched_duel.native_scalar_ops import emit_native_scalar_source


def probe_source(image, schema):
    source, _ = emit_native_soa_source(image, schema, VmLimits())
    marker = '/* Static native-array execute.'
    if source.count(marker) != 1:
        raise ValueError('Canonical scalar device library boundary changed')
    # The component probe uses real canonical libraries, but no VM execute
    # kernel or rule PC dispatch. This keeps the device compile bounded.
    source = source[:source.index(marker)] + emit_native_scalar_source()
    source += '''
template<int OP> __device__ inline int scalar_original_binary(Context* c,int a,int b){
 int out=MISSING;if(stdlib_binary(c,a,b,OP,&out))return out;return numeric_binary(c,a,b,OP);
}
__device__ inline int scalar_original_negative(Context* c,int ref){
 if(kind(c,ref)==TAG_FLOAT){double a=0;read_float64(&c->h,ref,&a);return make_float64(&c->h,-a);}
 long long a=0;numeric_int(c,ref,&a);
 if(a==(-9223372036854775807LL-1)){c->h.error=ERROR_INT_OVERFLOW;return MISSING;}
 return make_int64(&c->h,-a);
}
} // namespace duel_rule_vm
extern "C" __global__ void native_scalar_probe(
 int* nodes,int* edges,unsigned char* payload,int* counts,const int* cn,const int* ce,
 const unsigned char* cp,int cnc,int cec,int cpc,int* output,int rows){
 using namespace duel_rule_vm;
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
 for(int side=0;side<2;side++){
  int index=row*2+side;
  Context c{};c.error_value=MISSING;c.raised=MISSING;
  init_heap(&c.h,nodes+index*64*4,edges+index*128*2,payload+index*1024,
   counts+index*3,64,128,1024,cn,ce,cp,cnc,cec,cpc);
  int a=MISSING,b=MISSING,result=MISSING;
  switch(row%16){
   case 0:a=make_int64(&c.h,9223372036854775807LL);b=make_int64(&c.h,1);
    result=side?native_scalar_binary<OP_BINARY_ADD>(&c,a,b):scalar_original_binary<OP_BINARY_ADD>(&c,a,b);break;
   case 1:a=make_int64(&c.h,(-9223372036854775807LL-1));b=make_int64(&c.h,-1);
    result=side?native_scalar_binary<OP_BINARY_MULTIPLY>(&c,a,b):scalar_original_binary<OP_BINARY_MULTIPLY>(&c,a,b);break;
   case 2:a=make_int64(&c.h,-7);b=make_int64(&c.h,3);
    result=side?native_scalar_binary<OP_BINARY_FLOOR_DIVIDE>(&c,a,b):scalar_original_binary<OP_BINARY_FLOOR_DIVIDE>(&c,a,b);break;
   case 3:a=make_int64(&c.h,7);b=make_int64(&c.h,2);
    result=side?native_scalar_binary<OP_BINARY_TRUE_DIVIDE>(&c,a,b):scalar_original_binary<OP_BINARY_TRUE_DIVIDE>(&c,a,b);break;
   case 4:a=make_bool(&c.h,1);b=make_bool(&c.h,1);
    result=side?native_scalar_binary<OP_BINARY_AND>(&c,a,b):scalar_original_binary<OP_BINARY_AND>(&c,a,b);break;
   case 5:a=make_int64(&c.h,(-9223372036854775807LL-1));
    result=side?native_scalar_negative(&c,a):scalar_original_negative(&c,a);break;
   case 6:a=make_bool(&c.h,1);result=side?native_scalar_not(&c,a):!truth_value(&c,a);break;
   case 7:a=make_int64(&c.h,1);b=make_bool(&c.h,1);
    result=side?native_scalar_compare<2>(&c,a,b):compare(&c,a,b,2);break;
   case 8:a=make_int64(&c.h,9007199254740993LL);b=make_float64(&c.h,9007199254740992.0);
    result=side?native_scalar_compare<4>(&c,a,b):compare(&c,a,b,4);break;
   case 9:a=make_float64(&c.h,0.0);result=side?native_scalar_negative(&c,a):scalar_original_negative(&c,a);break;
   case 10:a=make_utf8(&c.h,"ab",2);b=make_utf8(&c.h,"cd",2);
    result=side?native_scalar_binary<OP_BINARY_ADD>(&c,a,b):scalar_original_binary<OP_BINARY_ADD>(&c,a,b);break;
   case 11:a=make_list(&c.h);list_append(&c.h,a,make_int64(&c.h,7));b=make_int64(&c.h,3);
    result=side?native_scalar_binary<OP_BINARY_MULTIPLY>(&c,a,b):scalar_original_binary<OP_BINARY_MULTIPLY>(&c,a,b);break;
   case 12:{a=stdlib_new_counter(&c);b=stdlib_new_counter(&c);
    int key=make_utf8(&c.h,"hp",2);dict_set(&c.h,a,key,make_int64(&c.h,2));dict_set(&c.h,b,key,make_int64(&c.h,3));
    result=side?native_scalar_binary<OP_BINARY_ADD>(&c,a,b):scalar_original_binary<OP_BINARY_ADD>(&c,a,b);break;}
   case 13:a=make_int64(&c.h,1);b=make_int64(&c.h,2);c.h.max_nodes=c.h.counts[0];
    result=side?native_scalar_binary<OP_BINARY_ADD>(&c,a,b):scalar_original_binary<OP_BINARY_ADD>(&c,a,b);break;
   case 14:a=make_bool(&c.h,1);b=make_bool(&c.h,0);
    result=side?native_scalar_compare<3>(&c,a,b):compare(&c,a,b,3);break;
   case 15:a=make_int64(&c.h,(-9223372036854775807LL-1));b=make_int64(&c.h,-1);
    result=side?native_scalar_binary<OP_BINARY_MODULO>(&c,a,b):scalar_original_binary<OP_BINARY_MODULO>(&c,a,b);break;
  }
  output[index*3]=result;output[index*3+1]=c.error;output[index*3+2]=c.h.error;
 }
}
'''
    return source


def run_probe(image, schema):
    import torch
    from app.modules.card_game.rl.batched_duel.cuda_runtime import CudaModule
    rows = 32
    nodes = torch.zeros((rows, 2, 64, 4), dtype=torch.int32, device='cuda')
    edges = torch.zeros((rows, 2, 128, 2), dtype=torch.int32, device='cuda')
    payload = torch.zeros((rows, 2, 1024), dtype=torch.uint8, device='cuda')
    counts = torch.zeros((rows, 2, 3), dtype=torch.int32, device='cuda')
    output = torch.zeros_like(counts)
    constants = [torch.tensor(getattr(image, key).copy(), device='cuda') for key in ('nodes', 'edges', 'payload')]
    source = probe_source(image, schema)
    with CudaModule(source, {'native_scalar_probe': ['int32*','int32*','uint8*','int32*',
            'int32*','int32*','uint8*','int32','int32','int32','int32*','int32']}, binary_format='cubin') as module:
        module.ensure_stack(131072)
        module.launch('native_scalar_probe', [nodes, edges, payload, counts, *constants,
            len(image.nodes), len(image.edges), len(image.payload), output, rows], grid=1, block=rows).synchronize()
        for name, tensor in (('nodes', nodes), ('edges', edges), ('payload', payload), ('counts', counts), ('output', output)):
            if not torch.equal(tensor[:, 0], tensor[:, 1]):
                raise AssertionError('Scalar device parity failed: ' + name)
        if not (counts[:, :, 0] > 0).all():
            raise AssertionError('Scalar device probe did not execute every row')
        host_output = output[:, 1].cpu().tolist()
        host_nodes, host_payload = nodes[:, 1].cpu(), payload[:, 1].cpu()
        for row, (ref, error, heap_error) in enumerate(host_output):
            case = row % 16
            expected = (0, 7) if case in (0, 1, 5) else (0, 1) if case == 13 else (106, 0) if case == 12 else (0, 0)
            if (error, heap_error) != expected:
                raise AssertionError(('Unexpected scalar status', case, error, heap_error))
            if case in (6, 7, 8, 14) and ref != (0 if case == 6 else 1):
                raise AssertionError(('Unexpected scalar predicate', case, ref))
            if case in (2, 4, 15):
                tag, offset, size, _ = host_nodes[row, ref].tolist()
                value = int.from_bytes(bytes(host_payload[row, offset:offset+size].tolist()), 'little', signed=True)
                if tag != 2 or value != {2: -3, 4: 1, 15: 0}[case]:
                    raise AssertionError(('Unexpected scalar value', case, tag, value))
        return dict(passed=True, rows=rows, cases=16, outputs=output[:, 1].cpu().tolist(),
                    compilation=module.compilation_profile)


class NativeScalarCUDATests(unittest.TestCase):
    def test_actual_cuda_scalar_heap_and_error_parity(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from tests.test_duel_v2_native_shard_runs import fixture_program, fixture_schema
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        self.assertTrue(run_probe(lower_program(fixture_program()), fixture_schema())['passed'])


if __name__ == '__main__':
    unittest.main()
