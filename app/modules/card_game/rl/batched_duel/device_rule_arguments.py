"""CUDA-resident actor/seed VM argument binder. Not a rule executor."""
from dataclasses import dataclass

from .cuda_vm import DeviceGraph
from .schema import TAG_INT, TAG_STR
from .vm_heap import (
    ERROR_INVALID_REF,
    ERROR_NODE_CAPACITY,
    ERROR_OK,
    ERROR_PAYLOAD_CAPACITY,
    ERROR_TYPE_MISMATCH,
    emit_vm_heap_source,
)


@dataclass(frozen=True)
class BoundDeviceArguments:
    arena: DeviceGraph
    argument_refs: object
    errors: object


_BIND_SPEC = (
    'int32*', 'int32*', 'uint8*', 'int32*', 'int32*',
    'int64*', 'int64*', 'uint8*', 'int32*', 'int32*',
    'int32', 'int32', 'int32', 'int32', 'int32',
)
_ACTION_SPEC = ('int32*','int32*','uint8*','int32*','int32*') * 2 + (
    'int64*','uint8*','int32*','int32*','int32','int32','int32','int32',
    'int32','int32','int32',
)


def emit_binder_source():
    return emit_vm_heap_source() + r'''
using namespace duel_vm_heap;
__device__ inline int make_seed_int(Heap* heap, const long long* limbs){
 if(heap->error!=ERROR_OK) return -1;
 unsigned char raw[41];
 for(int i=0;i<5;i++){
  unsigned long long u=(unsigned long long)limbs[i];
  for(int b=0;b<8;b++) raw[i*8+b]=(unsigned char)((u>>(b*8))&0xFF);
 }
 int len=40;
 while(len>1 && raw[len-1]==0 && (raw[len-2]&0x80)==0) len--;
 if(raw[len-1]&0x80){ raw[len]=0; len++; }
 if(heap->counts[2]+len>heap->max_payload){ heap->error=ERROR_PAYLOAD_CAPACITY; return -1; }
 if(heap->counts[0]+1>heap->max_nodes){ heap->error=ERROR_NODE_CAPACITY; return -1; }
 int offset=heap->counts[2];
 for(int i=0;i<len;i++) heap->payload[offset+i]=raw[i];
 heap->counts[2]+=len;
 int ref=heap->counts[0]++;
 heap->nodes[ref*4+0]=TAG_INT; heap->nodes[ref*4+1]=offset;
 heap->nodes[ref*4+2]=len; heap->nodes[ref*4+3]=0;
 return ref;
}
extern "C" __global__ void bind_actor_seed_args(
 int* nodes,int* edges,unsigned char* payload,int* counts,int* roots,
 const long long* actors,const long long* seeds,const unsigned char* active,
 int* arg_refs,int* errors,int rows,int maxn,int maxe,int maxp,int with_seed){
 int row=(int)(blockIdx.x*blockDim.x+threadIdx.x); if(row>=rows) return;
 int cols=with_seed?3:2;
 errors[row]=ERROR_OK; arg_refs[row*cols+0]=roots[row]; arg_refs[row*cols+1]=-1;
 if(with_seed) arg_refs[row*cols+2]=-1;
 if(!active[row]) return;
 int nN=counts[row*3+0], nE=counts[row*3+1], nP=counts[row*3+2];
 if(nN<0||nE<0||nP<0||nN>maxn||nE>maxe||nP>maxp){ errors[row]=ERROR_INVALID_REF; return; }
 if(roots[row]<0||roots[row]>=nN){errors[row]=ERROR_INVALID_REF;return;}
 long long actor=actors[row];
 if(actor!=0 && actor!=1){ errors[row]=ERROR_TYPE_MISMATCH; return; }
 Heap h; init_heap(&h,nodes+(long long)row*maxn*4,edges+(long long)row*maxe*2,
  payload+(long long)row*maxp,counts+row*3,maxn,maxe,maxp,0,0,0,0,0,0);
 const char* name=actor? "b":"a";
 int side=make_utf8(&h,name,1);
 if(side<0){ errors[row]=h.error?h.error:ERROR_NODE_CAPACITY; h.counts[0]=nN; h.counts[1]=nE; h.counts[2]=nP; return; }
 if(!with_seed){ arg_refs[row*cols+1]=side; return; }
 int seed=make_seed_int(&h,seeds+((long long)row*5));
 if(seed<0){ errors[row]=h.error?h.error:ERROR_PAYLOAD_CAPACITY; h.counts[0]=nN; h.counts[1]=nE; h.counts[2]=nP; return; }
 arg_refs[row*cols+1]=side; arg_refs[row*cols+2]=seed;
}
extern "C" __global__ void bind_actor_action_args(
 int* nodes,int* edges,unsigned char* payload,int* counts,int* roots,
 const int* an,const int* ae,const unsigned char* ap,const int* ac,const int* ar,
 const long long* actors,const unsigned char* active,int* refs,int* errors,
 int rows,int maxn,int maxe,int maxp,int amn,int ame,int amp){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
 errors[row]=0;refs[row*3]=roots[row];refs[row*3+1]=-1;refs[row*3+2]=-1;
 if(!active[row])return;
 int n=counts[row*3],e=counts[row*3+1],p=counts[row*3+2];
 int na=ac[row*3],ea=ac[row*3+1],pa=ac[row*3+2],root=ar[row];
 if(n<1||e<0||p<0||n>maxn||e>maxe||p>maxp||roots[row]<0||roots[row]>=n
     ||na<1||ea<0||pa<0||na>amn||ea>ame||pa>amp||root<0||root>=na){errors[row]=ERROR_INVALID_REF;return;}
 if(actors[row]!=0&&actors[row]!=1){errors[row]=ERROR_TYPE_MISMATCH;return;}
 if((long long)n+na+1>maxn){errors[row]=ERROR_NODE_CAPACITY;return;}
 if((long long)e+ea>maxe){errors[row]=ERROR_EDGE_CAPACITY;return;}
 if((long long)p+pa+1>maxp){errors[row]=ERROR_PAYLOAD_CAPACITY;return;}
 const int* src=an+(long long)row*amn*4;
 const int* sedges=ae+(long long)row*ame*2;
 const unsigned char* spayload=ap+(long long)row*amp;
 if(src[root*4]!=TAG_DICT){errors[row]=ERROR_TYPE_MISMATCH;return;}
 // Validate all references and offsets before any write to destination scratch.
 for(int i=0;i<na;i++){
  int tag=src[i*4],off=src[i*4+1],len=src[i*4+2];
  if(tag<0||tag>TAG_CARD_ENTITY){errors[row]=ERROR_INVALID_TAG;return;}
  if(tag==TAG_INT||tag==TAG_FLOAT||tag==TAG_STR){
   if(off<0||len<0||(long long)off+len>pa){errors[row]=ERROR_INVALID_REF;return;}
  }else if(tag>=TAG_LIST){
   if(off<0||len<0||(long long)off+len>ea){errors[row]=ERROR_INVALID_REF;return;}
   for(int j=0;j<len;j++){
    int k=sedges[(off+j)*2],v=sedges[(off+j)*2+1];
    if(k<0||k>=na||(tag>=TAG_DICT&&(v<0||v>=na))){errors[row]=ERROR_INVALID_REF;return;}
   }
  }
 }
 int* dest=nodes+(long long)row*maxn*4;
 int* dedges=edges+(long long)row*maxe*2;
 unsigned char* dpayload=payload+(long long)row*maxp;
 for(int i=0;i<pa;i++)dpayload[p+i]=spayload[i];
 for(int i=0;i<na;i++){
  int tag=src[i*4],off=src[i*4+1],len=src[i*4+2];
  int mapped=tag>=TAG_LIST?e+off:(tag>=TAG_INT&&tag<=TAG_STR?p+off:off);
  dest[(n+i)*4]=tag;dest[(n+i)*4+1]=mapped;
  dest[(n+i)*4+2]=len;dest[(n+i)*4+3]=src[i*4+3];
  if(tag>=TAG_LIST)for(int j=0;j<len;j++){
   dedges[(e+off+j)*2]=n+sedges[(off+j)*2];
   dedges[(e+off+j)*2+1]=tag>=TAG_DICT?n+sedges[(off+j)*2+1]:-1;
  }
 }
 int side=n+na;dpayload[p+pa]=actors[row]?98:97;
 dest[side*4]=TAG_STR;dest[side*4+1]=p+pa;dest[side*4+2]=1;dest[side*4+3]=0;
 counts[row*3]=n+na+1;counts[row*3+1]=e+ea;counts[row*3+2]=p+pa+1;
 refs[row*3+1]=side;refs[row*3+2]=n+root;
}
'''


def _seed_signed_bytes(limbs):
    raw = bytearray(41)
    for i in range(5):
        unsigned = int(limbs[i]) & 0xFFFFFFFFFFFFFFFF
        for bit in range(8):
            raw[i * 8 + bit] = (unsigned >> (8 * bit)) & 0xFF
    length = 40
    while length > 1 and raw[length - 1] == 0 and (raw[length - 2] & 0x80) == 0:
        length -= 1
    if raw[length - 1] & 0x80:
        raw[length] = 0
        length += 1
    return bytes(raw[:length])


class DeviceRuleArgumentBinder:
    """Bind actor/seed argument refs on device. Does not run duel rules."""
    supports_full_duel_rules = False

    def __init__(self, device='cuda', *, allow_cpu_reference=False):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != 'cuda' and not (
                self.device.type == 'cpu' and allow_cpu_reference is True):
            raise ValueError('CUDA required; CPU is an explicit numerical oracle only')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no CPU argument-binder fallback')
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self._cpu = self.device.type == 'cpu'
        self._dummy_seeds = torch.zeros((1, 5), dtype=torch.int64, device=self.device)
        self.module = None
        self.closed = False
        if not self._cpu:
            from .cuda_runtime import CudaModule
            self.module = CudaModule(
                emit_binder_source(), {'bind_actor_seed_args': _BIND_SPEC,
                                      'bind_actor_action_args': _ACTION_SPEC}, device=str(self.device))

    def close(self):
        if self.closed:
            return
        if self.module is not None:
            self.module.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _require_graph(self, graph):
        if self.closed:
            raise RuntimeError('Argument binder closed')
        if not isinstance(graph, DeviceGraph):
            raise TypeError('Device graph required')
        t = self.torch
        rows = graph.rows
        if rows < 1:
            raise ValueError('Nonempty device batch required')
        layout = graph.layout
        expected = (
            (graph.nodes, t.int32, (rows, layout.max_nodes, 4)),
            (graph.edges, t.int32, (rows, layout.max_edges, 2)),
            (graph.payload, t.uint8, (rows, layout.max_payload_bytes)),
            (graph.counts, t.int32, (rows, 3)),
            (graph.roots, t.int32, (rows,)),
        )
        for value, dtype, shape in expected:
            if (not isinstance(value, t.Tensor) or value.device != self.device
                    or value.dtype != dtype or tuple(value.shape) != shape
                    or not value.is_contiguous()):
                raise ValueError('Exact device graph buffers required')
        return rows, layout

    def _exact(self, value, dtype, shape, name):
        t = self.torch
        if (not isinstance(value, t.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != shape
                or not value.is_contiguous()):
            raise ValueError('Exact %s tensor required' % name)

    def bind_actor_device(self, graph, actors, active):
        return self._bind(graph, actors, None, active, with_seed=False)

    def bind_actor_seed_device(self, graph, actors, seeds, active):
        return self._bind(graph, actors, seeds, active, with_seed=True)

    def _bind(self, graph, actors, seeds, active, *, with_seed):
        rows, _ = self._require_graph(graph)
        t = self.torch
        dest = graph.fork()
        refs = t.empty((rows, 3 if with_seed else 2), dtype=t.int32, device=self.device)
        errors = t.zeros(rows, dtype=t.int32, device=self.device)
        return self.bind_into_device(dest, actors, active, argument_refs=refs,
                                     errors=errors, seeds=seeds)

    def bind_into_device(self, graph, actors, active, *, argument_refs, errors, seeds=None):
        """Bind actors/whole seed into reusable scratch without a heap fork.

        The caller refills these scratch lanes before reuse. Persistent leased
        states must not be passed directly. Inputs and results stay on device;
        invalid rows keep their original graph counts and receive row errors.
        """
        rows, layout = self._require_graph(graph)
        t = self.torch
        with_seed = seeds is not None
        self._exact(actors, t.int64, (rows,), 'actors')
        if with_seed:
            self._exact(seeds, t.int64, (rows, 5), 'seeds')
        self._exact(active, active.dtype if isinstance(active, t.Tensor) else t.bool, (rows,), 'active')
        if not isinstance(active, t.Tensor) or active.dtype not in (t.bool, t.uint8):
            raise ValueError('active must be contiguous bool/uint8[B]')
        active_u8 = active.view(t.uint8) if active.dtype == t.bool else active
        refs=argument_refs
        self._exact(refs,t.int32,(rows,3 if with_seed else 2),'argument refs')
        self._exact(errors,t.int32,(rows,),'argument errors')
        if any(refs.data_ptr()==getattr(graph,key).data_ptr()
               for key in ('nodes','edges','payload','counts','roots')):
            raise ValueError('Separate argument output storage required')
        if self._cpu:
            self._bind_cpu(graph, actors, seeds, active_u8, refs, errors, with_seed)
        else:
            self.module.launch(
                'bind_actor_seed_args',
                [graph.nodes, graph.edges, graph.payload, graph.counts, graph.roots,
                 actors, seeds if with_seed else self._dummy_seeds, active_u8, refs, errors,
                 rows, layout.max_nodes, layout.max_edges, layout.max_payload_bytes, int(with_seed)],
                grid=(rows + 127) // 128, block=128, capture_safe=True)
        return BoundDeviceArguments(graph, refs, errors)

    def bind_action_into_device(self, graph, actors, action_graph, active, *, argument_refs, errors):
        """Append a compact selected action graph to reusable rule lanes.

        The caller must validate the decision's owner/generation/index lease and
        compact the selected action first. No host action dict is reconstructed.
        Action/state buffers must be distinct. This preserves every action field
        and alias and fails before writes for a malformed or overflowing row.
        """
        rows,layout=self._require_graph(graph)
        action_rows,a=self._require_graph(action_graph)
        if self._cpu:raise RuntimeError('Actual CUDA required for action graph binding')
        if rows!=action_rows:raise ValueError('Matching action graph rows required')
        t=self.torch
        self._exact(actors,t.int64,(rows,),'actors')
        self._exact(active,t.uint8,(rows,),'active')
        self._exact(argument_refs,t.int32,(rows,3),'action argument refs')
        self._exact(errors,t.int32,(rows,),'action argument errors')
        if any(getattr(graph,key).untyped_storage().data_ptr()==getattr(action_graph,key).untyped_storage().data_ptr()
               for key in ('nodes','edges','payload','counts','roots')):
            raise ValueError('Separate action graph storage required')
        self.module.launch('bind_actor_action_args',[
            graph.nodes,graph.edges,graph.payload,graph.counts,graph.roots,
            action_graph.nodes,action_graph.edges,action_graph.payload,action_graph.counts,action_graph.roots,
            actors,active,argument_refs,errors,rows,
            layout.max_nodes,layout.max_edges,layout.max_payload_bytes,
            a.max_nodes,a.max_edges,a.max_payload_bytes],
            grid=(rows+127)//128,block=128,capture_safe=True)
        return BoundDeviceArguments(graph,argument_refs,errors)

    def _bind_cpu(self, dest, actors, seeds, active, refs, errors, with_seed):
        layout = dest.layout
        for row in range(dest.rows):
            refs[row, 0] = dest.roots[row]
            refs[row, 1:] = -1
            errors[row] = ERROR_OK
            if int(active[row]) == 0:
                continue
            n_nodes, n_edges, n_payload = (int(dest.counts[row, i]) for i in range(3))
            if min(n_nodes, n_edges, n_payload) < 0 or n_nodes > layout.max_nodes or n_edges > layout.max_edges or n_payload > layout.max_payload_bytes:
                errors[row] = ERROR_INVALID_REF
                continue
            if not 0 <= int(dest.roots[row]) < n_nodes:
                errors[row] = ERROR_INVALID_REF
                continue
            actor = int(actors[row])
            if actor not in (0, 1):
                errors[row] = ERROR_TYPE_MISMATCH
                continue
            raw_seed = _seed_signed_bytes(seeds[row]) if with_seed else b''
            extra_nodes = 1 + int(with_seed)
            extra_payload = 1 + len(raw_seed)
            if n_nodes + extra_nodes > layout.max_nodes:
                errors[row] = ERROR_NODE_CAPACITY
                continue
            if n_payload + extra_payload > layout.max_payload_bytes:
                errors[row] = ERROR_PAYLOAD_CAPACITY
                continue
            dest.payload[row, n_payload] = 98 if actor else 97
            dest.nodes[row, n_nodes] = self.torch.tensor([TAG_STR, n_payload, 1, 0], dtype=self.torch.int32)
            refs[row, 1] = n_nodes
            n_nodes += 1
            n_payload += 1
            if with_seed:
                dest.payload[row, n_payload:n_payload + len(raw_seed)] = self.torch.tensor(list(raw_seed), dtype=self.torch.uint8)
                dest.nodes[row, n_nodes] = self.torch.tensor([TAG_INT, n_payload, len(raw_seed), 0], dtype=self.torch.int32)
                refs[row, 2] = n_nodes
                n_nodes += 1
                n_payload += len(raw_seed)
            dest.counts[row, 0] = n_nodes
            dest.counts[row, 2] = n_payload
