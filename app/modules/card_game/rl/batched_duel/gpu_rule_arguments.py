"""Bounded CUDA construction of VM entry arguments from device graphs.

Merges only explicit CPU argument literals into an immutable device state graph.
Never exports state to compute a rule result. Status/capacity failures are fatal.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .codec import pack_states
from .cuda_vm import DeviceGraph
from .schema import GraphLayout, PackedStateBatch, TAG_BOOL, TAG_DICT, TAG_FLOAT, TAG_INT, TAG_LIST, TAG_NONE, TAG_STR, TAG_TUPLE
from .vm_heap import emit_vm_heap_source


class GraphArgumentError(RuntimeError):
    """Fail-closed error for GPU graph argument construction."""


STATUS_OK = 0
STATUS_NODE_CAPACITY = 1
STATUS_EDGE_CAPACITY = 2
STATUS_PAYLOAD_CAPACITY = 3
STATUS_BAD_ARITY = 4
STATUS_INVALID_TAG = 5
STATUS_INVALID_REF = 6
STATUS_TYPE_MISMATCH = 7

_STATUS_NAMES = {
    STATUS_OK: 'ok',
    STATUS_NODE_CAPACITY: 'node_capacity',
    STATUS_EDGE_CAPACITY: 'edge_capacity',
    STATUS_PAYLOAD_CAPACITY: 'payload_capacity',
    STATUS_BAD_ARITY: 'bad_arity',
    STATUS_INVALID_TAG: 'invalid_tag',
    STATUS_INVALID_REF: 'invalid_ref',
    STATUS_TYPE_MISMATCH: 'type_mismatch',
}

_APPEND_SPEC = (
    'int32*', 'int32*', 'uint8*', 'int32*', 'int32*',
    'int32*', 'int32*', 'uint8*', 'int32*', 'int32*',
    'int32*', 'int32*', 'uint8*', 'int32*', 'int32*',
    'int32*', 'int32*',
    'int32', 'int32', 'int32', 'int32', 'int32', 'int32', 'int32', 'int32',
)
_META_SPEC = (
    'int32*', 'int32*', 'uint8*', 'int32*', 'int32*',
    'uint8*', 'int32*', 'int32*', 'int32',
    'int32*', 'int32*', 'int32*',
    'int32', 'int32', 'int32', 'int32',
)


def emit_argument_source():
    return emit_vm_heap_source() + _ARGUMENT_SOURCE


_ARGUMENT_SOURCE = r'''
using namespace duel_vm_heap;
enum ArgStatus {
  ARG_OK=0, ARG_NODE=1, ARG_EDGE=2, ARG_PAYLOAD=3, ARG_ARITY=4, ARG_TAG=5, ARG_REF=6, ARG_TYPE=7
};
__device__ inline int remap_ref(int ref,int base,int count){
 if(ref<0)return ref;
 if(ref>=count)return -2147483647-1;
 return base+ref;
}
__device__ inline int intern_eq(Heap* h,int ref,const unsigned char* names,const int* off,const int* len,int ident){
 NodeView node; if(!get_node(h,ref,&node) || node.tag!=TAG_STR) return 0;
 if(node.count!=len[ident]) return 0;
 const unsigned char* p=get_payload_ptr(h,&node); if(!p) return 0;
 return vm_memcmp(p,names+off[ident],len[ident])==0;
}
__device__ inline int dict_get_intern(Heap* h,int dict,const unsigned char* names,const int* off,const int* len,int ident){
 NodeView node; if(!get_node(h,dict,&node)) return -1;
 if(node.tag!=TAG_DICT && node.tag<TAG_PLAYER_ENTITY) return -1;
 for(int i=0;i<node.count;i++){
  int k=0,v=0; if(!get_edge(h,&node,i,&k,&v)) return -1;
  if(intern_eq(h,k,names,off,len,ident)) return v;
 }
 return -1;
}
extern "C" __global__ void graph_append_arguments(
 const int* sn,const int* se,const unsigned char* sp,const int* sc,const int* sr,
 const int* an,const int* ae,const unsigned char* ap,const int* ac,const int* ar,
 int* dn,int* de,unsigned char* dp,int* dc,int* dr,
 int* arg_refs,int* status,
 int rows,int maxn,int maxe,int maxp,int extra_nargs,int arg_maxn,int arg_maxe,int arg_maxp){
 int row=(int)(blockIdx.x*blockDim.x+threadIdx.x); if(row>=rows) return;
 const int* scn=sc+row*3; int nN=scn[0],nE=scn[1],nP=scn[2];
 if(nN<0||nE<0||nP<0||nN>maxn||nE>maxe||nP>maxp){status[row]=ARG_REF;return;}
 long long nbase=(long long)row*maxn*4, ebase=(long long)row*maxe*2, pbase=(long long)row*maxp;
 for(int i=0;i<nN*4;i++) dn[nbase+i]=sn[nbase+i];
 for(int i=0;i<nE*2;i++) de[ebase+i]=se[ebase+i];
 for(int i=0;i<nP;i++) dp[pbase+i]=sp[pbase+i];
 dc[row*3+0]=nN; dc[row*3+1]=nE; dc[row*3+2]=nP; dr[row]=sr[row];
 int cols=1+extra_nargs; arg_refs[row*cols+0]=sr[row];
 for(int i=1;i<cols;i++) arg_refs[row*cols+i]=-1;
 if(extra_nargs==0){status[row]=ARG_OK;return;}
 const int* acn=ac+row*3; int aN=acn[0],aE=acn[1],aP=acn[2];
 if(aN<1||aE<0||aP<0||aN>arg_maxn||aE>arg_maxe||aP>arg_maxp){status[row]=ARG_REF;return;}
 if(nN+aN>maxn){status[row]=ARG_NODE;return;}
 if(nE+aE>maxe){status[row]=ARG_EDGE;return;}
 if(nP+aP>maxp){status[row]=ARG_PAYLOAD;return;}
 long long abase_n=(long long)row*arg_maxn*4, abase_e=(long long)row*arg_maxe*2, abase_p=(long long)row*arg_maxp;
 int outN=nN,outE=nE,outP=nP;
 for(int i=0;i<aN;i++){
  int tag=an[abase_n+i*4+0], off=an[abase_n+i*4+1], cnt=an[abase_n+i*4+2], extra=an[abase_n+i*4+3];
  int dest=outN+i;
  if(tag==TAG_INT||tag==TAG_FLOAT||tag==TAG_STR){
   if(off<0||cnt<0||off+cnt>aP){status[row]=ARG_REF;return;}
   int dest_off=outP;
   for(int b=0;b<cnt;b++) dp[pbase+dest_off+b]=ap[abase_p+off+b];
   outP+=cnt;
   dn[nbase+dest*4+0]=tag; dn[nbase+dest*4+1]=dest_off; dn[nbase+dest*4+2]=cnt; dn[nbase+dest*4+3]=extra;
  }else if(tag==TAG_LIST||tag==TAG_TUPLE||tag==TAG_DICT||tag>=TAG_PLAYER_ENTITY){
   if(off<0||cnt<0||off+cnt>aE){status[row]=ARG_REF;return;}
   int dest_off=outE;
   for(int e=0;e<cnt;e++){
    int k=ae[abase_e+(off+e)*2+0], v=ae[abase_e+(off+e)*2+1];
    int rk=remap_ref(k,nN,aN), rv=remap_ref(v,nN,aN);
    if((k>=0&&rk<0)||(v>=0&&rv<0)){status[row]=ARG_REF;return;}
    de[ebase+(dest_off+e)*2+0]=rk; de[ebase+(dest_off+e)*2+1]=rv;
   }
   outE+=cnt;
   dn[nbase+dest*4+0]=tag; dn[nbase+dest*4+1]=dest_off; dn[nbase+dest*4+2]=cnt; dn[nbase+dest*4+3]=extra;
  }else if(tag==TAG_NONE||tag==TAG_BOOL){
   dn[nbase+dest*4+0]=tag; dn[nbase+dest*4+1]=0; dn[nbase+dest*4+2]=0; dn[nbase+dest*4+3]=extra;
  }else{status[row]=ARG_TAG;return;}
 }
 outN+=aN;
 dc[row*3+0]=outN; dc[row*3+1]=outE; dc[row*3+2]=outP;
 int root=ar[row];
 if(root<0||root>=aN){status[row]=ARG_REF;return;}
 int mapped=nN+root;
 int tag=dn[nbase+mapped*4+0], off=dn[nbase+mapped*4+1], cnt=dn[nbase+mapped*4+2];
 if(tag!=TAG_TUPLE||cnt!=extra_nargs){status[row]=ARG_ARITY;return;}
 for(int j=0;j<extra_nargs;j++){
  int child=de[ebase+(off+j)*2+0];
  if(child<-1){status[row]=ARG_REF;return;}
  arg_refs[row*cols+1+j]=child;
 }
 status[row]=ARG_OK;
}
extern "C" __global__ void graph_read_meta(
 int* nodes,int* edges,unsigned char* payload,int* counts,int* roots,
 const unsigned char* names,const int* off,const int* len,int n_names,
 int* phase_code,int* winner_code,int* status,
 int rows,int maxn,int maxe,int maxp){
 int row=(int)(blockIdx.x*blockDim.x+threadIdx.x); if(row>=rows) return;
 Heap h; init_heap(&h,nodes+(long long)row*maxn*4,edges+(long long)row*maxe*2,payload+(long long)row*maxp,
  counts+row*3,maxn,maxe,maxp,0,0,0,0,0,0);
 int root=roots[row];
 phase_code[row]=-1; winner_code[row]=-1;
 if(n_names<9){status[row]=ARG_REF;return;}
 int phase=dict_get_intern(&h,root,names,off,len,0);
 int winner=dict_get_intern(&h,root,names,off,len,1);
 if(h.error){status[row]=ARG_TYPE;return;}
 if(phase>=0){
  if(intern_eq(&h,phase,names,off,len,2)) phase_code[row]=0;
  else if(intern_eq(&h,phase,names,off,len,3)) phase_code[row]=1;
  else if(intern_eq(&h,phase,names,off,len,4)) phase_code[row]=2;
  else if(intern_eq(&h,phase,names,off,len,5)) phase_code[row]=3;
  else {status[row]=ARG_TAG;return;}
 }else{status[row]=ARG_TYPE;return;}
 if(winner<0){winner_code[row]=-1;}
 else{
  NodeView node; if(!get_node(&h,winner,&node)){status[row]=ARG_REF;return;}
  if(node.tag==TAG_NONE) winner_code[row]=-1;
  else if(intern_eq(&h,winner,names,off,len,6)) winner_code[row]=0;
  else if(intern_eq(&h,winner,names,off,len,7)) winner_code[row]=1;
  else if(intern_eq(&h,winner,names,off,len,8)) winner_code[row]=2;
  else {status[row]=ARG_TAG;return;}
 }
 status[row]=ARG_OK;
}
'''

META_NAMES = ('phase', 'winner', 'mulligan', 'playing', 'choice', 'finished', 'a', 'b', 'draw')
PHASE_BY_CODE = {0: 'mulligan', 1: 'playing', 2: 'choice', 3: 'finished'}
WINNER_BY_CODE = {-1: None, 0: 'a', 1: 'b', 2: 'draw'}


def _intern_table(names):
    payload = bytearray(); offsets = []; lengths = []
    for name in names:
        raw = name.encode('utf-8')
        offsets.append(len(payload)); lengths.append(len(raw)); payload.extend(raw)
    return (
        np.frombuffer(bytes(payload), dtype=np.uint8).copy(),
        np.asarray(offsets, dtype=np.int32),
        np.asarray(lengths, dtype=np.int32),
    )


@dataclass(frozen=True)
class BoundArguments:
    arena: DeviceGraph
    argument_refs: object
    extra_nargs: int


class GraphArgumentKernel:
    """Compiled GPU graph argument / metadata kernels. CUDA-only."""

    def __init__(self, device='cuda'):
        import torch
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required for graph argument kernels; no CPU fallback')
        from .cuda_runtime import CudaModule
        self.torch = torch
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.module = CudaModule(
            emit_argument_source(),
            {'graph_append_arguments': _APPEND_SPEC, 'graph_read_meta': _META_SPEC},
            device=str(self.device),
        )
        payload, offsets, lengths = _intern_table(META_NAMES)
        self._names = torch.tensor(payload, dtype=torch.uint8, device=self.device)
        self._off = torch.tensor(offsets, dtype=torch.int32, device=self.device)
        self._len = torch.tensor(lengths, dtype=torch.int32, device=self.device)
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.module.close(); self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def bind(self, source, extras):
        """Copy immutable source rows and append packed CPU argument tuples on GPU."""
        if self.closed:
            raise RuntimeError('Graph argument kernel closed')
        if not isinstance(source, DeviceGraph):
            raise TypeError('Device graph required')
        rows = source.rows
        if rows < 1:
            raise ValueError('Nonempty device batch required')
        if extras is None:
            extras = [() for _ in range(rows)]
        extras = list(extras)
        if len(extras) != rows:
            raise ValueError('Argument row count must match the device graph')
        extra_nargs = len(extras[0]) if extras else 0
        if any(not isinstance(row, tuple) or len(row) != extra_nargs for row in extras):
            raise ValueError('Each row must supply the same explicit argument tuple')
        torch = self.torch
        dest = DeviceGraph(
            source.layout,
            torch.zeros_like(source.nodes),
            torch.zeros_like(source.edges),
            torch.zeros_like(source.payload),
            torch.zeros_like(source.counts),
            torch.zeros_like(source.roots),
        )
        cols = 1 + extra_nargs
        arg_refs = torch.empty((rows, cols), dtype=torch.int32, device=self.device)
        status = torch.empty(rows, dtype=torch.int32, device=self.device)
        if extra_nargs == 0:
            arg_layout = GraphLayout(rows, 1, 1, 1)
            packed = pack_states([() for _ in range(rows)], arg_layout)
        else:
            arg_layout = GraphLayout(rows, 4096, 8192, 65536)
            packed = pack_states(extras, arg_layout)
        args = DeviceGraph(
            packed.layout,
            *(torch.tensor(np.array(getattr(packed, name), copy=True), device=self.device)
              for name in ('nodes', 'edges', 'payload', 'counts', 'roots')),
        )
        event = self.module.launch(
            'graph_append_arguments',
            [source.nodes, source.edges, source.payload, source.counts, source.roots,
             args.nodes, args.edges, args.payload, args.counts, args.roots,
             dest.nodes, dest.edges, dest.payload, dest.counts, dest.roots,
             arg_refs, status,
             rows, source.layout.max_nodes, source.layout.max_edges, source.layout.max_payload_bytes,
             extra_nargs, args.layout.max_nodes, args.layout.max_edges, args.layout.max_payload_bytes],
            grid=(rows + 127) // 128, block=128,
        )
        event.synchronize()
        codes = status.cpu().numpy()
        bad = np.flatnonzero(codes)
        if len(bad):
            code = int(codes[int(bad[0])])
            raise GraphArgumentError(
                'GPU graph argument construction failed row=%s status=%s'
                % (int(bad[0]), _STATUS_NAMES.get(code, str(code)))
            )
        if not arg_refs.is_contiguous():
            arg_refs = arg_refs.contiguous()
        return BoundArguments(dest, arg_refs, extra_nargs)

    def read_meta(self, arena):
        """Read phase/winner from device state dicts. Not a formal rule entry."""
        if self.closed:
            raise RuntimeError('Graph argument kernel closed')
        if not isinstance(arena, DeviceGraph):
            raise TypeError('Device graph required')
        rows = arena.rows
        if rows < 1:
            raise ValueError('Nonempty device batch required')
        torch = self.torch
        phase = torch.empty(rows, dtype=torch.int32, device=self.device)
        winner = torch.empty(rows, dtype=torch.int32, device=self.device)
        status = torch.empty(rows, dtype=torch.int32, device=self.device)
        event = self.module.launch(
            'graph_read_meta',
            [arena.nodes, arena.edges, arena.payload, arena.counts, arena.roots,
             self._names, self._off, self._len, len(META_NAMES),
             phase, winner, status,
             rows, arena.layout.max_nodes, arena.layout.max_edges, arena.layout.max_payload_bytes],
            grid=(rows + 127) // 128, block=128,
        )
        event.synchronize()
        codes = status.cpu().numpy()
        bad = np.flatnonzero(codes)
        if len(bad):
            code = int(codes[int(bad[0])])
            raise GraphArgumentError(
                'GPU state metadata read failed row=%s status=%s'
                % (int(bad[0]), _STATUS_NAMES.get(code, str(code)))
            )
        phase_np = phase.cpu().numpy(); winner_np = winner.cpu().numpy()
        return (
            [PHASE_BY_CODE[int(code)] for code in phase_np],
            [WINNER_BY_CODE[int(code)] for code in winner_np],
        )


def pack_argument_tuples(rows, layout=None):
    """Host packing of explicit CPU argument tuples. Ingress only, not a rule result."""
    values = [tuple(row) for row in rows]
    if layout is None:
        layout = GraphLayout(max(1, len(values)), 4096, 8192, 65536)
    return pack_states(values, layout)
