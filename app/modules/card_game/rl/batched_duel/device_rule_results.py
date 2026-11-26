"""Decode frozen VM result references on CUDA; not a rule engine or approval.

This decoder consumes actual acting_side/finished outputs and the same graph.
It does not choose an actor, reconstruct a host state, or execute Python rules.
All row errors remain device tensors until the whole operation completes.
"""
from dataclasses import dataclass

from .cuda_vm import CudaVm
from .device_rule_protocol import DeviceMeta
from .vm_heap import emit_vm_heap_source


_RESULT_SOURCE = r'''
using namespace duel_vm_heap;
const int RESULT_INVALID=150;
__device__ inline int literal_equal(Heap* h,int ref,const char* s,int len){
 NodeView n;if(!get_node(h,ref,&n))return 0;
 if(n.tag!=TAG_STR||n.count!=len)return 0;
 const unsigned char* p=get_payload_ptr(h,&n);if(!p)return 0;
 return vm_memcmp(p,s,len)==0;
}
__device__ inline int field_ref(Heap* h,int root,const char* s,int len){
 NodeView n;if(!get_node(h,root,&n))return -2147483647-1;
 if(n.tag!=TAG_DICT){h->error=ERROR_TYPE_MISMATCH;return -2147483647-1;}
 for(int i=0;i<n.count;i++){
  int k=0,v=0;if(!get_edge(h,&n,i,&k,&v))return -2147483647-1;
  if(literal_equal(h,k,s,len))return v;
 }
 h->error=ERROR_INVALID_REF;return -2147483647-1;
}
__device__ inline int integer64(Heap* h,int ref,long long* result){
 NodeView n;if(!get_node(h,ref,&n)||n.tag!=TAG_INT||n.count<1)return 0;
 const unsigned char* p=get_payload_ptr(h,&n);if(!p)return 0;
 int negative=(p[n.count-1]&128)!=0;
 unsigned long long bits=negative?~0ULL:0ULL;
 for(int i=0;i<n.count&&i<8;i++)bits=(bits&~(255ULL<<(8*i)))|((unsigned long long)p[i]<<(8*i));
 if(((bits>>63)!=0)!=negative)return 0;
 for(int i=8;i<n.count;i++)if(p[i]!=(negative?255:0))return 0;
 *result=(long long)bits;return 1;
}
__device__ inline int side_code(Heap* h,int ref){
 NodeView n;if(!get_node(h,ref,&n))return -3;
 if(n.tag==TAG_NONE)return -1;
 if(literal_equal(h,ref,"a",1))return 0;
 if(literal_equal(h,ref,"b",1))return 1;
 return -3;
}
extern "C" __global__ void decode_metadata(
 int* nodes,int* edges,unsigned char* payload,int* counts,const int* roots,
 const int* cn,const int* ce,const unsigned char* cp,int cnc,int cec,int cpc,
 const int* actor_refs,const int* finished_refs,const int* input_errors,const unsigned char* active,
 long long* actors,int* phases,long long* winners,unsigned char* finished,long long* versions,int* errors,
 int rows,int maxn,int maxe,int maxp){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
 actors[row]=-1;phases[row]=-1;winners[row]=-2;finished[row]=0;versions[row]=0;errors[row]=0;
 if(!active[row])return;
 if(input_errors[row]){errors[row]=input_errors[row];return;}
 const int* c=counts+row*3;
 if(c[0]<1||c[0]>maxn||c[1]<0||c[1]>maxe||c[2]<0||c[2]>maxp){errors[row]=RESULT_INVALID;return;}
 Heap h;init_heap(&h,nodes+(long long)row*maxn*4,edges+(long long)row*maxe*2,
  payload+(long long)row*maxp,counts+row*3,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);
 int actor=side_code(&h,actor_refs[row]);
 NodeView is_finished;
 if(actor<-1||!get_node(&h,finished_refs[row],&is_finished)||is_finished.tag!=TAG_BOOL
     ||(is_finished.extra!=0&&is_finished.extra!=1)){errors[row]=RESULT_INVALID;return;}
 int phase_ref=field_ref(&h,roots[row],"phase",5),phase=-1;
 if(literal_equal(&h,phase_ref,"mulligan",8))phase=0;
 else if(literal_equal(&h,phase_ref,"playing",7))phase=1;
 else if(literal_equal(&h,phase_ref,"choice",6))phase=2;
 else if(literal_equal(&h,phase_ref,"finished",8))phase=3;
 long long version=0;
 int version_ref=field_ref(&h,roots[row],"version",7);
 int winner_ref=field_ref(&h,roots[row],"winner",6);
 int winner=side_code(&h,winner_ref);
 if(winner==-3&&literal_equal(&h,winner_ref,"draw",4))winner=-1;
 if(h.error||phase<0||!integer64(&h,version_ref,&version)||version<0||winner<-1
     ||(phase==3)!=(is_finished.extra!=0)||((phase==3)&&actor!=-1)
     ||((phase!=3)&&winner!=-1)){errors[row]=RESULT_INVALID;return;}
 actors[row]=actor;phases[row]=phase;winners[row]=phase==3?winner:-2;
 finished[row]=(unsigned char)is_finished.extra;versions[row]=version;
}
extern "C" __global__ void decode_pair(
 int* nodes,int* edges,unsigned char* payload,int* counts,const int* cn,const int* ce,const unsigned char* cp,
 int cnc,int cec,int cpc,const int* outputs,const int* input_errors,const unsigned char* active,
 int* first,int* second,int* errors,int rows,int maxn,int maxe,int maxp){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
 first[row]=-2147483647-1;second[row]=-2147483647-1;errors[row]=input_errors[row];
 if(!active[row]||errors[row])return;
 int* c=counts+row*3;
 if(c[0]<1||c[0]>maxn||c[1]<0||c[1]>maxe||c[2]<0||c[2]>maxp){errors[row]=RESULT_INVALID;return;}
 Heap h;init_heap(&h,nodes+(long long)row*maxn*4,edges+(long long)row*maxe*2,
  payload+(long long)row*maxp,c,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);
 NodeView n;int dummy=0,a=0,b=0;
 if(!get_node(&h,outputs[row],&n)||n.tag!=TAG_TUPLE||n.count!=2
     ||!get_edge(&h,&n,0,&a,&dummy)||!get_edge(&h,&n,1,&b,&dummy)){
  errors[row]=RESULT_INVALID;return;
 }
 NodeView child;
 if(!get_node(&h,a,&child)||!get_node(&h,b,&child)){errors[row]=RESULT_INVALID;return;}
 first[row]=a;second[row]=b;
}
extern "C" __global__ void decode_float(
 int* nodes,int* edges,unsigned char* payload,int* counts,const int* cn,const int* ce,const unsigned char* cp,
 int cnc,int cec,int cpc,const int* refs,const int* input_errors,const unsigned char* active,
 double* values,int* errors,int rows,int maxn,int maxe,int maxp){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
 values[row]=0;errors[row]=input_errors[row];if(!active[row]||errors[row])return;
 int* c=counts+row*3;
 if(c[0]<1||c[0]>maxn||c[1]<0||c[1]>maxe||c[2]<0||c[2]>maxp){errors[row]=RESULT_INVALID;return;}
 Heap h;init_heap(&h,nodes+(long long)row*maxn*4,edges+(long long)row*maxe*2,
  payload+(long long)row*maxp,c,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);
 double value=0;
 if(read_float64(&h,refs[row],&value)||!vm_isfinite(value)){errors[row]=RESULT_INVALID;return;}
 values[row]=value;
}
'''

_SPEC = ['int32*','int32*','uint8*','int32*','int32*',
         'int32*','int32*','uint8*','int32','int32','int32',
         'int32*','int32*','int32*','uint8*',
         'int64*','int32*','int64*','uint8*','int64*','int32*',
         'int32','int32','int32','int32']
_VALUE_PREFIX=['int32*','int32*','uint8*','int32*','int32*','int32*','uint8*',
               'int32','int32','int32','int32*','int32*','uint8*']


@dataclass(frozen=True)
class DevicePairReferences:
    reader:object
    first:object
    second:object
    errors:object


@dataclass(frozen=True)
class DeviceFloatResult:
    reader:object
    values:object
    errors:object


@dataclass(frozen=True)
class DeviceMetadataWorkspace:
    reader: object
    meta: DeviceMeta


class DeviceRuleResultReader:
    supports_full_duel_rules = False
    phase_codes = ('mulligan','playing','choice','finished')

    def __init__(self, vm):
        if not isinstance(vm,CudaVm) or vm.closed:
            raise TypeError('Live frozen CUDA VM required')
        if vm.device.type!='cuda':
            raise RuntimeError('Actual CUDA required; no rule-result fallback')
        from .cuda_runtime import CudaModule
        self.vm=vm;self.t=vm.torch;self.device=vm.device;self.closed=False
        self.module=CudaModule(emit_vm_heap_source()+_RESULT_SOURCE,
            {'decode_metadata':_SPEC,
             'decode_pair':_VALUE_PREFIX+['int32*','int32*','int32*']+['int32']*4,
             'decode_float':_VALUE_PREFIX+['float64*','int32*']+['int32']*4},device=str(self.device))

    def close(self):
        if not self.closed:
            self.module.close();self.closed=True

    def _prepare_rows(self,rows):
        if self.closed:raise RuntimeError('Rule result reader closed')
        if type(rows) is not int or not 0<rows<2**31:
            raise ValueError('Positive bounded row count required')
        if self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Allocate result outputs before capture')

    def prepare_metadata_workspace(self,rows):
        self._prepare_rows(rows)
        t=self.t
        def empty(dtype):return t.empty(rows,dtype=dtype,device=self.device)
        return DeviceMetadataWorkspace(self,DeviceMeta(
            empty(t.int64),empty(t.int32),empty(t.int64),empty(t.bool),empty(t.int64),empty(t.int32)))

    def prepare_pair_workspace(self,rows):
        self._prepare_rows(rows)
        t=self.t
        return DevicePairReferences(self,t.empty(rows,dtype=t.int32,device=self.device),
            t.empty(rows,dtype=t.int32,device=self.device),t.empty(rows,dtype=t.int32,device=self.device))

    def prepare_float_workspace(self,rows):
        self._prepare_rows(rows)
        return DeviceFloatResult(self,self.t.empty(rows,dtype=self.t.float64,device=self.device),
            self.t.empty(rows,dtype=self.t.int32,device=self.device))

    def _result_vectors(self,arena,refs,input_errors,active,outputs):
        if self.closed or self.vm.closed:raise RuntimeError('Rule result reader or VM closed')
        self.vm._validate_device_graph(arena)
        t=self.t;rows=arena.rows
        for value,dtype in ((refs,t.int32),(input_errors,t.int32),(active,t.uint8),*outputs):
            if (not isinstance(value,t.Tensor) or value.device!=self.device or value.dtype!=dtype
                    or value.shape!=(rows,) or not value.is_contiguous()):
                raise ValueError('Exact device result vectors required')
        b=self.vm._buffers;l=arena.layout
        return [arena.nodes,arena.edges,arena.payload,arena.counts,b['nodes'],b['edges'],b['payload'],
                len(self.vm.image.nodes),len(self.vm.image.edges),len(self.vm.image.payload),refs,input_errors,active],\
               [rows,l.max_nodes,l.max_edges,l.max_payload_bytes]

    def pair_into_device(self,call,*,active,output):
        if call.executor is not self.vm or not isinstance(output,DevicePairReferences) or output.reader is not self:
            raise ValueError('Pair outputs must bind this frozen VM')
        t=self.t
        args,limits=self._result_vectors(call.arena,call.outputs,call.statuses[:,0].contiguous(),active,
            ((output.first,t.int32),(output.second,t.int32),(output.errors,t.int32)))
        self.module.launch('decode_pair',args+[output.first,output.second,output.errors]+limits,
                           grid=(call.arena.rows+127)//128,block=128,capture_safe=True)
        return output

    def float_into_device(self,arena,refs,*,input_errors,active,output):
        if not isinstance(output,DeviceFloatResult) or output.reader is not self:
            raise ValueError('Scalar outputs must bind this frozen VM')
        args,limits=self._result_vectors(arena,refs,input_errors,active,
            ((output.values,self.t.float64),(output.errors,self.t.int32)))
        self.module.launch('decode_float',args+[output.values,output.errors]+limits,
                           grid=(arena.rows+127)//128,block=128,capture_safe=True)
        return output

    def metadata_into_device(self,arena,*,actor_refs,finished_refs,input_errors,active,output):
        """Decode actual formal output refs; source graphs remain unchanged.

        These references must come from this VM's acting_side and finished calls
        on the same live graph. Consumers bind the VM/source/generation identity
        and retain both outputs before reusing VM scratch.
        """
        if self.closed or self.vm.closed:raise RuntimeError('Rule result reader or VM closed')
        self.vm._validate_device_graph(arena)
        if not isinstance(output,DeviceMetadataWorkspace) or output.reader is not self:
            raise ValueError('Metadata workspace must belong to this reader')
        t=self.t;rows=arena.rows;meta=output.meta
        vectors=((actor_refs,t.int32),(finished_refs,t.int32),(input_errors,t.int32),(active,t.uint8),
                 (meta.actors,t.int64),(meta.phases,t.int32),(meta.winners,t.int64),
                 (meta.finished,t.bool),(meta.versions,t.int64),(meta.errors,t.int32))
        for value,dtype in vectors:
            if (not isinstance(value,t.Tensor) or value.device!=self.device or value.dtype!=dtype
                    or value.shape!=(rows,) or not value.is_contiguous()):
                raise ValueError('Exact device metadata vectors required')
        b=self.vm._buffers;l=arena.layout
        self.module.launch('decode_metadata',[
            arena.nodes,arena.edges,arena.payload,arena.counts,arena.roots,
            b['nodes'],b['edges'],b['payload'],len(self.vm.image.nodes),len(self.vm.image.edges),len(self.vm.image.payload),
            actor_refs,finished_refs,input_errors,active,meta.actors,meta.phases,meta.winners,
            meta.finished.view(t.uint8),meta.versions,meta.errors,
            rows,l.max_nodes,l.max_edges,l.max_payload_bytes],
            grid=(rows+127)//128,block=128,capture_safe=True)
        return meta
