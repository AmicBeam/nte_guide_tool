"""Frozen three-step SoA teacher and internal raw teacher action decoding.

This helper does not supply public policy observations, change legal ordering,
or qualify the rule/search backend. Prepare its decoder before graph capture.
"""
from ..batched_duel.device_rule_protocol import DeviceDecision
from ..batched_duel.vm_heap import emit_vm_heap_source


TEACHER_HORIZON=3
RAW_TEACHER_CAPACITY=801
TEACHER_INDEX_ERROR=154
INVALID_LEGAL=160

_SOURCE=emit_vm_heap_source()+r'''
using namespace duel_vm_heap;
__device__ inline int soa_teacher_integer64(Heap* h,int ref,long long* result){
 NodeView n;if(!get_node(h,ref,&n)||n.tag!=TAG_INT||n.count<1)return 0;
 const unsigned char* p=get_payload_ptr(h,&n);if(!p)return 0;
 int negative=(p[n.count-1]&128)!=0;
 unsigned long long bits=negative?~0ULL:0ULL;
 for(int i=0;i<n.count&&i<8;i++)bits=(bits&~(255ULL<<(8*i)))|((unsigned long long)p[i]<<(8*i));
 if(((bits>>63)!=0)!=negative)return 0;
 for(int i=8;i<n.count;i++)if(p[i]!=(negative?255:0))return 0;
 *result=(long long)bits;return 1;
}
extern "C" __global__ void soa_teacher_index(
 int* nodes,int* edges,unsigned char* payload,int* counts,const int* cn,const int* ce,const unsigned char* cp,
 int cnc,int cec,int cpc,const int* refs,const int* input_errors,const unsigned char* active,
 long long* values,int* errors,int rows,int maxn,int maxe,int maxp){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
 values[row]=-1;errors[row]=input_errors[row];if(!active[row]||errors[row])return;
 int* c=counts+row*3;
 if(c[0]<1||c[0]>maxn||c[1]<0||c[1]>maxe||c[2]<0||c[2]>maxp){errors[row]=154;return;}
 Heap h;init_heap(&h,nodes+(long long)row*maxn*4,edges+(long long)row*maxe*2,
  payload+(long long)row*maxp,c,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);
 long long value=0;
 if(!soa_teacher_integer64(&h,refs[row],&value)||value<0||h.error){errors[row]=154;return;}
 values[row]=value;
}
extern "C" __global__ void soa_teacher_refs(
 int* nodes,int* edges,unsigned char* payload,int* counts,int* roots,
 const int* cn,const int* ce,const unsigned char* cp,int cnc,int cec,int cpc,
 const long long* chosen,const int* input_errors,const unsigned char* active,
 int* refs,unsigned char* valid,long long* sizes,int* errors,
 int rows,int maxn,int maxe,int maxp,int width){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
 sizes[row]=0;errors[row]=input_errors[row];
 for(int i=0;i<width;i++){refs[row*width+i]=-1;valid[row*width+i]=0;}
 if(!active[row]||errors[row])return;
 int* c=counts+row*3;
 if(c[0]<1||c[0]>maxn||c[1]<0||c[1]>maxe||c[2]<0||c[2]>maxp){errors[row]=160;return;}
 Heap h;init_heap(&h,nodes+(long long)row*maxn*4,edges+(long long)row*maxe*2,
  payload+(long long)row*maxp,c,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);
 NodeView list;
 if(!get_node(&h,roots[row],&list)||list.tag!=TAG_LIST||list.count<1||list.count>width
    ||chosen[row]<0||chosen[row]>=list.count){errors[row]=160;return;}
 for(int i=0;i<list.count;i++){
  int ref=0,unused=0;NodeView action;
  if(!get_edge(&h,&list,i,&ref,&unused)||!get_node(&h,ref,&action)
     ||action.tag!=TAG_DICT){errors[row]=160;return;}
  refs[row*width+i]=ref;valid[row*width+i]=1;
 }
 if(h.error){errors[row]=160;return;}sizes[row]=list.count;
}
'''
_INDEX_SPEC=['int32*','int32*','uint8*','int32*','int32*','int32*','uint8*',
             'int32','int32','int32','int32*','int32*','uint8*','int64*','int32*',
             'int32','int32','int32','int32']
_REFS_SPEC=['int32*','int32*','uint8*','int32*','int32*','int32*','int32*','uint8*',
            'int32','int32','int32','int64*','int32*','uint8*','int32*','uint8*',
            'int64*','int32*','int32','int32','int32','int32','int32']


def inspect_soa_teacher_dependencies(rules):
    """List precise binding gaps without claiming readiness or reading states."""
    gaps=[]
    for name in ('_begin','_invoke','_bind','_commit'):
        if not callable(getattr(rules,name,None)):gaps.append('SoA '+name+' required')
    executor=getattr(rules,'executor',None)
    if executor is None:return tuple(gaps+['SoA executor required'])
    if getattr(executor,'closed',True):gaps.append('live SoA executor required')
    if getattr(executor,'execute_only',True):gaps.append('full SoA projection module required')
    entries=getattr(getattr(executor,'image',None),'entries',{})
    functions=getattr(getattr(executor,'image',None),'function_ids',())
    for entry in ('teacher_continue','rule_decision'):
        if entry not in entries:gaps.append('frozen '+entry+' entry required')
        else:
            fid=entries[entry]
            prefix='func:app.modules.card_game.rl.batched_duel.gpu_rule_entries:'
            expected=(prefix+entry+':',prefix+'seven_'+entry+':')
            if (type(fid) is not int or not 0<=fid<len(functions)
                    or not functions[fid].startswith(expected)):
                gaps.append('canonical frozen '+entry+' binding required')
    for name in ('prepare_result_path','select_result','export_projection'):
        if not callable(getattr(executor,name,None)):gaps.append('SoA '+name+' required')
    if getattr(rules,'action_store',None) is None:gaps.append('independent action arena required')
    elif rules.action_store is getattr(rules,'store',None):gaps.append('independent action arena required')
    return tuple(gaps)


class DeviceSoATeacher:
    """Shared static decoder for direct SoA lanes, not an old VM adapter."""
    root_policy_compatible=False
    host_rule_callbacks=False
    supports_full_duel_rules=False

    def __init__(self,rules,*,max_actions=RAW_TEACHER_CAPACITY):
        gaps=inspect_soa_teacher_dependencies(rules)
        if gaps:raise ValueError('; '.join(gaps))
        if type(max_actions) is not int or not 1<=max_actions<=RAW_TEACHER_CAPACITY:
            raise ValueError('Bounded complete raw teacher action capacity required')
        t=rules.t
        if rules.device.type!='cuda' or not t.cuda.is_available():
            raise RuntimeError('Actual CUDA required; CPU teacher fallback is forbidden')
        if t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare SoA teacher decoder before CUDA capture')
        self.rules,self.t,self.device,self.rows=rules,t,rules.device,rules.rows
        self.max_actions=max_actions;self.closed=False;self._keepalive=[]
        rules.executor.prepare_result_path((0,));rules.executor.prepare_result_path((1,))
        from ..batched_duel.cuda_runtime import CudaModule
        self.module=CudaModule(_SOURCE,{'soa_teacher_index':_INDEX_SPEC,'soa_teacher_refs':_REFS_SPEC},
                               device=str(self.device))

    def close(self):
        if not self.closed:
            self.module.close();self._keepalive.clear();self.closed=True

    def _capture(self,viewers):
        if self.closed or self.rules.executor.closed:
            raise RuntimeError('SoA teacher decoder closed')
        if not self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('SoA teacher execution requires CUDA capture')
        if (not isinstance(viewers,self.t.Tensor) or viewers.device!=self.device
                or viewers.dtype!=self.t.int64 or viewers.shape!=(self.rows,) or not viewers.is_contiguous()):
            raise ValueError('Exact device int64 teacher viewer rows required')

    def teacher_continue_device(self,states,viewers,*,horizon=TEACHER_HORIZON):
        if type(horizon) is not int or horizon!=TEACHER_HORIZON:
            raise ValueError('Frozen SoA teacher_continue requires horizon=3')
        self._capture(viewers)
        r=self.rules
        batch,scratch,errors,active=r._begin(states)
        # Viewer is a value perspective, not an extra argument to this frozen
        # unary continuation. Its three canonical steps preserve chance state.
        call=r._invoke('teacher_continue',batch,scratch,scratch.roots[:,None].contiguous(),active)
        return r._commit(call,states.owners,active,errors)

    def _decode_index(self,call,errors,active):
        t=self.t;r=self.rules;arena=call.scratch;c=r.executor.constants;l=arena.layout
        values=t.full((self.rows,),-1,dtype=t.int64,device=self.device)
        output_errors=t.zeros_like(errors)
        incoming=(errors|call.statuses[:,0]).contiguous()
        args=[arena.nodes,arena.edges,arena.payload,arena.counts,
            c['nodes'],c['edges'],c['payload'],len(c['nodes']),len(c['edges']),len(c['payload']),
            call.outputs,incoming,active,values,output_errors,self.rows,l.max_nodes,l.max_edges,l.max_payload_bytes]
        self.module.launch('soa_teacher_index',args,grid=(self.rows+127)//128,block=128,capture_safe=True)
        self._keepalive.extend((call,incoming,values,output_errors))
        return values,output_errors

    def rule_decision_device(self,states,actors):
        self._capture(actors)
        r=self.rules;t=self.t;e=r.executor
        batch,scratch,errors,active=r._begin(states)
        refs,extra=r._bind(scratch,actors,active)
        errors=errors|extra;active=(states.active&(errors==0)).view(t.uint8)
        call=r._invoke('rule_decision',batch,scratch,refs,active)
        errors=errors|call.statuses[:,0];active=(states.active&(errors==0)).view(t.uint8)
        chosen_call=e.select_result(call,(1,),active=active)
        chosen,index_errors=self._decode_index(chosen_call,errors,active)
        errors=errors|index_errors;active=(states.active&(errors==0)).view(t.uint8)
        legal_call=e.select_result(call,(0,),active=active)
        exported=e.export_projection(legal_call,active=active)
        errors=errors|exported.statuses[:,0];active=(states.active&(errors==0)).view(t.uint8)
        graph=exported.arena;l=graph.layout;c=e.constants
        action_refs=t.full((self.rows,self.max_actions),-1,dtype=t.int32,device=self.device)
        valid=t.zeros((self.rows,self.max_actions),dtype=t.bool,device=self.device)
        counts=t.zeros(self.rows,dtype=t.int64,device=self.device)
        decoded_errors=t.zeros_like(errors)
        incoming=errors.contiguous()
        args=[graph.nodes,graph.edges,graph.payload,graph.counts,graph.roots,
            c['nodes'],c['edges'],c['payload'],len(c['nodes']),len(c['edges']),len(c['payload']),
            chosen,incoming,active,action_refs,valid.view(t.uint8),counts,decoded_errors,
            self.rows,l.max_nodes,l.max_edges,l.max_payload_bytes,self.max_actions]
        self.module.launch('soa_teacher_refs',args,grid=(self.rows+127)//128,block=128,capture_safe=True)
        errors=errors|decoded_errors
        leased=r.action_store.put_device(graph,states.owners,states.active&(errors==0))
        errors=errors|leased.errors
        decision=DeviceDecision(states,r.action_store,leased.slots,leased.generations,action_refs,
            t.full_like(action_refs,-1,dtype=t.int64),counts,valid,
            t.empty((self.rows,0),dtype=t.float32,device=self.device),
            t.empty((self.rows,self.max_actions,0),dtype=t.float32,device=self.device),errors)
        self._keepalive.extend((call,chosen_call,legal_call,exported,incoming,decoded_errors,leased,decision))
        return decision,chosen,errors
