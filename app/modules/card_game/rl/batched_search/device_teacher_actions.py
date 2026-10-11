"""Internal GPU teacher legal/index plan; no public observation or NN input.

Consumes the exact frozen rule_decision tuple. Raw legal order, including
concede, stays intact. This is only for continuation step binding, never a root
policy/hash decision and never a full-rule or throughput validation receipt.
"""
from ..batched_duel.cuda_vm import DeviceCall, DeviceGraph
from ..batched_duel.cuda_runtime import CudaModule
from ..batched_duel.device_rule_protocol import DeviceDecision, DeviceStateBatch
from ..batched_duel.schema import GraphLayout
from ..batched_duel.vm_heap import emit_vm_heap_source
from .device_frozen_rules import DeviceFrozenRules, _ActiveRuleLanes

RAW_TEACHER_CAPACITY = 801  # Existing policy capacity 800 plus raw concede.
INVALID_LEGAL = 160

_SOURCE = emit_vm_heap_source() + r'''
using namespace duel_vm_heap;
extern "C" __global__ void teacher_refs(
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
_SPEC = ['int32*','int32*','uint8*','int32*','int32*','int32*','int32*','uint8*',
         'int32','int32','int32','int64*','int32*','uint8*','int32*','uint8*',
         'int64*','int32*','int32','int32','int32','int32','int32']


class DeviceTeacherActions:
    """One independent capture site returning internal (decision, index, errors)."""
    supports_full_duel_rules = False
    host_rule_callbacks = False
    root_policy_compatible = False

    def __init__(self, rules, *, max_actions=RAW_TEACHER_CAPACITY):
        if not isinstance(rules, DeviceFrozenRules) or rules.closed:
            raise TypeError('Live same-source DeviceFrozenRules owner required')
        if type(max_actions) is not int or not 1 <= max_actions <= RAW_TEACHER_CAPACITY:
            raise ValueError('Bounded raw teacher capacity required')
        c=rules.calls;p=rules.presentation;t=c.t
        if t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare teacher actions before CUDA capture')
        self.rules,self.calls,self.presentation=rules,c,p
        self.t,self.device,self.rows=t,c.device,c.rows
        self.max_actions=max_actions;self.closed=False;self.captured=False
        self.lanes=None;self.module=None
        try:
            self.lanes=_ActiveRuleLanes(c.rows,c.lanes,device=c.device,seconds=c.seconds)
            self.module=CudaModule(_SOURCE,{'teacher_refs':_SPEC},device=str(c.device))
            kw=dict(device=c.device)
            self.refs=t.empty((c.lanes,2),dtype=t.int32,**kw)
            self.load_errors=t.empty(c.lanes,dtype=t.int32,**kw)
            self.bind_errors=t.empty_like(self.load_errors)
            self.actor_rows=t.empty(c.lanes,dtype=t.int64,**kw)
            self.local_choices=t.empty(c.lanes,dtype=t.int64,**kw)
            self.local_errors=t.empty_like(self.load_errors)
            self.pair=p.reader.prepare_pair_workspace(c.lanes)
            l=c.input_graph.layout
            self.graph=DeviceGraph(l,t.zeros_like(c.input_graph.nodes),t.zeros_like(c.input_graph.edges),
                t.zeros_like(c.input_graph.payload),t.zeros_like(c.input_graph.counts),t.zeros_like(c.input_graph.roots))
            self.compact=c.vm.prepare_compaction_workspace(c.large,self.graph)
            self.local_refs=t.empty((c.lanes,max_actions),dtype=t.int32,**kw)
            self.local_valid=t.empty((c.lanes,max_actions),dtype=t.bool,**kw)
            self.local_counts=t.empty(c.lanes,dtype=t.int64,**kw)
            self.choices=t.full((c.rows,),-1,dtype=t.int64,**kw)
            self.errors=t.zeros(c.rows,dtype=t.int32,**kw)
            self.action_refs=t.full((c.rows,max_actions),-1,dtype=t.int32,**kw)
            self.valid=t.zeros((c.rows,max_actions),dtype=t.bool,**kw)
            self.counts=t.zeros(c.rows,dtype=t.int64,**kw)
            self.action_types=t.full((c.rows,max_actions),-1,dtype=t.int64,**kw)
            self.leases=DeviceStateBatch(rules.action_store,t.full((c.rows,),-1,dtype=t.int64,**kw),
                t.zeros(c.rows,dtype=t.int64,**kw),t.zeros(c.rows,dtype=t.int64,**kw),
                t.zeros(c.rows,dtype=t.bool,**kw),self.errors)
            self.observations=t.empty((c.rows,0),dtype=t.float32,**kw)
            self.candidates=t.empty((c.rows,max_actions,0),dtype=t.float32,**kw)
        except BaseException:
            self.close();raise

    def close(self):
        if not self.closed:
            if self.lanes is not None:self.lanes.close()
            if self.module is not None:self.module.close()
            self.closed=True

    def capture_device(self, states, actors):
        c=self.calls;p=self.presentation;t=self.t
        if self.closed or self.rules.closed or c.closed or p.closed:
            raise RuntimeError('Teacher actions closed')
        if self.captured:raise RuntimeError('Teacher action capture site already frozen')
        if not t.cuda.is_current_stream_capturing():raise RuntimeError('Embed teacher actions in CUDA capture')
        c._validate_states(states,c.rows,pool=c.state_store)
        c._exact(actors,t.int64,(c.rows,),'teacher actors')
        self.errors.copy_(states.errors);self.choices.fill_(-1)
        self.action_refs.fill_(-1);self.valid.zero_();self.counts.zero_()
        self.leases.slots.fill_(-1);self.leases.generations.zero_()
        self.leases.owners.copy_(states.owners);self.leases.active.zero_()
        def operation(local, ids, valid):
            gathered=c.state_store.gather_into_device(local,c.input_graph)
            errors=local.errors|gathered.errors
            active=(local.active&(errors==0)).view(t.uint8)
            c.vm.load_graph_into_device(c.workspace,c.input_graph,active=active,errors=self.load_errors)
            errors=errors|self.load_errors;active=(local.active&(errors==0)).view(t.uint8)
            self.actor_rows.copy_(t.where(valid.bool(),actors[ids],0))
            c.binder.bind_into_device(c.large,self.actor_rows,active,argument_refs=self.refs,errors=self.bind_errors)
            errors=errors|self.bind_errors;active=(local.active&(errors==0)).view(t.uint8)
            call=c.vm.launch_into_device(c.workspace,'rule_decision',argument_refs=self.refs,active=active,gpu_deadlines=c.deadlines)
            errors=errors|call.statuses[:,0];active=(local.active&(errors==0)).view(t.uint8)
            pair=p.reader.pair_into_device(call,active=active,output=self.pair)
            errors=errors|pair.errors;active=(local.active&(errors==0)).view(t.uint8)
            self.rules._decode_teacher_index(call.arena,pair.second,errors.contiguous(),active,self.local_choices,self.local_errors)
            errors=errors|self.local_errors;active=(local.active&(errors==0)).view(t.uint8)
            # The result tuple is decoded before changing the compaction root.
            legal_call=DeviceCall(c.vm,call.arena,pair.first,call.statuses,call.frames,None)
            c.vm.compact_into_device(legal_call,self.compact,active=active)
            errors=errors|self.compact.errors;active=(local.active&(errors==0)).view(t.uint8)
            b=c.vm._buffers;l=self.graph.layout
            self.module.launch('teacher_refs',[self.graph.nodes,self.graph.edges,self.graph.payload,
                self.graph.counts,self.graph.roots,b['nodes'],b['edges'],b['payload'],
                len(c.vm.image.nodes),len(c.vm.image.edges),len(c.vm.image.payload),self.local_choices,
                errors.contiguous(),active,self.local_refs,self.local_valid.view(t.uint8),self.local_counts,
                self.local_errors,c.lanes,l.max_nodes,l.max_edges,l.max_payload_bytes,self.max_actions],
                grid=(c.lanes+127)//128,block=128,capture_safe=True)
            errors=errors|self.local_errors;active=(local.active&(errors==0)).contiguous()
            saved=self.rules.action_store.put_device(self.graph,local.owners,active)
            saved.errors.bitwise_or_(errors)
            self.lanes.scatter_states_device(saved,self.leases)
            self.lanes.scatter_tensor_device(self.local_choices,self.choices)
            self.lanes.scatter_tensor_device(self.local_refs,self.action_refs)
            self.lanes.scatter_tensor_device(self.local_valid,self.valid)
            self.lanes.scatter_tensor_device(self.local_counts,self.counts)
        self.lanes.capture_dispatch(states,operation)
        self.errors.bitwise_or_(states.active.to(t.int32)*(self.lanes.loop.errors!=0).to(t.int32)[0]*152)
        decision=DeviceDecision(states,self.rules.action_store,self.leases.slots,self.leases.generations,
            self.action_refs,self.action_types,self.counts,self.valid,self.observations,self.candidates,self.errors)
        self.captured=True
        return decision,self.choices,self.errors
