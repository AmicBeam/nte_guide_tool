"""Device-controlled shared rule scratch lanes; no host row dispatch.

Each static operation has a bounded nested CUDA WHILE. Persistent game roots
remain device resident while a fixed number of large VM lanes are reused.
This dispatcher alone is neither an actual RuleAdapter nor capacity evidence.
"""
from .device_rule_protocol import DeviceStateBatch


_SOURCE = r'''
extern "C" __global__ void scatter_rule_bytes(
 const long long* ids,const unsigned char* valid,const unsigned char* local,
 unsigned char* output,long long stride,int rows,int lanes){
 int lane=blockIdx.x;if(lane>=lanes||!valid[lane])return;
 long long row=ids[lane];if(row<0||row>=rows)return;
 for(long long byte=threadIdx.x;byte<stride;byte+=blockDim.x)
  output[row*stride+byte]=local[(long long)lane*stride+byte];
}
extern "C" __global__ void select_rule_lanes(
 const long long* cursor,const unsigned char* active,const int* errors,
 long long* ids,unsigned char* selected,unsigned char* valid,
 long long* processed,long long* peak,int rows,int lanes){
 if(blockIdx.x||threadIdx.x)return;
 long long begin=*cursor,live=0;
 for(int lane=0;lane<lanes;lane++){
  long long row=begin+lane;int in=row>=0&&row<rows;
  ids[lane]=in?row:0;valid[lane]=(unsigned char)in;
  selected[lane]=(unsigned char)(in&&active[row]&&!errors[row]);
  live+=selected[lane]!=0;
 }
 *processed+=live;if(live>*peak)*peak=live;
}
extern "C" __global__ void advance_rule_lanes(
 long long* cursor,unsigned char* pending,int rows,int lanes){
 if(blockIdx.x||threadIdx.x)return;
 *cursor+=lanes;*pending=(unsigned char)(*cursor<rows);
}
extern "C" __global__ void scatter_rule_leases(
 const long long* ids,const unsigned char* valid,
 const long long* slots,const long long* generations,const long long* owners,
 const unsigned char* active,const int* errors,
 long long* all_slots,long long* all_generations,long long* all_owners,
 unsigned char* all_active,int* all_errors,int rows,int lanes){
 int lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=lanes||!valid[lane])return;
 long long row=ids[lane];if(row<0||row>=rows)return;
 all_slots[row]=slots[lane];all_generations[row]=generations[lane];all_owners[row]=owners[lane];
 all_active[row]=active[lane];all_errors[row]|=errors[lane];
}
'''


class DeviceRuleLanes:
    supports_full_duel_rules=False

    def __init__(self,rows,lanes=32,*,device='cuda',seconds=60):
        import torch
        if any(type(v) is not int or not 0<v<2**31 for v in (rows,lanes)):
            raise ValueError('Positive bounded root and scratch lane counts required')
        self.t=torch;self.rows=rows;self.lanes=min(rows,lanes)
        self.device=torch.device(device)
        if self.device.type!='cuda' or not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no host scratch dispatch fallback')
        if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        from .cuda_runtime import CudaModule
        from ..batched_search.cuda_conditional_graph import CudaConditionalWhile
        self.module=CudaModule(_SOURCE,{
            'scatter_rule_bytes':['int64*','uint8*','uint8*','uint8*','int64','int32','int32'],
            'select_rule_lanes':['int64*','uint8*','int32*','int64*','uint8*','uint8*',
                                 'int64*','int64*','int32','int32'],
            'advance_rule_lanes':['int64*','uint8*','int32','int32'],
            'scatter_rule_leases':['int64*','uint8*','int64*','int64*','int64*','uint8*','int32*',
                                   'int64*','int64*','int64*','uint8*','int32*','int32','int32'],
        },device=str(self.device))
        try:
            self.loop=CudaConditionalWhile(device=str(self.device),
                max_ticks=(rows+self.lanes-1)//self.lanes+1,seconds=seconds)
        except BaseException:
            self.module.close();raise
        self.cursor=torch.zeros(1,dtype=torch.int64,device=self.device)
        self.pending=torch.zeros(1,dtype=torch.uint8,device=self.device)
        self.ids=torch.zeros(self.lanes,dtype=torch.int64,device=self.device)
        self.selected=torch.zeros(self.lanes,dtype=torch.uint8,device=self.device)
        self.valid=torch.zeros(self.lanes,dtype=torch.uint8,device=self.device)
        self.processed=torch.zeros(1,dtype=torch.int64,device=self.device)
        self.peak=torch.zeros(1,dtype=torch.int64,device=self.device)
        self.closed=False

    def close(self):
        if not self.closed:
            self.loop.close();self.module.close();self.closed=True

    def _validate_states(self,states,rows):
        if not isinstance(states,DeviceStateBatch):raise TypeError('DeviceStateBatch required')
        t=self.t
        for value,dtype in ((states.slots,t.int64),(states.generations,t.int64),
                (states.owners,t.int64),(states.active,t.bool),(states.errors,t.int32)):
            if (not isinstance(value,t.Tensor) or value.device!=self.device or value.dtype!=dtype
                    or value.shape!=(rows,) or not value.is_contiguous()):
                raise ValueError('Exact device lease vectors required')

    def select_states_device(self,states):
        """Gather current device-selected leases; padding has no valid lease."""
        self._validate_states(states,self.rows)
        t=self.t
        self.module.launch('select_rule_lanes',[
            self.cursor,states.active.view(t.uint8),states.errors,self.ids,self.selected,self.valid,
            self.processed,self.peak,self.rows,self.lanes],grid=1,block=1,capture_safe=True)
        valid=self.valid.bool()
        # Active describes a lease, not permission to execute an errored row.
        # Retain failed live leases so cleanup can still release their pages.
        return DeviceStateBatch(states.pool,
            t.where(valid,states.slots[self.ids],-1),
            t.where(valid,states.generations[self.ids],0),
            t.where(valid,states.owners[self.ids],0),
            valid & states.active[self.ids],t.where(valid,states.errors[self.ids],0))

    def scatter_states_device(self,local,output):
        """Copy real lanes only, so padding cannot overwrite root zero."""
        self._validate_states(local,self.lanes);self._validate_states(output,self.rows)
        if local.pool is not output.pool:raise ValueError('Exact output graph pool required')
        self.module.launch('scatter_rule_leases',[
            self.ids,self.valid,local.slots,local.generations,local.owners,
            local.active.view(self.t.uint8),local.errors,
            output.slots,output.generations,output.owners,output.active.view(self.t.uint8),output.errors,
            self.rows,self.lanes],grid=(self.lanes+127)//128,block=128,capture_safe=True)
        return output

    def scatter_tensor_device(self,local,output):
        """Scatter fixed-shape row payloads bit exactly, without padded writes."""
        t=self.t
        if (not isinstance(local,t.Tensor) or not isinstance(output,t.Tensor)
                or local.device!=self.device or output.device!=self.device
                or local.dtype!=output.dtype or local.ndim<1 or output.ndim!=local.ndim
                or local.shape[0]!=self.lanes or output.shape[0]!=self.rows
                or local.shape[1:]!=output.shape[1:] or not local.is_contiguous()
                or not output.is_contiguous()):
            raise ValueError('Matching contiguous device row payloads required')
        src=local.reshape(self.lanes,-1).view(t.uint8)
        dst=output.reshape(self.rows,-1).view(t.uint8)
        if src.shape[1]<1:raise ValueError('Nonempty row payload required')
        if src.untyped_storage().data_ptr()==dst.untyped_storage().data_ptr():
            raise ValueError('Separate scatter output storage required')
        self.module.launch('scatter_rule_bytes',[
            self.ids,self.valid,src,dst,src.shape[1],self.rows,self.lanes],
            grid=self.lanes,block=128,capture_safe=True)
        return output

    def capture_dispatch(self,states,operation):
        """Embed one static operation in the enclosing device search graph.

        operation(lane_states, root_ids, valid_mask) runs once during capture.
        GPU replay selects/executes/scatters each cohort until all roots are
        visited, including a final partial cohort. No host predicate or row loop
        runs during execution. Allocate each operation's dispatcher before capture.
        Gate rule execution with local.active & (local.errors == 0); errored
        leases are preserved for cleanup rather than silently discarded.
        """
        if self.closed:raise RuntimeError('Rule scratch dispatcher closed')
        self._validate_states(states,self.rows)
        if not callable(operation):raise TypeError('Static operation callback required')
        if not self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Rule dispatch must be embedded in CUDA capture')
        def initialize():
            self.cursor.zero_();self.pending.fill_(1);self.processed.zero_();self.peak.zero_()
        def body():
            local=self.select_states_device(states)
            operation(local,self.ids,self.valid)
            self.module.launch('advance_rule_lanes',[
                self.cursor,self.pending,self.rows,self.lanes],grid=1,block=1,capture_safe=True)
        self.loop.capture_loop(self.pending,body,initialize=initialize)
