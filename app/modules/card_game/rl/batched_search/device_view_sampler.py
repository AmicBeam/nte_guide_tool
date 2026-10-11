"""Device chance packet with mandatory inner AND outer exact encoding guards.

No NumPy encoding inside the sampling VM; the same ordered public views are
encoded and compared by typed GPU kernels before any world gets a live lease.
This experimental adapter has no inherited full-rules qualification.
"""
from ..batched_duel.cuda_vm import DeviceCall,VmCapabilityError
from ..batched_duel.gpu_rule_projection import GraphProjectionKernel
from ..batched_duel.gpu_seven_candidates import encode_tokens_device,get_card_template_table
from ..batched_duel.device_rule_protocol import DeviceStateBatch
from .device_split_sampler import DeviceTransitionSampleCalls,_PacketPlan

TRANSITION_VIEWS_ENTRY='sample_world_guard_views_packet'
VALIDATION_ERROR=158


def compare_projected_views_device(left,right,active,template_table):
    import torch as t
    if active.device.type!='cuda':raise ValueError('Actual CUDA required')
    x,y=left.observations.observations,right.observations.observations
    if x.dtype!=t.float32 or y.dtype!=t.float32 or x.shape!=y.shape:
        raise ValueError('Exact float32 observation shape required')
    lc,le=encode_tokens_device(left.candidates,device=active.device,active=active,template_table=template_table)
    rc,re=encode_tokens_device(right.candidates,device=active.device,active=active,template_table=template_table)
    if lc.candidates.shape!=rc.candidates.shape:raise ValueError('Full ordered candidate width required')
    errors=left.errors|right.errors|le|re
    prefix=t.arange(lc.mask.shape[1],device=active.device)[None,:]<lc.counts[:,None]
    mask_bad=((lc.mask!=prefix)|(rc.mask!=prefix)).any(-1)
    obs_bad=((x!=y)|~t.isfinite(x)|~t.isfinite(y)).any(-1)
    relevant=(lc.mask|rc.mask)[:,:,None]
    cand_bad=(relevant&((lc.candidates!=rc.candidates)|~t.isfinite(lc.candidates)|~t.isfinite(rc.candidates))).any(-1).any(-1)
    bad=(lc.counts!=rc.counts)|(lc.counts<1)|mask_bad|obs_bad|cand_bad
    errors=t.where((errors==0)&bad,VALIDATION_ERROR,errors)
    return t.where(active,errors,0).to(t.int32)


class DeviceViewSampleCalls(DeviceTransitionSampleCalls):
    def __init__(self,vm,*args,max_actions=800,**kwargs):
        if TRANSITION_VIEWS_ENTRY not in vm.image.entries or not vm.capability(TRANSITION_VIEWS_ENTRY)['ready']:
            raise VmCapabilityError('Frozen mandatory dual-guard packet entry required')
        self.projection=None
        super().__init__(vm,*args,**kwargs)
        try:
            self.projection=GraphProjectionKernel(device=self.device,max_candidates=max_actions,runtime='seven')
            self.template_table=get_card_template_table(self.device)
            self._view_guards=self._packet_reader.prepare_pair_workspace(self.lanes)
            self._view_inner=self._packet_reader.prepare_pair_workspace(self.lanes)
            self._view_outer=self._packet_reader.prepare_pair_workspace(self.lanes)
        except BaseException:self.close();raise
    def close(self):
        if self.projection is not None:self.projection.close();self.projection=None
        super().close()

    def _execute(self, plan, local, ids, valid):
        if plan.entry != 'sample_world':
            return super()._execute(plan, local, ids, valid)
        if not isinstance(plan, _PacketPlan):
            raise RuntimeError('Guarded sampler packet plan required')
        t = self.t
        self._lanes = plan.lanes
        self.live_active.copy_(local.active.bool())
        self.lane_owners.copy_(local.owners)
        self.lane_errors.copy_(local.errors)
        self._refresh_active()
        gathered = self.state_store.gather_into_device(local, self.input_graph)
        self._or_errors(gathered.errors)
        self.vm.load_graph_into_device(self.workspace, self.input_graph,
                                      active=self.lane_active, errors=self.load_errors)
        self._or_errors(self.load_errors)
        self.lane_actors.copy_(self._pick(plan.actors))
        self.lane_seeds.copy_(self._pick(plan.seeds))
        self.binder.bind_into_device(self.large, self.lane_actors, self.lane_active,
                                    argument_refs=self.refs_bound, errors=self.bind_errors,
                                    seeds=self.lane_seeds)
        self._or_errors(self.bind_errors)
        call = self.vm.launch_into_device(self.workspace, TRANSITION_VIEWS_ENTRY,
            argument_refs=self.refs_bound, active=self.lane_active, gpu_deadlines=self.deadlines)
        self._or_errors(call.statuses[:,0])
        pair = self._packet_reader.pair_into_device(call, active=self.lane_active, output=self._packet_pair)
        self._or_errors(pair.errors)
        guards = self._packet_reader.pair_into_device(
            DeviceCall(self.vm,call.arena,pair.second,call.statuses,call.frames,None),
            active=self.lane_active,output=self._view_guards)
        self._or_errors(guards.errors)
        for ref,workspace in ((guards.first,self._view_inner),(guards.second,self._view_outer)):
            views=self._packet_reader.pair_into_device(
                DeviceCall(self.vm,call.arena,ref,call.statuses,call.frames,None),
                active=self.lane_active,output=workspace)
            self._or_errors(views.errors)
            left=self.projection.project_tokens_device(
                DeviceCall(self.vm,call.arena,views.first,call.statuses,call.frames,None),active=self.lane_active)
            right=self.projection.project_tokens_device(
                DeviceCall(self.vm,call.arena,views.second,call.statuses,call.frames,None),active=self.lane_active)
            self._or_errors(plan.guard_compare(left,right,self.lane_active.bool(),self.template_table))
        target=t.where(valid.bool(),ids,self.rows).long()
        plan.validation_required.index_copy_(0,target,t.where(self.lane_active.bool(),2,0).to(t.int64))
        state_call = DeviceCall(self.vm,call.arena,pair.first,call.statuses,call.frames,None)
        self.vm.compact_into_device(state_call,self.output_compact,active=self.lane_active)
        self._or_errors(self.output_compact.errors)
        put = self.state_store.put_device(self.output_graph,self.lane_owners,self.lane_active)
        self._or_errors(put.errors)
        self.out_slots.copy_(put.slots)
        self.out_gens.copy_(put.generations)
        self.out_active.copy_((put.slots>=0)&(self.lane_errors==0))
        plan.lanes.scatter_states_device(DeviceStateBatch(self.state_store,self.out_slots,
            self.out_gens,self.lane_owners,self.out_active,self.lane_errors),plan.output)


class DeviceViewValidatedSampler:
    """Only the internally checked packet may produce active sampled states."""
    def __init__(self,rules):
        if not isinstance(rules.calls,DeviceViewSampleCalls):raise TypeError('Mandatory dual-guard calls required')
        self.rules=rules;self.closed=False;self.captured=False
        self.compare=compare_projected_views_device
    def close(self):self.closed=True
    def capture_device(self,states,viewers,seeds):
        if self.closed or self.captured:raise RuntimeError('Sampler site closed or already captured')
        r=self.rules;r.calls._exact(seeds,r.t.int64,(r.rows,5),'sampler seeds')
        plan=r._consume('sample_world')
        if not isinstance(plan,_PacketPlan):raise TypeError('Guarded sampler plan required')
        plan.guard_compare=self.compare
        result=plan._capture_packet_device(states,actors=viewers,seeds=seeds)
        self.captured=True
        return result
