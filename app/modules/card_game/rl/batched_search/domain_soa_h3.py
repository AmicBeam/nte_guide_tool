"""Three physical covered-teacher steps followed by independent auxiliary value.

This uses named state columns, actual anonymous rule previews and a separate
value-only inference bank. It does not use the VM teacher or qualify training.
"""
from ..batched_duel.domain_soa_execution import DomainSoAExecution
from ..batched_duel.domain_soa_rule_previews import DomainRulePreviews
from ..batched_duel.domain_soa_deadline import DomainDeviceDeadline
from ..batched_duel.domain_soa_observation import DomainSoAObservation
from .device_teacher_ranking import DeviceTeacherRanking,KIND_CODES
from .device_value_only_inference import DeviceValueOnlyInference

TEACHER_HORIZON=3
supports_full_duel_rules=False
h3_qualified=False


class DomainSoAH3:
    """Unqualified physical H3 integration; no primary-WDL or greedy substitution."""
    supports_full_duel_rules=False
    h3_qualified=False
    gpu_qualified=False
    host_rule_callbacks=False
    teacher_horizon=3

    def __init__(self,layout,auxiliary_inference,primary_inference,*,device='cuda',max_actions=800,allow_cpu_oracle=False,allow_unqualified=False):
        import torch
        if allow_unqualified is not True:raise RuntimeError('Physical H3 has not passed end-to-end acceptance')
        if not isinstance(auxiliary_inference,DeviceValueOnlyInference):raise TypeError('Separate auxiliary value-only bank required')
        if auxiliary_inference is primary_inference:raise ValueError('Auxiliary value cannot alias primary inference')
        if not getattr(auxiliary_inference,'weights',None)or not getattr(primary_inference,'weights',None):raise ValueError('Actual independently loaded parameter banks required')
        primary_addresses={value.data_ptr()for value in primary_inference.weights.values()}
        if any(value.data_ptr()in primary_addresses for value in auxiliary_inference.weights.values()):raise ValueError('Auxiliary parameters alias primary parameters')
        self.t=torch;self.layout=layout;self.auxiliary=auxiliary_inference;self.device=torch.device(device);self.closed=False
        if self.device.type=='cuda'and self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        if auxiliary_inference.device!=self.device or auxiliary_inference.feature_dim!=3479 or auxiliary_inference.candidate_dim!=1576:raise ValueError('Bound auxiliary 3479/1576 model contract required')
        self.executor=None;self.previews=None;self.observation=None;self.rankers=[]
        try:
            self.executor=DomainSoAExecution(layout,device=self.device,max_actions=max_actions,allow_cpu_oracle=allow_cpu_oracle,allow_unqualified=True)
            self.previews=DomainRulePreviews(layout,device=self.device,max_actions=max_actions,allow_cpu_oracle=allow_cpu_oracle,allow_unqualified=True)
            self.observation=DomainSoAObservation(layout,device=self.device,allow_cpu_oracle=allow_cpu_oracle)
            names=('end_turn','attack','ultimate','play_card','choose','mulligan','choose_deck_order','exit_ultimate','choose_cards')
            self.kinds=torch.tensor([KIND_CODES[name]for name in names],dtype=torch.int64,device=self.device)
            self.max_actions=max_actions
        except BaseException:self.close();raise

    def prepare(self,fields,viewers,model_index,active,deadline):
        from types import SimpleNamespace
        t=self.t;B=viewers.shape[0];A=self.max_actions
        if self.closed:raise RuntimeError('Closed H3')
        if not isinstance(deadline,DomainDeviceDeadline)or deadline.device!=self.device or deadline.rows!=B:raise TypeError('Shared resident deadline required; leaf calls never reset it')
        for v,dtype in ((viewers,t.int64),(model_index,t.int64),(active,t.bool)):
            if v.device!=self.device or v.dtype!=dtype or v.shape!=(B,):raise ValueError('Resident H3 routing columns required')
        world={name:value.clone(memory_format=t.contiguous_format)for name,value in fields.items()}
        actor=t.zeros_like(viewers);choice=t.zeros_like(viewers);on=active.clone()
        action_buffers=self.executor.actions.prepare(world,actor,on)
        # Buffers are resident and reused by each of the exact three steps.
        action_table={'counts':action_buffers.playing.counts,'errors':action_buffers.playing.errors,
                      'payloads':action_buffers.playing.payloads,'actions':action_buffers.playing.actions,
                      'selected':action_buffers.selected,'top':action_buffers.top,'bottom':action_buffers.bottom,
                      'bottom_plural':action_buffers.bottom_plural,'public_cards':action_buffers.playing.public.cards}
        preview_buffers=self.previews.prepare(world,actor,on,action_table)
        apply_buffers=self.executor.prepare(world,actor,choice,on)
        observed=self.observation.prepare(world,viewers,on)
        ranking=DeviceTeacherRanking(rows=B,max_actions=A,device=self.device,allow_cpu_reference=self.device.type=='cpu')
        self.rankers.append(ranking)
        return SimpleNamespace(owner=self,sources=dict(fields),world=world,viewers=viewers,model_index=model_index,source_active=active,
            actor=actor,choice=choice,active=on,action_buffers=action_buffers,preview=preview_buffers,apply=apply_buffers,observation=observed,
            ranking=ranking,deadline=deadline,errors=t.zeros(B,dtype=t.int32,device=self.device),steps=t.zeros(B,dtype=t.int32,device=self.device),
            values=t.zeros(B,dtype=t.float64,device=self.device),complete=t.zeros(B,dtype=t.bool,device=self.device),
            empty_candidates=t.empty((B,0,1576),dtype=t.float32,device=self.device),empty_valid=t.empty((B,0),dtype=t.bool,device=self.device),
            value_actor=t.empty((B,1),dtype=t.float32,device=self.device),value_only=t.ones(B,dtype=t.bool,device=self.device))

    def _actor(self,world):
        t=self.t;phase=world['phase'];done_a=((t.arange(2,device=self.device)[None,:]<world['mulligan_count'][:,None])&(world['mulligan_sides']==0)).any(-1)
        return t.where(phase==0,done_a.long(),t.where(phase==2,world['choice_side'].long(),world['active_side'].long()))

    def launch(self,p):
        if self.closed or p.owner is not self:raise ValueError('Closed or foreign H3 buffers')
        t=self.t;B=p.viewers.shape[0];A=self.max_actions
        for name,value in p.world.items():value.copy_(p.sources[name])
        p.errors.copy_(p.world['row_error']|((p.viewers<0)|(p.viewers>1)).int());p.steps.zero_();p.values.zero_();p.complete.zero_()
        for unused in range(TEACHER_HORIZON):
            terminal=p.world['phase']==3
            p.actor.copy_(self._actor(p.world))
            p.active.copy_(p.source_active&~terminal&(p.errors==0))
            live=p.deadline.refresh(p.active)
            p.errors.copy_(t.where(p.active&~live,p.errors|128,p.errors))
            p.active.copy_(live)
            actions=self.executor.actions.launch(p.action_buffers)
            preview=self.previews.launch(p.preview)
            codes=actions['payloads'][...,0]
            valid=(t.arange(A,device=self.device)[None,:]<actions['counts'][:,None])&p.active[:,None]
            ranking=p.ranking.rank_device(self.kinds[codes.clamp(0,8)],preview['battle'],preview['attack'],preview['attack_count'],preview['genesis'],preview['counter'],
                valid,actions['counts'],p.active,preview['errors'])
            p.choice.copy_(ranking.choices)
            p.errors.copy_(t.where(p.active,p.errors|ranking.errors,p.errors))
            p.active.copy_(p.active&(p.errors==0))
            applied=self.executor.launch(p.apply)
            p.errors.copy_(t.where(p.active,p.errors|applied['errors'],p.errors))
            succeeded=p.active&(applied['errors']==0)
            for name,value in p.world.items():value.copy_(t.where(succeeded.reshape((B,)+(1,)*(value.ndim-1)),applied['fields'][name],value))
            p.steps.add_(succeeded.int())
        terminal=p.world['phase']==3
        p.active.copy_(p.source_active&~terminal&(p.steps==TEACHER_HORIZON)&(p.errors==0))
        observed=self.observation.launch(p.observation)
        p.errors.copy_(t.where(p.active,p.errors|observed['errors'],p.errors))
        p.value_actor.copy_((self._actor(p.world)==p.viewers).float()[:,None])
        inferred=self.auxiliary.predict_device(observed['features'],p.empty_candidates,p.empty_valid,p.model_index,p.value_actor,value_only=p.value_only)
        p.errors.copy_(t.where(p.active,p.errors|inferred.invalid_rows.int(),p.errors))
        value=(inferred.wdl[:,2]-inferred.wdl[:,0]).double()
        valid_value=t.isfinite(value)&(value.abs()<=1+1e-6)
        p.errors.copy_(t.where(p.active&~valid_value,p.errors|1,p.errors))
        # Stored winner enum is a=0, b=1, draw=2; values stay in original viewer perspective.
        terminal_value=t.where(p.world['winner']==2,0.,t.where(p.world['winner'].long()==p.viewers,1.,-1.)).double()
        p.complete.copy_(p.source_active&(p.errors==0)&(terminal|(p.steps==TEACHER_HORIZON)))
        p.values.copy_(t.where(p.complete,t.where(terminal,terminal_value,value),t.zeros_like(value)))
        return dict(values=p.values,complete=p.complete,errors=p.errors,steps=p.steps,fields=p.world,
                    h3_qualified=False,full_duel_qualified=False,gpu_qualified=False)

    def close(self):
        if self.closed:return
        for ranking in self.rankers:ranking.close()
        if self.observation:self.observation.close()
        if self.previews:self.previews.close()
        if self.executor:self.executor.close()
        self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
