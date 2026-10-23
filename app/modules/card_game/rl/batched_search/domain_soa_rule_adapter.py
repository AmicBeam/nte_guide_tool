"""Unqualified no-VM rule protocol over named state and checked action leases.

All preparation is explicit and outside capture. Hot methods use resident tensor
routing and direct physical C++ kernels. No Python rules execute in this adapter.
"""
import numpy as np
from ..batched_duel import domain_soa_schema as schema
from ..batched_duel.device_rule_protocol import DeviceStateBatch,DeviceMeta,DeviceDecision,DeviceValueObservation
from ..batched_duel.domain_soa_state_pool import DomainSoAStatePool
from ..batched_duel.domain_soa_execution import DomainSoAExecution
from ..batched_duel.domain_soa_actions import DomainSoAActions
from ..batched_duel.domain_soa_candidates import DomainSoACandidates
from ..batched_duel.domain_soa_observation import DomainSoAObservation,PI
from ..batched_duel.domain_soa_sampler import DomainSoASampler
from ..batched_duel.domain_soa_rule_previews import DomainRulePreviews
from ..batched_duel.domain_soa_deadline import DomainDeviceDeadline
from .device_teacher_ranking import DeviceTeacherRanking,KIND_CODES
from .device_action_leases import release_actions_device
from .domain_soa_compact_candidates import DomainCandidateWorkspace,DomainCompactCandidates

supports_full_duel_rules=False
gpu_qualified=False
h3_qualified=False
search400_qualified=False


class DomainSoARuleAdapter:
    supports_full_duel_rules=False
    gpu_qualified=False
    h3_qualified=False
    search400_qualified=False
    require_teacher=True
    host_rule_callbacks=False
    no_vm=True

    def __init__(self,layout,*,rows,max_actions=800,capacity=None,device='cuda',allow_cpu_oracle=False,allow_unqualified=False):
        import torch
        if allow_unqualified is not True:raise RuntimeError('Physical rule/search integration has not passed acceptance')
        if type(rows)is not int or not 1<=rows<=min(400,layout.game_capacity):raise ValueError('Bounded physical cohort required')
        if type(max_actions)is not int or not 1<=max_actions<=800:raise ValueError('Full bounded candidate capacity required')
        self.t=torch;self.layout=layout;self.rows=rows;self.max_actions=max_actions;self.device=torch.device(device);self.closed=False
        if self.device.type=='cuda'and self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        if self.device.type=='cpu'and allow_cpu_oracle is not True:raise RuntimeError('Explicit CPU compiler oracle required')
        self.pool=None;self.action_pool=None;self.execution=None;self.actions=None;self.candidates=None;self.observation=None;self.sampler=None;self.previews=None;self.ranking=None;self.deadline=None
        specs=schema.array_specs(layout);S=max(4,layout.max_choice_cards);B=rows;A=max_actions;t=torch
        try:
            self.pool=DomainSoAStatePool(capacity or 8*rows,device=self.device,allow_cpu_reference=allow_cpu_oracle,
                specs=[(name,np.dtype(dtype).name,shape)for name,(dtype,shape)in specs.items()])
            action_specs=[('payloads','int64',(A,7)),('descriptions','int64',(A,6)),('selected','int64',(A,S)),('top','int64',(A,4)),('bottom','int64',(A,4)),('bottom_plural','bool',(A,)),('counts','int64',()),('state_versions','int64',())]
            self.action_pool=DomainSoAStatePool(3*rows,device=self.device,allow_cpu_reference=allow_cpu_oracle,specs=action_specs)
            self.fields={name:t.zeros((B,)+shape,dtype=getattr(t,np.dtype(dtype).name),device=self.device)for name,(dtype,shape)in specs.items()}
            self.viewers=t.zeros(B,dtype=t.int64,device=self.device);self.actors=t.zeros_like(self.viewers);self.indices=t.zeros_like(self.viewers)
            self.active=t.zeros(B,dtype=t.bool,device=self.device);self.sample_active=t.zeros(B,dtype=t.int32,device=self.device)
            self.seeds=t.zeros((B,5),dtype=t.int64,device=self.device);self.turn_counts=t.zeros(B,dtype=t.int64,device=self.device)
            self.row=t.arange(B,device=self.device);self.column=t.arange(A,device=self.device)[None,:]
            names=('end_turn','attack','ultimate','play_card','choose','mulligan','choose_deck_order','exit_ultimate','choose_cards')
            self.kind_codes=t.tensor([KIND_CODES[name]for name in names],dtype=t.int64,device=self.device)
            self.execution=DomainSoAExecution(layout,device=self.device,max_actions=A,allow_cpu_oracle=allow_cpu_oracle,allow_unqualified=True)
            self.actions=DomainSoAActions(layout,device=self.device,max_actions=A,allow_cpu_oracle=allow_cpu_oracle)
            self.action_buffers=self.actions.prepare(self.fields,self.viewers,self.active)
            ap=self.action_buffers
            self.action_table=dict(actions=ap.playing.actions,payloads=ap.playing.payloads,counts=ap.playing.counts,errors=ap.playing.errors,
                selected=ap.selected,top=ap.top,bottom=ap.bottom,bottom_plural=ap.bottom_plural,public_cards=ap.playing.public.cards)
            self.candidates=DomainCandidateWorkspace(rows=min(32,B),actions=A,cards=2*layout.max_cards,selection_width=S,device=self.device,allow_cpu_oracle=allow_cpu_oracle)
            self.observation=DomainSoAObservation(layout,device=self.device,allow_cpu_oracle=allow_cpu_oracle)
            self.observation_buffers=self.observation.prepare(self.fields,self.viewers,self.active)
            self.sampler=DomainSoASampler(layout,device=self.device,allow_cpu_oracle=allow_cpu_oracle,guard=True,seed_words=5)
            self.sample_buffers=self.sampler.prepare(self.fields,self.viewers,self.seeds,self.sample_active)
            self.sample_original_phase=t.zeros_like(self.fields['phase']);self.sample_original_actor=t.zeros_like(self.fields['active_side'])
            self.apply_buffers=self.execution.prepare(self.fields,self.actors,self.indices,self.active)
            self.previews=DomainRulePreviews(layout,device=self.device,max_actions=A,allow_cpu_oracle=allow_cpu_oracle,allow_unqualified=True)
            self.preview_buffers=self.previews.prepare(self.fields,self.viewers,self.active,self.action_table)
            self.ranking=DeviceTeacherRanking(rows=B,max_actions=A,device=self.device,allow_cpu_reference=allow_cpu_oracle)
            self.deadline=DomainDeviceDeadline(B,device=self.device,allow_cpu_oracle=allow_cpu_oracle)
            self.deadline_ms=t.zeros(B,dtype=t.float64,device=self.device)
            self.empty_candidates=t.empty((B,0,1576),dtype=t.float32,device=self.device)
        except BaseException:self.close();raise

    def _vector(self,value,dtype,shape):
        if value.device!=self.device or value.dtype!=dtype or value.shape!=shape:raise ValueError('Resident protocol column required')

    def _load(self,states):
        if self.closed or not isinstance(states,DeviceStateBatch)or states.pool is not self.pool:raise ValueError('Live bound state pool required')
        if states.slots.shape!=(self.rows,):raise ValueError('Fixed cohort size required')
        fields=self.pool.gather_device(states,shared_reads=True)
        for name,target in self.fields.items():target.copy_(fields[name])
        errors=states.errors|self.fields['row_error']
        self.fields['row_error'].copy_(errors)
        self.active.copy_(states.active&(errors==0))
        return errors

    def _actors(self):
        t=self.t;f=self.fields;phase=f['phase'];done_a=((t.arange(2,device=self.device)[None,:]<f['mulligan_count'][:,None])&(f['mulligan_sides']==0)).any(-1)
        actors=t.where(phase==0,done_a.long(),t.where(phase==2,f['choice_side'].long(),f['active_side'].long()))
        return t.where(phase==3,-1,actors)

    def begin_search_device(self,milliseconds):
        self._vector(milliseconds,self.t.float64,(self.rows,));self.deadline_ms.copy_(milliseconds);self.deadline.start(self.deadline_ms)

    def metadata_device(self,states):
        t=self.t;errors=self._load(states);f=self.fields;finished=f['phase']==3;actors=self._actors()
        bad=(f['phase']<0)|(f['phase']>3)|(~finished&((actors<0)|(actors>1)))
        bad|=finished&((f['winner_presence']!=2)|(f['winner']<0)|(f['winner']>2))
        errors=errors|(states.active&bad).int()
        winner=t.where(finished,t.where(f['winner']==2,-1,f['winner'].long()),-2)
        return DeviceMeta(t.where(states.active,actors,-1).clone(),f['phase'].int().clone(),winner.clone(),finished.clone(),f['version'].long().clone(),errors.clone())

    def _decision(self,states,observers,features):
        t=self.t;errors=self._load(states);self._vector(observers,t.int64,(self.rows,));self.viewers.copy_(observers)
        self.active.logical_and_(self.fields['phase']!=3)
        out=self.actions.launch(self.action_buffers);errors=errors|out['errors']
        turn_index=PI['turn_count'];side=self.viewers.clamp(0,1)
        self.turn_counts.copy_(t.where(self.fields['player_int_presence'][self.row,side,turn_index]>=2,self.fields['player_int_value'][self.row,side,turn_index],0).long())
        if features:
            public_fields=dict(cards=out['public_cards'],actions=out['actions'],selected=out['selected'],top=out['top'],bottom=out['bottom'],
                counts=out['counts'],viewers=self.viewers,turn_counts=self.turn_counts,active=self.active,prior_errors=errors,
                mask=(self.column<out['counts'][:,None])&self.active[:,None]&(errors==0)[:,None])
            candidates=DomainCompactCandidates(public_fields,self.candidates)
        else:candidates=self.empty_candidates
        active=self.active&(errors==0)
        lease_fields=dict(payloads=out['payloads'],descriptions=out['actions'],selected=out['selected'],top=out['top'],bottom=out['bottom'],bottom_plural=out['bottom_plural'],counts=out['counts'],state_versions=self.fields['version'].long())
        lease=self.action_pool.put_device(lease_fields,states.owners,active,errors=errors)
        valid=(self.column<out['counts'][:,None])&lease.active[:,None]
        refs=t.where(valid,self.column.expand(self.rows,-1),-1).int()
        kinds=self.kind_codes[out['payloads'][...,0].clamp(0,8)]
        decision=DeviceDecision(states,self.action_pool,lease.slots,lease.generations,refs,kinds.clone(),out['counts'].clone(),valid.clone(),out['observation'].clone(),candidates,lease.errors.clone())
        return decision,out

    def decision_device(self,states,observers):return self._decision(states,observers,True)[0]

    def rule_decision_device(self,states,observers):
        decision,out=self._decision(states,observers,False)
        preview=self.previews.launch(self.preview_buffers)
        ranked=self.ranking.rank_device(decision.action_types,preview['battle'],preview['attack'],preview['attack_count'],preview['genesis'],preview['counter'],decision.valid,decision.counts,self.active,decision.errors|preview['errors'])
        return decision,ranked.choices.clone(),ranked.errors.clone()

    def sample_world_device(self,states,viewers,seeds):
        t=self.t;errors=self._load(states);self._vector(viewers,t.int64,(self.rows,));self._vector(seeds,t.int64,(self.rows,5))
        self.viewers.copy_(viewers);self.seeds.copy_(seeds);self.sample_active.copy_(self.active.int())
        # Match information_search.sample_world: sample a playing proxy for
        # mulligans, restore phase/actor, then check the original visible input.
        self.sample_original_phase.copy_(self.fields['phase']);self.sample_original_actor.copy_(self.fields['active_side'])
        mulligan=self.active&(self.sample_original_phase==0)
        self.fields['phase'].copy_(t.where(mulligan,1,self.fields['phase']))
        self.fields['active_side'].copy_(t.where(mulligan,viewers.to(self.fields['active_side'].dtype),self.fields['active_side']))
        sampled=self.sampler.launch(self.sample_buffers)
        fields=sampled.fields if hasattr(sampled,'fields')else sampled
        self.fields['phase'].copy_(self.sample_original_phase);self.fields['active_side'].copy_(self.sample_original_actor)
        fields['phase'].copy_(t.where(mulligan,self.sample_original_phase,fields['phase']))
        fields['active_side'].copy_(t.where(mulligan,self.sample_original_actor,fields['active_side']))
        self.sample_buffers.information_active.copy_(mulligan&(fields['row_error']==0))
        checked=self.sampler._information_guard.launch(self.sample_buffers.information)
        bad=mulligan&(checked['errors']!=0)
        fields['row_error'].bitwise_or_(t.where(mulligan,checked['errors'],t.zeros_like(checked['errors'])))
        for name,value in fields.items():
            if name!='row_error':value.copy_(t.where(bad.reshape((self.rows,)+(1,)*(value.ndim-1)),self.fields[name],value))
        return self.pool.put_device(dict(fields),states.owners,self.active,errors=errors|fields['row_error'])

    def step_device(self,states,actors,decision,action_indices,*,physical=False):
        t=self.t;errors=self._load(states);self._vector(actors,t.int64,(self.rows,));self._vector(action_indices,t.int64,(self.rows,))
        if decision.action_pool is not self.action_pool or decision.states.pool is not self.pool:raise ValueError('Bound immutable action lease required')
        lease=DeviceStateBatch(self.action_pool,decision.action_slots,decision.action_generations,states.owners,self.active,t.zeros_like(errors))
        action_fields=self.action_pool.gather_device(lease,shared_reads=True)
        errors=errors|lease.errors|decision.errors
        foreign=(decision.states.owners!=states.owners)|(decision.states.slots!=states.slots)|(decision.states.generations!=states.generations)
        self.viewers.copy_(actors);self.actors.copy_(actors);self.indices.copy_(action_indices)
        fresh=self.actions.launch(self.action_buffers);safe=action_indices.clamp(0,self.max_actions-1)
        invalid=foreign|(action_indices<0)|(action_indices>=action_fields['counts'])|(action_indices>=fresh['counts'])
        invalid|=action_fields['state_versions']!=self.fields['version'].long()
        for name in ('payloads','selected','top','bottom'):
            expected=action_fields[name][self.row,safe];actual=fresh[name][self.row,safe]
            invalid|=(expected!=actual).flatten(1).any(-1)
        invalid|=action_fields['bottom_plural'][self.row,safe]!=fresh['bottom_plural'][self.row,safe]
        errors=errors|fresh['errors']|(self.active&invalid).int()
        self.active.logical_and_(errors==0)
        advanced=self.execution.launch(self.apply_buffers)
        return self.pool.put_device(advanced['fields'],states.owners,self.active,errors=errors|advanced['errors'])

    def clean_device(self,states):
        self.pool.validate(states,shared_reads=True)
        return states

    def value_observation_device(self,states,viewers):
        t=self.t;errors=self._load(states);self._vector(viewers,t.int64,(self.rows,));self.viewers.copy_(viewers)
        finished=self.fields['phase']==3;actor=self._actors();self.active.logical_and_(~finished)
        observed=self.observation.launch(self.observation_buffers)
        value=t.where(self.fields['winner']==2,0.,t.where(self.fields['winner'].long()==viewers,1.,-1.)).double()
        return DeviceValueObservation(observed['features'].clone(),(actor==viewers).float()[:,None],finished.clone(),t.where(finished,value,0.).clone(),(errors|observed['errors']).clone())

    def teacher_continue_device(self,states,viewers,*,horizon=3):
        if horizon!=3:raise ValueError('Frozen covered teacher requires H3')
        t=self.t;current=states
        for unused in range(3):
            meta=self.metadata_device(current);todo=current.active&~meta.finished&(meta.errors==0)
            live=self.deadline.refresh(todo).clone();errors=meta.errors|(todo&~live).int()*128
            request=DeviceStateBatch(self.pool,current.slots,current.generations,current.owners,live,errors)
            decision,choice,teacher_errors=self.rule_decision_device(request,meta.actors)
            request=DeviceStateBatch(self.pool,current.slots,current.generations,current.owners,live,errors|teacher_errors)
            advanced=self.step_device(request,meta.actors,decision,choice,physical=False)
            action_errors=release_actions_device(decision,live,current.owners)
            changed=(current.slots!=states.slots)|(current.generations!=states.generations)
            release=DeviceStateBatch(self.pool,current.slots,current.generations,current.owners,live&changed,t.zeros_like(errors))
            self.pool.release_device(release)
            current=DeviceStateBatch(self.pool,t.where(live,advanced.slots,current.slots),t.where(live,advanced.generations,current.generations),current.owners,
                current.active,errors|teacher_errors|advanced.errors|action_errors|release.errors)
        return current

    def release_device(self,states):self.pool.release_device(states)

    def close(self):
        if self.closed:return
        for component in (self.ranking,self.deadline,self.previews,self.sampler,self.observation,self.candidates,self.actions,self.execution,self.action_pool,self.pool):
            if component is not None:component.close()
        self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def make_gumbel_search(rules,inference,auxiliary_inference,statistics,information_keys,*,random=None,noise=False,budget_seconds=120,allow_cpu_oracle=False):
    """Bind the unchanged 32/cap16/H3/depth10/per-simulation/mctx_default profile.

    Still an explicit integration sandbox. This is not a training approval gate.
    """
    from .device_search import DeviceGumbelSearch
    if not isinstance(rules,DomainSoARuleAdapter):raise TypeError('Named-column rule adapter required')
    if inference is auxiliary_inference:raise ValueError('Independent auxiliary bank required')
    from .device_value_only_inference import DeviceValueOnlyInference
    if not isinstance(auxiliary_inference,DeviceValueOnlyInference):raise TypeError('Independent auxiliary value-only bank required')
    if not getattr(inference,'weights',None)or not getattr(auxiliary_inference,'weights',None):raise ValueError('Loaded primary and auxiliary parameter banks required')
    primary_addresses={value.data_ptr()for value in inference.weights.values()}
    if any(value.data_ptr()in primary_addresses for value in auxiliary_inference.weights.values()):raise ValueError('Auxiliary parameters alias primary parameters')
    if not isinstance(budget_seconds,(int,float))or not 0<budget_seconds<=86400:raise ValueError('Bounded shared decision budget required')
    class DomainGumbelSearch(DeviceGumbelSearch):
        def initialize(self,*args,**kwargs):
            self.rules.begin_search_device(self.domain_budget_ms)
            return super().initialize(*args,**kwargs)
    search=DomainGumbelSearch(rules,inference,auxiliary_inference,statistics,information_keys,
        rows=rules.rows,max_actions=rules.max_actions,simulations=32,gumbel_candidates=16,max_depth=10,terminal_horizon=3,
        teacher_mode='covered_value',random=random,noise=noise,device=rules.device,
        allow_cpu_reference=allow_cpu_oracle,allow_synthetic=allow_cpu_oracle,
        opponent_mode='rule',reuse_sampled_world=False,reuse_root_decision=False,q_scale='mctx_default')
    search.domain_budget_ms=rules.t.full((rules.rows,),budget_seconds*1000.,dtype=rules.t.float64,device=rules.device)
    return search
