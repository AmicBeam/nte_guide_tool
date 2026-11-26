"""Explicit three-step SoA continuation with stored states between rule calls.

Prepare for a direct SoA work-lane owner before capture. This composition
does not change the default teacher or grant full-rule/GPU qualification.
"""
from ..batched_duel.device_rule_protocol import DeviceStateBatch
from .device_action_leases import release_actions_device


LEASE_ERROR=2048
METADATA_ERROR=64


class DeviceSoATeacherPipeline:
    supports_full_duel_rules=False
    host_rule_callbacks=False
    teacher_horizon=3

    def __init__(self,rules,teacher_actions):
        from ..batched_duel.device_soa_storage import DeviceSoAStore
        if not isinstance(getattr(rules,'store',None),DeviceSoAStore):
            raise TypeError('Direct SoA typed state owner required')
        for name in ('_begin','metadata_device','step_device','clean_device'):
            if not callable(getattr(rules,name,None)):
                raise TypeError('Direct SoA '+name+' required')
        if getattr(teacher_actions,'rules',None) is not rules:
            raise ValueError('Raw teacher must use the same direct SoA owner')
        if not callable(getattr(teacher_actions,'rule_decision_device',None)):
            raise TypeError('Raw SoA teacher action decoder required')
        if rules.action_store is rules.store:
            raise ValueError('Independent teacher action arena required')
        if getattr(rules,'inner',None) is not None:
            raise ValueError('Prepare the staged teacher on direct work lanes')
        if rules.device.type!='cuda' or not rules.t.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no CPU continuation fallback')
        if rules.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare staged SoA teacher before CUDA capture')
        self._prepare(rules,teacher_actions)

    def _prepare(self,rules,teacher_actions):
        """Allocate static handle vectors; CPU tests call this as an oracle."""
        self.rules,self.teacher_actions=rules,teacher_actions
        self.t,self.device,self.rows,self.pool=rules.t,rules.device,rules.rows,rules.store
        self.closed=False;self.captured=False;self._owned=[];self._keepalive=[]
        t=self.t;kw=dict(device=self.device)
        def empty():
            return DeviceStateBatch(self.pool,t.full((self.rows,),-1,dtype=t.int64,**kw),
                t.zeros(self.rows,dtype=t.int64,**kw),t.zeros(self.rows,dtype=t.int64,**kw),
                t.zeros(self.rows,dtype=t.bool,**kw),t.zeros(self.rows,dtype=t.int32,**kw))
        self.current=empty();self.work=empty();self._owned.extend((self.current,self.work))
        self.stage_errors=t.zeros((3,6,self.rows),dtype=t.int32,**kw)
        self._indices=t.arange(self.rows,dtype=t.int64,**kw)

    def close(self):
        # The returned current belongs to the caller. Never release it, close
        # shared rules/stores/decoder, or reclaim another graph's live output.
        self.closed=True;self._owned.clear();self._keepalive.clear()

    def teacher_continue_device(self,states,viewers,*,horizon=3):
        """Optional rule-adapter spelling; retains the single capture-site guard."""
        return self.capture_device(states,viewers,horizon=horizon)

    def _check(self,states,viewers):
        t=self.t
        if not isinstance(states,DeviceStateBatch) or states.pool is not self.pool:
            raise ValueError('Same typed SoA parent arena required')
        for name,dtype in (('slots',t.int64),('generations',t.int64),('owners',t.int64),
                           ('active',t.bool),('errors',t.int32)):
            value=getattr(states,name)
            if (not isinstance(value,t.Tensor) or value.device!=self.device or value.dtype!=dtype
                    or value.shape!=(self.rows,) or not value.is_contiguous()):
                raise ValueError('Exact device parent lease vectors required')
        if (not isinstance(viewers,t.Tensor) or viewers.device!=self.device or viewers.dtype!=t.int64
                or viewers.shape!=(self.rows,) or not viewers.is_contiguous()):
            raise ValueError('Exact device teacher viewer vector required')

    def _remember(self,source):
        if source.pool is not self.pool:
            raise ValueError('Teacher result must use the same typed state arena')
        # Rule adapters may return aliases. Own the handle tensors before any
        # cleanup so no borrowed parent/work/result metadata gets mutated.
        saved=DeviceStateBatch(self.pool,*[getattr(source,name).clone()
            for name in ('slots','generations','owners','active','errors')])
        self._owned.append(saved);self._keepalive.append(source)
        return saved

    def _assign(self,target,source,mask=None):
        for name in ('slots','generations','owners','active'):
            old,new=getattr(target,name),getattr(source,name)
            old.copy_(new if mask is None else self.t.where(mask,new,old))

    def _aliases(self,slots,generations,refs):
        return ((slots[:,None]==refs.slots[None,:])
            &(generations[:,None]==refs.generations[None,:])
            &(slots[:,None]>=0)&(refs.slots[None,:]>=0)).any(-1)

    def _accepted(self,source,mask,parent):
        """Reject stale/foreign/physical outputs before adopting any handle."""
        t=self.t;p=self.pool;slots=source.slots;gens=source.generations
        fit=(slots>=0)&(slots<p.live.numel())&(gens>=0)
        safe=slots.clamp(0,p.live.numel()-1)
        owned=fit&p.live[safe].bool()&(p.generation[safe]==gens)&(p.owner[safe]==source.owners)
        valid=owned&(source.owners==parent.owners)&source.active
        # Physical roots are protected across every row, including inactive
        # parents; a malicious cross-row alias is never an independent output.
        valid&=~self._aliases(slots,gens,parent)
        duplicate=((slots[:,None]==slots[None,:])&(gens[:,None]==gens[None,:])
                   &mask[None,:]&mask[:,None]&(self._indices[:,None]!=self._indices[None,:])).any(-1)
        valid&=~duplicate
        bad=mask&~valid&(source.errors==0)
        self.current.errors.bitwise_or_(bad.to(t.int32)*LEASE_ERROR)
        return mask&valid&(source.errors==0)&(self.current.errors==0)

    def _release(self,source,mask,parent,*,protected=()):
        """Release checked owned identities once and null internal aliases."""
        t=self.t;p=self.pool
        slots,gens,owners=(getattr(source,name).clone() for name in ('slots','generations','owners'))
        fit=(slots>=0)&(slots<p.live.numel())&(gens>=0)
        safe=slots.clamp(0,p.live.numel()-1)
        live=fit&p.live[safe].bool()&(p.generation[safe]==gens)
        owner_ok=(p.owner[safe]==owners)&(owners==parent.owners)
        parent_alias=self._aliases(slots,gens,parent)
        retained=parent_alias.clone()
        for other in protected:retained|=self._aliases(slots,gens,other)
        bad=mask&(((slots>=0)&~fit)|(live&~owner_ok))
        self.current.errors.bitwise_or_(bad.to(t.int32)*LEASE_ERROR)
        take=mask&live&owner_ok&~retained
        repeated=((slots[:,None]==slots[None,:])&(gens[:,None]==gens[None,:])
                  &take[None,:]&(self._indices[:,None]>self._indices[None,:])).any(-1)
        take&=~repeated
        request=DeviceStateBatch(p,slots,gens,owners,take.contiguous(),t.zeros_like(source.errors))
        p.release_device(request)
        self.current.errors.bitwise_or_((request.errors!=0).to(t.int32)*LEASE_ERROR)
        freed=take&(request.errors==0)
        # Clear stale aliases and invalid parent borrows, without releasing
        # either a reused generation or the caller's physical parent lease.
        clear=mask&(~live|parent_alias|freed)
        for other in self._owned:
            aliases=((other.slots[:,None]==slots[None,:])
                &(other.generations[:,None]==gens[None,:])&clear[None,:]
                &(other.slots[:,None]>=0))
            same_owner=other.owners[:,None]==owners[None,:]
            mismatch=(aliases&~same_owner).any(-1)
            self.current.errors.bitwise_or_(mismatch.to(t.int32)*LEASE_ERROR)
            drop=(aliases&same_owner).any(-1)
            other.slots.masked_fill_(drop,-1);other.generations.masked_fill_(drop,0)
            other.active.masked_fill_(drop,False)

    def capture_device(self,states,viewers,*,horizon=3):
        if (self.closed or getattr(self.teacher_actions,'closed',False)
                or getattr(getattr(self.rules,'executor',None),'closed',False)):
            raise RuntimeError('Staged SoA teacher closed')
        if self.captured:raise RuntimeError('Staged SoA teacher capture site already frozen')
        if type(horizon) is not int or horizon!=3:
            raise ValueError('Staged SoA teacher requires exactly horizon 3')
        self._check(states,viewers)
        t=self.t
        if not t.cuda.is_current_stream_capturing():
            raise RuntimeError('Embed staged SoA teacher in CUDA capture')
        self.captured=True
        self.current.slots.fill_(-1);self.current.generations.zero_();self.current.active.zero_()
        self.current.owners.copy_(states.owners);self.current.errors.copy_(states.errors)
        self.stage_errors.zero_()
        # Exactly one typed clone. Each subsequent operation gathers its stored
        # input and resets its own transient scratch through the SoA adapter.
        parent=DeviceStateBatch(self.pool,states.slots,states.generations,states.owners,
                               states.active,states.errors.clone())
        batch,_,errors,active=self.rules._begin(parent)
        copied=self._remember(self.pool.put_device(batch,states.owners,active.bool(),errors=errors))
        self.current.errors.bitwise_or_(copied.errors)
        clone_ok=self._accepted(copied,states.active&(self.current.errors==0),states)
        self._assign(self.current,copied,clone_ok)
        self._release(copied,(copied.slots>=0)&~clone_ok,states,protected=(self.current,))
        for stage in range(3):
            self._assign(self.work,self.current)
            self.work.active.copy_(self.current.active&(self.current.errors==0))
            self.work.errors.copy_(self.current.errors)
            meta=self.rules.metadata_device(self.work)
            self.stage_errors[stage,0].copy_(meta.errors)
            self.current.errors.bitwise_or_(meta.errors)
            advance=self.work.active&(self.current.errors==0)&~meta.finished.bool()
            invalid=advance&((meta.actors<0)|(meta.actors>1))
            self.current.errors.bitwise_or_(invalid.to(t.int32)*METADATA_ERROR)
            advance&=self.current.errors==0
            self.work.active.copy_(advance);self.work.errors.copy_(self.current.errors)
            actors=meta.actors.clamp(0,1).contiguous()
            decision,chosen,decision_errors=self.teacher_actions.rule_decision_device(self.work,actors)
            self.stage_errors[stage,1].copy_(decision_errors)
            self.current.errors.bitwise_or_(decision_errors|decision.errors)
            self.work.active.copy_(advance&(self.current.errors==0))
            self.work.errors.copy_(self.current.errors)
            stepped=self._remember(self.rules.step_device(self.work,actors,decision,chosen,physical=False))
            self.stage_errors[stage,2].copy_(stepped.errors)
            self.current.errors.bitwise_or_(stepped.errors)
            step_ok=self._accepted(stepped,self.work.active,states)
            clean_input=DeviceStateBatch(self.pool,stepped.slots,stepped.generations,stepped.owners,
                                        step_ok,self.current.errors.clone())
            cleaned=self._remember(self.rules.clean_device(clean_input))
            self.stage_errors[stage,3].copy_(cleaned.errors)
            self.current.errors.bitwise_or_(cleaned.errors)
            clean_ok=self._accepted(cleaned,step_ok,states)
            action_errors=release_actions_device(decision,advance,states.owners,
                protected=(states,self.current,stepped,cleaned))
            self.stage_errors[stage,4].copy_(action_errors)
            self.current.errors.bitwise_or_((action_errors!=0).to(t.int32)*LEASE_ERROR)
            keep=clean_ok&(self.current.errors==0)
            self._release(self.current,keep,states,protected=(cleaned,))
            self._assign(self.current,cleaned,keep)
            self._release(stepped,stepped.slots>=0,states,protected=(self.current,))
            self._release(cleaned,(cleaned.slots>=0)&~keep,states,protected=(self.current,))
            self._release(self.current,(self.current.slots>=0)&(self.current.errors!=0),states)
            self.stage_errors[stage,5].copy_(self.current.errors)
            self._keepalive.extend((meta,decision,chosen,decision_errors,clean_input,action_errors))
        self.current.active.copy_(states.active&(self.current.errors==0)&(self.current.slots>=0))
        return self.current
