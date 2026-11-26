"""Bounded numerical value-only sink for genuine zero-simulation games.

This accepts the GameLoop sink protocol but never emits policy/search labels.
It is a component, not a collector or driver qualification. Actual initial
states/actions, seed binding and rollout provenance remain driver obligations.
"""
from dataclasses import dataclass
from typing import Any

from ..batched_duel.device_rule_protocol import DeviceDecision, DeviceStateBatch
from ..batched_duel.device_graph_storage import CompactDeviceGraphStore


OWNER_ERROR, GEN_ERROR, STEP_ERROR, NONFINITE_ERROR = 1, 2, 4, 8
PURE_ERROR, CAPACITY_ERROR, FINISH_ERROR, MASK_ERROR = 16, 32, 64, 128
CAVEAT = ('Value-only numerical sink; zero simulations and no policy targets. '
          'Canonical terminal checks do not qualify the rollout driver or prove raw replay provenance.')


@dataclass(frozen=True)
class DevicePureValueTapeResult:
    record_count: Any
    learning_eligible: Any
    z: Any
    errors: Any
    winners: Any
    complete: Any
    finished: Any
    truncated: Any
    caveat: str = CAVEAT


class DevicePureValueTape:
    """One record per physical decision, storing both public viewer inputs."""
    supports_full_duel_rules = False
    gpu_qualified = False
    policy_targets = False
    simulations = 0

    def __init__(self, rules, *, rows, max_records, max_bytes, observation_dim,
                 max_actions=800, device='cuda', allow_cpu_reference=False, allow_synthetic=False):
        import torch
        if type(allow_cpu_reference) is not bool or type(allow_synthetic) is not bool:
            raise TypeError('Explicit boolean CPU/synthetic controls required')
        for name, value in (('rows', rows), ('max_records', max_records),
                ('max_bytes', max_bytes), ('observation_dim', observation_dim), ('max_actions', max_actions)):
            if type(value) is not int or not 1 <= value < 2**63:
                raise ValueError('Positive bounded '+name+' required')
        if rows > 4096 or max_actions > 16384 or max_records >= 2**31:
            raise ValueError('Bounded pure-value cohort/action/record dimensions required')
        for name in ('metadata_device', 'value_observation_device'):
            if not callable(getattr(rules, name, None)):
                raise ValueError('Device rules require '+name)
        self.t, self.device = torch, torch.device(device)
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('Actual CUDA required; no CPU fallback')
            if self.device.index is None:self.device = torch.device('cuda', torch.cuda.current_device())
        elif not (self.device.type == 'cpu' and allow_cpu_reference and allow_synthetic):
            raise ValueError('CPU requires explicit synthetic and reference controls')
        self.rules, self.rows, self.max_records = rules, rows, max_records
        self.observation_dim, self.max_actions, self.max_bytes = observation_dim, max_actions, max_bytes
        self.allow_synthetic = allow_synthetic
        n = max_records+1
        # All owned tensors, including the ignored write sentinel, are budgeted
        # before allocating anything on the device.
        layout = {
            '_row_ids': ((rows,), torch.int64, 0), '_record_ids': ((n,), torch.int64, 0),
            '_seats': ((2,), torch.int64, 0), '_sink': ((), torch.int64, max_records),
            '_count': ((), torch.int64, 0), '_owners': ((rows,), torch.int64, -1),
            '_initial_version': ((rows,), torch.int64, 0), '_initial_slots': ((rows,), torch.int64, -1),
            '_initial_gen': ((rows,), torch.int64, 0), '_expected_step': ((rows,), torch.int64, 0),
            '_job_seeds': ((rows,), torch.int64, 0), '_policy_ids': ((rows,2), torch.int64, 0),
            '_aux_ids': ((rows,2), torch.int64, 0), '_train_sides': ((rows,2), torch.bool, False),
            '_cohort': ((rows,), torch.bool, False), '_begun': ((rows,), torch.bool, False),
            '_game_errors': ((rows,), torch.int32, 0), '_game_winners': ((rows,), torch.int64, -2),
            '_game_steps': ((rows,), torch.int64, 0), '_finished': ((rows,), torch.bool, False),
            '_complete': ((rows,), torch.bool, False), '_truncated': ((rows,), torch.bool, False),
            '_rec_game': ((n,), torch.int64, 0), '_rec_owner': ((n,), torch.int64, -1),
            '_rec_seed': ((n,), torch.int64, 0), '_rec_actor': ((n,), torch.int64, 0),
            '_rec_step': ((n,), torch.int64, 0), '_rec_version': ((n,), torch.int64, 0),
            '_rec_slot': ((n,), torch.int64, -1), '_rec_generation': ((n,), torch.int64, 0),
            '_rec_errors': ((n,), torch.int32, 0), '_x': ((n,2,observation_dim), torch.float32, 0),
            '_ctx': ((n,2), torch.float32, 0), '_z': ((n,2), torch.float32, 0),
            '_eligible': ((n,2), torch.bool, False),
        }
        widths = {torch.int64:8, torch.int32:4, torch.float32:4, torch.bool:1}
        required = 0
        for shape, dtype, _ in layout.values():
            count = 1
            for size in shape:count *= size
            required += count*widths[dtype]
        if required > max_bytes:raise MemoryError('Pure value tape exceeds declared byte budget')
        self._tensor_names = tuple(layout)
        for name, (shape, dtype, fill) in layout.items():
            setattr(self, name, torch.full(shape, fill, dtype=dtype, device=self.device))
        self._row_ids.copy_(torch.arange(rows, device=self.device))
        self._record_ids.copy_(torch.arange(n, device=self.device))
        self._seats.copy_(torch.arange(2, device=self.device))
        self.allocated_bytes = sum(getattr(self,k).numel()*getattr(self,k).element_size() for k in layout)
        if self.allocated_bytes != required:raise AssertionError('Pure value allocation estimate drift')
        self._pool, self._ready = None, False

    def _capture(self):
        if self.device.type == 'cuda' and not self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Pure value tape device operations require CUDA graph capture')

    def _vector(self, value, dtype, shape):
        if (not isinstance(value,self.t.Tensor) or value.device != self.device or value.dtype != dtype
                or tuple(value.shape) != shape or not value.is_contiguous()):
            raise ValueError('Exact pure-value bound-device tensor metadata required')

    def _roots(self, roots):
        if not isinstance(roots,DeviceStateBatch):raise TypeError('Checked device roots required')
        for name,dtype in (('slots',self.t.int64),('generations',self.t.int64),('owners',self.t.int64),
                ('active',self.t.bool),('errors',self.t.int32)):
            self._vector(getattr(roots,name),dtype,(self.rows,))
        if self._pool is not None and roots.pool is not self._pool:
            raise ValueError('Pure value tape requires the same checked state arena')
        validate = getattr(roots.pool,'validate',None)
        if callable(validate):
            _,good = validate(roots,shared_reads=True)
            self._vector(good,self.t.bool,(self.rows,))
            return good
        if not self.allow_synthetic:raise ValueError('Checked state lease validator required')
        return roots.active & (roots.errors==0) & (roots.slots>=0) & (roots.generations>0)

    def _metadata(self, roots):
        meta = self.rules.metadata_device(roots)
        for name,dtype in (('actors',self.t.int64),('versions',self.t.int64),('errors',self.t.int32),
                ('finished',self.t.bool),('winners',self.t.int64)):
            self._vector(getattr(meta,name),dtype,(self.rows,))
        return meta

    def begin_device(self, roots, policy_ids, aux_ids, job_seeds, train_sides, active):
        """Begin/reset a cohort; seeds/policy IDs are nonnegative, unused aux IDs >= -1."""
        self._capture();t=self.t;valid=self._roots(roots)
        self._vector(policy_ids,t.int64,(self.rows,2));self._vector(aux_ids,t.int64,(self.rows,2))
        self._vector(job_seeds,t.int64,(self.rows,));self._vector(train_sides,t.bool,(self.rows,2))
        self._vector(active,t.bool,(self.rows,));meta=self._metadata(roots)
        same_owner=roots.owners[:,None]==roots.owners[None,:]
        same_lease=(roots.slots[:,None]==roots.slots[None,:])&(roots.generations[:,None]==roots.generations[None,:])
        duplicate=((same_owner|same_lease)&active[:,None]&active[None,:]
                   &(self._row_ids[:,None]!=self._row_ids[None,:])).any(-1)
        bad=active&(~valid|~roots.active|(roots.owners<0)|(roots.generations<=0)|duplicate)
        protocol=active&(meta.finished|(meta.actors<0)|(meta.actors>1)|(meta.versions<0))
        protocol |= active & ((job_seeds < 0) | (policy_ids < 0).any(-1) | (aux_ids < -1).any(-1))
        errors=t.where(active,roots.errors|meta.errors,0)|bad.to(t.int32)*OWNER_ERROR|protocol.to(t.int32)*GEN_ERROR
        self._pool=roots.pool;self._count.zero_();self._expected_step.zero_()
        self._cohort.copy_(active);self._begun.copy_(active&(errors==0));self._owners.copy_(roots.owners)
        self._initial_slots.copy_(roots.slots);self._initial_gen.copy_(roots.generations)
        self._initial_version.copy_(meta.versions);self._job_seeds.copy_(job_seeds)
        self._policy_ids.copy_(policy_ids);self._aux_ids.copy_(aux_ids);self._train_sides.copy_(train_sides)
        self._game_errors.copy_(errors);self._game_winners.fill_(-2);self._game_steps.zero_()
        self._finished.zero_();self._complete.zero_();self._truncated.zero_()
        self._eligible.zero_();self._z.zero_();self._rec_errors.zero_()
        self._ready=True
        return errors

    def append_device(self, roots, actors, steps, pure_result, active):
        """Accept a completed policy-only forward, with real zero search work."""
        self._capture()
        if not self._ready:raise RuntimeError('begin_device required before pure append')
        t=self.t;valid=self._roots(roots)
        for value in (actors,steps):self._vector(value,t.int64,(self.rows,))
        self._vector(active,t.bool,(self.rows,))
        decision=pure_result.decision
        if not isinstance(decision,DeviceDecision) and not self.allow_synthetic:
            raise TypeError('Actual device decision required for pure value capture')
        self._vector(decision.counts,t.int64,(self.rows,))
        self._vector(decision.valid,t.bool,(self.rows,self.max_actions))
        self._vector(decision.observations,t.float32,(self.rows,self.observation_dim))
        self._vector(decision.errors,t.int32,(self.rows,))
        for name,dtype,shape in (('actions',t.int64,(self.rows,)),('requested',t.int64,(self.rows,)),
                ('completed',t.int64,(self.rows,)),('errors',t.int32,(self.rows,)),
                ('complete',t.bool,(self.rows,)),('visits',t.int32,(self.rows,self.max_actions))):
            self._vector(getattr(pure_result,name),dtype,shape)
        states=decision.states
        if not isinstance(states,DeviceStateBatch) or states.pool is not roots.pool:
            raise ValueError('Pure decision must bind its actual state lease')
        for name,dtype in (('slots',t.int64),('generations',t.int64),('owners',t.int64),('active',t.bool),('errors',t.int32)):
            self._vector(getattr(states,name),dtype,(self.rows,))
        pool = decision.action_pool
        if (not isinstance(pool, CompactDeviceGraphStore) or pool.closed
                or pool.device != self.device):
            raise ValueError('Live bound-device compact action arena required')
        if getattr(self.rules, 'action_store', pool) is not pool:
            raise ValueError('Decision action arena differs from the rule executor')
        self._vector(decision.action_slots,t.int64,(self.rows,))
        self._vector(decision.action_generations,t.int64,(self.rows,))
        self._vector(decision.action_refs,t.int32,(self.rows,self.max_actions))
        # Read-only registry validation uses only admitted lanes. It neither
        # gathers graphs nor acquires/releases leases or clears prior errors.
        checking=active & valid & (roots.errors==0) & (decision.errors==0)
        action_slots,action_good,action_errors,_=pool._leases(
            decision.action_slots,decision.action_generations,states.owners,
            checking.contiguous(),live=True,shared_reads=True)
        meta=self._metadata(roots)
        # The actor input already belongs to the checked physical decision.
        # Reuse it and obtain only the other public viewer through existing rules.
        value=self.rules.value_observation_device(roots,(1-actors.clamp(0,1)).contiguous())
        for name,dtype,shape in (('observations',t.float32,(self.rows,self.observation_dim)),
                ('viewer_is_actor',t.float32,(self.rows,1)),('errors',t.int32,(self.rows,)),
                ('terminal',t.bool,(self.rows,)),('terminal_value',t.float64,(self.rows,))):
            self._vector(getattr(value,name),dtype,shape)
        counts=decision.counts;prefix=t.arange(self.max_actions,device=self.device)[None,:]<counts[:,None]
        legal=(counts>=1)&(counts<=self.max_actions)&(decision.valid==prefix).all(-1)
        legal&=(pure_result.actions>=0)&(pure_result.actions<counts)
        chosen=decision.action_refs.gather(1,pure_result.actions.clamp(0,self.max_actions-1)[:,None])[:,0]
        legal &= (chosen>=0) & (chosen.long()<pool.counts[action_slots,0])
        pure=(pure_result.requested==0)&(pure_result.completed==0)&(pure_result.visits==0).all(-1)&pure_result.complete
        owner=self._begun&valid&roots.active&(roots.errors==0)&(roots.owners==self._owners)&~self._finished
        generation=states.active&(states.errors==0)&(states.slots==roots.slots)&(states.generations==roots.generations)&(states.owners==roots.owners)
        generation&=(roots.generations>0)&(meta.actors==actors)&(actors>=0)&(actors<=1)&~meta.finished
        generation&=(value.viewer_is_actor[:,0]==0)&~value.terminal&(value.terminal_value==0)
        generation &= action_good
        initial=(steps!=0)|((roots.slots==self._initial_slots)&(roots.generations==self._initial_gen))
        step=(steps==self._expected_step)&(steps>=0)&(meta.versions==self._initial_version+steps)&initial
        finite=t.isfinite(decision.observations).all(-1)&t.isfinite(value.observations).all(-1)
        finite&=t.isfinite(value.viewer_is_actor).all(-1)&t.isfinite(value.terminal_value)
        incoming=roots.errors|meta.errors|decision.errors|pure_result.errors|value.errors|self._game_errors|action_errors
        errors=t.where(active,incoming,0)
        for ok,code in ((owner,OWNER_ERROR),(generation,GEN_ERROR),(step,STEP_ERROR),
                (finite,NONFINITE_ERROR),(pure,PURE_ERROR),(legal,MASK_ERROR)):
            errors.bitwise_or_((active&~ok).to(t.int32)*code)
        admitted=active&(errors==0);fits=self._count+admitted.long().sum()<=self.max_records
        errors.bitwise_or_((admitted&~fits).to(t.int32)*CAPACITY_ERROR)
        commit=admitted&fits;rank=commit.long().cumsum(0)-commit.long()
        destination=t.where(commit,self._count+rank,self._sink).clamp(0,self.max_records)
        data={'_rec_game':self._row_ids,'_rec_owner':roots.owners,'_rec_seed':self._job_seeds,
              '_rec_actor':actors,'_rec_step':steps,'_rec_version':meta.versions,
              '_rec_slot':roots.slots,'_rec_generation':roots.generations,'_rec_errors':errors}
        for name,source in data.items():getattr(self,name).index_copy_(0,destination,source)
        own=actors==0
        x=t.stack((t.where(own[:,None],decision.observations,value.observations),
                   t.where(own[:,None],value.observations,decision.observations)),dim=1)
        ctx=(actors[:,None]==self._seats[None,:]).to(t.float32)
        self._x.index_copy_(0,destination,x);self._ctx.index_copy_(0,destination,ctx)
        self._count.add_(commit.long().sum());self._expected_step.add_(commit.long())
        self._game_errors.bitwise_or_(errors)
        return errors

    def finish_device(self, roots, winners, steps, complete, errors):
        self._capture()
        if not self._ready:raise RuntimeError('begin_device required before pure finish')
        t=self.t;valid=self._roots(roots)
        for value in (winners,steps):self._vector(value,t.int64,(self.rows,))
        self._vector(complete,t.bool,(self.rows,));self._vector(errors,t.int32,(self.rows,))
        meta=self._metadata(roots);active=roots.active
        owner=self._begun&valid&(roots.owners==self._owners)
        step=(steps==self._expected_step)&(steps>=0)&(meta.versions==self._initial_version+steps)
        terminal=meta.finished&(winners==meta.winners)&(winners>=-1)&(winners<=1)&(steps>0)
        finish=t.where(active,roots.errors|meta.errors,0)
        finish.bitwise_or_((active&~owner).to(t.int32)*OWNER_ERROR)
        finish.bitwise_or_((active&~step).to(t.int32)*STEP_ERROR)
        finish.bitwise_or_((active&complete&~terminal).to(t.int32)*FINISH_ERROR)
        finish.bitwise_or_((active&self._finished).to(t.int32)*FINISH_ERROR)
        combined=finish|t.where(active,errors,0)
        self._game_errors.bitwise_or_(combined)
        self._finished.logical_or_(active)
        qualifies=active&complete&terminal&(self._game_errors==0)
        self._complete.logical_or_(qualifies)
        self._truncated.logical_or_(active&~complete)
        self._game_winners.copy_(t.where(active,winners,self._game_winners))
        self._game_steps.copy_(t.where(active,steps,self._game_steps))
        game=self._rec_game.clamp(0,self.rows-1);occupied=self._record_ids<self._count
        valid_records=occupied&self._complete[game]&(self._game_errors[game]==0)&(self._rec_errors==0)
        winner=self._game_winners[game]
        z=t.where(winner[:,None]==-1,0.,t.where(winner[:,None]==self._seats[None,:],1.,-1.))
        self._z.copy_(t.where(valid_records[:,None],z,0.))
        self._eligible.copy_(valid_records[:,None]&self._train_sides[game])
        return t.where(active,finish,0)

    def result_device(self):
        if not self._ready:raise RuntimeError('No pure-value cohort initialized')
        valid=self._game_errors[self._rec_game.clamp(0,self.rows-1)]==0
        return DevicePureValueTapeResult(self._count,self._eligible&valid[:,None],
            self.t.where(valid[:,None],self._z,0.),self._game_errors,self._game_winners,
            self._complete&(self._game_errors==0)&~self._truncated,self._finished,self._truncated)

    def value_rows_after_batch(self, *, include_frozen=False):
        """Export only after the entire begun cohort finishes naturally/error-free."""
        if type(include_frozen) is not bool:raise TypeError('Explicit frozen-viewer audit control required')
        if not self._ready:raise RuntimeError('No pure-value cohort initialized')
        t=self.t
        if self.device.type=='cuda':
            if t.cuda.is_current_stream_capturing():raise RuntimeError('Host export forbidden inside CUDA capture')
            t.cuda.synchronize()
        acceptable=~self._cohort|(self._finished&self._complete&~self._truncated&(self._game_errors==0))
        if not bool(acceptable.all().item()):
            raise RuntimeError('Only an entirely complete error-free natural batch may export pure value rows')
        count=self._count.item()
        arrays={name:getattr(self,name)[:count].detach().cpu() for name in (
            '_x','_ctx','_z','_eligible','_rec_owner','_rec_seed','_rec_actor','_rec_step','_rec_version',
            '_rec_slot','_rec_generation','_rec_errors')}
        if bool((arrays['_rec_errors']!=0).any().item()):raise RuntimeError('Invalid pure value record')
        rows=[]
        for record in range(count):
            for viewer in (0,1):
                eligible=bool(arrays['_eligible'][record,viewer])
                if not include_frozen and not eligible:continue
                rows.append(dict(x=arrays['_x'][record,viewer].numpy().copy(),
                    value_actor=float(arrays['_ctx'][record,viewer]),z=float(arrays['_z'][record,viewer]),
                    actor=viewer,viewer=viewer,physical_actor=int(arrays['_rec_actor'][record]),
                    step=int(arrays['_rec_step'][record]),version=int(arrays['_rec_version'][record]),
                    job_seed=int(arrays['_rec_seed'][record]),owner=int(arrays['_rec_owner'][record]),
                    slot=int(arrays['_rec_slot'][record]),generation=int(arrays['_rec_generation'][record]),
                    learning_eligible=eligible,pure_value=True,caveat=CAVEAT))
        return rows
