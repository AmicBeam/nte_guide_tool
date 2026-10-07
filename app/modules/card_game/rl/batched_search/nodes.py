"""Bounded batched Gumbel statistics, with the same FP64 formulas as NumPy.

The pool stores statistics rather than complete game states. One scheduler owns
it and all CUDA operations must use the stream active at construction. Rules,
observations, hidden worlds and node identity hashing belong to the backend.
"""
from dataclasses import dataclass
import heapq
import math
from uuid import uuid4
import numpy as np


@dataclass(frozen=True)
class NodeRef:
    pool_id: str
    slot: int
    generation: int
    owner: str


class BatchedNodePool:
    def __init__(self, capacity, max_actions, *, device='cpu', q_scale='natural_wdl'):
        import torch
        if type(capacity) is not int or type(max_actions) is not int or min(capacity,max_actions)<1:
            raise ValueError('Positive node and full-action capacities required')
        if q_scale not in ('legacy','natural_wdl'):
            raise ValueError('Explicit completed-Q scale required')
        self.capacity=capacity;self.max_actions=max_actions;self.q_scale=q_scale
        self.device=torch.device(device)
        if self.device.type not in ('cpu','cuda'):
            raise ValueError('Only CPU oracle and CUDA devices supported')
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no implicit fallback')
        self._torch=torch;self._id=uuid4().hex
        self._stream=torch.cuda.current_stream(self.device) if self.device.type=='cuda' else None
        self._prior=torch.zeros((capacity,max_actions),dtype=torch.float64,device=self.device)
        self._total=torch.zeros_like(self._prior)
        self._visits=torch.zeros((capacity,max_actions),dtype=torch.int32,device=self.device)
        self._value=torch.zeros(capacity,dtype=torch.float64,device=self.device)
        self._count=torch.zeros(capacity,dtype=torch.int64,device=self.device)
        self._generation=[1]*capacity;self._owner=[None]*capacity;self._host_count=[0]*capacity
        self._free=list(range(capacity));self._active=0

    @property
    def allocated_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self._prior,self._total,self._visits,self._value,self._count))

    @property
    def active_nodes(self):return self._active

    def _stream_check(self):
        if self._stream is not None and self._torch.cuda.current_stream(self.device)!=self._stream:
            raise RuntimeError('Node pool operations require its owning CUDA stream')

    def _slot_numbers(self,refs):
        self._stream_check();slots=[]
        for ref in refs:
            if (not isinstance(ref,NodeRef) or ref.pool_id!=self._id
                    or type(ref.slot) is not int or type(ref.generation) is not int
                    or not 0<=ref.slot<self.capacity
                    or self._generation[ref.slot]!=ref.generation
                    or self._owner[ref.slot]!=ref.owner or not isinstance(ref.owner,str)):
                raise ValueError('Foreign, stale or malformed node reference')
            slots.append(ref.slot)
        if not slots:raise ValueError('Nonempty node batch required')
        return slots

    def _slots(self,refs):
        return self._torch.tensor(self._slot_numbers(refs),dtype=self._torch.long,device=self.device)

    def allocate(self,priors,values,owners):
        self._stream_check();priors=list(priors);values=list(values);owners=list(owners)
        if not priors or len(priors)!=len(values) or len(priors)!=len(owners):
            raise ValueError('Matching nonempty node initialization required')
        if len(priors)>len(self._free):raise OverflowError('Node capacity exhausted')
        validated=[]
        for prior,value,owner in zip(priors,values,owners):
            p=np.asarray(prior,dtype=np.float64)
            if (p.ndim!=1 or not 1<=len(p)<=self.max_actions or not np.isfinite(p).all()
                    or (p<0).any() or not np.isclose(p.sum(),1.,atol=1e-12,rtol=1e-10)):
                raise ValueError('Finite normalized full legal-action prior required')
            if isinstance(value,bool) or not isinstance(value,(int,float,np.floating)) or not math.isfinite(value):
                raise ValueError('Finite node value required')
            if self.q_scale=='natural_wdl' and abs(value)>1+1e-6:
                raise ValueError('Natural WDL outside [-1,1]')
            if not isinstance(owner,str) or not owner:raise ValueError('Root owner identity required')
            validated.append((p,float(value),owner))
        slots=[heapq.heappop(self._free) for _ in validated]
        t=self._torch;idx=t.tensor(slots,device=self.device)
        padded=np.zeros((len(slots),self.max_actions),dtype=np.float64)
        for i,(p,_,_) in enumerate(validated):padded[i,:len(p)]=p
        self._prior[idx]=t.as_tensor(padded,device=self.device)
        self._visits[idx]=0;self._total[idx]=0
        self._value[idx]=t.tensor([v for _,v,_ in validated],dtype=t.float64,device=self.device)
        self._count[idx]=t.tensor([len(p) for p,_,_ in validated],device=self.device)
        refs=[]
        for slot,(p,_,owner) in zip(slots,validated):
            self._owner[slot]=owner;self._host_count[slot]=len(p);refs.append(NodeRef(self._id,slot,self._generation[slot],owner))
        self._active+=len(refs)
        return refs

    def release(self,refs):
        refs=list(refs);idx=self._slots(refs)
        if len(set(ref.slot for ref in refs))!=len(refs):raise ValueError('Duplicate node release')
        self._prior[idx]=0;self._total[idx]=0;self._visits[idx]=0;self._value[idx]=0;self._count[idx]=0
        for ref in refs:
            self._owner[ref.slot]=None;self._host_count[ref.slot]=0;self._generation[ref.slot]+=1;heapq.heappush(self._free,ref.slot)
        self._active-=len(refs)

    def _rows(self,refs,*,compact=False):
        idx=self._slots(refs);t=self._torch
        width=max(self._host_count[ref.slot] for ref in refs) if compact else self.max_actions
        prior=self._prior[:,:width].index_select(0,idx);visits=self._visits[:,:width].index_select(0,idx)
        total=self._total[:,:width].index_select(0,idx);value=self._value.index_select(0,idx)
        count=self._count.index_select(0,idx)
        valid=t.arange(width,device=self.device)[None,:]<count[:,None]
        return idx,prior,visits,total,value,valid

    def completed_q(self,refs):
        refs=tuple(refs);_,prior,visits,total,value,valid=self._rows(refs)
        return self._completed_q(prior,visits,total,value,valid)

    def _completed_q(self,prior,visits,total,value,valid):
        t=self._torch
        seen=(visits>0)&valid
        q=total/visits.clamp_min(1)
        mass=(prior*seen).sum(-1)
        mean=t.where(mass>0,(prior*seen*q).sum(-1)/mass.clamp_min(1e-300),value)
        visit_sum=visits.sum(-1)
        mixed=(value+visit_sum*mean)/(1+visit_sum)
        q=t.where(seen,q,mixed[:,None])
        gain=.1*(50+visits.max(-1).values.to(t.float64))
        if self.q_scale=='natural_wdl':
            if bool((q[valid].abs()>1+1e-6).any()):raise ValueError('Natural WDL estimate outside [-1,1]')
        else:
            low=q.masked_fill(~valid,float('inf')).min(-1).values
            high=q.masked_fill(~valid,-float('inf')).max(-1).values
            q=(q-low[:,None])/(high-low).clamp_min(1e-8)[:,None]
        return (q*gain[:,None]).masked_fill(~valid,0.)

    def policy(self,refs):
        refs=tuple(refs);_,prior,visits,total,value,valid=self._rows(refs)
        return (prior.clamp_min(1e-300).log()+self._completed_q(prior,visits,total,value,valid)).masked_fill(~valid,-float('inf')).softmax(-1)

    def select_root(self,refs,noise,visit_targets):
        refs=tuple(refs);t=self._torch;_,prior,visits,total,value,valid=self._rows(refs,compact=True)
        noise=t.as_tensor(noise,dtype=t.float64,device=self.device)
        targets=t.as_tensor(visit_targets,device=self.device)
        if noise.shape not in ((len(refs),self.max_actions),tuple(prior.shape)) or targets.shape!=(len(refs),) or targets.dtype not in (t.int32,t.int64):
            raise ValueError('Root noise and integer visit targets must match batch')
        if not bool(t.isfinite(noise).all()) or bool((targets<0).any()):raise ValueError('Invalid root selection inputs')
        eligible=(visits==targets[:,None])&valid
        if not bool(eligible.any(-1).all()):raise ValueError('Invalid sequential-halving visit schedule')
        score=prior.clamp_min(1e-300).log()+noise[:,:prior.shape[1]]+self._completed_q(prior,visits,total,value,valid)
        return score.masked_fill(~eligible,-float('inf')).argmax(-1)

    def select_interior(self,refs):
        refs=tuple(refs)
        _,prior,visits,total,value,valid=self._rows(refs,compact=True)
        policy=(prior.clamp_min(1e-300).log()+self._completed_q(prior,visits,total,value,valid)).masked_fill(~valid,-float('inf')).softmax(-1)
        scores=policy-visits.to(self._torch.float64)/(1+visits.sum(-1)[:,None])
        return scores.masked_fill(~valid,-float('inf')).argmax(-1)

    def final_summary(self,refs,noise):
        refs=tuple(refs);t=self._torch
        _,prior,visits,total,value,valid=self._rows(refs,compact=True)
        noise=t.as_tensor(noise,dtype=t.float64,device=self.device)
        if noise.shape!=prior.shape or not bool(t.isfinite(noise).all()):raise ValueError('Compact final noise shape mismatch')
        eligible=(visits==visits.max(-1).values[:,None])&valid
        score=prior.clamp_min(1e-300).log()+noise+self._completed_q(prior,visits,total,value,valid)
        selected=score.masked_fill(~eligible,-float('inf')).argmax(-1)
        return t.stack((selected,((visits>0)&valid).sum(-1)),dim=1)

    def backup(self,paths,values):
        self._apply_backup(self._backup_rows(paths,values))

    def _backup_rows(self,paths,values):
        paths=[list(path) for path in paths];values=list(values)
        if len(paths)!=len(values):raise ValueError('One terminal/leaf value per completed path required')
        flat=[];owners=set()
        for path,value in zip(paths,values):
            if isinstance(value,bool) or not isinstance(value,(int,float,np.floating)) or not math.isfinite(value):
                raise ValueError('Finite backup value required')
            if self.q_scale=='natural_wdl' and abs(value)>1+1e-6:raise ValueError('Natural WDL backup outside [-1,1]')
            if not path:continue
            refs=[ref for ref,_ in path];self._slot_numbers(refs)
            owner=refs[0].owner
            if owner in owners or any(ref.owner!=owner for ref in refs):
                raise ValueError('Only one completed simulation per root in a backup batch')
            owners.add(owner)
            for ref,action in path:
                if type(action) is not int or not 0<=action<self._host_count[ref.slot]:raise ValueError('Integer full legal action index required')
                # This scalar check is batched below; it is host metadata only.
                flat.append((ref,action,float(value)))
        return flat

    def _apply_backup(self,flat):
        if not flat:return
        t=self._torch;idx=self._slots([ref for ref,_,_ in flat])
        actions=t.tensor([a for _,a,_ in flat],device=self.device)
        keys=idx*self.max_actions+actions
        self._visits.view(-1).index_add_(0,keys,t.ones(len(flat),dtype=t.int32,device=self.device))
        self._total.view(-1).index_add_(0,keys,t.tensor([v for _,_,v in flat],dtype=t.float64,device=self.device))

    def backup_admitted(self,paths,values,deadlines,clock):
        """Back up completed work; roll back rows whose device completion is late."""
        paths=list(paths);values=list(values);deadlines=list(deadlines)
        if len(paths)!=len(deadlines) or len(values)!=len(paths):raise ValueError('Backup deadline count mismatch')
        if any(d is not None and (isinstance(d,bool) or not isinstance(d,(int,float)) or not math.isfinite(d)) for d in deadlines):
            raise ValueError('Finite absolute backup deadlines required')
        # Validate the entire request before any device mutation.
        self._backup_rows(paths,values)
        now=clock();accepted=[d is None or now<d for d in deadlines]
        indices=[i for i,ok in enumerate(accepted) if ok]
        admitted=[paths[i] for i in indices]
        flat=self._backup_rows(admitted,[values[i] for i in indices])
        owners={paths[i][0][0].owner:i for i in indices if paths[i]}
        unique={ref.slot*self.max_actions+action:owners[ref.owner] for ref,action,_ in flat}
        if not unique:return accepted
        t=self._torch;keys=t.tensor(list(unique),device=self.device,dtype=t.int64)
        old_visits=self._visits.view(-1).index_select(0,keys)
        old_total=self._total.view(-1).index_select(0,keys)
        self._apply_backup(flat)
        if self.device.type=='cuda':
            event=t.cuda.Event();event.record(self._stream);event.synchronize()
        finished=clock()
        late=[i for i in indices if deadlines[i] is not None and finished>=deadlines[i]]
        if late:
            late_set=set(late);mask=t.tensor([owner in late_set for owner in unique.values()],device=self.device,dtype=t.bool)
            restore=keys[mask]
            self._visits.view(-1)[restore]=old_visits[mask]
            self._total.view(-1)[restore]=old_total[mask]
            for i in late:accepted[i]=False
        return accepted

    def snapshot(self,refs):
        _,prior,visits,total,value,valid=self._rows(refs)
        return dict(prior=prior,visits=visits,total=total,value=value,valid=valid)

    def set_raw_value(self,refs,values):
        refs=list(refs);values=list(values)
        if len(refs)!=len(values):raise ValueError('Matching node/value updates required')
        if any(isinstance(v,bool) or not isinstance(v,(int,float,np.floating)) or not math.isfinite(v) for v in values):
            raise ValueError('Finite node values required')
        if self.q_scale=='natural_wdl' and any(abs(v)>1+1e-6 for v in values):
            raise ValueError('Natural WDL estimate outside [-1,1]')
        if len({ref.slot for ref in refs})!=len(refs):raise ValueError('Duplicate raw-value update')
        indices=self._slots(refs)
        self._value[indices]=self._torch.tensor(values,dtype=self._torch.float64,device=self.device)
