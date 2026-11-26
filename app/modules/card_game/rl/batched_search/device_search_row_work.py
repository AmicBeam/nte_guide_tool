"""Bounded device queues for full candidate NN, SHA and value evaluation.

All models, feature equations and exact ordered candidate encodings are reused.
Each operation captures one fixed-size body, independent of total root count.
No CPU rules, host row scheduling, candidate truncation or readiness receipt.
"""
from dataclasses import fields
from ..batched_duel import gpu_seven_candidates as encoder
from .compact_device_candidates import CompactDeviceCandidates
from .device_inference import DeviceInferenceResult
from .device_info_key import DeviceKeyResult
from .device_tensor_row_queue import DeviceTensorRowQueue


class _WorkPlan:
    def __init__(self,owner,kind):
        self.owner=owner;self.kind=kind;t=owner.t;b=owner.rows;m=owner.max_actions;d=owner.device
        self.queue=DeviceTensorRowQueue(rows=b,scratch_rows=owner.tile,seconds=owner.seconds,device=d)
        self.captured=False
        self.logits=t.zeros((b+1,m),dtype=t.float32,device=d) if kind=='policy' else None
        self.wdl=t.zeros((b+1,3),dtype=t.float32,device=d) if kind=='policy' else None
        self.invalid=t.zeros(b+1,dtype=t.bool,device=d) if kind!='hash' else None
        self.keys=t.zeros((b+1,8),dtype=t.int64,device=d) if kind=='hash' else None
        self.errors=t.zeros(b+1,dtype=t.int32,device=d) if kind=='hash' else None
        self.values=t.zeros(b+1,dtype=t.float64,device=d) if kind=='value' else None
        self.empty_candidates=t.empty((owner.tile,0,owner.auxiliary.candidate_dim),dtype=t.float32,device=d)
        self.empty_valid=t.empty((owner.tile,0),dtype=t.bool,device=d)
        self.value_only=t.ones(owner.tile,dtype=t.bool,device=d)
    def close(self):self.queue.close()
    def _encode(self,candidates,ids,on):
        raw=encoder.RawCandidateTokens(**{f.name:getattr(candidates.tokens,f.name).index_select(0,ids)
            for f in fields(encoder.RawCandidateTokens) if f.name!='request_ids'},request_ids=())
        return encoder.encode_tokens_device(raw,device=self.owner.device,active=on,template_table=candidates.template_table)
    def capture(self,active,*,decision=None,ids=None,context=None,observed=None):
        o=self.owner;t=o.t;b=o.rows
        if self.captured:raise RuntimeError('Row work site already frozen')
        if active.device!=o.device or active.dtype!=t.bool or active.shape!=(b,) or not active.is_contiguous():
            raise ValueError('Exact contiguous CUDA row mask required')
        if self.kind!='value':
            if type(decision.candidates) is not CompactDeviceCandidates or decision.candidates.shape[:2]!=(b,o.max_actions):
                raise TypeError('Complete compact candidate rows required')
        for name in ('logits','wdl','invalid','keys','errors','values'):
            output=getattr(self,name)
            if output is not None:output.zero_()
        def operation(rows,on,destinations):
            if self.kind=='value':
                prediction=o.auxiliary.predict_device(observed.observations.index_select(0,rows),
                    self.empty_candidates,self.empty_valid,ids.index_select(0,rows),
                    observed.viewer_is_actor.index_select(0,rows),value_only=self.value_only)
                value=(prediction.wdl[:,2]-prediction.wdl[:,0]).to(t.float64)
                self.values.index_copy_(0,destinations,t.where(on,value,0.))
                self.invalid.index_copy_(0,destinations,on&prediction.invalid_rows)
                return
            packed,encoding_errors=self._encode(decision.candidates,rows,on)
            x=decision.observations.index_select(0,rows)
            if self.kind=='policy':
                valid=decision.valid.index_select(0,rows)
                mismatch=on&(valid!=packed.mask).any(-1)
                mismatch|=on&(valid!=decision.candidates.tokens.mask.index_select(0,rows)).any(-1)
                prediction=o.inference.predict_device(x,packed.candidates,valid,
                    ids.index_select(0,rows),context.index_select(0,rows))
                usable=on&(encoding_errors==0)&~mismatch&~prediction.invalid_rows
                self.logits.index_copy_(0,destinations,t.where(usable[:,None],prediction.logits,0.))
                self.wdl.index_copy_(0,destinations,t.where(usable[:,None],prediction.wdl,0.))
                self.invalid.index_copy_(0,destinations,on&~usable)
            else:
                counts=decision.counts.index_select(0,rows)
                hashed=o.information_keys.hash(x.contiguous(),packed.candidates.contiguous(),counts.contiguous(),on.contiguous())
                errors=hashed.errors
                errors=t.where((encoding_errors!=0)&(errors==0),encoding_errors,errors)
                mismatch=on&(counts!=decision.candidates.tokens.counts.index_select(0,rows))
                errors=t.where(mismatch&(errors==0),1,errors)
                self.keys.index_copy_(0,destinations,t.where((errors!=0)[:,None],0,hashed.keys))
                self.errors.index_copy_(0,destinations,errors.to(t.int32))
        guard=self.queue.capture_device(active,operation)
        self.captured=True
        bad=active&(guard.reshape(-1)[0]!=0)
        if self.kind=='hash':
            self.errors[:b].bitwise_or_(bad.to(t.int32))
            return DeviceKeyResult(self.keys[:b],self.errors[:b])
        self.invalid[:b].logical_or_(bad)
        if self.kind=='value':return self.values[:b],self.invalid[:b]
        return DeviceInferenceResult(self.logits[:b],self.wdl[:b],self.invalid[:b])


class DeviceSearchRowWork:
    def __init__(self,inference,auxiliary,information_keys,*,rows,max_actions,tile=2,seconds=120,policy_sites=2):
        import torch as t
        if any(type(v)is not int or v<1 for v in (rows,max_actions,tile)):
            raise ValueError('Positive bounded row work dimensions required')
        if type(policy_sites) is not int or not 1<=policy_sites<=3:
            raise ValueError('One to three static policy capture sites required')
        self.t=t;self.inference=inference;self.auxiliary=auxiliary;self.information_keys=information_keys
        self.device=inference.device;self.rows=rows;self.max_actions=max_actions;self.tile=tile;self.seconds=seconds
        if self.device.type!='cuda' or self.device!=auxiliary.device or self.device!=information_keys.device:
            raise ValueError('Same actual CUDA numerical backends required')
        if t.cuda.is_current_stream_capturing():raise RuntimeError('Prepare row work before capture')
        self.plans={};self.indices={};self.closed=False
        try:
            for kind,count in (('policy',policy_sites),('hash',3),('value',3)):
                self.plans[kind]=[];self.indices[kind]=0
                for _ in range(count):self.plans[kind].append(_WorkPlan(self,kind))
            # Each native child stream needs actual numerical warmup before capture.
            x=t.zeros((tile,inference.feature_dim),dtype=t.float32,device=self.device)
            c=t.zeros((tile,max_actions,inference.candidate_dim),dtype=t.float32,device=self.device)
            mask=t.zeros((tile,max_actions),dtype=t.bool,device=self.device);mask[:,0]=True
            ids=t.zeros(tile,dtype=t.int64,device=self.device);context=t.ones((tile,1),dtype=t.float32,device=self.device)
            for plans in self.plans.values():
                for plan in plans:
                    stream=plan.queue.loop.body_stream;stream.wait_stream(t.cuda.current_stream(self.device))
                    with t.cuda.stream(stream):
                        if plan.kind=='hash':information_keys.hash(x,c,t.ones(tile,dtype=t.int64,device=self.device),t.ones(tile,dtype=t.bool,device=self.device))
                        elif plan.kind=='policy':inference.predict_device(x,c,mask,ids,context)
                        else:auxiliary.predict_device(x,plan.empty_candidates,plan.empty_valid,ids,context,value_only=plan.value_only)
                    stream.synchronize()
        except BaseException:self.close();raise
    def _consume(self,kind):
        if self.closed:raise RuntimeError('Row work closed')
        index=self.indices[kind]
        if index>=len(self.plans[kind]):raise RuntimeError('Row work capture quota exhausted: '+kind)
        self.indices[kind]=index+1;return self.plans[kind][index]
    def policy(self,decision,ids,context,active):return self._consume('policy').capture(active,decision=decision,ids=ids,context=context)
    def hash(self,decision,active):return self._consume('hash').capture(active,decision=decision)
    def value(self,observed,ids,active):return self._consume('value').capture(active,observed=observed,ids=ids)
    def close(self):
        if not self.closed:
            for plans in self.plans.values():
                for plan in plans:plan.close()
            self.closed=True
