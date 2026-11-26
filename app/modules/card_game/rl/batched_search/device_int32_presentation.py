"""Collector-owned int32 storage for native parser candidate indices.

Keeps every candidate/selection/order slot and float32 column. The underlying
parser already emits int32; widened int64 buffers are unnecessary. Counts and
lease IDs stay int64. Any out-of-range integer fails the row before use.
"""
from dataclasses import replace
from ..batched_duel.device_rule_presentation import DeviceRulePresentation
from .compact_device_candidates import CompactDeviceCandidates

INDEX_FIELDS=('int_tokens','mulligan_cards','order_cards','select_cards')
RANGE_ERROR=159


def narrow_candidate_indices(tokens,*,allow_cpu_reference=False):
    import torch as t
    device=tokens.int_tokens.device
    if device.type!='cuda' and not (device.type=='cpu' and allow_cpu_reference is True):
        raise ValueError('CUDA required; CPU explicit storage oracle only')
    rows=tokens.int_tokens.shape[0];errors=t.zeros(rows,dtype=t.int32,device=device);values={}
    for name in INDEX_FIELDS:
        field=getattr(tokens,name)
        if field.device!=device or field.dtype not in (t.int32,t.int64) or field.shape[0]!=rows:
            raise ValueError('Bound device integer index arrays required')
        bad=((field<-(1<<31))|(field>(1<<31)-1)).reshape(rows,-1).any(-1)
        errors.bitwise_or_(bad.to(t.int32)*RANGE_ERROR)
        values[name]=field.to(t.int32)
    return replace(tokens,**values),errors


class _NarrowingProjection:
    def __init__(self,projection):self.projection=projection
    def __getattr__(self,name):return getattr(self.projection,name)
    def project_tokens_device(self,call,*,active):
        result=self.projection.project_tokens_device(call,active=active)
        candidates,errors=narrow_candidate_indices(result.candidates)
        return replace(result,candidates=candidates,errors=result.errors|errors)


class DeviceInt32Presentation(DeviceRulePresentation):
    """Same public plans and encodings; narrower index storage only."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.projection=_NarrowingProjection(self.projection)
    def _prepare(self,kind,action_store=None):
        plan=super()._prepare(kind,action_store)
        if kind=='decision':
            old=plan.candidates
            # Fresh buffers contain only parser sentinels (-1); no narrowing risk.
            tokens=replace(old.tokens,**{name:getattr(old.tokens,name).to(self.t.int32) for name in INDEX_FIELDS})
            plan.candidates=CompactDeviceCandidates(tokens,scratch_rows=old.scratch_rows,
                device=self.device,template_table=old.template_table)
        return plan
