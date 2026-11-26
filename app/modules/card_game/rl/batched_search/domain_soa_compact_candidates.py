"""Exact domain candidate snapshots with one shared bounded dense workspace.

All 1576 columns and every candidate remain present. Root/current decisions
retain public numeric descriptors, while predict/hash materialize 32 rows at a
time. No full [400,800,1576] root/current matrices are retained.
"""
from types import SimpleNamespace
from ..batched_duel.domain_soa_candidates import DomainSoACandidates
from .compact_device_candidates import CompactDeviceCandidates


class DomainCandidateWorkspace:
    def __init__(self,*,rows,actions,cards,selection_width,device='cuda',allow_cpu_oracle=False):
        import torch
        t=torch;self.t=t;self.rows=rows;self.device=t.device(device)
        if self.device.type=='cuda'and self.device.index is None:self.device=t.device('cuda',t.cuda.current_device())
        self.encoder=DomainSoACandidates(device=self.device,allow_cpu_oracle=allow_cpu_oracle)
        self.inputs={
            'cards':t.zeros((rows,cards,13),dtype=t.float64,device=self.device),
            'actions':t.full((rows,actions,6),-1,dtype=t.int64,device=self.device),
            'selected':t.full((rows,actions,selection_width),-1,dtype=t.int64,device=self.device),
            'top':t.full((rows,actions,4),-1,dtype=t.int64,device=self.device),
            'bottom':t.full((rows,actions,4),-1,dtype=t.int64,device=self.device),
            'counts':t.zeros(rows,dtype=t.int64,device=self.device),
            'viewers':t.zeros(rows,dtype=t.int64,device=self.device),
            'turn_counts':t.zeros(rows,dtype=t.int64,device=self.device),
            'active':t.zeros(rows,dtype=t.bool,device=self.device),
            'prior_errors':t.zeros(rows,dtype=t.int32,device=self.device),
        }
        self.prepared=self.encoder.prepare(**self.inputs)
    def close(self):self.encoder.close()


class DomainCompactCandidates(CompactDeviceCandidates):
    """Compact field snapshots implementing the existing search candidate API."""
    def __init__(self,fields,workspace,*,clone=True):
        self.t=workspace.t;self.device=workspace.device;self.workspace=workspace
        self.fields={name:value.clone(memory_format=self.t.contiguous_format)if clone else value for name,value in fields.items()}
        expected=set(workspace.inputs)|{'mask'}
        if set(self.fields)!=expected:raise ValueError('Complete public domain candidate descriptors required')
        self._rows,self._max_candidates=self.fields['actions'].shape[:2]
        self.scratch_rows=workspace.rows;self._template_table=None;self.request_ids=()
        for name,value in self.fields.items():
            if value.device!=self.device or value.shape[0]!=self._rows:raise ValueError('Resident aligned candidate snapshot required')
        self.tokens=SimpleNamespace(mask=self.fields['mask'],counts=self.fields['counts'])
    @property
    def allocated_bytes(self):return sum(v.numel()*v.element_size()for v in self.fields.values())
    @property
    def template_allocated_bytes(self):return 0
    def clone(self):return DomainCompactCandidates(self.fields,self.workspace,clone=True)
    def copy_masked_device(self,other,mask):
        t=self.t
        if type(other)is not DomainCompactCandidates or other.workspace is not self.workspace or other.shape!=self.shape:raise ValueError('Bound compact domain snapshots required')
        self._require_tensor(mask,t.bool,(self._rows,),'active')
        for name,dst in self.fields.items():
            src=other.fields[name]
            if src.shape!=dst.shape or src.dtype!=dst.dtype:raise ValueError('Stable public descriptor layout required')
            t.where(mask.reshape((self._rows,)+(1,)*(dst.ndim-1)),src,dst,out=dst)
        return t.zeros(self._rows,dtype=t.int32,device=self.device)
    def encode_slice_device(self,start,stop,active):
        if type(start)is not int or type(stop)is not int or not 0<=start<stop<=self._rows or stop-start>self.scratch_rows:raise ValueError('Bounded static candidate row slice required')
        local=stop-start;self._require_tensor(active,self.t.bool,(local,),'active')
        w=self.workspace
        for name,dst in w.inputs.items():
            if name=='active':
                dst.zero_();dst[:local].copy_(self.fields[name][start:stop]&active)
            else:dst[:local].copy_(self.fields[name][start:stop])
        out=w.encoder.launch(w.prepared)
        return SimpleNamespace(candidates=out['features'][:local],mask=out['valid'][:local]),out['errors'][:local]
