"""Batched device statistics queries with root-owned, bounded node lifetimes."""
from collections import Counter
import queue
import time
from uuid import uuid4
import weakref
import numpy as np
from .nodes import BatchedNodePool,NodeRef
from .cooperative import _operation,OperationQuery,OperationResponse


class _Owner:
    def __init__(self):self.id=uuid4().hex


class BatchedStatistics:
    read_only_finalization=True
    def __init__(self,capacity,max_actions,*,device='cuda',q_scale='natural_wdl',clock=time.monotonic):
        self.pool=BatchedNodePool(capacity,max_actions,device=device,q_scale=q_scale)
        self.q_scale=q_scale;self.clock=clock;self._owners={};self._counts={};self._releases=queue.SimpleQueue()
        self._closed=False;self._operations=Counter();self._wall=0.;self._peak=0
        self._completed_backups=0;self._late_backups=0

    def new_owner(self):
        self._check();owner=_Owner();self._owners[owner.id]=[]
        weakref.finalize(owner,self._releases.put,owner.id)
        return owner

    def _check(self):
        if self._closed:raise RuntimeError('Device statistics are closed')
        self.pool._stream_check()

    def _flush(self):
        self._check()
        released=[]
        while True:
            try:owner=self._releases.get_nowait()
            except queue.Empty:break
            refs=self._owners.pop(owner,[])
            released.extend(refs)
        if released:
            self.pool.release(released)
            for ref in released:self._counts.pop(ref,None)

    def create(self,prior,value,owner):return _operation(self,'create',prior,value,owner.id)
    def root_choice(self,node,noise,visit):return _operation(self,'root_choice',node,noise,visit)
    def interior_choice(self,node):return _operation(self,'interior_choice',node)
    def backup(self,path,value,deadline=None):return _operation(self,'backup',path,value,deadline)
    def snapshot(self,node):return _operation(self,'snapshot',node,_finalization=True)
    def final_summary(self,node,noise):return _operation(self,'final_summary',node,noise,_finalization=True)
    def set_value(self,node,value):return _operation(self,'set_value',node,value)

    def resolve_operations(self,queries):
        self._flush();queries=list(queries)
        if not queries:return []
        method=queries[0].method
        if method not in ('create','root_choice','interior_choice','backup','snapshot','set_value','final_summary'):
            raise ValueError('Unknown device statistics operation')
        if len({q.id for q in queries})!=len(queries):raise ValueError('Duplicate statistics query')
        if any(not isinstance(q,OperationQuery) or q.backend is not self or q.method!=method or q.kwargs for q in queries):
            raise ValueError('Statistics dispatch requires one bound operation batch')
        start=time.monotonic();out=[]
        if method=='create':
            if any(len(q.args)!=3 or q.args[2] not in self._owners for q in queries):
                raise ValueError('Unknown statistics root owner')
            refs=self.pool.allocate([q.args[0] for q in queries],[q.args[1] for q in queries],[q.args[2] for q in queries])
            for query,ref in zip(queries,refs):
                self._owners[ref.owner].append(ref);self._counts[ref]=len(query.args[0]);out.append(ref)
            self._peak=max(self._peak,self.pool.active_nodes)
        else:
            arity={'root_choice':3,'interior_choice':1,'backup':3,'snapshot':1,'set_value':2,'final_summary':2}[method]
            if any(len(q.args)!=arity for q in queries):raise ValueError('Statistics signature mismatch')
            if method!='backup':
                refs=[q.args[0] for q in queries]
                if any(not isinstance(ref,NodeRef) or ref not in self._counts for ref in refs):
                    raise ValueError('Foreign or released statistics node')
            if method in ('root_choice','final_summary'):
                width=max(self._counts[ref] for ref in refs)
                noise=np.zeros((len(queries),width),dtype=np.float64)
                visits=[]
                for index,query in enumerate(queries):
                    n=self._counts[query.args[0]];value=np.asarray(query.args[1],dtype=np.float64)
                    if value.shape!=(n,):raise ValueError('Full legal-action root noise required')
                    if method=='root_choice':
                        if type(query.args[2]) is not int:raise TypeError('Integer visit target required')
                        visits.append(query.args[2])
                    noise[index,:n]=value
                if method=='root_choice':out=self.pool.select_root(refs,noise,visits).cpu().tolist()
                else:out=[dict(search_choice=row[0],roots_covered=row[1]) for row in self.pool.final_summary(refs,noise).cpu().tolist()]
            elif method=='interior_choice':out=self.pool.select_interior(refs).cpu().tolist()
            elif method=='backup':
                out=self.pool.backup_admitted([q.args[0] for q in queries],[q.args[1] for q in queries],
                                             [q.args[2] for q in queries],self.clock)
                self._completed_backups+=sum(out);self._late_backups+=len(out)-sum(out)
            elif method=='set_value':
                self.pool.set_raw_value(refs,[q.args[1] for q in queries]);out=[None]*len(queries)
            else:
                snapshot={key:value.cpu().numpy() for key,value in self.pool.snapshot(refs).items()}
                for index,ref in enumerate(refs):
                    n=self._counts[ref]
                    out.append(dict(prior=snapshot['prior'][index,:n].copy(),visits=snapshot['visits'][index,:n].copy(),
                        total=snapshot['total'][index,:n].copy(),raw_value=float(snapshot['value'][index])))
        self._operations[method]+=len(queries);self._wall+=time.monotonic()-start
        return [OperationResponse(query.id,value) for query,value in zip(queries,out)]

    def stats(self):
        self._flush()
        return dict(device=str(self.pool.device),active_nodes=self.pool.active_nodes,peak_nodes=self._peak,
            allocated_bytes=self.pool.allocated_bytes,operations=dict(self._operations),dispatch_wall_seconds=self._wall,
            completed_backups=self._completed_backups,late_backups_discarded=self._late_backups)

    def close(self):
        if self._closed:return
        self._flush()
        refs=[ref for owned in self._owners.values() for ref in owned]
        if refs:self.pool.release(refs)
        self._owners.clear();self._counts.clear();self._closed=True
        if self.pool.device.type=='cuda':self.pool._stream.synchronize()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
