"""Shared-page residency for lossless typed nonzero columns.

Logical offsets, scalar bits and per-channel limits match the sparse store.
The byte allocator holds four fixed typed streams, with no graph nodes or
boxed Game. Admission and every read retain owner/generation/page guards.
"""
from .device_sparse_soa_storage import DeviceSparseSoAStore,DEFAULT_CELLS,WIDTHS,CORRUPT_COLUMN
from .device_soa_storage import INVALID_LEASE,CAPACITY
from .device_rule_protocol import DeviceStateBatch
from ..batched_search.device_stable_page_pool import DeviceStableGraphPagePool
from .device_graph_pool import DevicePageHandle
from .soa_paged_columns import paged_column_source,paged_column_signatures


def paged_arena_plan(lengths,capacity,*,column_cells=None,max_bytes=4<<30,resident_bytes=1536<<20,page_bytes=16384):
    if type(capacity) is not int or not 1<=capacity<2**31:
        raise ValueError('Positive bounded typed slot count required')
    for value in (max_bytes,resident_bytes,page_bytes):
        if type(value) is not int or value<1:
            raise ValueError('Positive byte/page budgets required')
    if page_bytes<16 or page_bytes>=2**31 or page_bytes&(page_bytes-1):
        raise ValueError('Power-of-two pages of at least 16 bytes required')
    requested=DEFAULT_CELLS if column_cells is None else column_cells
    if set(requested)!=set(WIDTHS) or set(lengths)!=set(WIDTHS):
        raise ValueError('Four typed channel capacities required')
    if any(type(v) is not int or not 0<=v<2**31 for v in (*requested.values(),*lengths.values())):
        raise ValueError('Bounded nonnegative typed capacities required')
    caps={k:min(requested[k],lengths[k]) for k in WIDTHS}
    per_state=max(1,sum(caps[k]*(4+WIDTHS[k]) for k in WIDTHS))
    per_slot=(per_state+page_bytes-1)//page_bytes
    fixed=capacity*(per_slot*8+68)+per_slot*8+16
    pages=min(capacity*per_slot,(min(max_bytes,resident_bytes)-fixed)//(page_bytes+16))
    if pages<per_slot or pages>=2**31:
        raise MemoryError('Page budget must fit metadata and one complete typed state')
    return dict(column_cells=caps,pages=pages,page_bytes=page_bytes,max_pages_per_slot=per_slot,
                allocated_bytes=fixed+pages*(page_bytes+16))


class DevicePagedSoAStore(DeviceSparseSoAStore):
    storage_kind='shared_paged_typed_columns'

    def __init__(self,schema,capacity,*,column_cells=None,max_bytes=4<<30,resident_bytes=1536<<20,
                 page_bytes=16384,device='cuda',allow_cpu_reference=False):
        plan=paged_arena_plan(schema.lengths,capacity,column_cells=column_cells,max_bytes=max_bytes,
                              resident_bytes=resident_bytes,page_bytes=page_bytes)
        import torch
        self.t,self.device,self.schema,self.capacity=torch,torch.device(device),schema,capacity
        self.pages=DeviceStableGraphPagePool(slots=capacity,pages=plan['pages'],page_bytes=page_bytes,
            max_pages_per_slot=plan['max_pages_per_slot'],device=device,allow_cpu_reference=allow_cpu_reference)
        self.device=self.pages.device;self.owner=self.pages.owner
        self.column_cells=plan['column_cells'];self.max_bytes=max_bytes
        self.dtypes={'i64':torch.int64,'f64':torch.float64,'i32':torch.int32,'u8':torch.uint8}
        self.counts={k:torch.zeros(capacity,dtype=torch.int32,device=self.device) for k in WIDTHS}
        self.roots=torch.zeros(capacity,dtype=torch.int32,device=self.device)
        self.ids=torch.arange(capacity,dtype=torch.int64,device=self.device)
        self.peak_live=torch.zeros(1,dtype=torch.int64,device=self.device)
        self._workspaces,self.module,self.closed={},None,False
        if self.device.type=='cuda':
            from .cuda_runtime import CudaModule
            self.module=CudaModule(paged_column_source(page_bytes),paged_column_signatures(),
                                   device=str(self.device),binary_format='cubin')
        if self.allocated_bytes!=plan['allocated_bytes']:
            raise AssertionError('Typed shared-page allocation differs from its exact byte plan')

    @property
    def live(self):return self.pages.live.bool()

    @property
    def generation(self):return self.pages.generation+1

    @property
    def allocated_bytes(self):
        return self.pages.allocated_bytes+sum(v.numel()*v.element_size() for v in
            (*self.counts.values(),self.roots,self.ids,self.peak_live))

    def _handle(self,states):
        return DevicePageHandle(states.slots,states.generations-1,states.owners)

    def put_device(self,batch,owners,active,*,errors=None):
        self._open();t=self.t;rows=self._check_batch(batch)
        self._vector(owners,t.int64,rows);self._vector(active,t.bool,rows)
        errors=t.zeros(rows,dtype=t.int32,device=self.device) if errors is None else errors.clone()
        self._vector(errors,t.int32,rows)
        root=self.schema.pools[self.schema.root]
        valid=(owners>=0)&(batch.roots>root.base)&(batch.roots<=root.base+root.capacity)
        errors.bitwise_or_((active&~valid).to(t.int32)*INVALID_LEASE)
        wanted=active&valid&(errors==0);work=self.prepare_workspace(rows);self._count(batch,wanted,work)
        for key in WIDTHS:
            errors.bitwise_or_((wanted&(work[key]['total']>self.column_cells[key])).to(t.int32)*CAPACITY)
        wanted=wanted&(errors==0);live=self.live
        fits=wanted.sum()<=(~live).sum()
        errors.bitwise_or_((wanted&~fits).to(t.int32)*CAPACITY);wanted=wanted&fits
        free=t.argsort(live.long()*self.capacity+self.ids)
        rank=(wanted.long().cumsum(0)-1).clamp(0,self.capacity-1);slots=free[rank]
        totals=[work[k]['total'] for k in WIDTHS]
        sizes=sum(v.long()*(4+WIDTHS[k]) for k,v in zip(WIDTHS,totals)).clamp_min(1)
        reserved=self.pages.reserve(slots,self.pages.generation[slots],owners,sizes,wanted)
        errors.bitwise_or_((reserved.errors!=0).to(t.int32)*CAPACITY)
        admitted=wanted&(errors==0)
        if self.device.type=='cpu':
            self._pack_cpu(batch,slots,admitted,work)
        else:
            for key in WIDTHS:
                source=batch.buffers[key]
                tile=work[key]
                self.module.launch('pack_page_'+key,[source,admitted.view(t.uint8),slots,tile['prefix'],*totals,
                    self.pages.data,self.pages.page_table,self.pages.page_owner,self.pages.logical_bytes,errors,
                    rows,self.schema.lengths[key],tile['tiles'],self.pages.page_capacity,self.pages.max_pages_per_slot,
                    self.column_cells[key]],grid=(tile['tiles'],rows),block=256,capture_safe=True)
        rollback=admitted&(errors!=0)
        failed=self.pages.release(reserved.handle,rollback)
        errors.bitwise_or_((rollback&(failed!=0)).to(t.int32)*CORRUPT_COLUMN)
        good=admitted&(errors==0)
        for key in WIDTHS:
            target=self.counts[key]
            target.index_add_(0,slots,t.where(good,work[key]['total']-target[slots],0))
        self.roots.index_add_(0,slots,t.where(good,batch.roots-self.roots[slots],0))
        self._clear_free_metadata()
        t.maximum(self.peak_live,self.pages.live.sum().reshape(1),out=self.peak_live)
        return DeviceStateBatch(self,t.where(good,slots,-1),t.where(good,self.generation[slots],0),owners.clone(),good,errors)

    def _pack_cpu(self,batch,slots,good,work):
        t=self.t
        for row in range(batch.rows):
            if not bool(good[row]):continue
            slot=int(slots[row]);base=0
            for key in WIDTHS:
                source=batch.buffers[key]
                bits=source.view(t.int64) if key=='f64' else source
                columns=t.nonzero(bits[:,row],as_tuple=False)[:,0];n=len(columns);width=WIDTHS[key]
                if n:
                    record=t.empty((n,4+width),dtype=t.uint8)
                    record[:,:4]=columns.to(t.int32).contiguous().view(t.uint8).reshape(n,4)
                    record[:,4:]=source[columns,row].contiguous().view(t.uint8).reshape(n,width)
                    offsets=t.arange(n*(4+width),dtype=t.int64)+base
                    physical=self.pages.page_table[slot,offsets//self.pages.page_bytes]
                    address=physical*self.pages.page_bytes+offsets%self.pages.page_bytes
                    self.pages.data.flatten().index_copy_(0,address,record.flatten())
                base+=n*(4+width)

    def _validate_pages(self,states,slots,good):
        t=self.t
        if self.device.type=='cuda':
            self.module.launch('validate_paged_columns',[*self.counts.values(),slots,good.view(t.uint8),
                self.pages.logical_bytes,self.pages.page_count,self.pages.page_table,self.pages.page_owner,
                states.errors,len(slots),self.pages.page_capacity,self.pages.max_pages_per_slot,
                *self.column_cells.values()],grid=(len(slots)+127)//128,block=128,capture_safe=True)
        else:
            for row in range(len(slots)):
                if not bool(good[row]):continue
                slot=int(slots[row]);counts={k:int(v[slot]) for k,v in self.counts.items()}
                size=max(1,sum(counts[k]*(4+WIDTHS[k]) for k in WIDTHS))
                n=int(self.pages.page_count[slot]);table=self.pages.page_table[slot,:max(0,min(n,self.pages.max_pages_per_slot))].tolist()
                valid=(all(0<=counts[k]<=self.column_cells[k] for k in WIDTHS)
                       and size==int(self.pages.logical_bytes[slot])
                       and n==(size+self.pages.page_bytes-1)//self.pages.page_bytes
                       and 1<=n<=self.pages.max_pages_per_slot and len(set(table))==n
                       and all(0<=p<self.pages.page_capacity and int(self.pages.page_owner[p])==slot for p in table))
                if not valid:states.errors[row]|=CORRUPT_COLUMN
        return good&(states.errors==0)

    def gather_into_device(self,states,destination,*,shared_reads=False):
        self._open();t=self.t;rows=self._check_batch(destination)
        if rows!=len(states.slots):raise ValueError('Matching bounded typed work rows required')
        slots,good=self.validate(states,shared_reads=shared_reads);good=self._validate_pages(states,slots,good)
        for target in destination.buffers.values():target.zero_()
        if self.device.type=='cpu':
            self._gather_cpu(states,slots,good,destination)
        else:
            for key in WIDTHS:
                target=destination.buffers[key]
                limit=self.column_cells[key]
                if not limit:continue
                self.module.launch('gather_page_'+key,[self.pages.data,self.pages.page_table,self.pages.page_owner,
                    self.pages.logical_bytes,*self.counts.values(),slots,good.view(t.uint8),target,states.errors,
                    rows,self.schema.lengths[key],self.pages.page_capacity,self.pages.max_pages_per_slot,limit],
                    grid=((limit+255)//256,rows),block=256,capture_safe=True)
        good=good&(states.errors==0)
        for target in destination.buffers.values():target.masked_fill_(~good[None,:],0)
        t.index_select(self.roots,0,slots,out=destination.roots);destination.roots.masked_fill_(~good,0)
        return destination

    def _gather_cpu(self,states,slots,good,destination):
        t=self.t
        for row in range(destination.rows):
            if not bool(good[row]):continue
            slot=int(slots[row]);base=0
            for key in WIDTHS:
                target=destination.buffers[key]
                n=int(self.counts[key][slot]);width=WIDTHS[key]
                if n:
                    offsets=t.arange(n*(4+width),dtype=t.int64)+base
                    physical=self.pages.page_table[slot,offsets//self.pages.page_bytes]
                    address=physical*self.pages.page_bytes+offsets%self.pages.page_bytes
                    record=self.pages.data.flatten().index_select(0,address).reshape(n,4+width)
                    columns=record[:,:4].contiguous().view(t.int32).reshape(n).long()
                    if bool(((columns<0)|(columns>=target.shape[0])).any()) or bool((columns[1:]<=columns[:-1]).any()):
                        states.errors[row]|=CORRUPT_COLUMN
                    else:target[columns,row]=record[:,4:].clone(memory_format=t.contiguous_format).view(self.dtypes[key]).reshape(n)
                base+=n*(4+width)

    def _clear_free_metadata(self):
        free=~self.live
        for values in self.counts.values():values.masked_fill_(free,0)
        self.roots.masked_fill_(free,0)

    def release_device(self,states):
        self._open();slots,good=self.validate(states);good=self._validate_pages(states,slots,good)
        failed=self.pages.release(self._handle(states),good)
        states.errors.bitwise_or_((good&(failed!=0)).to(self.t.int32)*CORRUPT_COLUMN)
        self._clear_free_metadata()

    def metrics_device(self):
        return dict(super().metrics_device(),pages=self.pages.metrics_device())
