"""One captured fixed-size tensor body over a device-resident row cohort."""
from .cuda_conditional_graph import CudaConditionalWhile


class DeviceTensorRowQueue:
    def __init__(self,*,rows,scratch_rows,seconds,device='cuda'):
        import torch as t
        if type(rows)is not int or rows<1 or type(scratch_rows)is not int or scratch_rows<1:
            raise ValueError('Positive bounded row and scratch counts required')
        import math
        if type(seconds)not in (int,float) or not math.isfinite(seconds) or seconds<=0:
            raise ValueError('Positive independent queue budget required')
        self.device=t.device(device)
        if self.device.type!='cuda' or not t.cuda.is_available():raise ValueError('Actual CUDA required')
        if self.device.index is None:self.device=t.device('cuda',t.cuda.current_device())
        if t.cuda.is_current_stream_capturing():raise RuntimeError('Prepare queue outside capture')
        self.t=t;self.rows=rows;self.scratch_rows=scratch_rows;self.closed=False;self.captured=False
        self.cursor=t.zeros(1,dtype=t.int64,device=self.device)
        self.pending=t.zeros(1,dtype=t.uint8,device=self.device)
        self.positions=t.arange(scratch_rows,dtype=t.int64,device=self.device)
        self.loop=CudaConditionalWhile(device=str(self.device),max_ticks=(rows+scratch_rows-1)//scratch_rows+1,seconds=seconds)
    def capture_device(self,active,operation):
        t=self.t
        if self.closed or self.captured:raise RuntimeError('Queue closed or already captured')
        if not t.cuda.is_current_stream_capturing():raise RuntimeError('Queue requires parent capture')
        if not isinstance(active,t.Tensor) or active.device!=self.device or active.dtype!=t.bool or active.shape!=(self.rows,) or not active.is_contiguous():
            raise ValueError('Exact contiguous CUDA boolean cohort required')
        if not callable(operation):raise TypeError('Static tensor operation required')
        def initialize():
            self.cursor.zero_();self.pending.copy_(active.any().reshape(1).to(t.uint8))
        def body():
            raw=self.cursor[0]+self.positions;inside=raw<self.rows
            ids=raw.clamp(0,self.rows-1)
            valid=inside&active.index_select(0,ids)
            destinations=t.where(inside,raw,self.rows)
            operation(ids,valid,destinations)
            self.cursor.add_(self.scratch_rows)
            self.pending.copy_(((self.cursor<self.rows)&active.any()).to(t.uint8))
        self.loop.capture_loop(self.pending,body,initialize=initialize)
        self.captured=True
        return self.loop.errors
    def close(self):
        if not self.closed:self.loop.close();self.closed=True
