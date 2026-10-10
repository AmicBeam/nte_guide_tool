"""One host launch, a device-controlled bounded WHILE graph, no CPU predicate.

This graph runner is infrastructure, not full rules or game-capacity evidence.
The body callback runs once during static capture; CUDA replays it until the
device predicate clears or the device iteration/time guard fails closed.
Requires typed CUDA 13 bindings. There is no eager CPU-controlled fallback.
"""
from contextlib import contextmanager


_SOURCE = r'''
typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle, unsigned int);
__device__ unsigned long long device_clock(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
extern "C" __global__ void initialize_guard(long long* counter,long long* start,int* errors){
 if(threadIdx.x||blockIdx.x)return;*counter=0;*start=(long long)device_clock();*errors=0;
}
extern "C" __global__ void update_condition(unsigned long long handle,const unsigned char* active,int rows,
 long long* counter,const long long* start,int* errors,long long max_ticks,long long budget_ns,int advance){
 if(threadIdx.x||blockIdx.x)return;
 if(advance)(*counter)++;
 int unfinished=0;for(int i=0;i<rows;i++)unfinished|=active[i]!=0;
 unsigned long long elapsed=device_clock()-(unsigned long long)*start;
 if(unfinished&&(*counter>=max_ticks||elapsed>=(unsigned long long)budget_ns))*errors|=1;
 cudaGraphSetConditional(handle,unfinished&&!*errors);
}
'''


def _checked(result):
    error, *values = result
    if int(error) != 0:
        raise RuntimeError('CUDA conditional graph API failed: ' + str(error))
    return tuple(values)


class CudaConditionalWhile:
    """Capture an exact static body with a persistent uint8[B] CUDA predicate.

    The caller retains all input/output buffers, and supplies an original
    remaining budget before capture. No graph execution reads CPU conditions.
    Device guard errors are inspected only after the entire batch completes.
    """
    def __init__(self, *, device='cuda', max_ticks=10000, seconds=60):
        import torch
        if type(max_ticks) is not int or not 0 < max_ticks < 2**63:
            raise ValueError('Positive bounded iteration count required')
        if isinstance(seconds, bool) or not 0 < seconds <= 86400:
            raise ValueError('Positive bounded time budget required')
        if not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no host loop fallback')
        self.t = torch
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CUDA device required')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        try:
            from cuda.bindings import driver, runtime
        except ImportError as exc:
            raise RuntimeError('Typed cuda-bindings>=13 required for device WHILE') from exc
        self.driver, self.runtime = driver, runtime
        self.max_ticks, self.budget_ns = max_ticks, int(seconds * 1e9)
        from ..batched_duel.cuda_runtime import CudaModule
        self.module = CudaModule(_SOURCE, {
            'initialize_guard': ['int64*', 'int64*', 'int32*'],
            'update_condition': ['uint64','uint8*','int32','int64*','int64*','int32*','int64','int64','int32'],
        }, str(self.device))
        self.counter = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.start = torch.zeros_like(self.counter)
        self.errors = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.body_stream = torch.cuda.Stream(device=self.device)
        self.graph = None

    def _condition(self, handle, predicate, advance):
        self.module.launch('update_condition', [int(handle), predicate, predicate.numel(),
                           self.counter,self.start,self.errors,self.max_ticks,self.budget_ns,advance],
                           grid=1,block=1,capture_safe=True)

    @contextmanager
    def _while_body(self, predicate):
        t, rt, dr = self.t, self.runtime, self.driver
        parent_stream = t.cuda.current_stream(self.device)
        capture = _checked(rt.cudaStreamGetCaptureInfo(parent_stream.cuda_stream))
        if capture[0] != rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
            raise RuntimeError('Device WHILE must be built inside CUDA capture')
        parent = capture[2]
        handle, = _checked(rt.cudaGraphConditionalHandleCreate(parent, 0, 0))
        self._condition(handle, predicate, 0)
        capture = _checked(rt.cudaStreamGetCaptureInfo(parent_stream.cuda_stream))
        dependencies = capture[3]
        params = dr.CUgraphNodeParams()
        params.type = dr.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        params.conditional.handle = handle
        params.conditional.type = dr.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_WHILE
        params.conditional.size = 1
        params.conditional.ctx, = _checked(dr.cuCtxGetCurrent())
        node, = _checked(dr.cuGraphAddNode(parent, dependencies, None, len(dependencies), params))
        _checked(rt.cudaStreamUpdateCaptureDependencies(parent_stream.cuda_stream,[node],None,1,
                    rt.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies))
        body_stream = self.body_stream
        body_graph = params.conditional.phGraph_out[0]
        _checked(rt.cudaStreamBeginCaptureToGraph(body_stream.cuda_stream, body_graph, None, None, 0,
                    rt.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
        active = True
        try:
            with t.cuda.stream(body_stream):
                yield
                self._condition(handle, predicate, 1)
                _checked(rt.cudaStreamEndCapture(body_stream.cuda_stream))
                active = False
        finally:
            if active:
                _checked(rt.cudaStreamEndCapture(body_stream.cuda_stream))

    def capture(self, predicate, body, *, initialize=None):
        t = self.t
        if self.graph is not None:
            raise RuntimeError('Graph is already frozen')
        if (not isinstance(predicate, t.Tensor) or predicate.device != self.device
                or predicate.dtype != t.uint8 or predicate.ndim != 1
                or not predicate.is_contiguous() or predicate.numel() < 1):
            raise ValueError('Persistent contiguous uint8[B] CUDA predicate required')
        if not callable(body) or (initialize is not None and not callable(initialize)):
            raise TypeError('Static graph callbacks required')
        graph = t.cuda.CUDAGraph()
        stream = self.capture_stream
        stream.wait_stream(t.cuda.current_stream(self.device))
        with t.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
            self.module.launch('initialize_guard',[self.counter,self.start,self.errors],grid=1,block=1,capture_safe=True)
            if initialize is not None:
                initialize()
            with self._while_body(predicate):
                body()
        self.graph, self.predicate = graph, predicate

    def replay(self):
        if self.graph is None:
            raise RuntimeError('No frozen device WHILE graph')
        self.graph.replay()

    def close(self):
        self.graph = None
        self.module.close()
