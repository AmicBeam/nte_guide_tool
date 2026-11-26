"""One host launch, a device-controlled bounded WHILE graph, no CPU predicate.

This graph runner is infrastructure, not full rules or game-capacity evidence.
The body callback runs once during static capture; CUDA replays it until the
device predicate clears or the device iteration/time guard fails closed.
Requires PyTorch native conditional graph capture and typed CUDA 13 bindings.
There is no raw child-graph or eager CPU-controlled fallback.
"""
from contextlib import contextmanager


_SOURCE = r'''
__device__ unsigned long long device_clock(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
extern "C" __global__ void initialize_guard(long long* counter,long long* start,int* errors){
 if(threadIdx.x||blockIdx.x)return;*counter=0;*start=(long long)device_clock();*errors=0;
}
extern "C" __global__ void update_condition(unsigned char* proceed,const unsigned char* active,int rows,
 long long* counter,const long long* start,int* errors,long long max_ticks,long long budget_ns,int advance){
 if(threadIdx.x||blockIdx.x)return;
 if(advance)(*counter)++;
 int unfinished=0;for(int i=0;i<rows;i++)unfinished|=active[i]!=0;
 unsigned long long elapsed=device_clock()-(unsigned long long)*start;
 if(unfinished&&(*counter>=max_ticks||elapsed>=(unsigned long long)budget_ns))*errors|=1;
 *proceed=(unsigned char)(unfinished&&!*errors);
}
'''


def _checked(result):
    error, *values = result
    if int(error) != 0:
        raise RuntimeError('CUDA conditional graph API failed: ' + str(error))
    return tuple(values)


# Torch's caching allocator may retain an allocation's originating stream after
# its tensor or captured graph owner closes. As with Torch's own stream pools,
# these native streams remain valid until the bounded worker process exits.
_PROCESS_CAPTURE_STREAMS = []


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
        self._require_native_api(torch)
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
            'update_condition': ['uint8*','uint8*','int32','int64*','int64*','int32*','int64','int64','int32'],
        }, str(self.device))
        self.counter = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.start = torch.zeros_like(self.counter)
        self.errors = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._condition_scalar = torch.zeros((), dtype=torch.bool, device=self.device)
        self._condition_bytes = self._condition_scalar.view(torch.uint8)
        self._capture_stream_callbacks = []
        # torch.cuda.Stream() round-robins a finite pool. Distinct Python
        # objects can therefore alias an actively capturing parent's stream.
        # Keep unique parent and prewarm streams. Native conditional capture
        # replaces body_stream with its allocator-owned actual child stream.
        self._owned_streams = []
        self.graph = None
        self._loop_built = False
        self._closed = False
        try:
            with torch.cuda.device(self.device):
                self.capture_stream = self._new_stream()
                self.body_stream = self._new_stream()
        except BaseException:
            self.close()
            raise

    def _new_stream(self):
        native, = _checked(self.runtime.cudaStreamCreateWithFlags(1))
        try:
            stream = self.t.cuda.ExternalStream(int(native), device=self.device)
        except BaseException:
            _checked(self.runtime.cudaStreamDestroy(native))
            raise
        _PROCESS_CAPTURE_STREAMS.append(native)
        self._owned_streams.append(native)
        return stream

    @staticmethod
    def _require_native_api(t):
        required = ('get_currently_capturing_graph', 'begin_capture_to_while_node',
                    'end_capture_to_conditional_node',
                    'set_conditional_handle_for_current_node')
        if any(not callable(getattr(t.cuda.CUDAGraph, name, None)) for name in required):
            raise RuntimeError('PyTorch native conditional graph capture required; no raw fallback')

    def register_capture_stream_callback(self, callback):
        """Observe exact native parent/body streams during static capture only.

        The native graph owns the actual body stream and its allocator pool.
        Consumers with stream ownership checks register this observer before
        capture, then admit only the observed dependency-linked stream.
        """
        if self._closed or self._loop_built or self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Register stream observers before static graph construction')
        if not callable(callback):
            raise TypeError('Capture stream observer must be callable')
        self._capture_stream_callbacks.append(callback)

    def _condition(self, predicate, advance):
        self.module.launch('update_condition', [self._condition_bytes, predicate, predicate.numel(),
                           self.counter,self.start,self.errors,self.max_ticks,self.budget_ns,advance],
                           grid=1,block=1,capture_safe=True)

    @contextmanager
    def _while_body(self, predicate):
        t = self.t
        parent_stream = t.cuda.current_stream(self.device)
        graph = t.cuda.CUDAGraph.get_currently_capturing_graph()
        if graph is None:
            raise RuntimeError('Device WHILE requires an owning PyTorch CUDA graph')
        self._condition(predicate, 0)
        # PyTorch must see the conditional child capture: it switches the
        # current stream and retains the child's allocator pool in the parent.
        # A raw cudaStreamBeginCaptureToGraph bypasses that ownership and lets
        # a subsequent independent capture reclaim live child temporaries.
        graph.begin_capture_to_while_node(self._condition_scalar)
        try:
            self.body_stream = t.cuda.current_stream(self.device)
            for callback in self._capture_stream_callbacks:
                callback(parent_stream, self.body_stream)
            yield
            self._condition(predicate, 1)
            graph.set_conditional_handle_for_current_node(self._condition_scalar)
        finally:
            graph.end_capture_to_conditional_node()

    def _validate_inputs(self, predicate, body, initialize):
        t = self.t
        if self._closed:
            raise RuntimeError('Device loop closed')
        if self.graph is not None or self._loop_built:
            raise RuntimeError('Device loop is already frozen')
        if (not isinstance(predicate, t.Tensor) or predicate.device != self.device
                or predicate.dtype != t.uint8 or predicate.ndim != 1
                or not predicate.is_contiguous() or predicate.numel() < 1):
            raise ValueError('Persistent contiguous uint8[B] CUDA predicate required')
        if not callable(body) or (initialize is not None and not callable(initialize)):
            raise TypeError('Static graph callbacks required')

    def capture_loop(self, predicate, body, *, initialize=None):
        """Embed this guarded WHILE in an already captured device graph.

        A separate instance owns each nested loop; PyTorch owns its native
        child stream and allocator pool. The enclosing graph retains this
        instance and all captured buffers. This
        method never launches a graph or evaluates the predicate on the host.
        """
        self._validate_inputs(predicate, body, initialize)
        if not self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Embedded device loop requires active CUDA capture')
        self.module.launch('initialize_guard', [self.counter, self.start, self.errors],
                           grid=1, block=1, capture_safe=True)
        if initialize is not None:
            initialize()
        with self._while_body(predicate):
            body()
        self.predicate = predicate
        self._loop_built = True

    def capture(self, predicate, body, *, initialize=None):
        self._validate_inputs(predicate, body, initialize)
        t = self.t
        graph = t.cuda.CUDAGraph()
        stream = self.capture_stream
        stream.wait_stream(t.cuda.current_stream(self.device))
        with t.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
            self.capture_loop(predicate, body, initialize=initialize)
        self.graph, self.predicate = graph, predicate

    def replay(self):
        if self.graph is None:
            raise RuntimeError('No frozen device WHILE graph')
        self.graph.replay()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.graph = None
        # Drop logical ownership, but do not invalidate allocator-held stream
        # handles. Worker exit releases the CUDA context and all native streams.
        self._owned_streams.clear()
        self.module.close()
