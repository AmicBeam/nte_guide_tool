"""Persistent device search FSM across deliberately short graph replays.

A planned search-tick boundary keeps all FSM/RNG/statistics/leases resident.
Only a completed entire root cohort may enter the physical commit gate. This
infrastructure is not full-rules, training, or CUDA qualification evidence.
"""
from contextlib import nullcontext
import time
import math

from .device_games import DeviceGameLoop, GUARD_ERROR


class DeviceChunkedGameLoop(DeviceGameLoop):
    def __init__(self,*args,chunk_search_ticks=1,**kwargs):
        if type(chunk_search_ticks)is not int or not 1<=chunk_search_ticks<=10000:
            raise ValueError('One to10000 deliberate search ticks per chunk required')
        self._chunk_seconds=kwargs.get('seconds',60)
        if type(self._chunk_seconds)not in (int,float) or not math.isfinite(self._chunk_seconds) or not 0<self._chunk_seconds<=90:
            raise ValueError('Each persistent CUDA chunk requires at most90seconds')
        super().__init__(*args,**kwargs)
        t,z=self.t,dict(device=self.device)
        self.chunk_search_ticks=chunk_search_ticks
        self.chunk_pending_device=t.zeros(1,dtype=t.uint8,**z)
        self.chunk_search_ticks_device=t.zeros(1,dtype=t.int64,**z)
        self._root_ticks=t.zeros(1,dtype=t.int64,**z)
        self._start_pred=t.zeros(1,dtype=t.uint8,**z)
        self._commit_pred=t.zeros_like(self._start_pred)
        self._search_pred=t.zeros(self.rows,dtype=t.uint8,**z)
        self._chunk_pred=t.zeros_like(self._start_pred)
        self._abort_pred=t.zeros_like(self._start_pred)
        self._abort_force=t.zeros(1,dtype=t.bool,**z)
        self._release_roots=t.zeros(1,dtype=t.bool,**z)
        self._captured=False;self._chunk_closed=False;self._roots_released=False
        self._start_gate=self._commit_gate=self._abort_gate=None
        self._bootstrap_graph=self._abort_graph=None
        if self.device.type=='cuda' and not self.allow_synthetic:
            from .cuda_conditional_graph import CudaConditionalWhile
            self._start_gate=CudaConditionalWhile(device=str(self.device),max_ticks=2,seconds=self._chunk_seconds)
            self._commit_gate=CudaConditionalWhile(device=str(self.device),max_ticks=2,seconds=self._chunk_seconds)
            self._abort_gate=CudaConditionalWhile(device=str(self.device),max_ticks=2,seconds=self._chunk_seconds)
            # Search guard allows the final planned predicate-clear tick.
            self._inner.max_ticks=chunk_search_ticks+1
            self._outer.max_ticks=2

    def initialize(self,*args):
        if self._ready:raise RuntimeError('Persistent chunk cohort can only initialize once')
        super().initialize(*args)
        self.chunk_pending_device.zero_();self._root_ticks.zero_()

    def _begin_chunk(self):
        self.chunk_search_ticks_device.zero_()
        self._chunk_pred.fill_(1)
        self._start_pred.copy_(((self._unfinished!=0).any() & (self.chunk_pending_device==0)).to(self.t.uint8))

    def _start_once(self):
        self._start_search()
        self.chunk_pending_device.fill_(1)
        self._root_ticks.zero_();self._start_pred.zero_()

    def _refresh_search_predicate(self):
        self._search_pred.copy_(((self.search.unfinished_device!=0)
            & (self.chunk_pending_device!=0)
            & (self.chunk_search_ticks_device<self.chunk_search_ticks)
            & (self._root_ticks<self._inner_max_ticks)).to(self.t.uint8))

    def _search_once(self):
        self.search.tick()
        self._root_ticks.add_(1);self.chunk_search_ticks_device.add_(1)
        self._refresh_search_predicate()

    def _commit_once(self):
        super()._commit_turn()
        self.chunk_pending_device.zero_();self._commit_pred.zero_()

    def _abort_pending(self):
        mask=self._search_mask & (self.chunk_pending_device!=0)
        self._error(mask,mask,GUARD_ERROR)
        self.search.abort_device(mask,error_code=GUARD_ERROR)
        # Consume the failed search result solely to release its action leases
        # and finish failed sinks. Error masks prevent all physical commits.
        super()._commit_turn()
        self.chunk_pending_device.zero_();self._abort_pred.zero_()

    def _mark_failed(self,failed):
        live=self._present & ~self._finished & ~self._truncated
        self._errors.bitwise_or_((live & failed).to(self.t.int32)*GUARD_ERROR)
        self._abort_pred.copy_((failed & (self.chunk_pending_device!=0)).to(self.t.uint8))
        self._update_predicate()

    def _chunk_body(self):
        self._start_gate.capture_loop(self._start_pred,self._start_once)
        self._refresh_search_predicate()
        self._inner.capture_loop(self._search_pred,self._search_once)
        exhausted=(self._root_ticks>=self._inner_max_ticks)&(self.search.unfinished_device!=0).any()
        failed=exhausted | (self._inner.errors!=0).any() | (self._start_gate.errors!=0).any()
        self._mark_failed(failed)
        # A planned pause leaves search unfinished and cannot enter commit.
        self._commit_pred.copy_(((self.chunk_pending_device!=0)
            & ~(self.search.unfinished_device!=0).any() & ~failed).to(self.t.uint8))
        self._commit_gate.capture_loop(self._commit_pred,self._commit_once)
        self._chunk_pred.zero_()

    def capture(self,roots,job_seeds,policy_index_by_side,auxiliary_index_by_side,train_sides):
        if self._captured or self._ready:raise RuntimeError('Persistent chunk graph already initialized/frozen')
        if self.device.type!='cuda' or self.allow_synthetic:
            raise RuntimeError('Production chunk capture requires actual conditional CUDA graphs')
        t=self.t
        # Register exactly the actual nested dependency-linked capture streams.
        runners=(self._outer,self._inner,self._start_gate,self._commit_gate,self._abort_gate)
        streams=[r.body_stream for r in runners]+[self._outer.capture_stream]
        for owner in (self.search.statistics if self._counts_as_search else None,self.random_plan):
            if owner is not None and callable(getattr(owner,'bind_capture_streams',None)):
                owner.bind_capture_streams(streams[0],streams[1])
                owner._capture_streams=frozenset(s.cuda_stream for s in streams)
        scope=self.search.statistics.capture_scope()if self._counts_as_search else nullcontext()
        stream=self._outer.capture_stream
        stream.wait_stream(t.cuda.current_stream(self.device))
        with scope:
            # This graph is launched once only. Reusable chunks never execute
            # game.initialize or reset sink/RNG/statistics/physical counters.
            self._bootstrap_graph=t.cuda.CUDAGraph()
            with t.cuda.graph(self._bootstrap_graph,stream=stream,capture_error_mode='thread_local'):
                self.initialize(roots,job_seeds,policy_index_by_side,auxiliary_index_by_side,train_sides)
            self._outer.capture(self._chunk_pred,self._chunk_body,initialize=self._begin_chunk)
            self._abort_graph=t.cuda.CUDAGraph()
            with t.cuda.graph(self._abort_graph,stream=stream,capture_error_mode='thread_local'):
                self._abort_pred.copy_(self.chunk_pending_device)
                force=self._abort_force.reshape(())
                self._errors.bitwise_or_((self._present & force).to(t.int32)*GUARD_ERROR)
                self._abort_gate.capture_loop(self._abort_pred,self._abort_pending)
                self._update_predicate()
                self._release_checked(self._physical(self._present),self._present & self._release_roots.reshape(()))
        self._bootstrap_graph.replay();t.cuda.synchronize(self.device)
        self._captured=True

    def _cpu_chunk(self):
        """Explicit synthetic oracle only; production never invokes host FSM."""
        if self.device.type!='cpu' or not self.allow_synthetic:
            raise RuntimeError('CPU chunk execution is a synthetic fixture only')
        if not self._ready:raise RuntimeError('Initialize synthetic cohort first')
        self._begin_chunk()
        if bool(self._start_pred.any()):self._start_once()
        self._refresh_search_predicate()
        while bool(self._search_pred.any()):self._search_once()
        exhausted=bool((self._root_ticks>=self._inner_max_ticks).any()and self.search.unfinished_device.any())
        if exhausted:
            self._mark_failed(self.t.tensor(True,device=self.device))
            if bool(self.chunk_pending_device.any()):self._abort_pending()
        elif bool(self.chunk_pending_device.any())and not bool(self.search.unfinished_device.any()):
            self._commit_once()
        self._chunk_pred.zero_()

    def abort(self,*,release_roots=False,force=True):
        """Batch-level cancellation; never turns a paused root into success."""
        if self._chunk_closed or self._roots_released:return
        if self.device.type=='cpu' and self.allow_synthetic:
            if force:self._errors.bitwise_or_(self._present.to(self.t.int32)*GUARD_ERROR)
            if bool(self.chunk_pending_device.any()):self._abort_pending()
            self._update_predicate()
            if release_roots:
                self._roots_released=True
                self._release_checked(self._physical(self._present),self._present)
            return
        if self._abort_graph is None:return
        self._abort_force.fill_(force);self._release_roots.fill_(release_roots)
        if release_roots:self._roots_released=True
        self._abort_graph.replay();self.t.cuda.synchronize(self.device)

    def tick(self):
        raise RuntimeError('Persistent chunk cohorts advance only through replay_chunk')

    def replay_chunk(self):
        if self._chunk_closed or self._roots_released:raise RuntimeError('Persistent chunk loop closed or roots released')
        if self.device.type=='cpu' and self.allow_synthetic:
            self._cpu_chunk();return
        if not self._captured:raise RuntimeError('No frozen persistent chunk graph')
        # A final slow tick can clear a conditional predicate before its guard
        # notices elapsed time. Inspect the completed CHUNK wall duration only;
        # no host root choice, simulation, or partial success is admitted.
        self.t.cuda.synchronize(self.device);started=time.monotonic()
        self._outer.replay();self.t.cuda.synchronize(self.device)
        failed=time.monotonic()-started>=self._chunk_seconds
        failed=failed or bool(self._errors.any().item())
        failed=failed or any(bool(r.errors.any().item())for r in
            (self._outer,self._inner,self._start_gate,self._commit_gate))
        if failed:self.abort(force=True)

    replay=replay_chunk

    def close(self):
        if self._chunk_closed:return
        try:
            if self._ready:self.abort(release_roots=True,force=False)
        finally:
            self._chunk_closed=True
            self._bootstrap_graph=self._abort_graph=None
            for runner in (self._start_gate,self._commit_gate,self._abort_gate):
                if runner is not None:runner.close()
            super().close()
