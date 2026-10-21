"""Cooperative rule-operation suspension and bounded cross-root dispatch.

Only calls to the explicit backend protocol are rewritten. Rule code, search
formulas, RNG streams and model inputs are not rewritten. CPU oracle dispatch
requires an explicit opt-in; a GPU backend must provide a bulk resolver.

Async rule dispatch API for the primary agent:
- CooperativeScheduler(..., async_rule_dispatch=None)
  None (default): capability-gated. A rule backend is used asynchronously iff
  supports_async_operations is True and submit_operations/poll_operations exist.
  GPU rule backends and other unsupported backends keep the synchronous
  resolve_operations wave.
  False: force the legacy global wave barrier for every backend.
  True: require async capability on every rule backend that receives
  OperationQuery; statistics and inference stay bulk-dispatched among
  currently ready roots only.
- Ready roots may advance and infer while an unrelated worker is still
  executing. Each root still has at most one outstanding operation and the
  same internal deterministic RNG/sequence.
- Final statistics snapshots remain synchronous read-only finalization.
- progress_metrics reports actual pending_operations, ready_roots and
  idle_workers. It does not invent queue wait.
"""
from dataclasses import dataclass,replace
from hashlib import sha256
import ast
import inspect
import itertools
import time
from uuid import uuid4


@dataclass(frozen=True)
class OperationQuery:
    id: str
    backend: object
    method: str
    args: tuple
    kwargs: tuple
    finalization: bool = False
    deadline: float | None = None


@dataclass(frozen=True)
class RequestExpired:
    id: str
    started: bool = False


@dataclass(frozen=True)
class OperationResponse:
    id: str
    value: object


@dataclass(frozen=True)
class RootJob:
    root_id: str
    game_id: str
    generation: int
    generator: object
    deadline: float | None = None


_PROTOCOL_METHODS=frozenset(('acting_side','finished','terminal_value','decision',
    'sample_world','step','clean','value_observation','serving_root_metadata','public_terminal_probe','rule_decision'))
_OPTIONAL_PROTOCOL_METHODS=frozenset(('covered_value_continuation',))
_SUSPEND_METHODS=_PROTOCOL_METHODS|_OPTIONAL_PROTOCOL_METHODS
_PREFIX=uuid4().hex
_COUNTER=itertools.count()
_CACHE={}


def _operation(backend,method,*args,_finalization=False,**kwargs):
    if type(_finalization) is not bool:raise TypeError('Boolean finalization marker required')
    deadline=kwargs.pop('deadline',None)
    inline=getattr(backend,'inline_operation',None)
    if inline is not None and not kwargs and not _finalization:
        handled,value=inline(method,args)
        if handled:return value
    query=OperationQuery(f'{_PREFIX}:{next(_COUNTER)}',backend,method,args,tuple(kwargs.items()),_finalization,deadline)
    response=yield query
    if not isinstance(response,OperationResponse) or response.id!=query.id:
        raise ValueError('Rule operation response belongs to a different request')
    return response.value


class _SuspendBackend(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='backend' and node.func.attr in _SUSPEND_METHODS):
            call=ast.Call(func=ast.Name(id='_batch_operation',ctx=ast.Load()),
                args=[ast.Name(id='backend',ctx=ast.Load()),ast.Constant(node.func.attr),*node.args],
                keywords=node.keywords)
            return ast.copy_location(ast.YieldFrom(value=call),node)
        return node


def cooperative_search(*args,**kwargs):
    """Return the same traversal with explicit backend calls suspended as requests.

    Pass an absolute monotonic ``deadline`` anchored before queueing; setting a
    relative time limit inside a lazily started generator would omit queue wait.
    """
    if kwargs.get('time_limit') is not None:
        raise ValueError('Cooperative roots require a queue-anchored absolute deadline')
    from . import traversal
    names=('_follow_terminal_generator','_short_terminal_value_generator',
           '_teacher_terminal_generator','_teacher_value_generator','search_traversal')
    functions=tuple(getattr(traversal,name) for name in names)
    if functions not in _CACHE:
        source='\n\n'.join(inspect.getsource(fn) for fn in functions)
        identity=sha256(source.encode()).hexdigest()
        tree=_SuspendBackend().visit(ast.parse(source));ast.fix_missing_locations(tree)
        namespace=dict(vars(traversal));namespace['_batch_operation']=_operation
        exec(compile(tree,'<batch-cooperative-'+identity[:12]+'>','exec'),namespace)
        _CACHE.clear();_CACHE[functions]=namespace['search_traversal']
    return _CACHE[functions](*args,**kwargs)


class CooperativeScheduler:
    """One host scheduler with one outstanding query per independently owned root."""
    def __init__(self,inference,*,max_roots=1600,allow_cpu_oracle=False,async_rule_dispatch=None):
        if type(max_roots) is not int or max_roots<1:raise ValueError('Positive root capacity required')
        if type(allow_cpu_oracle) is not bool:raise TypeError('CPU oracle opt-in must be boolean')
        if async_rule_dispatch is not None and type(async_rule_dispatch) is not bool:
            raise TypeError('async_rule_dispatch must be boolean or None')
        self.inference=inference;self.max_roots=max_roots;self.allow_cpu_oracle=allow_cpu_oracle
        self.async_rule_dispatch=async_rule_dispatch
        self._running=False
        self.dispatch_metrics={}
        self.progress_metrics=dict(pending_operations=0,ready_roots=0,idle_workers=None,
            peak_pending_operations=0,peak_ready_roots=0)

    def _record_dispatch(self,name,items,started):
        elapsed=time.perf_counter()-started
        row=self.dispatch_metrics.setdefault(name,dict(batches=0,items=0,wall_seconds=0.,max_seconds=0.))
        row['batches']+=1;row['items']+=items;row['wall_seconds']+=elapsed
        row['max_seconds']=max(row['max_seconds'],elapsed)

    def _use_async(self,backend):
        if getattr(backend,'read_only_finalization',False):
            return False
        capable=(bool(getattr(backend,'supports_async_operations',False))
                 and callable(getattr(backend,'submit_operations',None))
                 and callable(getattr(backend,'poll_operations',None)))
        if self.async_rule_dispatch is False:return False
        if self.async_rule_dispatch is True and not capable:
            raise RuntimeError('Explicit async rule dispatch requires submit_operations and poll_operations')
        return capable

    def _async_queries(self,queries):
        return [query for query in queries if isinstance(query,OperationQuery) and self._use_async(query.backend)]

    def _abandon_async(self,queries):
        grouped={}
        for query in queries:
            if not isinstance(query,OperationQuery):
                continue
            backend=query.backend
            capable=(bool(getattr(backend,'supports_async_operations',False))
                     and callable(getattr(backend,'submit_operations',None))
                     and callable(getattr(backend,'poll_operations',None))
                     and not getattr(backend,'read_only_finalization',False))
            if not capable:
                continue
            grouped.setdefault(id(query.backend),(query.backend,[]))[1].append(query.id)
        for backend,ids in grouped.values():
            cancel=getattr(backend,'cancel_operations',None)
            if callable(cancel) and ids:cancel(ids)

    def _wait_timeout(self,queries):
        now=time.monotonic();remain=[]
        for query in queries:
            deadline=getattr(query,'deadline',None)
            if deadline is not None and not isinstance(deadline,bool) and isinstance(deadline,(int,float)):
                remain.append(max(0.,deadline-now))
        if remain:return min(1.,min(remain))
        return 1.

    def _update_progress(self,queries,outstanding,responses):
        pending=len(outstanding)
        ready=sum(1 for query in queries if query.id in responses)
        idle=None
        seen=set()
        idle_total=0;have=False
        for query in queries:
            if not isinstance(query,OperationQuery) or id(query.backend) in seen:continue
            seen.add(id(query.backend))
            value=getattr(query.backend,'idle_workers',None)
            if type(value) is int:
                idle_total+=value;have=True
        if have:idle=idle_total
        self.progress_metrics['pending_operations']=pending
        self.progress_metrics['ready_roots']=ready
        self.progress_metrics['idle_workers']=idle
        self.progress_metrics['peak_pending_operations']=max(self.progress_metrics['peak_pending_operations'],pending)
        self.progress_metrics['peak_ready_roots']=max(self.progress_metrics['peak_ready_roots'],ready)

    def _poll_async(self,queries,outstanding,timeout):
        responses={}
        backends=[];seen=set()
        for query in queries:
            if query.id in outstanding and isinstance(query,OperationQuery) and self._use_async(query.backend):
                if id(query.backend) not in seen:
                    seen.add(id(query.backend));backends.append(query.backend)
        waited=False
        for backend in backends:
            slice_timeout=0. if waited else timeout
            started=time.perf_counter()
            rows=backend.poll_operations(timeout=slice_timeout)
            if slice_timeout>0:waited=True
            self._record_dispatch('rules.poll',len(rows),started)
            for row in rows:
                if not isinstance(row,OperationResponse):
                    raise ValueError('Bulk operation response identity mismatch')
                if row.id not in outstanding or row.id in responses:
                    raise ValueError('Bulk operation response identity mismatch')
                outstanding.discard(row.id)
                responses[row.id]=row.value if isinstance(row.value,RequestExpired) else row
        return responses

    def _dispatch_available(self,queries,outstanding):
        from .traversal import InferenceQuery
        if len({q.id for q in queries})!=len(queries):raise ValueError('Duplicate outstanding query identity')
        # Harvest ready terminal results before considering newly elapsed root
        # deadlines. Retire expired queued roots instead of spinning on them.
        responses=self._poll_async(queries,outstanding,0.)
        groups={};neural=[];async_groups={}
        now=time.monotonic()
        for query in queries:
            if query.id in responses:
                continue
            deadline=getattr(query,'deadline',None)
            expired=(not getattr(query,'finalization',False) and deadline is not None and now>=deadline)
            if query.id in outstanding:
                cancel=getattr(query.backend,'cancel_operations',None)
                if expired and callable(cancel):
                    cancel([query.id])
                    outstanding.discard(query.id)
                    responses[query.id]=RequestExpired(query.id)
                continue
            if expired:
                responses[query.id]=RequestExpired(query.id);continue
            if isinstance(query,OperationQuery):
                if self._use_async(query.backend):
                    async_groups.setdefault(id(query.backend),[]).append(query)
                else:
                    method=None if getattr(query.backend,'supports_mixed_operations',False) else query.method
                    groups.setdefault((id(query.backend),method),[]).append(query)
            elif isinstance(query,InferenceQuery):neural.append(query)
            else:raise TypeError('Unknown cooperative request type')
        for group in async_groups.values():
            backend=group[0].backend
            started=time.perf_counter()
            backend.submit_operations(group)
            for query in group:outstanding.add(query.id)
            prefix='statistics.' if getattr(backend,'read_only_finalization',False) else 'rules.'
            method=None if getattr(backend,'supports_mixed_operations',False) else group[0].method
            self._record_dispatch(prefix+(method or 'mixed')+'.submit',len(group),started)
        if groups:
            sync_queries=[query for group in groups.values() for query in group]
            responses.update(self._resolve(sync_queries))
        if neural:
            responses.update(self._resolve(neural))
        responses.update(self._poll_async(queries,outstanding,0.))
        return responses

    def _pump(self,queries,outstanding):
        completed=self._dispatch_available(queries,outstanding)
        if completed:return completed
        return self._poll_async(queries,outstanding,self._wait_timeout(queries))

    def run_stream(self,admit,completed,*,max_waves=100000,cancelled=None):
        """Refill a bounded window after publishing each wave's completed roots.

        ``admit`` constructs a root only when a slot is available; its deadline
        starts at admission. ``completed`` commits physical actions before the
        next admission, so a game's next generation cannot overlap its old one.
        No per-decision result history is retained in the scheduler.
        """
        if self._running:raise RuntimeError('Scheduler already owns an active run')
        if not callable(admit) or not callable(completed):raise TypeError('Admission and completion callbacks required')
        if type(max_waves) is not int or max_waves<1:raise ValueError('Positive wave limit required')
        import math
        from .traversal import TraversalCancelled,_DeadlineReached
        pending={};generations={};owned={};root_ids=set();waves=0;stalled_waves=0;count=0;peak=0;outstanding=set()
        start=time.monotonic();self._running=True
        def stopped():return cancelled is not None and cancelled()
        def publish(rows):
            nonlocal stalled_waves
            for job,_ in rows:
                owned.pop(job.game_id,None)
                job.generator.close()
            if rows:
                completed(rows)
                stalled_waves=0
        try:
            while True:
                immediate=[]
                while len(pending)<self.max_roots and not stopped():
                    job=admit()
                    if job is None:break
                    if (not isinstance(job,RootJob) or not isinstance(job.root_id,str) or not job.root_id
                        or not isinstance(job.game_id,str) or not job.game_id
                        or type(job.generation) is not int or job.generation<1
                        or not inspect.isgenerator(job.generator)):
                        raise ValueError('Versioned root jobs with generator traversal required')
                    # Include the new generator in cleanup even if validation fails.
                    if job.game_id in owned or job.root_id in root_ids or job.generation<=generations.get(job.game_id,0):
                        job.generator.close()
                        raise ValueError('Duplicate root, overlapping game, or stale generation')
                    owned[job.game_id]=job;root_ids.add(job.root_id);generations[job.game_id]=job.generation
                    if job.deadline is not None and (isinstance(job.deadline,bool)
                        or not isinstance(job.deadline,(int,float)) or not math.isfinite(job.deadline)):
                        raise ValueError('Finite monotonic root deadline required')
                    count+=1
                    try:pending[job.root_id]=(job,next(job.generator))
                    except StopIteration as stop:immediate.append((job,stop.value))
                peak=max(peak,len(pending))
                publish(immediate)
                if stopped():
                    finished=[];finalizing=[]
                    self._abandon_async([query for _,query in pending.values()]);outstanding.clear()
                    for job,_ in pending.values():
                        try:query=job.generator.throw(TraversalCancelled())
                        except StopIteration as stop:finished.append((job,stop.value))
                        except TraversalCancelled:finished.append((job,dict(stop_reason='cancelled_before_policy',complete=False,simulations=0)))
                        else:
                            if (not isinstance(query,OperationQuery) or not query.finalization
                                or query.method not in ('snapshot','final_summary')
                                or not getattr(query.backend,'read_only_finalization',False)):
                                raise RuntimeError('Traversal ignored root cancellation')
                            finalizing.append((job,query))
                    if finalizing:
                        responses=self._resolve([query for _,query in finalizing])
                        for job,query in finalizing:
                            try:job.generator.send(responses[query.id])
                            except StopIteration as stop:finished.append((job,stop.value))
                            else:raise RuntimeError('Traversal continued after final statistics snapshot')
                    pending.clear();publish(finished);break
                if not pending:
                    if immediate:continue
                    break
                # A continuous run may legitimately contain millions of waves.
                # Bound lack of progress, not the total work of all real games.
                if stalled_waves>=max_waves:raise TimeoutError('Cooperative wave allowance exhausted')
                queries=[replace(q,deadline=job.deadline) for job,q in pending.values()]
                if outstanding or self._async_queries(queries):
                    responses=self._pump(queries,outstanding)
                else:
                    responses=self._resolve(queries)
                waves+=1;stalled_waves+=1
                self._update_progress(queries,outstanding,responses)
                finished=[];next_pending={}
                for key,(job,query) in pending.items():
                    try:
                        if query.id not in responses:
                            next_pending[key]=(job,query);continue
                        if isinstance(responses[query.id],RequestExpired):
                            value=job.generator.throw(_DeadlineReached('Queued operation exceeded its root deadline'))
                        else:value=job.generator.send(responses[query.id])
                        next_pending[key]=(job,value)
                    except StopIteration as stop:finished.append((job,stop.value))
                pending=next_pending
                publish(finished)
            return dict(root_count=count,peak_outstanding_roots=peak,waves=waves,
                wall_seconds=time.monotonic()-start,cpu_oracle_opt_in=self.allow_cpu_oracle,
                dispatch_metrics=self.dispatch_metrics,async_rule_dispatch=self.async_rule_dispatch,
                progress_metrics=self.progress_metrics)
        finally:
            import sys
            original_error=sys.exc_info()[1]
            cleanup_errors=[]
            try:
                self._abandon_async([query for _,query in pending.values()] if pending else [])
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
            for job in owned.values():
                try:job.generator.close()
                except Exception as cleanup_error:cleanup_errors.append(cleanup_error)
            self._running=False
            if cleanup_errors:
                if original_error is None:raise cleanup_errors[0]
                if hasattr(original_error,'add_note'):
                    for cleanup_error in cleanup_errors:
                        original_error.add_note('Asynchronous cleanup also failed: '+repr(cleanup_error))

    def _resolve(self,queries):
        from .traversal import InferenceQuery,InferenceResponse
        from .inference import InferenceRequest
        if len({q.id for q in queries})!=len(queries):raise ValueError('Duplicate outstanding query identity')
        responses={};groups={};neural=[]
        for query in queries:
            deadline=getattr(query,'deadline',None)
            if not getattr(query,'finalization',False) and deadline is not None and time.monotonic()>=deadline:
                responses[query.id]=RequestExpired(query.id);continue
            if isinstance(query,OperationQuery):
                method=None if getattr(query.backend,'supports_mixed_operations',False) else query.method
                groups.setdefault((id(query.backend),method),[]).append(query)
            elif isinstance(query,InferenceQuery):neural.append(query)
            else:raise TypeError('Unknown cooperative request type')
        for (_,method),group in groups.items():
            fresh=[]
            for query in group:
                if (not query.finalization and query.deadline is not None and time.monotonic()>=query.deadline):
                    responses[query.id]=RequestExpired(query.id)
                else:fresh.append(query)
            group=fresh
            if not group:continue
            backend=group[0].backend
            dispatch_started=time.perf_counter()
            if callable(getattr(backend,'resolve_operations',None)):
                result=backend.resolve_operations(group)
                if len(result)!=len(group):raise ValueError('Bulk backend dropped or added operations')
                expected={q.id for q in group}
                for row in result:
                    if not isinstance(row,OperationResponse) or row.id not in expected or row.id in responses:
                        raise ValueError('Bulk operation response identity mismatch')
                    responses[row.id]=row.value if isinstance(row.value,RequestExpired) else row
                if expected!={row.id for row in result}:raise ValueError('Bulk operation response set mismatch')
            else:
                if not self.allow_cpu_oracle or getattr(backend,'supports_gpu',False):
                    raise RuntimeError('No bulk rule resolver; CPU oracle fallback is not authorized')
                if method not in _PROTOCOL_METHODS:raise ValueError('Unknown backend operation')
                for q in group:
                    responses[q.id]=OperationResponse(q.id,getattr(backend,method)(*q.args,**dict(q.kwargs)))
            prefix='statistics.' if getattr(backend,'read_only_finalization',False) else 'rules.'
            self._record_dispatch(prefix+(method or 'mixed'),len(group),dispatch_started)
        if neural:
            fresh=[]
            for query in neural:
                if query.deadline is not None and time.monotonic()>=query.deadline:
                    responses[query.id]=RequestExpired(query.id)
                else:fresh.append(query)
            neural=fresh
        if neural:
            requests=[InferenceRequest(q.id,q.policy_key,q.model_version,q.x,q.c,q.value_actor,q.value_only,q.deadline) for q in neural]
            dispatch_started=time.perf_counter()
            result=self.inference.predict(requests)
            self._record_dispatch('inference',len(requests),dispatch_started)
            if len(result)!=len(requests):raise ValueError('Inference dropped or added requests')
            expected={q.id:q for q in neural}
            for row in result:
                if row.request_id not in expected or row.request_id in responses:raise ValueError('Neural response identity mismatch')
                query=expected[row.request_id]
                if row.model_key!=query.policy_key or row.model_version!=query.model_version:
                    raise ValueError('Neural result model identity mismatch')
                responses[row.request_id]=(RequestExpired(query.id,True) if getattr(row,'expired',False) or query.deadline is not None and time.monotonic()>=query.deadline
                    else InferenceResponse(query.id,row.logits,row.wdl,row.model_version))
        if set(responses)!={q.id for q in queries}:raise ValueError('Incomplete batch response set')
        return responses

    def run(self,jobs,*,max_waves=100000,cancelled=None):
        if self._running:raise RuntimeError('Scheduler already owns an active run')
        jobs=list(jobs)
        if not jobs or len(jobs)>self.max_roots:raise ValueError('Nonempty root batch within capacity required')
        if type(max_waves) is not int or max_waves<1:raise ValueError('Positive wave limit required')
        roots=set();games=set()
        for job in jobs:
            if (not isinstance(job,RootJob) or not isinstance(job.root_id,str) or not job.root_id
                or not isinstance(job.game_id,str) or not job.game_id
                or type(job.generation) is not int or job.generation<1
                or not inspect.isgenerator(job.generator)):
                raise ValueError('Versioned root jobs with generator traversal required')
            if job.root_id in roots or job.game_id in games:raise ValueError('A game may own only one active root')
            if job.deadline is not None:
                import math
                if isinstance(job.deadline,bool) or not isinstance(job.deadline,(int,float)) or not math.isfinite(job.deadline):
                    raise ValueError('Finite monotonic root deadline required')
            roots.add(job.root_id);games.add(job.game_id)
        self._running=True;pending={};results={};start=time.monotonic();waves=0;outstanding=set()
        try:
            for job in jobs:
                try:pending[job.root_id]=(job,next(job.generator))
                except StopIteration as stop:results[job.root_id]=stop.value
            peak=len(pending)
            while pending:
                if waves>=max_waves:raise TimeoutError('Cooperative wave allowance exhausted')
                if cancelled is not None and cancelled():
                    from .traversal import TraversalCancelled
                    finishing=[]
                    self._abandon_async([query for _,query in pending.values()])
                    for key,(job,_) in list(pending.items()):
                        try:query=job.generator.throw(TraversalCancelled())
                        except StopIteration as stop:results[key]=stop.value
                        except TraversalCancelled:results[key]=dict(stop_reason='cancelled_before_policy',complete=False,simulations=0)
                        else:
                            if (not isinstance(query,OperationQuery) or not query.finalization or
                                query.method not in ('snapshot','final_summary') or not getattr(query.backend,'read_only_finalization',False)):
                                raise RuntimeError('Traversal ignored root cancellation')
                            finishing.append((key,job,query))
                    if finishing:
                        final=self._resolve([query for _,_,query in finishing])
                        for key,job,query in finishing:
                            try:job.generator.send(final[query.id])
                            except StopIteration as stop:results[key]=stop.value
                            else:raise RuntimeError('Traversal continued after final statistics snapshot')
                    pending.clear();break
                queries=[replace(q,deadline=job.deadline) for job,q in pending.values()]
                if outstanding or self._async_queries(queries):
                    responses=self._pump(queries,outstanding)
                else:
                    responses=self._resolve(queries)
                waves+=1
                self._update_progress(queries,outstanding,responses)
                next_pending={}
                for key,(job,query) in pending.items():
                    try:
                        if query.id not in responses:
                            next_pending[key]=(job,query);continue
                        if isinstance(responses[query.id],RequestExpired):
                            from .traversal import _DeadlineReached
                            value=job.generator.throw(_DeadlineReached('Queued operation exceeded its root deadline'))
                        else:value=job.generator.send(responses[query.id])
                        next_pending[key]=(job,value)
                    except StopIteration as stop:results[key]=stop.value
                pending=next_pending
            if set(results)!=roots:raise ValueError('A root disappeared from scheduler accounting')
            return dict(results=results,root_count=len(jobs),peak_outstanding_roots=peak,waves=waves,
                        wall_seconds=time.monotonic()-start,cpu_oracle_opt_in=self.allow_cpu_oracle,
                        async_rule_dispatch=self.async_rule_dispatch,progress_metrics=self.progress_metrics)
        finally:
            self._abandon_async([query for _,query in pending.values()] if pending else [])
            for job in jobs:job.generator.close()
            self._running=False
