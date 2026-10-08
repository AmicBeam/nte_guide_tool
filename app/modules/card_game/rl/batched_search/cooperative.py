"""Cooperative rule-operation suspension and bounded cross-root dispatch.

Only calls to the explicit backend protocol are rewritten. Rule code, search
formulas, RNG streams and model inputs are not rewritten. CPU oracle dispatch
requires an explicit opt-in; a GPU backend must provide a bulk resolver.
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
    'sample_world','step','clean','value_observation','serving_root_metadata','public_terminal_probe'))
_PREFIX=uuid4().hex
_COUNTER=itertools.count()
_CACHE={}


def _operation(backend,method,*args,_finalization=False,**kwargs):
    if type(_finalization) is not bool:raise TypeError('Boolean finalization marker required')
    inline=getattr(backend,'inline_operation',None)
    if inline is not None and not kwargs and not _finalization:
        handled,value=inline(method,args)
        if handled:return value
    query=OperationQuery(f'{_PREFIX}:{next(_COUNTER)}',backend,method,args,tuple(kwargs.items()),_finalization)
    response=yield query
    if not isinstance(response,OperationResponse) or response.id!=query.id:
        raise ValueError('Rule operation response belongs to a different request')
    return response.value


class _SuspendBackend(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='backend' and node.func.attr in _PROTOCOL_METHODS):
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
    names=('_follow_terminal_generator','_short_terminal_value_generator','search_traversal')
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
    def __init__(self,inference,*,max_roots=1600,allow_cpu_oracle=False):
        if type(max_roots) is not int or max_roots<1:raise ValueError('Positive root capacity required')
        if type(allow_cpu_oracle) is not bool:raise TypeError('CPU oracle opt-in must be boolean')
        self.inference=inference;self.max_roots=max_roots;self.allow_cpu_oracle=allow_cpu_oracle
        self._running=False

    def _resolve(self,queries):
        from .traversal import InferenceQuery,InferenceResponse
        from .inference import InferenceRequest
        if len({q.id for q in queries})!=len(queries):raise ValueError('Duplicate outstanding query identity')
        responses={};groups={};neural=[]
        for query in queries:
            deadline=getattr(query,'deadline',None)
            if not getattr(query,'finalization',False) and deadline is not None and time.monotonic()>=deadline:
                responses[query.id]=RequestExpired(query.id);continue
            if isinstance(query,OperationQuery):groups.setdefault((id(query.backend),query.method),[]).append(query)
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
        if neural:
            fresh=[]
            for query in neural:
                if query.deadline is not None and time.monotonic()>=query.deadline:
                    responses[query.id]=RequestExpired(query.id)
                else:fresh.append(query)
            neural=fresh
        if neural:
            requests=[InferenceRequest(q.id,q.policy_key,q.model_version,q.x,q.c,q.value_actor,q.value_only,q.deadline) for q in neural]
            result=self.inference.predict(requests)
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
        self._running=True;pending={};results={};start=time.monotonic();waves=0
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
                responses=self._resolve(queries);waves+=1
                next_pending={}
                for key,(job,query) in pending.items():
                    try:
                        if isinstance(responses[query.id],RequestExpired):
                            from .traversal import _DeadlineReached
                            value=job.generator.throw(_DeadlineReached('Queued operation exceeded its root deadline'))
                        else:value=job.generator.send(responses[query.id])
                        next_pending[key]=(job,value)
                    except StopIteration as stop:results[key]=stop.value
                pending=next_pending
            if set(results)!=roots:raise ValueError('A root disappeared from scheduler accounting')
            return dict(results=results,root_count=len(jobs),peak_outstanding_roots=peak,waves=waves,
                        wall_seconds=time.monotonic()-start,cpu_oracle_opt_in=self.allow_cpu_oracle)
        finally:
            for job in jobs:job.generator.close()
            self._running=False
