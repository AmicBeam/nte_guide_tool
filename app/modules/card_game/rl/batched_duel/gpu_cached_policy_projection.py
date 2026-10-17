"""One public-query scope indexes pure modifier effects by property.

Canonical predicates, aura providers, reduction order and formulas are reused.
The index lives only in this projection call; no Game field or global cache is
written. This optional capture entry requires independent numerical/CUDA proof.
"""
import ast
from copy import deepcopy
from ...engine.duel_v2 import modifiers,projection,state
from .gpu_policy_projection import (
    _compile,_Observe,_policy_card,_policy_state,_REQUIRED,ProjectionSpecializationError)


def _indexed_effects(game,side,subject,property,_cache):
    index=_cache['property_effects'] if game is _cache['state'] and game is not None else None
    if index is None:
        index={}
        holders=list(modifiers.entities(game)) if game is not None else [(side,subject)]
        for owner_side,holder in holders:
            for effect in modifiers.all_effects(holder):
                index.setdefault(effect['property'],[]).append((owner_side,holder,effect))
        if game is _cache['state'] and game is not None:
            _cache['property_effects']=index
    return index.get(property,[])


class _QueryConsumer(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        node.args.kwonlyargs.append(ast.arg(arg='_cache'));node.args.kw_defaults.append(None)
        return self.generic_visit(node)
    def visit_ImportFrom(self,node):
        if node.module=='modifiers':
            node.names=[n for n in node.names if n.name!='evaluate']
            if not node.names:return None
        return node
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id=='evaluate':
            node.func.id='_indexed_evaluate'
            node.keywords.append(ast.keyword(arg='_cache',value=ast.Name('_cache',ast.Load())))
        return self.generic_visit(node)


class _IndexedEvaluate(_QueryConsumer):
    def __init__(self):self.assignments=0;self.loops=0
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='holders':
            self.assignments+=1;return None
        return self.generic_visit(node)
    def visit_For(self,node):
        if isinstance(node.iter,ast.Name) and node.iter.id=='holders':
            if (ast.dump(node.target)!=ast.dump(ast.parse('for holder_side,holder in []: pass').body[0].target)
                    or len(node.body)!=1 or not isinstance(node.body[0],ast.For)
                    or ast.dump(node.body[0].iter)!=ast.dump(ast.parse('all_effects(holder)',mode='eval').body)
                    or not isinstance(node.body[0].target,ast.Name) or node.body[0].target.id!='effect'
                    or node.orelse or node.body[0].orelse):
                raise ProjectionSpecializationError('Canonical modifier iteration changed')
            self.loops+=1
            node.target=ast.Tuple(elts=[ast.Name(v,ast.Store()) for v in ('holder_side','holder','effect')],ctx=ast.Store())
            node.iter=ast.Call(func=ast.Name('_indexed_effects',ast.Load()),
                args=[ast.Name(v,ast.Load()) for v in ('state','side','subject','property','_cache')],keywords=[])
            node.body=node.body[0].body
        return self.generic_visit(node)


_transform=_IndexedEvaluate()
_indexed_evaluate,_=_compile(modifiers.evaluate,'_indexed_evaluate',_transform,{'_indexed_effects':_indexed_effects})
if (_transform.assignments,_transform.loops)!=(1,1):
    raise ProjectionSpecializationError('Exactly one canonical modifier inventory required')
_queries={'_indexed_evaluate':_indexed_evaluate}
_query_names={}
for _name in ('attack_value','damage_immune','damage_limit'):
    _alias='_query_'+_name
    _queries[_alias],_=_compile(getattr(state,_name),_alias,_QueryConsumer(),_queries)
    _query_names[_name]=_alias


def _query_call(node):
    if isinstance(node.func,ast.Name) and node.func.id in _query_names:
        node.func.id=_query_names[node.func.id]
        node.keywords.append(ast.keyword(arg='_cache',value=ast.Name('_cache',ast.Load())))
    return node


class _IndexedObserve(_Observe):
    def visit_Assign(self,node):
        initialize=(len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='actions')
        result=super().visit_Assign(node)
        # The base specialization emits a new attack_value assignment directly.
        # Visit its generated call without running its assignment sites twice.
        for part in result if isinstance(result,list) else [result]:
            if part is not None:
                for item in ast.walk(part):
                    if isinstance(item,ast.Call):_query_call(item)
        if initialize:
            self.hit('query_cache')
            return [result,ast.copy_location(ast.parse("_cache = {'state': state, 'property_effects': None}").body[0],node)]
        return result
    def visit_Call(self,node):
        return _query_call(super().visit_Call(node))


_observe_transform=_IndexedObserve()
_cached_observe,_=_compile(projection.observe,'_cached_observe',_observe_transform,
    dict(_queries,public_card=_policy_card,_policy_state=_policy_state))
if any(_observe_transform.counts.get(k)!=v for k,v in dict(_REQUIRED,query_cache=1).items()):
    raise ProjectionSpecializationError('Canonical indexed projection sites changed')


def cached_policy_public_legal_decision(game,side):
    view=_cached_observe(game,side,include_previews=False)
    return [deepcopy(entry['action']) for entry in view['legal_actions']],view
