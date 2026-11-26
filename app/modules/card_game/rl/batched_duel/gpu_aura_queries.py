"""Opt-in empty-aura generator guard derived from canonical evaluate.

Context construction is ALWAYS retained. The frozen VM has no complete
function.__code__/class-descriptor runtime guard, so this module does not claim
safe pre-Context elision. No effects, predicates or provider results are cached.
The default capture pipeline is unchanged; capture_aura_evaluate is explicit.
"""
import ast
from hashlib import sha256
import inspect
import textwrap

from ...engine.duel_v2 import modifiers,context
from ...content.duel_v2.characters.bohe import Bohe
from ...content.duel_v2.characters.xiaozhi import Xiaozhi
from ...content.duel_v2.characters.adler import Adler
from .gpu_policy_projection import _compile


class AuraQuerySpecializationError(RuntimeError):
    pass


_ORIGINAL_EVALUATE=modifiers.evaluate
_EVALUATE_CODE=_ORIGINAL_EVALUATE.__code__
_CONTEXT=context.EffectContext
_CTOR=_CONTEXT.__init__
_CTOR_CODE=_CTOR.__code__
_BOHE=Bohe.aura_effects
_XIAOZHI=Xiaozhi.aura_effects
_ADLER=Adler.aura_effects
_PROVIDERS=((Bohe,_BOHE,_BOHE.__code__,'attack.override'),
            (Xiaozhi,_XIAOZHI,_XIAOZHI.__code__,'attack.hit'),
            (Adler,_ADLER,_ADLER.__code__,'attack.panel'))
_HELPERS=tuple((name,getattr(modifiers,name),getattr(modifiers,name).__code__)
    for name in ('entities','all_effects','_applies','reduce_modifiers'))
_EXPECTED_AST=(
    '6b2c037a3d9f29bf6c99a4cb1e474f0652439a33b29e7e21f785d002c45cb677',
    '749197ca7902f33c237633f05ddec7120a8453f516b1be42933a8d66f30a2d0c',
    '825169317dcc0ae68ea5e05b57eeb1b0f5e883e54c956159079f94e1d59a0e31',
    'f8c2ed960c3a1112a3dadd0fc7094fbb3e05bc7f70871c24d0e7b2a71b6c1868',
    '0a19324bce5c02a121b363f071c3f47cacd549837eaa3e227c07ce505127d852')


def _dump(tree):
    for node in ast.walk(tree):
        if 'type_params' in node._fields or hasattr(node,'type_params'):
            value=getattr(node,'type_params',None)
            if type(value) is not list or value:
                raise AuraQuerySpecializationError('Unknown generic source shape')
            node._fields=tuple(name for name in node._fields if name!='type_params')
    options=dict(include_attributes=False)
    if 'show_empty' in inspect.signature(ast.dump).parameters:options['show_empty']=True
    return ast.dump(tree,**options)


def _source_tree(fn):
    try:return ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError,TypeError,SyntaxError) as error:
        raise AuraQuerySpecializationError('Unavailable canonical aura source') from error


def _empty_property(fn):
    """Only a bare-return first if whose leftmost test is property != str."""
    tree=_source_tree(fn)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise AuraQuerySpecializationError('Single canonical provider definition required')
    statements=tree.body[0].body
    if statements and isinstance(statements[0],ast.Expr) and isinstance(statements[0].value,ast.Constant) and isinstance(statements[0].value.value,str):
        statements=statements[1:]
    if not statements or not isinstance(statements[0],ast.If):
        raise AuraQuerySpecializationError('Provider has work before its property guard')
    first=statements[0]
    if first.orelse or len(first.body)!=1 or not isinstance(first.body[0],ast.Return) or first.body[0].value is not None:
        raise AuraQuerySpecializationError('Only an unconditional bare-return guard is eligible')
    test=first.test
    if isinstance(test,ast.BoolOp) and isinstance(test.op,ast.Or):test=test.values[0]
    if not (isinstance(test,ast.Compare) and isinstance(test.left,ast.Name) and test.left.id=='property'
            and len(test.ops)==len(test.comparators)==1 and isinstance(test.ops[0],ast.NotEq)
            and isinstance(test.comparators[0],ast.Constant) and type(test.comparators[0].value) is str):
        raise AuraQuerySpecializationError('Leftmost property != literal guard required')
    if not inspect.isgeneratorfunction(fn):
        raise AuraQuerySpecializationError('Only canonical generator providers are eligible')
    return test.comparators[0].value


def validate_aura_query_sources():
    """Host-only admission immediately before explicit frozen capture/CPU use."""
    if modifiers.evaluate is not _ORIGINAL_EVALUATE or _ORIGINAL_EVALUATE.__code__ is not _EVALUATE_CODE:
        raise AuraQuerySpecializationError('Canonical evaluate object/code changed')
    if context.EffectContext is not _CONTEXT or _CONTEXT.__init__ is not _CTOR or _CTOR.__code__ is not _CTOR_CODE:
        raise AuraQuerySpecializationError('Canonical Context constructor object/code changed')
    if (type(_CONTEXT) is not type or _CONTEXT.__mro__!=(_CONTEXT,object)
            or _CONTEXT.__new__ is not object.__new__ or _CONTEXT.__setattr__ is not object.__setattr__
            or _CONTEXT.__getattribute__ is not object.__getattribute__):
        raise AuraQuerySpecializationError('Custom Context allocation/attribute behavior')
    if any(name in cls.__dict__ for cls in _CONTEXT.__mro__ for name in ('state','side','cid','operation','card')):
        raise AuraQuerySpecializationError('Context assigned field gained a descriptor/default')
    for name,fn,code in _HELPERS:
        if getattr(modifiers,name) is not fn or fn.__code__ is not code:
            raise AuraQuerySpecializationError('Canonical evaluate dependency changed: '+name)
    if any(actual is not item[1]for actual,item in zip((_BOHE,_XIAOZHI,_ADLER),_PROVIDERS)):
        raise AuraQuerySpecializationError('Provider identity bindings changed')
    for cls,fn,code,property in _PROVIDERS:
        descriptor=inspect.getattr_static(cls,'aura_effects')
        if type(descriptor) is not staticmethod or descriptor.__func__ is not fn or fn.__code__ is not code:
            raise AuraQuerySpecializationError('Canonical provider object/code changed: '+cls.__name__)
        if _empty_property(fn)!=property:
            raise AuraQuerySpecializationError('Canonical empty property changed')
    functions=(_ORIGINAL_EVALUATE,_CTOR,_BOHE,_XIAOZHI,_ADLER)
    for fn,expected in zip(functions,_EXPECTED_AST):
        if sha256(_dump(_source_tree(fn)).encode()).hexdigest()!=expected:
            raise AuraQuerySpecializationError('Canonical source shape changed: '+fn.__qualname__)
    return dict(context_elision=False,generator_elision=True,default_capture_enabled=False,gpu_qualified=False)


def _host_skip_aura(provider,property,query,operation,plain_query):
    for _,fn,code,literal in _PROVIDERS:
        if provider is fn:
            if fn.__code__ is not code:raise AuraQuerySpecializationError('Provider code changed during query')
            return (plain_query and type(property) is str and type(query) is dict
                and (operation is None or type(operation) is dict) and property!=literal)
    return False


def _frozen_skip_aura(provider,property,query,operation,plain_query):
    # Only function-reference identity is required inside the immutable frozen
    # program. __code__ writes are unsupported there; host source validation is
    # mandatory before freezing. Context is retained even for skipped providers.
    if not plain_query or type(property) is not str or type(query) is not dict:
        return False
    if operation is not None and type(operation) is not dict:return False
    if provider is _BOHE:return property!='attack.override'
    if provider is _XIAOZHI:return property!='attack.hit'
    if provider is _ADLER:return property!='attack.panel'
    return False


class _AuraEvaluate(ast.NodeTransformer):
    def __init__(self):self.sites=0
    def visit_FunctionDef(self,node):
        if node.name=='_aura_evaluate':
            node.body[:0]=ast.parse('_aura_plain_query = query is None or type(query) is dict').body
        return self.generic_visit(node)
    def visit_If(self,node):
        if isinstance(node.test,ast.Name) and node.test.id=='provider':
            expected=ast.parse("c = EffectContext(state, holder_side, cid, query.get('operation') or {'side': holder_side})").body[0]
            if (len(node.body)!=2 or _dump(node.body[0])!=_dump(expected)
                    or not isinstance(node.body[1],ast.For)
                    or _dump(node.body[1].iter)!=_dump(ast.parse('provider(c, property=property, query=query)',mode='eval').body)
                    or node.orelse):
                raise AuraQuerySpecializationError('Canonical provider loop changed')
            self.sites+=1
            # Preserve callee/argument evaluation order and the ONE query.get;
            # retain constructor, operation truth/get effects and every context.
            node.body=ast.parse("""
_aura_ctor = EffectContext
_aura_state, _aura_side, _aura_cid = state, holder_side, cid
_aura_raw_operation = query.get('operation')
_aura_operation = _aura_raw_operation or {'side': holder_side}
c = _aura_ctor(_aura_state, _aura_side, _aura_cid, _aura_operation)
if _aura_skip(provider, property, query, _aura_raw_operation, _aura_plain_query):
    continue
""").body+[node.body[1]]
        return self.generic_visit(node)


def _derive(skip):
    validate_aura_query_sources()
    transform=_AuraEvaluate()
    result,_=_compile(_ORIGINAL_EVALUATE,'_aura_evaluate',transform,{'_aura_skip':skip})
    if transform.sites!=1:raise AuraQuerySpecializationError('Exactly one canonical provider loop required')
    result.__module__=__name__
    return result


_CPU_EVALUATE=_derive(_host_skip_aura)


def query_evaluate(state,side,subject,property,base,*,query=None,extra=()):
    """CPU parity/reference interface, validating mutable Python code each call."""
    validate_aura_query_sources()
    return _CPU_EVALUATE(state,side,subject,property,base,query=query,extra=extra)


def capture_aura_evaluate():
    """Return explicit immutable-VM replacement after host source admission.

    Caller must freeze immediately and include this source in its source-hash
    contract. This factory itself is not a rule entry or a qualification flag.
    """
    return _derive(_frozen_skip_aura)
