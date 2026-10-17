"""Derive an encoder-only projection from canonical public projection source.

No rule formula or visibility branch is reimplemented. The specialization drops
presentation-only work and reuses the already computed compatibility flags.
Canonical observe and all website callers remain unchanged. Numerical/legal
parity and actual CUDA execution are separate required checks.
"""
import ast
import inspect
import importlib.util
import textwrap
from copy import deepcopy

from app.modules.card_game.engine.duel_v2 import projection, presentation, policy_state


class ProjectionSpecializationError(RuntimeError):
    pass


def _key(node):
    if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name):
        index=node.slice
        if isinstance(index,ast.Constant):return node.value.id,index.value
    return None


DISPLAY_FIELDS = frozenset(('name', 'description', 'resource_name', 'faction',
    'avatar', 'portrait', 'passive', 'awakened_passive', 'flavor'))


def _trim_display_tuple(node):
    if isinstance(node, ast.Tuple) and all(isinstance(v, ast.Constant) and isinstance(v.value, str)
                                         for v in node.elts):
        node.elts = [v for v in node.elts if v.value not in DISPLAY_FIELDS]
    return node


class _Observe(ast.NodeTransformer):
    def __init__(self):self.counts={}
    def hit(self,key):self.counts[key]=self.counts.get(key,0)+1
    def visit_Name(self,node):
        if node.id=='include_previews' and isinstance(node.ctx,ast.Load):
            return ast.copy_location(ast.Constant(False),node)
        return node
    def visit_FunctionDef(self,node):
        if node.name=='legal_entry':
            if [arg.arg for arg in node.args.args]!=['action','preview_value','interaction']:
                raise ProjectionSpecializationError('Canonical legal entry signature changed')
            value=node.body[0].value
            if not isinstance(value,ast.Dict) or {k.value for k in value.keys}!= {'action','entity_action','label','preview','interaction'}:
                raise ProjectionSpecializationError('Canonical legal entry fields changed')
            node.args.args=[ast.arg(arg='action')]
            node.body=[ast.copy_location(ast.Return(ast.Dict(keys=[ast.Constant('action')],values=[ast.Call(func=ast.Name('deepcopy',ast.Load()),args=[ast.Name('action',ast.Load())],keywords=[])])),node.body[0])]
            self.hit('entry');return node
        return self.generic_visit(node)
    def visit_Assign(self,node):
        target=node.targets[0] if len(node.targets)==1 else None
        if isinstance(target,ast.Name) and target.id in ('card_index','playable','form','template'):
            self.hit(target.id);return None
        field=_key(target)
        if field==('c','attack_breakdown'):
            self.hit('attack')
            return ast.copy_location(ast.Assign(targets=[ast.Subscript(ast.Name('c',ast.Load()),ast.Constant('attack'),ast.Store())],value=ast.Call(ast.Name('attack_value',ast.Load()),args=[ast.Name('h',ast.Load()),ast.Name('state',ast.Load()),ast.Name('ps',ast.Load())],keywords=[])),node)
        if field in {('c','stat_layers'),('c','equipment'),('c','effect_markers'),('team','effect_markers')}:
            self.hit(str(field));return None
        if isinstance(target,ast.Name) and target.id=='flags':
            self.hit('cache_flags')
            assignment=ast.parse('_flags[ps][cid] = flags').body[0]
            return [self.generic_visit(node),ast.copy_location(assignment,node)]
        return self.generic_visit(node)
    def visit_If(self,node):
        if isinstance(node.test,ast.Call) and ast.unparse(node.test)=="h.get('summoned')":
            if len(node.body)!=2 or _key(node.body[-1].targets[0])!=('c','passive'):
                raise ProjectionSpecializationError('Summon text branch gained non-display work')
            self.hit('summon_text');return None
        if any(isinstance(n,ast.Name) and n.id=='template' for n in ast.walk(node.test)):
            if (len(node.body)!=1 or not isinstance(node.body[0],ast.For)
                    or ast.literal_eval(node.body[0].iter)!=('passive','awakened_passive')):
                raise ProjectionSpecializationError('Current passive text branch changed')
            self.hit('template_text');return None
        if isinstance(node.test,ast.Name) and node.test.id=='form':
            self.hit('form_if');return None
        if any(isinstance(n,ast.Name) and n.id=='playable' for n in ast.walk(node.test)):
            fields=[_key(n.targets[0]) for n in ast.walk(node) if isinstance(n,ast.Assign) and len(n.targets)==1]
            if ('card','unavailable_reason') not in fields:
                raise ProjectionSpecializationError('Canonical unavailable-card branch changed')
            self.hit('unavailable');return None
        return self.generic_visit(node)
    def visit_For(self,node):
        if isinstance(node.target,ast.Name) and node.target.id=='ps':
            node=self.generic_visit(node)
            first=ast.copy_location(ast.parse('_flags = {}').body[0],node)
            node.body.insert(0,ast.copy_location(ast.parse('_flags[ps] = {}').body[0],node))
            self.hit('cache_sides');return [first,node]
        return self.generic_visit(node)
    def visit_ImportFrom(self,node):
        if (node.module=='tutorial' and [n.name for n in node.names]==['is_tutorial']
                or node.module=='equipment' and [n.name for n in node.names]==['stat_layers','public_equipment']
                or node.module=='app.modules.card_game.content.duel_v2.tutorial' and [n.name for n in node.names]==['lookup_card']):
            return None
        if node.module=='policy_state' and any(n.name=='public_policy_state' for n in node.names):
            self.hit('policy_import');return None
        return node
    def visit_DictComp(self,node):
        # Only specialize the canonical copied team/character field tuples.
        if (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                and node.value.func.id=='deepcopy' and len(node.generators)==1
                and isinstance(node.value.args[0],ast.Subscript)
                and isinstance(node.value.args[0].value,ast.Name)
                and node.value.args[0].value.id in ('source','h')):
            node.generators[0].iter=_trim_display_tuple(node.generators[0].iter)
            self.hit('display_fields')
        return self.generic_visit(node)
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id=='legal_entry':
            if len(node.args)!=3:raise ProjectionSpecializationError('Canonical legal call changed')
            node.args=node.args[:1]
        elif isinstance(node.func,ast.Name) and node.func.id=='public_policy_state':
            node.func=ast.Name('_policy_state',ast.Load());node.args.append(ast.Name('_flags',ast.Load()));self.hit('policy_call')
        elif isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='c' and node.func.attr=='update':
            removed={'attack','shape','shape_description'}
            node.keywords=[k for k in node.keywords if k.arg not in removed]
            self.hit('character_update')
        return self.generic_visit(node)


class _Card(ast.NodeTransformer):
    def visit_DictComp(self,node):
        for generator in node.generators:
            generator.iter=_trim_display_tuple(generator.iter)
        return self.generic_visit(node)
    def visit_If(self,node):
        fields=[_key(n.targets[0]) for n in ast.walk(node) if isinstance(n,ast.Assign) and len(n.targets)==1]
        if ('result','description') in fields:
            if any(field is not None and field!=('result','description') for field in fields):
                raise ProjectionSpecializationError('Description branch gained non-display output')
            return None
        if ('result','action_point_free') in fields:
            if any(field is not None and field!=('result','action_point_free') for field in fields):
                raise ProjectionSpecializationError('Cost-display branch gained numerical output')
            return None
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if len(node.targets)==1:
            target=node.targets[0]
            if _key(target)==('result','action_point_free') or isinstance(target,ast.Name) and target.id=='extra_ap':return None
        return self.generic_visit(node)


class _PolicyState(ast.NodeTransformer):
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id=='compatibility_flags':
            return ast.copy_location(ast.parse('_flags[side][cid]',mode='eval').body,node)
        return self.generic_visit(node)
    def visit_Dict(self,node):
        keep=[i for i,k in enumerate(node.keys) if not isinstance(k,ast.Constant) or k.value not in ('stat_layers','equipment')]
        node.keys=[node.keys[i] for i in keep];node.values=[node.values[i] for i in keep]
        return self.generic_visit(node)


def _compile(original,name,transform,extra=None,extra_arg=None):
    lines,start=inspect.getsourcelines(original)
    tree=ast.parse(textwrap.dedent(''.join(lines)))
    fn=tree.body[0]
    if not isinstance(fn,ast.FunctionDef):raise ProjectionSpecializationError('Canonical function definition required')
    fn.name=name
    if extra_arg:fn.args.args.append(ast.arg(arg=extra_arg))
    tree=transform.visit(tree)
    package=original.__globals__.get('__package__') or original.__module__.rpartition('.')[0]
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.level:
            node.module=importlib.util.resolve_name('.'*node.level+(node.module or ''),package)
            node.level=0
    ast.fix_missing_locations(tree);ast.increment_lineno(tree,start-1)
    namespace=dict(original.__globals__)
    namespace.update(extra or {})
    exec(compile(tree,inspect.getsourcefile(original),'exec'),namespace)
    result=namespace[name];result.__module__=__name__
    return result,namespace


_policy_card,_card_globals=_compile(presentation.public_card,'_policy_card',_Card(),
    {'CARD_FIELDS':tuple(k for k in presentation.CARD_FIELDS if k not in DISPLAY_FIELDS)})
_card_globals['public_card']=_policy_card
_policy_state,_state_globals=_compile(policy_state.public_policy_state,'_policy_state',_PolicyState(),extra_arg='_flags')
_observe_transform=_Observe()
_policy_observe,_observe_globals=_compile(projection.observe,'_policy_observe',_observe_transform,
    {'public_card':_policy_card,'_policy_state':_policy_state})
_REQUIRED={'entry':1,'card_index':1,'playable':1,'form':1,'form_if':1,'attack':1,'cache_flags':1,'cache_sides':1,'policy_import':1,'policy_call':1,'unavailable':1,'character_update':1,
           'template':1,'template_text':1,'summon_text':1,'display_fields':2}
if any(_observe_transform.counts.get(key)!=count for key,count in _REQUIRED.items()):
    raise ProjectionSpecializationError('Canonical projection specialization sites changed')


def policy_public_legal_decision(state,side):
    view=_policy_observe(state,side,include_previews=False)
    return [deepcopy(entry['action']) for entry in view['legal_actions']],view


def policy_projection_probe(packet):
    actions,view=policy_public_legal_decision(packet['state'],packet['side'])
    return actions,view,packet['state']
