"""Canonical transitions for disposable VM scratch, never persistent roots.

DeviceRuleGraphCalls first gathers immutable leased state into a separate VM
arena. These explicitly selected capture inputs consume that arena, retaining
canonical hydration, validation, effects, events and tutorial branches while
avoiding a second full state graph copy. Do not call them on a shared state.
"""
import ast
import inspect
import textwrap
from app.modules.card_game.engine.duel_v2 import flow, simulation
from app.modules.card_game.engine.duel_v2.entities import hydrate_entities, clone_state
from .gpu_policy_projection import _compile, ProjectionSpecializationError


class _OwnedClone(ast.NodeTransformer):
    def __init__(self):self.sites=0
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id=='clone_state':
            if (len(node.args)!=1 or not isinstance(node.args[0],ast.Name)
                    or node.args[0].id!='state' or node.keywords):
                raise ProjectionSpecializationError('Canonical state clone changed')
            self.sites+=1
            node.func=ast.copy_location(ast.Name('hydrate_entities',ast.Load()),node.func)
        return self.generic_visit(node)


class _OwnedSimulation(ast.NodeTransformer):
    def __init__(self):self.sites=0
    def visit_ImportFrom(self,node):
        if node.module=='flow' and any(a.name=='apply_action' for a in node.names):
            if node.level!=1 or len(node.names)!=1 or node.names[0].asname:
                raise ProjectionSpecializationError('Canonical simulation import changed')
            self.sites+=1
            return None
        return node


_clone_body=ast.parse(textwrap.dedent(inspect.getsource(clone_state))).body[0].body
_expected=ast.parse('return hydrate_entities(copy_state_graph(state))').body
if ast.dump(ast.Module(body=_clone_body,type_ignores=[]))!=ast.dump(ast.Module(body=_expected,type_ignores=[])):
    raise ProjectionSpecializationError('Canonical clone gained non-copy behavior')
_clone=_OwnedClone()
scratch_apply_action,_apply_globals=_compile(flow.apply_action,'scratch_apply_action',_clone,
    {'hydrate_entities':hydrate_entities})
if _clone.sites!=1:raise ProjectionSpecializationError('Exactly one canonical clone required')
scratch_apply_action.__module__=__name__
_simulation=_OwnedSimulation()
scratch_simulate_action,_simulation_globals=_compile(simulation.simulate_action,
    'scratch_simulate_action',_simulation,{'apply_action':scratch_apply_action})
if _simulation.sites!=1:raise ProjectionSpecializationError('Exactly one simulation apply import required')
scratch_simulate_action.__module__=__name__
