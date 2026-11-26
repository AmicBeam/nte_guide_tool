"""AST-derived public value inputs without legal-action enumeration.

Only the value-query entry may use this view: legal_actions is deliberately
empty. Policy/action consumers retain the original cached projection. Formula,
visibility, hydration and per-query modifier indexing are inherited unchanged.
"""
import ast

from ...engine.duel_v2 import projection
from .gpu_cached_policy_projection import _IndexedObserve, _queries
from .gpu_policy_projection import (
    _compile, _policy_card, _policy_state, _REQUIRED, ProjectionSpecializationError,
)


def _same(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


class _ValueObserve(_IndexedObserve):
    def __init__(self):
        super().__init__()
        self.value_sites = dict(actions=0, entries=0, entry_function=0)

    def visit_Assign(self, node):
        actions = (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == 'actions')
        if actions:
            expected = ast.parse('actions = legal_actions(state, side)').body[0]
            if not _same(node, expected):
                raise ProjectionSpecializationError('Value-query legal enumeration changed')
            self.value_sites['actions'] += 1
        result = super().visit_Assign(node)
        if actions:
            # Keep the inherited cache initialization at precisely this site.
            if not isinstance(result, list) or len(result) != 2 or result[0] is not node:
                raise ProjectionSpecializationError('Indexed query initialization changed')
            node.value = ast.copy_location(ast.List(elts=[], ctx=ast.Load()), node.value)
        return result

    def visit_FunctionDef(self, node):
        if node.name == 'legal_entry':
            result = super().visit_FunctionDef(node)
            if not isinstance(result, ast.FunctionDef):
                raise ProjectionSpecializationError('Policy legal-entry derivation changed')
            self.value_sites['entry_function'] += 1
            return None
        result = super().visit_FunctionDef(node)
        if any(isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load)
               and part.id in ('actions', 'legal_actions', 'legal_entry')
               for part in ast.walk(result)):
            raise ProjectionSpecializationError('Value view gained an action dependency')
        return result

    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == 'include_previews':
            expected = ast.parse('''if include_previews:
    with prepared_preview_sandbox(state, side):
        legal_entries = [legal_entry(action, preview(state, side, action), interaction_for_action(state, side, action)) for action in actions]
else:
    legal_entries = [legal_entry(action, {}, {}) for action in actions]
''').body[0]
            if not _same(node, expected):
                raise ProjectionSpecializationError('Canonical legal-entry preview branches changed')
            self.value_sites['entries'] += 1
            return ast.copy_location(ast.parse('legal_entries = []').body[0], node)
        return super().visit_If(node)


def derive_value_observation(original=projection.observe):
    """Fail closed if the canonical or inherited policy-query shape changes."""
    transform = _ValueObserve()
    result, _ = _compile(original, '_value_observe', transform,
                        dict(_queries, public_card=_policy_card, _policy_state=_policy_state))
    expected = dict(_REQUIRED, query_cache=1)
    expected.update({str(field): 1 for field in (
        ('team', 'effect_markers'), ('c', 'stat_layers'),
        ('c', 'equipment'), ('c', 'effect_markers'))})
    if transform.counts != expected or transform.value_sites != dict(actions=1, entries=1, entry_function=1):
        raise ProjectionSpecializationError('Canonical value-query specialization sites changed')
    return result


_value_observe = derive_value_observation()


def cached_policy_value_observation(game, side):
    """Return the same public value fields, with no action/candidate contract."""
    return _value_observe(game, side, include_previews=False)
