"""Canonical array-rule specializations vs formal reduce_max_hp snapshots."""
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import ast
import copy
import dis
import inspect
import json
import random
import textwrap
import unittest

from app.modules.card_game.engine.duel_v2 import new_game, acting_side, legal_actions
from app.modules.card_game.engine.duel_v2.combat import attack as combat_attack
from app.modules.card_game.engine.duel_v2.state import hero, reduce_max_hp, round_half_up
from app.modules.card_game.rl.batched_duel.gpu_owned_transitions import (
    scratch_apply_action, scratch_simulate_action,
)
from app.modules.card_game.rl.batched_duel.array_rule_sources import (
    ArrayRuleSpecializationError,
    ReduceMaxHpRatioTransformer,
    array_reduce_max_hp,
    array_rule_bindings_for,
    array_scratch_apply_action,
    array_scratch_simulate_action,
    canonical_array_rule_identities,
    canonical_ratio_operands,
    rebind_function_globals,
    specialize_canonical_source,
    substitute_ast_names,
)


def aliases(value):
    paths = {}
    stack = [((), value)]
    while stack:
        path, item = stack.pop()
        if not isinstance(item, (list, tuple, dict)):
            continue
        old = paths.setdefault(id(item), [])
        old.append(path)
        if len(old) > 1:
            continue
        pairs = item.items() if isinstance(item, dict) else enumerate(item)
        stack.extend((path + (str(k),), v) for k, v in pairs)
    return sorted(sorted(group) for group in paths.values() if len(group) > 1)


def entity_kinds(state):
    kinds = []
    for side in ('a', 'b'):
        team = state['sides'][side]
        kinds.append((side, type(team).__name__, team.get('entity_id')))
        for cid, unit in team['characters'].items():
            kinds.append((side, cid, type(unit).__name__, unit.get('entity_id')))
    return kinds


def unit_of(state, side, cid):
    return state['sides'][side] if cid == 'player' else hero(state, side, cid)


def run_both(state, side, cid, amount, **kwargs):
    expected_state = deepcopy(state)
    actual_state = deepcopy(state)
    expected = reduce_max_hp(expected_state, side, cid, amount, **kwargs)
    actual = array_reduce_max_hp(actual_state, side, cid, amount, **kwargs)
    return expected_state, actual_state, expected, actual


def assert_same_rule(test, expected_state, actual_state, expected, actual):
    test.assertEqual(actual, expected)
    test.assertEqual(actual_state, expected_state)
    test.assertEqual(aliases(actual_state), aliases(expected_state))
    test.assertEqual(entity_kinds(actual_state), entity_kinds(expected_state))
    test.assertEqual(actual_state.get('events'), expected_state.get('events'))
    test.assertEqual(actual_state.get('phase'), expected_state.get('phase'))
    test.assertEqual(actual_state.get('winner'), expected_state.get('winner'))


def decimal_cut(hp, max_hp, amount):
    amount = max(0, int(amount))
    if hp <= 0:
        return hp, max_hp, 0
    cut = 0
    for _ in range(amount):
        old = max_hp
        if old <= 0:
            break
        new = old - 1
        hp = min(int((Decimal(hp) / Decimal(old) * Decimal(new)).quantize(
            Decimal('1'), rounding=ROUND_HALF_UP)), new)
        max_hp = new
        cut += 1
        if hp <= 0 or max_hp <= 0:
            hp = 0
            break
    return hp, max_hp, cut


def canonical_tree():
    return ast.parse(textwrap.dedent(inspect.getsource(reduce_max_hp)))


class StripDecimal(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == 'Decimal'
                and len(node.args) == 1 and not node.keywords):
            return node.args[0]
        return node


class ArrayRuleSourcesTest(unittest.TestCase):
    def setUp(self):
        self.state = new_game(seed=9, first_side='a', skip_mulligan=True)
        self.characters = ['player'] + list(self.state['sides']['a']['characters'])

    def test_canonical_players_and_characters_snapshots(self):
        cases = [
            ('a', 'nanali', 5, 7, 1), ('a', 'nanali', 5, 7, 2), ('a', 'nanali', 5, 5, 2),
            ('a', 'iloy', 3, 6, 1), ('a', 'zero', 1, 5, 4), ('a', 'jiuyuan', 4, 4, 4),
            ('b', 'player', 5, 7, 1), ('b', 'player', 5, 7, 2), ('b', 'player', 20, 30, 1),
            ('a', 'player', 30, 30, 0), ('a', 'nanali', 8, 8, 1),
        ]
        original = deepcopy(self.state)
        for side, cid, hp, mx, amt in cases:
            state = deepcopy(self.state)
            unit_of(state, side, cid).update(hp=hp, max_hp=mx, shield=3)
            expected_state, actual_state, expected, actual = run_both(
                state, side, cid, amt, settle=False, source=f'{side}:{cid}', group_id='g1')
            assert_same_rule(self, expected_state, actual_state, expected, actual)
            got = unit_of(actual_state, side, cid)
            self.assertEqual((got['hp'], got['max_hp'], actual), decimal_cut(hp, mx, amt))
            self.assertEqual(got['shield'], 3)
            self.assertIs(type(got), type(unit_of(state, side, cid)))
        self.assertEqual(self.state, original)

    def test_life_locked_zero_negative_and_finished(self):
        original = deepcopy(self.state)
        locked = deepcopy(self.state)
        locked['sides']['a'].update(hp=10, max_hp=30, life_locked=True)
        expected_state, actual_state, expected, actual = run_both(
            locked, 'a', 'player', 4, settle=False)
        assert_same_rule(self, expected_state, actual_state, expected, actual)
        self.assertEqual(actual, 0)
        self.assertEqual(actual_state['sides']['a']['hp'], 10)
        self.assertEqual(actual_state['sides']['a']['max_hp'], 30)

        zero = deepcopy(self.state)
        hero(zero, 'a', 'nanali').update(hp=0, max_hp=5)
        before_events = len(zero['events'])
        expected_state, actual_state, expected, actual = run_both(
            zero, 'a', 'nanali', 3, settle=False)
        assert_same_rule(self, expected_state, actual_state, expected, actual)
        self.assertEqual(actual, 0)
        self.assertEqual(len(actual_state['events']), before_events)

        negative_hp = deepcopy(self.state)
        hero(negative_hp, 'b', 'zero').update(hp=-2, max_hp=6)
        expected_state, actual_state, expected, actual = run_both(
            negative_hp, 'b', 'zero', 2, settle=False)
        assert_same_rule(self, expected_state, actual_state, expected, actual)
        self.assertEqual(actual, 0)

        negative_amount = deepcopy(self.state)
        unit_of(negative_amount, 'a', 'player').update(hp=8, max_hp=30)
        expected_state, actual_state, expected, actual = run_both(
            negative_amount, 'a', 'player', -4, settle=False)
        assert_same_rule(self, expected_state, actual_state, expected, actual)
        self.assertEqual(actual, 0)
        self.assertEqual(actual_state['sides']['a']['hp'], 8)
        self.assertEqual(actual_state['events'][-1]['type'], 'max_hp')

        finished = deepcopy(self.state)
        finished.update(phase='finished', winner='b')
        expected_state, actual_state, expected, actual = run_both(
            finished, 'a', 'player', 5, settle=True)
        assert_same_rule(self, expected_state, actual_state, expected, actual)
        self.assertEqual(actual, 0)
        self.assertEqual(self.state, original)

    def test_repeated_rounding_and_settlement(self):
        state = deepcopy(self.state)
        hero(state, 'b', 'zero').update(hp=5, max_hp=7, shield=5)
        first_e, first_a, cut_e, cut_a = run_both(state, 'b', 'zero', 1, settle=False)
        assert_same_rule(self, first_e, first_a, cut_e, cut_a)
        self.assertEqual((hero(first_a, 'b', 'zero')['hp'], hero(first_a, 'b', 'zero')['max_hp'],
                          hero(first_a, 'b', 'zero')['shield']), (4, 6, 5))
        second_e, second_a, cut_e, cut_a = run_both(first_a, 'b', 'zero', 1, settle=False)
        assert_same_rule(self, second_e, second_a, cut_e, cut_a)
        self.assertEqual((hero(second_a, 'b', 'zero')['hp'], hero(second_a, 'b', 'zero')['max_hp']), (3, 5))

        lethal = deepcopy(self.state)
        hero(lethal, 'a', 'nanali').update(hp=1, max_hp=1)
        expected_state, actual_state, expected, actual = run_both(
            lethal, 'a', 'nanali', 1, settle=True, source='b:zero')
        assert_same_rule(self, expected_state, actual_state, expected, actual)
        self.assertEqual(actual, 1)
        self.assertEqual(hero(actual_state, 'a', 'nanali')['down_turns'], 3)
        self.assertIn('down', [event['type'] for event in actual_state['events']])
        self.assertEqual(actual_state['phase'], 'playing')

        skip = deepcopy(lethal)
        expected_state, actual_state, expected, actual = run_both(
            skip, 'a', 'nanali', 1, settle=False)
        assert_same_rule(self, expected_state, actual_state, expected, actual)
        self.assertEqual(hero(actual_state, 'a', 'nanali')['down_turns'], 0)
        self.assertNotIn('down', [event['type'] for event in actual_state['events']])

        player = deepcopy(self.state)
        player['sides']['a'].update(hp=1, max_hp=1)
        expected_state, actual_state, expected, actual = run_both(
            player, 'a', 'player', 1, settle=True)
        assert_same_rule(self, expected_state, actual_state, expected, actual)
        self.assertEqual(actual_state['phase'], 'finished')
        self.assertEqual(actual_state['winner'], 'b')
        self.assertEqual(actual_state['sides']['a']['hp'], 0)

    def test_random_bounded_states(self):
        rng = random.Random(20261016)
        original = deepcopy(self.state)
        for index in range(160):
            state = deepcopy(self.state)
            side = rng.choice(('a', 'b'))
            cid = rng.choice(self.characters if side == 'a'
                             else ['player'] + list(state['sides']['b']['characters']))
            mx = rng.randint(0, 48)
            hp = rng.randint(-2, mx if mx else 4)
            amount = rng.randint(-3, 12)
            settle = bool(rng.getrandbits(1))
            unit = unit_of(state, side, cid)
            unit.update(hp=hp, max_hp=mx, shield=rng.randint(0, 4))
            if cid == 'player' and rng.random() < 0.12:
                unit['life_locked'] = True
            expected_state, actual_state, expected, actual = run_both(
                state, side, cid, amount, settle=settle,
                source=f'{side}:{cid}', group_id=('g' if rng.random() < 0.3 else None))
            assert_same_rule(self, expected_state, actual_state, expected, actual)
            if hp > 0 and mx > 0 and amount > 0 and not unit.get('life_locked'):
                want_hp, want_max, want_cut = decimal_cut(hp, mx, amount)
                got = unit_of(actual_state, side, cid)
                if (not settle) or (got.get('down_turns', 0) == 0 and actual_state['phase'] != 'finished'):
                    self.assertEqual((got['hp'], got['max_hp'], actual), (want_hp, want_max, want_cut))
            self.assertEqual(self.state, original)
        self.assertGreater(index, 100)

    def test_json_entities_alias_immutability(self):
        state = deepcopy(self.state)
        unit = hero(state, 'a', 'nanali')
        unit.update(hp=5, max_hp=7)
        before = deepcopy(state)
        live = deepcopy(state)
        live_unit = hero(live, 'a', 'nanali')
        array_reduce_max_hp(live, 'a', 'nanali', 2, settle=False)
        self.assertIs(hero(live, 'a', 'nanali'), live_unit)
        self.assertEqual(type(live_unit).__name__, 'CharacterEntity')
        expected_state, actual_state, expected, actual = run_both(
            state, 'a', 'nanali', 2, settle=False)
        assert_same_rule(self, expected_state, actual_state, expected, actual)
        self.assertEqual(state, before)
        self.assertEqual(type(hero(actual_state, 'a', 'nanali')).__name__, 'CharacterEntity')
        self.assertEqual(type(actual_state['sides']['a']).__name__, 'PlayerEntity')
        self.assertEqual(hero(actual_state, 'a', 'nanali')['entity_id'], unit['entity_id'])

        plain = json.loads(json.dumps(state))
        self.assertEqual(type(plain['sides']['a']['characters']['nanali']), dict)
        expected_state, actual_state, expected, actual = run_both(
            plain, 'a', 'nanali', 2, settle=False)
        assert_same_rule(self, expected_state, actual_state, expected, actual)
        self.assertEqual(type(unit_of(actual_state, 'a', 'nanali')), dict)
        self.assertEqual(plain, json.loads(json.dumps(state)))

    def test_canonical_source_integrity(self):
        source = inspect.getsource(reduce_max_hp)
        self.assertIn(
            "round_half_up(Decimal(h['hp']) / Decimal(old_max) * Decimal(new_max))",
            source)
        self.assertNotIn('Decimal', array_reduce_max_hp.__code__.co_names)
        self.assertNotIn('round_half_up', array_reduce_max_hp.__code__.co_names)
        ops = {item.opname for item in dis.get_instructions(array_reduce_max_hp)}
        self.assertIn('BINARY_FLOOR_DIVIDE', ops)
        self.assertNotIn('BINARY_TRUE_DIVIDE', ops)
        self.assertIsNot(array_reduce_max_hp, reduce_max_hp)
        self.assertIs(reduce_max_hp.__globals__['reduce_max_hp'], reduce_max_hp)
        self.assertIsNot(array_reduce_max_hp.__globals__, reduce_max_hp.__globals__)
        self.assertEqual(array_reduce_max_hp.__module__,
                         'app.modules.card_game.rl.batched_duel.array_rule_sources')
        self.assertIn('knockdowns', array_reduce_max_hp.__code__.co_names)
        self.assertIn('victory', array_reduce_max_hp.__code__.co_names)
        self.assertIn('log', array_reduce_max_hp.__code__.co_names)
        identities = canonical_array_rule_identities()
        self.assertIs(identities[reduce_max_hp], array_reduce_max_hp)
        self.assertIs(combat_attack.__globals__['reduce_max_hp'], reduce_max_hp)
        tree = canonical_tree()
        operands = None
        for node in ast.walk(tree):
            found = canonical_ratio_operands(node)
            if found is not None:
                self.assertIsNone(operands)
                operands = found
        self.assertIsNotNone(operands)
        specialized = specialize_canonical_source(
            reduce_max_hp, 'array_reduce_max_hp_again', ReduceMaxHpRatioTransformer())
        state = deepcopy(self.state)
        hero(state, 'a', 'nanali').update(hp=5, max_hp=7)
        expected_state, actual_state, expected, actual = run_both(
            state, 'a', 'nanali', 2, settle=False)
        again = deepcopy(state)
        self.assertEqual(specialized(again, 'a', 'nanali', 2, settle=False), expected)
        self.assertEqual(again, expected_state)
        from app.modules.card_game.engine.duel_v2 import state as state_mod
        from app.modules.card_game.engine.duel_v2 import combat as combat_mod
        self.assertIs(state_mod.reduce_max_hp, reduce_max_hp)
        self.assertIs(combat_mod.reduce_max_hp, reduce_max_hp)

    def test_fail_closed_non_integer_domain_and_structure(self):
        for hp, maximum in ((1.5, 6), (3, 6.0), (True, 6), (3, 2147483648)):
            state = deepcopy(self.state)
            target = hero(state, 'a', 'nanali')
            target.update(hp=hp, max_hp=maximum)
            before = deepcopy(state)
            with self.assertRaisesRegex(ValueError, 'integer'):
                array_reduce_max_hp(state, 'a', 'nanali', 1, settle=False)
            self.assertEqual(state, before)
        tree = canonical_tree()
        with self.assertRaises(ArrayRuleSpecializationError):
            ReduceMaxHpRatioTransformer().visit(StripDecimal().visit(copy.deepcopy(tree)))

        mutated = copy.deepcopy(tree)
        for node in ast.walk(mutated):
            operands = canonical_ratio_operands(node)
            if operands is None:
                continue
            node.args[0].left.right.func = ast.Name('float', ast.Load())
            break
        with self.assertRaises(ArrayRuleSpecializationError):
            ReduceMaxHpRatioTransformer().visit(mutated)

        renamed = copy.deepcopy(tree)
        renamed.body[0].args.args[0].arg = 'board'
        with self.assertRaises(ArrayRuleSpecializationError):
            ReduceMaxHpRatioTransformer().visit(renamed)

        self.assertEqual(
            ast.dump(ast.parse(textwrap.dedent(inspect.getsource(round_half_up)))),
            ast.dump(ast.parse(
                "def round_half_up(value):\n"
                "    return int(Decimal(value).quantize(Decimal('1'), rounding=ROUND_HALF_UP))\n"
            )),
        )

    def test_substitution_helpers_preserve_production_bindings(self):
        bindings = array_rule_bindings_for(combat_attack)
        self.assertEqual(set(bindings), {'reduce_max_hp'})
        self.assertIs(bindings['reduce_max_hp'], array_reduce_max_hp)
        bound = rebind_function_globals(combat_attack, bindings, name='array_attack')
        self.assertIs(bound.__globals__['reduce_max_hp'], array_reduce_max_hp)
        self.assertIs(combat_attack.__globals__['reduce_max_hp'], reduce_max_hp)
        self.assertIs(bound.__code__, combat_attack.__code__)
        self.assertIsNot(bound.__globals__, combat_attack.__globals__)
        self.assertEqual(bound.__module__,
                         'app.modules.card_game.rl.batched_duel.array_rule_sources')
        with self.assertRaises(ArrayRuleSpecializationError):
            rebind_function_globals(combat_attack, {'missing_name': array_reduce_max_hp})
        with self.assertRaises(ArrayRuleSpecializationError):
            rebind_function_globals(reduce_max_hp, {1: array_reduce_max_hp})
        self.assertEqual(array_rule_bindings_for(array_scratch_apply_action), {})

        tree = ast.parse('def entry(state, side, cid, n):\n    return reduce_max_hp(state, side, cid, n)\n')
        rewritten = substitute_ast_names(tree, {'reduce_max_hp': 'array_reduce_max_hp'})
        dumped = ast.dump(rewritten)
        self.assertIn("'array_reduce_max_hp'", dumped)
        self.assertNotIn("'reduce_max_hp'", dumped)
        with self.assertRaises(ArrayRuleSpecializationError):
            substitute_ast_names(tree, {object(): 'array_reduce_max_hp'})

    def test_owned_scratch_wrappers_match_canonical_owned(self):
        state = deepcopy(self.state)
        actor = acting_side(state)
        action = next(item for item in legal_actions(state, actor) if item['type'] != 'concede')
        original = deepcopy(state)
        for formal, derived in (
            (scratch_apply_action, array_scratch_apply_action),
            (scratch_simulate_action, array_scratch_simulate_action),
        ):
            expected = formal(deepcopy(state), actor, deepcopy(action))
            actual = derived(deepcopy(state), actor, deepcopy(action))
            self.assertEqual(actual, expected)
            self.assertEqual(aliases(actual), aliases(expected))
            self.assertEqual(entity_kinds(actual), entity_kinds(expected))
        self.assertEqual(state, original)
        self.assertIsNot(array_scratch_apply_action, scratch_apply_action)
        self.assertIsNot(array_scratch_simulate_action, scratch_simulate_action)


if __name__ == '__main__':
    unittest.main()
