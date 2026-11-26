"""Differential checks of resident domain stages against formal primitives."""
import random
import unittest

import torch

from app.modules.card_game.rl.batched_duel.domain_soa_rules import (
    INVALID_DOMAIN, INTEGER_OVERFLOW, UNKNOWN_OPERATION, capped_add,
    shield_add, damage_fields, resource_caps, reduce_integer_modifiers,
    expire_effects, compact_slots,
)


def ints(values):
    return torch.tensor(values, dtype=torch.int32)


def bits(values):
    return torch.tensor(values, dtype=torch.bool)


class DomainSoARulesTest(unittest.TestCase):
    def test_damage_matches_formal_unclamped_loss_and_inactive_tail(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.engine.duel_v2.state import damage
        from app.modules.card_game.engine.duel_v2.modifiers import add_effect
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        deck = next(d for d in STARTER_DECKS if d['id'] == 'weave-rush')
        cases = [(5, 2, 8, False, False, 0, -1),
                 (5, 2, 8, True, False, 0, -1),
                 (1, 3, 10, False, False, 1, -1),
                 (4, 2, 8, False, True, 0, -1),
                 (5, 2, -1, False, False, 0, -1),
                 (5, 2, 8, False, False, 0, 1)]
        expected = []
        for hp, shield, amount, bypass, immune, floor, limit in cases:
            state = new_game(seed=17, decks={'a': deck, 'b': deck}, skip_mulligan=True)
            h = state['sides']['a']['characters']['baicang']
            h.update(hp=hp, shield=shield)
            if immune:
                add_effect(h, 'test.immunity', 'rule.damage_immune', True, op='set')
            if floor:
                h['flags']['hp_floor'] = floor
            if limit >= 0:
                add_effect(h, 'test.cap', 'damage.limit', limit, op='set')
            loss = damage(state, 'a', 'baicang', amount, bypass=bypass, settle=False)
            expected.append((h['hp'], h['shield'], loss))
        result = damage_fields(ints([c[0] for c in cases]), ints([c[1] for c in cases]),
            ints([c[2] for c in cases]), bits([c[3] for c in cases]),
            bits([c[4] for c in cases]), bits([False] * len(cases)),
            ints([c[5] for c in cases]), ints([c[6] for c in cases]), bits([True] * len(cases)))
        self.assertEqual(list(zip(result.hp.tolist(), result.shield.tolist(), result.loss.tolist())), expected)
        self.assertEqual(result.errors.tolist(), [0] * len(cases))
        inactive = damage_fields(ints([-9]), ints([-2]), ints([5]), bits([False]),
            bits([False]), bits([False]), ints([0]), ints([-1]), bits([False]))
        self.assertEqual(inactive.hp.tolist(), [-9])
        self.assertEqual(inactive.errors.tolist(), [0])

    def test_life_lock_consumes_shield_without_life_loss(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.engine.duel_v2.state import damage
        state = new_game(skip_mulligan=True)
        state['sides']['a'].update(hp=7, shield=2, life_locked=True)
        loss = damage(state, 'a', 'player', 9, settle=False)
        result = damage_fields(ints([7]), ints([2]), ints([9]), bits([False]), bits([False]),
            bits([True]), ints([0]), ints([-1]), bits([True]))
        self.assertEqual(result.hp.item(), state['sides']['a']['hp'])
        self.assertEqual(result.shield.item(), state['sides']['a']['shield'])
        self.assertEqual(result.loss.item(), loss)

    def test_resources_formal_clamp_and_signed_delta(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.engine.duel_v2.state import energy
        amounts = [-2, 0, 1, 9]
        expected = []
        for amount in amounts:
            state = new_game(skip_mulligan=True)
            state['sides']['a']['characters']['zero']['energy'] = 2
            energy(state, 'a', 'zero', amount)
            expected.append(state['sides']['a']['characters']['zero']['energy'])
        result = capped_add(ints([2]*4), ints(amounts), ints([5]*4), bits([True]*4), bits([True]*4))
        self.assertEqual(result.value.tolist(), expected)
        overflow = capped_add(ints([-2147483648]), ints([-1]), ints([5]), bits([True]), bits([True]))
        self.assertEqual(overflow.errors.tolist(), [INTEGER_OVERFLOW])
        self.assertEqual(overflow.value.tolist(), [-2147483648])

    def test_shield_accumulates_and_overflow_fails_before_store(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.engine.duel_v2.state import shield
        state = new_game(skip_mulligan=True)
        h = state['sides']['a']['characters']['zero']
        h['shield'] = 2
        shield(state, 'a', 'zero', 3)
        result = shield_add(ints([2, 2147483647, -1]), ints([3, 1, 5]),
            bits([True]*3), bits([True]*3))
        self.assertEqual(result.value.tolist(), [h['shield'], 2147483647, -1])
        self.assertEqual(result.errors.tolist(), [0, INTEGER_OVERFLOW, INVALID_DOMAIN])

    def test_integer_modifiers_match_canonical_stable_layers(self):
        from app.modules.card_game.engine.duel_v2.modifiers import reduce_modifiers
        rng = random.Random(1016)
        names = ['set', 'add', 'multiply', 'allow', 'forbid']
        rows, width = 37, 9
        base = [rng.randint(-3, 8) for _ in range(rows)]
        layers = [[rng.randint(-2, 2) for _ in range(width)] for _ in range(rows)]
        ops = [[rng.randrange(5) for _ in range(width)] for _ in range(rows)]
        amounts = [[rng.randint(-2, 3) for _ in range(width)] for _ in range(rows)]
        present = [[rng.random() < .8 for _ in range(width)] for _ in range(rows)]
        expected = []
        types = []
        for r in range(rows):
            effects = [dict(layer=layers[r][c], op=names[ops[r][c]], value=amounts[r][c])
                       for c in range(width) if present[r][c]]
            value = reduce_modifiers(base[r], effects)['value']
            expected.append(int(value))
            types.append(type(value) is bool)
        result = reduce_integer_modifiers(ints(base), ints(layers), ints(ops), ints(amounts),
            bits(present), bits([True]*rows))
        self.assertEqual(result.value.tolist(), expected)
        self.assertEqual(result.boolean.tolist(), types)
        self.assertEqual(result.errors.tolist(), [0]*rows)
        bad = reduce_integer_modifiers(ints([1, 2]), ints([[0], [0]]), ints([[99], [2]]),
            ints([[1], [2147483647]]), bits([[True], [True]]), bits([True, True]))
        self.assertEqual(bad.errors.tolist(), [UNKNOWN_OPERATION, INTEGER_OVERFLOW])
        self.assertEqual(bad.value.tolist(), [1, 2])
        self.assertFalse(bad.applied.any())
        boolean_set = reduce_integer_modifiers(ints([7]), ints([[0, 0]]), ints([[0, 4]]),
            ints([[1, 0]]), bits([[True, True]]), bits([True]),
            amount_boolean=bits([[True, False]]))
        self.assertEqual(boolean_set.value.tolist(), [1])
        self.assertEqual(boolean_set.boolean.tolist(), [True])

    def test_effect_expiry_formal_clocks_and_stable_compaction(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.engine.duel_v2.modifiers import add_effect, tick
        state = new_game(skip_mulligan=True)
        h = state['sides']['a']['characters']['zero']
        h['effects'] = []
        records = [('a', 'turn_start', 2), ('b', 'turn_start', 1), ('a', 'turn_end', 1), ('a', 'turn_start', 5)]
        for index, (side, phase, turn) in enumerate(records):
            add_effect(h, 'test.'+str(index), 'attack.panel', index,
                       expiry=dict(side=side, phase=phase, turn=turn))
        state['sides']['a']['turn_count'] = 3
        original_ids = [e['effect_id'] for e in h['effects']]
        expired = tick(state, 'a', 'turn_start')
        expected = [original_ids.index(e['effect_id']) for _, _, e in expired]
        kept, mask, errors = expire_effects(ints([[0, 1, 0, 0]]), ints([[0, 0, 1, 0]]),
            ints([[2, 1, 1, 5]]), bits([[True]*4]), ints([[0]*4]), ints([0]), ints([0]),
            ints([[3, 1]]), bits([True]))
        self.assertEqual(mask.nonzero()[:, 1].tolist(), expected)
        values, compact_mask, permutation = compact_slots(ints([[10, 20, 30, 40]]), kept, bits([True]))
        self.assertEqual(values.tolist(), [[20, 30, 40, 10]])
        self.assertEqual(compact_mask.tolist(), [[True, True, True, False]])
        self.assertEqual(errors.tolist(), [0])

    def test_caps_match_formal_thresholds_for_all_public_characters(self):
        from app.modules.card_game.content.duel_v2 import CHARACTERS
        from app.modules.card_game.engine.duel_v2.escalation import energy_cap, harmony_cap
        from app.modules.card_game.engine.duel_v2.state import new_character, energy_max
        turns = [1, 6, 12, 13, 20, 13]
        enabled = [True]*5 + [False]
        characters = [new_character(k) for k in CHARACTERS]
        base = [[energy_max(h) for h in characters] for _ in turns]
        energy_out, harmony_out = resource_caps(ints(turns), bits(enabled), ints(base))
        for row, (turn, on) in enumerate(zip(turns, enabled)):
            state = dict(turn=turn, escalation_enabled=on)
            self.assertEqual(energy_out[row].tolist(), [energy_cap(state, h) for h in characters])
            self.assertEqual(harmony_out[row].item(), harmony_cap(state))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; CPU parity does not qualify device execution')
    def test_cuda_graph_replay_stages(self):
        tensors = [ints([5, 1]).cuda(), ints([2, 0]).cuda(), ints([8, 2]).cuda()]
        flags = [bits([False, False]).cuda() for _ in range(3)]
        floor, limit, active = ints([0, 1]).cuda(), ints([-1, -1]).cuda(), bits([True, True]).cuda()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(2):
                damage_fields(*tensors, *flags, floor, limit, active)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = damage_fields(*tensors, *flags, floor, limit, active)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(output.hp.cpu().tolist(), [0, 1])
        tensors[2].zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(output.hp.cpu().tolist(), [5, 1])


if __name__ == '__main__':
    unittest.main()
