"""CPU oracle for direct domain SoA public projection. No full-3479 claim."""
import ast
import inspect
import unittest
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import torch

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2.flow import legal_actions, new_game
from app.modules.card_game.engine.duel_v2.projection import observe
from app.modules.card_game.rl import seven_lineup
from app.modules.card_game.rl.batched_duel import domain_soa_projection as proj
from app.modules.card_game.rl.batched_duel.domain_soa_codec import encode_domain_states
from app.modules.card_game.rl.batched_duel.domain_soa_schema import (
    CARD_VOCAB,
    DomainSoALayout,
    PRESENCE_VALUE,
)


KEYS = ('starter', 'weave-rush')
MODULE_PATH = Path(proj.__file__)


def _decks():
    return {item['id']: item for item in STARTER_DECKS}


def _state(seed=7, first='a', a='starter', b='weave-rush'):
    decks = _decks()
    return new_game(seed=seed, decks={'a': decks[a], 'b': decks[b]}, first_side=first)


def _pack(*states):
    return encode_domain_states(list(states), presentation='drop').to_torch(device='cpu')


def _project(packed, viewers, active=None, **kwargs):
    rows = packed.arrays['row_valid'].shape[0]
    if active is None:
        active = torch.ones(rows, dtype=torch.bool)
    return cpu().project(packed, torch.as_tensor(viewers, dtype=torch.int64), active, **kwargs)


def cpu(**kwargs):
    return proj.cpu_oracle(**kwargs)


class DomainSoAProjectionTest(unittest.TestCase):
    def test_coverage_refuses_full_rules_and_full_3479(self):
        report = proj.coverage()
        self.assertFalse(proj.supports_full_duel_rules)
        self.assertFalse(proj.DomainSoAProjection.supports_full_duel_rules)
        self.assertFalse(report['supports_full_duel_rules'])
        self.assertFalse(report['full_3479_observation'])
        self.assertEqual(report['seven_observation_dim'], 3479)
        self.assertLess(report['observation_dim_claimed'], 3479)
        self.assertEqual(report['full_action_capacity'], 800)
        self.assertEqual(report['search_candidate_cap'], 16)
        self.assertNotEqual(proj.FULL_ACTION_CAPACITY, proj.SEARCH_CANDIDATE_CAP)
        self.assertEqual(len(seven_lineup.all_features()), 3479)
        self.assertTrue(set(proj.IMPLEMENTED_FEATURE_NAMES).issubset(set(seven_lineup.all_features())))

    def test_source_stays_on_named_tensors_without_vm_or_hot_host_sync(self):
        tree = ast.parse(MODULE_PATH.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append((node.module or '') + '.' + ','.join(alias.name for alias in node.names))
        joined = ' '.join(imports)
        self.assertNotIn('domain_soa_codec', joined)
        self.assertNotIn('native_soa', joined)
        self.assertNotIn('cuda_vm', joined)
        self.assertNotIn('vm_heap', joined)
        self.assertNotIn('gpu_rule_projection', joined)
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ('item', 'cpu', 'numpy', 'tolist'):
                forbidden.append(node.attr)
        self.assertEqual(forbidden, [])
        source = inspect.getsource(proj.DomainSoAProjection.project)
        self.assertNotIn('for row in', source)
        self.assertNotIn('legal_actions(', source)

    def test_named_columns_match_seven_lineup_oracle_on_mulligan(self):
        states = [_state(11, 'a'), _state(12, 'b', 'weave-rush', 'starter')]
        packed = _pack(*states)
        arena = cpu()
        viewers = torch.tensor([0, 1], dtype=torch.int64)
        out = arena.project(packed, viewers, torch.ones(2, dtype=torch.bool))
        self.assertTrue(torch.equal(out.errors, torch.zeros(2, dtype=torch.int32)))
        self.assertTrue(out.public_flags[:, 1].all())
        for row, (state, viewer) in enumerate(zip(states, ('a', 'b'))):
            view = observe(state, viewer)
            obs, _cand = seven_lineup.encode(view, [])
            got = out.packed_named[row].detach().float()
            oracle = torch.as_tensor(obs[list(out.packed_index)], dtype=torch.float32)
            self.assertEqual(got.shape, oracle.shape)
            self.assertTrue(torch.equal(got, oracle), msg=self._diff(out.packed_names, got, oracle))
            self.assertEqual(int(out.named['player_hp'][row, 0]), view['sides'][viewer]['hp'])
            self.assertEqual(int(out.named['player_front_seat'][row, 0]), -1)

    def test_all_seven_presets_both_first_sides_and_viewers_match(self):
        from tests.test_duel_v2_domain_soa_codec import _fourteen_initial_states
        states = _fourteen_initial_states()
        packed = _pack(*states)
        arena = cpu()
        for viewer in (0, 1):
            out = arena.project(packed, torch.full((len(states),), viewer, dtype=torch.int64),
                                torch.ones(len(states), dtype=torch.bool))
            self.assertTrue((out.errors == 0).all())
            for row, state in enumerate(states):
                obs, _ = seven_lineup.encode(observe(state, 'ab'[viewer]), [])
                expected = torch.tensor(obs[list(out.packed_index)], dtype=torch.float32)
                self.assertTrue(torch.equal(out.packed_named[row], expected),
                                msg=self._diff(out.packed_names, out.packed_named[row], expected))

    def test_hidden_opponent_cards_rng_and_deck_order_do_not_enter(self):
        state = _state(21)
        packed = _pack(state)
        viewers = torch.tensor([0], dtype=torch.int64)
        active = torch.ones(1, dtype=torch.bool)
        baseline = cpu().project(packed, viewers, active)
        mutated = {name: value.clone() for name, value in packed.arrays.items()}
        mutated['rng'][0] = mutated['rng'][0] + 123456789
        opp = 1
        deck_n = int(mutated['zone_count'][0, opp, 1])
        self.assertGreater(deck_n, 1)
        i0 = int(mutated['zone_card_index'][0, opp, 1, 0])
        i1 = int(mutated['zone_card_index'][0, opp, 1, 1])
        mutated['zone_card_index'][0, opp, 1, 0] = i1
        mutated['zone_card_index'][0, opp, 1, 1] = i0
        hand_n = int(mutated['zone_count'][0, opp, 0])
        self.assertGreater(hand_n, 0)
        hid = int(mutated['zone_card_index'][0, opp, 0, 0])
        other = next(v for v in CARD_VOCAB.values() if v != int(mutated['card_template'][0, hid]))
        mutated['card_template'][0, hid] = other
        leaked = cpu().project(mutated, viewers, active)
        self.assertTrue(torch.equal(baseline.packed_named, leaked.packed_named))
        self.assertTrue(torch.equal(baseline.tokens, leaked.tokens))
        own = 0
        own_i = int(mutated['zone_card_index'][0, own, 0, 0])
        mutated['card_template'][0, own_i] = other
        changed = cpu().project(mutated, viewers, active)
        self.assertFalse(torch.equal(baseline.packed_named, changed.packed_named))

    def test_mulligan_tokens_preserve_engine_order_including_tail(self):
        state = _state(31, 'a')
        packed = _pack(state)
        out = _project(packed, [0])
        self.assertEqual(int(out.errors[0]), 0)
        engine = legal_actions(state, 'a')
        count = int(out.counts[0])
        self.assertEqual(count, len(engine))
        self.assertTrue(bool(out.mask[0, :count].all()))
        self.assertFalse(bool(out.mask[0, count:].any()))
        self.assertEqual(int(out.tokens[0, 0, 0]), proj.ACTION_CONCEDE)
        self.assertEqual(engine[0]['type'], 'concede')
        hand = [card['instance_id'] for card in state['sides']['a']['hand']]
        decoded = []
        for slot in range(1, count):
            self.assertEqual(int(out.tokens[0, slot, 0]), proj.ACTION_MULLIGAN)
            ordinals = [int(v) for v in out.mulligan_ordinals[0, slot].tolist() if int(v) >= 0]
            decoded.append(tuple(hand[i] for i in ordinals))
        expected = [tuple(group) for size in range(min(3, len(hand)) + 1)
                    for group in combinations(hand, size)]
        self.assertEqual(decoded, expected)
        self.assertEqual(decoded[-1], expected[-1])
        self.assertEqual(int(out.actors[0]), 0)

    def test_inactive_unknown_overflow_and_unhandled_fail_closed(self):
        packed = _pack(_state(41), _state(42), _state(43), _state(44))
        fields = {name: value.clone() for name, value in packed.arrays.items()}
        fields['phase'][1] = 1
        fields['character_template'][2, 0, 0] = 0
        fields['zone_count'][3, 0, 0] = 17
        active = torch.tensor([False, True, True, True])
        viewers = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        out = cpu().project(fields, viewers, active)
        self.assertEqual(int(out.errors[0]), 0)
        self.assertEqual(int(out.counts[0]), 0)
        self.assertTrue(torch.equal(out.packed_named[0], torch.zeros_like(out.packed_named[0])))
        self.assertEqual(int(out.actors[0]), -1)
        self.assertEqual(int(out.errors[1]), proj.UNHANDLED_ACTION_CATEGORY)
        self.assertEqual(int(out.counts[1]), 0)
        self.assertFalse(bool(out.public_flags[1, 1]))
        self.assertEqual(int(out.errors[2]), proj.UNKNOWN_TEMPLATE)
        self.assertEqual(int(out.counts[2]), 0)
        self.assertEqual(int(out.errors[3]), proj.COUNT_OVERFLOW)
        self.assertEqual(int(out.counts[3]), 0)
        small = cpu(max_actions=16)
        overflow = small.project(_pack(_state(41)).arrays, torch.tensor([0], dtype=torch.int64),
                                 torch.ones(1, dtype=torch.bool))
        self.assertEqual(int(overflow.errors[0]), proj.COUNT_OVERFLOW)
        self.assertEqual(int(overflow.counts[0]), 0)
        self.assertEqual(small.max_actions, proj.SEARCH_CANDIDATE_CAP)
        self.assertEqual(proj.FULL_ACTION_CAPACITY, 800)

    def test_finished_is_empty_and_non_actor_playing_is_concede_only(self):
        packed = _pack(_state(51), _state(52))
        fields = {name: value.clone() for name, value in packed.arrays.items()}
        fields['phase'][0] = proj.PHASE_FINISHED
        fields['winner_presence'][0] = PRESENCE_VALUE
        fields['phase'][1] = proj.PHASE_PLAYING
        fields['active_side'][1] = 0
        out = cpu().project(fields, torch.tensor([0, 1], dtype=torch.int64), torch.ones(2, dtype=torch.bool))
        self.assertEqual(int(out.errors[0]), 0)
        self.assertEqual(int(out.counts[0]), 0)
        self.assertTrue(bool(out.public_flags[0, 3]))
        self.assertEqual(int(out.errors[1]), 0)
        self.assertEqual(int(out.counts[1]), 1)
        self.assertEqual(int(out.tokens[1, 0, 0]), proj.ACTION_CONCEDE)
        self.assertTrue(bool(out.public_flags[1, 1]))

    def test_cuda_constructor_without_fallback(self):
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                proj.DomainSoAProjection(device='cuda')
            return
        arena = proj.DomainSoAProjection(device='cuda')
        packed = _pack(_state(61))
        fields = {name: value.to(arena.device) for name, value in packed.arrays.items()}
        out = arena.project(fields, torch.zeros(1, dtype=torch.int64, device=arena.device),
                            torch.ones(1, dtype=torch.bool, device=arena.device))
        self.assertEqual(int(out.errors[0]), 0)
        self.assertGreater(int(out.counts[0]), 1)

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual 400-row CUDA graph required')
    def test_400_resident_projection_graph_matches_cpu_oracle(self):
        from tests.test_duel_v2_domain_soa_codec import _fourteen_initial_states
        states = _fourteen_initial_states()
        layout = DomainSoALayout(game_capacity=400)
        packed = encode_domain_states([states[i % len(states)] for i in range(400)],
                                      layout, presentation='drop').to_torch(device='cpu')
        viewers = torch.arange(400, dtype=torch.int64) % 2
        active = torch.ones(400, dtype=torch.bool)
        expected = cpu(layout=layout).project(packed, viewers, active)
        fields = {name: value.cuda() for name, value in packed.arrays.items()}
        gv, ga = viewers.cuda(), active.cuda()
        arena = proj.DomainSoAProjection(layout=layout, device='cuda')
        arena.project(fields, gv, ga)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = arena.project(fields, gv, ga)
        for offset in (0, 1000):
            fields['rng'].copy_(packed.arrays['rng'].cuda() + offset)
            graph.replay()
            torch.cuda.synchronize()
            for name in ('packed_named', 'tokens', 'mask', 'counts', 'mulligan_ordinals',
                         'actors', 'public_flags', 'errors'):
                self.assertTrue(torch.equal(getattr(actual, name).cpu(), getattr(expected, name)),
                                msg=name)
        del graph
        torch.cuda.synchronize()

    def _diff(self, names, got, oracle):
        bad = []
        for i, name in enumerate(names):
            if got[i] != oracle[i]:
                bad.append(f'{name}: {float(got[i])} != {float(oracle[i])}')
                if len(bad) >= 8:
                    break
        return '\n'.join(bad)
