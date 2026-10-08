"""Optional compiled opening mulligan: keep-all default, subset swap, both first sides."""
from __future__ import annotations

import tempfile
import unittest

from app.modules.card_game.content.duel_v2 import STARTER_DECK
from app.modules.card_game.engine.duel_v2 import acting_side, apply_action, legal_actions, new_game
from app.modules.card_game.rl.gpu_duel.catalog import (
    ACT_MULLIGAN, DECK_SLOTS, DISCARD_SLOTS, HAND_SLOTS, encode_public_deck_row,
)
from app.modules.card_game.rl.rule_ir.compiled_oracle import (
    legal_entries, map_python_action, mechan_from_row, pack_python_row, rebuild,
)
from app.modules.card_game.rl.rule_ir.layout import OFFSETS


def _hand(row, side):
    n = row[OFFSETS['hand_n'] + side]
    return [
        (row[OFFSETS['hand_kind'] + side * HAND_SLOTS + i],
         row[OFFSETS['hand_inst'] + side * HAND_SLOTS + i])
        for i in range(n)
    ]


def _deck(row, side):
    n = row[OFFSETS['deck_n'] + side]
    return [
        (row[OFFSETS['deck_kind'] + side * DECK_SLOTS + i],
         row[OFFSETS['deck_inst'] + side * DECK_SLOTS + i])
        for i in range(n)
    ]


def _zone_cards(row, side):
    return _hand(row, side) + _deck(row, side) + [
        (row[OFFSETS['discard_kind'] + side * DISCARD_SLOTS + i],
         row[OFFSETS['discard_inst'] + side * DISCARD_SLOTS + i])
        for i in range(row[OFFSETS['discard_n'] + side])
    ]


class CompiledMulliganNativeTest(unittest.TestCase):
    def setUp(self):
        from app.modules.card_game.rl.gpu_duel.compiled_backend import CompiledStarterBackend
        self.tmp = tempfile.TemporaryDirectory()
        try:
            self.backend = CompiledStarterBackend(self.tmp.name, backend='native')
        except Exception as exc:
            self.tmp.cleanup()
            self.fail(f'native compiled backend unavailable: {exc}')
        self.deck = encode_public_deck_row(STARTER_DECK)

    def tearDown(self):
        self.backend.close()
        self.tmp.cleanup()

    def _reset(self, seed, *, mulligan=False, escalation=True):
        return self.backend.reset_public_lists(
            [seed], [self.deck], [self.deck], escalation=escalation, mulligan=mulligan,
        )[0]

    def _legal_mulligan(self, row):
        return [item for item in legal_entries(row) if item['type'] == ACT_MULLIGAN]

    def test_backend_advertises_optional_mulligan(self):
        identity = self.backend.identity()
        self.assertTrue(self.backend.supports_mulligan)
        self.assertTrue(identity['supports_mulligan'])

    def test_keepall_legacy_matches_default_playing_start(self):
        for seed in (0, 1, 7):
            default = self.backend.reset_public_lists([seed], [self.deck], [self.deck], True)[0]
            explicit = self._reset(seed, mulligan=False)
            self.assertEqual(default, explicit)
            snap = mechan_from_row(explicit)
            self.assertEqual(snap['phase'], 1)
            first = explicit[OFFSETS['first']]
            self.assertEqual(snap['active'], first)
            self.assertEqual(snap['turn'], 1)
            self.assertEqual(explicit[OFFSETS['pending_count']], 0)
            self.assertEqual(snap['sides'][first]['hand_n'], 7)
            self.assertEqual(snap['sides'][1 - first]['hand_n'], 5)

    def test_mulligan_reset_both_first_sides_before_turn_draw(self):
        for seed, first in ((0, 0), (1, 1)):
            row = self._reset(seed, mulligan=True)
            self.assertEqual(row[OFFSETS['error']], 0)
            self.assertEqual(row[OFFSETS['phase']], 0)
            self.assertEqual(row[OFFSETS['first']], first)
            self.assertEqual(row[OFFSETS['active']], 0)
            self.assertEqual(row[OFFSETS['turn']], 0)
            self.assertEqual(row[OFFSETS['pending_count']], 0)
            for side in (0, 1):
                self.assertEqual(row[OFFSETS['hand_n'] + side], 5)
                self.assertEqual(row[OFFSETS['deck_n'] + side], 27)
                self.assertEqual(row[OFFSETS['nanali_seed'] + side], 0)
                self.assertEqual(row[OFFSETS['ap'] + side], 0)
                self.assertEqual(row[OFFSETS['shield'] + side], 0 if side == first else 5)
            legal = self._legal_mulligan(row)
            self.assertEqual(len(legal), 26)
            self.assertTrue(all(item['actor'] == -1 and item['tside'] == -1 and item['tseat'] == -1 for item in legal))
            masks = sorted(item['hand'] for item in legal)
            expected = [mask for mask in range(32) if bin(mask).count('1') <= 3]
            self.assertEqual(masks, expected)
            self.assertFalse(any(item['type'] != ACT_MULLIGAN for item in legal_entries(row)))

    def test_python_pack_maps_skip_mulligan_false_and_pending_mask(self):
        for first in ('a', 'b'):
            state = new_game(seed=9, first_side=first, skip_mulligan=False,
                             decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            packed = pack_python_row(state)
            self.assertEqual(packed[OFFSETS['phase']], 0)
            self.assertEqual(packed[OFFSETS['first']], 0 if first == 'a' else 1)
            self.assertEqual(packed[OFFSETS['active']], 0)
            self.assertEqual(packed[OFFSETS['pending_count']], 0)
            self.assertEqual(acting_side(state), 'a')
            rebuilt = rebuild(self.backend, packed)
            self.assertEqual(len(self._legal_mulligan(rebuilt)), 26)
            keep = {'type': 'mulligan', 'card_ids': []}
            index = map_python_action(rebuilt, state, keep)
            nxt = self.backend.step_lists([rebuilt], [index])[0]
            state = apply_action(state, 'a', keep)
            packed_after = pack_python_row(state)
            self.assertEqual(nxt[OFFSETS['phase']], 0)
            self.assertEqual(nxt[OFFSETS['active']], 1)
            self.assertEqual(nxt[OFFSETS['pending_count']], 1)
            self.assertEqual(packed_after[OFFSETS['pending_count']], 1)
            self.assertEqual(packed_after[OFFSETS['active']], 1)
            self.assertEqual(acting_side(state), 'b')

    def test_no_immediate_redraw_of_returned_instances(self):
        row = self._reset(4, mulligan=True)
        before = _hand(row, 0)
        mask = 0b00111
        legal = {item['hand']: item['index'] for item in self._legal_mulligan(row)}
        nxt = self.backend.step_lists([row], [legal[mask]])[0]
        removed = [before[i] for i in range(5) if mask & (1 << i)]
        kept = [before[i] for i in range(5) if not (mask & (1 << i))]
        after = _hand(nxt, 0)
        after_inst = {inst for _kind, inst in after}
        for card in removed:
            self.assertNotIn(card, after)
            self.assertNotIn(card[1], after_inst)
            self.assertIn(card, _deck(nxt, 0))
        for card in kept:
            self.assertIn(card, after)
        self.assertEqual(len(after), 5)
        self.assertEqual(nxt[OFFSETS['deck_n']], 27)
        insts = [inst for _k, inst in _zone_cards(nxt, 0)]
        self.assertEqual(len(insts), 32)
        self.assertEqual(len(set(insts)), 32)
        self.assertEqual(nxt[OFFSETS['phase']], 0)
        self.assertEqual(nxt[OFFSETS['active']], 1)
        self.assertEqual(nxt[OFFSETS['pending_count']], 1)

    def test_both_seats_choose_then_first_turn_starts_once(self):
        for seed, first in ((0, 0), (1, 1)):
            row = self._reset(seed, mulligan=True)
            keep = {item['hand']: item['index'] for item in self._legal_mulligan(row)}[0]
            after_a = self.backend.step_lists([row], [keep])[0]
            self.assertEqual(after_a[OFFSETS['phase']], 0)
            self.assertEqual(after_a[OFFSETS['active']], 1)
            self.assertEqual(after_a[OFFSETS['turn']], 0)
            self.assertEqual(after_a[OFFSETS['hand_n'] + first], 5)
            keep_b = {item['hand']: item['index'] for item in self._legal_mulligan(after_a)}[0]
            after = self.backend.step_lists([after_a], [keep_b])[0]
            self.assertEqual(after[OFFSETS['phase']], 1)
            self.assertEqual(after[OFFSETS['active']], first)
            self.assertEqual(after[OFFSETS['turn']], 1)
            self.assertEqual(after[OFFSETS['pending_count']], 0)
            self.assertEqual(after[OFFSETS['hand_n'] + first], 7)
            self.assertEqual(after[OFFSETS['hand_n'] + (1 - first)], 5)
            self.assertEqual(after[OFFSETS['nanali_seed'] + first], 1)
            self.assertEqual(after[OFFSETS['nanali_seed'] + (1 - first)], 0)
            self.assertEqual(after[OFFSETS['ap'] + first], 1)

    def test_determinism_and_native_batch_mask(self):
        first = self._reset(12, mulligan=True)
        second = self._reset(12, mulligan=True)
        self.assertEqual(first, second)
        keep = {item['hand']: item['index'] for item in self._legal_mulligan(first)}[0]
        a = self.backend.step_lists([list(first)], [keep])[0]
        b = self.backend.step_lists([list(second)], [keep])[0]
        self.assertEqual(a, b)
        preserved = list(first)
        preserved[OFFSETS['ap']] = 9
        flags = 1 | 2
        out = self.backend.launch.public_reset_lists(
            [preserved, list(first)], [-1, 12], [self.deck, self.deck], [self.deck, self.deck], flags,
        )
        self.assertEqual(out[0], preserved)
        self.assertEqual(out[1], first)

    def test_python_native_keepall_and_swap_invariants(self):
        for first in ('a', 'b'):
            state = new_game(seed=11, first_side=first, skip_mulligan=False,
                             decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            packed = rebuild(self.backend, pack_python_row(state))
            py_legal = [action for action in legal_actions(state, 'a') if action['type'] == 'mulligan']
            self.assertEqual(len(py_legal), 26)
            hand_a = list(state['sides']['a']['hand'])
            swap_ids = [card['instance_id'] for card in hand_a[:2]]
            swap = {'type': 'mulligan', 'card_ids': swap_ids}
            compiled = self.backend.step_lists(
                [packed], [map_python_action(packed, state, swap)],
            )[0]
            python = apply_action(state, 'a', swap)
            swapped = {int(card_id.rsplit('-', 1)[-1]) for card_id in swap_ids}
            compiled_hand = {inst for _k, inst in _hand(compiled, 0)}
            compiled_deck = {inst for _k, inst in _deck(compiled, 0)}
            self.assertTrue(swapped.isdisjoint(compiled_hand))
            self.assertTrue(swapped.issubset(compiled_deck))
            py_hand = {int(card['instance_id'].rsplit('-', 1)[-1]) for card in python['sides']['a']['hand']}
            py_deck = {int(card['instance_id'].rsplit('-', 1)[-1]) for card in python['sides']['a']['deck']}
            self.assertTrue(swapped.isdisjoint(py_hand))
            self.assertTrue(swapped.issubset(py_deck))
            self.assertEqual(compiled[OFFSETS['phase']], 0)
            self.assertEqual(compiled[OFFSETS['active']], 1)
            self.assertEqual(compiled[OFFSETS['pending_count']], 1)
            self.assertEqual(pack_python_row(python)[OFFSETS['pending_count']], 1)
            self.assertEqual(len(python['sides']['a']['hand']), 5)
            self.assertEqual(compiled[OFFSETS['hand_n']], 5)

            keep_b = {'type': 'mulligan', 'card_ids': []}
            packed_b = rebuild(self.backend, pack_python_row(python))
            compiled_b = self.backend.step_lists(
                [packed_b], [map_python_action(packed_b, python, keep_b)],
            )[0]
            python_b = apply_action(python, 'b', keep_b)
            self.assertEqual(compiled_b[OFFSETS['phase']], 1)
            self.assertEqual(python_b['phase'], 'playing')
            self.assertEqual(compiled_b[OFFSETS['pending_count']], 0)
            self.assertEqual(compiled_b[OFFSETS['active']], 0 if first == 'a' else 1)
            self.assertEqual(python_b['active_side'], first)
            self.assertEqual(compiled_b[OFFSETS['turn']], 1)
            first_i = 0 if first == 'a' else 1
            self.assertEqual(len(python_b['sides'][first]['hand']), compiled_b[OFFSETS['hand_n'] + first_i])
            self.assertEqual(len(python_b['sides'][first]['hand']), 7)


if __name__ == '__main__':
    unittest.main()
