"""Isolated pacing checks; these do not approve incomplete legacy training engines."""
from copy import deepcopy
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

from app.modules.card_game.engine.duel_v2 import new_game, apply_action
from app.modules.card_game.engine.duel_v2.state import card_instance
from app.modules.card_game.rl.gpu_duel.catalog import CARD_INDEX, N_SEATS, SEAT_INDEX
from app.modules.card_game.rl.rule_ir.layout import OFFSETS
from app.modules.card_game.rl.rule_ir.pack_row import pack_python_row


def fixture(before, enabled=True):
    state = new_game(seed=7, first_side='a', skip_mulligan=True, escalation=enabled)
    state['turn'] = before
    for team in state['sides'].values():
        for h in team['characters'].values():
            h.update(energy=2, harmony=2)
    state['sides']['b']['discard'] = [card_instance(state, cid, 'b') for cid in ('NF01', 'J01')]
    return state


class NativeEscalationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.modules.card_game.rl.rule_ir.compile_engine import compile_native_engine
        cls.tmp = tempfile.TemporaryDirectory()
        # Known content/IR completeness gaps remain blocked in production.
        with patch('app.modules.card_game.rl.rule_ir.emit_engine.validate_compiled_deck'):
            cls.engine = compile_native_engine(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        cls.tmp.cleanup()

    def test_thresholds_recovery_identity_and_disabled_baseline(self):
        from app.modules.card_game.rl.rule_ir.compiled_oracle import map_python_action
        for before, enabled in ((4, True), (5, True), (12, True), (19, True), (19, False)):
            with self.subTest(before=before, enabled=enabled):
                state = fixture(before, enabled)
                row = self.engine.launch_lists([pack_python_row(state)], [-1])[0]
                action = {'type': 'end_turn'}
                index = map_python_action(row, state, action)
                got = self.engine.launch_lists([row], [index])[0]
                expected = pack_python_row(apply_action(state, 'a', action))
                for field, width in (('ap', 2), ('ch_energy', 2 * N_SEATS),
                                     ('ch_harmony', 2 * N_SEATS), ('hand_n', 2), ('deck_n', 2),
                                     ('discard_n', 2), ('hand_kind', 20), ('hand_inst', 20)):
                    offset = OFFSETS[field]
                    self.assertEqual(got[offset:offset + width], expected[offset:offset + width], field)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class TensorEscalationTest(unittest.TestCase):
    def test_thresholds_and_batch_mask(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.state import empty_state
        from app.modules.card_game.rl.rule_ir.pack_row import unpack_gpu_row
        from app.modules.card_game.rl.gpu_duel.rules import begin_turn
        # Covers pre-6, 6, 13, 20, disabled, and an untouched batch row.
        states = [fixture(t, enable) for t, enable in
                  ((4, True), (5, True), (12, True), (19, True), (19, False), (19, True))]
        s = empty_state(len(states), 'cpu')
        for index, state in enumerate(states):
            unpack_gpu_row(s, pack_python_row(state), index)
        mask = torch.tensor([True] * 5 + [False])
        begin_turn(s, torch.ones(6, dtype=torch.long), mask)
        self.assertEqual(s.ap[:, 1].tolist(), [2, 2, 2, 3, 2, 0])
        for i in range(5):
            expected = apply_action(states[i], 'a', {'type': 'end_turn'})
            for si, side in enumerate(('a', 'b')):
                for cid, h in expected['sides'][side]['characters'].items():
                    seat = SEAT_INDEX[cid]
                    self.assertEqual(s.ch_energy[i, si, seat].item(), h['energy'])
                    self.assertEqual(s.ch_harmony[i, si, seat].item(), h['harmony'])
            team = expected['sides']['b']
            self.assertEqual(s.hand_n[i, 1].item(), len(team['hand']))
            self.assertEqual(s.deck_n[i, 1].item(), len(team['deck']))
            self.assertEqual(s.discard_n[i, 1].item(), len(team['discard']))
        self.assertEqual(s.turn[5].item(), 19)

    def test_random_recovery_preserves_instances_and_full_hand(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.state import empty_state
        from app.modules.card_game.rl.rule_ir.pack_row import unpack_gpu_row
        from app.modules.card_game.rl.gpu_duel.rules import _recover_buildable
        states = []
        for seed in range(16):
            state = fixture(19)
            state['rng'] = seed
            state['sides']['b']['discard'] += [card_instance(state, 'N01', 'b')]
            states.append(state)
        s = empty_state(len(states), 'cpu')
        for index, state in enumerate(states):
            unpack_gpu_row(s, pack_python_row(state), index)
        other = deepcopy(s)
        mask = torch.ones(len(states), dtype=torch.bool)
        _recover_buildable(s, 1, mask)
        _recover_buildable(other, 1, mask)
        self.assertTrue(torch.equal(s.hand_inst, other.hand_inst))
        self.assertEqual(s.discard_kind[:, 1, 0].tolist(), [CARD_INDEX['NF01']] * len(states))
        self.assertEqual(s.hand_n[:, 1].tolist(), [6] * len(states))
        picked = set(s.hand_kind[:, 1, 5].tolist())
        self.assertEqual(picked, {CARD_INDEX['J01'], CARD_INDEX['N01']})
        self.assertTrue(torch.all(s.hand_inst[:, 1, 5] > 0))
        s.hand_n[:, 1] = 10
        _recover_buildable(s, 1, mask)
        self.assertEqual(s.hand_n[:, 1].tolist(), [10] * len(states))
        self.assertEqual(s.discard_n[:, 1].tolist(), [2] * len(states))
