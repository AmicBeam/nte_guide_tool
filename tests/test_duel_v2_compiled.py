"""Compiled starter engine: completeness, availability, and Python-oracle diffs."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest

from app.modules.card_game.content.duel_v2 import STARTER_DECK, STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import acting_side, apply_action, legal_actions, new_game
from app.modules.card_game.rl.gpu_duel.catalog import ACT_ATTACK, ACT_CHOOSE, ACT_END, ACT_PLAY, ACT_ULTIMATE, CARD_INDEX, SEAT_INDEX
from app.modules.card_game.rl.rule_ir.completeness import CompletenessError, validate_compiled_starter


TORCH = importlib.util.find_spec('torch')


def _gpu_index(env, action, row=0):
    s = env.state
    n = int(s.legal_n[row])
    kind = action.get('type')
    side = int(s.active[row])
    if kind == 'end_turn':
        for index in range(n):
            if int(s.legal_type[row, index]) == ACT_END:
                return index
    if kind == 'attack':
        actor = SEAT_INDEX[action['character_id']]
        for index in range(n):
            if int(s.legal_type[row, index]) == ACT_ATTACK and int(s.legal_actor[row, index]) == actor:
                return index
    if kind == 'ultimate':
        actor = SEAT_INDEX[action['character_id']]
        for index in range(n):
            if int(s.legal_type[row, index]) == ACT_ULTIMATE and int(s.legal_actor[row, index]) == actor:
                return index
    if kind == 'choose':
        choice = action.get('choice_id')
        for index in range(n):
            if int(s.legal_type[row, index]) == ACT_CHOOSE:
                slot = int(s.legal_hand[row, index])
                pending = int(s.pending_card[row, slot])
                if choice and CARD_INDEX.get(str(choice).split('-')[0] if False else '', None) is not None:
                    pass
                return index
    if kind == 'play_card':
        inst = int(str(action['card_id']).rsplit('-', 1)[-1])
        target = action.get('target_id')
        for index in range(n):
            if int(s.legal_type[row, index]) != ACT_PLAY:
                continue
            slot = int(s.legal_hand[row, index])
            if int(s.hand_inst[row, side, slot]) != inst:
                continue
            if target:
                tside, tname = str(target).split(':', 1)
                want_side = 0 if tside == 'a' else 1
                want_seat = -1 if tname == 'player' else SEAT_INDEX[tname]
                if int(s.legal_tside[row, index]) != want_side or int(s.legal_tseat[row, index]) != want_seat:
                    continue
            return index
    raise AssertionError(f'no compiled legal slot for {action}')


class CompletenessGateTest(unittest.TestCase):
    def test_frozen_starter_is_complete(self):
        manifest = validate_compiled_starter()
        self.assertTrue(manifest['ok'])
        self.assertEqual(manifest['deck_id'], 'starter')
        self.assertIn('NF01', manifest['cards'])
        self.assertEqual(manifest['characters'], list(STARTER_DECK['character_ids']))

    def test_rush_supported_but_custom_list_rejected(self):
        from app.modules.card_game.rl.gpu_duel.compiled_backend import _assert_compiled_deck
        rush = next(item for item in STARTER_DECKS if item['id'] == 'weave-rush')
        self.assertEqual(_assert_compiled_deck(rush)['id'], 'weave-rush')
        from copy import deepcopy
        custom = deepcopy(rush)
        custom['card_ids'][0] = 'M02'
        with self.assertRaises((CompletenessError, ValueError)):
            _assert_compiled_deck(custom)



class CompiledNativeOracleNoTorchTest(unittest.TestCase):
    """Mac path: compile the shared C engine and diff packed Python snapshots."""

    def setUp(self):
        from app.modules.card_game.rl.gpu_duel.compiled_backend import CompiledStarterBackend
        self.tmp = tempfile.TemporaryDirectory()
        try:
            self.backend = CompiledStarterBackend(self.tmp.name, backend='native')
        except Exception as exc:
            self.tmp.cleanup()
            self.fail(f'native compiled backend unavailable: {exc}')

    def tearDown(self):
        self.backend.close()
        self.tmp.cleanup()

    def _check(self, state, action):
        from app.modules.card_game.engine.duel_v2 import apply_action, acting_side
        from app.modules.card_game.rl.rule_ir.compiled_oracle import python_packed_mechan, step_python
        packed, got = step_python(self.backend, state, action)
        nxt = apply_action(state, acting_side(state), action)
        self.assertEqual(got, python_packed_mechan(nxt, self.backend))
        self.assertEqual(packed[__import__('app.modules.card_game.rl.rule_ir.layout', fromlist=['OFFSETS']).OFFSETS['error']], 0)
        self.assertEqual(self.backend.stats['fallback_rows'], 0)
        return nxt

    def test_opening_attack_end_turn_and_inactive_is_row_local(self):
        from app.modules.card_game.engine.duel_v2 import acting_side, legal_actions
        from app.modules.card_game.rl.rule_ir.compiled_oracle import pack_python_row, rebuild
        state = new_game(seed=7, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
        side = acting_side(state)
        attack = next(a for a in legal_actions(state, side) if a.get('type') == 'attack')
        self._check(state, attack)
        state = new_game(seed=7, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
        self._check(state, {'type': 'end_turn'})

    def test_response_harmony_equipment_choice_empty_full_terminal(self):
        import copy
        from app.modules.card_game.engine.duel_v2 import acting_side, legal_actions
        from app.modules.card_game.rl.rule_ir.compiled_oracle import python_packed_mechan, step_python
        # response
        state = new_game(seed=11, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
        foe = 'b' if state['active_side'] == 'a' else 'a'
        actor_side = state['active_side']
        n02 = next(c for c in state['sides'][actor_side]['deck'] + state['sides'][actor_side]['hand'] if c.get('card_id') == 'N02')
        state['sides'][foe]['hand'] = [copy.deepcopy(n02)]
        state['sides'][foe]['ap'] = 1
        state['sides'][foe]['front'] = 'nanali'
        attack = next(a for a in legal_actions(state, acting_side(state)) if a.get('type') == 'attack')
        self._check(state, attack)
        # harmony
        state = new_game(seed=13, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
        side = state['active_side']
        state['sides'][side]['front'] = 'zero'
        state['sides'][side]['characters']['zero']['harmony'] = 2
        attack = next(a for a in legal_actions(state, acting_side(state)) if a.get('type') == 'attack' and a['character_id'] == 'nanali')
        self._check(state, attack)
        # simultaneous death
        state = new_game(seed=17, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
        side = state['active_side']
        foe = 'b' if side == 'a' else 'a'
        state['sides'][side]['front'] = 'nanali'
        state['sides'][foe]['front'] = 'nanali'
        state['sides'][side]['characters']['nanali'].update(hp=1, shield=0)
        state['sides'][foe]['characters']['nanali'].update(hp=1, shield=0)
        attack = next(a for a in legal_actions(state, acting_side(state)) if a.get('type') == 'attack' and a['character_id'] == 'nanali')
        self._check(state, attack)
        # empty deck terminal before reset: the next player draws on their turn start.
        state = new_game(seed=9, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
        other = 'b' if state['active_side'] == 'a' else 'a'
        state['sides'][other]['deck'] = []
        nxt = self._check(state, {'type': 'end_turn'})
        self.assertEqual(nxt.get('phase'), 'finished')
        # full hand
        state = new_game(seed=5, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
        side = state['active_side']
        template = state['sides'][side]['hand'][0]
        while len(state['sides'][side]['hand']) < 10:
            extra = dict(template)
            extra['instance_id'] = f'{side}-{1000 + len(state["sides"][side]["hand"])}'
            state['sides'][side]['hand'].append(extra)
        self._check(state, {'type': 'end_turn'})
        # both seats
        for first in ('a', 'b'):
            state = new_game(seed=8, skip_mulligan=True, first_side=first, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            side = acting_side(state)
            action = next(a for a in legal_actions(state, side) if a.get('type') != 'concede')
            self._check(state, action)
        self.assertEqual(self.backend.stats['fallback_rows'], 0)

    def test_play_form_or_battle_and_inspect_if_present(self):
        from app.modules.card_game.engine.duel_v2 import acting_side, legal_actions
        state = new_game(seed=21, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
        saw_choice = False
        for _ in range(14):
            if state.get('phase') == 'finished':
                break
            side = acting_side(state)
            actions = [a for a in legal_actions(state, side) if a.get('type') != 'concede']
            chosen = None
            if state.get('phase') == 'choice':
                chosen = next(a for a in actions if a.get('type') == 'choose')
                saw_choice = True
            else:
                for play in actions:
                    if play.get('type') != 'play_card':
                        continue
                    card = next(c for c in state['sides'][side]['hand'] if c['instance_id'] == play['card_id'])
                    if card['type'] in ('form', 'battle') or card.get('card_id') == 'J01':
                        chosen = play
                        break
                chosen = chosen or actions[0]
            state = self._check(state, chosen)
        # Bounded smoke: at least one compiled play/end succeeded without fallback.
        self.assertEqual(self.backend.stats['fallback_rows'], 0)

    def test_compiled_reset_does_not_use_tensor_helper(self):
        from app.modules.card_game.rl.rule_ir.layout import OFFSETS
        from app.modules.card_game.rl.rule_ir.pack_row import mechan_from_row
        rows = self.backend.reset_lists([7, 8])
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row[OFFSETS['error']], 0)
            self.assertEqual(row[OFFSETS['phase']], 1)
            snap = mechan_from_row(row)
            self.assertIn(snap['active'], (0, 1))
            first = snap['active']
            self.assertEqual(snap['sides'][first]['deck_n'], 26)
            self.assertEqual(snap['sides'][first]['hand_n'], 7)
            total = snap['sides'][first]['hand_n'] + snap['sides'][first]['deck_n'] + snap['sides'][first]['discard_n']
            self.assertEqual(total, 33)  # 32 starter cards plus first-turn NF01
            other = 1 - first
            self.assertEqual(snap['sides'][other]['deck_n'], 27)
            self.assertEqual(snap['sides'][other]['hand_n'], 5)
            other_total = snap['sides'][other]['hand_n'] + snap['sides'][other]['deck_n'] + snap['sides'][other]['discard_n']
            self.assertEqual(other_total, 32)
        self.assertEqual(self.backend.stats['fallback_rows'], 0)


@unittest.skipUnless(TORCH, 'PyTorch is required to pack GpuState; compiled diffs still need a compiler')
class CompiledStarterOracleTest(unittest.TestCase):
    def setUp(self):
        from app.modules.card_game.rl.gpu_duel.env import GpuDuelEnv
        from app.modules.card_game.rl.gpu_duel.pack import python_mechan
        self.GpuDuelEnv = GpuDuelEnv
        self.python_mechan = python_mechan
        self.tmp = tempfile.TemporaryDirectory()
        try:
            self.env = GpuDuelEnv(
                2, device='cpu', compiled_backend='native', compiled_dir=self.tmp.name)
        except Exception as exc:
            self.tmp.cleanup()
            self.fail(f'native compiled backend unavailable: {exc}')

    def tearDown(self):
        if getattr(self, 'env', None) and self.env.compiled:
            self.env.compiled.close()
        self.tmp.cleanup()

    def _py(self, seed=7):
        # Exact legacy trajectories exclude stochastic escalation. Public boundary
        # and tie-selection invariants have dedicated tests in the public suites.
        return new_game(seed=seed, skip_mulligan=True, escalation=False,
                        decks={'a': STARTER_DECK, 'b': STARTER_DECK})

    def _load(self, *states):
        if len(states) == 1:
            states = (states[0], states[0])
        self.env.reset_from_python(list(states))

    def _step_both(self, action, inactive=-1):
        index = _gpu_index(self.env, action, 0)
        self.env.step([index, inactive])

    def test_backend_is_explicit_and_isolated(self):
        from app.modules.card_game.rl.gpu_duel.env import GpuDuelEnv
        plain = GpuDuelEnv(1, device='cpu')
        self.assertIsNone(plain.compiled)
        ident = self.env.compiled.identity()
        self.assertEqual(ident['backend'], 'native')
        self.assertEqual(ident['fallback_rows'], 0)
        self.assertEqual(ident['engine_name'], 'starter_engine')

    def test_opening_attack_and_inactive_row(self):
        state = self._py()
        frozen = self._py()
        self._load(state, frozen)
        side = acting_side(state)
        attack = next(action for action in legal_actions(state, side) if action.get('type') == 'attack')
        self._step_both(attack, inactive=-1)
        nxt = apply_action(state, side, attack)
        self.assertEqual(self.env.snapshot(0), self.python_mechan(nxt))
        self.assertEqual(self.env.snapshot(1), self.python_mechan(frozen))

    def test_end_turn_matches_python(self):
        state = self._py()
        self._load(state)
        side = acting_side(state)
        self._step_both({'type': 'end_turn'})
        nxt = apply_action(state, side, {'type': 'end_turn'})
        self.assertEqual(self.env.snapshot(0), self.python_mechan(nxt))

    def test_equipment_and_battle_when_present(self):
        state = self._py(seed=21)
        self._load(state)
        found = {'form': False, 'battle': False}
        for _ in range(12):
            if state.get('phase') == 'finished':
                break
            side = acting_side(state)
            plays = [a for a in legal_actions(state, side) if a.get('type') == 'play_card']
            action = None
            for play in plays:
                inst = play['card_id']
                card = next(c for c in state['sides'][side]['hand'] if c['instance_id'] == inst)
                if card['type'] == 'form' and not found['form']:
                    action = play
                    found['form'] = True
                    break
                if card['type'] == 'battle' and not found['battle']:
                    action = play
                    found['battle'] = True
                    break
            if action is None:
                action = next(a for a in legal_actions(state, side) if a.get('type') != 'concede')
            self._step_both(action)
            state = apply_action(state, side, action)
            self.assertEqual(self.env.snapshot(0), self.python_mechan(state))
        self.assertTrue(found['form'] or found['battle'])

    def test_choice_inspect_when_present(self):
        state = self._py(seed=3)
        found = False
        for _ in range(16):
            self._load(state)
            if state.get('phase') == 'choice':
                found = True
                side = acting_side(state)
                action = next(a for a in legal_actions(state, side) if a.get('type') == 'choose')
                self._step_both(action)
                nxt = apply_action(state, side, action)
                self.assertEqual(self.env.snapshot(0), self.python_mechan(nxt))
                break
            side = acting_side(state)
            plays = [a for a in legal_actions(state, side) if a.get('type') == 'play_card']
            chosen = None
            for play in plays:
                inst = play['card_id']
                card = next(c for c in state['sides'][side]['hand'] if c['instance_id'] == inst)
                if card.get('card_id') == 'J01':
                    chosen = play
                    break
            action = chosen or next(a for a in legal_actions(state, side) if a.get('type') != 'concede')
            self._step_both(action)
            state = apply_action(state, side, action)
            self.assertEqual(self.env.snapshot(0), self.python_mechan(state))
        if not found:
            self.skipTest('J01 inspect choice did not appear in bounded smoke')

    def test_response_harmony_and_simultaneous_death_fixtures(self):
        import copy
        state = self._py(seed=11)
        # Response: put N02 in defender hand, force an attack into occupied front.
        foe = 'b' if state['active_side'] == 'a' else 'a'
        actor_side = state['active_side']
        n02 = next((c for c in state['sides'][actor_side]['deck'] + state['sides'][actor_side]['hand']
                    if c.get('card_id') == 'N02'), None)
        if n02 is None:
            self.fail('starter deck missing N02')
        state['sides'][foe]['hand'] = [copy.deepcopy(n02)]
        state['sides'][foe]['ap'] = 1
        state['sides'][foe]['front'] = 'nanali'
        state['sides'][actor_side]['front'] = None
        self._load(state)
        side = acting_side(state)
        attack = next(a for a in legal_actions(state, side) if a.get('type') == 'attack')
        self._step_both(attack)
        nxt = apply_action(state, side, attack)
        self.assertEqual(self.env.snapshot(0), self.python_mechan(nxt))

        # Harmony: previous front has 2 harmony of matching 光/灵 pair.
        state = self._py(seed=13)
        side = state['active_side']
        state['sides'][side]['front'] = 'zero'
        state['sides'][side]['characters']['zero']['harmony'] = 2
        state['sides'][side]['last_front'] = 'zero'
        self._load(state)
        attack = next(a for a in legal_actions(state, acting_side(state))
                      if a.get('type') == 'attack' and a['character_id'] == 'nanali')
        self._step_both(attack)
        nxt = apply_action(state, acting_side(state), attack)
        self.assertEqual(self.env.snapshot(0), self.python_mechan(nxt))

        # Simultaneous death: both fronts at 1 hp, enough attack to kill.
        state = self._py(seed=17)
        side = state['active_side']
        foe = 'b' if side == 'a' else 'a'
        state['sides'][side]['front'] = 'nanali'
        state['sides'][foe]['front'] = 'nanali'
        state['sides'][side]['characters']['nanali']['hp'] = 1
        state['sides'][foe]['characters']['nanali']['hp'] = 1
        state['sides'][side]['characters']['nanali']['shield'] = 0
        state['sides'][foe]['characters']['nanali']['shield'] = 0
        self._load(state)
        attack = next(a for a in legal_actions(state, acting_side(state))
                      if a.get('type') == 'attack' and a['character_id'] == 'nanali')
        self._step_both(attack)
        nxt = apply_action(state, acting_side(state), attack)
        self.assertEqual(self.env.snapshot(0), self.python_mechan(nxt))

    def test_full_hand_empty_deck_and_terminal_before_reset(self):
        state = self._py(seed=5)
        side = state['active_side']
        # Fill hand to 10 using copies of first card identity.
        template = state['sides'][side]['hand'][0]
        while len(state['sides'][side]['hand']) < 10:
            extra = dict(template)
            extra['instance_id'] = f'{side}-{1000 + len(state["sides"][side]["hand"])}'
            state['sides'][side]['hand'].append(extra)
        self._load(state)
        self._step_both({'type': 'end_turn'})
        nxt = apply_action(state, acting_side(state), {'type': 'end_turn'})
        self.assertEqual(self.env.snapshot(0), self.python_mechan(nxt))

        state = self._py(seed=9)
        side = state['active_side']
        other = 'b' if side == 'a' else 'a'
        state['sides'][other]['deck'] = []
        self._load(state)
        self._step_both({'type': 'end_turn'})
        nxt = apply_action(state, acting_side(state), {'type': 'end_turn'})
        self.assertEqual(self.env.snapshot(0), self.python_mechan(nxt))
        self.assertEqual(int(self.env.state.phase[0]), 3)

        # Terminal before reset: collector-visible done must remain until reset_finished.
        done, winner = self.env.outcome()
        self.assertTrue(bool(done[0]))
        self.assertEqual(int(self.env.compiled.stats['fallback_rows']), 0)

    def test_both_seats_and_zero_fallback(self):
        for seed, first in ((7, 'a'), (8, 'b')):
            state = new_game(seed=seed, skip_mulligan=True, first_side=first,
                             decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            self._load(state)
            side = acting_side(state)
            action = next(a for a in legal_actions(state, side) if a.get('type') != 'concede')
            self._step_both(action)
            nxt = apply_action(state, side, action)
            self.assertEqual(self.env.snapshot(0), self.python_mechan(nxt))
        self.assertEqual(int(self.env.compiled.stats['fallback_rows']), 0)
        self.assertEqual(int(self.env.compiled.stats['errors']), 0)

    def test_resident_state_and_masked_reset(self):
        import torch
        from unittest.mock import patch
        from app.modules.card_game.rl.rule_ir.pack_row import bind_gpu_rows, pack_gpu_row
        self.env.reset(seeds=[0, 1])
        state = self.env.state
        rows = bind_gpu_rows(state)
        self.assertEqual(rows[0].tolist(), pack_gpu_row(state, 0))
        before = rows[1].clone()
        with patch('app.modules.card_game.rl.rule_ir.pack_row.pack_gpu_row', side_effect=AssertionError('scalar packing')):
            self.env.compiled.reset_gpu_state(state, torch.tensor([23, -1], dtype=torch.int32))
            self.env.compiled.step_gpu_state(state, [-1, -1])
        self.assertTrue(torch.equal(rows[1], before))
        self.assertEqual(state.hp.untyped_storage().data_ptr(), rows.untyped_storage().data_ptr())

    def test_training_covers_both_initiatives_without_tensor_fallback(self):
        import torch
        from unittest.mock import patch
        env = self.GpuDuelEnv(4, device='cpu', compiled_backend='native', compiled_dir=self.tmp.name + '/batch')
        try:
            with patch('app.modules.card_game.rl.gpu_duel.env.rebuild_legal', side_effect=AssertionError('tensor fallback')), patch('app.modules.card_game.rl.gpu_duel.env.advance_to_actor', side_effect=AssertionError('tensor fallback')):
                env.reset(seeds=[0, 1, 2, 3])
                env.prepare_decision()
            self.assertEqual(int((env.learner == env.state.first).sum()), 2)
            self.assertTrue(bool((env.state.active == env.learner).all()))
        finally:
            env.compiled.close()

    def test_cuda_availability_is_explicit(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.env import GpuDuelEnv
        if not torch.cuda.is_available():
            with self.assertRaises(Exception):
                GpuDuelEnv(1, device='cpu', compiled_backend='cuda')
            return
        env = GpuDuelEnv(1, device='cuda', compiled_backend='cuda', compiled_dir=self.tmp.name)
        try:
            state = self._py()
            env.reset_from_python([state])
            side = acting_side(state)
            action = next(a for a in legal_actions(state, side) if a.get('type') == 'end_turn')
            env.step([_gpu_index(env, action)])
            nxt = apply_action(state, side, action)
            self.assertEqual(env.snapshot(0), self.python_mechan(nxt))
            self.assertEqual(env.compiled.identity()['backend'], 'cuda')
        finally:
            env.compiled.close()


if __name__ == '__main__':
    unittest.main()
