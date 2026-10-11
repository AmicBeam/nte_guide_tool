"""CPU/CUDA tests for the bounded device training tape; not full-rules evidence."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from app.modules.card_game.content.duel_v2 import STARTER_DECKS, validate_deck
from app.modules.card_game.engine.duel_v2 import new_game, observe
from app.modules.card_game.rl import seven_lineup
from app.modules.card_game.rl.batched_duel import gpu_seven_candidates as encoder
from app.modules.card_game.rl.batched_duel.device_rule_protocol import (
    DeviceMeta, DeviceStateBatch, DeviceValueObservation,
)
from app.modules.card_game.rl.batched_search.compact_device_candidates import (
    CompactDeviceCandidates, _TOKEN_TENSORS,
)
from app.modules.card_game.rl.batched_search.device_training_tape import (
    CAPACITY_ERROR, CAVEAT, DeviceTrainingTape, PI_ERROR, STEP_ERROR,
)


def _cases(skips=(True,)):
    cases = []
    for definition in STARTER_DECKS:
        if definition['id'] not in seven_lineup.KEYS:
            continue
        deck = validate_deck(deepcopy(definition))
        for skip in skips:
            view = observe(new_game(seed=7, first_side='a',
                                    decks={'a': deck, 'b': deepcopy(deck)},
                                    skip_mulligan=skip), 'a', include_previews=False)
            actions = [row['action'] for row in view['legal_actions']
                       if row['action']['type'] != 'concede']
            cases.append((view, actions))
    return cases


def _tokens(cases, device='cpu', capacity=None):
    import torch
    raw = encoder.extract_raw_tokens([view for view, _ in cases],
                                     [actions for _, actions in cases],
                                     capacity=capacity)
    values = {field: torch.as_tensor(getattr(raw, field), device=device).contiguous()
              for field in _TOKEN_TENSORS}
    return encoder.RawCandidateTokens(**values)


def _observations(cases, device='cpu'):
    import torch
    rows = [seven_lineup.encode(view, actions)[0] for view, actions in cases]
    return torch.as_tensor(np.stack(rows), dtype=torch.float32, device=device).contiguous()


def _compact(tokens, scratch_rows=4, device='cpu'):
    table = encoder.get_card_template_table(tokens.int_tokens.device)
    return CompactDeviceCandidates(
        tokens, scratch_rows=scratch_rows, device=device,
        allow_cpu_reference=device == 'cpu', template_table=table,
    )


class ControllableRules:
    def __init__(self, rows, device, observation_dim):
        import torch
        self.t = torch
        self.device = device
        self.actors = torch.zeros(rows, dtype=torch.int64, device=device)
        self.phases = torch.zeros(rows, dtype=torch.int32, device=device)
        self.winners = torch.full((rows,), -2, dtype=torch.int64, device=device)
        self.finished = torch.zeros(rows, dtype=torch.bool, device=device)
        self.versions = torch.arange(rows, dtype=torch.int64, device=device) + 1
        self.meta_errors = torch.zeros(rows, dtype=torch.int32, device=device)
        self.value_obs = torch.zeros((rows, observation_dim), dtype=torch.float32, device=device)
        self.terminal = torch.zeros(rows, dtype=torch.bool, device=device)
        self.terminal_value = torch.zeros(rows, dtype=torch.float64, device=device)
        self.value_errors = torch.zeros(rows, dtype=torch.int32, device=device)

    def metadata_device(self, states):
        return DeviceMeta(self.actors, self.phases, self.winners, self.finished,
                          self.versions, self.meta_errors)

    def value_observation_device(self, states, viewers):
        ctx = (self.actors == viewers).to(self.t.float32)[:, None].contiguous()
        return DeviceValueObservation(self.value_obs, ctx, self.terminal,
                                      self.terminal_value, self.value_errors)


def _roots(rows, device, active=None):
    import torch
    slots = torch.arange(rows, dtype=torch.int64, device=device)
    gens = torch.ones(rows, dtype=torch.int64, device=device)
    owners = torch.arange(rows, dtype=torch.int64, device=device)
    if active is None:
        mask = torch.ones(rows, dtype=torch.bool, device=device)
    else:
        mask = torch.tensor(active, dtype=torch.bool, device=device)
    errors = torch.zeros(rows, dtype=torch.int32, device=device)
    return DeviceStateBatch('fixture', slots, gens, owners, mask, errors)


def _search_result(compact, observations, counts, chosen, requested=8):
    import torch
    device = compact.device
    rows, width = compact.batch_size, compact.max_candidates
    prefix = torch.arange(width, device=device)[None, :] < counts[:, None]
    pi = prefix.to(torch.float32)
    pi = pi / counts.clamp(min=1).to(torch.float32)[:, None]
    visit = (torch.arange(width, device=device)[None, :] == chosen[:, None]).to(torch.float32)
    visits = visit.to(torch.int32) * requested
    zeros = torch.zeros(rows, dtype=torch.int32, device=device)
    decision = SimpleNamespace(
        counts=counts.contiguous(), valid=prefix.contiguous(),
        observations=observations.contiguous(), candidates=compact, errors=zeros,
        states=None,
    )
    full = torch.full((rows,), requested, dtype=torch.int64, device=device)
    return SimpleNamespace(
        decision=decision, actions=chosen.contiguous(), policy=pi.contiguous(),
        visit_policy=visit.contiguous(), visits=visits.contiguous(),
        requested=full, completed=full.clone(), errors=zeros.clone(),
    )


class DeviceTrainingTapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.openings = _cases(skips=(False,))
        cls.skip_pairs = _cases(skips=(False, True))

    def _tape(self, cases, *, device='cpu', capacity=32, max_records=16,
              max_legal_entries=256, max_bytes=10 ** 9, scratch_rows=4):
        import torch
        tokens = _tokens(cases, device=device, capacity=capacity)
        compact = _compact(tokens, scratch_rows=scratch_rows, device=device)
        obs = _observations(cases, device=device)
        rules = ControllableRules(len(cases), compact.device, obs.shape[1])
        tape = DeviceTrainingTape(
            compact, rules, rows=len(cases), max_records=max_records,
            max_legal_entries=max_legal_entries, max_bytes=max_bytes,
            observation_dim=int(obs.shape[1]), device=str(compact.device),
            allow_cpu_reference=device == 'cpu', allow_synthetic=device == 'cpu',
        )
        roots = _roots(len(cases), compact.device)
        seeds = torch.arange(len(cases), dtype=torch.int64, device=compact.device)
        ids = torch.zeros((len(cases), 2), dtype=torch.int64, device=compact.device)
        train = torch.ones((len(cases), 2), dtype=torch.bool, device=compact.device)
        return tape, compact, rules, obs, roots, seeds, ids, train

    def test_constructor_requires_cuda_and_refuses_readiness_claims(self):
        tape, compact, rules, obs, roots, seeds, ids, train = self._tape(self.openings[:2])
        with self.assertRaises(ValueError):
            DeviceTrainingTape(compact, rules, rows=2, max_records=4, max_legal_entries=32,
                               max_bytes=10 ** 9, observation_dim=obs.shape[1], device='cpu')
        self.assertFalse(tape.supports_full_duel_rules)
        self.assertFalse(hasattr(tape, 'device_validation_receipt'))
        dense = tape.max_records * tape.max_actions * encoder.CAND_DIM * 4
        self.assertLess(tape.allocated_bytes, dense)
        self.assertGreater(tape.max_actions, 16)

    def test_cpu_seven_openings_pack_and_reconstruct_without_weights(self):
        import torch
        cases = self.openings
        self.assertEqual(len(cases), 7)
        tape, compact, rules, obs, roots, seeds, ids, train = self._tape(
            cases, capacity=32, max_records=8, max_legal_entries=200)
        counts = compact.tokens.counts.clone()
        self.assertTrue(bool((counts > 16).any()))
        chosen = torch.zeros(7, dtype=torch.int64)
        result = _search_result(compact, obs, counts, chosen)
        rules.actors.zero_()
        rules.value_obs.copy_(obs)
        active = torch.ones(7, dtype=torch.bool)
        begin = tape.begin_device(roots, ids, ids, seeds, train, active)
        append = tape.append_device(roots, rules.actors, torch.zeros(7, dtype=torch.int64),
                                    result, active)
        winners = torch.tensor([0, 1, 0, 1, -1, 0, 1], dtype=torch.int64)
        complete = torch.ones(7, dtype=torch.bool)
        finish = tape.finish_device(roots, winners, torch.ones(7, dtype=torch.int64),
                                    complete, torch.zeros(7, dtype=torch.int32))
        self.assertEqual(begin.tolist(), [0] * 7)
        self.assertEqual(append.tolist(), [0] * 7)
        self.assertEqual(finish.tolist(), [0] * 7)
        self.assertEqual(int(tape._record_count), 7)
        self.assertEqual(int(tape._legal_count), int(counts.sum()))
        offsets = tape._rec_offset[:7]
        packed = tape._packed['int_tokens']
        for i in range(7):
            start = int(offsets[i])
            n = int(counts[i])
            self.assertTrue(torch.equal(packed[start:start + n], compact.tokens.int_tokens[i, :n]))
            if i:
                self.assertEqual(start, int(counts[:i].sum()))
        encoded, errors = encoder.encode_tokens_device(
            compact.tokens, device='cpu', allow_cpu_reference=True,
            template_table=compact.template_table)
        self.assertEqual(errors.tolist(), [0] * 7)
        rows = tape.numerical_rows_after_batch()
        self.assertEqual(len(rows), 7)
        expected_z = [1, -1, 1, -1, 0, 1, -1]
        for i, row in enumerate(rows):
            self.assertTrue(row['learning_eligible'])
            self.assertEqual(row['selected'], 0)
            self.assertEqual(row['job_seed'], i)
            self.assertEqual(row['owner'], i)
            self.assertEqual(row['version'], i + 1)
            self.assertEqual(row['actor'], 0)
            self.assertEqual(row['caveat'], CAVEAT)
            self.assertNotIn('action', row)
            self.assertEqual(row['z'], expected_z[i])
            np.testing.assert_array_equal(row['x'], obs[i].numpy())
            np.testing.assert_array_equal(row['c'], encoded.candidates[i, :int(counts[i])].numpy())
            self.assertEqual(row['encode_error'], 0)
        self.assertEqual(tape.unfinished_records_after_batch(), [])

    def test_scatter_mask_sends_invalid_and_inactive_to_sink(self):
        import torch
        cases = self.openings[:3]
        tape, compact, rules, obs, roots, seeds, ids, train = self._tape(
            cases, max_records=4, max_legal_entries=80)
        counts = compact.tokens.counts.clone()
        chosen = torch.zeros(3, dtype=torch.int64)
        result = _search_result(compact, obs, counts, chosen)
        tape.begin_device(roots, ids, ids, seeds, train, torch.ones(3, dtype=torch.bool))
        result.policy[0].zero_()
        active = torch.tensor([True, False, True])
        errors = tape.append_device(roots, rules.actors, torch.zeros(3, dtype=torch.int64),
                                    result, active)
        self.assertTrue(int(errors[0]) & PI_ERROR)
        self.assertEqual(int(errors[1]), 0)
        self.assertEqual(int(errors[2]), 0)
        self.assertEqual(int(tape._record_count), 1)
        self.assertEqual(int(tape._rec_game[0]), 2)
        start = int(tape._rec_offset[0])
        n = int(counts[2])
        self.assertTrue(torch.equal(
            tape._packed['int_tokens'][start:start + n], compact.tokens.int_tokens[2, :n]))
        before = tape._packed['int_tokens'][:n].clone()
        inactive = torch.zeros(3, dtype=torch.bool)
        again = tape.append_device(roots, rules.actors, torch.zeros(3, dtype=torch.int64),
                                   result, inactive)
        self.assertEqual(again.tolist(), [0, 0, 0])
        self.assertEqual(int(tape._record_count), 1)
        self.assertTrue(torch.equal(tape._packed['int_tokens'][:n], before))

    def test_capacity_is_all_or_none_and_blocks_advancement(self):
        import torch
        cases = self.openings[:2]
        counts = _tokens(cases, capacity=32).counts
        need = int(counts.sum())
        tape, compact, rules, obs, roots, seeds, ids, train = self._tape(
            cases, max_records=2, max_legal_entries=need - 1)
        chosen = torch.zeros(2, dtype=torch.int64)
        result = _search_result(compact, obs, compact.tokens.counts.clone(), chosen)
        active = torch.ones(2, dtype=torch.bool)
        tape.begin_device(roots, ids, ids, seeds, train, active)
        packed_before = tape._packed['int_tokens'].clone()
        errors = tape.append_device(roots, rules.actors, torch.zeros(2, dtype=torch.int64),
                                    result, active)
        self.assertTrue(bool((errors & CAPACITY_ERROR).all()))
        self.assertEqual(int(tape._record_count), 0)
        self.assertEqual(int(tape._legal_count), 0)
        self.assertEqual(tape._expected_step.tolist(), [0, 0])
        self.assertTrue(torch.equal(tape._packed['int_tokens'][:-1], packed_before[:-1]))
        small, compact, rules, obs, roots, seeds, ids, train = self._tape(
            cases, max_records=1, max_legal_entries=need)
        small.begin_device(roots, ids, ids, seeds, train, active)
        overflow = small.append_device(roots, rules.actors, torch.zeros(2, dtype=torch.int64),
                                       _search_result(compact, obs, compact.tokens.counts.clone(), chosen),
                                       active)
        self.assertTrue(bool((overflow & CAPACITY_ERROR).all()))
        self.assertEqual(int(small._record_count), 0)

    def test_duplicate_and_out_of_order_steps_are_rejected(self):
        import torch
        cases = self.openings[:1]
        tape, compact, rules, obs, roots, seeds, ids, train = self._tape(
            cases, max_records=4, max_legal_entries=64)
        counts = compact.tokens.counts.clone()
        chosen = torch.zeros(1, dtype=torch.int64)
        result = _search_result(compact, obs, counts, chosen)
        active = torch.ones(1, dtype=torch.bool)
        tape.begin_device(roots, ids, ids, seeds, train, active)
        first = tape.append_device(roots, rules.actors, torch.zeros(1, dtype=torch.int64),
                                   result, active)
        dup = tape.append_device(roots, rules.actors, torch.zeros(1, dtype=torch.int64),
                                 result, active)
        skip = tape.append_device(roots, rules.actors, torch.full((1,), 2, dtype=torch.int64),
                                  result, active)
        nxt = tape.append_device(roots, rules.actors, torch.ones(1, dtype=torch.int64),
                                 result, active)
        self.assertEqual(int(first), 0)
        self.assertTrue(int(dup) & STEP_ERROR)
        self.assertTrue(int(skip) & STEP_ERROR)
        self.assertEqual(int(nxt), 0)
        self.assertEqual(int(tape._record_count), 2)
        self.assertEqual(tape._rec_step[:2].tolist(), [0, 1])

    def test_terminal_signs_both_sides_and_eligibility(self):
        import torch
        cases = self.openings[:4]
        tape, compact, rules, obs, roots, seeds, ids, train = self._tape(
            cases, max_records=8, max_legal_entries=160)
        counts = compact.tokens.counts.clone()
        chosen = torch.zeros(4, dtype=torch.int64)
        actors = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        rules.actors.copy_(actors)
        result = _search_result(compact, obs, counts, chosen)
        train[:, 1] = True
        active = torch.ones(4, dtype=torch.bool)
        tape.begin_device(roots, ids, ids, seeds, train, active)
        self.assertEqual(tape.append_device(roots, actors, torch.zeros(4, dtype=torch.int64),
                                            result, active).tolist(), [0, 0, 0, 0])
        winners = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        complete = torch.tensor([True, True, False, True])
        errors = torch.tensor([0, 0, 0, 7], dtype=torch.int32)
        finish_active = DeviceStateBatch(
            roots.pool, roots.slots, roots.generations, roots.owners,
            torch.tensor([True, True, True, True]), roots.errors)
        tape.finish_device(finish_active, winners, torch.ones(4, dtype=torch.int64),
                           complete, errors)
        rows = tape.numerical_rows_after_batch()
        self.assertEqual([row['actor'] for row in rows], [0, 1])
        self.assertEqual([row['z'] for row in rows], [1.0, -1.0])
        self.assertTrue(all(row['learning_eligible'] for row in rows))
        self.assertFalse(bool(tape._eligible[2]))
        self.assertFalse(bool(tape._eligible[3]))
        self.assertEqual(float(tape._z[2]), 0.0)
        self.assertEqual(float(tape._z[3]), 0.0)
        self.assertTrue(bool(tape._game_truncated[2]))
        unfinished = tape.unfinished_records_after_batch()
        self.assertEqual(unfinished, [])
        self.assertEqual(int(tape._record_count), 4)

    def test_truncated_and_error_zero_labels_are_not_learning_eligible(self):
        import torch
        cases = self.openings[:2]
        tape, compact, rules, obs, roots, seeds, ids, train = self._tape(cases)
        result = _search_result(compact, obs, compact.tokens.counts.clone(),
                                torch.zeros(2, dtype=torch.int64))
        active = torch.ones(2, dtype=torch.bool)
        tape.begin_device(roots, ids, ids, seeds, train, active)
        tape.append_device(roots, rules.actors, torch.zeros(2, dtype=torch.int64), result, active)
        complete = torch.zeros(2, dtype=torch.bool)
        errors = torch.tensor([0, 3], dtype=torch.int32)
        winners = torch.full((2,), -2, dtype=torch.int64)
        tape.finish_device(roots, winners, torch.ones(2, dtype=torch.int64), complete, errors)
        self.assertEqual(tape.numerical_rows_after_batch(), [])
        self.assertEqual(tape._z[:2].tolist(), [0.0, 0.0])
        self.assertEqual(tape.value_rows_after_batch(), [])
        self.assertEqual(tape._eligible[:2].tolist(), [False, False])
        self.assertTrue(bool(tape._game_truncated[0]))
        self.assertFalse(bool(tape._unfinished[:2].any()))

    def test_frozen_actor_retains_learning_viewer_opponent_turn_values(self):
        import torch
        tape, compact, rules, obs, roots, seeds, ids, train = self._tape(self.openings[:2])
        train[:, 0] = False
        rules.value_obs.copy_(obs + 3)
        result = _search_result(compact, obs, compact.tokens.counts,
                                torch.zeros(2, dtype=torch.int64))
        tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        tape.append_device(roots, rules.actors, torch.zeros(2, dtype=torch.int64), result, roots.active)
        self.assertEqual(tape.value_rows_after_batch(), [])
        tape.finish_device(roots, torch.tensor([0, -1]), torch.ones(2, dtype=torch.int64),
                           roots.active, torch.zeros(2, dtype=torch.int32))
        self.assertEqual(tape.numerical_rows_after_batch(), [])
        values = tape.value_rows_after_batch()
        self.assertEqual([row['actor'] for row in values], [1, 1])
        self.assertEqual([row['value_actor'] for row in values], [0, 0])
        self.assertEqual([row['z'] for row in values], [-1, 0])
        for i, row in enumerate(values):
            np.testing.assert_array_equal(row['x'], (obs + 3)[i].numpy())
            self.assertNotIn('pi', row)
        tape.finish_device(roots, torch.tensor([0, -1]), torch.ones(2, dtype=torch.int64),
                           roots.active, torch.zeros(2, dtype=torch.int32))
        self.assertEqual(tape.value_rows_after_batch(), [])

    def test_opponent_value_context_cannot_claim_to_be_the_actor(self):
        import torch
        tape, compact, rules, obs, roots, seeds, ids, train = self._tape(self.openings[:1])
        result = _search_result(compact, obs, compact.tokens.counts,
                                torch.zeros(1, dtype=torch.int64))
        tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        wrong = DeviceValueObservation(rules.value_obs, torch.ones((1, 1)), rules.terminal,
                                      rules.terminal_value, rules.value_errors)
        with patch.object(rules, 'value_observation_device', return_value=wrong):
            errors = tape.append_device(roots, rules.actors, torch.zeros(1, dtype=torch.int64),
                                        result, roots.active)
        self.assertNotEqual(int(errors[0]), 0)
        self.assertEqual(int(tape._record_count), 0)

    def test_byteguard_refuses_oversize_allocation(self):
        tokens = _tokens(self.openings[:1], capacity=32)
        compact = _compact(tokens, scratch_rows=1)
        rules = ControllableRules(1, compact.device, 8)
        with self.assertRaises(MemoryError):
            DeviceTrainingTape(
                compact, rules, rows=1, max_records=1, max_legal_entries=8,
                max_bytes=64, observation_dim=8, device='cpu',
                allow_cpu_reference=True, allow_synthetic=True,
            )

    def test_hot_path_has_no_host_tensor_reads(self):
        import torch
        cases = self.openings[:2]
        tape, compact, rules, obs, roots, seeds, ids, train = self._tape(cases)
        result = _search_result(compact, obs, compact.tokens.counts.clone(),
                                torch.zeros(2, dtype=torch.int64))
        active = torch.ones(2, dtype=torch.bool)
        steps = torch.zeros(2, dtype=torch.int64)
        winners = torch.zeros(2, dtype=torch.int64)
        complete = torch.ones(2, dtype=torch.bool)
        errors = torch.zeros(2, dtype=torch.int32)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU copy')), \
             patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy copy')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar copy')), \
             patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('Host list')), \
             patch.object(torch.Tensor, 'nonzero', side_effect=AssertionError('Dynamic index')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Host branch')):
            tape.begin_device(roots, ids, ids, seeds, train, active)
            tape.append_device(roots, rules.actors, steps, result, active)
            tape.finish_device(roots, winners, torch.ones(2, dtype=torch.int64), complete, errors)
            tape.result_device()

    def test_dense_decision_cannot_reuse_stale_template_tokens(self):
        import torch
        tape, compact, rules, obs, roots, seeds, ids, train = self._tape(self.openings[:2])
        tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        result = _search_result(compact, obs, compact.tokens.counts,
                                torch.zeros(2, dtype=torch.int64))
        result.decision.candidates = torch.zeros((2, 32, encoder.CAND_DIM))
        with self.assertRaisesRegex(ValueError, 'actual compact'):
            tape.append_device(roots, rules.actors, torch.zeros(2, dtype=torch.int64),
                               result, roots.active)
        self.assertEqual(int(tape._record_count), 0)

    def test_claimed_completed_budget_requires_matching_visits(self):
        import torch
        from app.modules.card_game.rl.batched_search.device_training_tape import BUDGET_ERROR
        tape, compact, rules, obs, roots, seeds, ids, train = self._tape(self.openings[:2])
        tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        result = _search_result(compact, obs, compact.tokens.counts,
                                torch.zeros(2, dtype=torch.int64))
        result.visits.div_(8, rounding_mode='floor')
        errors = tape.append_device(roots, rules.actors, torch.zeros(2, dtype=torch.int64),
                                    result, roots.active)
        self.assertTrue(bool(((errors & BUDGET_ERROR) != 0).all()))
        self.assertEqual(int(tape._record_count), 0)

    def test_terminal_step_mismatch_and_repeated_finish_do_not_relabel(self):
        import torch
        tape, compact, rules, obs, roots, seeds, ids, train = self._tape(self.openings[:2])
        tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        result = _search_result(compact, obs, compact.tokens.counts,
                                torch.zeros(2, dtype=torch.int64))
        tape.append_device(roots, rules.actors, torch.zeros(2, dtype=torch.int64), result, roots.active)
        error = tape.finish_device(roots, torch.zeros(2, dtype=torch.int64),
            torch.tensor([0, 1]), roots.active, torch.zeros(2, dtype=torch.int32))
        self.assertTrue(bool(error[0] & STEP_ERROR))
        self.assertEqual(tape.result_device().learning_eligible[:2].tolist(), [False, True])
        tape.finish_device(roots, torch.ones(2, dtype=torch.int64), torch.ones(2, dtype=torch.int64),
                           roots.active, torch.zeros(2, dtype=torch.int32))
        self.assertFalse(bool(tape.result_device().learning_eligible[:2].any()))

    def test_actual_cuda_graph_replay_packs_skip_aware_full_widths(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; training tape packing/masks only')
        wide = next(item for item in self.skip_pairs if len(item[1]) > 16)
        other = next(item for item in self.skip_pairs if item is not wide)
        cases = [wide, other]
        device = 'cuda'
        tape, compact, rules, obs, roots, seeds, ids, train = self._tape(
            cases, device=device, capacity=32, max_records=4, max_legal_entries=80, scratch_rows=2)
        self.assertGreater(tape.max_actions, 16)
        dense = tape.max_records * tape.max_actions * encoder.CAND_DIM * 4
        self.assertLess(tape.allocated_bytes, dense)
        counts = compact.tokens.counts.clone()
        chosen = torch.zeros(2, dtype=torch.int64, device=compact.device)
        result = _search_result(compact, obs, counts, chosen)
        active = torch.ones(2, dtype=torch.bool, device=compact.device)
        result.decision.states = roots
        steps = torch.zeros(2, dtype=torch.int64, device=compact.device)
        winners = torch.tensor([0, 1], dtype=torch.int64, device=compact.device)
        complete = torch.ones(2, dtype=torch.bool, device=compact.device)
        errors = torch.zeros(2, dtype=torch.int32, device=compact.device)
        rules.actors.zero_()
        tape.begin_device(roots, ids, ids, seeds, train, active)
        tape.append_device(roots, rules.actors, steps, result, active)
        tape.finish_device(roots, winners, torch.ones(2, dtype=torch.int64, device=compact.device),
                           complete, errors)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tape.begin_device(roots, ids, ids, seeds, train, active)
            tape.append_device(roots, rules.actors, steps, result, active)
            tape.finish_device(roots, winners, torch.ones(2, dtype=torch.int64, device=compact.device),
                               complete, errors)
        rules.versions.add_(9)
        compact.tokens.int_tokens[0, 0, 0] = compact.tokens.int_tokens[0, 1, 0]
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('Host loop predicate')), \
             patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Host root stepping')), \
             patch.object(torch.Tensor, 'nonzero', side_effect=AssertionError('Dynamic index')):
            graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(int(tape._record_count), 2)
        self.assertEqual(int(tape._legal_count), int(counts.sum()))
        self.assertEqual(tape._rec_version[:2].cpu().tolist(), [10, 11])
        start = int(tape._rec_offset[0])
        n0 = int(counts[0])
        self.assertTrue(torch.equal(
            tape._packed['int_tokens'][start:start + n0], compact.tokens.int_tokens[0, :n0]))
        rows = tape.numerical_rows_after_batch()
        self.assertEqual(len(rows), 2)
        self.assertEqual([row['z'] for row in rows], [1.0, -1.0])
        self.assertGreater(len(rows[0]['c']), 16)
        self.assertEqual(rows[0]['c'].shape[1], encoder.CAND_DIM)
        self.assertEqual(rows[0]['caveat'], CAVEAT)
        values = tape.value_rows_after_batch()
        self.assertEqual([row['actor'] for row in values], [0, 1, 0, 1])
        self.assertEqual([row['z'] for row in values], [1, -1, -1, 1])
        self.assertEqual([row['value_actor'] for row in values], [1, 0, 1, 0])


if __name__ == '__main__':
    unittest.main()
