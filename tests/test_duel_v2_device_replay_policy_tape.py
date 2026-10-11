"""CPU/CUDA tests for the light device replay-policy archive; not raw replay evidence."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
from app.modules.card_game.rl.batched_search.device_replay_policy_tape import (
    BUDGET_ERROR, CAPACITY_ERROR, CAVEAT, DeviceReplayPolicyTape, FINISH_ERROR,
    GEN_ERROR, OWNER_ERROR, PI_ERROR, STEP_ERROR, _layout_bytes,
)


def _roots(rows, device, *, gens=None, slots=None, owners=None, active=None, pool=None):
    import torch
    pool = object() if pool is None else pool
    def tensor(values, default, dtype):
        if values is None:
            return default
        return torch.tensor(values, dtype=dtype, device=device)
    slots = tensor(slots, torch.arange(rows, dtype=torch.int64, device=device), torch.int64)
    gens = tensor(gens, torch.zeros(rows, dtype=torch.int64, device=device), torch.int64)
    owners = tensor(owners, torch.arange(rows, dtype=torch.int64, device=device), torch.int64)
    mask = tensor(active, torch.ones(rows, dtype=torch.bool, device=device), torch.bool)
    errors = torch.zeros(rows, dtype=torch.int32, device=device)
    return DeviceStateBatch(pool, slots, gens, owners, mask, errors)


def _search(roots, counts, chosen, *, requested=8, max_actions=32, complete=True, errors=None):
    import torch
    device = counts.device
    rows = counts.shape[0]
    prefix = torch.arange(max_actions, device=device)[None, :] < counts[:, None]
    pi = prefix.to(torch.float32)
    pi = pi / counts.clamp(min=1).to(torch.float32)[:, None]
    visit = (torch.arange(max_actions, device=device)[None, :] == chosen[:, None]).to(torch.float32)
    visits = visit.to(torch.int32) * int(requested)
    zeros = torch.zeros(rows, dtype=torch.int32, device=device)
    if errors is None:
        errors = zeros
    decision = SimpleNamespace(
        counts=counts.contiguous(), valid=prefix.contiguous(), errors=zeros.clone(),
        states=roots,
    )
    full = torch.full((rows,), int(requested), dtype=torch.int64, device=device)
    if type(complete) is bool:
        complete = torch.full((rows,), complete, dtype=torch.bool, device=device)
    return SimpleNamespace(
        decision=decision, actions=chosen.contiguous(), policy=pi.contiguous(),
        visit_policy=visit.contiguous(), visits=visits.contiguous(),
        requested=full, completed=full.clone(), complete=complete.contiguous(),
        errors=errors.clone(),
    )


class DeviceReplayPolicyTapeTest(unittest.TestCase):
    def _tape(self, rows=2, max_records=8, max_legal_entries=128, max_actions=32,
              device='cpu', max_bytes=10 ** 9, allow_synthetic=True):
        return DeviceReplayPolicyTape(
            rows, max_records, max_legal_entries, max_actions, max_bytes,
            device=device, allow_cpu_reference=device == 'cpu',
            allow_synthetic=allow_synthetic,
        )

    def _ids(self, tape, rows=None):
        import torch
        rows = tape.rows if rows is None else rows
        seeds = torch.arange(rows, dtype=torch.int64, device=tape.device)
        ids = torch.arange(2, dtype=torch.int64, device=tape.device).repeat(rows, 1)
        train = torch.ones((rows, 2), dtype=torch.bool, device=tape.device)
        keys = torch.arange(rows * 8, dtype=torch.int64, device=tape.device).reshape(rows, 8)
        return seeds, ids, train, keys

    def test_constructor_requires_cuda_and_refuses_readiness_claims(self):
        tape = self._tape()
        with self.assertRaises(ValueError):
            DeviceReplayPolicyTape(2, 4, 32, 32, 10 ** 9, device='cpu')
        with self.assertRaises(ValueError):
            DeviceReplayPolicyTape(2, 4, 32, 16385, 10 ** 9, device='cpu',
                                   allow_cpu_reference=True, allow_synthetic=True)
        with self.assertRaises(ValueError):
            DeviceReplayPolicyTape(2, 2 ** 31, 32, 32, 10 ** 9, device='cpu',
                                   allow_cpu_reference=True, allow_synthetic=True)
        with self.assertRaises(MemoryError):
            DeviceReplayPolicyTape(2, 4, 32, 32, 64, device='cpu',
                                   allow_cpu_reference=True, allow_synthetic=True)
        self.assertFalse(tape.supports_full_duel_rules)
        self.assertFalse(hasattr(tape, 'device_validation_receipt'))
        self.assertFalse(hasattr(tape, 'learning_ready'))
        self.assertGreater(tape.max_actions, 16)
        self.assertEqual(
            tape.allocated_bytes,
            _layout_bytes(tape.rows, tape.max_records + 1, tape.max_legal_entries + 1,
                          tape.max_actions))
        self.assertEqual(
            tape.allocated_bytes,
            sum(v.numel() * v.element_size() for v in tape._persistent_tensors()))
        self.assertIn('DeviceRawActionTape', CAVEAT)
        self.assertIn('immutable initial-state collector', CAVEAT)
        self.assertIn('do not prove exact physical raw replay', CAVEAT)

    def test_generation_zero_and_allocated_slot_are_valid(self):
        import torch
        tape = self._tape(rows=2)
        roots = _roots(2, tape.device, gens=[0, 0], slots=[0, 3])
        seeds, ids, train, keys = self._ids(tape)
        begin = tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        self.assertEqual(begin.tolist(), [0, 0])
        counts = torch.tensor([3, 4], dtype=torch.int64)
        chosen = torch.zeros(2, dtype=torch.int64)
        result = _search(roots, counts, chosen, max_actions=32)
        versions = torch.zeros(2, dtype=torch.int64)
        append = tape.append_device(roots, torch.zeros(2, dtype=torch.int64),
                                    versions, versions, result, keys, roots.active)
        self.assertEqual(append.tolist(), [0, 0])
        bad = self._tape(rows=2)
        missing = _roots(2, bad.device, slots=[-1, 1], owners=[-1, 1], gens=[-1, 0])
        seeds, ids, train, keys = self._ids(bad)
        denied = bad.begin_device(missing, ids, ids, seeds, train, missing.active)
        self.assertTrue(int(denied[0]) & OWNER_ERROR)
        self.assertEqual(int(denied[1]), 0)

    def test_full_legal_width_above_sixteen_exports_metadata_pi_only(self):
        import torch
        tape = self._tape(rows=2, max_actions=32, max_legal_entries=64)
        roots = _roots(2, tape.device)
        seeds, ids, train, keys = self._ids(tape)
        counts = torch.tensor([20, 5], dtype=torch.int64)
        chosen = torch.tensor([19, 0], dtype=torch.int64)
        result = _search(roots, counts, chosen, max_actions=32)
        tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        errors = tape.append_device(
            roots, torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64),
            torch.ones(2, dtype=torch.int64), result, keys, roots.active)
        self.assertEqual(errors.tolist(), [0, 0])
        self.assertEqual(int(tape._record_count), 2)
        self.assertEqual(int(tape._legal_count), 25)
        start = int(tape._rec_offset[0])
        self.assertTrue(all(abs(v - 0.05) < 1e-6 for v in tape._packed_pi[start:start + 20].tolist()))
        winners = torch.tensor([0, 1], dtype=torch.int64)
        tape.finish_device(roots, winners, torch.ones(2, dtype=torch.int64),
                           roots.active, torch.zeros(2, dtype=torch.int32))
        rows = tape.records_after_batch()
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]['pi']), 20)
        self.assertEqual(int(rows[0]['selected']), 19)
        self.assertEqual(rows[0]['replay_verified'], False)
        self.assertEqual(rows[0]['learning_ready'], False)
        self.assertNotIn('x', rows[0])
        self.assertNotIn('c', rows[0])
        self.assertEqual(rows[0]['caveat'], CAVEAT)
        self.assertEqual([row['z'] for row in rows], [1.0, -1.0])
        self.assertEqual([row['learning_scope'] for row in rows], [True, True])

    def test_capacity_is_all_or_none_before_allocation(self):
        import torch
        counts = torch.tensor([6, 7], dtype=torch.int64)
        need = int(counts.sum())
        tape = self._tape(rows=2, max_records=2, max_legal_entries=need - 1)
        roots = _roots(2, tape.device)
        seeds, ids, train, keys = self._ids(tape)
        chosen = torch.zeros(2, dtype=torch.int64)
        result = _search(roots, counts, chosen)
        tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        packed = tape._packed_pi.clone()
        errors = tape.append_device(
            roots, torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64), result, keys, roots.active)
        self.assertTrue(bool((errors & CAPACITY_ERROR).all()))
        self.assertEqual(int(tape._record_count), 0)
        self.assertEqual(int(tape._legal_count), 0)
        self.assertEqual(tape._expected_step.tolist(), [0, 0])
        self.assertTrue(torch.equal(tape._packed_pi[:-1], packed[:-1]))
        small = self._tape(rows=2, max_records=1, max_legal_entries=need)
        small.begin_device(roots, ids, ids, seeds, train, roots.active)
        overflow = small.append_device(
            roots, torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64), result, keys, roots.active)
        self.assertTrue(bool((overflow & CAPACITY_ERROR).all()))
        self.assertEqual(int(small._record_count), 0)

    def test_step_owner_and_source_failures_reject(self):
        import torch
        tape = self._tape(rows=2)
        roots = _roots(2, tape.device)
        seeds, ids, train, keys = self._ids(tape)
        counts = torch.tensor([3, 3], dtype=torch.int64)
        chosen = torch.zeros(2, dtype=torch.int64)
        result = _search(roots, counts, chosen)
        actors = torch.zeros(2, dtype=torch.int64)
        steps = torch.zeros(2, dtype=torch.int64)
        versions = torch.ones(2, dtype=torch.int64)
        tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        first = tape.append_device(roots, actors, steps, versions, result, keys, roots.active)
        self.assertEqual(first.tolist(), [0, 0])
        dup = tape.append_device(roots, actors, steps, versions, result, keys, roots.active)
        skip = tape.append_device(
            roots, actors, torch.full((2,), 2, dtype=torch.int64), versions, result, keys,
            roots.active)
        nxt = tape.append_device(
            roots, actors, torch.ones(2, dtype=torch.int64), versions, result, keys,
            roots.active)
        self.assertTrue(bool((dup & STEP_ERROR).all()))
        self.assertTrue(bool((skip & STEP_ERROR).all()))
        self.assertTrue(bool((nxt & STEP_ERROR).all()))
        self.assertEqual(int(tape._record_count), 2)
        self.assertEqual(tape._rec_step[:2].tolist(), [0, 0])
        foreign = _roots(2, tape.device, owners=[9, 1], pool=roots.pool)
        owner_err = tape.append_device(
            foreign, actors, torch.full((2,), 2, dtype=torch.int64), versions, result, keys,
            roots.active)
        self.assertTrue(int(owner_err[0]) & OWNER_ERROR)
        result.complete[0] = False
        result.errors[1] = 7
        source = tape.append_device(
            roots, actors, torch.full((2,), 2, dtype=torch.int64), versions, result, keys,
            roots.active)
        self.assertTrue(int(source[0]) & GEN_ERROR)
        self.assertTrue(int(source[1]) & GEN_ERROR)
        result.complete.fill_(True)
        result.errors.zero_()
        result.decision.states = _roots(2, tape.device, slots=[0, 8], pool=roots.pool)
        lease = tape.append_device(
            roots, actors, torch.full((2,), 2, dtype=torch.int64), versions, result, keys,
            roots.active)
        self.assertTrue(int(lease[1]) & GEN_ERROR)
        result.decision.states = _roots(2, tape.device)
        with self.assertRaisesRegex(ValueError, 'arenas must match'):
            tape.append_device(
                roots, actors, torch.full((2,), 2, dtype=torch.int64), versions, result, keys,
                roots.active)
        result.decision.states = roots
        actors[0] = 2
        versions[1] = -1
        bad = tape.append_device(
            roots, actors, torch.full((2,), 2, dtype=torch.int64), versions, result, keys,
            roots.active)
        self.assertTrue(int(bad[0]) & GEN_ERROR)
        self.assertTrue(int(bad[1]) & GEN_ERROR)

    def test_failed_append_cannot_recover_terminal_labels(self):
        import torch
        tape = self._tape(rows=1)
        roots = _roots(1, tape.device)
        seeds, ids, train, keys = self._ids(tape)
        zero = torch.zeros(1, dtype=torch.int64)
        result = _search(roots, torch.tensor([20]), zero)
        tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        tape.append_device(roots, zero, zero, zero, result, keys, roots.active)
        duplicate = tape.append_device(roots, zero, zero, zero, result, keys, roots.active)
        self.assertTrue(int(duplicate[0]) & STEP_ERROR)
        next_attempt = tape.append_device(roots, zero, zero + 1, zero + 1,
                                          result, keys, roots.active)
        self.assertTrue(int(next_attempt[0]) & STEP_ERROR)
        tape.finish_device(roots, zero, zero + 1, roots.active, roots.errors)
        self.assertFalse(bool(tape.result_device().normal_complete.any()))
        self.assertFalse(bool(tape.result_device().eligible.any()))
        self.assertEqual(tape.records_after_batch(), [])

    def test_stale_version_and_append_after_terminal_revoke_labels(self):
        import torch
        for after_terminal in (False, True):
            tape = self._tape(rows=1)
            roots = _roots(1, tape.device)
            seeds, ids, train, keys = self._ids(tape)
            zero = torch.zeros(1, dtype=torch.int64)
            result = _search(roots, torch.tensor([20]), zero)
            tape.begin_device(roots, ids, ids, seeds, train, roots.active)
            tape.append_device(roots, zero, zero, zero + 4, result, keys, roots.active)
            if after_terminal:
                tape.finish_device(roots, zero, zero + 1, roots.active, roots.errors)
                self.assertEqual(len(tape.records_after_batch()), 1)
            flags = tape.append_device(roots, zero, zero + 1, zero + 4,
                                       result, keys, roots.active)
            self.assertTrue(int(flags[0]) & STEP_ERROR)
            if after_terminal:
                self.assertTrue(int(flags[0]) & FINISH_ERROR)
            else:
                tape.finish_device(roots, zero, zero + 1, roots.active, roots.errors)
            self.assertFalse(bool(tape.result_device().eligible.any()))
            self.assertFalse(bool(tape.result_device().normal_complete.any()))
            self.assertEqual(tape.records_after_batch(), [])

    def test_bad_initial_state_cannot_enter_a_later_valid_lease(self):
        import torch
        tape = self._tape(rows=1)
        roots = _roots(1, tape.device)
        roots.errors.fill_(107)
        seeds, ids, train, keys = self._ids(tape)
        zero = torch.zeros(1, dtype=torch.int64)
        tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        roots.errors.zero_()
        result = _search(roots, torch.tensor([20]), zero)
        flags = tape.append_device(roots, zero, zero, zero, result, keys, roots.active)
        self.assertNotEqual(int(flags[0]), 0)
        tape.finish_device(roots, zero, zero, roots.active, roots.errors)
        self.assertEqual(tape.records_after_batch(), [])

    def test_pure_budget_zero_is_rejected(self):
        import torch
        tape = self._tape(rows=1, max_actions=8)
        roots = _roots(1, tape.device)
        seeds, ids, train, keys = self._ids(tape)
        counts = torch.tensor([3], dtype=torch.int64)
        chosen = torch.zeros(1, dtype=torch.int64)
        result = _search(roots, counts, chosen, requested=0, max_actions=8)
        result.visit_policy.zero_()
        result.visits.zero_()
        tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        errors = tape.append_device(
            roots, torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64), result, keys, roots.active)
        self.assertTrue(int(errors[0]) & BUDGET_ERROR)
        self.assertEqual(int(tape._record_count), 0)

    def test_finish_label_signs_separate_learning_scope_from_audit(self):
        import torch
        tape = self._tape(rows=4, max_legal_entries=64)
        roots = _roots(4, tape.device)
        seeds, ids, train, keys = self._ids(tape)
        train[:, 1] = False
        actors = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        counts = torch.tensor([2, 2, 2, 2], dtype=torch.int64)
        chosen = torch.zeros(4, dtype=torch.int64)
        result = _search(roots, counts, chosen, max_actions=32)
        tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        self.assertEqual(
            tape.append_device(
                roots, actors, torch.zeros(4, dtype=torch.int64),
                torch.ones(4, dtype=torch.int64), result, keys, roots.active).tolist(),
            [0, 0, 0, 0])
        winners = torch.tensor([0, 0, 1, -1], dtype=torch.int64)
        complete = torch.tensor([True, True, False, True])
        finish = tape.finish_device(
            roots, winners, torch.ones(4, dtype=torch.int64), complete,
            torch.zeros(4, dtype=torch.int32))
        self.assertEqual(finish.tolist(), [0, 0, 0, 0])
        public = tape.result_device()
        self.assertEqual(public.normal_complete.tolist(), [True, True, False, True])
        self.assertEqual(public.eligible[:4].tolist(), [True, False, False, False])
        self.assertTrue(bool(public.truncated[2]))
        self.assertEqual(public.z[:4].tolist(), [1.0, -1.0, 0.0, 0.0])
        rows = tape.records_after_batch()
        self.assertEqual([row['actor'] for row in rows], [0, 1, 1])
        self.assertEqual([row['z'] for row in rows], [1.0, -1.0, 0.0])
        self.assertEqual([row['learning_scope'] for row in rows], [True, False, False])
        self.assertEqual([row['learning_eligible'] for row in rows], [True, False, False])
        self.assertTrue(all(row['replay_verified'] is False for row in rows))
        self.assertTrue(all(row['learning_ready'] is False for row in rows))
        self.assertNotIn('x', rows[0])
        self.assertNotIn('c', rows[0])

    def test_repeated_finish_permanently_revokes_eligibility(self):
        import torch
        tape = self._tape(rows=2)
        roots = _roots(2, tape.device)
        seeds, ids, train, keys = self._ids(tape)
        counts = torch.tensor([2, 2], dtype=torch.int64)
        result = _search(roots, counts, torch.zeros(2, dtype=torch.int64))
        tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        tape.append_device(
            roots, torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64),
            torch.ones(2, dtype=torch.int64), result, keys, roots.active)
        error = tape.finish_device(
            roots, torch.zeros(2, dtype=torch.int64), torch.tensor([0, 1]),
            roots.active, torch.zeros(2, dtype=torch.int32))
        self.assertTrue(int(error[0]) & STEP_ERROR)
        self.assertEqual(tape.result_device().eligible[:2].tolist(), [False, True])
        tape.finish_device(
            roots, torch.ones(2, dtype=torch.int64), torch.ones(2, dtype=torch.int64),
            roots.active, torch.zeros(2, dtype=torch.int32))
        self.assertFalse(bool(tape.result_device().eligible[:2].any()))
        self.assertTrue(int(tape.result_device().errors[1]) & FINISH_ERROR)

    def test_dummy_slot_does_not_overwrite_committed_records(self):
        import torch
        tape = self._tape(rows=3, max_legal_entries=64)
        roots = _roots(3, tape.device)
        seeds, ids, train, keys = self._ids(tape)
        counts = torch.tensor([4, 4, 5], dtype=torch.int64)
        chosen = torch.zeros(3, dtype=torch.int64)
        result = _search(roots, counts, chosen)
        result.policy[0].zero_()
        tape.begin_device(roots, ids, ids, seeds, train, roots.active)
        active = torch.tensor([True, False, True])
        errors = tape.append_device(
            roots, torch.zeros(3, dtype=torch.int64), torch.zeros(3, dtype=torch.int64),
            torch.ones(3, dtype=torch.int64), result, keys, active)
        self.assertTrue(int(errors[0]) & PI_ERROR)
        self.assertEqual(int(errors[1]), 0)
        self.assertEqual(int(errors[2]), 0)
        self.assertEqual(int(tape._record_count), 1)
        self.assertEqual(int(tape._rec_game[0]), 2)
        start = int(tape._rec_offset[0])
        n = int(counts[2])
        self.assertTrue(all(abs(v - 0.2) < 1e-6 for v in tape._packed_pi[start:start + n].tolist()))
        before = tape._packed_pi[:n].clone()
        again = tape.append_device(
            roots, torch.zeros(3, dtype=torch.int64), torch.zeros(3, dtype=torch.int64),
            torch.ones(3, dtype=torch.int64), result, keys, torch.zeros(3, dtype=torch.bool))
        self.assertEqual(again.tolist(), [0, 0, 0])
        self.assertEqual(int(tape._record_count), 1)
        self.assertTrue(torch.equal(tape._packed_pi[:n], before))
        self.assertEqual(int(tape._rec_game[0]), 2)
        self.assertEqual(int(tape._rec_chosen[0]), 0)

    def test_hot_path_has_no_host_tensor_reads(self):
        import torch
        tape = self._tape(rows=2)
        roots = _roots(2, tape.device)
        seeds, ids, train, keys = self._ids(tape)
        counts = torch.tensor([3, 3], dtype=torch.int64)
        result = _search(roots, counts, torch.zeros(2, dtype=torch.int64))
        actors = torch.zeros(2, dtype=torch.int64)
        steps = torch.zeros(2, dtype=torch.int64)
        versions = torch.ones(2, dtype=torch.int64)
        winners = torch.zeros(2, dtype=torch.int64)
        complete = torch.ones(2, dtype=torch.bool)
        errors = torch.zeros(2, dtype=torch.int32)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU copy')), \
             patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy copy')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar copy')), \
             patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('Host list')), \
             patch.object(torch.Tensor, 'nonzero', side_effect=AssertionError('Dynamic index')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Host branch')):
            tape.begin_device(roots, ids, ids, seeds, train, roots.active)
            tape.append_device(roots, actors, steps, versions, result, keys, roots.active)
            tape.finish_device(roots, winners, torch.ones(2, dtype=torch.int64), complete, errors)
            tape.result_device()

    def test_actual_cuda_capture_append_finish_replay_with_synthetic_policy(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; synthetic replay-policy archive only')
        tape = self._tape(rows=2, max_actions=24, max_legal_entries=48, device='cuda',
                          allow_synthetic=False)
        roots = _roots(2, tape.device, gens=[0, 0])
        seeds, ids, train, keys = self._ids(tape)
        counts = torch.tensor([20, 4], dtype=torch.int64, device=tape.device)
        chosen = torch.tensor([3, 1], dtype=torch.int64, device=tape.device)
        result = _search(roots, counts, chosen, max_actions=24)
        actors = torch.zeros(2, dtype=torch.int64, device=tape.device)
        steps = torch.zeros(2, dtype=torch.int64, device=tape.device)
        versions = torch.ones(2, dtype=torch.int64, device=tape.device)
        winners = torch.tensor([0, 1], dtype=torch.int64, device=tape.device)
        end_steps = torch.ones(2, dtype=torch.int64, device=tape.device)
        errors = torch.zeros(2, dtype=torch.int32, device=tape.device)
        def record():
            tape.begin_device(roots, ids, ids, seeds, train, roots.active)
            tape.append_device(roots, actors, steps, versions, result, keys, roots.active)
            tape.finish_device(roots, winners, end_steps, roots.active, errors)
        record()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            record()
        versions.add_(9)
        keys.add_(1)
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('Host loop predicate')), \
             patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Host root stepping')), \
             patch.object(torch.Tensor, 'nonzero', side_effect=AssertionError('Dynamic index')):
            graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(int(tape._record_count), 2)
        self.assertEqual(int(tape._legal_count), 24)
        self.assertEqual(tape._rec_version[:2].cpu().tolist(), [10, 10])
        self.assertGreater(int(tape._rec_count[0]), 16)
        rows = tape.records_after_batch()
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]['pi']), 20)
        self.assertEqual(rows[0]['replay_verified'], False)
        self.assertEqual(rows[0]['learning_ready'], False)
        self.assertNotIn('x', rows[0])
        self.assertNotIn('c', rows[0])
        self.assertEqual([row['z'] for row in rows], [1.0, -1.0])
        self.assertEqual(rows[0]['information_keys'].tolist()[0], int(keys[0, 0].cpu()))

    def test_actual_cuda_failed_append_permanently_revokes_labels(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; synthetic policy archive only')
        tape = self._tape(rows=2, device='cuda', allow_synthetic=False)
        roots = _roots(2, tape.device)
        seeds, ids, train, keys = self._ids(tape)
        zero = torch.zeros(2, dtype=torch.int64, device=tape.device)
        result = _search(roots, torch.tensor([20, 20], device=tape.device), zero)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tape.begin_device(roots, ids, ids, seeds, train, roots.active)
            tape.append_device(roots, zero, zero, zero, result, keys, roots.active)
            tape.append_device(roots, zero, zero, zero, result, keys, roots.active)
            tape.finish_device(roots, zero, zero + 1, roots.active, roots.errors)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(bool((tape.result_device().errors & STEP_ERROR).all()))
        self.assertFalse(bool(tape.result_device().normal_complete.any()))
        self.assertFalse(bool(tape.result_device().eligible.any()))
        self.assertEqual(tape.records_after_batch(), [])


if __name__ == '__main__':
    unittest.main()




