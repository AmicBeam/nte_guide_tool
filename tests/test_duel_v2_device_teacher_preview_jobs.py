"""Opt-in GPU battle-preview worklist: compact, capacity, scatter alignment."""
import unittest
from unittest.mock import patch

from app.modules.card_game.rl.batched_search.device_teacher_preview_jobs import (
    CLOSED_SERVICE_READY, DeviceTeacherPreviewJobs, ERR_CAPACITY, ERR_DUPLICATE,
    ERR_FOREIGN, ERR_MALFORMED, ERR_OUTSIDE, ERR_STALE, OPT_IN, QUALIFIED)


def _cpu(rows=4, max_actions=5, capacity=8):
    return DeviceTeacherPreviewJobs(
        rows=rows, max_actions=max_actions, capacity=capacity, device='cpu',
        allow_cpu_reference=True)


def _batch(engine):
    import torch as t
    b, m, d = engine.rows, engine.max_actions, engine.device
    valid = t.zeros((b, m), dtype=t.bool, device=d)
    battle = t.zeros((b, m), dtype=t.bool, device=d)
    counts = t.zeros(b, dtype=t.int64, device=d)
    active = t.zeros(b, dtype=t.bool, device=d)
    input_errors = t.zeros(b, dtype=t.int32, device=d)
    slots = t.full((b,), -1, dtype=t.int64, device=d)
    generations = t.zeros(b, dtype=t.int64, device=d)
    owners = t.full((b,), -1, dtype=t.int64, device=d)
    actors = t.zeros(b, dtype=t.int64, device=d)
    return (valid, battle, counts, active, input_errors, slots, generations, owners, actors)


def _row(batch, row, n, battle_cols, slot, generation, owner, actor=0):
    valid, battle, counts, active, input_errors, slots, generations, owners, actors = batch
    counts[row] = n
    active[row] = True
    valid[row, :n] = True
    for col in battle_cols:
        battle[row, col] = True
    slots[row] = slot
    generations[row] = generation
    owners[row] = owner
    actors[row] = actor


class DeviceTeacherPreviewJobsTest(unittest.TestCase):
    def test_unqualified_opt_in_flags(self):
        self.assertFalse(QUALIFIED)
        self.assertFalse(CLOSED_SERVICE_READY)
        self.assertTrue(OPT_IN)
        engine = _cpu()
        self.assertFalse(engine.qualified)
        self.assertFalse(engine.closed_service_ready)
        self.assertTrue(engine.opt_in)

    def test_cpu_reference_must_be_explicit(self):
        with self.assertRaises(RuntimeError):
            DeviceTeacherPreviewJobs(rows=2, max_actions=2, capacity=2, device='cpu')

    def test_cpu_ragged_tail_poisoned_padding_and_stable_order(self):
        engine = _cpu(rows=4, max_actions=5, capacity=8)
        batch = _batch(engine)
        valid, battle, counts, active, input_errors, slots, generations, owners, actors = batch
        _row(batch, 0, 3, [0, 2], slot=10, generation=1, owner=100)
        _row(batch, 1, 1, [], slot=11, generation=1, owner=101)
        # Inactive poisoned padding must not emit jobs or error.
        valid[2, 0] = True
        battle[2, 0] = True
        counts[2] = 4
        slots[2] = 99
        owners[2] = 7
        _row(batch, 3, 4, [3], slot=12, generation=2, owner=103, actor=1)
        before = valid.clone(), battle.clone(), slots.clone()
        work = engine.compact_device(*batch)
        self.assertFalse(work.qualified)
        self.assertEqual(int(work.needed[0]), 3)
        self.assertEqual(int(work.job_count[0]), 3)
        self.assertEqual(int(work.overflow[0]), 0)
        self.assertEqual(work.parent_row[:3].tolist(), [0, 0, 3])
        self.assertEqual(work.ordinal[:3].tolist(), [0, 2, 3])
        self.assertEqual(work.owners[:3].tolist(), [100, 100, 103])
        self.assertEqual(work.actors[:3].tolist(), [0, 0, 1])
        self.assertEqual(work.row_workload.tolist(), [2, 0, 0, 1])
        self.assertEqual(work.errors.tolist(), [0, 0, 0, 0])
        self.assertTrue(t_equal(valid, before[0]))
        self.assertTrue(t_equal(battle, before[1]))
        self.assertTrue(t_equal(slots, before[2]))
        first_epoch = int(work.epoch[0])
        again = engine.compact_device(*batch)
        self.assertEqual(again.parent_row[:3].tolist(), [0, 0, 3])
        self.assertEqual(int(again.epoch[0]), first_epoch + 1)


def t_equal(left, right):
    import torch as t
    return bool(t.equal(left, right))


class DeviceTeacherPreviewCapacityTest(unittest.TestCase):
    def test_capacity_overflow_rejects_whole_eligible_workload(self):
        engine = _cpu(rows=3, max_actions=4, capacity=2)
        batch = _batch(engine)
        _row(batch, 0, 2, [0, 1], slot=1, generation=1, owner=10)
        _row(batch, 1, 2, [0], slot=2, generation=1, owner=11)
        work = engine.compact_device(*batch)
        self.assertEqual(int(work.needed[0]), 3)
        self.assertEqual(int(work.job_count[0]), 0)
        self.assertEqual(int(work.overflow[0]), 1)
        self.assertEqual(int(work.errors[0]), ERR_CAPACITY)
        self.assertEqual(int(work.errors[1]), ERR_CAPACITY)
        self.assertEqual(int(work.errors[2]), 0)
        self.assertEqual(work.parent_row.tolist(), [-1, -1])

    def test_declaration_shapes_without_throughput_claim(self):
        for rows in (8, 32, 128, 400, 1600):
            engine = _cpu(rows=rows, max_actions=4, capacity=8)
            batch = _batch(engine)
            _row(batch, 0, 2, [0], slot=1, generation=1, owner=1)
            _row(batch, 1, 3, [1, 2], slot=2, generation=1, owner=2)
            work = engine.compact_device(*batch)
            self.assertEqual(int(work.job_count[0]), 3)
            self.assertEqual(work.parent_row[:3].tolist(), [0, 1, 1])
            self.assertEqual(work.ordinal[:3].tolist(), [0, 1, 2])
            self.assertFalse(work.qualified)
            engine.close()


def _dest(engine):
    import torch as t
    b, m, d = engine.rows, engine.max_actions, engine.device
    return (
        t.full((b, m), -1.0, dtype=t.float64, device=d),
        t.full((b, m), -1.0, dtype=t.float64, device=d),
        t.full((b, m), -1.0, dtype=t.float64, device=d),
        t.full((b, m), -1.0, dtype=t.float64, device=d),
        t.zeros(b, dtype=t.int32, device=d),
    )


def _scatter(engine, job_ids, *args, epochs=None, actors=None):
    import torch as t
    if epochs is None:
        epochs = t.ones_like(job_ids) * engine.epoch[0]
    if actors is None:
        actors = engine.actors[job_ids.clamp(0, engine.capacity - 1)]
    return engine.scatter_previews_device(job_ids, *args, epochs=epochs, actors=actors)


class DeviceTeacherPreviewScatterTest(unittest.TestCase):
    def test_scatter_rejects_duplicate_outside_stale_foreign_and_isolates_roots(self):
        import torch as t
        engine = _cpu(rows=2, max_actions=3, capacity=4)
        batch = _batch(engine)
        _row(batch, 0, 2, [0], slot=5, generation=3, owner=20)
        _row(batch, 1, 2, [1], slot=6, generation=4, owner=21)
        work = engine.compact_device(*batch)
        self.assertEqual(int(work.job_count[0]), 2)
        engine.compact_device(*batch)
        dest = _dest(engine)
        jobs = 2
        job_ids = t.tensor([0, 1], dtype=t.int64)
        parents = t.tensor([0, 1], dtype=t.int64)
        ordinals = t.tensor([0, 1], dtype=t.int64)
        owners = t.tensor([20, 21], dtype=t.int64)
        generations = t.tensor([3, 4], dtype=t.int64)
        slots = t.tensor([5, 6], dtype=t.int64)
        active = t.ones(jobs, dtype=t.bool)
        attack = t.tensor([1.5, 2.5], dtype=t.float64)
        attack_count = t.tensor([1.0, 2.0], dtype=t.float64)
        genesis = t.tensor([0.0, 4.0], dtype=t.float64)
        counter = t.tensor([0.5, 0.0], dtype=t.float64)
        errors = _scatter(engine,
            job_ids, parents, ordinals, owners, generations, slots, active,
            attack, attack_count, genesis, counter, *dest)
        self.assertEqual(errors.tolist(), [0, 0])
        self.assertEqual(float(dest[0][0, 0]), 1.5)
        self.assertEqual(float(dest[1][0, 0]), 1.0)
        self.assertEqual(float(dest[2][1, 1]), 4.0)
        self.assertEqual(float(dest[0][1, 0]), -1.0)
        self.assertEqual(float(dest[0][0, 1]), -1.0)
        # Duplicate job 0.
        engine.compact_device(*batch)
        dest = _dest(engine)
        dup_ids = t.tensor([0, 0], dtype=t.int64)
        dup_parents = t.tensor([0, 0], dtype=t.int64)
        dup_ord = t.tensor([0, 0], dtype=t.int64)
        dup_own = t.tensor([20, 20], dtype=t.int64)
        dup_gen = t.tensor([3, 3], dtype=t.int64)
        dup_slot = t.tensor([5, 5], dtype=t.int64)
        errors = _scatter(engine,
            dup_ids, dup_parents, dup_ord, dup_own, dup_gen, dup_slot, active,
            attack, attack_count, genesis, counter, *dest)
        self.assertEqual(int(errors[0]), ERR_DUPLICATE)
        self.assertEqual(int(errors[1]), ERR_DUPLICATE)
        self.assertEqual(float(dest[0][0, 0]), -1.0)
        # Outside / foreign / stale.
        engine.compact_device(*batch)
        dest = _dest(engine)
        bad_ids = t.tensor([9, 1], dtype=t.int64)
        errors = _scatter(engine,
            bad_ids, parents, ordinals, owners, generations, slots, active,
            attack, attack_count, genesis, counter, *dest)
        self.assertEqual(int(errors[0]), ERR_OUTSIDE)
        foreign_own = t.tensor([20, 99], dtype=t.int64)
        engine.compact_device(*batch)
        dest = _dest(engine)
        errors = _scatter(engine,
            job_ids, parents, ordinals, foreign_own, generations, slots, active,
            attack, attack_count, genesis, counter, *dest)
        self.assertEqual(int(errors[1]), ERR_FOREIGN)
        self.assertEqual(float(dest[0][1, 1]), -1.0)
        self.assertEqual(float(dest[0][0, 0]), 1.5)
        cross = t.tensor([1, 0], dtype=t.int64)
        engine.compact_device(*batch)
        dest = _dest(engine)
        errors = _scatter(engine,
            job_ids, cross, ordinals, owners, generations, slots, active,
            attack, attack_count, genesis, counter, *dest)
        self.assertEqual(int(errors[0]), ERR_FOREIGN)
        self.assertEqual(int(errors[1]), ERR_FOREIGN)
        engine.compact_device(*batch)
        engine.compact_device(*batch)
        dest = _dest(engine)
        errors = _scatter(engine,
            job_ids, parents, ordinals, owners, generations, slots, active,
            attack, attack_count, genesis, counter, *dest)
        self.assertEqual(errors.tolist(), [0, 0])
        stale_gen = t.tensor([3, 1], dtype=t.int64)
        engine.compact_device(*batch)
        dest = _dest(engine)
        errors = _scatter(engine,
            job_ids, parents, ordinals, owners, stale_gen, slots, active,
            attack, attack_count, genesis, counter, *dest)
        self.assertEqual(int(errors[1]), ERR_STALE)

    def test_malformed_and_errored_rows_are_skipped(self):
        engine = _cpu(rows=3, max_actions=3, capacity=4)
        batch = _batch(engine)
        valid, battle, counts, active, input_errors, slots, generations, owners, actors = batch
        _row(batch, 0, 2, [0], slot=1, generation=1, owner=1)
        _row(batch, 1, 2, [0], slot=2, generation=1, owner=2)
        valid[1, 1] = False
        _row(batch, 2, 1, [0], slot=3, generation=1, owner=3)
        input_errors[2] = 7
        work = engine.compact_device(*batch)
        self.assertEqual(int(work.job_count[0]), 1)
        self.assertEqual(int(work.errors[1]), ERR_MALFORMED)
        self.assertEqual(int(work.errors[2]), 7)
        self.assertEqual(work.parent_row[:1].tolist(), [0])


    def test_same_parent_old_batch_receipt_is_stale_and_cannot_write(self):
        import torch as t
        engine = _cpu(rows=2, max_actions=2, capacity=2)
        batch = _batch(engine)
        _row(batch, 0, 1, [0], slot=5, generation=0, owner=7)
        engine.compact_device(*batch)
        old_epochs = engine.job_epoch[:1].clone()
        engine.compact_device(*batch)  # SAME owner/slot/gen/ordinal, new batch.
        dest = _dest(engine)
        ids = t.tensor([0], dtype=t.int64)
        claims = [t.tensor([x], dtype=t.int64) for x in (0, 0, 7, 0, 5)]
        active = t.ones(1, dtype=t.bool)
        values = [t.tensor([x], dtype=t.float64) for x in (9., 1., 0., 0.)]
        errors = _scatter(engine, ids, *claims, active, *values, *dest, epochs=old_epochs)
        self.assertTrue(int(errors[0]) & ERR_STALE)
        self.assertEqual(float(dest[0][0, 0]), -1.)
        self.assertFalse(bool(engine.finalize_device(dest[-1])[0]))

    def test_chunked_results_require_all_jobs_and_reject_later_duplicate(self):
        import torch as t
        from app.modules.card_game.rl.batched_search.device_teacher_preview_jobs import ERR_MISSING
        engine = _cpu(rows=2, max_actions=2, capacity=2)
        batch = _batch(engine)
        _row(batch, 0, 2, [0, 1], slot=4, generation=0, owner=9)
        engine.compact_device(*batch)
        dest = _dest(engine)
        def deliver(ordinal):
            ids = t.tensor([ordinal], dtype=t.int64)
            claims = [t.tensor([x], dtype=t.int64) for x in (0, ordinal, 9, 0, 4)]
            return _scatter(engine, ids, *claims, t.ones(1, dtype=t.bool),
                *(t.tensor([x], dtype=t.float64) for x in (2., 1., 0., 0.)), *dest)
        deliver(0)
        complete = engine.finalize_device(dest[-1])
        self.assertFalse(bool(complete[0]))
        self.assertTrue(int(dest[-1][0]) & ERR_MISSING)
        engine.compact_device(*batch)
        dest = _dest(engine)
        deliver(0)
        deliver(1)
        self.assertEqual(engine.finalize_device(dest[-1]).tolist(), [True, False])
        errors = deliver(0)
        self.assertTrue(int(errors[0]) & ERR_DUPLICATE)
        self.assertFalse(bool(engine.finalize_device(dest[-1])[0]))

    def test_foreign_claim_does_not_poison_claimed_other_root(self):
        import torch as t
        engine = _cpu(rows=2, max_actions=2, capacity=2)
        batch = _batch(engine)
        _row(batch, 0, 1, [0], slot=4, generation=0, owner=9)
        _row(batch, 1, 1, [0], slot=5, generation=0, owner=10)
        engine.compact_device(*batch)
        dest = _dest(engine)
        # Job 0 belongs to root 0; receipt falsely names root 1.
        ids = t.tensor([0], dtype=t.int64)
        claims = [t.tensor([x], dtype=t.int64) for x in (1, 0, 9, 0, 4)]
        errors = _scatter(engine, ids, *claims, t.ones(1, dtype=t.bool),
            *(t.tensor([x], dtype=t.float64) for x in (2., 1., 0., 0.)), *dest)
        self.assertTrue(int(errors[0]) & ERR_FOREIGN)
        self.assertTrue(int(dest[-1][0]) & ERR_FOREIGN)
        self.assertEqual(int(dest[-1][1]), 0)

    def test_actual_cuda_graph_replay_resets_buffers(self):
        import torch as t
        if not t.cuda.is_available():
            self.skipTest('Actual CUDA required; preview jobs unqualified')
        engine = DeviceTeacherPreviewJobs(
            rows=4, max_actions=4, capacity=8, device='cuda')
        self.addCleanup(engine.close)
        batch = _batch(engine)
        _row(batch, 0, 2, [0, 1], slot=1, generation=1, owner=10)
        _row(batch, 1, 1, [0], slot=2, generation=1, owner=11)
        engine.compact_device(*batch)
        dest = _dest(engine)
        with patch.object(t.Tensor, 'cpu', side_effect=AssertionError('CPU copy')), patch.object(t.Tensor, 'numpy', side_effect=AssertionError('NumPy copy')), patch.object(t.Tensor, 'item', side_effect=AssertionError('Scalar copy')), patch.object(t.Tensor, 'tolist', side_effect=AssertionError('Host list')), patch.object(t.Tensor, '__bool__', side_effect=AssertionError('Host branch')):
            work = engine.compact_device(*batch)
        self.assertEqual(int(work.job_count[0].cpu()), 3)
        job_ids = t.tensor([0, 1, 2], dtype=t.int64, device='cuda')
        parents = t.tensor([0, 0, 1], dtype=t.int64, device='cuda')
        ordinals = t.tensor([0, 1, 0], dtype=t.int64, device='cuda')
        owners = t.tensor([10, 10, 11], dtype=t.int64, device='cuda')
        generations = t.tensor([1, 1, 1], dtype=t.int64, device='cuda')
        slots = t.tensor([1, 1, 2], dtype=t.int64, device='cuda')
        active = t.ones(3, dtype=t.bool, device='cuda')
        attack = t.tensor([1.0, 2.0, 3.0], dtype=t.float64, device='cuda')
        attack_count = t.ones(3, dtype=t.float64, device='cuda')
        genesis = t.zeros(3, dtype=t.float64, device='cuda')
        counter = t.zeros(3, dtype=t.float64, device='cuda')
        t.cuda.synchronize()
        graph = t.cuda.CUDAGraph()
        with t.cuda.graph(graph):
            work = engine.compact_device(*batch)
            _scatter(engine,
                job_ids, parents, ordinals, owners, generations, slots, active,
                attack, attack_count, genesis, counter, *dest)
        battle = batch[1]
        battle[1, 0] = False
        graph.replay()
        t.cuda.synchronize()
        self.assertEqual(int(work.job_count[0].cpu()), 2)
        # A third stale active result is outside the new workload: reject it.
        self.assertTrue(bool((dest[-1] != 0).any().cpu()))
        active[2] = False
        dest[-1].zero_()
        graph.replay()
        t.cuda.synchronize()
        self.assertEqual(int(work.job_count[0].cpu()), 2)
        self.assertEqual(dest[-1].cpu().tolist(), [0, 0, 0, 0])
        complete = engine.finalize_device(dest[-1])
        self.assertTrue(bool(complete[0].cpu()))


if __name__ == '__main__':
    unittest.main()
