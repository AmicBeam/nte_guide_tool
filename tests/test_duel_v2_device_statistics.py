"""Numerical and capture-boundary tests for device Gumbel statistics."""
import unittest
from unittest.mock import patch
import numpy as np
from app.modules.card_game.rl.batched_search.device_statistics import DeviceStatisticsPool
from app.modules.card_game.rl.information_search import Node
from app.modules.card_game.rl import recovery_search, search_policy


def cpu_pool(capacity, max_actions, **kwargs):
    return DeviceStatisticsPool(capacity, max_actions, device='cpu', allow_cpu_reference=True, **kwargs)


def pad_prior(prior, width):
    prior = np.asarray(prior, dtype=np.float64)
    out = np.zeros(width, dtype=np.float64)
    out[:len(prior)] = prior
    legal = np.zeros(width, dtype=np.bool_)
    legal[:len(prior)] = True
    return out, legal


def as_node(prior, visits, total, value, legal):
    node = Node.create(np.asarray(prior, dtype=np.float64)[legal], float(value))
    node.visits[:] = np.asarray(visits)[legal]
    node.total[:] = np.asarray(total, dtype=np.float64)[legal]
    return node


class DeviceStatisticsTest(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.torch = torch

    def t(self, value, dtype=None):
        kwargs = dict(device='cpu')
        if dtype is not None:
            kwargs['dtype'] = dtype
        return self.torch.as_tensor(value, **kwargs)

    def write_compact(self, pool, slots, owners, priors, values):
        width = pool.max_actions
        padded = np.zeros((len(priors), width), dtype=np.float64)
        legal = np.zeros((len(priors), width), dtype=np.bool_)
        for i, prior in enumerate(priors):
            padded[i], legal[i] = pad_prior(prior, width)
        written = pool.write_nodes(
            self.t(slots, self.torch.int64), self.t(owners, self.torch.int64),
            self.t(padded, self.torch.float64), self.t(values, self.torch.float64),
            self.t(legal))
        return written, legal

    def backup_actions(self, pool, slots, generations, actions, values, owners, active=None):
        return pool.backup(
            slots[:, None], generations[:, None], actions[:, None],
            self.torch.ones((slots.shape[0], 1), dtype=self.torch.bool, device=slots.device),
            values, owners, active)

    def test_cpu_requires_explicit_reference(self):
        with self.assertRaises(ValueError):
            DeviceStatisticsPool(4, 8, device='cpu')
        with self.assertRaises(ValueError):
            DeviceStatisticsPool(4, 8, device='cpu', q_scale='legacy', allow_cpu_reference=True)

    def test_unvisited_mixed_q_and_full_policy_match_numpy(self):
        pool = cpu_pool(4, 8)
        priors = [np.array([.75, .25]), np.array([.4, .2, .4])]
        values = [0., -.2]
        written, legal = self.write_compact(pool, [0, 1], [10, 11], priors, values)
        self.assertFalse(bool(written.error.any()))
        slots = self.t([0, 1], self.torch.int64)
        gens = written.generations
        owners = self.t([10, 11], self.torch.int64)
        self.backup_actions(pool, slots, gens, self.t([0, 2], self.torch.int64),
                            self.t([1., -.5], self.torch.float64), owners)
        self.backup_actions(pool, slots[:1], gens[:1], self.t([0], self.torch.int64),
                            self.t([1.], self.torch.float64), owners[:1])
        snap = pool.snapshot(slots, gens)
        q = pool.completed_q(slots, gens)
        pi = pool.improved_policy(slots, gens)
        self.assertFalse(bool(q.error.any() | pi.error.any() | snap.error.any()))
        host_q = q.completed_q.detach().cpu().numpy()
        host_pi = pi.policy.detach().cpu().numpy()
        host_snap = {k: getattr(snap, k).detach().cpu().numpy() for k in ('prior', 'visits', 'total', 'value', 'legal')}
        for i, prior in enumerate(priors):
            node = as_node(host_snap['prior'][i], host_snap['visits'][i], host_snap['total'][i],
                           host_snap['value'][i], host_snap['legal'][i])
            expected_q = recovery_search.completed_q(node, 'natural_wdl')
            expected_pi = recovery_search.improved_policy(node, 'natural_wdl')
            np.testing.assert_allclose(host_q[i, host_snap['legal'][i]], expected_q, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(host_pi[i, host_snap['legal'][i]], expected_pi, atol=1e-12, rtol=1e-12)
            self.assertTrue(np.all(host_q[i, ~host_snap['legal'][i]] == 0))
            self.assertTrue(np.all(host_pi[i, ~host_snap['legal'][i]] == 0))
            self.assertGreater(len(prior), 0)
        self.assertEqual(int(host_snap['visits'][0, 0]), 2)
        self.assertEqual(int(host_snap['visits'][0, 1]), 0)

    def test_padding_and_inactive_rows_are_ignored(self):
        pool = cpu_pool(5, 8)
        prior = np.array([.2, .5, .3])
        written, _ = self.write_compact(pool, [0, 2], [1, 2], [prior, prior], [0.1, -0.3])
        gens = written.generations
        slots = self.t([0, 1, 2, 4], self.torch.int64)
        active = self.t([True, False, True, False])
        padded_gen = self.t([int(gens[0]), 99, int(gens[1]), -3], self.torch.int64)
        noise = self.torch.zeros((4, 8), dtype=self.torch.float64)
        noise[0, 1] = 4.
        targets = self.t([0, 0, 0, 0], self.torch.int64)
        choice = pool.select_root(slots, padded_gen, noise, targets, active=active)
        self.assertFalse(bool(choice.error[0] | choice.error[2]))
        self.assertFalse(bool(choice.error[1] | choice.error[3]))
        self.assertEqual(int(choice.actions[0]), 1)
        self.assertEqual(int(choice.actions[1]), 0)
        interior = pool.select_interior(slots, padded_gen, active=active)
        q = pool.completed_q(slots, padded_gen, active=active)
        self.assertTrue(np.all(q.completed_q[1].detach().cpu().numpy() == 0))
        self.assertTrue(np.all(q.completed_q[3].detach().cpu().numpy() == 0))
        before = pool.snapshot(self.t([0, 2], self.torch.int64), gens)
        err = self.backup_actions(
            pool, slots, padded_gen, self.t([1, 7, 0, 7], self.torch.int64),
            self.t([.2, .9, -.1, .9], self.torch.float64),
            self.t([1, 99, 2, 99], self.torch.int64), active=active)
        self.assertFalse(bool(err.error[0] | err.error[2]))
        after = pool.snapshot(self.t([0, 2], self.torch.int64), gens)
        self.assertEqual(int(after.visits[0, 1]), 1)
        self.assertEqual(int(after.visits[1, 0]), 1)
        self.assertEqual(int(before.visits[0, 7]), 0)
        self.assertEqual(int(after.visits[0, 7]), 0)
        self.assertFalse(bool(pool._occupied[1] | pool._occupied[3] | pool._occupied[4]))
        colliding = cpu_pool(3, 4)
        padded, legal = pad_prior(np.array([.25, .75]), 4)
        other, other_legal = pad_prior(np.array([1.]), 4)
        result = colliding.write_nodes(
            self.t([0, -1], self.torch.int64), self.t([5, 6], self.torch.int64),
            self.t(np.stack([padded, other]), self.torch.float64),
            self.t([-.4, .3], self.torch.float64),
            self.t(np.stack([legal, other_legal])),
            active=self.t([True, False]),
        )
        self.assertFalse(bool(result.error[0]))
        self.assertFalse(bool(result.error[1]))
        snap = colliding.snapshot(self.t([0], self.torch.int64), result.generations[:1])
        np.testing.assert_allclose(snap.prior[0, :2].detach().cpu().numpy(), [.25, .75], atol=1e-12)
        np.testing.assert_allclose(snap.value[0].detach().cpu().numpy(), -.4, atol=1e-12)

    def test_max_visit_final_and_first_index_ties(self):
        pool = cpu_pool(3, 6)
        prior = np.array([.25, .25, .25, .25])
        written, _ = self.write_compact(pool, [0], [7], [prior], [0.])
        slots = self.t([0], self.torch.int64)
        gens = written.generations[:1]
        owners = self.t([7], self.torch.int64)
        for action in (1, 1, 2, 3, 3):
            self.backup_actions(pool, slots, gens, self.t([action], self.torch.int64),
                                self.t([0.], self.torch.float64), owners)
        noise = self.torch.zeros((1, 6), dtype=self.torch.float64)
        final = pool.select_final(slots, gens, noise)
        self.assertFalse(bool(final.error.any()))
        self.assertEqual(int(final.actions[0]), 1)
        self.assertEqual(int(final.roots_covered[0]), 3)
        node = Node.create(prior, 0.)
        node.visits[:] = [0, 2, 1, 2]
        expected = recovery_search.choose_root(node, np.zeros(4), int(node.visits.max()), 'natural_wdl')
        self.assertEqual(int(final.actions[0]), expected)
        tied = cpu_pool(2, 4)
        written, _ = self.write_compact(tied, [0], [1], [np.array([.5, .5])], [0.])
        root = tied.select_root(self.t([0], self.torch.int64), written.generations[:1],
                                self.torch.zeros((1, 4), dtype=self.torch.float64),
                                self.t([0], self.torch.int64))
        interior = tied.select_interior(self.t([0], self.torch.int64), written.generations[:1])
        self.assertEqual(int(root.actions[0]), 0)
        self.assertEqual(int(interior.actions[0]), 0)

    def test_candidate_width_beyond_16_is_not_truncated(self):
        pool = cpu_pool(2, 24)
        rng = np.random.default_rng(3)
        prior = search_policy.normalized_exp(rng.normal(size=20))
        written, legal = self.write_compact(pool, [0], [4], [prior], [.15])
        slots = self.t([0], self.torch.int64)
        gens = written.generations[:1]
        noise = self.torch.zeros((1, 24), dtype=self.torch.float64)
        noise[0, 18] = 8.
        choice = pool.select_root(slots, gens, noise, self.t([0], self.torch.int64))
        self.assertFalse(bool(choice.error.any()))
        self.assertEqual(int(choice.actions[0]), 18)
        with self.assertRaises(ValueError):
            pool.select_root(slots, gens, self.torch.zeros((1, 20), dtype=self.torch.float64),
                             self.t([0], self.torch.int64))
        node = Node.create(prior, .15)
        expected = recovery_search.choose_root(node, np.concatenate([np.zeros(18), [8.], np.zeros(1)]), 0, 'natural_wdl')
        self.assertEqual(int(choice.actions[0]), expected)

    def test_illegal_duplicate_stale_and_ownership_do_not_mutate_other_rows(self):
        pool = cpu_pool(6, 8)
        priors = [np.array([.5, .5]), np.array([.2, .3, .5]), np.array([1.])]
        written, _ = self.write_compact(pool, [0, 1, 2], [10, 11, 12], priors, [0., .1, -.1])
        self.assertFalse(bool(written.error.any()))
        slots = self.t([0, 1, 2], self.torch.int64)
        gens = written.generations
        owners = self.t([10, 11, 12], self.torch.int64)
        self.backup_actions(pool, slots[:1], gens[:1], self.t([1], self.torch.int64),
                            self.t([.3], self.torch.float64), owners[:1])
        before = pool.snapshot(slots, gens)
        mixed = pool.backup(
            self.t([[0, 0], [1, 1], [2, 2], [1, 1]], self.torch.int64),
            self.torch.stack([gens[0].expand(2), gens[1].expand(2), gens[2].expand(2), gens[1].expand(2)]),
            self.t([[1, 9], [0, 0], [0, 0], [1, 1]], self.torch.int64),
            self.t([[True, True], [True, True], [True, True], [True, True]]),
            self.t([.4, .2, .2, .7], self.torch.float64),
            self.t([10, 11, 11, 12], self.torch.int64),
        )
        after = pool.snapshot(slots, gens)
        self.assertTrue(bool(mixed.error[0]))
        self.assertTrue(bool(mixed.error[1] and mixed.error[2]))
        np.testing.assert_array_equal(after.visits.detach().cpu().numpy(), before.visits.detach().cpu().numpy())
        np.testing.assert_array_equal(after.total.detach().cpu().numpy(), before.total.detach().cpu().numpy())
        released = pool.release_nodes(slots[:1], gens[:1])
        self.assertFalse(bool(released.error.any()))
        stale = self.backup_actions(pool, slots[:1], gens[:1], self.t([0], self.torch.int64),
                                    self.t([.9], self.torch.float64), owners[:1])
        self.assertTrue(bool(stale.error[0]))
        still = pool.snapshot(slots[1:], gens[1:])
        np.testing.assert_array_equal(still.visits.detach().cpu().numpy(), after.visits[1:].detach().cpu().numpy())
        dup_write = pool.write_nodes(
            self.t([3, 3], self.torch.int64), self.t([1, 2], self.torch.int64),
            self.t(np.stack([pad_prior([1.], 8)[0], pad_prior([1.], 8)[0]]), self.torch.float64),
            self.t([0., 0.], self.torch.float64),
            self.t(np.stack([pad_prior([1.], 8)[1], pad_prior([1.], 8)[1]])),
        )
        self.assertTrue(bool(dup_write.error.all()))
        self.assertFalse(bool(pool._occupied[3]))
        bad_value = pool.backup(
            slots[1:2, None], gens[1:2, None], self.t([[0]], self.torch.int64),
            self.t([[True]]), self.t([1.5], self.torch.float64), owners[1:2])
        self.assertTrue(bool(bad_value.error[0]))
        self.assertEqual(int(pool.snapshot(slots[1:2], gens[1:2]).visits.sum()),
                         int(after.visits[1].sum()))

    def test_fixed_observer_path_backup_and_sequential_budget(self):
        pool = cpu_pool(6, 8)
        rng = np.random.default_rng(17)
        priors = [search_policy.normalized_exp(rng.normal(size=n)) for n in (1, 3, 7)]
        values = rng.uniform(-1, 1, 3)
        written, _ = self.write_compact(pool, [0, 1, 2], [1, 2, 3], priors, values)
        slots = self.t([0, 1, 2], self.torch.int64)
        gens = written.generations
        owners = self.t([1, 2, 3], self.torch.int64)
        nodes = [Node.create(p, v) for p, v in zip(priors, values)]
        schedules = [search_policy.visit_schedule(min(len(p), 16, 32), 32) for p in priors]
        noise = self.torch.zeros((3, 8), dtype=self.torch.float64)
        for step in range(32):
            targets = self.t([schedule[step] for schedule in schedules], self.torch.int64)
            selected = pool.select_root(slots, gens, noise, targets)
            interior = pool.select_interior(slots, gens)
            q = pool.completed_q(slots, gens)
            pi = pool.improved_policy(slots, gens)
            self.assertFalse(bool(selected.error.any() | interior.error.any() | q.error.any() | pi.error.any()))
            host_sel = selected.actions.detach().cpu().numpy()
            host_int = interior.actions.detach().cpu().numpy()
            host_q = q.completed_q.detach().cpu().numpy()
            host_pi = pi.policy.detach().cpu().numpy()
            for i, node in enumerate(nodes):
                self.assertEqual(int(host_sel[i]), recovery_search.choose_root(node, np.zeros(len(node.prior)), schedules[i][step], 'natural_wdl'))
                expected_int = int(np.argmax(recovery_search.improved_policy(node, 'natural_wdl') - node.visits / (1 + node.visits.sum())))
                self.assertEqual(int(host_int[i]), expected_int)
                np.testing.assert_allclose(host_q[i, :len(node.prior)], recovery_search.completed_q(node, 'natural_wdl'), atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(host_pi[i, :len(node.prior)], recovery_search.improved_policy(node, 'natural_wdl'), atol=1e-12, rtol=1e-12)
                self.assertTrue(np.all(host_pi[i, len(node.prior):] == 0))
            rewards = rng.uniform(-1, 1, 3)
            backed = self.backup_actions(pool, slots, gens, selected.actions,
                                         self.t(rewards, self.torch.float64), owners)
            self.assertFalse(bool(backed.error.any()))
            for i, node in enumerate(nodes):
                node.visits[host_sel[i]] += 1
                node.total[host_sel[i]] += rewards[i]
        snap = pool.snapshot(slots, gens)
        for i, node in enumerate(nodes):
            np.testing.assert_array_equal(snap.visits[i, :len(node.visits)].detach().cpu().numpy(), node.visits)
            np.testing.assert_allclose(snap.total[i, :len(node.total)].detach().cpu().numpy(), node.total, atol=1e-12)
        final = pool.select_final(slots, gens, noise)
        for i, node in enumerate(nodes):
            self.assertEqual(int(final.actions[i]), recovery_search.choose_root(node, np.zeros(len(node.prior)), int(node.visits.max()), 'natural_wdl'))
            self.assertEqual(int(final.roots_covered[i]), int((node.visits > 0).sum()))
        observer = cpu_pool(4, 4)
        written, _ = self.write_compact(observer, [0, 1], [9, 9], [np.array([.5, .5]), np.array([1.])], [0., 0.])
        path = observer.backup(
            self.t([[0, 1]], self.torch.int64),
            written.generations[None, :].expand(1, 2).contiguous(),
            self.t([[0, 0]], self.torch.int64),
            self.t([[True, True]]),
            self.t([.5], self.torch.float64),
            self.t([9], self.torch.int64),
        )
        self.assertFalse(bool(path.error.any()))
        seen = observer.snapshot(self.t([0, 1], self.torch.int64), written.generations)
        np.testing.assert_array_equal(seen.visits.detach().cpu().numpy()[:, 0], [1, 1])
        np.testing.assert_allclose(seen.total.detach().cpu().numpy()[:, 0], [.5, .5], atol=1e-12)
        self.assertEqual(int(seen.total[0, 1]), 0)

    def test_hot_path_stays_on_device_and_workspace_is_bounded(self):
        import torch
        pool = cpu_pool(8, 12)
        written, _ = self.write_compact(pool, [0, 1], [1, 2], [np.array([.6, .4]), np.array([1.])], [0., .2])
        slots = self.t([0, 1], self.torch.int64)
        gens = written.generations
        noise = self.torch.zeros((2, 12), dtype=self.torch.float64)
        targets = self.t([0, 0], self.torch.int64)
        values = self.t([.1, -.2], self.torch.float64)
        owners = self.t([1, 2], self.torch.int64)
        active = self.t([True, True])
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Hot-path CPU copy')), \
             patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('Hot-path NumPy')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('Hot-path scalar')), \
             patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('Hot-path list')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Hot-path truth')):
            pool.completed_q(slots, gens, active=active)
            pool.improved_policy(slots, gens, active=active)
            selected = pool.select_root(slots, gens, noise, targets, active=active)
            pool.select_interior(slots, gens, active=active)
            pool.select_final(slots, gens, noise, active=active)
            self.backup_actions(pool, slots, gens, selected.actions, values, owners, active=active)
            pool.set_value(slots, gens, values, active=active)
            pool.clear_visits(slots, gens, active=active)
            pool.snapshot(slots, gens, active=active)
        expected = (9 * 12 * 8 * 2 + 9 * 12 * 4 + 9 * 8 + 9 * 12 + 9 * 8 * 2 + 9 + 8)
        self.assertEqual(pool.allocated_bytes, expected)
        self.assertGreater(pool.allocated_bytes, 0)

    def test_invalid_device_rows_are_reported_without_host_fallback(self):
        pool = cpu_pool(4, 6)
        written, _ = self.write_compact(pool, [0, 1], [1, 2], [np.array([.5, .5]), np.array([.5, .5])], [0., 0.])
        slots = self.t([0, 1], self.torch.int64)
        gens = written.generations
        noise = self.torch.zeros((2, 6), dtype=self.torch.float64)
        targets = self.t([3, 0], self.torch.int64)
        choice = pool.select_root(slots, gens, noise, targets)
        self.assertTrue(bool(choice.error[0]))
        self.assertFalse(bool(choice.error[1]))
        self.assertEqual(int(choice.actions[0]), 0)
        self.assertEqual(int(choice.actions[1]), 0)

    def test_rejected_nan_backup_cannot_poison_live_or_inactive_nodes(self):
        pool = cpu_pool(3, 3)
        written, _ = self.write_compact(pool, [0, 1], [1, 2],
            [np.array([.5, .5]), np.array([.5, .5])], [0., 0.])
        slots = self.t([[0], [1]], self.torch.int64)
        gens = written.generations[:, None]
        actions = self.t([[0], [0]], self.torch.int64)
        valid = self.t([[True], [True]])
        for bad in (float('nan'), float('inf')):
            leaf = self.t([bad, .25], self.torch.float64)
            result = pool.backup(slots, gens, actions, valid, leaf,
                                 self.t([1, 2], self.torch.int64))
            self.assertEqual(result.error.tolist(), [True, False])
            self.assertTrue(bool(self.torch.isfinite(pool._total[:3]).all()))
            self.assertEqual(int(pool._visits[0, 0]), 0)
        self.assertEqual(int(pool._visits[1, 0]), 2)

    def test_padding_cannot_hide_duplicate_active_slot_or_owner(self):
        pool = cpu_pool(4, 3)
        slots = self.t([0, 0, 0], self.torch.int64)
        owners = self.t([7, 7, 7], self.torch.int64)
        active = self.t([True, False, True])
        priors = self.torch.tensor([[.5, .5, 0.]]*3, dtype=self.torch.float64)
        legal = self.torch.tensor([[True, True, False]]*3)
        written = pool.write_nodes(slots, owners, priors,
            self.t([0., 0., 0.], self.torch.float64), legal, active=active)
        self.assertEqual(written.error.tolist(), [True, False, True])
        self.assertFalse(bool(pool._occupied[0]))

    def test_reused_node_requires_matching_generation_at_creation(self):
        pool = cpu_pool(3, 3)
        slots = self.t([0], self.torch.int64)
        owners = self.t([1], self.torch.int64)
        priors = self.torch.tensor([[.5, .5, 0.]], dtype=self.torch.float64)
        legal = self.torch.tensor([[True, True, False]])
        values = self.t([0.], self.torch.float64)
        first = pool.write_nodes(slots, owners, priors, values, legal)
        pool.release_nodes(slots, first.generations)
        self.assertTrue(bool(pool.write_nodes(slots, owners, priors, values, legal).error[0]))
        stale = pool.write_nodes(slots, owners, priors, values, legal,
                                 expected_generations=first.generations)
        self.assertTrue(bool(stale.error[0]))
        current = pool.write_nodes(slots, owners, priors, values, legal,
                                   expected_generations=first.generations+1)
        self.assertFalse(bool(current.error[0]))

    def test_actual_cuda_1600_rows_and_graph_replay(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self._cuda_1600_on_owned_stream(stream)

    def _cuda_1600_on_owned_stream(self, stream):
        import torch
        device = 'cuda'
        rows = 1600
        width = 8
        pool = DeviceStatisticsPool(rows, width, device=device, max_batch=rows, max_path_len=2)
        rng = np.random.default_rng(21)
        priors = np.zeros((rows, width), dtype=np.float64)
        legal = np.zeros((rows, width), dtype=np.bool_)
        values = rng.uniform(-.5, .5, rows)
        for i in range(rows):
            n = 3 + (i % 5)
            priors[i, :n] = search_policy.normalized_exp(rng.normal(size=n))
            legal[i, :n] = True
        slots = torch.arange(rows, device=device, dtype=torch.int64)
        owners = slots.clone()
        written = pool.write_nodes(slots, owners, torch.as_tensor(priors, device=device),
                                   torch.as_tensor(values, device=device, dtype=torch.float64),
                                   torch.as_tensor(legal, device=device))
        self.assertFalse(bool(written.error.any()))
        gens = written.generations
        noise = torch.zeros((rows, width), dtype=torch.float64, device=device)
        targets = torch.zeros(rows, dtype=torch.int64, device=device)
        path_slots = slots[:, None]
        path_gens = gens[:, None]
        path_actions = torch.zeros((rows, 1), dtype=torch.int64, device=device)
        path_valid = torch.ones((rows, 1), dtype=torch.bool, device=device)
        leaf = torch.as_tensor(rng.uniform(-.4, .4, rows), dtype=torch.float64, device=device)
        warm = stream
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                pool.clear_visits(slots, gens)
                pool.backup(path_slots, path_gens, path_actions, path_valid, leaf, owners)
                pool.select_root(slots, gens, noise, targets)
                pool.completed_q(slots, gens)
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cleared = pool.clear_visits(slots, gens)
            backed = pool.backup(path_slots, path_gens, path_actions, path_valid, leaf, owners)
            captured_q = pool.completed_q(slots, gens)
            captured = pool.select_final(slots, gens, noise)
        path_actions.copy_(torch.full_like(path_actions, 1))
        leaf.copy_(torch.full_like(leaf, .25))
        graph.replay()
        torch.cuda.synchronize()
        expected_clear = pool.clear_visits(slots, gens)
        expected_back = pool.backup(path_slots, path_gens, path_actions, path_valid, leaf, owners)
        expected_q = pool.completed_q(slots, gens)
        expected_final = pool.select_final(slots, gens, noise)
        torch.testing.assert_close(captured_q.completed_q, expected_q.completed_q)
        torch.testing.assert_close(captured.actions, expected_final.actions)
        torch.testing.assert_close(captured.roots_covered, expected_final.roots_covered)
        self.assertFalse(bool(captured.error.any() | backed.error.any() | cleared.error.any()))
        self.assertFalse(bool(expected_final.error.any() | expected_back.error.any() | expected_clear.error.any()))
        cpu = cpu_pool(4, width)
        sample = [0, 7, 19, 1599]
        cpu_written, _ = self.write_compact(cpu, [0, 1, 2, 3], sample,
                                            [priors[i, legal[i]] for i in sample], values[sample])
        cpu_slots = self.t([0, 1, 2, 3], self.torch.int64)
        cpu.backup(cpu_slots[:, None], cpu_written.generations[:, None],
                   self.t([[1], [1], [1], [1]], self.torch.int64),
                   self.t([[True], [True], [True], [True]]),
                   self.t([.25, .25, .25, .25], self.torch.float64),
                   self.t(sample, self.torch.int64))
        cpu_q = cpu.completed_q(cpu_slots, cpu_written.generations).completed_q.detach().cpu().numpy()
        gpu_q = expected_q.completed_q.detach().cpu().numpy()
        for row, index in enumerate(sample):
            np.testing.assert_allclose(gpu_q[index, legal[index]], cpu_q[row, :legal[index].sum()], atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()

