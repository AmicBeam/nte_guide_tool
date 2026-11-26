"""CPU and fake-device checks for bulk host materialization in inference/statistics."""
import traceback
import unittest
from unittest import mock
from uuid import uuid4
from dataclasses import replace
import numpy as np

from app.modules.card_game.rl.batched_search.cooperative import OperationQuery
from app.modules.card_game.rl.batched_search.inference import BatchedInference, InferenceRequest
from app.modules.card_game.rl.batched_search.nodes import BatchedNodePool
from app.modules.card_game.rl.batched_search.statistics import BatchedStatistics
from tests.test_duel_v2_batch_inference import (
    SyntheticNumericModel,
    make_legal_candidates,
    make_legal_observation,
)


OWNED = (
    'app/modules/card_game/rl/batched_search/inference.py',
    'app/modules/card_game/rl/batched_search/statistics.py',
)


def _from_owned(frame_filename):
    normalized = frame_filename.replace('\\', '/')
    return any(normalized.endswith(path) for path in OWNED)


class HostTransferProbe:
    """Count explicit host materializations originating in the owned modules."""

    def __init__(self):
        import torch
        self.torch = torch
        self.calls = []
        self._restore = {}

    def __enter__(self):
        tensor = self.torch.Tensor
        self._restore = {
            'to': tensor.to,
            'cpu': tensor.cpu,
            'item': tensor.item,
            'numpy': tensor.numpy,
            'tolist': tensor.tolist,
        }
        probe = self

        def record(method, owned_tensor):
            stack = traceback.extract_stack()
            if not any(_from_owned(frame.filename) for frame in stack):
                return
            probe.calls.append((method, tuple(owned_tensor.shape), int(owned_tensor.dim()), int(owned_tensor.numel())))

        def to(self, *args, **kwargs):
            device = kwargs.get('device', None)
            if device is None and args:
                first = args[0]
                if first is None or isinstance(first, (str, int, probe.torch.device)):
                    device = first
            if device is not None and str(device) == 'cpu':
                record('to_cpu', self)
            return probe._restore['to'](self, *args, **kwargs)

        def cpu(self, memory_format=None):
            record('cpu', self)
            if memory_format is None:
                return probe._restore['cpu'](self)
            return probe._restore['cpu'](self, memory_format=memory_format)

        def item(self):
            record('item', self)
            return probe._restore['item'](self)

        def numpy(self, *args, **kwargs):
            record('numpy', self)
            return probe._restore['numpy'](self, *args, **kwargs)

        def tolist(self):
            record('tolist', self)
            return probe._restore['tolist'](self)

        tensor.to = to
        tensor.cpu = cpu
        tensor.item = item
        tensor.numpy = numpy
        tensor.tolist = tolist
        return self

    def __exit__(self, exc_type, exc, tb):
        tensor = self.torch.Tensor
        for name, original in self._restore.items():
            setattr(tensor, name, original)
        return False

    def owned(self, *methods):
        return [row for row in self.calls if row[0] in methods]


def query(backend, method, *args, qid=None):
    return OperationQuery(qid or uuid4().hex, backend, method, args, ())


def resolve(backend, method, *batches):
    queries = [query(backend, method, *args, qid=f'{method}-{i}') for i, args in enumerate(batches)]
    return [row.value for row in backend.resolve_operations(queries)]


class DeviceSyncReductionTest(unittest.TestCase):
    def test_workspace_growth_keeps_the_candidate_element_limit(self):
        model = SyntheticNumericModel('bounds', hidden=16)
        width = make_legal_candidates(1).shape[1]
        runner = BatchedInference({'bounds': model}, device='cpu',
                                   max_batch_rows=8, max_candidate_elements=40 * width)
        x = make_legal_observation(2)
        for rows, actions in ((1, 40), (8, 1), (2, 20), (4, 10)):
            requests = [InferenceRequest(str(i), 'bounds', model.version, x,
                                          make_legal_candidates(actions, seed=i), True)
                        for i in range(rows)]
            results = runner.predict(requests)
            self.assertLessEqual(runner._host_workspace['c'].size, 40 * width)
            for req, result in zip(requests, results):
                np.testing.assert_allclose(result.logits, model.scores(x, req.candidates), atol=1e-5)

    def test_overlapping_host_calls_do_not_overwrite_each_others_inputs(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        model = SyntheticNumericModel('threads', hidden=16)
        runner = BatchedInference({'threads': model}, device='cpu')
        entered, release, second_started = Event(), Event(), Event()
        original = runner._fill_host_workspace
        calls = []
        def fill(*args):
            out = original(*args)
            calls.append(True)
            if len(calls) == 1:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('Test did not release first call')
            return out
        def request(seed):
            return InferenceRequest(str(seed), 'threads', model.version,
                                    make_legal_observation(seed), make_legal_candidates(3, seed=seed), True)
        a, b = request(4), request(9)
        def second():
            second_started.set()
            return runner.predict([b])
        with mock.patch.object(runner, '_fill_host_workspace', side_effect=fill), ThreadPoolExecutor(2) as pool:
            first = pool.submit(runner.predict, [a])
            try:
                self.assertTrue(entered.wait(5))
                other = pool.submit(second)
                self.assertTrue(second_started.wait(5))
                self.assertEqual(len(calls), 1)
            finally:
                release.set()
            one, two = first.result(5)[0], other.result(5)[0]
        np.testing.assert_allclose(one.logits, model.scores(a.x, a.candidates), atol=1e-5)
        np.testing.assert_allclose(two.logits, model.scores(b.x, b.candidates), atol=1e-5)

    def test_inference_packed_host_copy_matches_reference_and_reuses_workspace(self):
        model = SyntheticNumericModel('sync', hidden=16)
        runner = BatchedInference({'sync': model}, device='cpu', max_batch_rows=8, max_candidate_elements=40 * make_legal_candidates(1).shape[1])
        x = make_legal_observation(21)
        lengths = (1, 4, 7, 10)
        requests = [
            InferenceRequest(f'r{i}', 'sync', model.version, x, make_legal_candidates(n, seed=i), bool(i % 2))
            for i, n in enumerate(lengths)
        ]
        with HostTransferProbe() as probe:
            results = runner.predict(requests)
        self.assertEqual(len(results), 4)
        for req, res in zip(requests, results):
            self.assertEqual(res.request_id, req.request_id)
            self.assertEqual(res.model_version, model.version)
            np.testing.assert_allclose(res.logits, model.scores(req.x, req.candidates), atol=1e-5)
            np.testing.assert_allclose(res.wdl, model.wdl(req.x, req.value_actor), atol=1e-5)
            self.assertAlmostEqual(float(res.wdl.sum()), 1.0, places=5)
            original = res.logits.copy()
            res.logits[0] += 3.0
            np.testing.assert_allclose(original, model.scores(req.x, req.candidates), atol=1e-5)
            self.assertNotEqual(float(res.logits[0]), float(original[0]))
        st = runner.stats()
        self.assertEqual(st['batches'], 1)
        self.assertEqual(st['rows'], 4)
        self.assertEqual(st['candidate_padding'], 18)
        self.assertEqual(st['host_materializations'], 1)
        self.assertEqual(st['workspace_allocations'], 1)
        self.assertEqual(len(probe.owned('to_cpu', 'cpu')), 1)
        self.assertEqual(probe.owned('item'), [])
        self.assertEqual(probe.owned('tolist'), [])
        runner.predict(requests)
        self.assertEqual(runner.stats()['workspace_allocations'], 1)
        self.assertEqual(runner.stats()['host_materializations'], 2)

    def test_inference_value_actor_and_value_only_stay_bound_after_workspace_reuse(self):
        model = SyntheticNumericModel('leaf', hidden=16)
        runner = BatchedInference({'leaf': model}, device='cpu')
        x = make_legal_observation(8)
        c = make_legal_candidates(3, seed=8)
        empty = np.empty((0, c.shape[1]), dtype=np.float32)
        first = [
            InferenceRequest('a1', 'leaf', model.version, x, c, True),
            InferenceRequest('a0', 'leaf', model.version, x, c, 0),
        ]
        second = [
            InferenceRequest('v0', 'leaf', model.version, x, empty, False, True),
            InferenceRequest('v1', 'leaf', model.version, x, empty, True, True),
        ]
        policy = runner.predict(first)
        leaves = runner.predict(second)
        np.testing.assert_allclose(policy[0].wdl, model.wdl(x, True), atol=1e-5)
        np.testing.assert_allclose(policy[1].wdl, model.wdl(x, False), atol=1e-5)
        self.assertEqual(leaves[0].logits.shape, (0,))
        self.assertEqual(leaves[1].logits.shape, (0,))
        np.testing.assert_allclose(leaves[0].wdl, model.wdl(x, False), atol=1e-5)
        np.testing.assert_allclose(leaves[1].wdl, model.wdl(x, True), atol=1e-5)
        again = runner.predict([InferenceRequest('short', 'leaf', model.version, x, make_legal_candidates(1, seed=9), True)])
        np.testing.assert_allclose(again[0].logits, model.scores(x, make_legal_candidates(1, seed=9)), atol=1e-5)
        self.assertEqual(runner.stats()['host_materializations'], 3)

    def test_inference_keeps_request_identity_and_validation(self):
        model = SyntheticNumericModel('err', hidden=16)
        runner = BatchedInference({'err': model}, device='cpu')
        x = make_legal_observation(3)
        c = make_legal_candidates(2, seed=3)
        with self.assertRaises(ValueError):
            runner.predict([InferenceRequest('ver', 'err', 'nope', x, c, True)])
        packed = np.zeros((1, c.shape[0] + 3), dtype=np.float32)
        packed[0, :c.shape[0]] = np.nan
        packed[0, c.shape[0]:] = (0.2, 0.3, 0.5)
        with mock.patch.object(runner, '_materialize_microbatch', return_value=packed):
            with self.assertRaises(ValueError) as ctx:
                runner.predict([InferenceRequest('bad-logit', 'err', model.version, x, c, True)])
        self.assertIn('bad-logit', str(ctx.exception))
        packed = np.zeros((1, c.shape[0] + 3), dtype=np.float32)
        packed[0, c.shape[0]:] = (0.5, 0.5, 0.5)
        with mock.patch.object(runner, '_materialize_microbatch', return_value=packed):
            with self.assertRaises(ValueError) as ctx:
                runner.predict([InferenceRequest('bad-wdl', 'err', model.version, x, c, True)])
        self.assertIn('bad-wdl', str(ctx.exception))

    def test_statistics_snapshot_choices_and_backup_match_pool(self):
        priors = [np.array([0.2, 0.8]), np.array([0.5, 0.25, 0.25]), np.array([1.0])]
        values = [0.1, -0.2, 0.0]
        noise = [np.zeros(len(p), dtype=np.float64) for p in priors]
        with BatchedStatistics(6, 4, device='cpu') as stats:
            owners = [stats.new_owner() for _ in priors]
            refs = resolve(stats, 'create', *[(p, v, owner.id) for p, v, owner in zip(priors, values, owners)])
            pool = BatchedNodePool(6, 4, device='cpu')
            pool_refs = pool.allocate(priors, values, [owner.id for owner in owners])
            with HostTransferProbe() as probe:
                chosen = resolve(stats, 'root_choice', *[(ref, n, 0) for ref, n in zip(refs, noise)])
            expected = pool.select_root(pool_refs, np.array([
                np.pad(n, (0, 3 - len(n))) for n in noise
            ])[:, :max(len(p) for p in priors)], [0, 0, 0]).cpu().tolist()
            self.assertEqual(chosen, expected)
            self.assertEqual(probe.owned('item'), [])
            self.assertEqual(len(probe.owned('to_cpu', 'cpu')), 1)
            interior = resolve(stats, 'interior_choice', *[(ref,) for ref in refs])
            self.assertEqual(interior, pool.select_interior(pool_refs).cpu().tolist())
            accepted = resolve(stats, 'backup', *[( [(ref, int(action))], 0.4, None) for ref, action in zip(refs, chosen)])
            self.assertEqual(accepted, [True, True, True])
            pool.backup([[(ref, int(action))] for ref, action in zip(pool_refs, expected)], [0.4, 0.4, 0.4])
            before = stats.stats()['host_materializations']
            with HostTransferProbe() as snap_probe:
                snaps = resolve(stats, 'snapshot', *[(ref,) for ref in refs])
            self.assertEqual(stats.stats()['host_materializations'], before + 1)
            self.assertEqual(len(snap_probe.owned('to_cpu', 'cpu')), 1)
            self.assertEqual(snap_probe.owned('item'), [])
            pool_snap = pool.snapshot(pool_refs)
            for i, snap in enumerate(snaps):
                n = len(priors[i])
                np.testing.assert_allclose(snap['prior'], pool_snap['prior'][i, :n].cpu().numpy())
                np.testing.assert_array_equal(snap['visits'], pool_snap['visits'][i, :n].cpu().numpy())
                np.testing.assert_allclose(snap['total'], pool_snap['total'][i, :n].cpu().numpy())
                self.assertEqual(snap['raw_value'], float(pool_snap['value'][i].cpu()))
                self.assertEqual(snap['visits'].dtype, np.int32)
            summary = resolve(stats, 'final_summary', *[(ref, n) for ref, n in zip(refs, noise)])
            rows = pool.final_summary(pool_refs, np.array([
                np.pad(n, (0, max(len(p) for p in priors) - len(n))) for n in noise
            ])).cpu().tolist()
            self.assertEqual(summary, [dict(search_choice=int(row[0]), roots_covered=int(row[1])) for row in rows])
            resolve(stats, 'root_choice', *[(ref, n, 1) for ref, n in zip(refs, noise)])
            self.assertEqual(stats.stats()['workspace_allocations'], 1)

    def test_gumbel_ties_visits_and_stale_nodes_stay_exact(self):
        prior = np.array([0.5, 0.5])
        with BatchedStatistics(4, 2, device='cpu') as stats:
            owners = [stats.new_owner(), stats.new_owner()]
            refs = resolve(stats, 'create', (prior, 0.0, owners[0].id), (prior, 0.0, owners[1].id))
            zero = np.zeros(2, dtype=np.float64)
            prefer_second = np.array([0.0, 8.0], dtype=np.float64)
            tied = resolve(stats, 'root_choice', (refs[0], zero, 0), (refs[1], prefer_second, 0))
            self.assertEqual(tied, [0, 1])
            resolve(stats, 'backup', ([(refs[0], 0)], 1.0, None), ([(refs[1], 1)], -0.25, None))
            snaps = resolve(stats, 'snapshot', (refs[0],), (refs[1],))
            np.testing.assert_array_equal(snaps[0]['visits'], [1, 0])
            np.testing.assert_array_equal(snaps[1]['visits'], [0, 1])
            np.testing.assert_allclose(snaps[0]['total'], [1.0, 0.0])
            np.testing.assert_allclose(snaps[1]['total'], [0.0, -0.25])
            after = resolve(stats, 'root_choice', (refs[0], zero, 0), (refs[1], prefer_second, 0))
            self.assertEqual(after, [1, 0])
            stale = replace(refs[0], generation=refs[0].generation + 1)
            with self.assertRaises(ValueError):
                resolve(stats, 'snapshot', (stale,))

    def test_cuda_host_sync_reduction_pending(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA tests pending: no actual CUDA device in this environment')
        model = SyntheticNumericModel('cuda-sync', hidden=16)
        runner = BatchedInference({'cuda-sync': model}, device='cuda')
        x = make_legal_observation(1)
        requests = [
            InferenceRequest(f'c{i}', 'cuda-sync', model.version, x, make_legal_candidates(2, seed=i), True)
            for i in range(4)
        ]
        results = runner.predict(requests)
        self.assertEqual(runner.stats()['host_materializations'], 1)
        for req, res in zip(requests, results):
            np.testing.assert_allclose(res.logits, model.scores(req.x, req.candidates), atol=1e-4)
        with BatchedStatistics(4, 3, device='cuda') as stats:
            owner = stats.new_owner()
            refs = resolve(stats, 'create', ([0.5, 0.5], 0.0, owner.id), ([1.0], 0.0, owner.id))
            before = stats.stats()['host_materializations']
            resolve(stats, 'snapshot', *[(ref,) for ref in refs])
            self.assertEqual(stats.stats()['host_materializations'] - before, 1)


if __name__ == '__main__':
    unittest.main()
