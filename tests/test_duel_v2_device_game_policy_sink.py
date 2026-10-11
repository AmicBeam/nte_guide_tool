"""Physical revision binding, independent of replay-policy tape implementation."""
from types import SimpleNamespace
import unittest

from app.modules.card_game.rl.batched_search.device_game_policy_sink import DeviceGamePolicySink


class DeviceGamePolicySinkTest(unittest.TestCase):
    def test_exact_physical_versions_and_root_hash_are_forwarded(self):
        seen = []
        tape = SimpleNamespace(begin_device=lambda *a: seen.append(a),
                               append_device=lambda *a: seen.append(a),
                               finish_device=lambda *a: seen.append(a))
        sink = DeviceGamePolicySink(tape)
        versions, keys = object(), object()
        loop = SimpleNamespace(sink=sink, source_versions_device=versions)
        sink.bind_game_loop(loop)
        result = SimpleNamespace(information_keys=keys)
        args = tuple(object() for _ in range(4))
        sink.append_device(*args[:3], result, args[3])
        self.assertEqual(seen[0], (*args[:3], versions, result, keys, args[3]))
        with self.assertRaises(ValueError):
            sink.bind_game_loop(loop)
        with self.assertRaises(ValueError):
            sink.append_device(*args[:3], SimpleNamespace(information_keys=None), args[3])

    def test_unbound_capture_and_foreign_loop_are_rejected(self):
        tape = SimpleNamespace(begin_device=lambda *a: None,
                               append_device=lambda *a: None,
                               finish_device=lambda *a: None)
        sink = DeviceGamePolicySink(tape)
        with self.assertRaises(RuntimeError):
            sink.begin_device(None, None, None, None, None, None)
        with self.assertRaises(ValueError):
            sink.bind_game_loop(SimpleNamespace(sink=object()))
        self.assertFalse(sink.supports_full_duel_rules)

    def _complete_fixture(self, device):
        import torch
        from tests.test_duel_v2_device_games import DeviceGameLoopTest, FixtureWorldSeeds
        from app.modules.card_game.rl.batched_search.device_games import DeviceGameLoop
        from app.modules.card_game.rl.batched_search.device_replay_policy_tape import DeviceReplayPolicyTape
        helper = DeviceGameLoopTest()
        base, fixture, _, roots, seeds, ids, train = helper.make(
            rows=3, device=device, simulations=32, actions=20, active=[True, False, True])
        tape = DeviceReplayPolicyTape(3, 8, 160, 20, 100000,
            device=str(fixture.device), allow_cpu_reference=device == 'cpu',
            allow_synthetic=True)
        sink = DeviceGamePolicySink(tape)
        loop = DeviceGameLoop(fixture, base.search, sink, FixtureWorldSeeds(),
            max_decisions=4, device=str(fixture.device),
            allow_cpu_reference=device == 'cpu', allow_synthetic=device == 'cpu',
            inner_max_ticks=2000, seconds=60)
        self.addCleanup(loop.close)
        sink.bind_game_loop(loop)
        if device == 'cpu':
            result = helper.finish_cpu(loop, roots, seeds, ids, train)
        else:
            loop.capture(roots, seeds, ids, ids, train)
            loop.replay()
            torch.cuda.synchronize()
            result = loop.result_device()
        self.assertEqual(result.errors.cpu().tolist(), [0, 0, 0])
        self.assertEqual(result.complete.cpu().tolist(), [True, False, True])
        records = tape.records_after_batch()
        self.assertEqual(len(records), 6)
        self.assertEqual(sorted((r['game'], r['step'], r['version']) for r in records),
                         [(g, s, s) for g in (0, 2) for s in range(3)])
        for record in records:
            self.assertEqual(record['counts'], 20)
            self.assertEqual(record['requested'], 32)
            self.assertEqual(record['completed'], 32)
            self.assertEqual(int(record['visits'].sum()), 32)
            self.assertEqual(record['information_keys'].shape, (8,))
            self.assertFalse(record['learning_ready'])
            self.assertFalse(record['replay_verified'])

    def test_full_gumbel_fixture_keeps_all_legal_pi_and_physical_versions(self):
        self._complete_fixture('cpu')

    def test_actual_cuda_nested_gumbel_policy_sink(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; zero formal NTE games')
        self._complete_fixture('cuda')
