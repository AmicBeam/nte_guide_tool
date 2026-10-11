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
