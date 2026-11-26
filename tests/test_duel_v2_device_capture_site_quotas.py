"""Unused capture sites allocate nothing and cannot be silently invoked."""
import unittest
from app.modules.card_game.rl.batched_search.device_frozen_rules import (
    _positive_sites, _PlanQueue, DEFAULT_SITES)
from app.modules.card_game.rl.batched_search.device_teacher_pipeline import _teacher_capture_sites


class CaptureSiteQuotaTest(unittest.TestCase):
    def test_default_preserved_and_disabled_queue_rejects_without_consuming(self):
        self.assertEqual(_positive_sites(None), DEFAULT_SITES)
        configured = _positive_sites({'rule_decision': 0, 'value': 3})
        self.assertEqual(configured['metadata'], DEFAULT_SITES['metadata'])
        queue = _PlanQueue('rule_decision', [object() for _ in range(configured['rule_decision'])])
        with self.assertRaisesRegex(RuntimeError, 'No remaining prepared rule_decision'):
            queue.consume()
        self.assertEqual(queue.index, 0)
        with self.assertRaises(ValueError): _positive_sites({'typo': 0})
        for value in (-1, True, False, 0.0, None):
            with self.assertRaises(ValueError): _positive_sites({'decision': value})

    def test_teacher_composition_reserves_all_three_steps_and_choice_backend(self):
        for raw in (False, True):
            sites = _positive_sites(_teacher_capture_sites(raw))
            for name in ('metadata', 'simulate_action', 'clean'):
                queue = _PlanQueue(name, [object() for _ in range(sites[name])])
                for _ in range(3): queue.consume()
                with self.assertRaises(RuntimeError): queue.consume()
            self.assertEqual(sites['rule_decision'], 0 if raw else 3)
            for name in ('decision', 'value', 'sample_world', 'apply_action', 'teacher_continue'):
                self.assertEqual(sites[name], 0)
        for value in (0, 1, None):
            with self.assertRaises(TypeError): _teacher_capture_sites(value)


if __name__ == '__main__': unittest.main()
