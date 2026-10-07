"""Lifecycle and cancellation invariants for independent batch-search roots."""
import unittest
from dataclasses import replace
from app.modules.card_game.rl.batched_search import BudgetExhausted, DecisionBudget, SearchLimits


class Clock:
    def __init__(self): self.now = 10.
    def __call__(self): return self.now


class BatchBudgetTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.budget = DecisionBudget('game-1', 1, clock=self.clock)

    def run_one(self, kind='tree'):
        ticket = self.budget.reserve(kind)
        self.budget.mark_started(ticket)
        self.assertTrue(self.budget.complete(ticket))
        return ticket

    def test_proof_and_tree_share_exact_total_allowance(self):
        for _ in range(8): self.run_one('proof')
        with self.assertRaises(BudgetExhausted): self.budget.reserve('proof')
        for _ in range(24): self.run_one()
        with self.assertRaises(BudgetExhausted): self.budget.reserve('tree')
        snap = self.budget.snapshot()
        self.assertEqual(snap['reserved'], {'proof': 8, 'tree': 24})
        self.assertEqual(snap['started'], snap['completed'])

    def test_single_outstanding_simulation_preserves_visit_dependency(self):
        ticket = self.budget.reserve('tree')
        with self.assertRaises(RuntimeError): self.budget.reserve('tree')
        self.budget.abort(ticket)
        self.run_one()
        self.assertEqual(self.budget.snapshot()['reserved']['tree'], 2)

    def test_queue_wait_consumes_original_three_seconds(self):
        ticket = self.budget.reserve('tree')
        self.clock.now = 13.
        with self.assertRaises(BudgetExhausted): self.budget.mark_started(ticket)
        self.assertEqual(self.budget.snapshot()['started']['tree'], 0)
        self.budget.abort(ticket)
        with self.assertRaises(BudgetExhausted): self.budget.reserve('tree')

    def test_late_gpu_result_is_not_a_completed_simulation(self):
        ticket = self.budget.reserve('tree'); self.budget.mark_started(ticket)
        self.clock.now = 13.
        self.assertFalse(self.budget.complete(ticket))
        snap = self.budget.snapshot()
        self.assertEqual(snap['completed']['tree'], 0)
        self.assertEqual(snap['discarded'], 1)
        with self.assertRaises(ValueError): self.budget.complete(ticket)

    def test_close_rejects_inflight_result_and_future_reservations(self):
        ticket = self.budget.reserve('tree'); self.budget.mark_started(ticket)
        self.budget.close()
        self.assertFalse(self.budget.complete(ticket))
        with self.assertRaises(BudgetExhausted): self.budget.reserve('tree')
        self.budget.close()  # Closing twice cannot reopen it.

    def test_cross_root_and_altered_ticket_rejected(self):
        ticket = self.budget.reserve('tree')
        other = DecisionBudget('game-1', 2, clock=self.clock)
        for invalid in (ticket, replace(ticket, budget_id='fake')):
            with self.assertRaises(ValueError): other.mark_started(invalid)
        with self.assertRaises(ValueError): self.budget.mark_started(replace(ticket, kind='proof'))
        for altered in (replace(ticket, sequence=False), replace(ticket, sequence=0.),
                        replace(ticket, root_generation=True)):
            with self.subTest(ticket=altered), self.assertRaises(ValueError):
                self.budget.mark_started(altered)
        self.budget.mark_started(ticket)
        self.assertTrue(self.budget.complete(ticket))

    def test_duplicate_start_and_unstarted_completion_rejected(self):
        ticket = self.budget.reserve('tree')
        with self.assertRaises(ValueError): self.budget.complete(ticket)
        self.budget.mark_started(ticket)
        with self.assertRaises(ValueError): self.budget.mark_started(ticket)
        self.budget.abort(ticket)
        with self.assertRaises(ValueError): self.budget.abort(ticket)

    def test_aborted_reservation_not_refunded(self):
        limits = SearchLimits(simulations=2, proof_cap=0)
        budget = DecisionBudget('g', 1, clock=self.clock, limits=limits)
        for _ in range(2): budget.abort(budget.reserve('tree'))
        with self.assertRaises(BudgetExhausted): budget.reserve('tree')
        self.assertEqual(budget.snapshot()['started']['tree'], 0)

    def test_proof_phase_does_not_resume_after_tree(self):
        self.run_one()
        with self.assertRaises(RuntimeError): self.budget.reserve('proof')

    def test_limits_and_monotonic_clock_fail_explicitly(self):
        for options in ({'seconds':True}, {'seconds':float('nan')}, {'seconds':0},
                        {'simulations':True}, {'simulations':0}, {'proof_cap':32},
                        {'proof_cap':-1}, {'terminal_horizon':False}, {'max_depth':0}):
            with self.subTest(options=options), self.assertRaises(ValueError): SearchLimits(**options)
        self.clock.now = 9.
        with self.assertRaises(ValueError): self.budget.snapshot()
        for game, generation in (('',1), ('g',0), ('g',True)):
            with self.assertRaises(ValueError): DecisionBudget(game,generation,clock=Clock())

    def test_depth_and_terminal_probe_have_separate_limits(self):
        limits = self.budget.limits
        self.assertEqual((limits.max_depth, limits.terminal_horizon), (10, 3))
        self.assertEqual((limits.seconds, limits.simulations, limits.gumbel_candidates), (3., 32, 16))


if __name__ == '__main__': unittest.main()
