import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from app.modules.card_game.rl.information_search import Node
from app.modules.card_game.rl.recovery_search import search, choose_root, completed_q
from app.modules.card_game.rl.root_recommendation import sweeps, recommend, MODES
from app.modules.card_game.rl.search_policy import visit_schedule


class Runtime:
    @staticmethod
    def decision(state, side):
        return [{'type': 'wait'}, {'type': 'finish'}], (np.array([0.], np.float32), np.eye(2, dtype=np.float32))

    @staticmethod
    def determinize(state, viewer, seed):
        return deepcopy(state)

    @staticmethod
    def predict_encoded(policy, x, c):
        return np.array([6., 0.]), 0.


class RootRecommendationTest(unittest.TestCase):
    def run_search(self, budget=13, trace=True, act=None):
        if act is None:
            act = lambda s, side, a: dict(s, phase='finished', winner='b' if a['type'] == 'wait' else 'a')
        with patch('app.modules.card_game.rl.information_search.apply_action', side_effect=act):
            return search(dict(phase='playing', active_side='a', turn=1), 'a', {'a': None, 'b': None},
                          seed=9, runtime=Runtime, simulations=budget, root_trace=trace,
                          fast_simulation=False, terminal_horizon=0)

    def test_sweeps_reproduce_reference_prefix_including_odd_widths(self):
        for actions in range(1, 33):
            for budget in range(actions, 65):
                passes = sweeps(actions, budget)
                sequence = [p['visit'] for p in passes for _ in range(p['width'])]
                self.assertEqual(sequence[:budget], visit_schedule(actions, budget))

    def test_uneven_tail_cannot_eliminate_last_completed_sweep_rival(self):
        r = self.run_search()
        # Visits 7 vs 6; simulate the real low-prior action's nonterminal Q gap.
        r['root_prior'] = np.array([.9978, .0022])
        r['mean_values'] = np.array([-.275, .208])
        r['root_diagnostics']['completed_q'] = r['mean_values'] * 5.7
        r['visits'] = np.array([7, 6]); r['search_choice'] = 0
        r['root_trace'] = [dict(chosen=i % 2, considered_visit=i // 2) for i in range(13)]
        self.assertEqual(recommend(r, 'legacy')[0], 0)
        self.assertEqual(recommend(r, 'completed_sweep_gumbel')[0], 0)
        chosen, audit = recommend(r, 'completed_sweep_value')
        self.assertEqual(chosen, 1)
        self.assertEqual(set(audit['eligible']), {0, 1})

    def test_even_budget_can_cut_a_pair_after_an_odd_halving_width(self):
        passes = sweeps(10, 24)
        self.assertEqual(passes[0]['width'], 10)
        self.assertEqual(passes[1]['width'], 5)
        tail = next(p for p in passes if p['start'] < 24 < p['stop'])
        self.assertEqual(tail, dict(visit=6, start=23, stop=25, width=2))
        self.assertEqual(passes[passes.index(tail) - 1]['stop'], 23)

    def test_single_sample_outlier_does_not_override_prior(self):
        r = self.run_search(budget=2)
        self.assertEqual(r['visits'].tolist(), [1, 1])
        r['mean_values'] = np.array([-.1, .1])
        r['root_diagnostics']['completed_q'] = r['mean_values'] * 5.1
        r['search_choice'] = 0
        self.assertEqual(recommend(r, 'completed_sweep_value')[0], 0)
        self.assertEqual(recommend(r, 'completed_sweep_value')[1]['fallback'],
                         'fewer_than_two_visits_per_finalist')

    def test_trace_changes_neither_allocation_nor_original_target(self):
        plain = self.run_search(trace=False)
        traced = self.run_search()
        for field in ('pi', 'visits', 'mean_values', 'root_prior', 'root_gumbel'):
            np.testing.assert_array_equal(plain[field], traced[field])
        self.assertEqual(plain['search_choice'], traced['search_choice'])
        self.assertNotIn('root_trace', plain)
        self.assertEqual(len(traced['root_trace']), traced['simulations'])
        for mode in MODES:
            self.assertIn(recommend(traced, mode)[0], (0, 1))

    def test_interrupted_simulation_is_not_evidence(self):
        from app.modules.card_game.rl import recovery_search
        clock = [0.]
        def act(s, side, a):
            clock[0] = 2.
            return dict(s, phase='playing')
        with patch.object(recovery_search.time, 'time', side_effect=lambda: clock[0]), \
                patch('app.modules.card_game.rl.information_search.apply_action', side_effect=act):
            r = search(dict(phase='playing', active_side='a', turn=1), 'a', {'a': None, 'b': None},
                       seed=9, runtime=Runtime, simulations=13, root_trace=True,
                       fast_simulation=False, terminal_horizon=0, deadline=1.)
        self.assertEqual(r['simulations'], 0)
        self.assertEqual(r['root_trace'], [])
        chosen, audit = recommend(r, 'completed_sweep_value')
        self.assertEqual(chosen, r['search_choice'])
        self.assertEqual(audit['fallback'], 'no_completed_sweep')

    def test_missing_trace_and_invalid_statistics_fail(self):
        r = self.run_search(trace=False)
        with self.assertRaises(ValueError): recommend(r, 'completed_sweep_value')
        r = self.run_search(); r['root_trace'].pop()
        with self.assertRaises(ValueError): recommend(r, 'completed_sweep_value')
        r = self.run_search(); r['mean_values'][0] = np.nan
        with self.assertRaises(ValueError): recommend(r)

    def test_recommendation_does_not_mutate_search_or_train_target(self):
        r = self.run_search(); before = deepcopy(r)
        for mode in MODES: recommend(r, mode)
        for field in ('pi', 'visits', 'mean_values', 'root_prior', 'root_gumbel'):
            np.testing.assert_array_equal(r[field], before[field])
        self.assertEqual(r['search_choice'], before['search_choice'])
        self.assertEqual(r['root_trace'], before['root_trace'])

    def test_final_selection_matches_mctx_max_visit_semantics(self):
        node = Node.create(np.array([.8, .2]), 0.)
        node.visits[:] = [7, 6]; node.total[:] = [-2., 2.]
        self.assertEqual(choose_root(node, np.zeros(2), 7, 'natural_wdl'), 0)
        self.assertGreater(completed_q(node, 'natural_wdl')[1], 0)


if __name__ == '__main__':
    unittest.main()
