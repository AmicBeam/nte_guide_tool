import unittest
import numpy as np
from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.engine.duel_v2.state import card_instance
from app.modules.card_game.engine.duel_v2.summons import summon_front
from app.modules.card_game.rl.information_search import search
from app.modules.card_game.rl.murk_lineup import all_features, deck, encode
from app.modules.card_game.rl import murk_runtime as runtime
from app.modules.card_game.engine.duel_v2 import observe


class Zero:
    def scores_only(self, x, c):
        return np.zeros(len(c), dtype=np.float32)

    def wdl(self, x, is_actor):
        return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)

    def scores_value(self, x, c):
        return self.scores_only(x, c), 0.0


class MurkEncodingTest(unittest.TestCase):
    def game(self):
        build = deck()
        return new_game(seed=7, first_side='a', skip_mulligan=True, decks={'a': build, 'b': build})

    def test_opening_encode_and_hidden_resample_match(self):
        state = self.game()
        actions, (x, c) = runtime.decision(state, 'a')
        self.assertEqual(x.shape, (len(all_features()),))
        self.assertEqual(c.shape[0], len(actions))
        sampled = runtime.determinize(state, 'a', 19)
        again, (xx, cc) = runtime.decision(sampled, 'a')
        self.assertEqual(actions, again)
        np.testing.assert_array_equal(x, xx)
        np.testing.assert_array_equal(c, cc)

    def test_summon_burn_and_generated_card_stay_encodable(self):
        state = self.game()
        summon_front(state, 'a', name='鬼郎丸', attack=2, hp=4)
        state['sides']['b']['front_debuff']['burn'] = {'stacks': 3, 'left': 2, 'by': 'a'}
        state['sides']['b']['front_debuff']['dots'] = {'nightmare': {'stacks': 5, 'by': 'a'}}
        state['sides']['b']['hand'].append(card_instance(state, 'AF01', 'b'))
        actions, (x, c) = runtime.decision(state, 'a')
        self.assertTrue(np.isfinite(x).all() and np.isfinite(c).all())
        names = all_features()
        self.assertGreater(x[names.index('murk:burn_stacks:1')], 0)
        self.assertGreater(x[names.index('murk:nightmare:1')], 0)
        sampled = runtime.determinize(state, 'a', 4)
        view = observe(state, 'a', include_previews=False)
        after = observe(sampled, 'a', include_previews=False)
        self.assertEqual(actions_of(view), actions_of(after))
        left, left_c = encode(view, actions_of(view))
        right, right_c = encode(after, actions_of(after))
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(left_c, right_c)
        result = search(state, 'a', {'a': Zero(), 'b': Zero()}, seed=3, simulations=2,
                        algorithm='gumbel', gumbel_candidates=4, runtime=runtime, terminal_horizon=1)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['actions']), len(actions))


def actions_of(view):
    return [entry['action'] for entry in view['legal_actions'] if entry['action']['type'] != 'concede']
