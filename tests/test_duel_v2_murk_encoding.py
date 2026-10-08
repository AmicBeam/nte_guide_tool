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
    def test_legacy_dot_counter_encoding_is_invariant_to_entity_id_renaming(self):
        from copy import deepcopy
        from app.modules.card_game.rl import cross_lineup
        state = self.game()
        # Frozen encoders still accept counters from old snapshots; current
        # Canhong no longer creates this field or gains attack from DoTs.
        cid = state['sides']['a']['characters']['canhong']['entity_id']
        state['sides']['a']['used'][f'canhong_dot_attack_turn:{cid}'] = state['turn']
        for viewer in ('a', 'b'):
            view = observe(state, viewer, include_previews=False)
            renamed = deepcopy(view)
            for team in renamed['sides'].values():
                for h in team['characters']:
                    old = h['entity_id']; h['entity_id'] = old + '-renamed'
                    key = f'canhong_dot_attack_turn:{old}'
                    if key in team.get('used', {}):
                        team['used'][key + '-renamed'] = team['used'].pop(key)
            for contract in (runtime, cross_lineup):
                encoded, _ = contract.encode(view, [])
                again, _ = contract.encode(renamed, [])
                np.testing.assert_array_equal(encoded, again)
                names = all_features() if contract is runtime else cross_lineup.all_features()
                own = 0 if viewer == 'a' else 1
                self.assertEqual(encoded[names.index(f'murk:dot_attack_used:{own}')], 1)
                self.assertEqual(encoded[names.index(f'murk:dot_attack_used:{1-own}')], 0)

    def test_current_dots_leave_retired_attack_counter_zero_in_both_contracts(self):
        from app.modules.card_game.engine.duel_v2.continuous import apply_dot
        from app.modules.card_game.rl import cross_lineup
        state = self.game()
        for kind in ('etch', 'venom'):
            apply_dot(state, 'b', kind, by='a', source_cid='canhong', stacks=1)
        for viewer in ('a', 'b'):
            view = observe(state, viewer, include_previews=False)
            for contract in (runtime, cross_lineup):
                encoded, _ = contract.encode(view, [])
                names = all_features() if contract is runtime else cross_lineup.all_features()
                for side in (0, 1):
                    self.assertEqual(encoded[names.index(f'murk:dot_attack_used:{side}')], 0)

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
