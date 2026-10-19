"""Real CPU rules and cooperative G32/H3, with lossless process-boundary prefixes."""
from copy import deepcopy
from types import SimpleNamespace
import time
import unittest

import numpy as np

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.rl import seven_runtime
from app.modules.card_game.rl.batched_duel.process_backend import ResidentRuleBackend
from app.modules.card_game.rl.batched_search.inference import InferenceResult
from app.modules.card_game.rl.batched_search.training_rollout import TrainingRollout
from app.modules.card_game.rl.batched_search.traversal import PolicyIdentity
from app.modules.card_game.rl.device_auxiliary_continuation import encode_auxiliary_payload, decode_auxiliary_payload
from app.modules.card_game.rl.seven_lineup import ACTION_TYPES


class FrozenInference:
    def predict(self, requests):
        result = []
        for req in requests:
            logits = np.full(len(req.candidates), -20, dtype=np.float32)
            if len(logits):
                logits[np.asarray(req.candidates)[:, 0] == ACTION_TYPES.index('attack')] = 20
            result.append(InferenceResult(req.request_id, req.model_key, req.model_version,
                logits, np.asarray([.3, .4, .3], dtype=np.float32)))
        return result


def plan():
    deck = deepcopy(next(d for d in STARTER_DECKS if d['id'] == 'starter'))
    return [dict(id='resume-'+str(i), seed=93197+i, first='a' if i == 0 else 'b',
        pure=False, training=True, train_sides=['a','b'], simulations=32,
        gumbel_candidates=16, terminal_horizon=3, max_depth=1, max_actions=20,
        q_scale='natural_wdl', teacher_mode='covered_value', deadline=time.time()+180,
        decks={'a': deck, 'b': deck},
        policies={s: PolicyIdentity('starter','fixture-v1') for s in ('a','b')}) for i in range(2)]


def states(jobs):
    result = []
    for j in jobs:
        state = new_game(seed=j['seed'], decks=j['decks'], first_side=j['first'])
        for side in ('a', 'b'):
            state['sides'][side]['hp'] = 1
            state['sides'][side]['shield'] = 0
        enemy = 'b' if j['first'] == 'a' else 'a'
        for character in state['sides'][enemy]['characters'].values():
            character['hp'] = 0
            character['down_turns'] = 2
        result.append(state)
    return result


class HybridResumeTest(unittest.TestCase):
    def collect(self, jobs, **kwargs):
        model = SimpleNamespace(version='fixture-v1', weights={'probe': np.asarray([1.], dtype=np.float32)})
        bank = {'starter': model}
        override = kwargs.pop('model', None)
        if override is not None:
            bank = {'starter': override}
        with ResidentRuleBackend(workers=1, max_games=len(jobs), runtime=seven_runtime) as backend:
            rollout = TrainingRollout(FrozenInference(), backend, bank, search_roots=len(jobs),
                create_states=kwargs.pop('factory', states), allow_cpu_oracle=True,
                value_models={'starter': model})
            games = rollout.collect(jobs, **kwargs)
            self.assertEqual(backend.stats()['live_states'], 0)
        return games

    def test_split_matches_uninterrupted_and_retains_only_complete_searches(self):
        jobs = plan()
        uninterrupted = self.collect(jobs)
        captures = []
        partial = self.collect(jobs, segment_max_decisions=1, checkpoint_callback=captures.append)
        self.assertTrue(all(g['reason'] == 'segment_pause' and not g['learning_eligible'] for g in partial))
        self.assertTrue(all(g['rows'] == g['value_rows'] == [] for g in partial))
        checkpoint = captures[-1]
        self.assertEqual([r['step'] for r in checkpoint['records'].values()], [1, 1])
        self.assertTrue(all('z' not in row for r in checkpoint['records'].values() for row in r['rows']+r['values']))
        for _ in range(5):
            # Same closed graph used for on-disk process continuation; no pickles.
            checkpoint = decode_auxiliary_payload(encode_auxiliary_payload(checkpoint))
            games = self.collect(jobs, resume_checkpoint=checkpoint,
                segment_max_decisions=1, checkpoint_callback=captures.append,
                factory=lambda _: self.fail('Restored games must not be initialized again'))
            checkpoint = captures[-1]
            if all(g['complete'] for g in games):
                break
        self.assertTrue(all(g['complete'] and g['learning_eligible'] for g in games))
        for left, right in zip(uninterrupted, games):
            for field in ('episode_actions','final_state_sha256','searched','decisions','winner','model_versions'):
                self.assertEqual(left[field], right[field])
            for kind in ('rows', 'value_rows'):
                self.assertEqual(len(left[kind]), len(right[kind]))
                for lrow, rrow in zip(left[kind], right[kind]):
                    self.assertEqual(set(lrow), set(rrow))
                    for field in lrow:
                        if isinstance(lrow[field], np.ndarray):
                            np.testing.assert_array_equal(lrow[field], rrow[field])
                        else:
                            self.assertEqual(lrow[field], rrow[field])

    def test_restore_rejects_identity_changes_corruption_and_no_sink(self):
        jobs = plan()
        captures = []
        self.collect(jobs, segment_deadline=time.time()-1, checkpoint_callback=captures.append)
        checkpoint = captures[-1]
        self.assertEqual([r['step'] for r in checkpoint['records'].values()], [0,0])
        changed = deepcopy(jobs); changed[0]['terminal_horizon'] = 2
        with self.assertRaisesRegex(ValueError, 'profile changed'):
            self.collect(changed, resume_checkpoint=checkpoint, checkpoint_callback=lambda _:None)
        changed_model = SimpleNamespace(version='fixture-v1', weights={'probe':np.asarray([2.],dtype=np.float32)})
        with self.assertRaisesRegex(ValueError, 'profile changed'):
            self.collect(jobs, model=changed_model, resume_checkpoint=checkpoint, checkpoint_callback=lambda _:None)
        damaged = deepcopy(checkpoint); damaged['records']['resume-0']['step'] += 1
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.collect(jobs, resume_checkpoint=damaged, checkpoint_callback=lambda _:None)
        with self.assertRaisesRegex(ValueError, 'callback'):
            self.collect(jobs, segment_max_decisions=1)

    def test_finished_games_survive_later_segments_without_replaying(self):
        jobs = plan()
        def mixed(rows):
            result = states(rows)
            # One genuine game begins after its setup decision; its peer still
            # has both setup decisions to make. Their completion times differ.
            result[0] = new_game(seed=rows[0]['seed'], decks=rows[0]['decks'],
                                 first_side=rows[0]['first'], skip_mulligan=True)
            for side in ('a', 'b'):
                result[0]['sides'][side]['hp'] = 1
                result[0]['sides'][side]['shield'] = 0
            for char in result[0]['sides']['b']['characters'].values():
                char['hp'] = 0; char['down_turns'] = 2
            return result
        captures = []
        games = self.collect(jobs, factory=mixed, segment_max_decisions=1,
                             checkpoint_callback=captures.append)
        self.assertTrue(games[0]['complete'], captures[-1]['records']['resume-0']['trace'])
        self.assertFalse(games[1]['complete'])
        original = deepcopy(games[0])
        for _ in range(4):
            games = self.collect(jobs, resume_checkpoint=captures[-1],
                                 segment_max_decisions=1, checkpoint_callback=captures.append)
            if all(g['complete'] for g in games):
                break
        self.assertTrue(all(g['complete'] for g in games))
        self.assertEqual(original['episode_actions'], games[0]['episode_actions'])
        self.assertEqual(original['final_state_sha256'], games[0]['final_state_sha256'])


if __name__ == '__main__':
    unittest.main()
