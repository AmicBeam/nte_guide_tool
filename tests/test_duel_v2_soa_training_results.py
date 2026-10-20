"""CPU postbatch verification fixtures; never GPU or training qualification."""
from copy import deepcopy
import pickle
import time
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.rl.batched_search.soa_training_jobs import adapt_training_jobs
from app.modules.card_game.rl.batched_search.soa_training_results import verified_training_games
from tests.test_duel_v2_soa_training_jobs import jobs, models


def precheck(bank):
    from app.modules.card_game.rl.seven_lineup import identity
    return dict(schema='seven_value_precheck_v1', complete=True, passed=True, heldout=True,
        rule_hash=identity(), by_version={model.version: dict(version=model.version,
            metrics=dict(normal_games=8, brier=.2, log_loss=.4)) for model in bank.values()})


def observe_stub(state, side, *, include_previews):
    assert include_previews is False
    return dict(version=state['version'], viewer_side=side, phase=state['phase'],
        legal_actions=[dict(action=dict(type='attack', target=0)), dict(action=dict(type='end_turn')),
                       dict(action=dict(type='concede'))])


def actor_stub(state):
    return state['active_side']


def apply_stub(state, side, action):
    result = deepcopy(state)
    result['version'] += 1
    result['rng'] += 1
    result['active_side'] = 'b' if side == 'a' else 'a'
    if result['version'] == 2:
        result.update(phase='finished', winner=result['first_side'])
    return result


def clean_stub(state):
    return state


def encode_stub(view, actions):
    x = np.asarray([view['version'], int(view['viewer_side'] == 'b')], dtype=np.float32)
    c = np.asarray([[index, view['version']] for index, _ in enumerate(actions)], dtype=np.float32).reshape(len(actions), 2)
    return x, c


STUB = (observe_stub, actor_stub, apply_stub, clean_stub, encode_stub)


def dataset_from_component_states(metadata, bank, initial, oracle, steps):
    """Make explicit CPU fixture rows, not an execution or missing-data fallback."""
    observe, actor, apply, clean, encode = oracle
    policies, values, records, references, final = [], [], [], {}, []
    for owner, start in enumerate(initial):
        state = deepcopy(start)
        winner = metadata.jobs[owner]['first']
        for step in range(steps):
            side = actor(state)
            view = observe(state, side, include_previews=False)
            legal = [v['action'] for v in view['legal_actions'] if v['action']['type'] != 'concede']
            chosen = next(i for i, action in enumerate(legal) if action['type'] == 'attack')
            x, c = encode(view, legal)
            budget = 1 if len(legal) == 1 else 32
            pi = np.full(len(legal), 1/len(legal), np.float32)
            visits = np.zeros(len(legal), np.int32); visits[chosen] = budget
            train_sides = metadata.jobs[owner]['train_sides']
            policies.append(dict(owner=owner, step=step, actor=int(side == 'b'), x=x, c=c, pi=pi,
                visits=visits, visit_pi=visits.astype(np.float32)/budget, selected=chosen,
                version=state['version'], job_seed=metadata.jobs[owner]['seed'], requested=budget, completed=budget,
                encode_error=0, learning_eligible=side in train_sides, z=1. if side == winner else -1.))
            for viewer in ('a', 'b'):
                vx, _ = encode(view if viewer == side else observe(state, viewer, include_previews=False), [])
                values.append(dict(owner=owner, step=step, actor=int(viewer == 'b'), x=vx,
                    value_actor=int(viewer == side), version=state['version'], job_seed=metadata.jobs[owner]['seed'],
                    learning_eligible=viewer in train_sides, z=1. if viewer == winner else -1.))
            records.append(dict(game=owner, step=step, actor=side, raw_action=deepcopy(legal[chosen]),
                                chosen=chosen, requested=budget, completed=budget))
            references[owner, step] = dict(actions=deepcopy(legal), pi=pi.copy(), visits=visits.copy())
            state = clean(apply(state, side, legal[chosen]))
        final.append(state)
    dataset = dict(complete=True, policy=policies, values=values, optimizer_steps=0,
                   search=deepcopy(metadata.search), model_versions=deepcopy(list(metadata.model_versions)))
    return dict(dataset=dataset, initial=initial, final=final, raw_records=records,
                metadata=metadata, models=bank, policy_reference=references)


def fixture(q_scale='natural_wdl'):
    bank = models()
    submitted = jobs()[:2]
    submitted[0]['train_sides'] = ['a']
    for job in submitted:
        job['value_precheck'] = precheck(bank)
        job['q_scale'] = q_scale
    metadata = adapt_training_jobs(submitted, bank)
    initial = []
    for job in metadata.jobs:
        shared = ['one-shared-object']
        initial.append(dict(version=0, phase='playing', active_side=job['first'], first_side=job['first'],
            winner=None, reason=None, turn=1, rng=job['seed'],
            sides={s: dict(order=deepcopy(job['decks'][s]['character_ids'])) for s in ('a', 'b')},
            alias_left=shared, alias_right=shared))
    return dataset_from_component_states(metadata, bank, initial, STUB, 2)


class SoATrainingResultsTests(unittest.TestCase):
    def test_policy_source_records_the_actual_q_profile(self):
        for profile in ('natural_wdl', 'mctx_default'):
            games = self.checked(fixture(profile))
            self.assertTrue(all(row['q_scale'] == profile and row['policy_source'] == 'gumbel_q_'+profile
                                for game in games for row in game['rows']))

    def checked(self, arguments):
        with patch('app.modules.card_game.rl.batched_search.soa_training_results._oracle', return_value=STUB):
            return verified_training_games(**arguments)

    def test_original_job_identity_single_train_side_labels_and_no_input_mutation(self):
        arguments = fixture()
        before = pickle.dumps(arguments)
        games = self.checked(arguments)
        self.assertEqual(before, pickle.dumps(arguments))
        for owner, game in enumerate(games):
            job = arguments['metadata'].jobs[owner]
            for key in ('id', 'seed', 'first', 'left', 'right'):
                self.assertEqual(game[key], job[key])
            self.assertEqual(game['job'], job)
            self.assertEqual(game['decisions'], 2)
            self.assertEqual(game['searched'], 2)
            self.assertEqual(len(game['value_rows']), 4)
            self.assertEqual(game['rule_backend'], 'cuda_soa')
            self.assertTrue(game['learning_eligible'])
            self.assertFalse(game['formal_training_qualified'])
            self.assertTrue(all(row['teacher_mode'] == 'network' and row['terminal_horizon'] == 3 for row in game['rows']))
        self.assertEqual(games[0]['job']['train_sides'], ['a'])
        games[0]['rows'][0]['pi'][0] = 100
        self.assertEqual(before, pickle.dumps(arguments))

    def test_missing_duplicate_noncontinuous_actor_order_and_budget_rows_rejected(self):
        def missing_value(a): a['dataset']['values'].pop()
        def duplicate_policy(a): a['dataset']['policy'].append(deepcopy(a['dataset']['policy'][0]))
        def duplicate_value(a): a['dataset']['values'].append(deepcopy(a['dataset']['values'][0]))
        def duplicate_raw(a): a['raw_records'].append(deepcopy(a['raw_records'][0]))
        def reorder(a): a['raw_records'][0], a['raw_records'][1] = a['raw_records'][1], a['raw_records'][0]
        def legal_order(a): a['policy_reference'][0, 0]['actions'].reverse()
        def actor(a): a['raw_records'][0]['actor'] = 'b'
        def bad_budget(a): a['dataset']['policy'][0]['completed'] = 31
        def seed(a): a['dataset']['values'][0]['job_seed'] += 1
        def version(a): a['dataset']['policy'][0]['version'] += 1
        def coordinates(a): a['dataset']['policy'][0]['owner'] = True
        for mutate in (missing_value, duplicate_policy, duplicate_value, duplicate_raw, reorder,
                       legal_order, actor, bad_budget, seed, version, coordinates):
            arguments = fixture(); mutate(arguments)
            with self.subTest(mutate=mutate.__name__), self.assertRaises(ValueError):
                self.checked(arguments)

    def test_numerical_targets_viewer_final_rng_alias_and_profile_errors_rejected(self):
        def pi(a): a['dataset']['policy'][0]['pi'][0] = np.nan
        def reference(a): a['policy_reference'][0, 0]['pi'][0] += .1
        def visits(a): a['dataset']['policy'][0]['visits'][0] -= 1
        def x(a): a['dataset']['policy'][0]['x'][0] += 1
        def c(a): a['dataset']['policy'][0]['c'][0, 0] += 1
        def viewer(a): a['dataset']['values'][1]['value_actor'] = 1
        def z(a): a['dataset']['values'][1]['z'] = 0
        def encoding(a): a['dataset']['policy'][0]['encode_error'] = 4
        def rng(a): a['final'][0]['rng'] += 1
        def alias(a): a['final'][0]['alias_right'] = list(a['final'][0]['alias_left'])
        def unfinished(a): a['final'][0]['phase'] = 'playing'
        def h0(a): a['dataset']['search']['terminal_horizon'] = 0
        def no_profile(a): del a['dataset']['search']
        def no_actions(a): del a['policy_reference'][0, 0]['actions']
        def model(a): a['dataset']['model_versions'][0]['a'] = '0'*64
        for mutate in (pi, reference, visits, x, c, viewer, z, encoding, rng, alias,
                       unfinished, h0, no_profile, no_actions, model):
            arguments = fixture(); mutate(arguments)
            with self.subTest(mutate=mutate.__name__), self.assertRaises(ValueError):
                self.checked(arguments)

    def test_calibration_missing_or_stale_and_pure_or_missing_dataset_have_no_fallback(self):
        for field, value in (('value_precheck', None), ('value_precheck', dict(passed=True)), ('pure', True)):
            arguments = fixture()
            arguments['metadata'].jobs[0][field] = value
            with self.assertRaises(ValueError): self.checked(arguments)
        arguments = fixture()
        for entry in arguments['metadata'].jobs[0]['value_precheck']['by_version'].values():
            entry['version'] = '0'*64
        with self.assertRaises(ValueError): self.checked(arguments)
        for dataset in (None, {}, dict(complete=False)):
            arguments = fixture(); arguments['dataset'] = dataset
            with self.assertRaises(ValueError): self.checked(arguments)
        arguments = fixture()
        alias = arguments['metadata'].inference_keys[arguments['metadata'].model_indices[0][0]]
        arguments['models'][alias].version = '0'*64
        arguments['models'][alias].manifest['sha256'] = '0'*64
        with self.assertRaises(ValueError): self.checked(arguments)

    def test_proof_deadline_is_atomic_and_inputs_unchanged_after_mid_replay_failure(self):
        arguments = fixture(); before = pickle.dumps(arguments)
        with self.assertRaises(TimeoutError):
            self.checked(dict(arguments, deadline=time.monotonic()-1))
        with patch('app.modules.card_game.rl.batched_search.soa_training_results.time.monotonic',
                   side_effect=[0, 0, 0, 10]), self.assertRaises(TimeoutError):
            self.checked(dict(arguments, deadline=5))
        self.assertEqual(before, pickle.dumps(arguments))

    def test_nontraining_jobs_remain_ineligible_and_error_receipts_refuse_success(self):
        arguments = fixture()
        submitted = deepcopy(list(arguments['metadata'].jobs))
        for job in submitted:
            job['training'] = False
            job['value_precheck'] = None
        arguments['metadata'] = adapt_training_jobs(submitted, arguments['models'])
        for row in arguments['dataset']['policy'] + arguments['dataset']['values']:
            row['learning_eligible'] = False
        self.assertTrue(all(not game['learning_eligible'] for game in self.checked(arguments)))
        for change in (dict(error_rows=1), dict(optimizer_steps=1), dict(optimizer_steps=False),
                       dict(truncated=True), dict(truncated=0)):
            bad = fixture(); bad['dataset'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.checked(bad)

    def test_real_one_attack_natural_terminal_component_with_canonical_encoder_and_aliases(self):
        from app.modules.card_game.engine.duel_v2 import new_game, observe, acting_side, apply_action
        from app.modules.card_game.rl.league_rollout import clean
        from app.modules.card_game.rl.seven_lineup import encode
        bank = models(); submitted = jobs()[:1]
        submitted[0]['value_precheck'] = precheck(bank)
        metadata = adapt_training_jobs(submitted, bank)
        job = metadata.jobs[0]
        state = new_game(seed=job['seed'], first_side=job['first'], decks=job['decks'], skip_mulligan=True)
        state['sides']['b']['hp'], state['sides']['b']['shield'] = 1, 0
        arguments = dataset_from_component_states(metadata, bank, [state],
            (observe, acting_side, apply_action, clean, encode), 1)
        self.assertEqual(arguments['final'][0]['phase'], 'finished')
        before = pickle.dumps(arguments)
        games = verified_training_games(**arguments)
        self.assertTrue(games[0]['complete']); self.assertEqual(games[0]['decisions'], 1)
        self.assertEqual(before, pickle.dumps(arguments))
        # Near-terminal component input is explicit; this is not an initializer,
        # complete-GPU-game, capacity, driver or formal-learning qualification.


if __name__ == '__main__': unittest.main()
