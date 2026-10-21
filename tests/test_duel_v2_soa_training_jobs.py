"""Real seven_round job metadata, without CUDA, model loading or collection."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.rl.league_schema import build_hash
from app.modules.card_game.rl.seven_schedule import ROSTER
from app.modules.card_game.rl.seven_policy import SCHEMA
from app.modules.card_game.rl.seven_round import jobs_for, collection_jobs
from app.modules.card_game.rl.batched_search.soa_training_jobs import adapt_training_jobs, policy_source_alias


DECKS = {deck['id']: deck for deck in STARTER_DECKS if deck['id'] in ROSTER}
FOLDER = '/tmp/soa-fixed-models'
HISTORICAL = '/tmp/soa-fixed-history'


def models(*folders):
    bank = {}
    for folder in folders or (FOLDER,):
        for key in ROSTER:
            version = sha256((folder + ':' + key).encode()).hexdigest()
            build = deepcopy(DECKS[key])
            bank[policy_source_alias(folder, key)] = SimpleNamespace(directory=folder,
                version=version, serving_deck=build,
                manifest=dict(schema=SCHEMA, deck=key, sha256=version, build=deepcopy(build), build_sha256=build_hash(build)))
    return bank


def jobs(**kwargs):
    return jobs_for(FOLDER, DECKS, 10_000_000_000, 2_000_000_000,
                    pure=False, cycle=7, **kwargs)


class SoATrainingJobsTests(unittest.TestCase):
    def test_real_generators_preserve_mctx_for_learning_and_selection(self):
        from app.modules.card_game.rl.seven_round import compare_jobs, learning_collection_jobs, capacity_heldout_jobs
        from app.modules.card_game.rl.training_config import COLLECTION_REPLICA_SEED_STRIDE
        generated = jobs(q_scale='mctx_default')
        self.assertTrue(generated and all(j['q_scale']=='mctx_default' for j in generated))
        compared = compare_jobs(FOLDER, HISTORICAL, DECKS, 10_000_000_000, 2_000_000_000, 1,
                                search=True, q_scale='mctx_default')
        self.assertTrue(compared and all(j['q_scale']=='mctx_default' for j in compared))
        config = dict(q_scale='mctx_default', simulations=32, collection_replicas=1,
                      preflight=True, preflight_replicas=2, seeds=dict(foundation=10_000_000_000,probes=11_000_000_000))
        for batch in (learning_collection_jobs(FOLDER, DECKS, config, 1, 2_000_000_000),
                      capacity_heldout_jobs(FOLDER, DECKS, config, 2_000_000_000)):
            self.assertTrue(batch and all(j['q_scale']=='mctx_default' for j in batch))
        with self.assertRaisesRegex(ValueError,'Q profile'):
            jobs(q_scale='unknown')

    def test_q_profiles_bind_explicitly_and_cannot_share_a_batch(self):
        source = jobs(teacher_mode='covered_value', covered_value_source='/tmp/covered')[:2]
        for profile in ('natural_wdl', 'mctx_default'):
            batch = deepcopy(source)
            for job in batch: job['q_scale'] = profile
            result = adapt_training_jobs(batch, models())
            self.assertEqual(result.search['q_scale'], profile)
            self.assertTrue(all(row['search']['q_scale'] == profile for row in result.identities))
            self.assertFalse(result.gpu_qualified)
        source[0]['q_scale'] = 'mctx_default'
        with self.assertRaisesRegex(ValueError, 'Q profile'):
            adapt_training_jobs(source, models())
        for invalid in (None, True, 'unknown'):
            source[0]['q_scale'] = invalid
            with self.assertRaisesRegex(ValueError, 'q_scale'):
                adapt_training_jobs(source, models())

    def test_actual_seven_round_schedule_initializers_and_collection_identities(self):
        original = collection_jobs(FOLDER, DECKS, 10_000_000_000, 2_000_000_000,
                                   pure=False, cycle=7, replicas=2)
        before = deepcopy(original)
        bank = models()
        result = adapt_training_jobs(original, bank)
        self.assertEqual(result.jobs, tuple(original))
        self.assertEqual(result.inference_keys, tuple(sorted(bank)))
        self.assertEqual({job['left'] for job in original} | {job['right'] for job in original}, set(ROSTER))
        for owner, job in enumerate(original):
            initial = result.initializers[owner]
            self.assertEqual(initial, dict(seed=job['seed'], decks=job['decks'], first_side=job['first'],
                                           skip_mulligan=False, escalation=True))
            identity = result.identities[owner]
            for field in ('id', 'seed', 'first', 'left', 'right', 'train_sides', 'deadline', 'collection_replica'):
                self.assertEqual(identity[field], job[field])
            for side, key in (('a', job['left']), ('b', job['right'])):
                index = result.model_indices[owner][int(side == 'b')]
                source = policy_source_alias(*job['policies'][side])
                self.assertEqual(result.inference_keys[index], source)
                self.assertEqual(result.model_versions[owner][side], bank[source].version)
                self.assertEqual(result.keys[owner][side], key)
            self.assertEqual(result.train_masks[owner], tuple(side in job['train_sides'] for side in ('a', 'b')))
        self.assertEqual(result.search['terminal_horizon'], 3)
        self.assertEqual(result.search['teacher_mode'], 'network')
        self.assertEqual(result.search['world_sampling'], 'per_simulation')
        self.assertFalse(result.formal_training_qualified)
        self.assertFalse(result.gpu_qualified)
        result.jobs[0]['decks']['a']['card_ids'].reverse()
        result.initializers[0]['decks']['a']['card_ids'].reverse()
        self.assertEqual(original, before)

    def test_historical_same_key_uses_distinct_source_and_keeps_training_side(self):
        source = jobs(historical=HISTORICAL, keys=('starter',))
        bank = models(FOLDER, HISTORICAL)
        result = adapt_training_jobs(source, bank)
        for owner, job in enumerate(source):
            learner = job['train_sides'][0]
            enemy = 'b' if learner == 'a' else 'a'
            self.assertEqual(result.train_masks[owner], (learner == 'a', learner == 'b'))
            self.assertEqual(result.identities[owner]['policy_bindings'][learner]['folder'], str(Path(FOLDER).resolve()))
            self.assertEqual(result.identities[owner]['policy_bindings'][enemy]['folder'], str(Path(HISTORICAL).resolve()))
        mirrors = [i for i, j in enumerate(source) if j['left'] == j['right']]
        self.assertTrue(mirrors)
        self.assertTrue(all(result.model_indices[i][0] != result.model_indices[i][1] for i in mirrors))

    def test_bad_controls_seeds_settings_identities_and_unknown_fields_rejected(self):
        base = jobs()[0]
        changes = [dict(seed=value) for value in (True, False, -1, 2**63, 1.0, None)]
        changes += [dict(deadline=value) for value in (True, 0, float('nan'), float('inf'), 10**1000)]
        changes += [dict(training=1), dict(pure=0), dict(pure=True), dict(first=True), dict(first_side='b'),
            dict(skip_mulligan=True), dict(skip_mulligan=0), dict(escalation=False), dict(escalation=1),
            dict(train_sides=[]), dict(train_sides=['a', 'a']), dict(train_sides=['unknown']),
            dict(exploration=1), dict(collection_replica=True), dict(collection_replica=-1),
            dict(teacher_mode='network_value'), dict(terminal_horizon=0), dict(simulations=16),
            dict(simulations=True), dict(gumbel_candidates=32), dict(max_depth=11), dict(max_actions=801),
            dict(q_scale='legacy'), dict(search_algorithm='mctx'), dict(runtime='league'),
            dict(guided_policy={}), dict(covered_value_source='other'), dict(zhenhong_passive_reward=True),
            dict(setup_rewards=0), dict(unknown_switch=True), dict(id=''), dict(decks=None)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                adapt_training_jobs([dict(deepcopy(base), **change)], models())
        bad = deepcopy(base)
        bad['policies']['a'] = (FOLDER, 'lingke')
        with self.assertRaises(ValueError):
            adapt_training_jobs([bad], models())
        bad = deepcopy(base)
        bad['decks']['a']['character_ids'].reverse()
        with self.assertRaises(ValueError):
            adapt_training_jobs([bad], models())
        bad = deepcopy(base)
        bad['decks']['a']['card_ids'].reverse()
        with self.assertRaises(ValueError):
            adapt_training_jobs([bad], models())
        bad = deepcopy(base)
        bad['policies']['a'] = list(bad['policies']['a'])
        with self.assertRaises(ValueError):
            adapt_training_jobs([bad], models())
        for key in ('policies', 'decks', 'id', 'seed', 'train_sides'):
            bad = deepcopy(base)
            del bad[key]
            with self.assertRaises(ValueError):
                adapt_training_jobs([bad], models())

    def test_duplicate_jobs_missing_models_and_version_or_build_mismatch_rejected(self):
        base = jobs()[0]
        for duplicate in (deepcopy(base), dict(deepcopy(base), id='another-id')):
            with self.assertRaises(ValueError):
                adapt_training_jobs([base, duplicate], models())
        alias = policy_source_alias(*base['policies']['a'])
        for change in ('missing', 'directory', 'version', 'build', 'build_hash', 'schema'):
            bank = models()
            if change == 'missing': del bank[alias]
            if change == 'directory': bank[alias].directory = HISTORICAL
            if change == 'version': bank[alias].version = '0' * 64
            if change == 'build': bank[alias].serving_deck = None
            if change == 'build_hash': bank[alias].manifest['build_sha256'] = '0' * 64
            if change == 'schema': bank[alias].manifest['schema'] = 'legacy'
            with self.subTest(change=change), self.assertRaises(ValueError):
                adapt_training_jobs([base], bank)

    def test_source_alias_matches_existing_training_binding(self):
        from app.modules.card_game.rl.batched_search.training_rollout import source_alias
        for key in ROSTER:
            self.assertEqual(policy_source_alias(FOLDER, key), source_alias((FOLDER, key)))

    def test_covered_value_scope_is_explicit_normalized_and_never_upgrades_qualification(self):
        scope = '/tmp/soa-covered-scope-not-loaded'
        source = jobs(teacher_mode='covered_value', covered_value_source=scope)[:2]
        source[1]['covered_value_source'] = scope+'/child/..'
        original = deepcopy(source)
        result = adapt_training_jobs(source, models())
        canonical = str(Path(scope).resolve())
        self.assertEqual(result.jobs, tuple(original))
        self.assertEqual(result.search['teacher_mode'], 'covered_value')
        self.assertEqual(result.search['covered_value_source'], canonical)
        self.assertEqual(result.search['world_sampling'], 'per_simulation')
        self.assertEqual((result.search['terminal_horizon'], result.search['simulations'],
                          result.search['gumbel_candidates'], result.search['max_depth']), (3, 32, 16, 10))
        self.assertTrue(all(row['covered_value_source'] == canonical and row['search'] == result.search
                            for row in result.identities))
        self.assertFalse(result.formal_training_qualified)
        self.assertFalse(result.gpu_qualified)

    def test_mixed_teacher_or_scope_and_diagnostic_sampling_contracts_rejected(self):
        covered = jobs(teacher_mode='covered_value', covered_value_source='/tmp/covered-a')[:2]
        for value in (None, '', ' ', True, Path('/tmp/covered-a')):
            bad = deepcopy(covered[:1]); bad[0]['covered_value_source'] = value
            with self.subTest(scope=value), self.assertRaises(ValueError):
                adapt_training_jobs(bad, models())
        for change in (dict(teacher_mode='network', covered_value_source=None),
                       dict(covered_value_source='/tmp/covered-b')):
            bad = deepcopy(covered); bad[1].update(change)
            with self.assertRaises(ValueError): adapt_training_jobs(bad, models())
        for source in (jobs()[:1], covered[:1]):
            for value in ('per_decision', None, True):
                bad = deepcopy(source); bad[0]['world_sampling'] = value
                with self.assertRaises(ValueError): adapt_training_jobs(bad, models())
            for change in (dict(terminal_horizon=0), dict(teacher_mode='covered_terminal'),
                           dict(teacher_mode='covered_expert'), dict(teacher_mode='network_value')):
                bad = deepcopy(source); bad[0].update(change)
                with self.assertRaises(ValueError): adapt_training_jobs(bad, models())

    def test_covered_receipt_passes_through_postbatch_without_network_relabel(self):
        from tests.test_duel_v2_soa_training_results import fixture, STUB
        from app.modules.card_game.rl.batched_search.soa_training_results import verified_training_games
        arguments = fixture()
        source = deepcopy(list(arguments['metadata'].jobs))
        for job in source:
            job.update(teacher_mode='covered_value', covered_value_source='/tmp/covered-a')
        arguments['metadata'] = adapt_training_jobs(source, arguments['models'])
        arguments['dataset']['search'] = deepcopy(arguments['metadata'].search)
        with patch('app.modules.card_game.rl.batched_search.soa_training_results._oracle', return_value=STUB):
            games = verified_training_games(**arguments)
        self.assertTrue(all(row['teacher_mode'] == 'covered_value' and row['terminal_horizon'] == 3
                            for game in games for row in game['rows']))
        self.assertTrue(all(not game['formal_training_qualified'] for game in games))

    def test_import_and_metadata_admission_never_import_cuda_or_load_models(self):
        code = '''
import importlib.abc, sys
class RejectGPU(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch', 'cupy', 'jax', 'cuda') or fullname.endswith('.cuda_vm'):
   raise AssertionError('CUDA imported: '+fullname)
sys.meta_path.insert(0, RejectGPU())
from tests.test_duel_v2_soa_training_jobs import models, jobs
from app.modules.card_game.rl.batched_search.soa_training_jobs import adapt_training_jobs
assert adapt_training_jobs(jobs()[:1],models()).initializers[0]['skip_mulligan'] is False
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
