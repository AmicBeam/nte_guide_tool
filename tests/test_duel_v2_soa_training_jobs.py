"""Real seven_round job metadata, without CUDA, model loading or collection."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

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
