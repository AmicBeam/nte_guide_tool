"""Actual pure calibration/comparison job shapes; CPU metadata only."""
from copy import deepcopy
from pathlib import Path
import pickle
import subprocess
import sys
import unittest

from app.modules.card_game.rl.seven_round import jobs_for, compare_jobs
from app.modules.card_game.rl.batched_search.soa_pure_jobs import adapt_pure_jobs
from app.modules.card_game.rl.batched_search.soa_training_jobs import policy_source_alias
from tests.test_duel_v2_soa_training_jobs import DECKS, FOLDER, HISTORICAL, models


def calibration(**kwargs):
    return jobs_for(FOLDER, DECKS, 10_000_000_000, 2_000_000_000, pure=True, **kwargs)


def comparison(candidate=FOLDER, reference=HISTORICAL, **kwargs):
    return compare_jobs(candidate, reference, DECKS, 10_000_000_000,
                         2_000_000_000, 2, search=False, **kwargs)


class SoAPureJobsTests(unittest.TestCase):
    def test_real_calibration_jobs_preserve_mulligans_and_do_not_invent_search_or_precheck(self):
        jobs, bank = calibration(), models()
        before = pickle.dumps((jobs, bank))
        metadata = adapt_pure_jobs(jobs, bank)
        self.assertEqual(metadata.jobs, tuple(jobs))
        self.assertEqual(pickle.dumps((jobs, bank)), before)
        self.assertEqual(metadata.execution, dict(algorithm='argmax', counts_as_search=False,
            simulations=0, teacher_used=False, world_sampling=False))
        self.assertFalse(metadata.learning_eligible)
        self.assertFalse(metadata.formal_training_qualified)
        self.assertFalse(metadata.gpu_qualified)
        for row, job in enumerate(jobs):
            self.assertEqual(metadata.initializers[row], dict(seed=job['seed'], first_side=job['first'],
                decks=job['decks'], skip_mulligan=False, escalation=True))
            self.assertIsNone(metadata.jobs[row]['value_precheck'])
            self.assertEqual(metadata.jobs[row]['terminal_horizon'], 3)
            self.assertEqual(metadata.reference_indices[row], (-1, -1))
            self.assertEqual(metadata.reference_versions[row], {})
        metadata.jobs[0]['decks']['a']['card_ids'].reverse()
        metadata.initializers[0]['decks']['a']['card_ids'].reverse()
        self.assertEqual(pickle.dumps((jobs, bank)), before)

    def test_real_paired_jobs_retain_independent_reference_indices_and_all_roles(self):
        jobs = comparison()
        bank, refbank = models(FOLDER, HISTORICAL), models(HISTORICAL)
        metadata = adapt_pure_jobs(jobs, bank, reference_models=refbank)
        self.assertEqual(len(metadata.jobs), len(jobs))
        self.assertEqual(metadata.inference_keys, tuple(sorted(bank)))
        self.assertEqual(metadata.reference_inference_keys, tuple(sorted(refbank)))
        for row, job in enumerate(jobs):
            learner = job['learner']
            for field in ('key', 'replica', 'label', 'learner', 'references'):
                self.assertEqual(metadata.identities[row][field], job[field])
            for side in ('a', 'b'):
                alias = policy_source_alias(*job['policies'][side])
                self.assertEqual(metadata.inference_keys[metadata.model_indices[row][int(side == 'b')]], alias)
                self.assertEqual(metadata.model_versions[row][side], bank[alias].version)
            alias = policy_source_alias(*job['references'][learner])
            self.assertEqual(metadata.reference_inference_keys[metadata.reference_indices[row][int(learner == 'b')]], alias)
            self.assertEqual(metadata.reference_indices[row][int(learner == 'a')], -1)
            self.assertEqual(metadata.reference_versions[row], {learner: refbank[alias].version})
            self.assertEqual(metadata.initializers[row]['decks'], job['decks'])

    def test_equal_source_candidate_reference_and_mirror_learners_are_not_wrongly_deduplicated(self):
        jobs = comparison(FOLDER, FOLDER)
        metadata = adapt_pure_jobs(jobs, models(), reference_models=models())
        self.assertEqual(len(metadata.jobs), len(jobs))
        pairs = [j for j in jobs if j['left'] == j['right'] and j['key'] == 'starter']
        self.assertEqual({(j['label'], j['learner'], j['first']) for j in pairs},
                         {(label, learner, first) for label in ('candidate', 'reference')
                          for learner in ('a', 'b') for first in ('a', 'b')})

    def test_historical_same_team_versions_and_single_viewer_requests_remain_distinct(self):
        jobs = calibration(historical=HISTORICAL, keys=('starter',))
        metadata = adapt_pure_jobs(jobs, models(FOLDER, HISTORICAL))
        for row, job in enumerate(jobs):
            learner = job['train_sides'][0]
            self.assertEqual(metadata.train_masks[row], (learner == 'a', learner == 'b'))
            if job['left'] == job['right']:
                self.assertNotEqual(*metadata.model_indices[row])
                self.assertNotEqual(metadata.model_versions[row]['a'], metadata.model_versions[row]['b'])

    def test_duplicate_jobs_and_replica_renaming_cannot_create_extra_physical_attempts(self):
        original = calibration()[0]
        for other in (deepcopy(original), dict(deepcopy(original), id='renamed')):
            with self.assertRaises(ValueError): adapt_pure_jobs([original, other], models())
        original = comparison()[0]
        for other in (dict(deepcopy(original), id='renamed'),
                      dict(deepcopy(original), id='renamed-replica', replica=999)):
            with self.assertRaises(ValueError):
                adapt_pure_jobs([original, other], models(FOLDER, HISTORICAL), reference_models=models(HISTORICAL))

    def test_identical_weights_copied_to_another_folder_cannot_create_a_physical_game(self):
        bank = models(FOLDER, HISTORICAL)
        # Loaded copies of the same numerical npz have the same version/digest,
        # even though their exact source aliases and folders are different.
        for key in DECKS:
            current = bank[policy_source_alias(FOLDER, key)]
            copied = bank[policy_source_alias(HISTORICAL, key)]
            copied.version = current.version
            copied.manifest['sha256'] = current.version
        original = calibration()[0]
        copied = deepcopy(original)
        copied['id'] = 'copied-weights'
        copied['policies'] = {side: (HISTORICAL, spec[1])
                              for side, spec in original['policies'].items()}
        before = pickle.dumps((original, copied, bank))
        with self.assertRaisesRegex(ValueError, 'Duplicate physical pure job'):
            adapt_pure_jobs([original, copied], bank)
        self.assertEqual(pickle.dumps((original, copied, bank)), before)
        metadata = adapt_pure_jobs([copied], bank)
        for side in ('a', 'b'):
            binding = metadata.identities[0]['policy_bindings'][side]
            self.assertEqual(binding['folder'], str(Path(HISTORICAL).resolve()))
            self.assertEqual(binding['alias'], policy_source_alias(*copied['policies'][side]))
            self.assertEqual(binding['version'], bank[policy_source_alias(*original['policies'][side])].version)
        reordered = deepcopy(copied)
        reordered['id'] = 'same-weights-different-original-order'
        reordered['decks']['a']['card_ids'].reverse()
        self.assertNotEqual(reordered['decks']['a']['card_ids'], original['decks']['a']['card_ids'])
        ordered = adapt_pure_jobs([original, reordered], bank)
        self.assertEqual(len(ordered.jobs), 2)
        self.assertEqual(ordered.initializers[1]['decks'], reordered['decks'])
        # Policy role remains meaningful even when candidate/reference weights
        # are copied equivalents rather than identical directory strings.
        reference_bank = {alias: deepcopy(model) for alias, model in bank.items()
                          if model.directory == HISTORICAL}
        paired = comparison()
        self.assertEqual(len(adapt_pure_jobs(paired, bank, reference_models=reference_bank).jobs), len(paired))

    def test_wrong_source_version_build_flags_and_comparison_identity_rejected(self):
        original = comparison()[0]
        for field, value in (('pure', False), ('pure', 1), ('training', True), ('training', 0),
                ('seed', True), ('seed', -1), ('seed', 2**63), ('first', True),
                ('train_sides', []), ('train_sides', ['a', 'a']), ('skip_mulligan', True),
                ('skip_mulligan', 0), ('exploration', True), ('exploration', 0),
                ('label', 'other'), ('learner', 'other'), ('key', 'other'), ('replica', True),
                ('replica', -1), ('unknown_flag', True), ('guided_policy', {})):
            job = dict(deepcopy(original), **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                adapt_pure_jobs([job], models(FOLDER, HISTORICAL), reference_models=models(HISTORICAL))
        with self.assertRaises(ValueError): adapt_pure_jobs([original], models(FOLDER, HISTORICAL))
        for field in ('key', 'replica', 'label', 'learner', 'references'):
            job = deepcopy(original); del job[field]
            with self.assertRaises(ValueError):
                adapt_pure_jobs([job], models(FOLDER, HISTORICAL), reference_models=models(HISTORICAL))
        job = deepcopy(original); job['decks']['a']['character_ids'].reverse()
        with self.assertRaises(ValueError):
            adapt_pure_jobs([job], models(FOLDER, HISTORICAL), reference_models=models(HISTORICAL))
        job = deepcopy(original); job['label'] = 'reference'
        with self.assertRaises(ValueError):
            adapt_pure_jobs([job], models(FOLDER, HISTORICAL), reference_models=models(HISTORICAL))
        refbank = models(HISTORICAL)
        alias = policy_source_alias(*original['references'][original['learner']])
        refbank[alias].version = '0'*64; refbank[alias].manifest['sha256'] = '0'*64
        with self.assertRaises(ValueError):
            adapt_pure_jobs([original], models(FOLDER, HISTORICAL), reference_models=refbank)

    def test_import_and_admission_do_not_load_cuda_or_execute_rules(self):
        code = '''
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch','cupy','jax','cuda') or fullname.endswith('.cuda_vm'):
   raise AssertionError('execution module imported: '+fullname)
sys.meta_path.insert(0, Reject())
from tests.test_duel_v2_soa_pure_jobs import calibration, models
from app.modules.card_game.rl.batched_search.soa_pure_jobs import adapt_pure_jobs
assert adapt_pure_jobs(calibration()[:1],models()).execution['counts_as_search'] is False
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
