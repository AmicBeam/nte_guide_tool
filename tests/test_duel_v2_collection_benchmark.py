"""Receipt/plan tests only; synthetic fixtures do not measure GPU performance."""
from copy import deepcopy
from hashlib import sha256
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.rl.collection_benchmark import (
    SCHEMA, check_games, compare_receipts, prepare_case, read_receipt, digest, task_identity, run_case,
)


def case():
    return {'config': {'runtime': 'seven', 'device': 'cuda', 'rule_backend': 'resident_cpu',
                       'search_statistics_device': 'cuda', 'simulations': 32,
                       'teacher_mode': 'covered_value', 'search_roots': 1600,
                       'covered_value_source': 'scope-fixture',
                       'query_batch_rows': 32, 'workers': 12, 'training_buffer_mib': 8192},
            'jobs': [{'id': 'bench-0', 'seed': 12, 'first': 'a', 'deadline': 9999999999,
                      'simulations': 32, 'gumbel_candidates': 16, 'terminal_horizon': 3,
                      'teacher_mode': 'covered_value', 'q_scale': 'natural_wdl',
                      'covered_value_source': 'scope-fixture', 'value_precheck': {'fixture_only': True},
                      'policies': {'a': ['folder', 'weave'], 'b': ['folder', 'creation']},
                      'decks': {'a': {}, 'b': {}}, 'training': False}]}


def receipt(seconds=10):
    # Arithmetic contract fixture, never a device-produced receipt.
    return {'schema': SCHEMA, 'timing_qualified': True, 'gpu_runtime_verified': True,
            'error': None, 'complete_games': 1, 'attempted_games': 1,
            'collection_seconds': seconds, 'optimization_steps': 0,
            'task_sha256': 'synthetic-task', 'hardware': {'gpu': 'synthetic'},
            'episodes': {'bench-0': {'actions_sha256': 'fixture', 'decisions': 2}}}


class CollectionBenchmarkTest(unittest.TestCase):
    def test_unavailable_cuda_does_not_load_models_or_create_output(self):
        with TemporaryDirectory() as folder, patch('torch.cuda.is_available', return_value=False), \
             patch('app.modules.card_game.rl.batched_search.training_rollout.models_for_jobs') as load:
            output = Path(folder) / 'not-created'
            with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
                run_case(case(), output)
            load.assert_not_called()
            self.assertFalse(output.exists())

    def test_identity_uses_numerics_and_names_instead_of_relocated_folders(self):
        from app.modules.card_game.rl.batched_search.training_rollout import source_alias
        _, jobs = prepare_case(case())
        model = SimpleNamespace(weights={'w': np.array([1, 2], dtype=np.float32)},
                                manifest={'schema': 'test', 'features': ['f'], 'candidates': ['c']})
        models = {source_alias(spec): model for spec in jobs[0]['policies'].values()}
        old = task_identity(jobs, models, {'aux': model})
        moved = deepcopy(jobs)
        moved[0]['policies'] = {side: ('new-folder', spec[1])
                                for side, spec in moved[0]['policies'].items()}
        relocated = {source_alias(spec): model for spec in moved[0]['policies'].values()}
        self.assertEqual(old, task_identity(moved, relocated, {'aux': model}))
        changed = deepcopy(model)
        changed.weights['w'][0] = 3
        self.assertNotEqual(old, task_identity(jobs, models, {'aux': changed}))
        renamed = deepcopy(model)
        renamed.manifest['features'] = ['different']
        self.assertNotEqual(old, task_identity(jobs, models, {'aux': renamed}))

    def test_plan_is_non_mutating_and_enforces_exact_contract(self):
        raw = case()
        before = deepcopy(raw)
        config, jobs = prepare_case(raw)
        self.assertEqual(raw, before)
        self.assertEqual(config['max_games'], 1)
        self.assertIsInstance(jobs[0]['policies']['a'], tuple)
        for field, value in (('device', 'cpu'), ('rule_backend', 'serial_cpu'),
                             ('search_roots', True), ('query_batch_rows', 33)):
            bad = case()
            bad['config'][field] = value
            with self.assertRaises(ValueError):
                prepare_case(bad)
        for field, value in (('simulations', 16), ('terminal_horizon', 2),
                             ('id', '../unsafe'), ('uncalibrated_value_diagnostic', True)):
            bad = case()
            bad['jobs'][0][field] = value
            with self.assertRaises(ValueError):
                prepare_case(bad)
        bad = case()
        bad['chunk_games'] = True
        with self.assertRaises(ValueError):
            prepare_case(bad)

    def test_only_complete_full_search_labels_qualify(self):
        jobs = case()['jobs']
        row = {'c': np.ones((2, 3)), 'pi': np.array([.5, .5]), 'search_budget': 32,
               'teacher_mode': 'covered_value', 'search_algorithm': 'gumbel', 'z': 1}
        game = {'id': 'bench-0', 'complete': True, 'decisions': 1, 'searched': 1,
                'rows': [row], 'episode_actions': [{'side': 'a', 'action': {'type': 'end_turn'}}],
                'final_state_sha256': 'fixture', 'winner': 'a'}
        self.assertEqual(set(check_games(jobs, [game])), {'bench-0'})
        for field, value in (('complete', False), ('searched', 0), ('id', 'other')):
            bad = deepcopy(game)
            bad[field] = value
            with self.assertRaises(ValueError):
                check_games(jobs, [bad])
        for budget in (16, 1):
            bad = deepcopy(game)
            bad['rows'][0]['search_budget'] = budget
            with self.assertRaises(ValueError):
                check_games(jobs, [bad])
        forced = deepcopy(game)
        forced['rows'][0].update(c=np.ones((1, 3)), pi=np.ones(1), search_budget=1)
        check_games(jobs, [forced])

    def test_three_matched_trials_compare_medians_not_partial_or_cpu(self):
        old = [receipt(s) for s in (49, 50, 51)]
        new = [receipt(s) for s in (9, 10, 11)]
        result = compare_receipts(old, new)
        self.assertEqual(result['ratio'], 5)
        self.assertFalse(result['model_quality_approved'])
        with self.assertRaises(ValueError):
            compare_receipts(old[:1], new[:1])
        for field, value in (('complete_games', 0), ('task_sha256', 'other'),
                             ('hardware', {}), ('collection_seconds', float('nan')),
                             ('gpu_runtime_verified', False), ('optimization_steps', 1),
                             ('episodes', {})):
            bad = deepcopy(new)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                compare_receipts(old, bad)

    def test_receipt_requires_real_files_and_sha(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / 'episode.json.gz'
            data = {'job': {'id': 'bench-0'}, 'rows': [], 'actions': [{'side': 'a'}], 'winner': 'a'}
            with gzip.open(artifact, 'wt') as f:
                json.dump(data, f)
            raw = receipt()
            raw['episodes']['bench-0'].update(actions_sha256=digest(data['actions']), decisions=1, winner='a')
            raw['files'] = {'episode.json.gz': sha256(artifact.read_bytes()).hexdigest()}
            report = root / 'timing.json'
            report.write_text(json.dumps(raw))
            self.assertEqual(read_receipt(report), raw)
            artifact.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                read_receipt(report)
            raw['files'] = {'../outside': 'fake'}
            report.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                read_receipt(report)


if __name__ == '__main__':
    unittest.main()
