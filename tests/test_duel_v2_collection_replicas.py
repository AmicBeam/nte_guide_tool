"""Named seven-team search collection replicas: unique games, default 1, no eval multiply."""
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch
import io
import json
import runpy
import tempfile
import time
import unittest

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.rl.seven_lineup import identity
from app.modules.card_game.rl.seven_round import (
    capacity_heldout_jobs, collection_jobs, compare_jobs, expected_preflight_learning_games,
    jobs_for, learning_collection_jobs, projected_learning_work, rollout_contract,
    validate_launch, restore_round, fingerprint,
)
from app.modules.card_game.rl.seven_schedule import matrix_schedule
from app.modules.card_game.rl.training_config import (
    COLLECTION_GAMES_PER_REPLICA, COLLECTION_HISTORICAL_GAMES_PER_REPLICA,
    COLLECTION_REPLICA_SEED_STRIDE, DEFAULT_COLLECTION_REPLICAS,
    collection_episode_pool_limit, collection_replicas, planned_complete_games,
    validate_collection_option,
)

DECKS = {deck['id']: deck for deck in STARTER_DECKS}


def _team_counts(jobs):
    counts = Counter()
    for job in jobs:
        for side in job['train_sides']:
            counts[job['policies'][side][1]] += 1
    return counts


class CollectionReplicaConfigTest(unittest.TestCase):
    def test_default_and_named_bounds(self):
        self.assertEqual(collection_replicas({}), DEFAULT_COLLECTION_REPLICAS)
        self.assertEqual(collection_replicas(value=1), 1)
        self.assertEqual(planned_complete_games(1), 56)
        self.assertEqual(planned_complete_games(5, historical=True), 560)
        self.assertEqual(collection_episode_pool_limit(5), 320)
        self.assertEqual(validate_collection_option(5, search_roots=8, training_buffer_mib=8192,
                                                    rule_backend='resident_cpu', runtime='seven',
                                                    historical=True), 5)
        for bad in (True, False, 0, -1, 6, 1.5, '2'):
            with self.assertRaises(ValueError):
                validate_collection_option(bad, rule_backend='resident_cpu', runtime='seven')
        with self.assertRaisesRegex(ValueError, 'resident_cpu or cuda_vm'):
            validate_collection_option(5, rule_backend='serial_cpu', runtime='seven')
        with self.assertRaisesRegex(ValueError, 'seven-team'):
            validate_collection_option(2, rule_backend='resident_cpu', runtime='recovery')
        with self.assertRaisesRegex(ValueError, 'training trajectory buffer'):
            validate_collection_option(5, search_roots=1600, training_buffer_mib=0,
                                       rule_backend='cuda_vm', runtime='seven', historical=True)

    def test_roots_are_not_a_job_cap(self):
        replicas = validate_collection_option(
            5, search_roots=16, training_buffer_mib=8192,
            gpu_memory_budget_mib=14336, rule_backend='cuda_vm', runtime='seven')
        self.assertEqual(planned_complete_games(replicas), 280)
        self.assertGreater(planned_complete_games(replicas), 16)


class CollectionReplicaJobsTest(unittest.TestCase):
    def test_default_replica_keeps_ids_seeds_and_pairing(self):
        baseline = jobs_for('frozen', DECKS, 771, 99, pure=False, cycle=3)
        jobs = collection_jobs('frozen', DECKS, 771, 99, replicas=1, pure=False, cycle=3)
        self.assertEqual(len(baseline), COLLECTION_GAMES_PER_REPLICA)
        self.assertEqual([(j['id'], j['seed'], j['first'], j['left'], j['right']) for j in jobs],
                         [(j['id'], j['seed'], j['first'], j['left'], j['right']) for j in baseline])
        for a, b in zip(jobs[::2], jobs[1::2]):
            self.assertEqual(a['seed'], b['seed'])
            self.assertEqual((a['left'], a['right']), (b['left'], b['right']))
            self.assertEqual({a['first'], b['first']}, {'a', 'b'})
        self.assertEqual(len(_team_counts(jobs)), 7)
        self.assertEqual(len(set(_team_counts(jobs).values())), 1)

    def test_replicas_are_unique_balanced_and_collision_free(self):
        jobs = collection_jobs('frozen', DECKS, 1000, 99, replicas=5, pure=False, cycle=2)
        self.assertEqual(len(jobs), 280)
        self.assertEqual(len({j['id'] for j in jobs}), 280)
        self.assertEqual(len(set(_team_counts(jobs).values())), 1)
        self.assertTrue(all(j['policies']['a'][0] == 'frozen' and j['policies']['b'][0] == 'frozen' for j in jobs))
        by_replica = defaultdict(list)
        for job in jobs:
            by_replica[job['collection_replica']].append(job)
        self.assertEqual(sorted(by_replica), [0, 1, 2, 3, 4])
        seed_sets = {}
        for replica, group in by_replica.items():
            self.assertEqual(len(group), 56)
            seed_sets[replica] = {j['seed'] for j in group}
            self.assertEqual(len(seed_sets[replica]), 28)
            pairs = defaultdict(list)
            for job in group:
                pairs[(job['left'], job['right'])].append(job)
            for pair in pairs.values():
                self.assertEqual(len({j['seed'] for j in pair}), 1)
                self.assertEqual({j['first'] for j in pair}, {'a', 'b'})
            if replica:
                self.assertTrue(all(j['id'].endswith('-r%s' % replica) for j in group))
                self.assertEqual(min(seed_sets[replica]), 1000 + replica * COLLECTION_REPLICA_SEED_STRIDE)
        for left in seed_sets:
            for right in seed_sets:
                if left != right:
                    self.assertTrue(seed_sets[left].isdisjoint(seed_sets[right]))
        extra = jobs_for('frozen', DECKS, 1000 + 100, 99, pure=False, cycle=1001)
        self.assertTrue({j['seed'] for j in jobs}.isdisjoint({j['seed'] for j in extra}))
        self.assertTrue({j['id'] for j in jobs}.isdisjoint({j['id'] for j in extra}))

    def test_historical_collection_is_112_per_replica(self):
        jobs = collection_jobs('now', DECKS, 4000, 99, replicas=2, pure=False, cycle=4, historical='old')
        self.assertEqual(len(jobs), COLLECTION_HISTORICAL_GAMES_PER_REPLICA * 2)
        self.assertEqual(len({j['id'] for j in jobs}), 224)
        self.assertEqual(len(set(_team_counts(jobs).values())), 1)
        for job in jobs:
            self.assertEqual(len(job['train_sides']), 1)
            learner = job['train_sides'][0]
            enemy = 'b' if learner == 'a' else 'a'
            self.assertEqual(job['policies'][learner][0], 'now')
            self.assertEqual(job['policies'][enemy][0], 'old')

    def test_learning_vs_heldout_and_selection_partitions(self):
        config = dict(seeds={'foundation': 1000, 'probes': 2000000, 'selection': 3000000},
                      simulations=32, collection_replicas=5, preflight=True,
                      preflight_replicas=2, teacher_mode='network')
        learning = learning_collection_jobs('now', DECKS, config, 1, 99, report={})
        heldout = capacity_heldout_jobs('now', DECKS, config, 99, report={})
        self.assertEqual(len(learning), 56 * 5 + 56)
        self.assertEqual(len(heldout), 56 * 2)
        self.assertEqual(expected_preflight_learning_games(config), len(learning))
        self.assertTrue(all(j.get('data_partition') == 'capacity_heldout' for j in heldout))
        self.assertTrue(all(j.get('data_partition') != 'capacity_heldout' for j in learning))
        self.assertTrue({j['id'] for j in learning}.isdisjoint({j['id'] for j in heldout}))
        self.assertTrue({j['seed'] for j in learning}.isdisjoint({j['seed'] for j in heldout}))
        unsafe=dict(config,seeds=dict(config['seeds'],probes=2000))
        with self.assertRaisesRegex(ValueError,'seed blocks overlap'):
            learning_collection_jobs('now',DECKS,unsafe,1,99,report={})
        selection = compare_jobs('cand', 'ref', DECKS, 3000, 99, 2)
        self.assertEqual(len(selection), 7 * 2 * 2 * 2 * 2)
        self.assertEqual(len(matrix_schedule(16)), 7 * 7 * 16)


class CollectionReplicaCapacityTest(unittest.TestCase):
    def test_resume_cycle_must_match_the_actual_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory); folder=out/'recovery'/'0'; folder.mkdir(parents=True)
            config=dict(train_until=time.time()+60, hard_deadline=time.time()+120, planned_batches=4)
            pointer=dict(algorithm_sha256=fingerprint(),cycle=1,folder=str(folder),files={},
                         models=str(out/'models'),hard_deadline=config['hard_deadline'],
                         original_train_until=config['train_until'])
            (out/'recovery'/'pointer.json').write_text(json.dumps(pointer))
            data={'metadata':{'config':config,'cycle':2}}
            with patch('app.modules.card_game.rl.seven_checkpoint.restore',return_value=(None,None,data)):
                with self.assertRaisesRegex(ValueError,'pointer cycle'):
                    restore_round(out,'cpu',config)

    def test_projected_costs_scale_collection_not_eval(self):
        receipt = dict(search_seconds_per_game=10, update_and_numeric_seconds=100,
                       gate_roots=32, pure_gate_seconds=40, selected_value_seconds=50)
        base = dict(planned_batches=4, gate_roots=32, search_gate_roots=2, matrix_games=16)
        one = projected_learning_work(dict(base, collection_replicas=1), receipt)
        five = projected_learning_work(dict(base, collection_replicas=5), receipt)
        self.assertEqual(one['planned_complete_games'], 56)
        self.assertEqual(five['planned_complete_games'], 280)
        self.assertEqual(five['planned_historical_complete_games'], 560)
        self.assertEqual(one['needed_eval'], five['needed_eval'])
        self.assertGreater(five['needed'], one['needed'])
        self.assertAlmostEqual(five['per_cycle'] - one['per_cycle'], 4 * 56 * 10)
        resume = projected_learning_work(dict(base, collection_replicas=5), receipt,
                                          resume=True, completed_batches=2)
        self.assertAlmostEqual(resume['needed'],
                               (50 + 2 * five['per_cycle'] + 280 * 10) * 1.25)
        self.assertGreater(resume['needed'], 50 * 1.25)
        for completed in (True, -1, 5):
            with self.assertRaises(ValueError):
                projected_learning_work(base, receipt, resume=True, completed_batches=completed)

    def test_rollout_contract_default_identity(self):
        default = rollout_contract(dict(rule_backend='resident_cpu', runtime='seven'))
        explicit = rollout_contract(dict(rule_backend='resident_cpu', runtime='seven', collection_replicas=1))
        self.assertEqual(default['collection_replicas'], 1)
        self.assertEqual(default['planned_complete_games'], 56)
        self.assertEqual(default['planned_historical_complete_games'], 112)
        self.assertEqual(default, explicit)
        five = rollout_contract(dict(rule_backend='resident_cpu', runtime='seven', collection_replicas=5,
                                     search_roots=16, training_buffer_mib=8192))
        self.assertEqual(five['collection_replicas'], 5)
        self.assertEqual(five['search_roots'], 16)
        self.assertGreater(five['planned_complete_games'], five['search_roots'])

    def test_smoke_launch_accepts_default_and_rejects_unsupported(self):
        config = dict(runtime='seven', rule_hash=identity(), smoke=True, preflight=True,
                      preflight_replicas=1, planned_batches=1, workers=4, simulations=32,
                      started_at=0, hard_deadline=1800, train_until=1700, max_row_reuse=8,
                      policy_passes=8, rule_backend='resident_cpu', search_roots=16,
                      training_buffer_mib=8192)
        validate_launch(config)
        validate_launch(dict(config, collection_replicas=5))
        for changes in ({'collection_replicas': True}, {'collection_replicas': 0},
                        {'collection_replicas': 6}, {'collection_replicas': 5, 'rule_backend': 'serial_cpu'},
                        {'collection_replicas': 5, 'training_buffer_mib': 0}):
            with self.assertRaises(ValueError):
                validate_launch(dict(config, **changes))


class CollectionReplicaCliTest(unittest.TestCase):
    def _run(self, *args):
        out = io.StringIO(); err = io.StringIO()
        with tempfile.TemporaryDirectory() as folder:
            argv = ['train_duel_v2_current_round.py', '--output', str(Path(folder) / 'out'), *args]
            with patch('sys.argv', argv), patch('sys.stdout', out), patch('sys.stderr', err):
                try:
                    runpy.run_path('scripts/train_duel_v2_current_round.py', run_name='__main__')
                    code = 0
                except SystemExit as exc:
                    code = exc.code
        return out.getvalue(), err.getvalue(), code

    def test_default_cli_keeps_one_replica(self):
        stdout, _, code = self._run()
        self.assertEqual(code, 0)
        config = json.loads(stdout)
        self.assertEqual(config['runtime'], 'seven')
        self.assertEqual(config['collection_replicas'], 1)
        self.assertEqual(config['preflight_replicas'], 1)

    def test_cli_rejects_invalid_and_allows_pending_roots(self):
        for extra in (('--collection-replicas', '0'), ('--collection-replicas', '6'),
                      ('--collection-replicas', 'true'), ('--runtime', 'recovery', '--collection-replicas', '2'),
                      ('--collection-replicas', '5'),
                      ('--rule-backend', 'resident_cpu', '--collection-replicas', '5', '--training-buffer-mib', '0')):
            _, err, code = self._run(*extra)
            self.assertNotEqual(code, 0, extra)
            self.assertTrue(err)
        stdout, _, code = self._run('--rule-backend', 'resident_cpu', '--collection-replicas', '5',
                                    '--search-roots', '8', '--training-buffer-mib', '8192')
        self.assertEqual(code, 0)
        config = json.loads(stdout)
        self.assertEqual(config['collection_replicas'], 5)
        self.assertEqual(config['search_roots'], 8)
        self.assertGreater(planned_complete_games(5), config['search_roots'])
        self.assertNotIn('ready', config)  # A dry plan does not qualify real buffer capacity.

    def test_async_and_cuda_statistics_are_explicitly_wired(self):
        stdout, _, code = self._run('--rule-backend', 'resident_cpu', '--device', 'cuda',
                                    '--async-rule-dispatch', 'off', '--search-statistics-device', 'cuda')
        self.assertEqual(code, 0)
        config = json.loads(stdout)
        self.assertIs(config['async_rule_dispatch'], False)
        self.assertEqual(config['search_statistics_device'], 'cuda')
        for args in (('--async-rule-dispatch', 'on'), ('--search-statistics-device', 'cuda'),
                     ('--runtime', 'recovery', '--async-rule-dispatch', 'on')):
            _, _, code = self._run(*args)
            self.assertNotEqual(code, 0)


if __name__ == '__main__':
    unittest.main()
