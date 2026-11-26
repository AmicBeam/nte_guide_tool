"""Synthetic CPU equivalence and transaction failures; no readiness evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.rl import hybrid_capacity as stage
from app.modules.card_game.rl import recovery_capacity as baseline
from app.modules.card_game.rl import seven_runtime as rt
from app.modules.card_game.rl.batched_search.pure_checkpoint import read_payload

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Torch unavailable')
class HybridCapacityTest(unittest.TestCase):
    def fixture(self, root):
        from app.modules.card_game.rl.seven_schedule import ROSTER
        from app.modules.card_game.rl.next_training import plan
        from app.modules.card_game.engine.duel_v2 import new_game
        torch.set_num_threads(1)
        torch.manual_seed(2091)
        source = root/'source'
        builds = plan()['builds']
        for key in ROSTER:
            rt.export(rt.create_network(key, 16), source, key, builds[key],
                      {'initialization':'synthetic_capacity_test'})
        state = new_game(seed=2091, decks={s:builds['starter'] for s in ('a','b')},
                         first_side='a', skip_mulligan=True)
        _, (x,c) = rt.decision(state, 'a')
        pi = np.zeros(len(c), dtype=np.float32)
        pi[0] = 1
        jobs, games, held_jobs, held_games = [], [], [], []
        for seed in range(3):
            job = dict(seed=seed, training=True, train_sides=['a'], left='starter',
                       right='starter', first='a')
            rows = [dict(x=x.copy(), c=c.copy(), pi=pi.copy(), key='starter', side='a',
                         z=(i%3)-1, value_actor=1) for i in range(90)]
            jobs.append(job)
            games.append(dict(complete=True, rows=rows))
            held_jobs.append(dict(job, seed=seed+100, training=False))
            held_games.append(deepcopy(games[-1]))
        return source, jobs, games, held_jobs, held_games, dict(device='cpu', seeds={'probes':291},
                                                             learning_keys=['starter'])

    def test_segmented_matches_existing_full_fit_and_restores_exact_adam(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, jobs, games, hj, hg, config = self.fixture(root)
            initial_rng = torch.get_rng_state()
            captured = {}
            restore = rt.restore
            def capture(*args, **kwargs):
                value = restore(*args, **kwargs)
                captured['net'] = value[0]
                return value
            with patch.object(rt, 'restore', side_effect=capture):
                expected = baseline.fit(source, jobs, games, config, time.time()+60,
                    steps=2, held_jobs=hj, held_games=hg, keys=['starter'], runtime_name='seven')
            expected_net = deepcopy(captured['net'].state_dict())
            train, held, seeds, hs = stage._partitions(jobs, games, hj, hg, 'starter')
            torch.set_rng_state(initial_rng)
            now = [0.]
            update = rt.update
            def timed_update(*args, **kwargs):
                result = update(*args, **kwargs)
                now[0] += 1
                return result
            binding = {'contract':'synthetic-equivalence'}
            receipts = []
            with patch.object(stage.time, 'time', side_effect=lambda:now[0]), \
                    patch.object(rt, 'update', side_effect=timed_update):
                for _ in range(5):
                    receipt = stage._fit_segment(source, train, held, seeds, hs, config,
                        100, 'starter', root/'segments', 1.5, 2, binding)
                    receipts.append(receipt)
                    if receipt['complete']:
                        break
            self.assertGreater(len(receipts), 1)
            self.assertEqual(receipts[0]['status'], 'READY')
            self.assertEqual(receipts[0]['steps'], 0)  # paused inside first epoch
            self.assertEqual(receipt['metric'], expected['metrics']['starter'])
            self.assertEqual(receipt['row_uses'], len(train)*2)
            payload = read_payload(receipt['checkpoint'], binding, kind=stage.SCHEMA)
            self.assertTrue(all(torch.equal(t, payload['network'][n]) for n,t in expected_net.items()))
            self.assertTrue(payload['optimizer']['state'])
            before = list((root/'segments').iterdir())
            again = stage._fit_segment(source, train, held, seeds, hs, config,
                -1, 'starter', root/'segments', 1.5, 2, binding)
            self.assertEqual(again, receipt)
            self.assertEqual(before, list((root/'segments').iterdir()))
            status = Path(receipt['checkpoint']).parent/'status.json'
            tampered = json.loads(status.read_text()); tampered['passed'] = not tampered['passed']
            status.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, 'immutable numeric'):
                stage._fit_segment(source, train, held, seeds, hs, config, 100,
                                   'starter', root/'segments', 1.5, 2, binding)

    def test_failed_and_running_units_cannot_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, jobs, games, hj, hg, config = self.fixture(root)
            train, held, seeds, hs = stage._partitions(jobs, games, hj, hg, 'starter')
            binding = {'contract':'synthetic-failure'}
            with patch.object(rt, 'update', side_effect=RuntimeError('synthetic interruption')):
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    stage._fit_segment(source, train, held, seeds, hs, config, time.time()+60,
                                       'starter', root/'segments', 30, 2, binding)
            marker = root/'segments/unit-000000/status.json'
            receipt = json.loads(marker.read_text())
            self.assertEqual(receipt['status'], 'FAILED')
            for status in ('FAILED', 'RUNNING'):
                marker.write_text(json.dumps(dict(receipt, status=status)))
                with self.assertRaisesRegex(ValueError, 'refuses replay'):
                    stage._fit_segment(source, train, held, seeds, hs, config, time.time()+60,
                                       'starter', root/'segments', 30, 2, binding)

    def test_public_gate_rejects_synthetic_unqualified_data_before_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, jobs, games, hj, hg, config = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'Qualified Gumbel'):
                stage.fit_capacity_segment(source, jobs, games, config, time.time()+60,
                    hj, hg, 'starter', root/'unqualified', steps=2)
            self.assertFalse((root/'unqualified').exists())
            hj[0]['seed'] = jobs[0]['seed']
            with self.assertRaisesRegex(ValueError, 'independent'):
                stage._partitions(jobs, games, hj, hg, 'starter')


if __name__ == '__main__':
    unittest.main()
