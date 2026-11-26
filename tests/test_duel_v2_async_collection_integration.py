"""Physical commits must not restore a barrier behind another worker's GC releases."""
import gc
import time
import unittest

from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.rl.batched_duel.process_backend import ResidentRuleBackend
from app.modules.card_game.rl.batched_search.cooperative import OperationQuery


def delayed_ops(ops, releases):
    from app.modules.card_game.rl.batched_duel.process_backend import _worker_ops_batch
    time.sleep(0.7)
    return _worker_ops_batch(ops, releases)


class DelayFirstOperation:
    def __init__(self, executor):
        self.executor = executor
        self.used = False

    def __getattr__(self, name):
        return getattr(self.executor, name)

    def submit(self, function, *args, **kwargs):
        if not self.used and function.__name__ == '_worker_ops_batch':
            self.used = True
            return self.executor.submit(delayed_ops, *args, **kwargs)
        return self.executor.submit(function, *args, **kwargs)


class AsyncCollectionIntegrationTest(unittest.TestCase):
    def test_ready_real_commit_does_not_wait_for_other_workers_gc(self):
        game = new_game(seed=83, first_side='a', skip_mulligan=True)
        with ResidentRuleBackend(workers=2, max_games=2, max_states_per_worker=40) as backend:
            fast, slow = backend.import_states([game, game], ['fast', 'slow'], [1, 1])
            self.assertNotEqual(fast.worker, slow.worker)
            legal, _ = backend.decision(fast, 'a')
            temporary = backend.sample_world(slow, 'a', 88)
            backend._executors[slow.worker] = DelayFirstOperation(backend._executors[slow.worker])
            backend.submit_operations([OperationQuery('slow-observation', backend,
                                      'value_observation', (slow, 'a'), ())])
            del temporary
            gc.collect()
            started = time.monotonic()
            next_fast = backend.commit_action(fast, 'a', legal[0], 2)
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 0.55, 'An unrelated GC release reintroduced the worker barrier')
            self.assertEqual(next_fast.root_generation, 2)
            deadline = time.monotonic() + 5
            responses = []
            while not responses and time.monotonic() < deadline:
                responses = backend.poll_operations(timeout=0.1)
            self.assertEqual([r.id for r in responses], ['slow-observation'])
            self.assertEqual(backend.stats()['pending_operations'], 0)


if __name__ == '__main__':
    unittest.main()
