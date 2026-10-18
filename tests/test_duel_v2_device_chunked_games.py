"""CPU synthetic FSM equivalence only, never CUDA capture evidence."""
import unittest

from app.modules.card_game.rl.batched_search.device_chunked_games import DeviceChunkedGameLoop
from app.modules.card_game.rl.batched_search.device_games import GUARD_ERROR
from tests import test_duel_v2_device_games as fixture_games
from unittest.mock import patch,Mock


class ChunkedGameLoopTest(unittest.TestCase):
    def fixture(self,chunk=1,inner=400,**kwargs):
        base,fixture,sink,roots,seeds,ids,train=fixture_games.DeviceGameLoopTest().make(**kwargs)
        loop=DeviceChunkedGameLoop(fixture,base.search,sink,base.world_seed_provider,
            max_decisions=base.max_decisions,device='cpu',allow_cpu_reference=True,
            allow_synthetic=True,inner_max_ticks=inner,chunk_search_ticks=chunk)
        loop.initialize(roots,seeds,ids,ids,train)
        return loop,fixture,sink,roots,seeds,ids,train

    def complete(self,loop):
        for i in range(1000):
            if not bool(loop.unfinished_device.any()):return loop.result_device(),i
            loop.replay_chunk()
        self.fail('Synthetic fixture did not finish')

    def test_many_short_chunks_equal_uninterrupted_full_fsm(self):
        import torch
        reference,fixture,sink,roots,seeds,ids,train=fixture_games.DeviceGameLoopTest().make(simulations=32)
        expected=fixture_games.DeviceGameLoopTest().finish_cpu(reference,roots,seeds,ids,train)
        chunked,other,other_sink,*_=self.fixture(chunk=1,simulations=32)
        actual,chunks=self.complete(chunked)
        self.assertGreater(chunks,int(actual.steps.max()))
        for name in ('steps','winners','complete','truncated','errors','searched_count'):
            self.assertTrue(torch.equal(getattr(expected,name),getattr(actual,name)),name)
        self.assertTrue(torch.equal(sink.appends,other_sink.appends))
        self.assertTrue(torch.equal(reference.search.completed,chunked.search.completed))
        self.assertTrue(torch.equal(reference.search.visits,chunked.search.visits))
        self.assertTrue(torch.equal(reference.search.policy,chunked.search.policy))
        self.assertTrue(torch.equal(reference.search.keys,chunked.search.keys))
        self.assertEqual(other_sink.begins.tolist(),[1,1])
        self.assertEqual(other_sink.finishes.tolist(),[1,1])

    def test_planned_pause_does_not_commit_reset_or_abort(self):
        loop,fixture,sink,*_=self.fixture(chunk=1,simulations=32)
        initial=loop._slots.clone();calls=[];original=loop.search.initialize
        def initialized(*args):calls.append(1);return original(*args)
        loop.search.initialize=initialized
        for _ in range(3):
            loop.replay_chunk()
            self.assertEqual(loop.chunk_pending_device.tolist(),[1])
            self.assertEqual(loop.result_device().steps.tolist(),[0,0])
            self.assertEqual(sink.appends.tolist(),[0,0])
            self.assertEqual(loop._slots.tolist(),initial.tolist())
            self.assertEqual(loop._errors.tolist(),[0,0])
            self.assertEqual(loop.chunk_search_ticks_device.tolist(),[1])
        self.assertEqual(len(calls),1)
        self.assertEqual(loop._root_ticks.tolist(),[3])
        with self.assertRaises(RuntimeError):loop.initialize(None,None,None,None,None)

    def test_cumulative_search_guard_aborts_without_physical_step(self):
        loop,fixture,sink,*_=self.fixture(chunk=1,inner=2,simulations=32)
        loop.replay_chunk();loop.replay_chunk()
        result=loop.result_device()
        self.assertTrue(all(v&GUARD_ERROR for v in result.errors.tolist()))
        self.assertEqual(result.steps.tolist(),[0,0])
        self.assertEqual(loop.chunk_pending_device.tolist(),[0])
        self.assertFalse(bool(loop.search.unfinished_device.any()))
        self.assertFalse(bool(loop.search._world_live.any()))
        self.assertFalse(bool(loop.search._sampled_live.any()))
        self.assertFalse(bool(result.complete.any()))

    def test_explicit_cancellation_drains_pending_search(self):
        loop,fixture,sink,*_=self.fixture(chunk=2,simulations=32)
        loop.replay_chunk();loop.abort(force=True)
        self.assertEqual(loop.result_device().steps.tolist(),[0,0])
        self.assertTrue(bool(loop.result_device().errors.any()))
        self.assertEqual(loop.chunk_pending_device.tolist(),[0])
        self.assertFalse(bool(loop.search._world_live.any()))
        self.assertFalse(bool(loop.search._sampled_live.any()))

    def test_close_is_idempotent_and_releases_pending_parent_roots(self):
        loop,fixture,sink,*_=self.fixture(chunk=1,simulations=32)
        releases=[];original=fixture.release_device
        def release(states):releases.append(states.active.clone());return original(states)
        fixture.release_device=release
        loop.replay_chunk();loop.close();before=len(releases);loop.close()
        self.assertEqual(len(releases),before)
        self.assertTrue(any(mask.all().item()for mask in releases))
        self.assertEqual(loop.chunk_pending_device.tolist(),[0])
        with self.assertRaises(RuntimeError):loop.replay_chunk()

    def test_slow_final_tick_cannot_escape_batch_wall_guard(self):
        import torch
        loop,*_=self.fixture(chunk=1,simulations=32)
        loop.device=torch.device('cuda');loop.allow_synthetic=False
        loop._captured=True;loop._chunk_seconds=1
        runner=Mock();loop._outer=runner
        with patch.object(torch.cuda,'synchronize'),patch(
                'app.modules.card_game.rl.batched_search.device_chunked_games.time.monotonic',
                side_effect=[0.,2.]),patch.object(loop,'abort')as abort:
            loop.replay_chunk()
        runner.replay.assert_called_once();abort.assert_called_once_with(force=True)

    def test_explicit_root_release_then_close_does_not_release_twice(self):
        loop,fixture,*_=self.fixture(chunk=1,simulations=32)
        releases=[];original=fixture.release_device
        def release(states):
            releases.append(states.active.clone());return original(states)
        fixture.release_device=release
        loop.replay_chunk();loop.abort(release_roots=True)
        before=len(releases)
        loop.close()
        self.assertEqual(len(releases),before)
        self.assertTrue(loop._roots_released)
        with self.assertRaises(RuntimeError):loop.replay_chunk()

    def test_invalid_unbounded_chunk_controls_rejected(self):
        for chunk in (0,True,10001):
            with self.assertRaises(ValueError):DeviceChunkedGameLoop(chunk_search_ticks=chunk)
        with self.assertRaises(ValueError):DeviceChunkedGameLoop(seconds=91)


if __name__=='__main__':unittest.main()
