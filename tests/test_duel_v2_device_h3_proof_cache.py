"""Real CPU prefix/encoding with mocked full-budget oracle; no GPU qualification."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np

from app.modules.card_game.rl.batched_search.device_h3_collection import _proof_references
from app.modules.card_game.rl.batched_search.device_h3_segments import H3CheckpointSession,H3SegmentPaused,replay_h3_chunks
from app.modules.card_game.rl.batched_search.soa_training_jobs import policy_source_alias


class H3ProofCacheTests(unittest.TestCase):
    def fixture(self):
        from app.modules.card_game.engine.duel_v2 import new_game,acting_side,observe,apply_action
        from app.modules.card_game.rl.league_rollout import clean
        from app.modules.card_game.rl.seven_lineup import encode
        state=new_game(seed=41,first_side='a',skip_mulligan=False)
        initial=[deepcopy(state)];raw=[];rows=[];hashes=np.zeros((1,2,8),dtype=np.int64);references=[]
        diagnostics={name:np.zeros((1,2,800),dtype=np.float64)for name in ('q','prior','mean','gumbel')}
        diagnostics['value']=np.zeros((1,2),dtype=np.float64);diagnostics['errors']=np.zeros((1,2),dtype=np.bool_)
        for step in range(2):
            actor=acting_side(state);view=observe(state,actor,include_previews=False)
            legal=[v['action']for v in view['legal_actions']if v['action']['type']!='concede'];x,c=encode(view,legal);n=len(legal)
            hashes[0,step]=np.frombuffer(sha256(x.tobytes()+c.tobytes()).digest(),dtype='>u4').astype(np.int64)
            pi=np.full(n,1/n,dtype=np.float32);visits=np.zeros(n,dtype=np.int32);visits[0]=32
            rows.append(dict(owner=0,step=step,selected=0,requested=32,completed=32,pi=pi,visits=visits))
            raw.append(dict(game=0,owner=0,step=step,seed=41,version=state['version'],actor=int(actor=='b'),raw_action=deepcopy(legal[0])))
            prior=np.full(n,1/n);diagnostics['prior'][0,step,:n]=prior
            references.append(dict(complete=True,root_diagnostics=dict(completed_q=np.zeros(n)),root_prior=prior,
                mean_values=np.zeros(n),root_gumbel=np.zeros(n),value=0.,visits=visits.copy(),pi=pi.copy(),search_choice=0))
            state=clean(apply_action(state,actor,deepcopy(legal[0])))
        job=dict(seed=41,training=True,deadline=time.time()+600,policies=dict(a=('/tmp/frozen','starter'),b=('/tmp/frozen','starter')))
        metadata=NS(jobs=[job]);models={policy_source_alias('/tmp/frozen','starter'):NS()};auxiliary={'starter':NS()}
        binding=dict(metadata=dict(jobs=[deepcopy(job)]),main_versions=['frozen-main'],auxiliary_versions=['frozen-aux'])
        return initial,raw,rows,metadata,models,NS(),auxiliary,NS(),hashes,diagnostics,references,binding

    def prove(self,fixture,session):
        args=fixture[:10]
        return _proof_references(*deepcopy(args[:3]),*args[3:],time.monotonic()+500,root_proof_seconds=10,checkpoint=session)

    def test_two_root_proof_across_fresh_process_does_not_repeat_first_search(self):
        data=self.fixture()
        with tempfile.TemporaryDirectory()as root:
            first=H3CheckpointSession(root,None,data[-1],process_stop=None)
            def window(seconds,**kwargs):
                if seconds==10 and first.proof_writes==1:raise H3SegmentPaused(root)
            first.require_window=window
            with patch('app.modules.card_game.rl.recovery_search.search',return_value=data[-2][0])as oracle:
                with self.assertRaises(H3SegmentPaused):self.prove(data,first)
                self.assertEqual(oracle.call_count,1)
            fresh=H3CheckpointSession(root,None,data[-1])
            with patch('app.modules.card_game.rl.recovery_search.search',return_value=data[-2][1])as oracle:
                refs=self.prove(data,fresh)
                self.assertEqual(oracle.call_count,1)
            self.assertEqual(set(refs),{(0,0),(0,1)})
            self.assertEqual(fresh.proof_hits,1);self.assertEqual(fresh.proof_writes,1)
            with patch('app.modules.card_game.rl.recovery_search.search',side_effect=AssertionError('No repeated root searches')):
                self.assertEqual(len(self.prove(data,H3CheckpointSession(root,None,data[-1]))),2)

    def test_gpu_input_model_and_reference_corruption_fail_closed(self):
        data=self.fixture()
        with tempfile.TemporaryDirectory()as root:
            session=H3CheckpointSession(root,None,data[-1])
            with patch('app.modules.card_game.rl.recovery_search.search',side_effect=data[-2]):self.prove(data,session)
            changed=list(data);changed[2]=deepcopy(data[2]);changed[2][0]['pi'][0]+=.1
            with self.assertRaisesRegex(ValueError,'input/GPU'):
                self.prove(changed,H3CheckpointSession(root,None,data[-1]))
            binding=deepcopy(data[-1]);binding['main_versions']=['different']
            with self.assertRaisesRegex(ValueError,'identity'):
                self.prove(data,H3CheckpointSession(root,None,binding))
            path=Path(root)/'proof-0000-0000'/'arrays.npz';content=bytearray(path.read_bytes());content[-1]^=1;path.write_bytes(content)
            with self.assertRaisesRegex(ValueError,'integrity'):
                self.prove(data,H3CheckpointSession(root,None,data[-1]))

    def test_actual_root_budget_exhaustion_is_not_a_planned_pause_or_cached_success(self):
        data=self.fixture()
        with tempfile.TemporaryDirectory()as root:
            session=H3CheckpointSession(root,None,data[-1])
            session.require_window=lambda *a,**kw:None
            clock=[0.]
            def slow(*args,**kwargs):clock[0]=100.;return data[-2][0]
            with patch('app.modules.card_game.rl.recovery_search.search',side_effect=slow),patch(
                    'app.modules.card_game.rl.batched_search.device_h3_collection.time.monotonic',side_effect=lambda:clock[0]):
                with self.assertRaises(TimeoutError):self.prove(data,session)
            self.assertFalse((Path(root)/'proof-0000-0000').exists())

    def test_terminal_checkpoint_skips_physical_gpu_sampling(self):
        from tests import test_duel_v2_device_h3_segments as fixtures
        loop=fixtures.PersistentLoop([]);loop.current=fixtures.result((10,),(True,))
        collector=fixtures.H3SegmentsTests().collector()
        checkpoint=NS(restored=True)
        self.assertTrue(replay_h3_chunks(collector,loop,NS(jobs=[dict(deadline=time.time()+600)]),checkpoint=checkpoint).complete.all())
        self.assertEqual(loop.calls,0)


class H3ProcessBudgetTests(unittest.TestCase):
    def test_insufficient_process_window_pauses_before_new_gpu_chunk_without_reject(self):
        from tests import test_duel_v2_device_h3_segments as fixtures
        loop=fixtures.PersistentLoop([]);collector=fixtures.H3SegmentsTests().collector()
        with tempfile.TemporaryDirectory()as root:
            session=H3CheckpointSession(root,None,dict(metadata=dict(jobs=[])),process_stop=time.monotonic()+5)
            with self.assertRaises(H3SegmentPaused):
                replay_h3_chunks(collector,loop,NS(jobs=[dict(deadline=time.time()+600)]),checkpoint=session)
            self.assertEqual(loop.calls,0)
            self.assertFalse((Path(root)/'REJECTED.json').exists())

    def test_gpu_window_reserves_snapshot_and_root_proof_window_reserves_full_budget(self):
        with tempfile.TemporaryDirectory()as root:
            session=H3CheckpointSession(root,None,dict(metadata=dict(jobs=[])),process_stop=time.monotonic()+70)
            session.require_window(45,margin=20)
            with self.assertRaises(H3SegmentPaused):session.require_window(60,margin=20)
            session.require_window(60,margin=5)


if __name__=='__main__':unittest.main()
