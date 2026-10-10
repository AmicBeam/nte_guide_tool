"""Resident fused covered_value continuation parity. No CUDA or training claims."""
from copy import deepcopy
import time
import unittest

import numpy as np

from app.modules.card_game.content.duel_v2.catalog import STARTER_DECKS
from app.modules.card_game.engine.duel_v2.flow import new_game
from app.modules.card_game.rl.batched_duel.process_backend import ResidentRuleBackend
from app.modules.card_game.rl.batched_search.cooperative import (
    CooperativeScheduler,
    OperationQuery,
    OperationResponse,
    RequestExpired,
    RootJob,
    _OPTIONAL_PROTOCOL_METHODS,
    _PROTOCOL_METHODS,
    _SUSPEND_METHODS,
    cooperative_search,
)
from app.modules.card_game.rl.batched_search.inference import InferenceResult
from app.modules.card_game.rl.batched_search.traversal import (
    InferenceQuery,
    _can_fuse_covered_value,
    _teacher_value_generator,
    search_traversal,
)
from tests.test_duel_v2_batch_traversal import drive_traversal
from tests.test_duel_v2_gpu_training_teachers import TeacherBackend


def _legacy_payload(backend, world, viewer, horizon, *, deadline=None, clock=time.monotonic):
    temps=[]
    current=world
    try:
        for step in range(horizon+1):
            if backend.finished(current):
                return ('terminal', backend.terminal_value(current, viewer))
            if deadline is not None and clock()>=deadline:
                return RequestExpired('legacy', step>0)
            if step==horizon:
                break
            actor=backend.acting_side(current)
            legal,chosen=backend.rule_decision(current, actor)
            nxt=backend.step(current, actor, legal[chosen])
            if nxt is not current:
                temps.append(nxt)
            cleaned=backend.clean(nxt)
            if cleaned is not nxt:
                temps.append(cleaned)
            current=cleaned
        x,c=backend.value_observation(current, viewer)
        return ('observation', x, c, backend.acting_side(current)==viewer)
    finally:
        seen=set(); unique=[]
        for item in temps:
            key=(getattr(item,'worker',None), getattr(item,'slot',None), getattr(item,'generation',None))
            if key==(None,None,None) or item is world or key in seen:
                continue
            seen.add(key); unique.append(item)
        if unique and hasattr(backend,'release'):
            backend.release(unique)


def _same_payload(test, actual, expected):
    test.assertEqual(actual[0], expected[0])
    if actual[0]=='terminal':
        test.assertEqual(actual[1], expected[1])
        return
    test.assertEqual(actual[0], 'observation')
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    test.assertEqual(bool(actual[3]), bool(expected[3]))


class FakeInference:
    def predict(self, requests):
        return [InferenceResult(r.request_id, r.model_key, r.model_version,
            np.zeros(len(r.candidates), dtype=np.float32),
            np.array([0.2, 0.1, 0.7], dtype=np.float32)) for r in requests]


class FusedTeacherBackend(TeacherBackend):
    supports_covered_value_continuation=True
    supports_mixed_operations=True
    supports_gpu=False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.methods=[]

    def covered_value_continuation(self, world, viewer, horizon, *, deadline=None):
        return _legacy_payload(self, world, viewer, horizon, deadline=deadline)

    def resolve_operations(self, queries):
        rows=[]
        for query in queries:
            self.methods.append(query.method)
            kwargs=dict(query.kwargs)
            if query.method=='covered_value_continuation' and query.deadline is not None:
                kwargs['deadline']=query.deadline
            rows.append(OperationResponse(query.id, getattr(self, query.method)(*query.args, **kwargs)))
        return rows


class CudaLikeTeacher(TeacherBackend):
    supports_gpu=True
    supports_covered_value_continuation=True

    def covered_value_continuation(self, *args, **kwargs):
        raise AssertionError('cuda_vm must keep the per-step teacher generator')


class FusedCoveredValueTest(unittest.TestCase):
    def test_protocol_keeps_fused_optional_and_capability_gated(self):
        self.assertNotIn('covered_value_continuation', _PROTOCOL_METHODS)
        self.assertIn('covered_value_continuation', _OPTIONAL_PROTOCOL_METHODS)
        self.assertIn('covered_value_continuation', _SUSPEND_METHODS)
        self.assertTrue(_can_fuse_covered_value(ResidentRuleBackend))
        self.assertFalse(_can_fuse_covered_value(CudaLikeTeacher()))
        self.assertFalse(_can_fuse_covered_value(TeacherBackend()))

    def test_fake_fused_matches_legacy_for_both_observers_and_horizons(self):
        backend=FusedTeacherBackend(terminal_depth=6)
        root={'phase':'playing','actor':'a','depth':0}
        for viewer in ('a','b'):
            for horizon in (0,1,2,3):
                with self.subTest(viewer=viewer, horizon=horizon):
                    expected=_legacy_payload(TeacherBackend(terminal_depth=6), deepcopy(root), viewer, horizon)
                    actual=backend.covered_value_continuation(deepcopy(root), viewer, horizon)
                    _same_payload(self, actual, expected)

    def test_cuda_like_backend_keeps_legacy_generator(self):
        backend=CudaLikeTeacher(terminal_depth=4)
        world={'phase':'playing','actor':'a','depth':0}
        def factory(actor,x,c,value_actor=True,value_only=False,auxiliary=False):
            return InferenceQuery('q','aux' if auxiliary else 'p','v',x,c,value_actor,actor,value_only)
        result=drive_traversal(_teacher_value_generator(
            world,'a',backend=backend,query_factory=factory,deadline=None,
            clock=time.monotonic,horizon=2,q_scale='natural_wdl'))
        self.assertEqual(result[1], True)
        self.assertTrue(np.isfinite(result[0]))

    def test_resident_payload_parity_input_preservation_and_reduced_rpc(self):
        game=new_game(seed=1101, decks={'a':STARTER_DECKS[0],'b':STARTER_DECKS[1]},
                      first_side='a', skip_mulligan=True)
        with ResidentRuleBackend(workers=1, max_games=2, max_states_per_worker=80) as backend:
            refs=backend.import_states([game, deepcopy(game)], ['live','done'], [1,1])
            live, done_root=refs
            finished=backend.step(done_root,'a',{'type':'concede'})
            self.assertTrue(backend.finished(finished))
            live_states=backend.stats()['live_states']
            exported=backend.export_states([live])[0]
            for viewer in ('a','b'):
                for horizon in (0,1,2,3):
                    with self.subTest(kind='live', viewer=viewer, horizon=horizon):
                        expected=_legacy_payload(backend, live, viewer, horizon)
                        actual=backend.covered_value_continuation(live, viewer, horizon)
                        _same_payload(self, actual, expected)
                with self.subTest(kind='terminal', viewer=viewer):
                    expected=_legacy_payload(backend, finished, viewer, 3)
                    actual=backend.covered_value_continuation(finished, viewer, 3, deadline=0.)
                    _same_payload(self, actual, expected)
                    self.assertEqual(actual[0], 'terminal')
            after=backend.export_states([live])[0]
            self.assertEqual(after, exported)
            self.assertEqual(after['phase'], exported['phase'])
            self.assertEqual(after['turn'], exported['turn'])
            self.assertEqual(after['sides']['a']['hp'], exported['sides']['a']['hp'])
            self.assertEqual(backend.stats()['live_states'], live_states)
            self.assertIsInstance(
                backend.covered_value_continuation(live,'a',3,deadline=0.), RequestExpired)
            before_rpc=backend.stats()['methods'].copy()
            _legacy_payload(backend, live, 'a', 3)
            after_legacy=backend.stats()['methods'].copy()
            backend.covered_value_continuation(live, 'a', 3)
            after_fused=backend.stats()['methods'].copy()
            worker_methods=('rule_decision','step','clean','value_observation',
                            'covered_value_continuation')
            self.assertEqual(sum(after_legacy.get(k,0)-before_rpc.get(k,0)
                                 for k in worker_methods),10)
            self.assertEqual(sum(after_fused.get(k,0)-after_legacy.get(k,0)
                                 for k in worker_methods),1)
            methods=backend.stats()['methods']
            self.assertGreaterEqual(methods.get('covered_value_continuation',0), 9)
            fused_ops=methods.get('covered_value_continuation',0)
            self.assertLess(fused_ops, methods.get('rule_decision',0)+methods.get('step',0)+methods.get('value_observation',0))
            policies={'a':('m','sha'),'b':('m','sha')}
            values={'a':('aux','auxv'),'b':('aux','auxv')}
            common=dict(seed=8, simulations=2, gumbel_candidates=2, max_depth=2,
                        terminal_horizon=3, q_scale='natural_wdl', teacher_mode='covered_value',
                        value_policy_identities_by_side=values)
            deadline=time.monotonic()+60
            fused_job=RootJob('fused','live',1, cooperative_search(
                live,'a',policies,backend=backend,deadline=deadline,**common), deadline)
            fused_search=CooperativeScheduler(FakeInference()).run([fused_job])['results']['fused']
            backend.supports_covered_value_continuation=False
            legacy_job=RootJob('legacy','done',1, cooperative_search(
                done_root,'a',policies,backend=backend,deadline=deadline,**common), deadline)
            legacy_search=CooperativeScheduler(FakeInference()).run([legacy_job])['results']['legacy']
            self.assertEqual(fused_search['search_choice'], legacy_search['search_choice'])
            np.testing.assert_array_equal(fused_search['visits'], legacy_search['visits'])
            np.testing.assert_allclose(fused_search['pi'], legacy_search['pi'])
            np.testing.assert_allclose(fused_search['mean_values'], legacy_search['mean_values'])

    def test_expired_foreign_stale_and_mixed_batch_keep_ids(self):
        game=new_game(seed=1102, first_side='a', skip_mulligan=True)
        with ResidentRuleBackend(workers=1, max_games=1, max_states_per_worker=40) as backend:
            [ref]=backend.import_states([game],['one'],[1])
            with self.assertRaises(ValueError):
                backend.resolve_operations([
                    OperationQuery('foreign', object(), 'covered_value_continuation', (ref,'a',0), ())])
            with self.assertRaises(ValueError):
                backend.resolve_operations([
                    OperationQuery('bad-horizon', backend, 'covered_value_continuation', (ref,'a',-1), ())])
            with self.assertRaises(TypeError):
                backend.resolve_operations([
                    OperationQuery('bool-horizon', backend, 'covered_value_continuation', (ref,'a',True), ())])
            mixed=[
                OperationQuery('m1', backend, 'acting_side', (ref,), ()),
                OperationQuery('m2', backend, 'covered_value_continuation', (ref,'a',0), ()),
                OperationQuery('m3', backend, 'value_observation', (ref,'a'), ()),
            ]
            responses=backend.resolve_operations(mixed)
            self.assertEqual([row.id for row in responses], ['m1','m2','m3'])
            self.assertEqual(responses[0].value, 'a')
            self.assertEqual(responses[1].value[0], 'observation')
            np.testing.assert_array_equal(responses[1].value[1], responses[2].value[0])
            np.testing.assert_array_equal(responses[1].value[2], responses[2].value[1])
            self.assertEqual(responses[1].value[3], True)
            backend.release([ref])
            with self.assertRaises(ValueError):
                backend.covered_value_continuation(ref,'a',0)

    def test_cooperative_traversal_parity_and_fewer_rule_waves(self):
        fused=FusedTeacherBackend(terminal_depth=8)
        legacy=TeacherBackend(terminal_depth=8)
        class BulkLegacy(TeacherBackend):
            supports_mixed_operations=True
            def __init__(self):
                super().__init__(terminal_depth=8); self.methods=[]
            def resolve_operations(self, queries):
                rows=[]
                for query in queries:
                    self.methods.append(query.method)
                    rows.append(OperationResponse(query.id, getattr(TeacherBackend, query.method)(self, *query.args)))
                return rows
        bulk_legacy=BulkLegacy()
        root={'phase':'playing','actor':'a','depth':0}
        policies={'a':('a','v1'),'b':('b','v1')}
        values={'a':('aux','auxv'),'b':('aux','auxv')}
        common=dict(seed=21, simulations=4, gumbel_candidates=2, max_depth=2,
                    terminal_horizon=3, q_scale='natural_wdl', teacher_mode='covered_value',
                    value_policy_identities_by_side=values)
        deadline=time.monotonic()+60
        fused_job=RootJob('fused','g1',1, cooperative_search(
            root,'a',policies, backend=fused, deadline=deadline, **common), deadline)
        legacy_job=RootJob('legacy','g2',1, cooperative_search(
            root,'a',policies, backend=bulk_legacy, deadline=deadline, **common), deadline)
        run=CooperativeScheduler(FakeInference()).run([fused_job, legacy_job])
        actual=run['results']['fused']
        expected=run['results']['legacy']
        self.assertEqual(actual['search_choice'], expected['search_choice'])
        np.testing.assert_array_equal(actual['visits'], expected['visits'])
        np.testing.assert_allclose(actual['pi'], expected['pi'])
        np.testing.assert_allclose(actual['mean_values'], expected['mean_values'])
        self.assertIn('covered_value_continuation', fused.methods)
        self.assertNotIn('covered_value_continuation', bulk_legacy.methods)
        self.assertLess(fused.methods.count('rule_decision'), bulk_legacy.methods.count('rule_decision'))
        self.assertLess(fused.methods.count('value_observation'), bulk_legacy.methods.count('value_observation'))
        direct=drive_traversal(search_traversal(root,'a',policies, backend=legacy, **common),
            response_provider=lambda query:(np.zeros(len(query.c)), np.array([0.2,0.1,0.7])))
        self.assertEqual(actual['search_choice'], direct['search_choice'])


if __name__=='__main__':
    unittest.main()
