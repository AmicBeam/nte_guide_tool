"""Canonical GPU teacher ranking: policy identity, ties, fail-closed rows."""
import unittest
from unittest.mock import patch

from app.modules.card_game.rl.batched_search.device_teacher_ranking import (
    CLOSED_SERVICE_READY, DeviceTeacherRanking, ERR_BATTLE, ERR_COUNT, ERR_EMPTY,
    ERR_KIND, ERR_MASK, ERR_NONFINITE, KIND_ATTACK, KIND_CHOOSE, KIND_CONCEDE,
    KIND_CYCLE, KIND_END, KIND_MULLIGAN, KIND_PLAY, KIND_ULTIMATE,
    canonical_teacher_ranking_identity)


def _cpu(rows=4, max_actions=6):
    return DeviceTeacherRanking(
        rows=rows, max_actions=max_actions, device='cpu', allow_cpu_reference=True)


def _zeros(engine):
    import torch as t
    b, m, d = engine.rows, engine.max_actions, engine.device
    return dict(
        kinds=t.full((b, m), KIND_END, dtype=t.int64, device=d),
        battle=t.zeros((b, m), dtype=t.bool, device=d),
        attack=t.zeros((b, m), dtype=t.float64, device=d),
        attack_count=t.zeros((b, m), dtype=t.float64, device=d),
        genesis_damage=t.zeros((b, m), dtype=t.float64, device=d),
        counter=t.zeros((b, m), dtype=t.float64, device=d),
        valid=t.zeros((b, m), dtype=t.bool, device=d),
        counts=t.zeros(b, dtype=t.int64, device=d),
        active=t.zeros(b, dtype=t.bool, device=d),
        input_errors=t.zeros(b, dtype=t.int32, device=d),
    )


def _legal(payload, row, kinds, battle=None, attack=None, attack_count=None,
           genesis=None, counter=None):
    n = len(kinds)
    payload['counts'][row] = n
    payload['active'][row] = True
    payload['valid'][row, :n] = True
    payload['kinds'][row, :n] = payload['kinds'].new_tensor(kinds)
    if battle is not None:
        payload['battle'][row, :n] = payload['battle'].new_tensor(battle)
    if attack is not None:
        payload['attack'][row, :n] = payload['attack'].new_tensor(attack)
    if attack_count is not None:
        payload['attack_count'][row, :n] = payload['attack_count'].new_tensor(attack_count)
    if genesis is not None:
        payload['genesis_damage'][row, :n] = payload['genesis_damage'].new_tensor(genesis)
    if counter is not None:
        payload['counter'][row, :n] = payload['counter'].new_tensor(counter)


def _rank(engine, payload):
    return engine.rank_device(
        payload['kinds'], payload['battle'], payload['attack'], payload['attack_count'],
        payload['genesis_damage'], payload['counter'], payload['valid'], payload['counts'],
        payload['active'], payload['input_errors'])


class DeviceTeacherRankingTest(unittest.TestCase):
    def test_cpu_oracle_matches_visible_policy_and_zero_attack_count(self):
        engine = _cpu()
        payload = _zeros(engine)
        _legal(payload, 0,
               [KIND_ATTACK, KIND_ATTACK, KIND_END],
               battle=[True, True, False],
               attack=[1.0, 0.0, 0.0],
               attack_count=[0.0, 2.0, 0.0],
               genesis=[0.0, 0.5, 0.0],
               counter=[0.0, 0.0, 0.0])
        _legal(payload, 1,
               [KIND_PLAY, KIND_ULTIMATE, KIND_CYCLE, KIND_CONCEDE],
               battle=[False, False, False, False])
        result = _rank(engine, payload)
        self.assertEqual(result.identity, canonical_teacher_ranking_identity())
        self.assertFalse(CLOSED_SERVICE_READY)
        self.assertFalse(engine.closed_service_ready)
        self.assertEqual(int(result.choices[0]), 0)
        self.assertEqual(int(result.choices[1]), 1)
        self.assertEqual(float(result.scores[0, 0]), 4.0)
        self.assertEqual(float(result.scores[0, 1]), 3.5)
        self.assertEqual(float(result.scores[0, 2]), -100.0)
        self.assertEqual(float(result.scores[1, 0]), 2.0)
        self.assertEqual(float(result.scores[1, 1]), 6.0)
        self.assertEqual(float(result.scores[1, 2]), 1.0)
        self.assertEqual(float(result.scores[1, 3]), -1000.0)
        self.assertEqual(result.errors.tolist(), [0, 0, 0, 0])

    def test_first_mulligan_then_choose_beats_higher_scores(self):
        engine = _cpu()
        payload = _zeros(engine)
        _legal(payload, 0,
               [KIND_ATTACK, KIND_PLAY, KIND_MULLIGAN],
               battle=[True, True, False],
               attack=[9.0, 8.0, 0.0],
               attack_count=[3.0, 3.0, 0.0])
        _legal(payload, 1,
               [KIND_ULTIMATE, KIND_CHOOSE, KIND_ATTACK],
               battle=[False, False, True],
               attack=[0.0, 0.0, 9.0],
               attack_count=[0.0, 0.0, 2.0])
        result = _rank(engine, payload)
        self.assertEqual(int(result.choices[0]), 2)
        self.assertEqual(int(result.choices[1]), 1)
        self.assertEqual(result.errors.tolist(), [0, 0, 0, 0])

    def test_stable_tied_ranking_prefers_original_ordinal(self):
        engine = _cpu()
        payload = _zeros(engine)
        _legal(payload, 0,
               [KIND_PLAY, KIND_PLAY, KIND_PLAY],
               battle=[True, True, True],
               attack=[2.0, 2.0, 2.0],
               attack_count=[1.0, 1.0, 1.0],
               genesis=[0.0, 0.0, 0.0],
               counter=[0.0, 0.0, 0.0])
        result = _rank(engine, payload)
        self.assertEqual(int(result.choices[0]), 0)
        self.assertEqual(result.scores[0, :3].tolist(), [5.0, 5.0, 5.0])

    def test_inactive_padding_does_not_overwrite_other_rows(self):
        engine = _cpu()
        payload = _zeros(engine)
        _legal(payload, 0, [KIND_END])
        payload['kinds'][1, 0] = KIND_ULTIMATE
        payload['valid'][1, 0] = True
        payload['counts'][1] = 1
        payload['active'][1] = False
        payload['attack'][1, 0] = 99.0
        result = _rank(engine, payload)
        self.assertEqual(int(result.choices[0]), 0)
        self.assertEqual(int(result.choices[1]), -1)
        self.assertEqual(float(result.scores[1, 0]), 0.0)
        self.assertEqual(int(result.errors[1]), 0)

    def test_fail_closed_nonfinite_malformed_empty_and_unknown_kind(self):
        engine = _cpu()
        payload = _zeros(engine)
        _legal(payload, 0, [KIND_ATTACK], battle=[True], attack=[float('nan')])
        _legal(payload, 1, [KIND_END, KIND_END])
        payload['valid'][1, 1] = False
        _legal(payload, 2, [KIND_END])
        payload['counts'][2] = 0
        payload['valid'][2].zero_()
        _legal(payload, 3, [99], battle=[False])
        result = _rank(engine, payload)
        self.assertEqual(int(result.errors[0]), ERR_NONFINITE)
        self.assertEqual(int(result.errors[1]), ERR_MASK)
        self.assertEqual(int(result.errors[2]), ERR_EMPTY)
        self.assertEqual(int(result.errors[3]), ERR_KIND)
        self.assertEqual(result.choices.tolist(), [-1, -1, -1, -1])
        payload = _zeros(engine)
        _legal(payload, 0, [KIND_CYCLE], battle=[True])
        payload['counts'][1] = 9
        payload['active'][1] = True
        result = _rank(engine, payload)
        self.assertEqual(int(result.errors[0]), ERR_BATTLE)
        self.assertEqual(int(result.errors[1]), ERR_COUNT)

    def test_cpu_reference_must_be_explicit_and_source_freeze_rejects_unknown_abi(self):
        with self.assertRaises(RuntimeError):
            DeviceTeacherRanking(rows=2, max_actions=2, device='cpu')
        with patch(
                'inspect.getsource',
                return_value='not the canonical ranking'):
            with self.assertRaises(RuntimeError):
                canonical_teacher_ranking_identity()

    def test_actual_cuda_replay_reset_and_hot_path_has_no_host_sync(self):
        import torch as t
        if not t.cuda.is_available():
            self.skipTest('Actual CUDA required; ranking primitive only')
        engine = DeviceTeacherRanking(rows=4, max_actions=6, device='cuda')
        self.addCleanup(engine.close)
        payload = _zeros(engine)
        _legal(payload, 0,
               [KIND_PLAY, KIND_PLAY],
               battle=[True, True],
               attack=[2.0, 2.0],
               attack_count=[1.0, 1.0])
        _legal(payload, 1, [KIND_MULLIGAN, KIND_ATTACK], battle=[False, True],
               attack=[0.0, 8.0], attack_count=[0.0, 1.0])
        payload['active'][2] = False
        with patch.object(t.Tensor, 'cpu', side_effect=AssertionError('CPU copy')), patch.object(t.Tensor, 'numpy', side_effect=AssertionError('NumPy copy')), patch.object(t.Tensor, 'item', side_effect=AssertionError('Scalar copy')), patch.object(t.Tensor, 'tolist', side_effect=AssertionError('Host list')), patch.object(t.Tensor, '__bool__', side_effect=AssertionError('Host branch')):
            result = _rank(engine, payload)
        self.assertEqual(int(result.choices[0].cpu()), 0)
        self.assertEqual(int(result.choices[1].cpu()), 0)
        for _ in range(2):
            _rank(engine, payload)
        t.cuda.synchronize()
        graph = t.cuda.CUDAGraph()
        with t.cuda.graph(graph):
            captured = _rank(engine, payload)
        payload['attack'][0, 1] = 9.0
        graph.replay()
        t.cuda.synchronize()
        self.assertEqual(int(captured.choices[0].cpu()), 1)
        engine.reset_device()
        self.assertEqual(int(engine.choices[0].cpu()), -1)

    def test_seven_extra_actions_match_formal_teacher(self):
        from app.modules.card_game.rl.batched_search.device_teacher_ranking import KIND_CODES
        from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
        names = ('choose_cards', 'choose_deck_order', 'exit_ultimate')
        engine = _cpu(rows=3, max_actions=3)
        payload = _zeros(engine)
        for row, name in enumerate(names):
            _legal(payload, row, [KIND_CODES[name], KIND_ATTACK, KIND_END],
                   attack=[3., 3., 0.])
        result = _rank(engine, payload)
        policy = VisibleEngineRulePolicy()
        for row, name in enumerate(names):
            actions = ({'type': name}, {'type': 'attack'}, {'type': 'end_turn'})
            observation = {'legal_actions': [{'action': a, 'preview': {'attack': 3.}}
                                             for a in actions]}
            self.assertEqual(int(result.choices[row]), policy(observation, actions))
            self.assertEqual(result.scores[row].tolist(), policy.ranking_scores(observation, actions))

    def test_source_change_preserving_old_fragments_is_rejected(self):
        import inspect
        from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
        original = inspect.getsource
        altered = original(VisibleEngineRulePolicy).replace(
            "            kind = action.get('type')",
            "            kind = action.get('type')\n            if kind == 'attack':\n                return 100.0")
        def get_source(obj):
            return altered if obj is VisibleEngineRulePolicy else original(obj)
        with patch('inspect.getsource', side_effect=get_source):
            with self.assertRaisesRegex(RuntimeError, 'Unsupported canonical'):
                canonical_teacher_ranking_identity()

    def test_finite_inputs_with_overflow_fail_closed(self):
        engine = _cpu(rows=1, max_actions=2)
        payload = _zeros(engine)
        _legal(payload, 0, [KIND_ATTACK, KIND_END],
               attack=[1.e308, 0.], attack_count=[1.e308, 0.])
        result = _rank(engine, payload)
        self.assertEqual(int(result.choices[0]), -1)
        self.assertEqual(int(result.errors[0]), ERR_NONFINITE)

    def test_actual_cuda_full_kind_and_invalid_row_policy_parity(self):
        import torch as t
        from app.modules.card_game.rl.batched_search.device_teacher_ranking import KIND_NAMES, KIND_CODES
        if not t.cuda.is_available():
            self.skipTest('Actual CUDA required; full teacher ranking parity pending')
        cpu = _cpu(rows=15, max_actions=4)
        payload = _zeros(cpu)
        for row, name in enumerate(KIND_NAMES):
            _legal(payload, row, [KIND_CODES[name], KIND_END],
                   attack=[2.5, 0.], attack_count=[-0., 0.],
                   genesis=[.5, 0.], counter=[.25, 0.])
        _legal(payload, 11, [KIND_ATTACK, KIND_END], battle=[True, False])
        payload['kinds'][11, 0] = 99
        _legal(payload, 12, [KIND_ATTACK, KIND_END],
               attack=[1.e308, 0.], attack_count=[1.e308, 0.])
        _legal(payload, 13, [KIND_ATTACK, KIND_MULLIGAN], attack=[float('nan'), 0.])
        _legal(payload, 14, [KIND_ATTACK, KIND_ATTACK], attack=[3., 3.])
        expected = _rank(cpu, payload)
        gpu = DeviceTeacherRanking(rows=15, max_actions=4, device='cuda')
        self.addCleanup(gpu.close)
        received = _rank(gpu, {key: value.to(gpu.device) for key, value in payload.items()})
        t.cuda.synchronize()
        self.assertTrue(t.equal(received.choices.cpu(), expected.choices))
        self.assertTrue(t.equal(received.errors.cpu(), expected.errors))
        good = expected.errors == 0
        self.assertTrue(t.equal(received.scores.cpu()[good], expected.scores[good]))


class CanonicalTeacherBatchReferenceTest(unittest.TestCase):
    def test_fourteen_real_seven_team_roots_match_formal_teacher_and_preserve_state(self):
        """CPU integration oracle only; no GPU capacity or search qualification."""
        import copy
        import torch as t
        from app.modules.card_game.content.duel_v2 import get_catalog
        from app.modules.card_game.engine.duel_v2 import new_game, acting_side
        from app.modules.card_game.engine.duel_v2.projection import RULE_PREVIEW_SCORE_FIELDS
        from app.modules.card_game.rl.batched_duel.gpu_teacher_preview_parts import (
            teacher_prepare, teacher_preview, teacher_finish)
        from app.modules.card_game.rl.batched_search.device_teacher_ranking import KIND_CODES
        from app.modules.card_game.rl.batched_search.device_teacher_preview_jobs import DeviceTeacherPreviewJobs
        decks = get_catalog()['starter_decks']
        cases = []
        for skip in (False, True):
            for index, deck in enumerate(decks):
                state = new_game(seed=12000 + index, decks={'a': deck, 'b': deck},
                                 first_side='a', skip_mulligan=skip)
                before = copy.deepcopy(state)
                actor = acting_side(state)
                view, legal, indices = teacher_prepare(state, actor)
                previews = [teacher_preview(state, actor, view['legal_actions'][i]['action'])
                            for i in indices]
                _, expected = teacher_finish(view, legal, indices, previews)
                positions = [legal.index(view['legal_actions'][i]['action']) for i in indices]
                self.assertEqual(state, before)
                cases.append((actor, legal, positions, previews, expected))
        self.assertEqual(len(cases), 14)
        b, m = len(cases), max(len(case[1]) for case in cases)
        ranking = _cpu(rows=b, max_actions=m)
        payload = _zeros(ranking)
        for row, (actor, legal, positions, previews, expected) in enumerate(cases):
            _legal(payload, row, [KIND_CODES[a['type']] for a in legal])
            payload['battle'][row, positions] = True
        jobs = DeviceTeacherPreviewJobs(rows=b, max_actions=m, capacity=b*m,
                                       device='cpu', allow_cpu_reference=True)
        self.addCleanup(jobs.close)
        owners = t.arange(b, dtype=t.int64)
        actors = t.tensor([int(case[0] == 'b') for case in cases], dtype=t.int64)
        work = jobs.compact_device(payload['valid'], payload['battle'], payload['counts'],
            payload['active'], payload['input_errors'], owners, t.zeros_like(owners), owners, actors)
        count = int(work.job_count[0])
        if count:
            ids = t.arange(count, dtype=t.int64)
            values = t.zeros((count, 4), dtype=t.float64)
            for job in range(count):
                row, ordinal = int(work.parent_row[job]), int(work.ordinal[job])
                case = cases[row]
                preview = case[3][case[2].index(ordinal)]
                values[job] = t.tensor([float(preview.get(key) or 0)
                                       for key in RULE_PREVIEW_SCORE_FIELDS], dtype=t.float64)
            jobs.scatter_previews_device(ids, work.parent_row[:count].clone(),
                work.ordinal[:count].clone(), work.owners[:count].clone(),
                work.generations[:count].clone(), work.slots[:count].clone(),
                t.ones(count, dtype=t.bool), *(values[:,i].contiguous() for i in range(4)),
                payload['attack'], payload['attack_count'], payload['genesis_damage'],
                payload['counter'], payload['input_errors'],
                epochs=t.ones(count,dtype=t.int64)*work.epoch[0], actors=work.actors[:count].clone())
        self.assertTrue(bool(jobs.finalize_device(payload['input_errors']).all()))
        result = _rank(ranking, payload)
        self.assertEqual(result.errors.tolist(), [0]*b)
        self.assertEqual(result.choices.tolist(), [case[-1] for case in cases])


if __name__ == '__main__':
    unittest.main()
