"""CPU restoration/rejection protocol checks; no CUDA qualification claims."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from app.modules.card_game.rl.batched_search.device_pure_collection import _ValueAuditSink
from app.modules.card_game.rl.batched_search.device_pure_segments import (
    _snapshot, _upload_history, _restore_history, _validate_history, _check_progress,
    collect_segments,
)
from app.modules.card_game.rl.batched_search.device_raw_action_tape import DeviceRawActionTape
from tests import test_duel_v2_device_pure_value_tape as value_helpers
from tests.test_duel_v2_device_raw_action_tape import _graph, _actions


class PureSegmentsTests(unittest.TestCase):
    def fixture(self):
        helper = value_helpers.PureValueTapeTests()
        tape, rules, roots = helper.fixture(train=torch.ones((2, 2), dtype=torch.bool))
        roots = replace(roots, owners=torch.arange(2))
        roots.pool.owner.copy_(torch.arange(32) % 2)
        raw = DeviceRawActionTape(2, 8, 256, 256, 1024, device='cpu', allow_cpu_reference=True)
        self.addCleanup(raw.close)
        audit = _ValueAuditSink(tape, decisions=4)
        ids = torch.zeros((2, 2), dtype=torch.int64)
        seeds = torch.tensor([123, 456])
        raw.begin_device(roots.owners, seeds)
        audit.begin_device(roots, ids, ids, seeds, torch.ones((2, 2), dtype=torch.bool), roots.active)
        return helper, tape, rules, roots, raw, audit

    def append(self, fixture, step, active=None):
        helper, tape, rules, roots, raw, audit = fixture
        if active is not None:
            roots = replace(roots, active=torch.tensor(active))
        result = helper.pure(rules, roots)
        result.information_keys = torch.arange(16).reshape(2, 8) + step * 16
        steps = torch.full((2,), step, dtype=torch.int64)
        errors = audit.append_device(roots, rules.actors, steps, result, roots.active)
        self.assertEqual(errors.tolist(), [0, 0])
        errors = raw.append_device(_graph(_actions(), 'cpu'), torch.arange(2), roots.owners,
            steps, rules.versions, rules.actors, roots.active)
        self.assertEqual(errors.tolist(), [0, 0])
        return roots

    def history(self):
        fixture = self.fixture()
        helper, tape, rules, roots, raw, audit = fixture
        self.append(fixture, 0)
        initial = [{'version': 7, 'phase': 'playing'}, {'version': 8, 'phase': 'playing'}]
        final = [{'version': 8, 'phase': 'finished'}, {'version': 9, 'phase': 'playing'}]
        rules.versions.add_(1)
        rules.finished[0] = True
        rules.winners[0] = 0
        finishing = replace(roots, active=torch.tensor([True, False]))
        self.assertEqual(tape.finish_device(finishing, rules.winners, torch.ones(2, dtype=torch.int64),
            torch.tensor([True, False]), torch.zeros(2, dtype=torch.int32)).tolist(), [0, 0])
        history = _snapshot(initial, final, [1, 1], [True, False], raw, tape, audit)
        self.assertTrue(_validate_history(history, 2, 4))
        return history, fixture

    def test_fresh_arena_restores_raw_hash_choice_and_both_viewer_prefix(self):
        history, old = self.history()
        new = self.fixture()
        helper, tape, rules, roots, raw, audit = new
        uploaded = _upload_history(history, raw, tape, audit)
        # Simulate the actual graph's normal begin/reset before restore.
        self.assertEqual(int(raw._record_count), 0)
        self.assertEqual(int(tape._count), 0)
        _restore_history(raw, tape, audit, uploaded)
        for obj, saved in ((raw, history.raw), (tape, history.value), (audit, history.audit)):
            for name, value in saved.items():
                self.assertTrue(torch.equal(getattr(obj, name), value), name)
        self.assertIs(tape._pool, roots.pool)
        self.assertIsNot(tape._pool, old[3].pool)
        rules.versions.copy_(torch.tensor([8, 9]))
        rules.finished.copy_(torch.tensor([True, False]))
        rules.winners.copy_(torch.tensor([0, -2]))
        self.append(new, 1, [False, True])
        rules.versions[1] += 1
        rules.finished[1] = True
        rules.winners[1] = 1
        finishing = replace(roots, active=torch.tensor([False, True]))
        self.assertEqual(tape.finish_device(finishing, rules.winners, torch.tensor([1, 2]),
            torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.int32)).tolist(), [0, 0])
        exported = tape.value_rows_after_batch(include_frozen=True)
        self.assertEqual(len(exported), 6)
        self.assertEqual({(r['owner'], r['step'], r['viewer']) for r in exported},
            {(owner, step, viewer) for owner, count in enumerate((1, 2)) for step in range(count) for viewer in (0, 1)})
        for row in exported:
            self.assertTrue(row['learning_eligible'])
            self.assertEqual(row['z'], 1 if row['owner'] == row['viewer'] else -1)
        self.assertTrue(torch.equal(tape._x[:2], history.value['_x'][:2]))
        self.assertEqual(raw._expected_step.tolist(), [1, 2])
        self.assertEqual(raw._last_version.tolist(), [7, 9])
        self.assertTrue(torch.equal(audit.hashes[:5], history.audit['hashes'][:5]))
        self.assertFalse(history.value['_eligible'][1].any(), 'Saved pending prefix stays ineligible')

    def test_guard_gap_duplicate_version_capacity_and_partial_labels_rejected(self):
        history, _ = self.history()
        mutators = [
            lambda h: h.raw['_game_errors'].__setitem__(0, 256),
            lambda h: h.value['_game_errors'].__setitem__(0, 1),
            lambda h: h.audit['errors'].__setitem__(0, 1),
            lambda h: h.audit['valid'].__setitem__(0, False),
            lambda h: h.raw['_rec_step'].__setitem__(1, 2),
            lambda h: h.raw['_expected_step'].__setitem__(0, 0),
            lambda h: h.raw['_last_version'].__setitem__(1, 99),
            lambda h: h.value['_eligible'].__setitem__((1, 0), True),
            lambda h: h.value['_z'].__setitem__((1, 1), 1),
            lambda h: h.value['_x'].__setitem__((1, 0, 0), float('nan')),
            lambda h: h.raw['_node_used'].fill_(257),
            lambda h: h.final[1].__setitem__('version', 100),
        ]
        for mutate in mutators:
            with self.subTest(mutate=mutate):
                corrupt = deepcopy(history)
                mutate(corrupt)
                with self.assertRaises(ValueError):
                    _validate_history(corrupt, 2, 4)

    def test_partial_batch_cannot_export(self):
        history, fixture = self.history()
        with self.assertRaisesRegex(RuntimeError, 'entirely complete'):
            fixture[1].value_rows_after_batch(include_frozen=True)

    def test_layout_mismatch_and_progress_guards(self):
        history, _ = self.history()
        fixture = self.fixture()
        history.audit['hashes'] = history.audit['hashes'][:-1]
        with self.assertRaisesRegex(ValueError, 'layout'):
            _upload_history(history, fixture[4], fixture[1], fixture[5])
        _check_progress([16, 1], [0, 1], [False, True], 800, 16)
        for now, before, done in (([0], [0], [False]), ([17], [0], [False]), ([800], [784], [False])):
            with self.assertRaises(ValueError):
                _check_progress(now, before, done, 800, 16)

    def test_segment_error_does_not_resume_or_emit_learning(self):
        collector = SimpleNamespace(max_decisions=800, segment_decisions=16, seconds=1,
            proof_seconds=1, last_diagnostics={}, on_segment=None)
        metadata = SimpleNamespace(jobs=[{'deadline': 10**12}])
        with patch('app.modules.card_game.rl.batched_search.device_pure_segments._segment',
                   side_effect=RuntimeError('guard')) as segment:
            with self.assertRaisesRegex(RuntimeError, 'guard'):
                collect_segments(collector, metadata, {})
        self.assertEqual(segment.call_count, 1)

    def test_planned_pause_returns_only_after_complete_batch_and_keeps_deadline(self):
        callbacks, arenas = [], []
        collector = SimpleNamespace(max_decisions=800, segment_decisions=16, seconds=1,
            proof_seconds=1, last_diagnostics={}, on_segment=callbacks.append)
        metadata = SimpleNamespace(jobs=[{'deadline': 10**12}])
        frame, finished = object(), [object()]
        def segment(owner, actual, models, owned, resume):
            self.assertIs(actual, metadata)
            arenas.append(owned)
            if resume is None:
                return frame, None, dict(steps=[16], complete_rows=[False], natural_games=0)
            self.assertIs(resume, frame)
            return None, finished, dict(steps=[20], complete_rows=[True], natural_games=1)
        with patch('app.modules.card_game.rl.batched_search.device_pure_segments._segment', side_effect=segment):
            self.assertIs(collect_segments(collector, metadata, {}), finished)
        self.assertIsNot(arenas[0], arenas[1])
        self.assertEqual(metadata.jobs[0]['deadline'], 10**12)
        self.assertEqual([c['complete'] for c in callbacks], [False, True])
        self.assertEqual(len(callbacks[-1]['segments']), 2)


if __name__ == '__main__':
    unittest.main()
