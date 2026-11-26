"""Durable training-flow protocol tests. Synthetic fixtures are not CUDA quality."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import io
import json
import struct
import zipfile
import unittest

from app.modules.card_game.rl.training_flow import (
    CANDIDATE_STAGES,
    STAGE_ORDER,
    TrainingFlow,
    confirmation_questions,
)

SHANGHAI = timezone(timedelta(hours=8))
NOW = datetime(2026, 10, 9, 16, 0, tzinfo=SHANGHAI)
DEADLINE = '2026-10-10T09:00:00+08:00'
AFTER = datetime(2026, 10, 10, 9, 0, tzinfo=SHANGHAI)


def sample_build(prefix):
    return {
        'character_ids': ['%s-c%s' % (prefix, i) for i in range(4)],
        'card_ids': ['%s-k%02d' % (prefix, i) for i in range(32)],
    }


def sample_request(include_candidates=False):
    roster = ['alpha', 'beta']
    request = {
        'include_candidates': include_candidates,
        'hard_deadline': DEADLINE,
        'roster': roster,
        'builds': {key: sample_build(key) for key in roster},
        'default_agent': 'grok',
    }
    if include_candidates:
        request['mutable_characters'] = {'alpha': ['alpha-c0'], 'beta': ['beta-c0']}
    return request


def receipt(request):
    return {
        'source': 'user',
        'reference': 'msg-confirm-1',
        'include_candidates': request['include_candidates'],
        'hard_deadline': request['hard_deadline'],
        'roster': list(request['roster']),
        'builds': deepcopy(request['builds']),
        'mutable_characters': deepcopy(request.get('mutable_characters', {})),
    }


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {'path': str(path), 'sha256': sha256(data).hexdigest()}


def numeric_fixture():
    buffer = io.BytesIO()
    header = repr({'descr': '<f4', 'fortran_order': False, 'shape': (1,)}).encode('ascii') + b'\n'
    array = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + struct.pack('<f', 1.0)
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('weights.npy', array)
    return buffer.getvalue()


def matrix_fixture():
    return {'planned_cells': 4, 'completed_cells': 4, 'matrix_modes': {
        mode: {'planned_cells': 4, 'completed_cells': 4, 'cells': [
            {'row': row, 'column': col, 'planned': 2, 'completed': 2,
             'first_wins': 1, 'second_wins': 1, 'draws': 0, 'errors': 0, 'truncated': 0}
            for row in ('alpha', 'beta') for col in ('alpha', 'beta')]}
        for mode in ('pure', 'search')}}


def raw_matrix_fixture():
    records = []
    for mode, data in matrix_fixture()['matrix_modes'].items():
        for cell in data['cells']:
            for first in ('a', 'b'):
                records.append({'id': str(len(records)), 'stage': 'matrix', 'mode': mode,
                                'left': cell['row'] if first == 'a' else cell['column'],
                                'right': cell['column'] if first == 'a' else cell['row'],
                                'first': first, 'winner': 'a', 'complete': True, 'error': None})
    return json.dumps(records).encode()


def note_result(assignment, name='note.txt', data=b'ok', extra_metrics=None, extra_files=None):
    artifacts = [write_bytes(Path(assignment['output_directory']) / name, data)]
    for filename, payload in (extra_files or {}).items():
        artifacts.append(write_bytes(Path(assignment['output_directory']) / filename, payload))
    metrics = {} if extra_metrics is None else dict(extra_metrics)
    return {
        'verdict': 'passed',
        'summary': 'synthetic plumbing fixture; not CUDA or Gumbel quality evidence',
        'artifacts': artifacts,
        'metrics': metrics,
        'request_sha256': assignment['request_sha256'],
    }


class TrainingFlowTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name) / 'run'
        self.root.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def create(self, include_candidates=False, now=NOW):
        return TrainingFlow.create(self.root, sample_request(include_candidates), now=now)

    def confirmed(self, include_candidates=False, now=NOW):
        request = sample_request(include_candidates)
        flow = TrainingFlow.create(self.root, request, now=now)
        flow.confirm(receipt(request), now=now)
        flow.configure_executors({name: lambda assignment: None for name in STAGE_ORDER if name != 'confirmation' and (include_candidates or name not in CANDIDATE_STAGES)})
        return flow

    def test_confirmation_questions_cover_three_opening_decisions(self):
        questions = confirmation_questions()
        self.assertEqual(len(questions), 3)
        self.assertTrue(all(isinstance(question, str) and question.strip() for question in questions))

    def test_create_is_draft_and_confirm_rejects_mismatch(self):
        request = sample_request(False)
        flow = TrainingFlow.create(self.root, request, now=NOW)
        snap = flow.snapshot()
        self.assertEqual(snap['schema'], 'duel_v2_complete_training_flow_v1')
        self.assertIsNone(snap['confirmation'])
        self.assertFalse(snap['workflow_complete'])
        self.assertFalse(snap['quality_approved'])
        self.assertFalse(snap['complete_report'])
        self.assertEqual(Path(snap['directory']), flow.directory.resolve())
        self.assertEqual(flow.next_stage(), 'confirmation')
        with self.assertRaises(ValueError):
            flow.start('freeze', now=NOW)
        with self.assertRaises(ValueError):
            flow.start('confirmation', now=NOW)
        bad = receipt(request)
        bad['source'] = 'timeout'
        with self.assertRaises(ValueError):
            flow.confirm(bad, now=NOW)
        omitted = dict(receipt(request))
        omitted.pop('hard_deadline')
        with self.assertRaises(ValueError):
            flow.confirm(omitted, now=NOW)
        mismatch = receipt(request)
        mismatch['include_candidates'] = True
        with self.assertRaises(ValueError):
            flow.confirm(mismatch, now=NOW)
        other_deadline = receipt(request)
        other_deadline['hard_deadline'] = '2026-10-11T09:00:00+08:00'
        with self.assertRaises(ValueError):
            flow.confirm(other_deadline, now=NOW)
        other_roster = receipt(request)
        other_roster['roster'] = ['beta', 'alpha']
        with self.assertRaises(ValueError):
            flow.confirm(other_roster, now=NOW)
        self.assertEqual(flow.next_stage(), 'confirmation')
        flow.confirm(receipt(request), now=NOW)
        self.assertEqual(flow.next_stage(), 'freeze')
        self.assertEqual(flow.snapshot()['stages']['confirmation']['status'], 'passed')

    def test_candidates_false_skips_only_candidate_stages(self):
        flow = self.confirmed(False)
        snap = flow.snapshot()
        for stage in CANDIDATE_STAGES:
            self.assertEqual(snap['stages'][stage]['status'], 'skipped')
            self.assertIn('32-card', snap['stages'][stage]['skip_reason'])
            self.assertIn('mulligan', snap['stages'][stage]['skip_reason'])
            with self.assertRaises(ValueError):
                flow.start(stage, now=NOW)
        for stage in STAGE_ORDER:
            if stage in CANDIDATE_STAGES or stage == 'confirmation':
                continue
            self.assertEqual(snap['stages'][stage]['status'], 'pending')
        self.assertEqual(flow.next_stage(), 'freeze')

    def test_candidates_true_does_not_skip_candidate_chain(self):
        flow = self.confirmed(True)
        snap = flow.snapshot()
        for stage in CANDIDATE_STAGES:
            self.assertEqual(snap['stages'][stage]['status'], 'pending')
            self.assertIsNone(snap['stages'][stage]['skip_reason'])
        self.assertEqual(flow.next_stage(), 'freeze')
        with self.assertRaises(ValueError):
            flow.start('candidate_search', now=NOW)

    def test_missing_required_handler_fails_before_any_launch(self):
        flow = self.confirmed(True)
        launched = []

        def freeze(assignment):
            launched.append(assignment['stage_id'])
            return note_result(assignment)

        with self.assertRaises(ValueError) as ctx:
            flow.run({'freeze': freeze}, now=NOW)
        self.assertIn('before launch', str(ctx.exception))
        self.assertIn('candidate_adaptation', str(ctx.exception))
        self.assertIn('build_confirmation', str(ctx.exception))
        self.assertEqual(launched, [])
        self.assertEqual(flow.snapshot()['stages']['freeze']['status'], 'pending')

    def test_zero_updates_and_metadata_only_delivery_are_rejected(self):
        flow = self.confirmed(False)
        freeze = flow.start('freeze', now=NOW)
        flow.finish('freeze', freeze['token'], note_result(freeze), now=NOW)
        preflight = flow.start('preflight', now=NOW)
        flow.finish('preflight', preflight['token'], note_result(preflight), now=NOW)
        training = flow.start('base_training', now=NOW)
        zero = note_result(
            training,
            extra_metrics={
                'complete_training_games': {'alpha': 0, 'beta': 8},
                'formal_policy_updates': {'alpha': 4, 'beta': 4},
            },
        )
        with self.assertRaises(ValueError):
            flow.finish('base_training', training['token'], zero, now=NOW)
        self.assertEqual(flow.snapshot()['stages']['base_training']['status'], 'running')
        bool_metrics = note_result(
            training,
            extra_metrics={
                'complete_training_games': {'alpha': True, 'beta': True},
                'formal_policy_updates': {'alpha': 1, 'beta': 1},
            },
        )
        with self.assertRaises(ValueError):
            flow.finish('base_training', training['token'], bool_metrics, now=NOW)
        flow.finish(
            'base_training',
            training['token'],
            {'verdict': 'failed', 'summary': 'zero updates are not a pass', 'artifacts': [], 'metrics': {}},
            now=NOW,
        )
        self.assertEqual(flow.next_stage(), 'report')
        report = flow.start('report', now=NOW)
        flow.finish('report', report['token'], note_result(report, 'report.md', b'incomplete'), now=NOW)
        delivery = flow.start('delivery', now=NOW)
        metadata = note_result(delivery, 'manifest.json', b'{"models":["alpha.json"]}')
        with self.assertRaises(ValueError) as ctx:
            flow.finish('delivery', delivery['token'], metadata, now=NOW)
        self.assertIn('JSON-only', str(ctx.exception))
        flow.finish(
            'delivery',
            delivery['token'],
            {'verdict': 'failed', 'summary': 'metadata-only delivery rejected', 'artifacts': [], 'metrics': {}},
            now=NOW,
        )
        snap = flow.snapshot()
        self.assertFalse(snap['workflow_complete'])
        self.assertTrue(snap['flow_terminal'])
        self.assertFalse(snap['quality_approved'])
        self.assertFalse(snap['complete_report'])
        self.assertEqual(snap['stages']['delivery']['status'], 'failed')

    def test_sha_path_and_token_identity_guards(self):
        flow = self.confirmed(False)
        first = flow.start('freeze', now=NOW)
        with self.assertRaises(ValueError):
            flow.finish('freeze', 'deadbeef', note_result(first), now=NOW)
        wrong_hash = note_result(first)
        wrong_hash['artifacts'][0]['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            flow.finish('freeze', first['token'], wrong_hash, now=NOW)
        outside = note_result(first)
        leaked = Path(self._tmp.name) / 'outside.bin'
        leaked.write_bytes(b'secret')
        outside['artifacts'] = [write_bytes(leaked, b'secret')]
        with self.assertRaises(ValueError):
            flow.finish('freeze', first['token'], outside, now=NOW)
        traversal = note_result(first)
        traversal['artifacts'] = [{
            'path': str(Path(first['output_directory']) / '..' / '..' / 'outside.bin'),
            'sha256': sha256(b'secret').hexdigest(),
        }]
        with self.assertRaises(ValueError):
            flow.finish('freeze', first['token'], traversal, now=NOW)
        link = Path(first['output_directory']) / 'linked.txt'
        os.symlink(leaked, link)
        linked = note_result(first)
        linked['artifacts'] = [{
            'path': str(link),
            'sha256': sha256(b'secret').hexdigest(),
        }]
        with self.assertRaises(ValueError):
            flow.finish('freeze', first['token'], linked, now=NOW)
        stale = note_result(first)
        stale['request_sha256'] = 'a' * 64
        with self.assertRaises(ValueError):
            flow.finish('freeze', first['token'], stale, now=NOW)
        flow.finish('freeze', first['token'], note_result(first), now=NOW)

    def test_duplicate_claims_are_rejected_across_loaded_handles(self):
        flow = self.confirmed(False)
        assignment = flow.start('freeze', now=NOW)
        self.assertEqual(assignment['agent'], 'grok')
        self.assertEqual(assignment['deadline'], DEADLINE)
        self.assertTrue(Path(assignment['output_directory']).is_dir())
        with self.assertRaises(ValueError):
            flow.start('freeze', now=NOW)
        with self.assertRaises(ValueError):
            flow.start('preflight', now=NOW)
        other = TrainingFlow.load(self.root)
        self.assertIsNone(other.next_stage())
        with self.assertRaises(ValueError):
            other.start('freeze', now=NOW)
        other.finish('freeze', assignment['token'], note_result(assignment), now=NOW)
        self.assertEqual(flow.snapshot()['stages']['freeze']['status'], 'running')
        reloaded = TrainingFlow.load(self.root)
        self.assertEqual(reloaded.snapshot()['stages']['freeze']['status'], 'passed')
        with self.assertRaises(ValueError):
            reloaded.start('freeze', now=NOW)

    def test_original_cutoff_survives_recovery_and_blocks_new_learning(self):
        flow = self.confirmed(False)
        epoch = flow.snapshot()['hard_deadline_epoch']
        restored = TrainingFlow.load(self.root)
        self.assertEqual(restored.snapshot()['hard_deadline'], DEADLINE)
        self.assertEqual(restored.snapshot()['hard_deadline_epoch'], epoch)
        later = datetime(2026, 10, 11, 12, 0, tzinfo=SHANGHAI)
        restored2 = TrainingFlow.load(self.root)
        self.assertEqual(restored2.snapshot()['hard_deadline_epoch'], epoch)
        with self.assertRaises(ValueError):
            restored2.start('freeze', now=AFTER)
        report = restored2.start('report', now=AFTER)
        restored2.finish('report', report['token'], note_result(report, 'report.md', b'cutoff'), now=AFTER)
        delivery = restored2.start('delivery', now=AFTER)
        restored2.finish(
            'delivery',
            delivery['token'],
            {'verdict': 'failed', 'summary': 'cutoff collected existing files only', 'artifacts': [], 'metrics': {}},
            now=AFTER,
        )
        snap = restored2.snapshot()
        self.assertEqual(snap['hard_deadline_epoch'], epoch)
        self.assertEqual(snap['stages']['freeze']['status'], 'pending')
        self.assertEqual(snap['stages']['report']['status'], 'passed')
        self.assertFalse(snap['quality_approved'])

    def test_partial_failure_reports_honestly_and_never_passes_missing_steps(self):
        flow = self.confirmed(False)
        freeze = flow.start('freeze', now=NOW)
        flow.finish(
            'freeze',
            freeze['token'],
            {'verdict': 'failed', 'summary': 'source freeze failed', 'artifacts': [], 'metrics': {}},
            now=NOW,
        )
        self.assertEqual(flow.next_stage(), 'report')
        for stage in ('preflight', 'base_training', 'evaluation'):
            with self.assertRaises(ValueError):
                flow.start(stage, now=NOW)
            self.assertEqual(flow.snapshot()['stages'][stage]['status'], 'pending')
        report = flow.start('report', now=NOW)
        flow.finish('report', report['token'], note_result(report, 'report.md', b'missing freeze'), now=NOW)
        delivery = flow.start('delivery', now=NOW)
        complete_looking = note_result(
            delivery,
            'weights.npz',
            b'npz',
            extra_files={'evaluation-raw.json.gz': b'raw', 'public-replay.json': b'replay'},
        )
        with self.assertRaises(ValueError):
            flow.finish('delivery', delivery['token'], complete_looking, now=NOW)
        flow.finish(
            'delivery',
            delivery['token'],
            {'verdict': 'blocked', 'summary': 'required stages missing', 'artifacts': [], 'metrics': {}},
            now=NOW,
        )
        snap = flow.snapshot()
        self.assertEqual(snap['stages']['freeze']['status'], 'failed')
        self.assertEqual(snap['stages']['preflight']['status'], 'pending')
        self.assertEqual(snap['stages']['delivery']['status'], 'blocked')
        self.assertFalse(snap['workflow_complete'])
        self.assertTrue(snap['flow_terminal'])
        self.assertFalse(snap['complete_report'])
        self.assertFalse(snap['quality_approved'])

    def test_synthetic_success_plumbing_does_not_approve_quality(self):
        flow = self.confirmed(False)
        metrics = {
            'complete_training_games': {'alpha': 8, 'beta': 8},
            'formal_policy_updates': {'alpha': 4, 'beta': 4},
        }
        eval_metrics = matrix_fixture()

        def handler_for(stage):
            def handle(assignment):
                extra_files = {}
                extra_metrics = {}
                name = 'note.txt'
                if stage == 'base_training':
                    extra_metrics = metrics
                    extra_files = {'policy.npz': b'weights'}
                elif stage == 'evaluation':
                    extra_metrics = eval_metrics
                    name = 'evaluation-raw.json'
                elif stage == 'delivery':
                    extra_files = {
                        'alpha.npz': numeric_fixture(),
                        'beta.npz': numeric_fixture(),
                        'evaluation-raw.json.gz': b'raw',
                        'public-replay.json': b'replay',
                    }
                elif stage == 'report':
                    name = 'report.md'
                return note_result(assignment, name, data=raw_matrix_fixture() if stage == 'evaluation' else b'ok', extra_metrics=extra_metrics, extra_files=extra_files)
            return handle

        required = [
            stage for stage in STAGE_ORDER
            if stage != 'confirmation' and stage not in CANDIDATE_STAGES
        ]
        snap = flow.run({stage: handler_for(stage) for stage in required}, now=NOW)
        self.assertIsNone(flow.next_stage())
        for stage in required:
            self.assertEqual(snap['stages'][stage]['status'], 'passed', stage)
        for stage in CANDIDATE_STAGES:
            self.assertEqual(snap['stages'][stage]['status'], 'skipped', stage)
        self.assertFalse(snap['workflow_complete'])
        review = self.root / 'fresh-review.json'
        review.write_text(json.dumps({'verdict': 'allow', 'reviewer_session': 'synthetic-fresh-session',
                                     'basis_sha256': flow.audit_basis(), 'request_sha256': snap['request_sha256']}))
        snap = flow.record_final_review({'path': str(review), 'sha256': sha256(review.read_bytes()).hexdigest()}, now=NOW)
        self.assertFalse(snap['workflow_complete'])
        self.assertFalse(snap['complete_report'])
        self.assertFalse(snap['quality_approved'])
        self.assertEqual(snap['request']['default_agent'], 'grok')
        self.assertEqual(len(snap['stages']['base_training']['attempts']), 1)
        self.assertEqual(snap['stages']['base_training']['attempts'][0]['agent'], 'grok')

    def test_evaluation_rejects_boolean_denominators_and_missing_raw_evidence(self):
        flow = self.confirmed(False)
        training_metrics = {
            'complete_training_games': {'alpha': 2, 'beta': 2},
            'formal_policy_updates': {'alpha': 1, 'beta': 1},
        }
        for stage in ('freeze', 'preflight', 'base_training', 'freeze_selection'):
            assignment = flow.start(stage, now=NOW)
            extra = training_metrics if stage == 'base_training' else None
            flow.finish(stage, assignment['token'], note_result(assignment, extra_metrics=extra), now=NOW)
        evaluation = flow.start('evaluation', now=NOW)
        boolean_cells = note_result(
            evaluation,
            'evaluation-raw.json.gz',
            extra_metrics={'planned_cells': True, 'completed_cells': True},
        )
        with self.assertRaises(ValueError):
            flow.finish('evaluation', evaluation['token'], boolean_cells, now=NOW)
        no_raw = note_result(
            evaluation,
            'summary.json',
            extra_metrics={'planned_cells': 4, 'completed_cells': 4},
        )
        with self.assertRaises(ValueError):
            flow.finish('evaluation', evaluation['token'], no_raw, now=NOW)
        self.assertEqual(flow.snapshot()['stages']['evaluation']['status'], 'running')

if __name__ == '__main__':
    unittest.main()
