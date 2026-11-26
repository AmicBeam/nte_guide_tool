"""Flow transport, clock, evidence and fresh-review regressions. No training."""
from copy import deepcopy
from datetime import datetime, timedelta
from hashlib import sha256
import json
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.modules.card_game.rl.training_flow import TrainingFlow, STAGE_ORDER, CANDIDATE_STAGES
from app.modules.card_game.rl.training_flow_agents import bind_stage_handlers, run_stage_agent, parse_structured_result
from app.modules.card_game.rl.training_flow_report import write_flow_report, render_flow_report
from tests.test_duel_v2_training_flow import NOW, SHANGHAI, sample_request, receipt, note_result, matrix_fixture, raw_matrix_fixture, numeric_fixture, write_bytes


class FlowIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'run'
        self.flow = TrainingFlow.create(self.root, sample_request(), now=NOW)
        self.flow.confirm(receipt(sample_request()), now=NOW)
        self.flow.configure_executors({name: lambda assignment: None for name in STAGE_ORDER if name != 'confirmation' and name not in CANDIDATE_STAGES})

    def tearDown(self):
        self.tmp.cleanup()

    def test_runtime_clock_stops_learning_and_still_reports_incomplete(self):
        clock = [datetime(2026, 10, 10, 8, 59, 59, tzinfo=SHANGHAI)]
        called = []
        def handler(assignment):
            called.append(assignment['stage_id'])
            clock[0] += timedelta(seconds=3)
            return note_result(assignment) if assignment['stage_id'] != 'delivery' else {
                'verdict': 'failed', 'summary': 'incomplete delivery', 'artifacts': [], 'metrics': {}}
        handlers = {name: handler for name in STAGE_ORDER if name != 'confirmation' and name not in CANDIDATE_STAGES}
        result = self.flow.run(handlers, now=lambda: clock[0])
        self.assertEqual(called, ['freeze', 'report', 'delivery'])
        self.assertEqual(result['stages']['freeze']['status'], 'failed')
        self.assertFalse(result['workflow_complete'])

    def test_manual_start_requires_entire_execution_plan(self):
        flow = TrainingFlow.create(self.root / 'unconfigured', sample_request(True), now=NOW)
        flow.confirm(receipt(sample_request(True)), now=NOW)
        with self.assertRaises(ValueError):
            flow.start('freeze', now=NOW)
        with self.assertRaises(ValueError):
            flow.configure_executors({'freeze': lambda assignment: None})
        self.assertFalse(flow.snapshot()['stages']['freeze']['attempts'])
        empty_scope = sample_request(True)
        empty_scope['mutable_characters'] = {'alpha': [], 'beta': []}
        with self.assertRaises(ValueError):
            TrainingFlow.create(self.root / 'empty-scope', empty_scope, now=NOW)

    def test_after_cutoff_report_collects_locally_without_spawning(self):
        assignment = self.flow.start('freeze', now=NOW)
        self.flow.finish('freeze', assignment['token'], {'verdict': 'failed', 'summary': 'fixture failed', 'metrics': {}, 'artifacts': []}, now=NOW)
        after = datetime(2026, 10, 10, 10, 0, tzinfo=SHANGHAI)
        assignment = self.flow.start('report', now=after)
        def never_launch(*args, **kwargs):
            self.fail('post-cutoff collection must not spawn')
        result = run_stage_agent(self.flow, assignment, runner=never_launch, now=after)
        self.assertEqual(result['verdict'], 'passed')
        self.assertFalse(result['metrics']['launched'])
        self.flow.finish('report', assignment['token'], result, now=after)
        self.assertFalse(self.flow.snapshot()['complete_report'])

    def test_invalidation_preserves_attempts_and_original_deadline(self):
        assignment = self.flow.start('freeze', now=NOW)
        self.flow.finish('freeze', assignment['token'], note_result(assignment), now=NOW)
        old = self.flow.snapshot()
        new = self.flow.invalidate_evidence('freeze', 'fixture source changed', now=NOW)
        self.assertEqual(new['hard_deadline'], old['hard_deadline'])
        self.assertEqual(len(new['stages']['freeze']['attempts']), 1)
        self.assertEqual(self.flow.next_stage(), 'freeze')
        assignment = self.flow.start('freeze', now=NOW)
        self.assertEqual(assignment['attempt'], 2)

    def test_agent_dispatch_is_bound_and_cannot_repeat_same_attempt(self):
        assignment = self.flow.start('freeze', now=NOW)
        calls = []
        def runner(prompt, **kwargs):
            calls.append(kwargs['assignment']['token'])
            return {**note_result(assignment), **{key: assignment[key] for key in ('stage_id', 'attempt', 'token')}}
        result = run_stage_agent(self.flow, assignment, runner=runner, now=NOW)
        self.assertEqual(result['verdict'], 'passed')
        with self.assertRaises(ValueError):
            run_stage_agent(self.flow, assignment, runner=runner, now=NOW)
        self.assertEqual(calls, [assignment['token']])
        self.flow.finish('freeze', assignment['token'], result, now=NOW)

    def test_crashed_or_timed_out_child_cannot_self_report_pass(self):
        payload = json.dumps({'verdict': 'passed', 'summary': 'claimed pass', 'metrics': {}, 'artifacts': []})
        self.assertEqual(parse_structured_result(payload, returncode=1)['verdict'], 'failed')
        self.assertEqual(parse_structured_result(payload, returncode=0, timed_out=True)['verdict'], 'blocked')
        self.assertEqual(parse_structured_result(payload + '\ntrace after result', returncode=0)['verdict'], 'failed')

    def test_recovery_rejects_edited_deadline_even_if_request_unchanged(self):
        state = self.root / 'state.json'
        data = json.loads(state.read_text())
        data['hard_deadline_epoch'] += 3600
        state.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            TrainingFlow.load(self.root)

    def test_stale_upstream_does_not_start_expensive_work(self):
        assignment = self.flow.start('freeze', now=NOW)
        result = note_result(assignment)
        self.flow.finish('freeze', assignment['token'], result, now=NOW)
        Path(result['artifacts'][0]['path']).write_text('changed')
        with self.assertRaises(ValueError):
            self.flow.start('preflight', now=NOW)

    def test_candidate_adaptation_requires_predeclared_equal_conditions(self):
        root = self.root / 'candidates'
        request = sample_request(True)
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        decks = {deck['id']: deck for deck in STARTER_DECKS}
        request['builds'] = {'alpha': decks['starter'], 'beta': decks['weave-rush']}
        request['mutable_characters'] = {'alpha': ['nanali'], 'beta': ['bohe']}
        flow = TrainingFlow.create(root, request, now=NOW)
        flow.confirm(receipt(request), now=NOW)
        flow.configure_executors({name: lambda assignment: None for name in STAGE_ORDER if name != 'confirmation'})
        for stage in ('freeze', 'preflight', 'base_training', 'candidate_search'):
            assignment = flow.start(stage, now=NOW)
            metrics = {'adaptation_target_steps': 2} if stage == 'freeze' else {}
            if stage == 'base_training':
                metrics = {'complete_training_games': 8, 'formal_policy_updates': {'alpha': 1, 'beta': 1}}
            if stage == 'candidate_search':
                candidate = deepcopy(request['builds']['alpha'])
                candidate['card_ids'].remove('N02')
                candidate['card_ids'].append('N08')
                metrics = {'candidate_changes': [{'key': 'alpha', 'build': candidate, 'removed': {'N02': 1}, 'added': {'N08': 1}}]}
                locked = deepcopy(request['builds']['alpha'])
                locked['card_ids'].remove('Y02')
                locked['card_ids'].append('Y01')
                with self.assertRaises(ValueError):
                    flow.finish(stage, assignment['token'], note_result(assignment, extra_metrics={'candidate_changes': [
                        {'key': 'alpha', 'build': locked, 'removed': {'Y02': 1}, 'added': {'Y01': 1}}]}), now=NOW)
            flow.finish(stage, assignment['token'], note_result(assignment, extra_metrics=metrics), now=NOW)
        original = flow.start('baseline_adaptation', now=NOW)
        proof = {'initial_sha256': 'i' * 64, 'opponents_sha256': 'o' * 64,
                 'seed_allocation_sha256': 's' * 64, 'optimizer_steps': {'alpha': 2, 'beta': 2}}
        flow.finish('baseline_adaptation', original['token'], note_result(original, extra_metrics={'adaptation': proof}), now=NOW)
        candidate = flow.start('candidate_adaptation', now=NOW)
        for key, wrong in [('optimizer_steps', {'alpha': 1, 'beta': 2}), ('seed_allocation_sha256', 'different')]:
            changed = {**proof, key: wrong}
            with self.assertRaises(ValueError):
                flow.finish('candidate_adaptation', candidate['token'], note_result(candidate, extra_metrics={'adaptation': changed}), now=NOW)
        flow.finish('candidate_adaptation', candidate['token'], note_result(candidate, extra_metrics={'adaptation': proof}), now=NOW)
        self.assertEqual(flow.snapshot()['stages']['candidate_adaptation']['status'], 'passed')
        confirmation = flow.start('build_confirmation', now=NOW)
        decision = {'decision': 'retain_baseline', 'reason': 'synthetic paired confirmation',
                    'normal_games': 8, 'seed_allocation_sha256': proof['seed_allocation_sha256']}
        with self.assertRaises(ValueError):
            flow.finish('build_confirmation', confirmation['token'], note_result(confirmation, extra_metrics=decision), now=NOW)
        decision['seed_allocation_sha256'] = 'c' * 64
        flow.finish('build_confirmation', confirmation['token'], note_result(confirmation, extra_metrics=decision), now=NOW)
        audit = flow.start('build_audit', now=NOW)
        with self.assertRaises(ValueError):
            flow.finish('build_audit', audit['token'], note_result(audit, extra_metrics=decision), now=NOW)

    def test_fake_numeric_extension_and_object_arrays_are_not_weights(self):
        from app.modules.card_game.rl.training_flow import _validate_numerical_file
        file = self.root / 'fake.npz'
        file.write_text('{"weights": "metadata"}')
        with self.assertRaises(ValueError):
            _validate_numerical_file(file)
        file.write_bytes(numeric_fixture())
        _validate_numerical_file(file)

    def test_matrix_claim_cannot_disagree_with_actual_raw_records(self):
        flow = self.flow
        for stage in ('freeze', 'preflight', 'base_training', 'freeze_selection'):
            assignment = flow.start(stage, now=NOW)
            metrics = {'complete_training_games': 8, 'formal_policy_updates': {'alpha': 1, 'beta': 1}} if stage == 'base_training' else {}
            flow.finish(stage, assignment['token'], note_result(assignment, extra_metrics=metrics), now=NOW)
        assignment = flow.start('evaluation', now=NOW)
        false_claim = matrix_fixture()
        false_claim['matrix_modes']['pure']['cells'][0]['first_wins'] = 2
        false_claim['matrix_modes']['pure']['cells'][0]['second_wins'] = 0
        with self.assertRaises(ValueError):
            flow.finish('evaluation', assignment['token'], note_result(assignment, 'evaluation-raw.json', data=raw_matrix_fixture(), extra_metrics=false_claim), now=NOW)
        pure_only = matrix_fixture()
        pure_only['matrix_modes'].pop('search')
        with self.assertRaises(ValueError):
            flow.finish('evaluation', assignment['token'], note_result(assignment, 'evaluation-raw.json', data=raw_matrix_fixture(), extra_metrics=pure_only), now=NOW)
        records = json.loads(raw_matrix_fixture())
        for game in records:
            game.pop('stage')
            game['policies'] = {'a': ['models', game.pop('left')], 'b': ['models', game.pop('right')]}
        data = gzip.compress(('\n'.join(json.dumps(game) for game in records) + '\n').encode())
        flow.finish('evaluation', assignment['token'], note_result(assignment, 'matrix-raw.jsonl.gz', data=data, extra_metrics=matrix_fixture()), now=NOW)

    def test_complete_synthetic_receipts_need_current_review_and_report_semantics(self):
        def handler(assignment):
            stage = assignment['stage_id']
            metrics = {}
            files = {}
            filename = 'proof.json'
            if stage == 'base_training':
                metrics = {'complete_training_games': 8, 'formal_policy_updates': {'alpha': 1, 'beta': 1},
                           'algorithm': 'gumbel', 'simulations': 32, 'gumbel_candidates': 16, 'terminal_wdl': True}
            elif stage == 'freeze_selection':
                metrics = {'models': {key: {variant: 'synthetic-' + variant for variant in ('initial', 'selected', 'latest')} for key in ('alpha', 'beta')}}
            elif stage == 'evaluation':
                metrics = matrix_fixture()
                metrics['evaluation'] = {key: {'planned': 2, 'completed': 2} for key in ('baseline', 'new_old', 'mulligan', 'varied_opponents')}
                filename = 'evaluation-raw.json'
            elif stage == 'analysis':
                metrics = {'analysis': {}}
                for key in ('card', 'esper', 'summon', 'use', 'release', 'mechanism'):
                    path = Path(assignment['output_directory']) / (key + '.csv')
                    path.write_text('game_id,event_count\n')
                    item = {'path': str(path.relative_to(self.root.resolve())), 'sha256': sha256(path.read_bytes()).hexdigest()}
                    metrics['analysis'][key] = {'games': 8, 'rows': 0, 'complete': True, 'artifacts': [item]}
                    files[key + '.csv'] = path.read_bytes()
            elif stage == 'delivery':
                files = {'alpha.npz': numeric_fixture(), 'beta.npz': numeric_fixture(), 'evaluation-raw.json': b'[]', 'public-replay.json': b'{}'}
            return note_result(assignment, filename, data=raw_matrix_fixture() if stage == 'evaluation' else b'ok', extra_metrics=metrics, extra_files=files)
        handlers = {name: handler for name in STAGE_ORDER if name != 'confirmation' and name not in CANDIDATE_STAGES}
        result = self.flow.run(handlers, now=NOW)
        self.assertTrue(all(result['stages'][name]['status'] == 'passed' for name in handlers))
        self.assertFalse(result['workflow_complete'])
        reviewed = write_flow_report(self.flow, self.root / 'reviewed-report.md')
        text = render_flow_report(self.flow)
        self.assertIn('50.0%（1/2）', text)
        artifact = write_bytes(self.root / 'fresh-review.json', json.dumps({
            'verdict': 'allow', 'reviewer_session': 'fresh-synthetic-session',
            'basis_sha256': self.flow.audit_basis(), 'request_sha256': result['request_sha256']}).encode())
        result = self.flow.record_final_review(artifact, now=NOW)
        self.assertTrue(result['complete_report'])
        self.assertTrue(result['workflow_complete'])
        self.assertFalse(result['quality_approved'])
        Path(artifact['path']).write_text('tampered')
        self.assertFalse(self.flow.snapshot()['workflow_complete'])


if __name__ == '__main__':
    unittest.main()
