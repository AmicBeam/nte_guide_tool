"""Seven-team readiness keeps engineering evidence, not a disposable-fit veto.

These are validator fixtures, not actual CUDA or strength qualification.
"""
from contextlib import ExitStack, nullcontext
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from app.modules.card_game.rl import seven_round as rr
from app.modules.card_game.rl.five_cross_gates import schedule_gates


class SevenPreflightTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(rr, 'fingerprint', return_value='algorithm'))
        self.stack.enter_context(patch('app.modules.card_game.rl.seven_lineup.identity', return_value='rules'))
        self.versions = {key: 'frozen-'+key for key in rr.ROSTER}
        self.stack.enter_context(patch.object(rr, 'versions', return_value=self.versions))
        self.stack.enter_context(patch('app.modules.card_game.rl.seven_runtime.model', return_value=object()))
        self.value_check = self.stack.enter_context(patch(
            'app.modules.card_game.rl.seven_runtime.value_precheck_ok', return_value=True))
        self.stack.enter_context(patch.object(rr, 'projected_learning_work',
            return_value=dict(needed=1, needed_eval=1)))
        self.contract = {'rule_backend': 'resident_cpu'}
        self.stack.enter_context(patch.object(rr, 'rollout_contract', return_value=self.contract))
        now = time.time()
        self.config = dict(runtime='seven', rule_hash='rules', device='cuda',
            rule_backend='resident_cpu', search_roots=1600, training_buffer_mib=8192,
            gpu_memory_budget_mib=14336, teacher_mode='covered_expert',
            recovery_source='frozen-source', simulations=32, gumbel_candidates=16,
            terminal_horizon=0, rewards=0, planned_batches=4, gate_roots=5,
            search_gate_roots=1, gates=schedule_gates(4), workers=6,
            policy_passes=4, learning_rate=.0003, train_until=now+1000,
            evaluate_until=now+2000, hard_deadline=now+3000,
            readiness_report=str(self.root/'readiness.json'))
        self.receipt = dict(schema='seven_recovery_readiness_v1', device='cuda',
            algorithm_sha256='algorithm', rule_hash='rules', simulations=32,
            gumbel_candidates=16, terminal_horizon=0, q_scale='natural_wdl',
            workers=6, teacher_mode='covered_expert', covered_value_sha=None,
            measured_at=now, ready=True, initial_models=self.versions,
            rollout_contract=self.contract, policy_passes=4, learning_rate=.0003,
            health={key: dict(finite=True, gradient=True, coverage=1, policy_rows=2)
                    for key in rr.ROSTER}, files={})
        self.put('initial-value/value-precheck.json', {'passed': True})
        self.put('cycles/1/numeric.json', {key: dict(actions_equal=True, policy_wdl_equal=True)
                                         for key in rr.ROSTER})
        self.put('cycles/1/restore-check.json', {key: dict(adam_next_step_equal=True, rng_equal=True)
                                               for key in rr.ROSTER})
        self.put('cycles/1/gate.json', {'decision': 'keep'})
        self.records = [dict(id=str(i), complete=True) for i in range(56)]
        self.put('cycles/1/sampling/games.json', self.records)

    def put(self, relative, value):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        self.receipt['files'][relative] = sha256(path.read_bytes()).hexdigest()

    def validate(self):
        (self.root/'readiness.json').write_text(json.dumps(self.receipt))
        return rr.validate_launch(self.config)

    def test_complete_preflight_does_not_require_disposable_capacity_fit(self):
        self.validate()
        self.assertFalse((self.root/'cycles/1/capacity.json').exists())

    def test_failed_optional_probe_does_not_veto_qualified_candidate(self):
        self.put('cycles/1/capacity.json', dict(passed=False, complete=True,
            schema='seven_recovery_isolation_fit_v1'))
        self.validate()

    def test_preflight_sends_search_batch_directly_to_joint_learning(self):
        # Stop at the update boundary: no optimizer, generated game or CUDA
        # operation is needed to verify that no second corpus/fit intervenes.
        config = dict(self.config, device='cpu', preflight=True, hidden=16,
                      seeds={'foundation': 11}, max_row_reuse=8, planned_batches=1)
        jobs = [{'id': str(i)} for i in range(56)]
        games = [{'complete': True} for _ in jobs]
        with patch.object(rr, 'validate_launch'), \
             patch.object(rr, 'initialize', return_value=({}, {}, {})), \
             patch('app.modules.card_game.rl.offline_sources.snapshot'), \
             patch.object(rr, 'warm_value', return_value=(self.root/'value/models', {'passed': True})), \
             patch.object(rr, 'learning_collection_jobs', return_value=jobs), \
             patch.object(rr, 'collect_search_batch', return_value=games) as collect, \
             patch.object(rr, 'capacity_heldout_jobs') as heldout, \
             patch('app.modules.card_game.rl.recovery_capacity.fit') as fit, \
             patch.object(rr, 'update_batch', side_effect=RuntimeError('stop before gradient')) as update:
            with self.assertRaisesRegex(RuntimeError, 'stop before gradient'):
                rr._run_learning_core(self.root/'run', config, device_context=nullcontext(object()))
            collect.assert_called_once()
            heldout.assert_not_called()
            fit.assert_not_called()
            self.assertIs(update.call_args.args[3], jobs)
            self.assertIs(update.call_args.args[4], games)

    def test_p0_and_incomplete_games_still_reject(self):
        self.put('cycles/1/gate.json', {'decision': 'p0'})
        with self.assertRaisesRegex(ValueError, 'numeric/restore/independent behavior'):
            self.validate()
        self.put('cycles/1/gate.json', {'decision': 'keep'})
        self.records[0]['complete'] = False
        self.put('cycles/1/sampling/games.json', self.records)
        with self.assertRaisesRegex(ValueError, 'numeric/restore/independent behavior'):
            self.validate()

    def test_numeric_restore_gradient_and_value_failures_still_reject(self):
        for name in ('actions_equal', 'policy_wdl_equal'):
            self.put('cycles/1/numeric.json', {key: dict(actions_equal=name!='actions_equal',
                policy_wdl_equal=name!='policy_wdl_equal') for key in rr.ROSTER})
            with self.assertRaisesRegex(ValueError, 'numeric/restore/independent behavior'):
                self.validate()
        self.put('cycles/1/numeric.json', {key: dict(actions_equal=True, policy_wdl_equal=True)
                                         for key in rr.ROSTER})
        for name in ('adam_next_step_equal', 'rng_equal'):
            self.put('cycles/1/restore-check.json', {key: dict(
                adam_next_step_equal=name!='adam_next_step_equal', rng_equal=name!='rng_equal')
                for key in rr.ROSTER})
            with self.assertRaisesRegex(ValueError, 'numeric/restore/independent behavior'):
                self.validate()
        self.put('cycles/1/restore-check.json', {key: dict(adam_next_step_equal=True, rng_equal=True)
                                               for key in rr.ROSTER})
        health = self.receipt['health'][rr.ROSTER[0]]
        for name, bad in [('gradient', False), ('finite', False), ('coverage', 0), ('policy_rows', 0)]:
            before = health[name]
            health[name] = bad
            with self.assertRaisesRegex(ValueError, 'gradient and coverage'):
                self.validate()
            health[name] = before
        self.value_check.return_value = False
        with self.assertRaisesRegex(ValueError, 'value arrays mismatch'):
            self.validate()

    def test_vm_replay_is_still_mandatory_and_bound_to_real_records(self):
        self.config['rule_backend'] = 'cuda_vm'
        self.contract['rule_backend'] = 'cuda_vm'
        with self.assertRaisesRegex(ValueError, 'raw evidence is incomplete'):
            self.validate()
        self.records = [dict(id=str(i), complete=True, rule_backend='cuda_vm',
                             final_state_sha256='state-'+str(i)) for i in range(56)]
        self.put('cycles/1/sampling/games.json', self.records)
        proofs = [dict(id=r['id'], passed=True, final_state_sha256=r['final_state_sha256'])
                  for r in self.records]
        self.put('cycles/1/sampling/gpu-replay-check.json', proofs)
        self.validate()
        proofs[0]['final_state_sha256'] = 'different'
        self.put('cycles/1/sampling/gpu-replay-check.json', proofs)
        with self.assertRaisesRegex(ValueError, 'real GPU trajectories'):
            self.validate()

    def test_raw_file_tampering_still_rejects(self):
        (self.root/'cycles/1/numeric.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'raw evidence changed'):
            self.validate()

    def test_old_source_and_cpu_receipts_cannot_become_cuda_readiness(self):
        self.receipt['algorithm_sha256'] = 'old-algorithm'
        with self.assertRaisesRegex(ValueError, 'identity/age/config'):
            self.validate()
        self.receipt['algorithm_sha256'] = 'algorithm'
        self.receipt['device'] = 'cpu'
        with self.assertRaisesRegex(ValueError, 'identity/age/config'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
