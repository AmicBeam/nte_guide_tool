import json
import tempfile
import unittest
from copy import deepcopy
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import numpy as np

from app.modules.card_game.rl.information_search import Node, search, action_identity
from app.modules.card_game.rl.search_label_counterfactual import (
    INITIATIVES, PAIRED_ROOT_RUNTIME, ROSTER, SEARCH, accept_root_diagnostics,
    dry_run, planned_roots, validate_output_path, validate_seed_reservation,
)
from app.modules.card_game.rl.search_policy import visit_schedule, choose_root


class Prior:
    def scores_value(self, x, c):
        scores = np.linspace(5., 0., len(c), dtype=np.float64)
        return scores, 0.


class Many:
    n = 20

    @staticmethod
    def decision(s, side):
        actions = [{'type': 'wait', 'i': index} for index in range(Many.n)]
        return actions, (np.array([s.get('step', 0)], np.float32), np.eye(Many.n, dtype=np.float32))

    @staticmethod
    def determinize(s, viewer, seed):
        return {**s, 'latent': seed % 2}


def _act(s, side, action):
    state = deepcopy(s)
    state['step'] = state.get('step', 0) + 1
    if state['step'] > 4:
        state.update(phase='finished', winner='a')
    return state


def _search(**extra):
    with patch('app.modules.card_game.rl.information_search.apply_action', side_effect=_act):
        return search(dict(phase='playing', active_side='a', turn=1), 'a',
                      {'a': Prior(), 'b': Prior()}, runtime=Many, seed=11,
                      simulations=SEARCH['simulations'], fast_simulation=False,
                      algorithm='gumbel', gumbel_candidates=SEARCH['gumbel_candidates'],
                      **extra)


def _reservation(start=12_520_000_000):
    from app.modules.card_game.rl.build_acceptance import SEED_PHASES, SEED_WIDTH
    return {phase: start + index * SEED_WIDTH for index, phase in enumerate(SEED_PHASES)}


def _ledger(*runs):
    return dict(runs=list(runs))


def _write_models(root, rule_hash='rule'):
    import hashlib
    from app.modules.card_game.rl.cross_grounded import fingerprint
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for key in ROSTER:
        weights = root / f'{key}.npz'
        np.savez(weights, x=np.array([1.0], dtype=np.float32))
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        build = dict(id=key, character_ids=['a', 'b', 'c', 'd'], card_ids=['x'] * 32)
        build_digest = hashlib.sha256(json.dumps(
            {name: build[name] for name in ('character_ids', 'card_ids')},
            sort_keys=True, separators=(',', ':'),
        ).encode()).hexdigest()
        (root / f'{key}.json').write_text(json.dumps(dict(
            schema='cross_five_grounded_wdl_v1', rule_hash=rule_hash, sha256=digest,
            deck=key, build_sha256=build_digest,
            candidate_transform_sha256=fingerprint(), build=build,
        )), encoding='utf-8')
    return root


class SearchLabelCounterfactualTest(unittest.TestCase):
    def test_diagnostics_absent_keep_output_and_rng(self):
        plain = _search()
        diagnosed = _search(diagnostics=True)
        self.assertNotIn('root_diagnostics', plain)
        self.assertIn('root_diagnostics', diagnosed)
        for key, value in plain.items():
            other = diagnosed[key]
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, other)
            else:
                self.assertEqual(value, other)
        np.testing.assert_array_equal(plain['root_gumbel'], diagnosed['root_gumbel'])
        np.testing.assert_array_equal(plain['visits'], diagnosed['visits'])
        self.assertEqual(plain['search_choice'], diagnosed['search_choice'])

    def test_candidate_admission_covers_all_legal_actions_and_max16(self):
        result = _search(diagnostics=True)
        diagnostics = accept_root_diagnostics(result)
        self.assertEqual(len(diagnostics['action_identities']), Many.n)
        self.assertEqual(diagnostics['action_identities'], [action_identity(action) for action in result['actions']])
        self.assertEqual(diagnostics['candidate_cap'], 16)
        admitted = [index for index, flag in enumerate(diagnostics['admitted_to_max16']) if flag]
        self.assertEqual(len(admitted), 16)
        self.assertEqual(admitted, list(range(16)))
        self.assertTrue(all(diagnostics['visits'][index] > 0 for index in admitted))
        self.assertTrue(all(diagnostics['visits'][index] == 0 for index in range(16, Many.n)))
        self.assertEqual(int(np.sum(result['visits'] > 0)), 16)

    def test_gumbel_top_k_matches_zero_noise_admission(self):
        prior = np.linspace(0.2, 0.01, 20)
        prior = prior / prior.sum()
        node = Node.create(prior, 0.)
        schedule = visit_schedule(16, 32)
        for visit in schedule:
            chosen = choose_root(node, np.zeros(20), visit)
            node.visits[chosen] += 1
        self.assertEqual(sorted(np.flatnonzero(node.visits > 0).tolist()), list(range(16)))

    def test_nonfinite_and_incomplete_diagnostics_are_rejected(self):
        incomplete = _search(diagnostics=True, deadline=0)
        self.assertFalse(incomplete['complete'])
        with self.assertRaises(ValueError):
            accept_root_diagnostics(incomplete)
        result = _search(diagnostics=True)
        broken = deepcopy(result)
        broken['root_diagnostics']['completed_q'] = list(broken['root_diagnostics']['completed_q'])
        admitted = next(index for index, flag in enumerate(broken['root_diagnostics']['admitted_to_max16']) if flag)
        broken['root_diagnostics']['completed_q'][admitted] = float('nan')
        with self.assertRaises(ValueError):
            accept_root_diagnostics(broken)
        missing = dict(result)
        missing.pop('root_diagnostics')
        with self.assertRaises(ValueError):
            accept_root_diagnostics(missing)

    def test_duplicate_and_old_seeds_are_rejected(self):
        ledger = _ledger(dict(run_id='five-cross-20260924-formal', start=12_030_000_000,
                              end=12_100_000_000, seeds=_reservation(12_030_000_000)))
        with self.assertRaises(ValueError):
            validate_seed_reservation(_reservation(12_030_000_000), ledger)
        duplicate = _reservation()
        duplicate['audit'] = duplicate['confirmation']
        with self.assertRaises(ValueError):
            validate_seed_reservation(duplicate, ledger)
        ok = validate_seed_reservation(_reservation(), ledger)
        self.assertFalse(ok['archived_overlap'])
        self.assertEqual(len(ok['planned']), 10)

    def test_plan_covers_five_presets_and_both_initiatives(self):
        plan = planned_roots(_reservation())
        self.assertEqual([row['preset'] for row in plan], [preset for preset in ROSTER for _ in INITIATIVES])
        self.assertEqual({row['first'] for row in plan}, set(INITIATIVES))
        for index in range(0, 10, 2):
            self.assertEqual(plan[index]['preset'], plan[index + 1]['preset'])
            self.assertEqual((plan[index]['first'], plan[index + 1]['first']), ('a', 'b'))

    def test_dry_run_validates_identities_and_creates_no_output(self):
        self.assertFalse(PAIRED_ROOT_RUNTIME)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            models = _write_models(root / 'models')
            seeds_path = root / 'seeds.json'
            seeds_path.write_text(json.dumps(_reservation()), encoding='utf-8')
            output = root / 'new-run'
            ledger = _ledger(dict(run_id='old', start=10_000_000_000, end=12_520_000_000,
                                  seeds=_reservation(10_000_000_000)))
            result = dry_run(seed_reservation=seeds_path, models=models, output=output,
                             ledger=ledger, rule_hash='rule')
            self.assertFalse(output.exists())
            self.assertFalse(result['output_created'])
            self.assertEqual(result['games_played'], 0)
            self.assertEqual(result['roster'], list(ROSTER))
            self.assertEqual(result['initiatives'], list(INITIATIVES))
            self.assertEqual(len(result['planned']), 10)
            self.assertTrue(all(item['rule_hash'] == 'rule' for item in result['identities']['models'].values()))
            validate_output_path(output)
            output.mkdir()
            with self.assertRaises(ValueError):
                dry_run(seed_reservation=seeds_path, models=models, output=output,
                        ledger=ledger, rule_hash='rule')

    def test_dry_run_rejects_build_or_transform_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            models = _write_models(root / 'models')
            seeds = _reservation()
            ledger = _ledger()
            manifest_path = models / 'starter.json'
            original = json.loads(manifest_path.read_text(encoding='utf-8'))
            changed = dict(original, build_sha256='wrong')
            manifest_path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Build SHA mismatch'):
                dry_run(seed_reservation=seeds, models=models, output=root / 'out',
                        ledger=ledger, rule_hash='rule')
            changed = dict(original, candidate_transform_sha256='wrong')
            manifest_path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Candidate transform mismatch'):
                dry_run(seed_reservation=seeds, models=models, output=root / 'out',
                        ledger=ledger, rule_hash='rule')

    def test_cli_dry_run_prints_plan_without_runtime(self):
        from scripts.check_duel_v2_search_label_counterfactual import main
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            models = _write_models(root / 'models')
            seeds_path = root / 'seeds.json'
            seeds_path.write_text(json.dumps(_reservation()), encoding='utf-8')
            ledger_path = root / 'ledger.json'
            ledger_path.write_text(json.dumps(_ledger(dict(run_id='old', start=1, end=2, seeds={'foundation': 1}))), encoding='utf-8')
            output = root / 'cli-out'
            with self.assertRaises(SystemExit):
                main(['--seed-reservation', str(seeds_path), '--models', str(models),
                      '--output', str(output), '--seed-ledger', str(ledger_path), '--run'])
            with patch('sys.stdout', StringIO()):
                result = main(['--seed-reservation', str(seeds_path), '--models', str(models),
                               '--output', str(output), '--seed-ledger', str(ledger_path), '--rule-hash', 'rule'])
            self.assertFalse(output.exists())
            self.assertEqual(result['runtime_enabled'], False)


if __name__ == '__main__':
    unittest.main()
