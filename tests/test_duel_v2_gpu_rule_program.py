"""CPU tests for frozen GPU-rule source entries. Not a CUDA runtime claim."""
from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.content.duel_v2.catalog import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.rl.batched_duel import gpu_rule_entries as entries
from app.modules.card_game.rl.batched_duel import gpu_rule_program as gp
from app.modules.card_game.rl.rule_ir.portable_program import ValidationError, compute_identity


REPO = Path(__file__).resolve().parents[1]
CLI = REPO / 'scripts' / 'build_duel_v2_gpu_rule_program.py'
COLLAPSE = next(item for item in STARTER_DECKS if item['id'] == 'collapse')
LINGKE = next(item for item in STARTER_DECKS if item['id'] == 'lingke')


def _can_capture():
    return sys.version_info[:2] == (3, 10)


def _capture(names=entries.LIGHT_ENTRY_NAMES, lower=True):
    return gp.build_gpu_rule_program(names, lower=lower)


class WrapperCorrespondenceTest(unittest.TestCase):
    def test_fixed_teacher_continuation_matches_original_three_steps(self):
        from app.modules.card_game.rl.covered_rollout import rule_decision
        from app.modules.card_game.rl.league_rollout import clean
        from app.modules.card_game.engine.duel_v2 import acting_side
        from app.modules.card_game.engine.duel_v2.simulation import simulate_action
        state=new_game(seed=717,first_side='a',skip_mulligan=True,
                       decks={'a':COLLAPSE,'b':LINGKE})
        before=deepcopy(state);expected=deepcopy(state)
        for _ in range(3):
            if expected['phase']=='finished':break
            actor=acting_side(expected);legal,chosen=rule_decision(expected,actor)
            expected=clean(simulate_action(expected,actor,legal[chosen]))
        self.assertEqual(entries.seven_teacher_continue(state),expected)
        self.assertEqual(state,before)
        ended=deepcopy(state);ended['phase']='finished';ended['winner']='a'
        self.assertEqual(entries.seven_teacher_continue(ended),ended)
    @unittest.skipUnless(_can_capture(),'CPython3.10 source capture required')
    def test_captured_contract_exports_retain_function_attribute_binding(self):
        program=gp.capture_gpu_rule_program(['sample_world'],runtime='seven')
        table=program.modules['app.modules.card_game.rl.seven_lineup']
        self.assertEqual(table['feature_names']['kind'],'function')
        self.assertTrue(any(fn['id']==table['feature_names']['id'] for fn in program.functions))
    def test_device_setup_extension_preserves_exact_formal_new_game(self):
        parameters=dict(seed=20261011,decks={'a':COLLAPSE,'b':LINGKE},first_side='b',
                        skip_mulligan=False,escalation=True)
        before=deepcopy(parameters)
        self.assertEqual(entries.seven_initialize(parameters),new_game(**parameters))
        self.assertEqual(parameters,before)
        self.assertEqual(entries.DEVICE_SETUP_ENTRY_NAMES,('initialize',))
        self.assertNotIn('initialize',entries.FULL_ENTRY_NAMES)
        self.assertIs(entries.entry_functions('seven')['initialize'],entries.seven_initialize)
    def test_canonical_function_identities(self):
        from app.modules.card_game.engine.duel_v2.flow import acting_side, apply_action, legal_actions
        from app.modules.card_game.engine.duel_v2.projection import observe
        from app.modules.card_game.engine.duel_v2.simulation import simulate_action
        from app.modules.card_game.engine.duel_v2.state import finished
        from app.modules.card_game.rl.batched_duel.serving_protocol import public_terminal_probe, root_metadata
        from app.modules.card_game.rl.information_search import sample_world, terminal_value
        from app.modules.card_game.rl.league_rollout import clean
        self.assertIs(entries.CANONICAL_FUNCTIONS['acting_side'], acting_side)
        self.assertIs(entries.CANONICAL_FUNCTIONS['finished'], finished)
        self.assertIs(entries.CANONICAL_FUNCTIONS['terminal_value'], terminal_value)
        self.assertIs(entries.CANONICAL_FUNCTIONS['legal_actions'], legal_actions)
        self.assertIs(entries.CANONICAL_FUNCTIONS['public_view'], observe)
        self.assertIs(entries.CANONICAL_FUNCTIONS['apply_action'], apply_action)
        self.assertIs(entries.CANONICAL_FUNCTIONS['simulate_action'], simulate_action)
        self.assertIs(entries.CANONICAL_FUNCTIONS['clean'], clean)
        self.assertIs(entries.CANONICAL_FUNCTIONS['sample_world'], sample_world)
        self.assertIs(entries.CANONICAL_FUNCTIONS['serving_root_metadata'], root_metadata)
        self.assertIs(entries.CANONICAL_FUNCTIONS['public_terminal_probe'], public_terminal_probe)
        self.assertEqual(entries.FULL_ENTRY_NAMES, gp.FULL_ENTRY_NAMES)

    def test_new_game_wrappers_match_canonical_functions(self):
        state = new_game(seed=20261009, first_side='a', skip_mulligan=True,
                         decks={'a': COLLAPSE, 'b': LINGKE})
        side = entries.CANONICAL_FUNCTIONS['acting_side'](state)
        self.assertEqual(entries.acting_side(state), side)
        self.assertEqual(entries.finished(state), entries.CANONICAL_FUNCTIONS['finished'](state))
        self.assertFalse(entries.finished(state))
        self.assertEqual(entries.terminal_value(state, 'a'), 0.0)
        self.assertEqual(entries.terminal_value(state, 'a'),
                         entries.CANONICAL_FUNCTIONS['terminal_value'](state, 'a'))
        actions, view = entries.public_legal_decision(state, side)
        self.assertEqual(actions, entries.CANONICAL_FUNCTIONS['legal_actions'](state, side))
        canonical_view = entries.CANONICAL_FUNCTIONS['public_view'](state, side, include_previews=False)
        self.assertEqual(view, canonical_view)
        self.assertNotIsInstance(view, np.ndarray)
        self.assertFalse(any(isinstance(item, np.ndarray) for item in (actions, view)))
        seen = {hero['id'] for hero in view['sides']['a']['characters']}
        self.assertEqual(seen, set(COLLAPSE['character_ids']))
        self.assertIn('heiyu', seen)
        self.assertIn('dafutier', seen)
        self.assertEqual(len(seen), 4)
        self.assertEqual(entries.serving_root_metadata(state, side),
                         entries.CANONICAL_FUNCTIONS['serving_root_metadata'](state, side))
        end_turn = next(action for action in actions if action['type'] == 'end_turn')
        physical = entries.apply_action(deepcopy(state), side, end_turn)
        expected_physical = entries.CANONICAL_FUNCTIONS['apply_action'](deepcopy(state), side, end_turn)
        self.assertEqual(physical['version'], expected_physical['version'])
        self.assertEqual(physical['phase'], expected_physical['phase'])
        hypothetical = entries.simulate_action(deepcopy(state), side, end_turn)
        expected_hyp = entries.CANONICAL_FUNCTIONS['simulate_action'](deepcopy(state), side, end_turn)
        self.assertEqual(hypothetical['version'], expected_hyp['version'])
        self.assertEqual(entries.public_terminal_probe(state, side, end_turn),
                         entries.CANONICAL_FUNCTIONS['public_terminal_probe'](state, side, end_turn))
        cleaned = entries.clean(deepcopy(physical))
        self.assertEqual(cleaned['events'], [])
        self.assertNotIn('_public_board', cleaned)

    def test_sample_world_matches_canonical_sampler(self):
        from app.modules.card_game.rl import recovery_runtime
        state = new_game(seed=17, first_side='a', skip_mulligan=True)
        viewer = entries.acting_side(state)
        sampled = entries.sample_world(deepcopy(state), viewer, 99)
        expected = entries.CANONICAL_FUNCTIONS['sample_world'](deepcopy(state), viewer, 99, runtime=recovery_runtime)
        self.assertEqual(sampled['rng'], expected['rng'])
        self.assertEqual(sampled['phase'], expected['phase'])

    def test_finished_terminal_value_on_synthetic_result(self):
        state = new_game(seed=3, first_side='a', skip_mulligan=True)
        state['phase'] = 'finished'
        state['winner'] = 'b'
        self.assertTrue(entries.finished(state))
        self.assertIsNone(entries.acting_side(state))
        self.assertEqual(entries.terminal_value(state, 'b'), 1.0)
        self.assertEqual(entries.terminal_value(state, 'a'), -1.0)


@unittest.skipUnless(_can_capture() or os.environ.get('NTE_VM_TEST_PROGRAM_DIR'),
                     'CPython 3.10 capture or an explicit frozen program directory is required')
class FrozenProgramTest(unittest.TestCase):
    def test_entry_identity_is_stable(self):
        first, status = _capture()
        second, _ = _capture()
        self.assertTrue(status['compile_ready'])
        self.assertFalse(status['full_duel_rules'])
        self.assertFalse(status['full_coverage'])
        self.assertEqual(status['entry_set'], 'requested_subset')
        self.assertEqual(first.identity, second.identity)
        self.assertEqual(set(first.entries), set(entries.LIGHT_ENTRY_NAMES))
        first.validate()

    def test_source_tampering_is_rejected(self):
        program, _ = _capture(lower=False)
        bad_identity = program.to_dict()
        bad_identity['identity'] = '0' * 64
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            identity_path = root / 'tampered.json'
            identity_path.write_text(json.dumps(bad_identity))
            with self.assertRaisesRegex(ValidationError, 'Identity mismatch'):
                gp.load_program(identity_path, verify_sources=False)
            hashed = program.to_dict()
            hashed['source_hashes'] = dict(hashed['source_hashes'])
            victim = next(iter(hashed['source_hashes']))
            hashed['source_hashes'][victim] = 'ab' * 32
            hashed['identity'] = compute_identity(hashed)
            hash_path = root / 'hash.json'
            hash_path.write_text(json.dumps(hashed))
            with self.assertRaisesRegex(ValidationError, 'source hash mismatch'):
                gp.load_program(hash_path, verify_sources=True)

    def test_subset_is_never_full_coverage(self):
        program, status = _capture(('acting_side', 'finished'))
        self.assertEqual(status['entry_set'], 'requested_subset')
        self.assertFalse(status['full_coverage'])
        self.assertFalse(status['full_duel_rules'])
        self.assertEqual(set(program.entries), {'acting_side', 'finished'})
        check = gp.check_configuration()
        self.assertEqual(check['entry_set'], 'all_declared_entries')
        self.assertFalse(check['full_coverage'])
        self.assertFalse(check['full_duel_rules'])
        self.assertGreaterEqual(check['catalog_character_count'], 21)

    def test_source_validation_rejects_out_of_scope_paths_before_reading(self):
        program, _ = _capture(('acting_side',), lower=False)
        for name in ('../../outside.py', '.env'):
            payload=program.to_dict()
            payload['source_hashes']={name:'00'*32}
            payload['identity']=compute_identity(payload)
            altered=gp.FrozenRuleProgram(payload)
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('Out-of-scope source read')) as reader:
                with self.assertRaisesRegex(ValidationError, 'source hash mismatch'):
                    gp.verify_source_hashes(altered)
                reader.assert_not_called()

    def test_source_validation_rejects_external_parent_symlink(self):
        program, _ = _capture(('acting_side',), lower=False)
        with tempfile.TemporaryDirectory() as raw:
            base=Path(raw);root=base/'repo';root.mkdir()
            external=base/'external';external.mkdir();(external/'a.py').write_text('synthetic source')
            (root/'app').symlink_to(external,target_is_directory=True)
            payload=program.to_dict();payload['source_hashes']={'app/a.py':'00'*32}
            payload['identity']=compute_identity(payload)
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('External symlink source read')) as reader:
                with self.assertRaisesRegex(ValidationError, 'source hash mismatch'):
                    gp.verify_source_hashes(gp.FrozenRuleProgram(payload),repo_root=root)
                reader.assert_not_called()

    def test_unsupported_and_ready_entries_are_reported_separately(self):
        program, status = _capture()
        self.assertTrue(status['compile_ready'])
        self.assertTrue(status['image_ready'])
        self.assertFalse(status['device_verified'])
        self.assertFalse(status['full_duel_rules'])
        for name, report in status['entries'].items():
            self.assertEqual(report['entry'], name)
            self.assertIn('missing', report)
            self.assertIn('opcodes', report['missing'])
            self.assertIn('intrinsics', report['missing'])
            self.assertFalse(report['full_duel_rules'])
            self.assertFalse(report['device_verified'])
            self.assertFalse(report['device_ready'])
            self.assertIn('declared_capabilities_ready', report)
            self.assertEqual(report['actual_backend'], 'source_capture')
        with self.assertRaisesRegex(gp.GpuRuleProgramError, 'Unknown GPU rule entries'):
            gp.resolve_entry_names(('acting_side', 'not_a_real_entry'))

    def test_full_declared_program_keeps_full_duel_rules_false(self):
        program, status = gp.build_gpu_rule_program()
        self.assertTrue(status['compile_ready'])
        self.assertTrue(status['image_ready'])
        self.assertEqual(status['entry_set'], 'all_declared_entries')
        self.assertFalse(status['full_coverage'])
        self.assertFalse(status['full_duel_rules'])
        self.assertFalse(status['device_verified'])
        self.assertEqual(set(program.entries), set(entries.FULL_ENTRY_NAMES))
        missing = status['program_missing']
        self.assertEqual(missing['opcodes'], [])
        self.assertIn('uuid.uuid4', missing['intrinsics'])
        for name in ('app.modules.card_game.engine.duel_v2.entities.clone_state',
                     'enumerate', 'frozenset', 'set', 'sorted', 'numpy.array_equal'):
            self.assertNotIn(name, missing['intrinsics'])
        self.assertFalse(status['entries']['apply_action']['device_ready'])
        self.assertFalse(status['entries']['public_legal_decision']['device_ready'])
        self.assertFalse(status['entries']['sample_world']['device_ready'])
        self.assertGreaterEqual(len(program.functions), 600)

    def test_load_rejects_entry_set_mismatch(self):
        program, _ = _capture(('acting_side',), lower=False)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / 'one.json'
            path.write_text(program.to_json())
            loaded = gp.load_program(path, expected_entries=('acting_side',))
            self.assertEqual(loaded.identity, program.identity)
            with self.assertRaisesRegex(ValidationError, 'entry set mismatch'):
                gp.load_program(path, expected_entries=('acting_side', 'finished'))


class CliTest(unittest.TestCase):
    def test_help_documents_supported_flags(self):
        out = subprocess.check_output([sys.executable, str(CLI), '--help'], cwd=REPO, text=True)
        for flag in ('--check', '--build', '--output', '--load', '--entries', '--skip-lower'):
            self.assertIn(flag, out)

    def test_default_check_does_not_write_or_import_torch(self):
        out = subprocess.check_output([sys.executable, str(CLI)], cwd=REPO, text=True)
        payload = json.loads(out)
        self.assertEqual(payload['mode'], 'check')
        self.assertFalse(payload['full_duel_rules'])
        self.assertFalse(payload['full_coverage'])
        self.assertFalse(payload['torch_imported'])
        self.assertEqual(payload['requested_entries'], list(entries.FULL_ENTRY_NAMES))
        self.assertGreaterEqual(payload['catalog_character_count'], 21)

    @unittest.skipUnless(_can_capture(), 'CPython 3.10 source capture required')
    def test_build_writes_fresh_subset_program(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / 'gpu-rule-out'
            proc = subprocess.run(
                [sys.executable, str(CLI), '--build', '--output', str(output),
                 '--entries', 'acting_side,finished,terminal_value,clean', '--skip-lower'],
                cwd=REPO, text=True, capture_output=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload['mode'], 'build')
            self.assertFalse(payload['status']['full_coverage'])
            self.assertEqual(payload['status']['entry_set'], 'requested_subset')
            self.assertTrue((output / gp.PROGRAM_FILENAME).is_file())
            self.assertTrue((output / gp.MANIFEST_FILENAME).is_file())
            manifest = json.loads((output / gp.MANIFEST_FILENAME).read_text())
            self.assertEqual(manifest['identity'], payload['status']['identity'])
            self.assertFalse(manifest['status']['full_duel_rules'])
            again = subprocess.run(
                [sys.executable, str(CLI), '--build', '--output', str(output), '--entries', 'acting_side'],
                cwd=REPO, text=True, capture_output=True,
            )
            self.assertNotEqual(again.returncode, 0)


class RuntimeBindingTest(unittest.TestCase):
    def test_default_runtime_keeps_recovery_identity(self):
        from app.modules.card_game.rl.batched_duel.gpu_schema_contract import GPU_RUNTIME_CONTRACTS
        check = gp.check_configuration()
        self.assertEqual(check['runtime'], 'recovery')
        self.assertEqual(check['obs_dim'], 2420)
        self.assertEqual(check['cand_dim'], 254)
        self.assertEqual(check['keys'], ['starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk'])
        self.assertEqual(check['sampler_runtime'], '_FrozenRecoveryRuntime')
        self.assertIs(entries.entry_functions()['sample_world'], entries.sample_world)
        self.assertIs(entries.entry_functions('recovery')['sample_world'], entries.sample_world)
        self.assertEqual(GPU_RUNTIME_CONTRACTS['recovery']['obs_dim'], 2420)
        self.assertEqual(GPU_RUNTIME_CONTRACTS['recovery']['cand_dim'], 254)

    def test_seven_wrappers_are_distinct_and_do_not_replace_recovery_sampler(self):
        from app.modules.card_game.rl import recovery_runtime, seven_runtime
        self.assertIsNot(entries.sample_world, entries.seven_sample_world)
        self.assertIs(entries.entry_functions('seven')['sample_world'], entries.seven_sample_world)
        self.assertIs(entries.sampler_runtime('recovery').decision, recovery_runtime.decision)
        self.assertIs(entries.sampler_runtime('recovery').determinize, recovery_runtime.determinize)
        self.assertIs(entries.sampler_runtime('seven').decision, seven_runtime.decision)
        self.assertIs(entries.sampler_runtime('seven').determinize, seven_runtime.determinize)
        self.assertEqual(entries.sampler_runtime('seven').__name__, '_FrozenSevenRuntime')
        self.assertNotEqual(entries.sampler_runtime('recovery'), entries.sampler_runtime('seven'))
        state = new_game(seed=21, first_side='a', skip_mulligan=True)
        viewer = entries.acting_side(state)
        recovered = entries.sample_world(deepcopy(state), viewer, 99)
        expected = entries.CANONICAL_FUNCTIONS['sample_world'](
            deepcopy(state), viewer, 99, runtime=recovery_runtime
        )
        self.assertEqual(recovered['rng'], expected['rng'])
        seven_state = new_game(
            seed=21, first_side='a', skip_mulligan=True,
            decks={'a': COLLAPSE, 'b': LINGKE},
        )
        seven_viewer = entries.seven_acting_side(seven_state)
        seven_sampled = entries.seven_sample_world(deepcopy(seven_state), seven_viewer, 99)
        seven_expected = entries.CANONICAL_FUNCTIONS['sample_world'](
            deepcopy(seven_state), seven_viewer, 99, runtime=seven_runtime
        )
        self.assertEqual(seven_sampled['rng'], seven_expected['rng'])
        self.assertEqual(entries.seven_acting_side(seven_state), entries.acting_side(seven_state))

    def test_seven_check_reports_versioned_widths(self):
        status = gp.check_configuration(runtime='seven')
        self.assertEqual(status['runtime'], 'seven')
        self.assertEqual(status['obs_dim'], 3479)
        self.assertEqual(status['cand_dim'], 1576)
        self.assertEqual(
            status['keys'],
            ['starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk', 'collapse', 'lingke'],
        )
        self.assertEqual(status['sampler_runtime'], '_FrozenSevenRuntime')
        self.assertIn('existing recovery sampler is not replaced', status['sampler_contract'])

    @unittest.skipUnless(_can_capture(), 'CPython 3.10 source capture required')
    def test_load_rejects_runtime_mismatch(self):
        program, _ = gp.build_gpu_rule_program(('acting_side',), lower=False, runtime='seven')
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / 'seven.json'
            path.write_text(program.to_json())
            loaded = gp.load_program(path, expected_entries=('acting_side',), runtime='seven', verify_sources=False)
            self.assertEqual(loaded.identity, program.identity)
            with self.assertRaisesRegex(ValidationError, 'runtime mismatch'):
                gp.load_program(path, expected_entries=('acting_side',), runtime='recovery', verify_sources=False)


if __name__ == '__main__':
    unittest.main()
