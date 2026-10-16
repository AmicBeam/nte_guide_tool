"""SoA conformance-input collector: official 7-key / public17 routing, not a SoA engine."""
from __future__ import annotations

from copy import deepcopy
import gzip
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from app.modules.card_game.content.duel_v2 import PUBLIC_CHARACTER_IDS, STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import legal_actions, new_game
from app.modules.card_game.engine.duel_v2.entities import clone_state
from app.modules.card_game.rl.batched_duel.domain_soa_cases import (
    KEYS,
    MECHANISM_MODULES,
    REQUIRED_PUBLIC17_MECHANISMS,
    alias_groups,
    collect_mechanism_cases,
    coverage,
    empty_inventory,
    execute,
    missing_coverage,
    observe_state,
    official_seven_decks,
    plan_report,
    snapshot_bundle,
    supports_full_duel_rules,
    validate_modules,
    validate_output_dir,
)
from app.modules.card_game.rl.batched_duel.domain_soa_schema import COVERAGE
from app.modules.card_game.rl.batched_duel.native_array_conformance import MODULES
from scripts.collect_duel_v2_soa_cases import main


REPO = Path(__file__).resolve().parents[1]
EVALS = REPO / 'artifacts' / 'rl-evals'
PRIVATE_MODULES = (
    'test_duel_v2_xun',
    'test_duel_v2_haniya',
    'test_duel_v2_edgar',
    'test_duel_v2_september18',
)


class DomainSoACasesTest(unittest.TestCase):
    def _output(self):
        path = EVALS / f'soa-cases-unittest-{os.getpid()}-{id(self)}-{self.id().rsplit(".", 1)[-1]}'
        if path.exists():
            shutil.rmtree(path)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_coverage_does_not_claim_full_soa_or_rules(self):
        self.assertFalse(supports_full_duel_rules)
        report = coverage()
        self.assertFalse(report['supports_full_duel_rules'])
        self.assertFalse(report['complete_soa'])
        self.assertFalse(report['full_public17'])
        self.assertFalse(report['soa_rules_executed'])
        self.assertEqual(tuple(report['seven_official_presets']), KEYS)
        self.assertEqual(tuple(report['mechanism_modules']), MODULES)

    def test_seven_keys_are_official_32_card_public17_presets(self):
        self.assertEqual(KEYS, tuple(COVERAGE['seven_official_presets']))
        self.assertEqual(KEYS, tuple(item['id'] for item in STARTER_DECKS))
        decks = official_seven_decks()
        self.assertEqual(tuple(decks), KEYS)
        seen = set()
        for key, deck in decks.items():
            self.assertEqual(len(deck['card_ids']), 32, key)
            self.assertEqual(len(deck['character_ids']), 4, key)
            self.assertTrue(set(deck['character_ids']) <= set(PUBLIC_CHARACTER_IDS), key)
            seen.update(deck['character_ids'])
        self.assertEqual(len(PUBLIC_CHARACTER_IDS), 17)
        self.assertTrue(seen <= set(PUBLIC_CHARACTER_IDS))
        self.assertNotIn('haiyue', seen)

    def test_mechanism_modules_are_native_router_and_reject_private_legacy(self):
        self.assertEqual(MECHANISM_MODULES, MODULES)
        for name in PRIVATE_MODULES:
            self.assertNotIn(name, MECHANISM_MODULES)
            with self.assertRaises(ValueError):
                validate_modules((name,))
        self.assertEqual(validate_modules(('test_duel_v2_iloy_knockdown',)),
                         ('test_duel_v2_iloy_knockdown',))

    def test_plan_does_not_create_output(self):
        plan = plan_report()
        self.assertEqual(plan['mode'], 'plan')
        self.assertFalse(plan['output_created'])
        self.assertFalse(plan['training'])
        self.assertFalse(plan['soa_rules_executed'])
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main([])
        self.assertEqual(code, 0)
        printed = json.loads(buf.getvalue())
        self.assertEqual(printed['mode'], 'plan')
        self.assertEqual(tuple(printed['keys']), KEYS)

    def test_run_requires_new_artifacts_rl_evals_directory(self):
        with self.assertRaises(SystemExit):
            main(['--run'])
        with self.assertRaises(SystemExit):
            main(['--run', '--output', '/tmp/soa-cases-not-evals'])
        existing = self._output()
        existing.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            validate_output_dir(existing)
        with self.assertRaises(SystemExit):
            main(['--run', '--output', str(existing)])

    def test_inventory_records_nested_effects_flags_choice_operation_deferred_summons_aliases(self):
        decks = official_seven_decks()
        state = clone_state(new_game(seed=11, decks={'a': decks['starter'], 'b': decks['collapse']},
                                     first_side='a'))
        team = state['sides']['a']
        hero = next(iter(team['characters'].values()))
        hero['flags'] = {'timed_attack': 1, 'immunity': True}
        hero['effects'] = [{'id': 'test.aura', 'op': 'add', 'amount': 2, 'kind': 'persistent'}]
        team['flags'] = {'collapse': 1}
        team['effects'] = [{'id': 'player.buff', 'op': 'set', 'amount': 1}]
        team['front_debuff'] = {'delay': {'by': 'b'}, 'dots': {'etch': {'stacks': 1}}}
        borrowed = team['hand'][0]
        state['phase'] = 'choice'
        state['pending_choice'] = {'side': 'a', 'kind': 'discard', 'cards': [borrowed],
                                   'prompt': '选择 1 张手牌弃置'}
        state['operation'] = {'side': 'a', 'actor': hero['id'], 'card': borrowed}
        state['deferred_actions'] = [{'side': 'a', 'actor': hero['id'],
                                      'entity_id': hero['entity_id'], 'kind': 'extra_attack'}]
        summoned = deepcopy(hero)
        summoned.update(id='summon_e99', summoned=True, entity_id='e99')
        team['characters']['summon_e99'] = summoned
        inventory = empty_inventory()
        observe_state(inventory, state, 'fixture')
        aliases = alias_groups(authoritative := {k: v for k, v in state.items()
                                                 if k not in ('events', '_public_board')})
        bundle = snapshot_bundle(state)
        self.assertGreater(inventory['mechanisms']['pending_choice']['count'], 0)
        self.assertGreater(inventory['mechanisms']['operation']['count'], 0)
        self.assertGreater(inventory['mechanisms']['deferred_actions']['count'], 0)
        self.assertGreater(inventory['mechanisms']['summons']['count'], 0)
        self.assertGreater(inventory['mechanisms']['flags']['count'], 0)
        self.assertGreater(inventory['mechanisms']['effects']['count'], 0)
        self.assertGreater(inventory['mechanisms']['front_debuff']['count'], 0)
        self.assertGreater(inventory['mechanisms']['alias_relations']['count'], 0)
        self.assertTrue(any(len(group) > 1 for group in aliases))
        self.assertIn('discard', inventory['choice_kinds'])
        self.assertIn('etch', inventory['dot_kinds'])
        self.assertIn('events', state)
        self.assertNotIn('events', bundle['state'])
        self.assertNotIn('_public_board', bundle['state'])
        self.assertEqual(bundle['rng'], state['rng'])
        self.assertEqual(len(state['sides']['a']['hand']) + len(state['sides']['a']['deck'])
                         + len(state['sides']['a']['discard']) + len(state['sides']['a']['removed']), 32)
        self.assertIs(authoritative['pending_choice']['cards'][0], state['sides']['a']['hand'][0])

    def test_missing_required_public17_coverage_is_explicit(self):
        missing = missing_coverage(empty_inventory(), keys_seen=(), mechanism_modules=(),
                                   mechanism_commits=0)
        for name in REQUIRED_PUBLIC17_MECHANISMS:
            self.assertIn(name, missing)
        self.assertTrue(any(item.startswith('seven_keys:') for item in missing))
        self.assertTrue(any(item.startswith('public17:') for item in missing))
        self.assertIn('mechanism_modules_not_native_conformance_list', missing)
        self.assertIn('mechanism_physical_commits', missing)

    def test_tiny_natural_run_keeps_legal_order_rng_and_official_decks(self):
        path = self._output()
        report = execute(path, games=1, max_decisions=2, seconds=30, modules=(),
                         require_coverage=False)
        self.assertEqual(report['attempted_games'], 1)
        self.assertEqual(report['games'][0]['a'], 'starter')
        self.assertEqual(report['games'][0]['b'], 'weave-rush')
        self.assertEqual(report['games'][0]['decisions'], 2)
        self.assertFalse(report['coverage']['supports_full_duel_rules'])
        self.assertFalse(report['soa_rules_executed'])
        self.assertEqual(report['mechanism_tests'], 0)
        with gzip.open(path / 'natural' / 'game-000.json.gz', 'rt', encoding='utf-8') as stream:
            payload = json.load(stream)
        self.assertEqual(payload['initial']['rng'], payload['initial']['state']['rng'])
        step = payload['steps'][0]
        self.assertEqual(step['legal_count'], len(step['legal_actions']))
        self.assertEqual(step['legal_actions'][0]['type'], 'concede')
        decks = official_seven_decks()
        expected = new_game(seed=10160000, decks={'a': decks['starter'], 'b': decks['weave-rush']},
                            first_side='a')
        self.assertEqual(step['legal_actions'], json.loads(json.dumps(
            legal_actions(clone_state(expected), 'a'))))
        self.assertEqual(len(payload['initial']['state']['sides']['a']['hand'])
                         + len(payload['initial']['state']['sides']['a']['deck']), 32)
        inventory = json.loads((path / 'inventory.json').read_text(encoding='utf-8'))
        self.assertGreater(inventory['states_seen'], 0)
        self.assertIn('PlayerEntity', inventory['classes'])

    def test_mechanism_router_preserves_skip_and_commit_counts_on_public_module(self):
        with tempfile.TemporaryDirectory() as raw:
            report, traces = collect_mechanism_cases(
                raw, empty_inventory(), modules=('test_duel_v2_iloy_knockdown',),
                budget=lambda: None)
            self.assertGreater(report['native_commits'], 0)
            self.assertGreater(report['skipped'], 0)
            self.assertEqual(report['failures'], 0)
            self.assertEqual(report['errors'], 0)
            self.assertEqual(len(report['excluded']), report['skipped'])
            reasons = ' '.join(item['reason'] for item in report['excluded'])
            self.assertIn('no routed native physical commit', reasons)
            self.assertEqual(len([row for row in traces if row['accepted']]), report['native_commits'])
            self.assertTrue(any(row.get('legal_actions') for row in traces if row['accepted']))
            self.assertTrue((Path(raw) / 'mechanisms' / 'traces.jsonl.gz').exists())
            self.assertTrue((Path(raw) / 'mechanisms' / 'report.json').exists())


if __name__ == '__main__':
    unittest.main()
