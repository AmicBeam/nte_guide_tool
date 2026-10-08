"""Fake-engine/clock/launcher tests for V2 card-effect coverage eval. No real games."""
from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import importlib.util

from app.modules.card_game.content.duel_v2.catalog import CARD_ID_ORDER
from app.modules.card_game.rl.card_effect_eval import (
    HARD_MAX_GAMES,
    HARD_MAX_RAW_STEPS,
    HARD_MAX_SECONDS,
    CardEffectEvaluator,
    check_payload,
    coverage_deck,
    select_weighted_action,
    verify_manifest,
)

_CLI_PATH = Path(__file__).resolve().parents[1] / 'scripts/evaluate_duel_v2_cards.py'
_CLI_SPEC = importlib.util.spec_from_file_location('evaluate_duel_v2_cards_test', _CLI_PATH)
eval_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(eval_cli)


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def monotonic(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class FakeEngine:
    def __init__(self, script):
        self.script = list(script)
        self.new_game_calls = 0
        self.apply_calls = 0
        self.observe_hidden = False

    def new_game(self, seed=0, decks=None, first_side=None, skip_mulligan=False):
        self.new_game_calls += 1
        self.seed = seed
        self.first_side = first_side
        self.decks = decks
        return {
            'phase': 'mulligan', 'winner': None, 'reason': None, 'turn': 0,
            'events': [], 'acting': first_side or 'a', 'hidden_deck': 'MUST_NOT_SAVE',
        }

    def acting_side(self, state):
        return state.get('acting', 'a')

    def observe(self, state, side):
        if state.get('hidden_deck') and self.observe_hidden:
            raise AssertionError('policy path must not need hidden_deck')
        legal = state.get('legal') or [{'type': 'end_turn'}]
        observation = {
            'viewer_side': side, 'phase': state.get('phase'),
            'legal_actions': [{'action': deepcopy(item)} for item in legal],
            'sides': {side: {'hand': list(state.get('hand') or [])}},
            'resolving_card': state.get('resolving_card'),
        }
        return observation

    def apply_action(self, state, side, action):
        self.apply_calls += 1
        if not self.script:
            raise RuntimeError('unexpected apply')
        step = self.script.pop(0)
        if step.get('hang'):
            raise TimeoutError('hung apply')
        if 'raise' in step:
            raise step['raise']
        nxt = deepcopy(state)
        nxt.update(step.get('state') or {})
        nxt.setdefault('events', [])
        nxt['events'] = list(nxt.get('events') or []) + list(step.get('events') or [])
        nxt['last_action'] = deepcopy(action)
        return nxt


def _eval(tmp, engine, clock=None, **kwargs):
    return CardEffectEvaluator(
        Path(tmp), engine=engine, clock=clock or FakeClock(), **kwargs,
    )


class PolicyTest(unittest.TestCase):
    def test_never_concede_when_other_actions_exist(self):
        rng = __import__('random').Random(0)
        for _ in range(40):
            picked = select_weighted_action(
                [{'type': 'concede'}, {'type': 'play_card', 'card_id': 'a-1'}, {'type': 'end_turn'}],
                rng,
            )
            self.assertNotEqual(picked['type'], 'concede')

    def test_concede_only_when_forced(self):
        picked = select_weighted_action([{'type': 'concede'}], __import__('random').Random(1))
        self.assertEqual(picked['type'], 'concede')

    def test_coverage_deck_has_only_eight_buildable_cards_per_character(self):
        from app.modules.card_game.content.duel_v2 import CARDS, validate_deck
        deck = validate_deck(coverage_deck())
        self.assertEqual(len(deck['card_ids']), 32)
        self.assertEqual(len(set(deck['card_ids'])), 32)
        self.assertTrue(all(not CARDS[cid].get('derived') for cid in deck['card_ids']))
        for owner in deck['character_ids']:
            self.assertEqual(sum(CARDS[cid]['character_id'] == owner for cid in deck['card_ids']), 8)

    def test_check_payload_no_engine(self):
        payload = check_payload()
        self.assertEqual(payload['model'], 0)
        self.assertEqual(payload['training_updates'], 0)
        self.assertEqual(payload['card_ids'], list(CARD_ID_ORDER))
        self.assertIn('NOT RL-trained', payload['label'])


class FakeEvalTest(unittest.TestCase):
    def test_first_status_raw_steps_and_zero_card_counts(self):
        engine = FakeEngine([{
            'state': {'phase': 'finished', 'winner': 'a', 'turn': 1},
            'events': [{'type': 'finish', 'text': 'done'}],
        }])
        engine.script_observe_legal = None
        with tempfile.TemporaryDirectory() as tmp:
            def observe(state, side):
                return {
                    'viewer_side': side,
                    'legal_actions': [{'action': {'type': 'end_turn'}}],
                    'sides': {side: {'hand': []}},
                }
            engine.observe = observe
            result = _eval(tmp, engine, max_raw_steps=3, max_games=1, max_seconds=10).run()
            status = json.loads((Path(tmp) / 'status.json').read_text(encoding='utf-8'))
            summary = json.loads((Path(tmp) / 'summary.json').read_text(encoding='utf-8'))
            self.assertGreaterEqual(status['raw_steps'], 1)
            self.assertIn(status.get('sample_phase'), ('finished', 'mulligan', 'playing', 'choice'))
            self.assertEqual(status.get('last_action_type'), 'end_turn')
            self.assertEqual(result['model'], 0)
            self.assertEqual(result['training_updates'], 0)
            self.assertEqual(set(summary['card_uses']), set(CARD_ID_ORDER))
            self.assertTrue(all(v == 0 or k == 'ignored' for k, v in summary['card_uses'].items()))
            self.assertEqual(summary['card_uses']['N01'], 0)
            self.assertEqual(summary['coverage']['N01'], 'NOT_COVERED')
            self.assertNotIn('hidden_deck', json.dumps(summary))
            self.assertEqual(summary['winners']['a'], 1)
            self.assertTrue((Path(tmp) / 'progress.jsonl').exists())
            self.assertEqual(engine.new_game_calls, 1)

    def test_pending_choice_attribution_not_usage(self):
        hand = [{'instance_id': 'a-1', 'card_id': 'N03', 'copy': False}]
        engine = FakeEngine([
            {
                'state': {
                    'phase': 'choice', 'acting': 'a', 'turn': 1,
                    'legal': [{'type': 'choose', 'choice_id': 'c1'}],
                    'hand': hand, 'resolving_card': hand[0],
                },
                'events': [{'type': 'play', 'card': {'card_id': 'N03', 'instance_id': 'a-1'}}],
            },
            {
                'state': {'phase': 'finished', 'winner': 'draw', 'turn': 1, 'legal': [{'type': 'end_turn'}]},
                'events': [{'type': 'damage', 'amount': 1}],
            },
        ])

        def observe(state, side):
            legal = state.get('legal') or [{'type': 'play_card', 'card_id': 'a-1'}]
            return {
                'viewer_side': side,
                'legal_actions': [{'action': deepcopy(item)} for item in legal],
                'sides': {side: {'hand': hand}},
                'resolving_card': state.get('resolving_card'),
            }

        engine.observe = observe
        with tempfile.TemporaryDirectory() as tmp:
            result = _eval(tmp, engine, max_raw_steps=8, max_games=1, max_seconds=10).run()
            self.assertEqual(result['card_uses']['N03'], 1)
            self.assertEqual(result['copy_uses']['N03'], 0)
            self.assertEqual(result['card_effect_events']['N03'].get('damage'), 1)
            self.assertNotIn('play', result['card_effect_events']['N03'])
            self.assertEqual(result['coverage']['N03'], 'OBSERVED')
            self.assertEqual(result['coverage']['N01'], 'NOT_COVERED')
            self.assertEqual(result['event_counts']['play'], 1)
            self.assertEqual(result['event_counts']['damage'], 1)
            self.assertEqual(sum(1 for cid in CARD_ID_ORDER if result['coverage'][cid] == 'NOT_COVERED'), len(CARD_ID_ORDER) - 1)

    def test_legacy_snapshot_copy_use_separate_and_half_decks(self):
        # Historical saved copy: this is use accounting, not current X07 generation.
        copy_card = {'instance_id': 'a-9', 'card_id': 'X07', 'copy': True}
        scripts = []
        for i in range(2):
            scripts.append({
                'state': {'phase': 'finished', 'winner': 'b', 'turn': 1},
                'events': [{'type': 'effect', 'card': {'card_id': 'X07', 'copy': True}}],
            })
        engine = FakeEngine(scripts)

        def observe(state, side):
            return {
                'viewer_side': side,
                'legal_actions': [{'action': {'type': 'play_card', 'card_id': 'a-9'}}],
                'sides': {side: {'hand': [copy_card]}},
            }

        engine.observe = observe
        with tempfile.TemporaryDirectory() as tmp:
            ev = _eval(tmp, engine, max_raw_steps=10, max_games=2, max_seconds=10, base_seed=3)
            result = ev.run()
            self.assertEqual(result['copy_uses']['X07'], 2)
            self.assertEqual(result['card_uses']['X07'], 2)
            kinds = [ev.decks_for(0)[0], ev.decks_for(1)[0]]
            self.assertEqual(kinds, ['starter', 'coverage_all8'])
            self.assertEqual(len(coverage_deck()['card_ids']), 32)
            self.assertEqual(len(set(coverage_deck()['card_ids'])), 32)
            self.assertEqual({engine.decks['a']['id'] for _ in [0]}, {engine.decks['a']['id']})

    def test_budgets_truncate_midgame(self):
        engine = FakeEngine([
            {'state': {'phase': 'playing', 'acting': 'a', 'turn': 1, 'legal': [{'type': 'end_turn'}]}},
            {'state': {'phase': 'playing', 'acting': 'b', 'turn': 1, 'legal': [{'type': 'end_turn'}]}},
            {'state': {'phase': 'playing', 'acting': 'a', 'turn': 2, 'legal': [{'type': 'end_turn'}]}},
            {'state': {'phase': 'finished', 'winner': 'a'}},
        ])

        def observe(state, side):
            return {
                'viewer_side': side,
                'legal_actions': [{'action': {'type': 'end_turn'}}],
                'sides': {side: {'hand': []}},
            }

        engine.observe = observe
        with tempfile.TemporaryDirectory() as tmp:
            result = _eval(tmp, engine, max_raw_steps=2, max_games=4, max_seconds=30).run()
            self.assertEqual(result['raw_steps'], 2)
            self.assertEqual(result['stopped_reason'], 'max_raw_steps')
            self.assertEqual(result['winners']['truncated'], 1)
            self.assertEqual(result['games_started'], 1)

        clock = FakeClock()
        engine2 = FakeEngine([
            {'state': {'phase': 'playing', 'acting': 'a', 'legal': [{'type': 'end_turn'}]}},
            {'state': {'phase': 'playing', 'acting': 'a', 'legal': [{'type': 'end_turn'}]}},
        ])

        def observe_and_jump(state, side):
            clock.advance(8)
            return observe(state, side)

        engine2.observe = observe_and_jump
        with tempfile.TemporaryDirectory() as tmp:
            result = _eval(tmp, engine2, clock=clock, max_raw_steps=50, max_games=3, max_seconds=10).run()
            self.assertEqual(result['stopped_reason'], 'max_seconds')
            self.assertGreaterEqual(result['winners']['truncated'], 1)

        engine3 = FakeEngine([
            {'state': {'phase': 'playing', 'acting': 'a', 'legal': [{'type': 'end_turn'}]}}
            for _ in range(6)
        ] + [{'state': {'phase': 'finished', 'winner': 'draw'}}])
        engine3.observe = observe
        with tempfile.TemporaryDirectory() as tmp:
            result = _eval(tmp, engine3, max_raw_steps=40, max_games=2, max_seconds=30, per_game_actions=2).run()
            self.assertEqual(result['games_started'], 2)
            self.assertEqual(result['winners']['truncated'], 2)
            self.assertEqual(result['stopped_reason'], 'max_games')

    def test_exception_writes_summary_no_retry(self):
        engine = FakeEngine([{'raise': ValueError('当前不能执行此操作：bad')}])

        def observe(state, side):
            return {
                'viewer_side': side,
                'legal_actions': [{'action': {'type': 'attack', 'character_id': 'nanali'}}],
                'sides': {side: {'hand': []}},
            }

        engine.observe = observe
        with tempfile.TemporaryDirectory() as tmp:
            result = _eval(tmp, engine, max_raw_steps=5, max_games=3, max_seconds=10).run()
            self.assertEqual(engine.apply_calls, 1)
            self.assertEqual(result['exceptions'], 1)
            self.assertEqual(result['invalid_actions'], 1)
            self.assertEqual(result['phase'], 'failed')
            self.assertTrue((Path(tmp) / 'traceback.txt').read_text(encoding='utf-8'))
            summary = json.loads((Path(tmp) / 'summary.json').read_text(encoding='utf-8'))
            self.assertIn('ValueError', summary['error'])

    def test_watchdog_timeout_counter(self):
        engine = FakeEngine([])

        def observe(state, side):
            return {
                'viewer_side': side,
                'legal_actions': [{'action': {'type': 'end_turn'}}],
                'sides': {side: {'hand': []}},
            }

        def apply_action(state, side, action):
            raise AssertionError('should not apply when no time remains')

        engine.observe = observe
        engine.apply_action = apply_action
        clock = FakeClock(t=10)
        with tempfile.TemporaryDirectory() as tmp:
            result = CardEffectEvaluator(
                Path(tmp), engine=engine, clock=clock, max_raw_steps=5, max_games=1,
                max_seconds=10, apply_timeout=0,
            ).run()
            self.assertGreaterEqual(result['timeouts'], 1)
            self.assertEqual(result['phase'], 'failed')
            self.assertTrue((Path(tmp) / 'summary.json').exists())


class ManifestAndCliTest(unittest.TestCase):
    def test_manifest_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'a.txt'
            target.write_text('hello', encoding='utf-8')
            digest = __import__('hashlib').sha256(b'hello').hexdigest()
            man = root / 'source-manifest.json'
            man.write_text(json.dumps({'files': {'a.txt': digest}}), encoding='utf-8')
            self.assertEqual(verify_manifest(man, root=root), 1)
            man.write_text(json.dumps({'files': {'a.txt': '0' * 64}}), encoding='utf-8')
            with self.assertRaises(ValueError):
                verify_manifest(man, root=root)

    def test_check_does_not_new_game(self):
        buf = io.StringIO()
        with patch('sys.stdout', buf), \
             patch('app.modules.card_game.engine.duel_v2.new_game', side_effect=AssertionError('no new_game')), \
             patch('app.modules.card_game.engine.duel_v2.apply_action', side_effect=AssertionError('no steps')):
            code = eval_cli.main(['--check'])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload['ok'])
        self.assertEqual(payload['training_updates'], 0)
        self.assertEqual(payload['model'], 0)

    def test_cli_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / 'exists'
            existing.mkdir()
            with self.assertRaises(SystemExit):
                eval_cli.main(['--run', '--output', str(existing)])
            with self.assertRaises(SystemExit):
                eval_cli.main(['--run', '--output', str(Path(tmp) / 'n'), '--max-raw-steps', str(HARD_MAX_RAW_STEPS + 1)])
            with self.assertRaises(SystemExit):
                eval_cli.main(['--run', '--output', str(Path(tmp) / 'n'), '--max-games', str(HARD_MAX_GAMES + 1)])
            with self.assertRaises(SystemExit):
                eval_cli.main(['--run', '--output', str(Path(tmp) / 'n'), '--max-seconds', str(HARD_MAX_SECONDS + 1)])

    def test_launch_detached_returns_pid(self):
        class Proc:
            pid = 4242

        calls = {}

        def fake_popen(cmd, **kwargs):
            calls['cmd'] = cmd
            calls['kwargs'] = kwargs
            return Proc()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'exp'
            code = eval_cli.main(
                ['--launch', '--output', str(out), '--max-games', '2', '--max-raw-steps', '4', '--max-seconds', '9'],
                popen=fake_popen, executable='python-fake',
            )
            self.assertEqual(code, 0)
            self.assertEqual(calls['cmd'][0], 'python-fake')
            self.assertIn('--supervise', calls['cmd'])
            self.assertNotIn('--worker', calls['cmd'])
            config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
            self.assertEqual(config['pid'], 4242)
            self.assertEqual(config['model'], 0)
            self.assertEqual((out / 'supervisor.pid').read_text(encoding='utf-8').strip(), '4242')

    def test_supervise_timeout_writes_summary(self):
        class Proc:
            pid = 99
            returncode = None

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired(cmd='worker', timeout=timeout)

            def terminate(self):
                self.returncode = 1

            def kill(self):
                self.returncode = 1

        def fake_popen(cmd, **kwargs):
            self.assertIn('--worker', cmd)
            self.assertNotIn('--supervise', cmd)
            return Proc()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'exp'
            out.mkdir()
            ns = eval_cli._parser().parse_args([
                '--run', '--output', str(out), '--max-seconds', '3', '--max-games', '1', '--max-raw-steps', '2',
            ])
            code = eval_cli.supervise(ns, out, popen=fake_popen)
            self.assertEqual(code, 1)
            summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['stopped_reason'], 'timeout')
            self.assertEqual(summary['training_updates'], 0)


if __name__ == '__main__':
    unittest.main()
