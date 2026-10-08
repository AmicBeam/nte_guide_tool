"""Bounded offline analysis tests. Fake policies plus tiny official games."""
from __future__ import annotations

import importlib.util
import json
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2.replay_log import leak_markers
from app.modules.card_game.rl.capability import sample_public_deck
from app.modules.card_game.rl.offline_analysis import (
    export_game_replay,
    frozen_model_policy,
    resolve_numeric_model_dir,
    run_offline_analysis,
)
from app.modules.card_game.rl.replay_export import public_file_ok


_CLI_PATH = Path(__file__).resolve().parents[1] / 'scripts/analyze_duel_v2_policy.py'
_CLI_SPEC = importlib.util.spec_from_file_location('analyze_duel_v2_policy_test', _CLI_PATH)
analyze_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(analyze_cli)


def _priority_policy(priority):
    def policy(view, actions):
        assert isinstance(view, dict)
        assert 'legal_actions' in view
        dumped = json.dumps(view)
        assert '"deck":' not in dumped
        assert '"rng"' not in dumped
        ranked = sorted(
            enumerate(actions),
            key=lambda item: (priority.get(item[1].get('type'), 50), item[0]),
        )
        return ranked[0][0]
    return policy


LEARNER = _priority_policy({
    'mulligan': 0,
    'choose': 1,
    'attack': 2,
    'play_card': 3,
    'ultimate': 4,
    'end_turn': 5,
})
OPPONENT = _priority_policy({
    'mulligan': 0,
    'choose': 1,
    'attack': 2,
    'play_card': 3,
    'ultimate': 4,
    'end_turn': 5,
})


class ResolveModelPathTest(unittest.TestCase):
    def test_pt_requires_export_and_does_not_pretend_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'starter.pt'
            path.write_bytes(b'not-a-model')
            with self.assertRaisesRegex(ValueError, 'export'):
                resolve_numeric_model_dir(path, deck_id='starter')

    def test_npz_requires_adjacent_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'starter.npz'
            path.write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'adjacent manifest'):
                resolve_numeric_model_dir(path, deck_id='starter')
            (Path(tmp) / 'starter.json').write_text('{}', encoding='utf-8')
            directory, deck_id = resolve_numeric_model_dir(path, deck_id='starter')
            self.assertEqual(directory, Path(tmp))
            self.assertEqual(deck_id, 'starter')


class FrozenPolicyContractTest(unittest.TestCase):
    def test_uses_select_public_action_not_a_new_model(self):
        seen = []
        class Frozen:
            def select_public_action(self, view, actions):
                seen.append((view, actions))
                return 0
        policy = frozen_model_policy(Frozen())
        self.assertEqual(policy({'viewer_side': 'a'}, ({'type': 'end_turn'},)), 0)
        self.assertEqual(seen[0][1][0]['type'], 'end_turn')

    def test_missing_method_fails_closed(self):
        class Frozen:
            pass
        with self.assertRaisesRegex(TypeError, 'select_public_action'):
            frozen_model_policy(Frozen())


class TinyOfficialGameTest(unittest.TestCase):
    def test_paired_games_record_complete_actions_and_public_replays(self):
        opponent_deck = sample_public_deck(7)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'report'
            summary = run_offline_analysis(
                output,
                LEARNER,
                deck='starter',
                opponent_policy=OPPONENT,
                opponent_deck=opponent_deck,
                pairs=1,
                max_seconds=20,
                max_actions=8,
                seed_base=11,
            )
            self.assertEqual(summary['games'], 2)
            self.assertEqual(summary['engine']['first_turn_draw'], True)
            self.assertEqual(summary['engine']['escalation'], True)
            self.assertFalse(summary['engine']['skip_mulligan'])
            self.assertEqual(summary['coverage']['public_cards'], 48)
            self.assertIn('not a causal', ' '.join(summary['caveats']).lower() + summary['label'].lower())
            self.assertTrue((output / 'report.json').is_file())
            self.assertTrue((output / 'report.md').is_file())
            report_md = (output / 'report.md').read_text(encoding='utf-8')
            self.assertIn('不等于卡牌的因果强度', report_md)
            self.assertIn('| N01 ·', report_md)
            self.assertEqual(len(summary['card_usage']), 49)
            self.assertIn('NF01',summary['card_usage'])
            self.assertGreater(len((output/'report.md').read_text(encoding='utf-8').splitlines()),20)
            self.assertNotIn('seed_base',summary)
            ids = [row['game_id'] for row in summary['games_index']]
            self.assertEqual(ids, ['starter-00-first', 'starter-00-second'])
            raw_first = json.loads((output / 'raw' / 'starter-00-first.json').read_text(encoding='utf-8'))
            raw_second = json.loads((output / 'raw' / 'starter-00-second.json').read_text(encoding='utf-8'))
            self.assertEqual(raw_first['seed'], raw_second['seed'])
            self.assertEqual(raw_first['opponent_deck']['card_ids'], raw_second['opponent_deck']['card_ids'])
            self.assertEqual(raw_first['first_side'], 'a')
            self.assertEqual(raw_second['first_side'], 'b')
            types = {item['type'] for item in raw_first['actions']}
            self.assertIn('mulligan', types)
            self.assertIn('attack', types)
            self.assertIn('end_turn', types)
            self.assertTrue(any(item['side'] == 'b' for item in raw_first['actions']))
            payload = json.loads((output / 'public' / 'starter-00-first.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['visibility'],'spectator_public_v1')
            for view in payload['views'].values():
                for side in ('a','b'):
                    self.assertTrue(all(c.get('hidden') for c in view['opening_board']['sides'][side]['hand']))
                for event in view['events']:
                    if event['type']=='draw':
                        self.assertNotIn('card',event)
                        self.assertNotIn('抽到「',event['text'])
            self.assertIsNone(payload.get('seed'))
            dumped = json.dumps(payload)
            self.assertNotIn('"seed"', dumped)
            self.assertNotIn('"rng"', dumped)
            self.assertNotIn('"deck":', dumped)
            self.assertEqual(leak_markers(payload), [])
            public_file_ok(payload)
            self.assertGreaterEqual(len(payload['views']['a']['events']), 1)
            self.assertGreaterEqual(len(payload['views']['b']['events']), 1)
            event_types = {event.get('type') for event in payload['views']['a']['events']}
            self.assertTrue(event_types)
            exported = Path(tmp) / 'share.json'
            result = export_game_replay(output, 'starter-00-first', exported)
            self.assertTrue(result['ok'])
            copied = json.loads(exported.read_text(encoding='utf-8'))
            self.assertEqual(copied['room_code'], payload['room_code'])
            self.assertEqual(leak_markers(copied), [])

    def test_cli_export_does_not_load_models(self):
        opponent_deck = next(item for item in STARTER_DECKS if item['id'] == 'starter')
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'src'
            run_offline_analysis(
                source,
                LEARNER,
                deck='starter',
                opponent_policy=OPPONENT,
                opponent_deck=opponent_deck,
                pairs=1,
                max_seconds=20,
                max_actions=6,
                seed_base=3,
            )
            dest = Path(tmp) / 'out.json'
            with patch.object(analyze_cli, '_load_frozen', side_effect=AssertionError('must not load')):
                code = analyze_cli.main([
                    '--export-game', 'starter-00-second',
                    '--source', str(source),
                    '--replay-output', str(dest),
                ])
            self.assertEqual(code, 0)
            payload = json.loads(dest.read_text(encoding='utf-8'))
            public_file_ok(payload)
            self.assertNotIn('seed', payload)

    def test_cli_rejects_pt_without_loading_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'starter.pt'
            model.write_bytes(b'pt')
            stderr = StringIO()
            with patch('sys.stderr', stderr):
                with self.assertRaises(SystemExit) as raised:
                    analyze_cli.main(['--model', str(model), '--output', str(Path(tmp) / 'new')])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn('export', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
