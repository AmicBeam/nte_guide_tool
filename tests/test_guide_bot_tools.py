"""Bot integration components use synthetic identities and private snapshots only."""
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


access = module('bot_wiki_access_test', 'plugins/nte_guide_access.py')
cli = module('bot_wiki_cli_test', 'app/modules/guide/cli.py')
card = module('bot_wiki_card_test', 'plugins/nte_build_card.py')


def profile(role='role-one'):
    return {'role_id': role, 'role_name': '合成验证账号', 'updated_at': '2026-10-16T12:00:00Z',
        'characters': [{'id': '1042', 'name': '黑羽', 'level': 70, 'awaken_level': 2, 'resonance_level': 1,
            'properties': [{'id': 'atk', 'name': '攻击力', 'value': '2,100'},
                           {'id': 'crit', 'name': '暴击率', 'value': '65.0%'}],
            'skills': [{'id': 'GA_BlackBird_Melee', 'name': '掠影', 'level': 8, 'group': 'skills'}],
            'equipment': {'fork': {'id': 'fork_test', 'name': '合成测试弧盘', 'alev': '70', 'blev': '5', 'slev': '1',
                'properties': [{'id': 'atk', 'name': '基础攻击', 'value': '520'}]},
                'suit': {'id': 'suit_test', 'name': '合成测试套装', 'core': [{'id': 'piece-one', 'name': '核心卡带', 'level': 20,
                    'main_properties': [{'id': 'crit', 'name': '暴击率', 'value': '12.0%'}],
                    'properties': [{'id': 'atk', 'name': '攻击力', 'value': '8.0%'}]}], 'pie': []}}}]}


class BotTokenAccessTest(unittest.TestCase):
    def test_read_only_sender_scope_expiry_revocation_and_no_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / 'accounts.sqlite3'
            self.assertIsNone(access.get_sender_token('10001', db))
            self.assertFalse(db.exists())
            with sqlite3.connect(db) as connection:
                connection.executescript('''CREATE TABLE player(id INTEGER PRIMARY KEY, player_uid TEXT);
                    CREATE TABLE accesstoken(id INTEGER PRIMARY KEY, player_id INTEGER, token TEXT,
                    revoked INTEGER, expires_at TEXT, created_at TEXT);
                    INSERT INTO player VALUES(1,'10001'); INSERT INTO player VALUES(2,'10002');''')
                now = datetime.utcnow()
                rows = [(1, 1, 'owner-valid', 0, now + timedelta(days=1), now),
                        (2, 1, 'owner-revoked', 1, now + timedelta(days=1), now + timedelta(seconds=1)),
                        (3, 1, 'owner-expired', 0, now - timedelta(days=1), now + timedelta(seconds=2)),
                        (4, 2, 'other-valid', 0, now + timedelta(days=1), now + timedelta(seconds=3)),
                        (5, 1, 'bad-expiry', 0, 'invalid', now + timedelta(seconds=4))]
                connection.executemany('INSERT INTO accesstoken VALUES(?,?,?,?,?,?)', rows)
            before = db.read_bytes()
            self.assertEqual(access.get_sender_token('10001', db), 'owner-valid')
            self.assertEqual(access.get_token_for_qq_id('10002', db), 'other-valid')
            self.assertIsNone(access.get_sender_token('99999', db))
            self.assertIsNone(access.get_sender_token("10001' OR 1=1--", db))
            self.assertEqual(db.read_bytes(), before)
            with patch.dict(os.environ, {'NTE_DATABASE_PATH': str(db)}):
                self.assertEqual(access.get_sender_token('10001'), 'owner-valid')


class WikiPrivateCliTest(unittest.TestCase):
    def run_cli(self, args):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            result = cli.main(args)
        return result, output.getvalue(), error.getvalue()

    def test_missing_token_and_private_header_filtering(self):
        with patch.dict(os.environ, {'NTE_GUIDE_TOKEN': ''}):
            code, output, error = self.run_cli(['me'])
        self.assertEqual(code, 1)
        self.assertEqual(output, '')
        self.assertIn('NTE_GUIDE_TOKEN', error)
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({'data': [profile(), profile('role-two')]}).encode()
        response.headers.get_content_type.return_value = 'application/json'
        with patch.dict(os.environ, {'NTE_GUIDE_TOKEN': 'synthetic-secret'}), patch.object(cli.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value = response
            code, output, error = self.run_cli(['me', '--role-id', 'role-two', '--character', '1042'])
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual(request.get_header('Authorization'), 'Bearer synthetic-secret')
            self.assertEqual(request.full_url, 'http://127.0.0.1:5001/api/guide/me/profiles')
        self.assertEqual(code, 0, error)
        self.assertNotIn('synthetic-secret', output + error)
        self.assertEqual([p['role_id'] for p in json.loads(output)['data']], ['role-two'])

    def test_no_redirect_and_no_private_error_echo(self):
        import urllib.error
        secret = 'synthetic-secret'
        with patch.dict(os.environ, {'NTE_GUIDE_TOKEN': secret}), patch.object(cli.urllib.request, 'build_opener') as opener:
            opener.return_value.open.side_effect = urllib.error.HTTPError('http://localhost', 401, 'Unauthorized', {}, io.BytesIO(secret.encode()))
            code, output, error = self.run_cli(['me'])
            handler = opener.call_args.args[0]
            with self.assertRaises(ValueError):
                handler.redirect_request(None, None, 302, '', {}, 'https://elsewhere.invalid')
        self.assertEqual(code, 1)
        self.assertNotIn(secret, output + error)

    def test_filters_do_not_select_another_role_or_mutate_original(self):
        payload = {'data': [profile()]}
        with self.assertRaises(ValueError):
            cli.filter_profiles(payload, 'not-owned')
        with self.assertRaises(ValueError):
            cli.filter_profiles(payload, character_id='not-owned')
        self.assertEqual(len(payload['data'][0]['characters']), 1)


class BuildCardTest(unittest.TestCase):
    def test_selection_is_explicit_and_content_has_no_credentials(self):
        first = profile()
        first['token'] = 'do-not-render'
        chosen, character = card.choose_character({'data': [first]}, '1042')
        content = card.card_content(chosen, character)
        self.assertNotIn('do-not-render', str(content))
        self.assertIn('65.0%', str(content))
        self.assertEqual(content['fork']['title'], '合成测试弧盘')
        with self.assertRaises(ValueError):
            card.choose_character({'data': [first, profile('two')]}, '1042')
        self.assertEqual(card.choose_character({'data': [first, profile('two')]}, '1042', 'two')[0]['role_id'], 'two')
        with self.assertRaises(ValueError):
            card.choose_character(first, 'missing')

    def test_skill_labels_match_case_insensitive_ids(self):
        saved = profile()
        character = saved['characters'][0]
        character['skills'][0]['id'] = 'ga_blackbird_melee'
        character['skills'][0]['name'] = ''
        content = card.card_content(saved, character, {'abilities': [
            {'id': 'GA_BlackBird_Melee', 'type': 'melee', 'name': '掠影'}]})
        self.assertEqual(content['skills'][0], ('普通攻击 · 掠影', 'Lv.8'))

    def test_render_has_valid_image_and_private_permissions(self):
        try:
            from PIL import Image
            card.font_path()
        except (ImportError, ValueError):
            self.skipTest('Optional bot rendering requires Pillow and a configured CJK font')
        with tempfile.TemporaryDirectory() as folder:
            saved = profile()
            output = Path(folder) / 'build.png'
            result = card.render_build_card(saved, saved['characters'][0], output)
            self.assertEqual(result['width'], 1200)
            self.assertGreater(result['height'], 1000)
            with Image.open(output) as image:
                self.assertEqual(image.format, 'PNG')
                self.assertEqual(image.size, (result['width'], result['height']))
                image.verify()
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)


if __name__ == '__main__':
    unittest.main()
