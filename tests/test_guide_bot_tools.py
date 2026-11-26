"""Website wiki CLI tests; account and image helpers live in qqbot_fairy."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

cli = module('bot_wiki_cli_test', 'app/modules/guide/cli.py')


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


class WikiPrivateCliTest(unittest.TestCase):
    def setUp(self):
        env_patch = patch.dict(os.environ, {'NTE_GUIDE_URL': 'http://127.0.0.1:5001'})
        env_patch.start()
        self.addCleanup(env_patch.stop)

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

    def test_reference_commands_can_send_preview_token_without_redirecting(self):
        for args in (['list', 'characters'], ['search', 'Exe'],
                     ['get', 'characters', '1074'], ['get', 'characters', '1074', '--format', 'markdown']):
            response = unittest.mock.MagicMock()
            response.__enter__.return_value = response
            response.read.return_value = b'{"data": {}}'
            response.headers.get_content_type.return_value = 'application/json'
            with self.subTest(args=args), patch.dict(os.environ, {'NTE_GUIDE_TOKEN': 'synthetic-tester'}), patch.object(cli.urllib.request, 'build_opener') as opener:
                opener.return_value.open.return_value = response
                code, output, error = self.run_cli(args)
                self.assertEqual(code, 0, error)
                request = opener.return_value.open.call_args.args[0]
                self.assertEqual(request.get_header('Authorization'), 'Bearer synthetic-tester')
                with self.assertRaises(ValueError):
                    opener.call_args.args[0].redirect_request(None, None, 302, '', {}, 'https://elsewhere.invalid')
                self.assertNotIn('synthetic-tester', output + error)

    def test_filters_do_not_select_another_role_or_mutate_original(self):
        payload = {'data': [profile()]}
        with self.assertRaises(ValueError):
            cli.filter_profiles(payload, 'not-owned')
        with self.assertRaises(ValueError):
            cli.filter_profiles(payload, character_id='not-owned')
        self.assertEqual(len(payload['data'][0]['characters']), 1)


if __name__ == '__main__':
    unittest.main()
