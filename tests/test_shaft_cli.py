import io
import json
import os
import shutil
import subprocess
import sys
import unittest
import urllib.error
import urllib.parse
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from tests import test_solo_room_flow


class ShaftCliTest(unittest.TestCase):
    def setUp(self):
        saved_env = patch.dict(os.environ)
        saved_env.start()
        self.addCleanup(saved_env.stop)
        site = test_solo_room_flow.RoomFlowTestCase()
        self.addCleanup(site.tearDown)
        site.setUp()
        self.site = site
        self.client = site.client
        from app.modules.guide import cli
        self.cli = cli
        self.paths = []

    def open(self, request, timeout):
        url = urllib.parse.urlsplit(request.full_url)
        path = url.path + ('?' + url.query if url.query else '')
        self.paths.append(path)
        with self.client.open(path, method=request.get_method(), data=request.data,
                              headers=request.header_items()) as response:
            body = io.BytesIO(response.data)
            if response.status_code >= 400:
                raise urllib.error.HTTPError(request.full_url, response.status_code, '', response.headers, body)
            return body

    def call(self, token, *args):
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {'NTE_GUIDE_TOKEN': token}), \
                patch.object(self.cli.urllib.request, 'build_opener', return_value=self), \
                redirect_stdout(out), redirect_stderr(err):
            code = self.cli.main(['--base-url', 'https://example.test', 'shaft', *args])
        return code, out.getvalue(), err.getvalue()

    def run_cli(self, token, *args):
        code, out, err = self.call(token, *args)
        self.assertEqual(code, 0, err)
        return json.loads(out)

    def test_login_required_before_using_shaft_commands(self):
        self.assertEqual(self.client.get('/api/shaft/cli-runtime').status_code, 401)
        code, _, err = self.call('', 'catalog')
        self.assertEqual(code, 1)
        self.assertIn('NTE_GUIDE_TOKEN', err)
        self.assertEqual(self.paths, [])
        code, _, err = self.call('invalid-test-token', 'simulate', 'missing.json')
        self.assertEqual(code, 1)
        self.assertIn('401', err)
        self.assertNotIn('invalid-test-token', err)
        self.assertEqual(self.paths, ['/api/me'])

    def test_bundle_and_version(self):
        skill_url = '/guide/skills/shaft-axis-workflow.zip'
        self.assertIn(skill_url, self.client.get('/guide/llms.txt').text)
        self.assertEqual(self.client.get('/guide/skills/unknown.zip').status_code, 404)
        with self.client.get(skill_url) as bundle, self.client.get('/guide/cli.py') as script:
            self.assertEqual(bundle.mimetype, 'application/zip')
            with ZipFile(io.BytesIO(bundle.data)) as archive:
                self.assertCountEqual(archive.namelist(), [
                    'shaft-axis-workflow/SKILL.md',
                    'shaft-axis-workflow/examples/move-step.js',
                    'shaft-axis-workflow/nte-guide.py',
                ])
                packaged_cli = archive.read('shaft-axis-workflow/nte-guide.py')
                self.assertEqual(packaged_cli, script.data)
        target = Path(self.site.temp_dir.name) / 'nte-guide.py'
        target.write_bytes(packaged_cli)
        self.assertEqual(subprocess.check_output([sys.executable, str(target), '--version'], text=True).strip(),
                         f'nte-guide {self.cli.CLI_VERSION}')
        self.assertIn(skill_url, subprocess.check_output(
            [sys.executable, str(target), 'shaft', '--help'], text=True))

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for simulator integration')
    def test_share_simulate_create_update_and_guard_stale_writes(self):
        token = self.site._issue_login_and_get_token('shaft-cli-owner')
        other = self.site._issue_login_and_get_token('shaft-cli-visitor')
        filename = Path(self.site.temp_dir.name) / 'axis.json'

        def use(document):
            filename.write_text(json.dumps(document), encoding='utf-8')
            return str(filename)

        catalog = self.run_cli(token, 'catalog')
        calculated = self.run_cli(token, 'simulate', use(catalog['starter_axis']))
        self.assertIsInstance(calculated['self_check'], list)
        self.assertGreater(calculated['result']['summary']['duration_ticks'], 0)
        created = self.run_cli(token, 'create', use(calculated), '--title', 'CLI Test')
        axis_id = str(created['id'])
        self.assertEqual(created['visibility'], 'private')
        self.assertEqual(self.run_cli(token, 'axis', axis_id)['axis'], created['axis'])
        self.assertEqual(self.run_cli(token, 'axes')['items'][0]['id'], created['id'])
        self.assertEqual(self.call(other, 'axis', axis_id)[0], 1)
        share = self.site._post(f'/api/shaft/axes/{axis_id}/share', {}, token=token)
        self.assertEqual(self.run_cli(other, 'shared', share['share_path'])['id'], created['id'])
        self.assertEqual(self.call(other, 'shared', 'https://elsewhere.invalid' + share['share_path'])[0], 1)

        step = created['axis']['steps'][-1]
        new_tick = step['start_tick'] + 1
        example = Path(self.cli.__file__).parent / 'skills/shaft-axis-workflow/examples/move-step.js'
        edited = json.loads(subprocess.check_output([
            'node', str(example), use(created), step['id'], str(new_tick),
        ]))
        self.assertNotIn('result', edited)
        edit = self.run_cli(token, 'simulate', use({**edited, 'description': 'Changed with CLI'}))
        updated = self.run_cli(token, 'update', axis_id, use(edit))
        self.assertEqual(updated['description'], 'Changed with CLI')
        self.assertEqual(next(item['start_tick'] for item in updated['axis']['steps']
                              if item['id'] == step['id']), new_tick)
        self.assertIn('409', self.call(token, 'update', axis_id, use(edit))[2])
        edit['axis']['steps'].append(edit['axis']['steps'][0])
        self.assertIn('simulate', self.call(token, 'update', axis_id, use(edit))[2])
        self.assertEqual(self.run_cli(token, 'axis', axis_id)['updated_at'], updated['updated_at'])
