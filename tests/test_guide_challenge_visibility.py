"""All public readers share the release cutoff, including direct CLI reads."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.parse import urlsplit

from flask import Flask
from app.modules.guide import blueprint, service, chat_tools
from app.modules.guide.challenge_calendar import is_released_challenge


class ChallengeVisibilityTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        app = Flask(__name__, template_folder=str(root / 'app/templates'))
        app.register_blueprint(blueprint)
        self.client = app.test_client()
        self.clock = patch('app.modules.guide.challenge_calendar.current_time',
                           return_value=datetime.fromisoformat('2026-10-17T12:00:00+08:00')).start()
        self.addCleanup(patch.stopall)
        spec = importlib.util.spec_from_file_location('visibility_cli', root / 'app/modules/guide/cli.py')
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)

    def test_lists_search_counts_and_chat_exclude_future(self):
        result = self.client.get('/api/guide/challenges?limit=100').json
        expected = {'Abyss_Common'} | {f'Abyss_{number}' for number in range(1, 13)}
        self.assertEqual({row['id'] for row in result['items']}, expected)
        self.assertEqual(result['total'], 13)
        self.assertIsNone(result['next_offset'])
        for page in ['/guide?kind=challenges', '/guide', '/guide/account']:
            html = self.client.get(page).text
            self.assertNotIn('/guide/challenges/Abyss_13', html)
            self.assertNotIn('/guide/challenges/Abyss_14', html)
            self.assertIn('轨外之境 <span>13</span>', html)
        for identifier in ['Abyss_13', 'Abyss_14']:
            self.assertEqual(self.client.get('/api/guide/search?q=' + identifier).json['total'], 0)
            toolkit = chat_tools.Toolkit(None, None)
            result, _ = toolkit.execute('wiki_search', {'kind': 'challenges', 'query': identifier})
            self.assertEqual(result['total'], 0)
        self.assertEqual(len(service.catalog()['challenges']), 15)

    def test_future_direct_reads_are_not_found_in_every_format(self):
        for identifier in ['Abyss_13', 'Abyss_14']:
            for path in [f'/guide/challenges/{identifier}', f'/guide/challenges/{identifier}.md',
                         f'/api/guide/challenges/{identifier}?section=resonance&stage=12']:
                response = self.client.get(path)
                self.assertEqual(response.status_code, 404, path)
                self.assertEqual(response.json['error']['code'], 'not_found')
            with self.assertRaises(LookupError):
                chat_tools.Toolkit(None, None).execute('wiki_get', {'kind': 'challenges', 'id': identifier})
        for identifier in ['Abyss_1', 'Abyss_12', 'Abyss_Common']:
            for suffix in ['', '.md']:
                self.assertEqual(self.client.get('/guide/challenges/' + identifier + suffix).status_code, 200)

    def test_release_uses_beijing_five_am_without_caching(self):
        # Warm the offline catalog before advancing the request clock.
        service.catalog()
        for timestamp, released in [('2026-10-30T04:59:59+08:00', False),
                                    ('2026-10-29T21:00:00+00:00', True)]:
            self.clock.return_value = datetime.fromisoformat(timestamp)
            expected_status = 200 if released else 404
            self.assertEqual(self.client.get('/api/guide/challenges/Abyss_13').status_code, expected_status)
            ids = {row['id'] for row in self.client.get('/api/guide/challenges?limit=100').json['items']}
            self.assertEqual('Abyss_13' in ids, released)
            self.assertNotIn('Abyss_14', ids)
        self.assertTrue(is_released_challenge({'is_permanent': True}))
        for begin in [None, '', 'invalid']:
            self.assertFalse(is_released_challenge({'begin_date': begin}))
        self.clock.return_value = datetime.fromisoformat('2026-11-13T05:00:00+08:00')
        self.assertEqual(self.client.get('/api/guide/challenges/Abyss_14').status_code, 200)

    def test_cli_list_search_get_share_server_cutoff(self):
        def opening(url, **kwargs):
            parsed = urlsplit(url)
            response = self.client.get(parsed.path + '?' + parsed.query)
            if response.status_code >= 400:
                raise HTTPError(url, response.status_code, 'not found', {}, io.BytesIO(response.data))
            stream = MagicMock()
            stream.__enter__.return_value = stream
            stream.read.return_value = response.data
            stream.headers.get_content_type.return_value = response.mimetype
            return stream
        with patch.object(self.cli.urllib.request, 'urlopen', side_effect=opening):
            for args in [['list', 'challenges'], ['search', 'Abyss_13']]:
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(self.cli.main(args), 0)
                result = json.loads(output.getvalue())
                self.assertNotIn('Abyss_13', {row['id'] for row in result['items']})
                self.assertNotIn('Abyss_14', {row['id'] for row in result['items']})
            for form in ['json', 'markdown']:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
                    self.assertEqual(self.cli.main(['get', 'challenges', 'Abyss_13', '--format', form]), 1)
                    self.assertIn('HTTP 404', errors.getvalue())
