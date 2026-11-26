"""Narrow CLI/API/model reads retain exact values without bulk context."""
from tests.guide_release_fixtures import published_character_data
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
from contextlib import redirect_stdout
from urllib.parse import urlsplit, parse_qs

from flask import Flask
from app.modules.guide import blueprint, service, chat_tools, chat_service

ROOT = Path(__file__).resolve().parents[1]


@published_character_data
class NarrowReadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()

    def test_overviews_cover_roster_without_mutating_full_records(self):
        toolkit = chat_tools.Toolkit(None, None)
        for character in service.catalog()['characters']:
            with self.subTest(character=character['id']):
                before = json.dumps(character, ensure_ascii=False)
                result, _ = toolkit.execute('wiki_get', {'kind': 'characters', 'id': character['id']})
                self.assertLess(len(json.dumps(result, ensure_ascii=False)), 11000)
                self.assertNotIn('experience', result)
                self.assertNotIn('materials', result)
                self.assertEqual({a['id'] for a in result['abilities']}, {a['id'] for a in character['abilities']})
                self.assertTrue(all('param_rows' not in a for a in result['abilities']))
                self.assertEqual(json.dumps(character, ensure_ascii=False), before)
                self.assertIn('abilities', result['available_sections'])
        default = self.client.get('/api/guide/characters/1042').json['data']
        self.assertEqual(len(default['experience']), 80)
        self.assertIsInstance(default['abilities'][0]['param_rows'][0]['values'], list)

    def test_one_ability_one_level_matches_full_data_in_both_formats(self):
        from app.modules.guide.reference_view import at_level
        self.assertEqual(at_level({'values': ['first', 'last']}, 10),
                         {'values': {'level': 2, 'max_level': 2, 'value': 'last'}})
        original = service.detail('characters', '1042')['abilities'][0]
        query = '?ability_id=' + original['id'] + '&level=10'
        result = self.client.get('/api/guide/characters/1042' + query).json['data']
        self.assertEqual(len(result['abilities']), 1)
        ability = result['abilities'][0]
        row = ability['param_rows'][0]
        self.assertEqual(row['values'], {'level': 10, 'max_level': 13, 'value': original['param_rows'][0]['values'][9]})
        self.assertEqual(row['hit_details'][0]['attack_values']['value'], original['param_rows'][0]['hit_details'][0]['attack_values'][9])
        self.assertEqual(row['hit_details'][0]['energy_gain'], original['param_rows'][0]['hit_details'][0]['energy_gain'])
        rows = {row['key']: row for row in ability['param_rows']}
        for group in ability['groups']:
            for key in group.get('parameter_keys', []):
                self.assertIn(key, rows)
        md = self.client.get('/guide/characters/1042.md' + query)
        self.assertEqual(md.status_code, 200)
        self.assertIn(original['param_rows'][0]['values'][9], md.text)
        for query in ['?level=0', '?level=14', '?level=abc', '?ability_id=missing',
                      '?ability_id=' + original['id'] + '&section=materials']:
            self.assertIn(self.client.get('/api/guide/characters/1042' + query).status_code, (400, 404))
        self.assertEqual(self.client.get('/api/guide/arcs/fork_twinbirds?ability_id=anything').status_code, 400)

    def test_tool_uses_overview_and_cli_passes_narrowing_parameters(self):
        toolkit = chat_tools.Toolkit(None, None)
        for args in [{'kind': 'characters', 'id': '1042'},
                     {'kind': 'characters', 'id': '1042', 'section': 'abilities'}]:
            result, _ = toolkit.execute('wiki_get', args)
            self.assertIn('read_hint', result)
            self.assertNotIn('param_rows', result['abilities'][0])
        spec = importlib.util.spec_from_file_location('narrow_cli', ROOT / 'app/modules/guide/cli.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"data":{}}'
        response.headers.get_content_type.return_value = 'application/json'
        with patch.object(cli.urllib.request, 'urlopen', return_value=response) as opening, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['get', 'characters', '1042', '--ability', 'GA_BlackBird_Melee', '--level', '10']), 0)
        self.assertEqual(parse_qs(urlsplit(opening.call_args.args[0]).query),
                         {'ability_id': ['GA_BlackBird_Melee'], 'skill_level': ['10'], 'hit_details': ['0']})

    def test_status_distinguishes_model_processing_from_local_read(self):
        calls = 0
        def stream(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield {'type': 'message', 'message': {'tool_calls': [{'id': 'one', 'type': 'function',
                    'function': {'name': 'wiki_get', 'arguments': '{"kind":"characters","id":"1042"}'}}]}}
            else:
                yield {'type': 'delta', 'text': '回答'}
                yield {'type': 'message', 'message': {'content': '回答'}}
        with patch.object(chat_service.chat_model, 'stream', side_effect=stream):
            events = list(chat_service.events({}, [{'role': 'user', 'content': '黑羽怎么玩'}], None, None, '/guide'))
        statuses = [event['text'] for event in events if event['type'] == 'status']
        self.assertIn('正在查询本站资料…', statuses)
        self.assertEqual(statuses[-1], '查询已完成，正在整理回答…')
