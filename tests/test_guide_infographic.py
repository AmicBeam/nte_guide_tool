import copy
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from flask import Flask
from app.modules.guide import blueprint, infographic, infographic_chat, infographic_script, chat_service, chat_tools

ROOT = Path(__file__).resolve().parents[1]


class InfographicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        app.add_url_rule('/login', endpoint='main.login_page', view_func=lambda: 'login')
        cls.client = app.test_client()
        cls.options = infographic.options('1042')
        # Exercise the same defaults/import contract in JS and Python.
        raw = subprocess.check_output(['node', '-e',
            "const s=require('./app/modules/guide/static/infographic_state.js');let d=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(s.defaults(d)))"],
            input=json.dumps(cls.options).encode(), cwd=ROOT)
        cls.config = json.loads(raw)

    def context(self):
        return infographic.editor_context({'character_id': '1042', 'revision': 7, 'config': copy.deepcopy(self.config)},
                                          '/guide/characters/1042/infographic')

    def test_public_entry_data_and_preview_boundary(self):
        self.assertIn('/guide/characters/1042/infographic', self.client.get('/guide/characters/1042').text)
        response = self.client.get('/guide/characters/1042/infographic')
        self.assertEqual(response.status_code, 200)
        self.assertIn('ig-canvas', response.text)
        self.assertIn('infographic_state.js', response.text)
        self.assertIn('infographic_board.js', response.text)
        self.assertIn('id="ig-editor-workspace" data-character-id="1042" hidden', response.text)
        self.assertNotIn('default_script', response.text)
        self.assertEqual(self.client.get('/api/guide/characters/1042/infographic').status_code, 401)
        self.assertTrue(all(c['element_icon'] and c['rarity_icon'] for c in self.options['characters']))
        self.assertNotIn('ig-data', self.client.get('/guide/characters/1042').text)
        self.assertEqual(self.client.get('/guide/characters/1090/infographic').status_code, 404)
        self.assertEqual(self.client.get('/guide/characters/missing/infographic').status_code, 404)
        for char in infographic.service.catalog()['characters']:
            opts = infographic.options(char['id'])
            self.assertTrue(all(str(a['type_id']) == str(char['arc_type']['id']) for a in opts['arcs']))

    def test_bounded_config_rejects_unknown_ids_and_malformed_nested_fields(self):
        self.assertEqual(infographic.validate(self.config, self.options), self.config)
        for change in [{'title': 'x'*61}, {'order': ['stats']*7}, {'order': [{}]*7},
                       {'radar': [True]*6}, {'radar': [101]*6}, {'art_scale': float('nan')},
                       {'theme': []}, {'radar_labels': ['a']}, {'radar_labels': ['']*6}, {'positions': {'stats': {'x': float('nan'), 'y': 4}}}, {'positions': {'bad-key': {'x': 1, 'y': 2}}}, {'modules': {}}, {'plans': [None,None]}, {'extra': 'x'}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                infographic.validate({**self.config, **change}, self.options)
        for name, field, value in [('plans', 'arc', 'other-game'), ('plans', 'arc', []), ('teams', 'members', ['missing']*4)]:
            bad = copy.deepcopy(self.config)
            bad[name][0][field] = value
            with self.assertRaises(ValueError):
                infographic.validate(bad, self.options)

    def test_candidates_accept_full_roster_and_reject_same_slot_duplicates(self):
        config = copy.deepcopy(self.config)
        config['teams'][0]['candidates'][0] = [c['id'] for c in self.options['characters'] if c['id'] != '1042']
        self.assertEqual(infographic.validate(config, self.options), config)
        config['teams'][0]['candidates'][0].append('1042')
        with self.assertRaises(ValueError):
            infographic.validate(config, self.options)
        config = copy.deepcopy(self.config)
        config['teams'][0]['folded'][0] = 'true'
        with self.assertRaises(ValueError):
            infographic.validate(config, self.options)

    def test_script_declaration_and_unique_edits_are_available_to_page_bot(self):
        context = self.context()
        value = infographic_chat.update(context, {'patch': {}, 'script_edits': [{'old': "const FONT =", 'new': "const FONT ="}]})
        self.assertTrue(value['prepared'])
        with self.assertRaises(ValueError):
            infographic_chat.update(context, {'patch': {}, 'script_edits': [{'old': 'not-present-code', 'new': ''}]})
        source = '/* @editor {"builtin":[],"controls":[{"key":"heading","label":"标题","type":"text","default":"新标题"}]} */\nwindow.GuideInfographicCanvas={render:async()=>({})};'
        infographic_chat.update(context, {'patch': {'script': source}})
        self.assertEqual(context['config']['script_params'], {'heading':'新标题'})
        self.assertIn('@editor', infographic_chat.prompt(context))
        self.assertIn('Worker', infographic_chat.prompt(context))
        for code in ['while(true){}', '/* @editor {invalid} */', source.replace('heading', 'constructor')]:
            with self.assertRaises(ValueError):
                infographic_script.declaration(code)

    def test_editor_scope_and_patch_are_transient(self):
        context = self.context()
        for path in ('/guide', '/guide/characters/1042', '/guide/characters/1057/infographic'):
            with self.assertRaises(ValueError):
                infographic.editor_context({k: v for k, v in context.items() if k != 'options'}, path)
        original = copy.deepcopy(self.config)
        infographic_chat.update(context, {'patch': {'title': '黑羽 · 入门配装', 'orientation': 'portrait'}})
        self.assertEqual(context['config']['orientation'], 'portrait')
        self.assertEqual(self.config, original)
        self.assertFalse(any(t['function']['name'] == 'update_infographic' for t in chat_tools.TOOLS))
        with self.assertRaises(ValueError):
            infographic_chat.update(context, {'patch': {'__proto__': {}}})

    def test_model_tool_is_page_only_and_patch_waits_for_complete_answer(self):
        calls = [{'id': 'edit-1', 'type': 'function', 'function': {'name': 'update_infographic',
                  'arguments': json.dumps({'patch': {'orientation': 'portrait'}})}}]
        observed = []
        def stream(settings, messages, tools):
            observed.append(tools)
            yield {'type': 'message', 'message': {'tool_calls': calls} if len(observed) == 1 else {'content': '已改为竖排。'}}
        with patch.object(chat_service.chat_model, 'stream', side_effect=stream):
            result = list(chat_service.events({}, [{'role': 'user', 'content': '改成竖排'}], None, None,
                                              '/guide/characters/1042/infographic', editor=self.context()))
        self.assertTrue(any(t['function']['name'] == 'update_infographic' for t in observed[0]))
        changes = [event for event in result if event['type'] == 'infographic_patch']
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]['revision'], 7)
        self.assertEqual(changes[0]['config']['orientation'], 'portrait')
        self.assertEqual(result[-1]['type'], 'answer')
        observed.clear()
        def answer(*args):
            yield {'type':'message','message':{'content':'普通问答'}}
        with patch.object(chat_service.chat_model, 'stream', side_effect=answer):
            list(chat_service.events({}, [], None, None, '/guide'))
        def interrupted(*args):
            if interrupted.called:
                raise RuntimeError('synthetic interruption')
            interrupted.called = True
            yield {'type':'message','message':{'tool_calls':calls}}
        interrupted.called = False
        result = []
        with patch.object(chat_service.chat_model, 'stream', side_effect=interrupted), self.assertRaises(RuntimeError):
            for event in chat_service.events({}, [], None, None, '/guide/characters/1042/infographic', editor=self.context()):
                result.append(event)
        self.assertNotIn('infographic_patch', [event['type'] for event in result])
