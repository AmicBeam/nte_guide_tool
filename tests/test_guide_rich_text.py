import importlib.util
from html import unescape
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from flask import Flask
from app.modules.guide import blueprint, service
from app.modules.guide.rich_text import description_data, description_html

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sync_guide_rich_text', ROOT / 'scripts/sync_guide_rich_text.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class GuideRichTextTest(unittest.TestCase):
    def test_markup_parser_keeps_color_and_explicit_noun(self):
        parsed = sync.runs('<Hun>魂属性</>与<hot textstyle="TextEXPL" param="nouns&Blackbirdmalice">恶意</>\r\n尾行')
        self.assertEqual(parsed, [{'style': 'hun', 'text': '魂属性'},
            {'text': '与'}, {'style': 'textexpl', 'noun': 'blackbirdmalice', 'text': '恶意'}, {'text': '\n尾行'}])
        self.assertEqual(sync.runs('<unknown>文字</>'), [{'text': '文字'}])

    def test_render_escapes_untrusted_content_and_color(self):
        malicious = '<img src=x onerror=alert(1)>'
        data = {'colors': {'hun': 'red;position:fixed'}, 'characters': {'1': {
            'descriptions': {'test': [{'style': 'hun', 'text': malicious, 'noun': 'n'}]},
            'nouns': {'n': {'title': malicious, 'runs': [{'text': malicious}]}}}}}
        with patch('app.modules.guide.rich_text.description_data', return_value=data):
            result = str(description_html('test', '1'))
            self.assertNotIn('<img', result)
            self.assertNotIn('position:fixed', result)
            self.assertIn('&lt;img', result)
            self.assertIn('role="tooltip" hidden', result)
            self.assertEqual(str(description_html(malicious, 'missing')), '&lt;img src=x onerror=alert(1)&gt;')

    def test_all_current_skills_have_text_equivalent_segments(self):
        data = description_data()
        self.assertEqual(len(data['characters']), 25)
        for character in service.catalog()['characters']:
            rich = data['characters'][str(character['id'])]
            for ability in character['abilities']:
                for group in ability['groups']:
                    text = group.get('description')
                    if text:
                        segments = rich['descriptions'][text]
                        self.assertEqual(''.join(r['text'] for r in segments), text)
                        for row in segments:
                            if row.get('noun'):
                                self.assertIn(row['noun'], rich['nouns'])

    def test_page_has_colored_skill_and_noun_explanation(self):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        client = app.test_client()
        html = client.get('/guide/characters/1042').text
        self.assertIn('color:#80DDFF', html)
        self.assertIn('class="guide-term"', html)
        self.assertIn('黑羽发动', unescape(html))
        ids = re.findall(r'id="(guide-term-\d+)"', html)
        self.assertTrue(ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn('<hot ', html)
        self.assertNotIn('<Hun>', html)
        plain = client.get('/api/guide/characters/1042?section=abilities').json
        self.assertNotIn('guide-term', str(plain))
        self.assertNotIn('<hot ', str(plain))


if __name__ == '__main__':
    unittest.main()
