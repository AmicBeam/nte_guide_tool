"""Challenge presentation keeps enemy traits and local assets available."""
from html.parser import HTMLParser
from pathlib import Path
import unittest

from flask import Flask
from app.modules.guide import blueprint, service

ROOT = Path(__file__).resolve().parents[1]


class DetailsParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.details = []

    def handle_starttag(self, tag, attrs):
        if tag == 'details':
            self.details.append(dict(attrs))


class GuideChallengesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()

    def test_all_enemy_bindings_and_assets(self):
        checked = set()
        for challenge in service.catalog()['challenges']:
            for stage in challenge['stages']:
                for half in stage['halves']:
                    for wave in half['waves']:
                        for monster in wave['monsters']:
                            self.assertIsInstance(monster['swallowable'], bool, monster['name'])
                            self.assertTrue(monster['image'].startswith('/static/guide/monsters/'))
                            if monster['image'] not in checked:
                                response = self.client.get(monster['image'])
                                self.assertEqual(response.status_code, 200)
                                self.assertEqual(response.data[8:12], b'WEBP')
                                response.close()
                                checked.add(monster['image'])
        self.assertEqual(len(checked), 37)

    def test_stage_layout_and_collapsed_stats(self):
        response = self.client.get('/guide/challenges/Abyss_12?stage=12')
        self.assertEqual(response.status_code, 200)
        html = response.text
        self.assertIn('百川归海', html)
        self.assertIn('提高45%', html)
        self.assertNotIn('鬼郎丸：', html)
        self.assertEqual(html.count('class="guide-swallow-badge"'), 3)
        self.assertEqual(html.count('alt="可被鬼郎丸吞噬"'), 3)
        badge = self.client.get('/static/guide/monsters/swallow-badge.webp')
        self.assertEqual(badge.status_code, 200)
        badge.close()
        self.assertIn('alt="瑰夜魅影"', html)
        self.assertNotIn('id="stage-11"', html)
        parser = DetailsParser()
        parser.feed(html)
        monster_details = [d for d in parser.details if d.get('class') == 'guide-monster-details']
        self.assertEqual(len(monster_details), 5)
        self.assertTrue(all('open' not in d for d in monster_details))
        all_stages = self.client.get('/guide/challenges/Abyss_12').text
        self.assertLess(all_stages.index('id="stage-12"'), all_stages.index('id="stage-1"'))

    def test_traits_in_machine_formats(self):
        record = self.client.get('/api/guide/challenges/Abyss_12?stage=1').json['data']
        monster = record['stages'][0]['halves'][0]['waves'][0]['monsters'][0]
        self.assertEqual(monster['name'], '棉绒绒')
        self.assertTrue(monster['swallowable'])
        md = self.client.get('/guide/challenges/Abyss_12.md?stage=1').text
        self.assertIn('可被鬼郎丸吞噬: 是', md)


if __name__ == '__main__':
    unittest.main()
