from pathlib import Path
import unittest

from flask import Flask
from app.modules.guide import blueprint

ROOT = Path(__file__).resolve().parents[1]


class GuideBuildsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()

    def test_character_tools_link_to_separate_public_pages(self):
        detail = self.client.get('/guide/characters/1042').text
        self.assertIn('href="/guide/characters/1042/builds"', detail)
        self.assertIn('href="/guide/characters/1042/cultivation"', detail)
        self.assertNotIn('id="related-list"', detail)
        self.assertNotIn('related_axes.js', detail)
        builds = self.client.get('/guide/characters/1042/builds')
        self.assertEqual(builds.status_code, 200)
        self.assertNotIn('window.location.replace', builds.text)
        self.assertIn('data-guide-character-id="1042"', builds.text)
        self.assertIn('data-guide-character-name="黑羽"', builds.text)
        self.assertNotIn('data-guide-character-name="娜娜莉"', builds.text)
        self.assertIn('id="related-list"', builds.text)
        self.assertIn('related_axes.js', builds.text)

    def test_unknown_character_is_not_a_generic_unfiltered_page(self):
        self.assertEqual(self.client.get('/guide/characters/missing/builds').status_code, 404)
        self.assertEqual(self.client.get('/guide/characters/1010/builds').status_code, 200)
        for asset in ['character_tools.css', 'related_axes.css', 'related_axes.js']:
            with self.client.get('/static/guide/' + asset) as response:
                self.assertEqual(response.status_code, 200, asset)


if __name__ == '__main__':
    unittest.main()
