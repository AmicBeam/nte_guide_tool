from pathlib import Path
import unittest
from unittest.mock import patch
from flask import Flask
from app.modules.guide import blueprint, presentation, service

ROOT = Path(__file__).resolve().parents[1]


class GuideHeaderTest(unittest.TestCase):
    def test_header_uses_game_version_and_does_not_confuse_schema_version(self):
        with patch.object(presentation, 'site_metadata', return_value={'title':'攻略数据库','game_version':'1.4.16'}):
            with patch.object(service, 'catalog', return_value={'schema_version':1}):
                self.assertEqual(presentation.site_header()['game_version'], '1.4.16')
            with patch.object(service, 'catalog', return_value={'schema_version':1,'game_version':'1.4.7'}):
                self.assertEqual(presentation.site_header()['game_version'], '1.4.16')
        with patch.object(presentation, 'site_metadata', return_value={'title':'攻略数据库','game_version':None}), patch.object(service, 'catalog', return_value={'schema_version':1}):
            self.assertIsNone(presentation.site_header()['game_version'])

    def test_index_details_and_editor_share_header(self):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        app.add_url_rule('/login', endpoint='main.login_page', view_func=lambda: 'login')
        with patch.object(presentation, 'site_metadata', return_value={'title':'攻略数据库','game_version':'1.4.16'}):
            for path in ('/guide','/guide/characters/1042','/guide/endless/EndlessTrial_1_50','/guide/write'):
                page = app.test_client().get(path)
                self.assertEqual(page.status_code, 200, path)
                self.assertIn('class="guide-header-title"', page.text)
                self.assertIn('<strong>攻略数据库</strong>', page.text)
                self.assertIn('游戏数据 v1.4.16', page.text)
                self.assertNotIn('数据逻辑版本', page.text)
