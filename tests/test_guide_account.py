"""Account page is a public shell; saved profiles stay behind existing auth APIs."""
from pathlib import Path
import unittest
from unittest.mock import patch

from flask import Flask
from app.modules.guide import blueprint, service, user_data, game_sync


class GuideAccountTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(Path(__file__).resolve().parents[1] / 'app/templates'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()

    def test_account_shell_reads_only_local_catalog(self):
        with patch.object(user_data, 'profiles') as private_read, patch.object(game_sync, 'send_sms') as sms:
            page = self.client.get('/guide/account')
        self.assertEqual(page.status_code, 200)
        self.assertNotIn('window.location.replace', page.text)
        self.assertIn('data-account-profiles', page.text)
        self.assertIn('account-catalog', page.text)
        self.assertIn('aria-current="page">账号练度', page.text)
        private_read.assert_not_called()
        sms.assert_not_called()

    def test_account_tab_follows_endless_in_every_catalog(self):
        for kind in service.KINDS:
            page = self.client.get('/guide?kind=' + kind)
            self.assertEqual(page.status_code, 200)
            self.assertLess(page.text.index('全站深境'), page.text.index('>账号练度</a>'))
            self.assertIn('href="/guide/account"', page.text)

    def test_cultivation_keeps_shared_sync_dialog(self):
        page = self.client.get('/guide/characters/1042/cultivation')
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.text.count('id="sync-dialog"'), 1)
        self.assertIn('id="use-profile"', page.text)
