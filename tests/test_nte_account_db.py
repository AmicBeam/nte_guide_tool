import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path


class NTEAccountDatabaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_database_path = os.environ.get('NTE_DATABASE_PATH')
        os.environ['NTE_DATABASE_PATH'] = str(Path(self.temp_dir.name) / 'plugin-test.db')
        sys.modules.pop('plugins.nte_account_db', None)
        self.module = importlib.import_module('plugins.nte_account_db')

    def tearDown(self) -> None:
        if not self.module.nte_account_db.is_closed():
            self.module.nte_account_db.close()
        sys.modules.pop('plugins.nte_account_db', None)
        if self.original_database_path is None:
            os.environ.pop('NTE_DATABASE_PATH', None)
        else:
            os.environ['NTE_DATABASE_PATH'] = self.original_database_path
        self.temp_dir.cleanup()

    def test_publish_shaft_character_is_persistent_and_idempotent(self) -> None:
        self.assertEqual(self.module.publish_shaft_character('伊洛伊'), 'already_published')
        self.assertEqual(self.module.publish_shaft_character('残红'), 'published')
        self.assertEqual(self.module.publish_shaft_character('残红'), 'already_published')
        self.assertEqual(self.module.publish_shaft_character('不存在'), 'not_found')

        publication = self.module.NTEShaftCharacterPublication.get(
            self.module.NTEShaftCharacterPublication.character_name == '残红'
        )
        self.assertTrue(publication.is_published)
