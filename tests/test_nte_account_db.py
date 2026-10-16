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
        self.assertEqual(self.module.publish_shaft_character('残虹'), 'already_published')
        self.assertEqual(self.module.publish_shaft_character('黑羽'), 'already_published')
        self.assertEqual(self.module.publish_shaft_character('灵可'), 'already_published')
        self.assertEqual(self.module.publish_shaft_character('灵可'), 'already_published')
        self.assertEqual(self.module.publish_shaft_character('不存在'), 'not_found')

        publication = self.module.NTEShaftCharacterPublication.get(
            self.module.NTEShaftCharacterPublication.character_name == '灵可'
        )
        self.assertTrue(publication.is_published)

    def test_ensure_tables_synchronizes_character_name_by_stable_id(self) -> None:
        self.module.ensure_nte_tables()
        publication = self.module.NTEShaftCharacterPublication.get(
            self.module.NTEShaftCharacterPublication.character_id == 'char_076a1f4e53'
        )
        publication.character_name = '旧称'
        publication.save()

        self.module.ensure_nte_tables()

        publication = self.module.NTEShaftCharacterPublication.get_by_id(publication.id)
        self.assertEqual(publication.character_name, '残虹')

    def test_released_characters_migrate_existing_test_records(self) -> None:
        self.module.ensure_nte_tables()
        model = self.module.NTEShaftCharacterPublication
        for name in ('灵可', '黑羽'):
            with self.subTest(character=name):
                model.update(is_published=False, access_level='test').where(
                    model.character_name == name
                ).execute()
                self.module.ensure_nte_tables()
                publication = model.get(model.character_name == name)
                self.assertTrue(publication.is_published)
                self.assertEqual(publication.access_level, 'public')


    def test_registration_matches_website_nonnull_permission_columns(self) -> None:
        db = self.module.nte_account_db
        db.connect(reuse_if_open=True)
        db.execute_sql('CREATE TABLE player (id INTEGER PRIMARY KEY, player_uid TEXT UNIQUE, nickname TEXT NOT NULL, '
                       'shaft_invited INTEGER NOT NULL, shaft_test_whitelisted INTEGER NOT NULL, '
                       'created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)')
        code, created = self.module.issue_account_register_code('10001', '验证用户')
        self.assertTrue(created)
        self.assertEqual(len(code), 6)
        player = self.module.NTEPlayer.get(self.module.NTEPlayer.player_uid == '10001')
        self.assertFalse(player.shaft_invited)
        self.assertFalse(player.shaft_test_whitelisted)
        player.shaft_invited = True
        player.shaft_test_whitelisted = True
        player.save()
        new_code, created = self.module.issue_account_register_code('10001')
        self.assertFalse(created)
        self.assertEqual(len(new_code), 6)
        player = self.module.NTEPlayer.get_by_id(player.id)
        self.assertTrue(player.shaft_invited)
        self.assertTrue(player.shaft_test_whitelisted)

    def test_legacy_plugin_player_table_migrates_without_losing_user(self) -> None:
        db = self.module.nte_account_db
        db.connect(reuse_if_open=True)
        db.execute_sql('CREATE TABLE player (id INTEGER PRIMARY KEY, player_uid TEXT UNIQUE, nickname TEXT NOT NULL, '
                       'created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)')
        db.execute_sql("INSERT INTO player VALUES(1, '10001', '原用户', '2026-01-01', '2026-01-01')")
        code, created = self.module.issue_account_register_code('10001')
        self.assertFalse(created)
        self.assertEqual(len(code), 6)
        player = self.module.NTEPlayer.get_by_id(1)
        self.assertEqual(player.nickname, '原用户')
        self.assertFalse(player.shaft_invited)
        self.assertFalse(player.shaft_test_whitelisted)
