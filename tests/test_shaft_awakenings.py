import unittest

from app.modules.shaft.domain.catalog import load_shaft_catalog


class ShaftAwakeningDataTestCase(unittest.TestCase):
    def test_every_catalog_character_has_eight_awakening_descriptions(self) -> None:
        catalog = load_shaft_catalog()
        character_names = {character['name'] for character in catalog['characters']}

        self.assertEqual(character_names, set(catalog['awakenings']))
        self.assertTrue(all(len(entries) == 8 for entries in catalog['awakenings'].values()))

    def test_pending_and_out_of_scope_awakenings_are_counted_separately(self) -> None:
        awakenings = load_shaft_catalog()['awakenings']
        entries = [
            entry
            for character_entries in awakenings.values()
            for entry in character_entries
        ]
        pending = [
            entry
            for entry in entries
            if entry.get('implemented') is False
            and entry.get('implementation_status') != 'out_of_scope'
        ]
        out_of_scope = [
            entry
            for entry in entries
            if entry.get('implementation_status') == 'out_of_scope'
        ]

        self.assertEqual(len(entries), 176)
        self.assertEqual(len(pending), 3)
        self.assertEqual(len(out_of_scope), 28)
        self.assertNotIn(
            'implemented',
            next(entry for entry in awakenings['残虹'] if entry['title'] == '猩红盛宴'),
        )
        self.assertNotIn(
            'implemented',
            next(entry for entry in awakenings['残虹'] if entry['title'] == '鸩火灼心'),
        )
        self.assertEqual(
            [entry['title'] for entry in awakenings['残虹']],
            ['瞳中深渊', '渊底之吻', '吻痕窥梦', '梦魇生花', '花开见血', '血染双瞳', '猩红盛宴', '鸩火灼心'],
        )
        self.assertNotIn(
            'implemented',
            next(entry for entry in awakenings['九原'] if entry['title'] == '优势初成'),
        )
        self.assertNotIn(
            'implemented',
            next(entry for entry in awakenings['九原'] if entry['title'] == '知晓每一条秘密'),
        )

if __name__ == '__main__':
    unittest.main()
