import json
from pathlib import Path
import re
import unittest
from html import unescape

from app.modules.guide.description_parameters import bind_description_parameters
from app.modules.guide.rich_text import description_html

ROOT = Path(__file__).resolve().parents[1]


class DescriptionParametersTest(unittest.TestCase):
    def character(self, identifier):
        data = json.loads((ROOT / 'app/modules/guide/data/catalog.json').read_text())
        return next(c for c in data['characters'] if c['id'] == identifier)

    def test_lacrimosa_awaken_a_uses_nightmare_and_preserves_source_values(self):
        character = self.character('1004')
        row = next(r for r in character['abilities'][0]['param_rows'] if r['key'] == 'GA_Lacrimosa_Dot_num')
        original = list(row['values'])
        bind_description_parameters(character)
        binding = character['awaken'][0]['parameters'][0]
        self.assertEqual(binding['ability_id'], 'GA_Lacrimosa_Melee')
        self.assertEqual(binding['values'][0], '8')
        self.assertEqual(binding['values'][9], '16')
        self.assertEqual(binding['values'][-1], '20.13')
        self.assertEqual(row['values'], original)
        html = str(description_html(character['awaken'][0]['desc'], '1004', [binding]))
        plain = unescape(re.sub('<[^>]*>', '', html))
        self.assertIn('提升至16', plain)
        self.assertIn('1 级倍率：8%', plain)
        self.assertIn('当前技能等级：10 级', plain)
        self.assertNotIn('{0}', plain)
        self.assertIn('data-linked-ability="GA_Lacrimosa_Melee"', html)
        self.assertIn('<strong>噩梦伤害倍率</strong>', html)

    def test_other_awakenings_bind_damage_and_unknown_text_stays_safe(self):
        for identifier, index, first, tenth in [('1010', 2, '50', '100'), ('1072', 3, '100', '199.9')]:
            character = self.character(identifier)
            bind_description_parameters(character)
            binding = character['awaken'][index]['parameters'][0]
            self.assertEqual(binding['values'][0], first)
            self.assertEqual(binding['values'][9], tenth)
            html = str(description_html(character['awaken'][index]['desc'], identifier, [binding]))
            self.assertNotIn('{0}', html)
            self.assertIn('1 级倍率：' + first + '%', html)
            self.assertIn(binding['ability_name'], html)
        html = str(description_html('未知{0}% <script>bad</script>', 'missing'))
        self.assertIn('{0}', html)
        self.assertNotIn('<script>', html)


if __name__ == '__main__':
    unittest.main()
