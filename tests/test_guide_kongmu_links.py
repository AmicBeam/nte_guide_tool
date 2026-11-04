from pathlib import Path
from types import SimpleNamespace
import unittest
from urllib.parse import parse_qs, urlsplit

from flask import Flask
from app.modules.guide import blueprint, presentation, service
from app.modules.kongmu.service import get_kongmu_catalog_payload, get_character_detail


class GuideKongmuLinksTest(unittest.TestCase):
    def test_all_character_links_select_existing_character_and_cartridge(self):
        catalog = get_kongmu_catalog_payload(player=SimpleNamespace(shaft_test_whitelisted=True))
        character_ids = {str(identifier) for c in catalog['characters']
                         for identifier in [c['id'], *c['merged_ids']]}
        cartridge_ids = {c['id'] for c in catalog['cartridges']}
        for character in service.catalog()['characters']:
            entry = presentation.kongmu_entry(character)
            params = parse_qs(urlsplit(entry['url']).query)
            self.assertEqual(params['character'], [str(character['id'])])
            self.assertIn(params['character'][0], character_ids)
            self.assertIn(params['cartridge'][0], cartridge_ids)
        self.assertEqual(presentation.kongmu_entry({'id': '1042'})['url'],
                         '/kongmu?character=1042&cartridge=psyche')
        self.assertEqual(presentation.kongmu_entry({'id': '1010'})['url'],
                         '/kongmu?character=1010&cartridge=nature')

    def test_legacy_ids_resolve_to_canonical_ids(self):
        for legacy, current in [('char_heiyu', '1042'), ('char_akane', '1057'),
                                ('char_076a1f4e53', '1036')]:
            self.assertEqual(get_character_detail(legacy), get_character_detail(current))
            self.assertEqual(get_character_detail(current)[0]['id'], current)

    def test_link_is_in_tools_and_not_the_kongmu_section(self):
        app = Flask(__name__, template_folder=str(Path(__file__).resolve().parents[1] / 'app/templates'))
        app.register_blueprint(blueprint)
        with app.test_client() as client:
            html = client.get('/guide/characters/1042').text
        tools = html.split('class="guide-character-tools"', 1)[1].split('</nav>', 1)[0]
        self.assertIn('/kongmu?character=1042&amp;cartridge=psyche', tools)
        section = html.split('id="section-equip_slots"', 1)[1].split('</section>', 1)[0]
        self.assertNotIn('href="/kongmu', section)
        self.assertIn('guide-equip-description', section)
