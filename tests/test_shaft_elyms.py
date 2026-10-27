"""Test-only framework and deliberately incomplete character data."""
import importlib
import json
import subprocess
import unittest
from pathlib import Path

from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import normalize_axis_payload, simulate_shaft_axis
from tests.test_solo_room_flow import RoomFlowTestCase

CID = 'char_elyms'
ROOT = Path(__file__).resolve().parents[1]


class ElymsCatalogTest(unittest.TestCase):
    def test_framework_keeps_missing_data_and_individual_damage_entries(self):
        catalog = load_shaft_catalog()
        characters = catalog['characters']
        character = next(c for c in characters if c['id'] == CID)
        self.assertEqual((character['element'], character['rarity'], character['adaptation']), ('光', 'S', ''))
        self.assertEqual(character['base_stats'], {'atk':None, 'hp':None, 'def':None})
        self.assertEqual([c['element'] for c in characters], sorted(
            [c['element'] for c in characters], key=('光', '灵', '咒', '暗', '魂', '相').index))
        self.assertTrue((ROOT / 'app/modules/guide/static/character-previews/elyms-avatar.webp').is_file())
        actions = [a for a in catalog['actions_by_character'][CID] if a['name'] not in ('无', '闪')]
        self.assertEqual(len(actions), 28)
        self.assertEqual(len({a['id'] for a in actions}), 28)
        for action in actions:
            self.assertTrue(action['duration_pending'])
            self.assertEqual(action['duration_ticks'], 0)
            self.assertFalse(action.get('is_time_stop_zero'))
            self.assertFalse(action['resonance_skill_level_bonus'])
            self.assertEqual(action['hit_count'], 1)
            self.assertEqual((action['multipliers']['atk'], action['energy_gain'], action['harmony'], action['stagger']),
                             (.433, 2.707, 4.511, .536))
            self.assertIn('占位', action['validation_note'])
        self.assertEqual(len(catalog['awakenings']['曳尔沐']), 8)
        self.assertTrue(all(a['implemented'] is False for a in catalog['awakenings']['曳尔沐']))
        self.assertFalse(any(p.get('id') == CID for b in catalog['buffs'] for p in b.get('providers', [])))

    def test_save_normalization_and_engine_do_not_invent_awakening_effects(self):
        axis = {'team': [dict(slot=0, character_id=CID, arc_id='', cartridge_id='',
                             skill_levels=dict(basic=1, skill=1, ultimate=1, support=1))],
                'steps': [dict(id='a', slot=0, action_id='action_elyms_melee1', start_tick=0)]}
        self.assertEqual(normalize_axis_payload(axis)['team'][0]['character_id'], CID)
        details = []
        for nodes in ([], [1, 2, 3, 4, 5, 6]):
            axis['team'][0]['awakening_nodes'] = nodes
            detail = simulate_shaft_axis(axis)['result']['details'][0]
            self.assertTrue(any('占位' in w for w in detail['warnings']))
            details.append(detail)
        self.assertEqual(details[0]['direct_damage'], details[1]['direct_damage'])
        catalog = load_shaft_catalog()
        script = "const s=require(process.argv[1]);const p=JSON.parse(require('fs').readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(s.inspectAxis(p.axis,p.catalog)));"
        result = subprocess.run(['node', '-e', script, str(ROOT/'app/modules/shaft/static/js/shaft_self_check.js')],
                                input=json.dumps(dict(axis=axis, catalog=catalog)), text=True, capture_output=True, check=True)
        warnings = json.loads(result.stdout)
        self.assertTrue(any('基础框架' in w for w in warnings))
        self.assertTrue(any('时长未实装' in w for w in warnings))


class ElymsAccessTest(RoomFlowTestCase):
    def test_test_only_access_save_publish_and_market(self):
        models = importlib.import_module('app.models')
        regular = self._issue_login_and_get_token('elyms-regular')
        invited = self._issue_login_and_get_token('elyms-invited')
        tester = self._issue_login_and_get_token('elyms-tester')
        models.Player.update(shaft_invited=True).where(models.Player.player_uid == 'elyms-invited').execute()
        models.Player.update(shaft_test_whitelisted=True).where(models.Player.player_uid == 'elyms-tester').execute()
        for token in (None, regular, invited, tester):
            catalog = self._get('/api/shaft/catalog', token=token)
            character = next(c for c in catalog['characters'] if c['id'] == CID)
            self.assertEqual(character['selection_disabled'], token != tester)
        axis = {'team': [dict(slot=0, character_id=CID, arc_id='', cartridge_id='')],
                'steps': [dict(id='a', slot=0, action_id='action_elyms_melee1', start_tick=0)]}
        payload = dict(title='曳尔沐基础框架', axis=axis, result=self._shaft_client_result(axis))
        for token in (regular, invited):
            response = self.client.post('/api/shaft/axes', json=payload, headers={'Authorization': f'Bearer {token}'})
            self.assertEqual(response.status_code, 400)
        saved = self._post('/api/shaft/axes', payload, token=tester)
        public = self._post(f"/api/shaft/axes/{saved['id']}/publish", {}, token=tester)
        self.assertEqual(self._get(f"/api/shaft/axes/{saved['id']}", token=tester)['axis']['team'][0]['character_id'], CID)
        for token in (None, regular, invited, tester):
            ids = [a['id'] for a in self._get('/api/shaft/market', token=token)['items']]
            self.assertEqual(public['id'] in ids, token == tester)
            response = self.client.get(f"/api/shaft/axes/{public['id']}", headers={'Authorization': f'Bearer {token}'} if token else {})
            self.assertEqual(response.status_code, 200 if token == tester else 400)
