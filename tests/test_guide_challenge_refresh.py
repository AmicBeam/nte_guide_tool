"""Challenge refresh validates complete seasons before replacing the catalog."""
import copy
import unittest
from scripts.refresh_guide_data import build_challenges


class ChallengeRefreshTest(unittest.TestCase):
    def payloads(self):
        root = 'https://static.nanoka.cc/nte/test/zh/abyss'
        record = {'id': 'Abyss_14', 'name': '霜寒环线', 'stage_count': 1,
                  'begin_date': None, 'end_date': None,
                  'resonance': {'name': '寒光流转', 'description': '获得<lv>20%</>无视防御'},
                  'stages': [{'id': 1, 'halves': []}]}
        return root, {root + '/index.json': {'seasons': [{'id': 'Abyss_14'}]},
                      root + '/Abyss_14.json': record}

    def test_names_and_effects_survive_calendar_and_cleanup(self):
        root, values = self.payloads()
        result = build_challenges('test', lambda url: copy.deepcopy(values[url]))
        self.assertEqual(result[0]['name'], '霜寒环线')
        self.assertEqual(result[0]['begin_date'], '2026-11-13')
        self.assertEqual(result[0]['resonance']['description'], '获得20%无视防御')

    def test_rejects_incomplete_duplicate_and_mismatched_seasons(self):
        for corruption in ['empty', 'duplicate', 'missing_stage', 'wrong_id']:
            with self.subTest(corruption=corruption):
                root, values = self.payloads()
                if corruption == 'empty':
                    values[root + '/index.json']['seasons'] = []
                elif corruption == 'duplicate':
                    values[root + '/index.json']['seasons'] *= 2
                elif corruption == 'missing_stage':
                    values[root + '/Abyss_14.json']['stages'] = []
                else:
                    values[root + '/Abyss_14.json']['id'] = 'Abyss_15'
                with self.assertRaises(ValueError):
                    build_challenges('test', values.__getitem__)
