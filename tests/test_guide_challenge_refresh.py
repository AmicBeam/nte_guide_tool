"""Challenge refresh validates complete seasons before replacing the catalog."""
import copy
import unittest
from scripts.refresh_guide_data import build_challenges, enrich_challenge_resonances, enrich_challenge_monsters


class ChallengeRefreshTest(unittest.TestCase):
    def test_monster_stats_follow_explicit_pool_and_attribute_binding(self):
        monster = {'class_id': 'Boss_BP', 'level': 90, 'count': 1,
                   'stats': {'hp': 4605680, 'resistances': {'cosmos': 0.3}}}
        records = [{'id': 'season', 'stages': [{'id': 12, 'halves': [
            {'id': 'FirstHalf', 'waves': [{'wave': 1, 'monsters': [monster]}]}]}]}]
        levels = {'season': {'LevelConfigArray': [{'LevelID': 12, 'SpawnMonsterConfigMap': [
            {'Key': 'EAbyssFightStage::FirstHalf', 'Value': {'CloneSpawnMonsterConfigArray': [
                {'Wave': 1, 'MonsterPoolID': 'explicit_pool'}]}}]}]}}
        pools = {'explicit_pool': {'MonsterPoolArray': [{'MonsterClass': {'AssetPathName': '/Game/Boss.Boss_BP_C'},
            'MonsterCount': 1, 'MonsterLevel': 90, 'AttributeID': 'explicit_pack'}]}}
        pack = {'HPMaxBase': 4930440, 'AtkBase': 2726, 'DefBase': 1080, 'UnbalMax': 50,
                'DamageResistCosmosBase': 0.1}
        pack.update({prefix + suffix: 0 for prefix in ('HPMax', 'Atk', 'Def') for suffix in ('Up', 'Add')})
        enrich_challenge_monsters(records, levels, pools, {'explicit_pack': pack})
        self.assertEqual(monster['stats']['hp'], 4930440)
        self.assertEqual(monster['stats']['resistances']['cosmos'], 0.1)
        pack['HPMaxUp'] = 0.1
        with self.assertRaises(ValueError):
            enrich_challenge_monsters(records, levels, pools, {'explicit_pack': pack})

    def test_resonance_uses_season_binding_and_keeps_unknown_dates(self):
        records = [{'id': 'Abyss_15', 'begin_date': None},
                   {'id': 'Abyss_16', 'begin_date': None}]
        seasons = {row['id']: {'BuffID': 'shared'} for row in records}
        buffs = {'shared': {'BuffName': {'LocalizedString': '众生回唱'},
                            'BuffDesc': {'LocalizedString': '暗伤提高<lv>30%</>；召唤物伤害提高<lv>15%</>。'}}}
        enrich_challenge_resonances(records, seasons, buffs)
        for record in records:
            self.assertIsNone(record['begin_date'])
            self.assertEqual(record['resonance'], {'id': 'shared', 'name': '众生回唱',
                'description': '暗伤提高30%；召唤物伤害提高15%。'})

    def test_missing_buff_does_not_partially_replace_resonances(self):
        records = [{'id': 'first', 'resonance': {'name': '旧效果'}}, {'id': 'second'}]
        before = copy.deepcopy(records)
        seasons = {'first': {'BuffID': 'known'}, 'second': {'BuffID': 'missing'}}
        buffs = {'known': {'BuffName': {'LocalizedString': '效果'},
                           'BuffDesc': {'LocalizedString': '完整说明'}}}
        with self.assertRaises(KeyError):
            enrich_challenge_resonances(records, seasons, buffs)
        self.assertEqual(records, before)

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
