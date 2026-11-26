"""Balance import changes every Exe body level, preserving clone attacks and resources."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('exe_adjustments',ROOT/'scripts/exe_adjustments.py')
adjustments=importlib.util.module_from_spec(spec);spec.loader.exec_module(adjustments)


class ExeAdjustmentTest(unittest.TestCase):
    def test_all_levels_clone_exclusions_resources_text_and_idempotence(self):
        body=dict(AtkRateBaseArray=[.537,1.073],DefRateBaseArray=[],HPRateBaseArray=[],ChargeAdd=2.093)
        raw={'DT_SkillDamageData':{'GE_Player_Exe_Melee1_Damage':body,
                'GE_Player_Exe_Skill_Damage':{**body,'AtkRateBaseArray':[.483,.966]},
                'GE_Player_Exe_QTE_Damage':{**body,'AtkRateBaseArray':[.308,.333]},
                'GE_Player_Exe_QTE1_Damage':{**body,'AtkRateBaseArray':[2.464,2.661]},
                'GE_Player_Exe_CloneAtk_Damage':{**body,'AtkRateBaseArray':[.924,1.847]},
                'GE_Player_Exe_CloneBoom_Damage':{**body,'AtkRateBaseArray':[.154,.308]},
                'other':deepcopy(body)},
             'DT_GameplayAbilityTipsData':{'GA_Exe_Passive_2':{'Description':'每层10%增伤，最多4层；当前冷却时间低于2秒时不再触发'},'other':{'Description':'10%'}},
             'DT_ExeEffectFigure':{'ExeDmgUpPas2':{'Keys':[{'Value':.1}]},'other':{'Keys':[{'Value':.1}]}},
             '1074AwakenEffect':{'1074':{'AwakenEffectStructList':[{'Desc':{'LocalizedString':'评分获取50%；斩杀10%的BOSS'}}]}}}
        original=deepcopy(raw);result=adjustments.adjust_tables(raw)
        self.assertEqual(raw,original)
        self.assertEqual(result['DT_SkillDamageData']['GE_Player_Exe_Melee1_Damage']['AtkRateBaseArray'],[.405972,.811188])
        self.assertEqual(result['DT_SkillDamageData']['GE_Player_Exe_Skill_Damage']['AtkRateBaseArray'],[.365148,.730296])
        for key in [*adjustments.CLONE_EFFECTS,'other']:self.assertEqual(result['DT_SkillDamageData'][key],raw['DT_SkillDamageData'][key])
        self.assertEqual(result['DT_SkillDamageData']['GE_Player_Exe_Melee1_Damage']['ChargeAdd'],2.093)
        self.assertIn('2.5%',result['DT_GameplayAbilityTipsData']['GA_Exe_Passive_2']['Description'])
        self.assertEqual(result['DT_ExeEffectFigure']['ExeDmgUpPas2']['Keys'][0]['Value'],.025)
        self.assertEqual(result['1074AwakenEffect']['1074']['AwakenEffectStructList'][-1]['Desc']['LocalizedString'],'评分获取50%；斩杀5%的BOSS\nExe抗打断能力提高50%。')
        self.assertEqual(result['DT_SkillDamageData']['GE_Player_Exe_QTE_Damage']['AtkRateBaseArray'][0],.2)
        self.assertEqual(result['DT_SkillDamageData']['GE_Player_Exe_QTE1_Damage']['AtkRateBaseArray'][0],1.6)
        for key in adjustments.SUPPORT_EFFECTS:
            self.assertAlmostEqual(result['DT_SkillDamageData'][key]['AtkRateBaseArray'][1]/raw['DT_SkillDamageData'][key]['AtkRateBaseArray'][1],2/3.08,places=8)
        previous=deepcopy(raw)
        for key,row in previous['DT_SkillDamageData'].items():
            if key.startswith('GE_Player_Exe_')and key not in adjustments.CLONE_EFFECTS:
                row['AtkRateBaseArray']=[round(v*.9,9)for v in row['AtkRateBaseArray']]
        self.assertEqual(adjustments.adjust_tables(previous),result)
        self.assertEqual(adjustments.adjust_tables(result),result)

    def test_current_game_multipliers_and_descriptions_pass_through(self):
        raw = {
            'DT_SkillDamageData': {
                'GE_Player_Exe_Melee1_Damage': {'AtkRateBaseArray': [.406, .438]},
                'GE_Player_Exe_CloneAtk_Damage': {'AtkRateBaseArray': [.776, .838]},
            },
            'DT_GameplayAbilityTipsData': {'GA_Exe_Passive_2': {'Description': '当前冷却时间低于2秒时不再触发'}},
        }
        self.assertEqual(adjustments.adjust_tables(raw), raw)

    def test_wiki_and_shaft_current_values_are_consistent(self):
        p=ROOT/'app/modules/shaft/static/data'
        actions=json.loads((p/'actions.json').read_text())
        self.assertAlmostEqual(next(a for a in actions if a['id']=='action_exe_a1')['multipliers']['atk'],.406)
        buffs=json.loads((p/'buffs.json').read_text())
        self.assertEqual(next(b for b in buffs if b['id']=='character_exe_overclock')['effects']['all_dmg'],.025)
        role=next(c for c in json.loads((ROOT/'app/modules/guide/data/catalog.json').read_text())['characters']if c['id']=='1074')
        a=next(a for a in role['abilities']if a['id']=='GA_Exe_Melee')
        self.assertEqual(a['param_rows'][0]['values'][0],'40.6%')
        self.assertEqual(a['param_rows'][0]['hit_details'][0]['attack_values'][0],'40.6%')
        self.assertIn('2.5%',next(a for a in role['abilities']if a['id']=='GA_Exe_Passive_2')['groups'][0]['description'])
        self.assertIn('5%',role['resonance'][-1]['desc']);self.assertNotIn('10%',role['resonance'][-1]['desc'])

    def test_all_wiki_damage_levels_match_shaft_in_both_snapshots(self):
        from scripts import refresh_guide_data
        from scripts.sync_guide_exe import DATA, SHAFT_DATA, shaft_attack_curves
        from scripts.guide_skill_details import percentage

        curves = shaft_attack_curves(SHAFT_DATA)
        supplemental = json.loads((DATA / 'ntedata_characters.json').read_text())
        catalog = json.loads((DATA / 'catalog.json').read_text())
        for records in (catalog, supplemental):
            role = next(character for character in records['characters'] if character['id'] == '1074')
            covered = set()
            for ability in role['abilities']:
                for row in ability['param_rows']:
                    hits = row['hit_details']
                    if not hits:
                        continue
                    for hit in hits:
                        effect = hit['effect_id']
                        covered.add(effect)
                        self.assertEqual(hit['attack_values'], [percentage(level) for level in curves[effect]])
                    expected = [[(hit['attack_values'][level], hit['repeat']) for hit in hits]
                                for level in range(13)]
                    shown = [[(number, int(count or 1)) for number, count in
                              re.findall(r'(\d+(?:\.\d+)?%)(?:\*(\d+))?', value)]
                             for value in row['values']]
                    self.assertEqual(shown, expected, row['label'])

            shock = 'GE_Player_Exe_CloneBoom_Damage'
            awakening = next(item for item in role['awaken'] if item.get('parameters'))
            self.assertEqual(awakening['parameters'][0]['values'],
                             [percentage(level).removesuffix('%') for level in curves[shock]])
            self.assertEqual(covered | {shock}, set(curves))

        stale = {'characters': [{'id': '1074', 'abilities': []}], 'cultivation': {'level_caps': {}}}
        refresh_guide_data.merge_local_characters(stale, supplemental)
        self.assertEqual([character for character in stale['characters'] if character['id'] == '1074'], [role])
