"""Balance import changes every Exe body level, preserving clone attacks and resources."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
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

    def test_wiki_and_shaft_current_values_are_consistent(self):
        p=ROOT/'app/modules/shaft/static/data'
        actions=json.loads((p/'actions.json').read_text())
        self.assertAlmostEqual(next(a for a in actions if a['id']=='action_exe_a1')['multipliers']['atk'],.405972)
        buffs=json.loads((p/'buffs.json').read_text())
        self.assertEqual(next(b for b in buffs if b['id']=='character_exe_overclock')['effects']['all_dmg'],.025)
        role=next(c for c in json.loads((ROOT/'app/modules/guide/data/catalog.json').read_text())['characters']if c['id']=='1074')
        a=next(a for a in role['abilities']if a['id']=='GA_Exe_Melee')
        self.assertEqual(a['param_rows'][0]['values'][0],'40.5972%')
        self.assertEqual(a['param_rows'][0]['hit_details'][0]['attack_values'][0],'40.5972%')
        self.assertIn('2.5%',next(a for a in role['abilities']if a['id']=='GA_Exe_Passive_2')['groups'][0]['description'])
        self.assertIn('5%',role['resonance'][-1]['desc']);self.assertNotIn('10%',role['resonance'][-1]['desc'])
