#!/usr/bin/env python3
"""Import Exe's independently bound growth, damage and cultivation tables."""
import argparse
from pathlib import Path
import json
from copy import deepcopy
from guide_skill_details import table_rows
from guide_skill_cooldowns import CooldownData
from sync_guide_rich_text import character_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'app/modules/guide/data'
CID = '1074'


def assemble(directory, catalog):
    def table(name):
        return table_rows(json.loads((directory / (name + '.json')).read_text(encoding='utf-8')))
    names = {}
    def collect(value):
        if isinstance(value, dict):
            if 'id' in value and 'name' in value:
                names[str(value['id']).casefold()] = value['name']
            for child in value.values(): collect(child)
        elif isinstance(value, list):
            for child in value: collect(child)
    collect(catalog)
    collect(json.loads((DATA / 'character_previews.json').read_text(encoding='utf-8')))
    names.update(gold='甲硬币', fons='方斯')
    def items(pairs):
        return [{'id': key, 'name': names[key.casefold()], 'amount': amount} for key, amount in pairs]
    record = dict(id=CID, name='Exe', element='Chaos', element_id='CHAOS', element_name='暗', rarity=5,
                  stats=[], abilities=[], awaken=[], resonance=[], nouns={}, materials={'breakthrough': [], 'skills': []})
    growth = table('DT_PlayerModifyPackData')
    for prop, name in [('HPMaxBase','生命值'),('AtkBase','攻击力'),('DefBase','防御力')]:
        increments = []
        for level in range(1, 81):
            stage = sum(cap < level for cap in (20,30,40,50,60,70))
            total = 0
            for key in [f'Exe_lv_{level}', *([f'Exe_stage_{stage}'] if stage else [])]:
                row = growth[key]
                if row['ConditionArray']: raise ValueError('Conditional growth')
                for mod in row['ModifyData']:
                    if mod['PropName'] == prop:
                        if not mod['ModifierOp'].endswith('MODIFY_MODOP_ADDITIVE'): raise ValueError('Nonadditive growth')
                        total += mod['PropValue']
            increments.append(int(total))
        record['stats'].append(dict(id_stats=prop, name=name, values=[None]*80, growth_values=increments))
    for prop, name, value in [('CritBase','暴击率',5),('CritDamageBase','暴击伤害',50)]:
        record['stats'].append(dict(id_stats=prop,name=name,values=[value]*80))
    caps = []
    breakthrough = table('DT_CharacterBreakthroughDataTable')
    for stage in range(7):
        row = breakthrough[f'Exe_Breakthrough_{stage}']; caps.append(row['MaxCharacterLevel'])
        if stage:
            pairs = [(p.split(':')[0], int(p.split(':')[1])) for p in (row['NeedItems']+','+row['NeedGolds']).split(',') if p]
            record['materials']['breakthrough'].append(dict(level=str(row['MaxCharacterLevel']),stage=stage,items=items(pairs)))
    types = [('CharacterAbilityList',None),('PassiveAbilityList','passive'),('PeculiarityList','peculiarity'),('CityList','city')]
    active = [('melee','Melee','普通攻击'),('skill','Skill','变轨技能'),('ultraskill','UltraSkill','极轨终结'),('qte','QTE','援护技')]
    for section, fallback in types:
        for binding in table('DT_CharacterAbilityConfig')[CID][section]:
            identifier, raw = binding['Key'], binding['Value']
            index = raw['AbilityIndex']
            typ, type_name, name = active[index-1] if fallback is None else (fallback, {'passive':'Passive','peculiarity':'Peculiarity','city':'City'}[fallback], {'passive':'天赋','peculiarity':'特性','city':'都市技能'}[fallback]+str(index))
            ability = dict(id=identifier,type=typ,type_name=type_name,ability_type=raw['AbilityType'].split('::')[-1],
                           ability_index=index,name=name,phases=[],groups=[],param_rows=[],icon=None)
            group = dict(name=name,description='技能名称与效果说明暂未提供。以下为独立伤害配置，实际命中次数与招式对应暂未提供。',param_rows=[])
            for effect_id, damage in table('DT_SkillDamageData').items():
                if not effect_id.startswith('GE_Player_Exe_'): continue
                if damage['GAName'] != identifier and not (typ=='melee' and damage['GAName']=='GA_Exe_PerfectEvade'): continue
                def values(key): return [f'{round(v*100, 4):g}%' for v in damage[key]]
                hit = dict(effect_id=effect_id,hit_index=1,repeat=None,harmony_gain=damage['HeterochromeAdd'],energy_gain=damage['ChargeAdd'],stagger=damage['UnbalValue'],attack_values=values('AtkRateBaseArray'),defense_values=values('DefRateBaseArray'),hp_values=values('HPRateBaseArray'),form_ability_id=damage['GAName'])
                row = dict(key=effect_id,label=f'独立伤害项 {len(group["param_rows"])+1}',values=hit['attack_values'],hit_details=[hit],hit_detail_status='available')
                group['param_rows'].append(row)
            if not group['param_rows']:
                group['description'] = '技能名称与效果说明暂未提供。'
            ability['param_rows'] = deepcopy(group['param_rows']); ability['groups']=[group]
            if typ in ('skill','ultraskill'): ability['base_cooldown']=CooldownData(directory).resolve(identifier)
            record['abilities'].append(ability)
            if raw['LevelsCostItems']:
                record['materials']['skills'].append(dict(id=identifier,name=name,levels=[dict(level=str(v['Level']+1),require_breakthrough_level=v['RequireTupoLevel'],items=items([(i['ID'],i['Number']) for i in v['CostItems']])) for v in raw['LevelsCostItems']]))
    # Exported Exe descriptions and awakening icons currently point to another character.
    # Keep the real configured slots without borrowing those texts or icons.
    if len(table('1074AwakenEffect')[CID]['AwakenEffectStructList']) != 8:
        raise ValueError('Unexpected awakening slots')
    record['unavailable_sections'] = {'nouns': '机制词条暂未提供。'}
    for index in range(1,7): record['awaken'].append(dict(name=f'觉醒 {index}',desc='名称与效果暂未提供。',icon=None))
    for level in (3,6): record['resonance'].append(dict(name=f'共鸣 {level}',resonance_level=level,desc='名称与效果暂未提供。',icon=None))
    plan=table('DT_EquipmentPlanData')[CID]
    record['equip_slots']=dict(slots=[[(-1 if c=='-1' else 0) for c in r.split(',')] for r in plan['EquipmentSlots']],stats=[],desc='驱动特化与属性加成暂未提供。')
    return record,caps


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--ntedata-dir',type=Path,required=True);args=parser.parse_args()
    def load(name): return json.loads((DATA/name).read_text(encoding='utf-8'))
    catalog=load('catalog.json');record,caps=assemble(args.ntedata_dir,catalog)
    catalog['characters']=[c for c in catalog['characters'] if c['id']!=CID]+[record]
    catalog['cultivation']['level_caps'][CID]=caps
    supplement=load('ntedata_characters.json')
    supplement['characters']=[c for c in supplement['characters'] if c['id']!=CID]+[record];supplement['level_caps'][CID]=caps
    outputs={'catalog.json':catalog,'ntedata_characters.json':supplement}
    for name in ('skill_icons.json','passive_icons.json','character_rich_text.json'):
        payload=load(name)
        if name=='skill_icons.json':payload[CID]={a['id']:None for a in record['abilities']}
        elif name=='passive_icons.json':payload[CID]={s:[dict(name=a['name'],icon=None) for a in record[s]] for s in ('awaken','resonance')}
        else:payload['characters'][CID]=character_data(record)
        outputs[name]=payload
    outputs['character_previews.json']=[c for c in load('character_previews.json') if c['id']!='preview-exe']
    for name,payload in outputs.items():
        target=DATA/name;temp=target.with_suffix('.tmp')
        formatting = {'indent': 2} if name in ('skill_icons.json','passive_icons.json','character_previews.json') else {'separators':(',',':')}
        temp.write_text(json.dumps(payload,ensure_ascii=False,**formatting)+'\n',encoding='utf-8');temp.replace(target)


if __name__=='__main__':main()
