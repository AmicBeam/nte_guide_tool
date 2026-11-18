#!/usr/bin/env python3
"""Import Exe's complete Wiki record from explicit local data and image tables."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from PIL import Image

from guide_skill_details import table_rows, display_text
from refresh_guide_data import clean
from sync_guide_preview_details import build as build_skills
from sync_guide_rich_text import character_data
from sync_guide_ntedata_arcs import assemble as build_arc
from sync_guide_affinity import TABLES as AFFINITY_TABLES, build_affinity

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'app/modules/guide/data'
CID='1074'
ARC='fork_exe'
RELEASE='2026-11-26T12:00:00+08:00'


def assemble(directory):
    def table(name):return table_rows(json.loads((directory/(name+'.json')).read_text()))
    raw=table('DT_Character')[CID]
    element=raw['ElementData']
    base=table('DT_PlayerPackData')[element['PropModifyID']]
    growth=table('DT_PlayerModifyPackData')
    breakthroughs=table('DT_CharacterBreakthroughDataTable')
    stages=[breakthroughs[f"{element['BreakthroughPackId']}_{i}"] for i in range(7)]
    caps=[row['MaxCharacterLevel']for row in stages]
    if caps!=sorted(set(caps)) or caps[-1]!=80:raise ValueError('Invalid character level caps')
    details=build_skills(directory,CID,'Exe')
    record=dict(id=CID,name=clean(display_text(raw['ItemName'])),aliases=['EXE'],element='Chaos',element_id='CHAOS',element_name='暗',rarity=5,
        hp=base['HPMaxBase'],atk=base['AtkBase'],**{'def':base['DefBase']},**details)
    record['stats']=[]
    for prop,name in [('HPMaxBase','生命值'),('AtkBase','攻击力'),('DefBase','防御力')]:
        values=[]
        for level in range(1,81):
            stage=sum(cap<level for cap in caps[:-1]);value=base[prop]
            keys=[f"{element['UpgradeModifyPackId']}_{level}"]
            if stage:keys.append(stages[stage]['ModifyPackID'])
            for key in keys:
                row=growth[key]
                if row['ConditionArray']:raise ValueError('Conditional character growth')
                for modifier in row['ModifyData']:
                    if modifier['PropName']!=prop:continue
                    if not modifier['ModifierOp'].endswith('MODIFY_MODOP_ADDITIVE'):raise ValueError('Nonadditive character growth')
                    value+=modifier['PropValue']
            values.append(int(value))
        record['stats'].append(dict(id_stats=prop,name=name,values=values))
    for prop,name in [('CritBase','暴击率'),('CritDamageBase','暴击伤害')]:
        record['stats'].append(dict(id_stats=prop,name=name,values=[base[prop]*100]*80))
    item_names={key.casefold():clean(display_text(value['ItemName']))for key,value in table('DT_ItemConfig').items()}
    item_names.update(gold='甲硬币',fons='方斯')
    def items(pairs):
        return [dict(id=key,name=item_names[key.casefold()],amount=amount)for key,amount in pairs]
    record['materials']=dict(breakthrough=[],skills=[])
    for stage,row in enumerate(stages[1:],1):
        pairs=[(p.split(':')[0],int(p.split(':')[1]))for p in (row['NeedItems']+','+row['NeedGolds']).split(',')if p]
        record['materials']['breakthrough'].append(dict(level=str(row['MaxCharacterLevel']),stage=stage,items=items(pairs)))
    config=table('DT_CharacterAbilityConfig')[CID]
    by_id={a['id']:a for a in record['abilities']}
    categories={'GA_Exe_Melee':'melee','GA_Exe_Skill':'skill','GA_Exe_UltraSkill':'ultraskill','GA_Exe_QTE':'qte'}
    for section in ('CharacterAbilityList','PassiveAbilityList','PeculiarityList','CityList'):
        for binding in config[section]:
            identifier,binding=binding['Key'],binding['Value']
            if identifier not in by_id:
                if section!='CityList':raise ValueError('Missing skill: '+identifier)
                name='都市技能'+('一'if binding['AbilityIndex']==1 else'二')
                ability=dict(id=identifier,name=name,phases=[],param_rows=[],icon=None,groups=[dict(name=name,description='都市技能效果暂未提供。',param_rows=[])])
                record['abilities'].append(ability);by_id[identifier]=ability
            ability=by_id[identifier]
            category='city'if section=='CityList'else'passive'if section=='PassiveAbilityList'else'peculiarity'if section=='PeculiarityList'else categories[identifier]
            ability.update(type=category,type_name={'melee':'Melee','skill':'Skill','ultraskill':'UltraSkill','qte':'QTE','passive':'Passive','peculiarity':'Peculiarity','city':'City'}[category],
                ability_type=binding['AbilityType'].split('::')[-1],ability_index=binding['AbilityIndex'])
            if binding['LevelsCostItems']:
                record['materials']['skills'].append(dict(id=identifier,name=ability['name'],levels=[dict(level=str(row['Level']+1),require_breakthrough_level=row['RequireTupoLevel'],items=items([(i['ID'],i['Number'])for i in row['CostItems']]))for row in binding['LevelsCostItems']]))
    by_id['GA_Exe_Skill']['base_cooldown']=15
    by_id['GA_Exe_UltraSkill']['base_cooldown']=20
    record['nouns']={key.casefold():dict(id=key,title=clean(display_text(row['Title'])),desc=clean(display_text(row['Explanation'])))for key,row in table('DT_NounsExplanation').items()if key.startswith('Exe')}
    slots=table('DT_CharacterEquipmentSlotsData')[element['EquipmentSlotID']]
    record['equip_slots']=dict(owner_grid_count=slots['OwnerGridCount'],desc='每装配1个Ⅲ型驱动，暴击率增加8%。',special_desc='Ⅲ型驱动特化',
        stats=[dict(id_stats='CritBase',name='暴击率',value=8,b_show_percent=True)],
        type_name='「卡带」属性',slots=[[int(value)for value in row.split(',')]for row in slots['EquipmentSlots']])
    if sum(cell>=0 for row in record['equip_slots']['slots']for cell in row)!=20:raise ValueError('Invalid equipment slots')
    rich=deepcopy(record);tips=table('DT_GameplayAbilityTipsData')
    for a in rich['abilities']:
        if a['id']in tips:
            a['phases']=[dict(desc=row['Description'].get('LocalizedString',''))for row in tips[a['id']]['AbilityDescription']if row['AbilityDesType'].endswith('ADT_DES')];a.pop('groups',None)
    awake=table('1074AwakenEffect')[CID]['AwakenEffectStructList']
    for section,originals in [('awaken',awake[:6]),('resonance',awake[6:])]:
        for target,source in zip(rich[section],originals):target['desc']=source['Desc']['LocalizedString']
    for noun in rich['nouns'].values():noun['desc']=table('DT_NounsExplanation')[noun['id']]['Explanation']['LocalizedString']
    return record,caps,character_data(rich)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ntedata-dir',type=Path,required=True);p.add_argument('--ui-dir',type=Path,required=True);p.add_argument('--affinity-dir',type=Path,required=True)
    args=p.parse_args();directory=args.ntedata_dir
    # The skill assembler binds only icons already available locally.
    tips=table_rows(json.loads((directory/'DT_GameplayAbilityTipsData.json').read_text()))
    for key,raw in tips.items():
        if not key.startswith('GA_Exe_'):continue
        source=args.ui_dir/(raw['icon']['AssetPathName'].split('.')[0].removeprefix('/Game/')+'.png')
        target=ROOT/'app/modules/guide/static/skill-icons'/(source.stem+'.webp')
        if not source.is_file():raise ValueError('Missing bound skill icon')
        with Image.open(source)as image:image.convert('RGBA').save(target,format='WEBP',quality=90,method=6)
    for i in range(1,7):
        source=args.ui_dir/('UI/UI/Player/YH_UI_juexingICON/1074_'+str(i)+'.png')
        target=ROOT/'app/modules/guide/static/passive-icons'/('1074_'+str(i)+'.webp')
        with Image.open(source)as image:image.convert('RGBA').save(target,format='WEBP',quality=90,method=6)
    load=lambda name:json.loads((DATA/name).read_text())
    outputs={}
    record,caps,rich=assemble(directory)
    catalog=load('catalog.json');catalog['characters']=[c for c in catalog['characters']if c['id']!=CID]+[record];catalog['cultivation']['level_caps'][CID]=caps
    arc=build_arc(directory,catalog,ARC);catalog['arcs']=[a for a in catalog['arcs']if a['id']!=ARC]+[arc]
    outputs['catalog.json']=catalog
    supplemental=load('ntedata_characters.json');supplemental['characters']=[c for c in supplemental['characters']if c['id']!=CID]+[record];supplemental['level_caps'][CID]=caps;outputs['ntedata_characters.json']=supplemental
    supplemental=load('ntedata_arcs.json');supplemental['arcs']=[a for a in supplemental['arcs']if a['id']!=ARC]+[arc];outputs['ntedata_arcs.json']=supplemental
    outputs['character_previews.json']=[c for c in load('character_previews.json')if c['id']not in(CID,'preview-exe')]
    payload=load('character_preview_details.json');payload.pop('preview-exe',None);payload.pop(CID,None);outputs['character_preview_details.json']=payload
    for name in ['skill_icons.json','passive_icons.json','character_rich_text.json']:
        payload=load(name)
        if name=='skill_icons.json':payload[CID]={a['id']:a.get('icon')for a in record['abilities']}
        elif name=='passive_icons.json':payload[CID]={s:[dict(name=a['name'],icon=a.get('icon'))for a in record[s]]for s in ['awaken','resonance']}
        else:payload['characters'][CID]=rich
        outputs[name]=payload
    metadata=load('character_metadata.json');raw=table_rows(json.loads((directory/'DT_Character.json').read_text()))[CID]
    existing_arc_type=next(c['arc_type']for c in metadata['characters']if c['arc_type']['id']=='3')
    tags=[]
    for tag in raw['PlayerViewTagArray']:
        basename=tag['TagIcon']['AssetPathName'].split('.')[0].rsplit('/',1)[-1]
        tags.append(dict(name=clean(display_text(tag['TagText'])),description=clean(display_text(tag['TagDesc'])),icon='/static/images/character-tags/'+basename+'.webp'))
    metadata['characters']=[c for c in metadata['characters']if c['id']!=CID]+[dict(id=CID,arc_type=existing_arc_type,char_tags=tags)];outputs['character_metadata.json']=metadata
    tables={key:table_rows(json.loads((args.affinity_dir/Path(path).name).read_text()))for key,path in AFFINITY_TABLES.items()}
    own_affinity=build_affinity([CID],tables)['characters'][0];payload=load('affinity.json');payload['characters']=[r for r in payload['characters']if r['id']!=CID]+[own_affinity];outputs['affinity.json']=payload
    # Preserve all existing local recommendations; Exe's local default is his bound signature arc.
    arc_settings=load('arc_cultivation.json');arc_raw={k.casefold():v for k,v in table_rows(json.loads((directory/'DT_ForkItemData.json').read_text())).items()}['fork_exe']['ElementData']
    upgrades={k.casefold():v for k,v in table_rows(json.loads((directory/'DT_ForkUpgradeData.json').read_text())).items()}
    breaks={k.casefold():v for k,v in table_rows(json.loads((directory/'DT_ForkBreakthroughData.json').read_text())).items()}
    pack=arc_raw['UpgradePackId'].casefold();arc_settings['experience'][pack]=[upgrades[f'{pack}_{level}']['NeedExp']for level in range(1,81)]
    arc_settings['arcs'][ARC]=dict(level_caps=[breaks[f"{arc_raw['BreakthroughPackId']}_{i}".casefold()]['MaxForkLevel']for i in range(7)],experience_pack=pack)
    arc_settings['recommendations'][CID]=dict(id=ARC,name=arc['name']);outputs['arc_cultivation.json']=arc_settings
    releases=load('character_releases.json')
    for row in releases['characters']:
        if row['name']=='Exe':row.update(ids=[CID,'char_exe','preview-exe',ARC],release_at=RELEASE)
    outputs['character_releases.json']=releases
    art_path=ROOT/'app/static/images/characters/index.json';art=json.loads(art_path.read_text())
    art['characters']=[c for c in art['characters']if c['id']!=CID]+[dict(id=CID,name='Exe',avatar='/static/images/characters/avatar/Exe-square.webp',portrait='/static/images/characters/portrait/Exe.webp',fullbody='/static/images/characters/fullbody/Exe.webp')]
    # Images are copied from explicit bound resources; never use a small card as a portrait.
    image_jobs=[('UI/UI/Player/TAG_Icon/YH_UI_JueSe_Icon_ZhaoHuan.png',ROOT/'app/static/images/character-tags/YH_UI_JueSe_Icon_ZhaoHuan.webp'),('UI/UI_Icon/Appearance/1074/Fashion_1074_0/2560_Fashion_1074_0.png',ROOT/'app/static/images/characters/fullbody/Exe.webp'),('UI/UI_Icon/Fork/fork_exe_256.png',ROOT/'app/static/images/arcs/fork_exe.webp')]
    for a in record['abilities']:
        if a.get('icon'):
            basename=Path(a['icon']).stem;image_jobs.append(('UI/UI_Icon/Skill/'+basename+'.png',ROOT/'app/modules/guide/static/skill-icons'/str(basename+'.webp')))
    for i in range(1,7):image_jobs.append(('UI/UI/Player/YH_UI_juexingICON/1074_'+str(i)+'.png',ROOT/'app/modules/guide/static/passive-icons'/('1074_'+str(i)+'.webp')))
    for source,target in image_jobs:
        source=args.ui_dir/source
        if not source.is_file():raise ValueError('Missing bound image: '+str(source))
        target.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(source)as image:image.convert('RGBA').save(target,format='WEBP',quality=90,method=6)
    for name,payload in outputs.items():
        path=DATA/name;pretty=name not in ['catalog.json','ntedata_characters.json','ntedata_arcs.json','character_rich_text.json','arc_cultivation.json']
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(payload,ensure_ascii=False,**({'indent':2}if pretty else{'separators':(',',':')}))+'\n');tmp.replace(path)
    art_path.write_text(json.dumps(art,ensure_ascii=False,indent=2)+'\n')
    print('Imported Exe Wiki record and signature arc')


if __name__=='__main__':main()
