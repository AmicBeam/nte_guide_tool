#!/usr/bin/env python3
"""Import Exe's supplied damage workbook and local configuration tables."""
import argparse
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile
from exe_adjustments import adjust_tables

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'app/modules/shaft/static/data'
CID = 'char_exe'
ARC = 'arc_exe_dotted_horizon'


def table(directory, name):
    paths = list(directory.rglob(name + '.json'))
    if len(paths) != 1: raise ValueError('Missing or ambiguous table: ' + name)
    return next(row['Rows'] for row in json.loads(paths[0].read_text()) if 'Rows' in row)


def workbook_rows(path):
    ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as book:
        strings = [''.join(row.itertext()) for row in ET.fromstring(book.read('xl/sharedStrings.xml'))]
        rows = []
        for row in ET.fromstring(book.read('xl/worksheets/sheet1.xml')).findall('.//x:sheetData/x:row', ns):
            values = {}
            for cell in row:
                value = cell.find('x:v', ns)
                if value is not None:
                    values[re.sub(r'\d+', '', cell.attrib['r'])] = strings[int(value.text)] if cell.attrib.get('t') == 's' else value.text
            if str(values.get('D', '')).startswith('GE_Player_Exe_'): rows.append(values)
        return rows


def plain(value):
    return re.sub('<[^>]*>', '', value.get('LocalizedString') or value.get('SourceString') or '').replace('\\r\\n', '\n')


def build(directory, workbook):
    load = lambda name: json.loads((DATA / (name + '.json')).read_text())
    output = {name: load(name) for name in ['characters', 'actions', 'action_hit_profiles', 'awakenings', 'mechanisms', 'arcs', 'arc_refinements', 'buffs', 'formula_constants']}
    damage = table(directory, 'DT_SkillDamageData')
    tips = table(directory, 'DT_GameplayAbilityTipsData')
    curves = table(directory, 'DT_ExeEffectFigure')
    frames = table(directory, 'DT_ExeEffectFigure_Frame')
    scores = table(directory, 'DT_ExeEffectFigure_Score')
    val = lambda rows, key: rows[key]['Keys'][0]['Value']
    for row in workbook_rows(workbook):
        actual = damage[row['D']]
        for level, column in enumerate(['P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB']):
            if abs(float(row[column]) - actual['AtkRateBaseArray'][level]) > 1e-9:
                raise ValueError('Workbook and configuration disagree at skill level ' + str(level+1))
        for column, key in [('L','AtkRateBaseArray'), ('M','ChargeAdd'), ('N','HeterochromeAdd'), ('O','UnbalValue')]:
            expected = actual[key][0] if isinstance(actual[key], list) else actual[key]
            if abs(float(row[column]) - expected) > 1e-9: raise ValueError('Workbook and configuration disagree: ' + row['D'])
    adjusted=adjust_tables({'DT_SkillDamageData':damage,'DT_GameplayAbilityTipsData':tips,'DT_ExeEffectFigure':curves,'1074AwakenEffect':table(directory,'1074AwakenEffect')})
    damage,tips,curves=(adjusted[name]for name in ['DT_SkillDamageData','DT_GameplayAbilityTipsData','DT_ExeEffectFigure'])
    rank_thresholds = [val(curves, f'Exe_RankScore{i}') for i in range(1,7)]
    rank_mult = [k['Value'] for k in table(directory, 'DT_ExeEffectFigure_RankMult')['Exe_RankMult']['Keys']]
    decay = [k['Value'] for k in table(directory, 'DT_ExeEffectFigure_ScoreDecay')['Exe_ScoreDecay']['Keys']]
    element=table(directory,'DT_Character')['1074']['ElementData']
    base=table(directory,'DT_PlayerPackData')[element['PropModifyID']]
    growth=table(directory,'DT_PlayerModifyPackData')
    stage=table(directory,'DT_CharacterBreakthroughDataTable')[element['BreakthroughPackId']+'_6']['ModifyPackID']
    base_stats={}
    for key,prop in [('atk','AtkBase'),('hp','HPMaxBase'),('def','DefBase')]:
        value=base[prop]
        for modifier in [growth[element['UpgradeModifyPackId']+'_80'],growth[stage]]:
            if modifier['ConditionArray']:raise ValueError('Conditional character growth')
            for item in modifier['ModifyData']:
                if item['PropName']!=prop:continue
                if not item['ModifierOp'].endswith('MODIFY_MODOP_ADDITIVE'):raise ValueError('Nonadditive character growth')
                value+=item['PropValue']
        base_stats[key]=int(value)
    character = dict(id=CID, name='Exe', element='暗', adaptation='等离子', rarity='S', level=80,
        avatar='/static/images/characters/avatar/Exe-square.webp', portrait='/static/images/characters/portrait/Exe.webp',
        base_stats=base_stats, modifiers=dict(crit_rate=.05,crit_dmg=.5),
        bond_bonus=dict(label='8暴伤',modifiers={'crit_dmg':.08}),
        energy_capacity=120, initial_personal_resources={'帧数':0,'评分':0},
        timeline_personal_resources=['帧数','评分'],
        ranked_resource_system=dict(frame_resource='帧数',score_resource='评分',
            frame_cap=val(curves,'ExeFrameMax'),score_cap=rank_thresholds[-1],
            export_cost=-val(curves,'Exe_CloneCost'),promote_reaction='黯星',promote_score=rank_thresholds[3],
            reaction_score=val(curves,'Exe_Score_Reaction'),empowered_buff_key='character_exe_equation',
            heal_atk_per_frame=val(curves,'Exe_RecoverHpCoef_L6'),score_per_consumed_frame=val(curves,'Exe_AddScore_L6'),
            ranks=[dict(label=label,threshold=t,frame_multiplier=m,decay_per_second=d) for label,t,m,d in zip(['B','A','S','SS','SSS','Ω'],rank_thresholds,rank_mult,decay)]))
    chars=[c for c in output['characters'] if c['id']!=CID]
    index=max(i for i,c in enumerate(chars) if c['element']=='暗')+1;chars.insert(index,character);output['characters']=chars
    configurations=[('a1','Melee1',1,.6),('a2','Melee2',2,.6),('a3','Melee3',3,.7),
        ('a4',[('Melee4',2),('Melee4_1',2),('Melee4_2',3)],1,1.3),('a5','Melee5',6,.8),
        ('z','Branch',1,.7),('下落','AirAttack',1,.8),('闪反','PerfectEvadeAttack',1,1),
        ('e',[('Skill',3),('Skill1',3)],1,1.2),('q',[('UltraSkill1',13),('UltraSkill2',6)],1,0),
        ('援护',[('QTE',2),('QTE1',1)],1,1.5),('数字残像','CloneAtk',1,0)]
    actions=[]
    for order,(name,parts,count,seconds) in enumerate(configurations):
        if isinstance(parts,str):parts=[(parts,count)]
        hits=[('GE_Player_Exe_'+part+'_Damage',damage['GE_Player_Exe_'+part+'_Damage']) for part,repetitions in parts for _ in range(repetitions)]
        aggregate=lambda field: round(sum(float(row[field]) for _,row in hits),9)
        levels=[round(sum(row['AtkRateBaseArray'][level] for _,row in hits),9) for level in range(13)]
        action_type='E' if name=='e' else 'Q' if name=='q' else '援护' if name=='援护' else '闪反' if name=='闪反' else '普攻'
        action=dict(id='action_exe_'+{'下落':'air','闪反':'counter','援护':'support','数字残像':'clone'}.get(name,name),character_id=CID,character_name='Exe',name=name,
            action_type=action_type,damage_type=action_type,is_background_damage=name=='数字残像',duration_seconds=seconds,duration_ticks=round(seconds*10),
            duration_implemented=True,duration_estimated=name in ['下落','闪反','e','援护'],multipliers=dict(atk=levels[0],hp=0,**{'def':0},flat=0),
            atk_multipliers_by_level=levels,energy_gain=aggregate('ChargeAdd'),harmony=aggregate('HeterochromeAdd'),stagger=aggregate('UnbalValue'),
            energy_return=0,self_modifiers={},cooldown_ticks=150 if name=='e' else 200 if name=='q' else 0,energy_cost=120 if name=='q' else 0,
            personal_resource_cost={},personal_resource_gain={},hit_count=len(hits),display_order=order,
            can_background_override=name in ['a2','a3','a4','a5'],tags=['召唤物']+(['数字残像']if name=='数字残像'else []),
            ranked_resource_hits=[dict(frames=val(frames,key),score=val(scores,key),atk_levels=row['AtkRateBaseArray'])for key,row in hits],
            ranked_export_limit=5 if name=='e' else 10 if name=='q' else 1 if name in ['a5','闪反','援护'] else 0,
            ranked_export_limit_empowered=10 if name in ['e','q'] else 3 if name in ['闪反','援护'] else 1 if name!='数字残像' else 0,
            ranked_export_last_hit=name=='a5')
        if name=='q':
            action['is_time_stop_zero']=True
            action['ranked_export_feedback']=True
        if name=='数字残像':action['duration_estimated']=False
        if action_type=='援护':action['resonance_skill_level_bonus']=False
        cumulative=[dict(hit_count=0,damage=0,energy=0,harmony=0,stagger=0)]
        for i,(_,hit) in enumerate(hits,1):
            prev=cumulative[-1]; cumulative.append(dict(hit_count=i,damage=round(prev['damage']+hit['AtkRateBaseArray'][0],9),energy=round(prev['energy']+hit['ChargeAdd'],9),harmony=round(prev['harmony']+hit['HeterochromeAdd'],9),stagger=round(prev['stagger']+hit['UnbalValue'],9)))
        if seconds and action_type!='援护':output['action_hit_profiles'][action['id']]={'cumulative':cumulative}
        if name=='数字残像':
            action.pop('is_background_damage',None)
            action['skill_level_category']='basic'
            character['ranked_resource_system']['export_action']=action
        else:
            actions.append(action)
    output['actions']=[a for a in output['actions']if a['character_id']!=CID]+actions
    # Explicit summon classes supplied for this character's team interaction.
    for a in output['actions']:
        summon=(a['character_id']==CID or (a['character_name']=='娜娜莉' and '副手'in a['name']) or
            (a['character_name']=='伊洛伊' and a['action_type']!='Q') or (a['character_name']=='黑羽'and a['name']=='失乐鸟') or
            (a['character_name']=='卡厄斯'and a['action_type']=='E') or (a['character_name']=='哈尼娅'and '后台'in a['name']) or
            (a['character_name']=='阿德勒'and a['action_type']in['E','Q']))
        if summon and a.get('damage_type')!='无':a['tags']=list(dict.fromkeys([*a.get('tags',[]),'召唤物']))
    output['formula_constants']['personal_resource_caps'][CID]={'帧数':100,'评分':5000}
    output['formula_constants']['default_build_options'][CID]={'curtain_bonus':{'value':8,'stat':'暴击','passive_type':'type3','fixed':True}}
    awakened=adjusted['1074AwakenEffect']['1074']['AwakenEffectStructList']
    output['awakenings']['Exe']=[dict(title=plain(a['Title']),description=plain(a['Desc']).replace('{0}',str(damage['GE_Player_Exe_CloneBoom_Damage']['AtkRateBaseArray'][0]*100)),implemented=True)for a in awakened]
    output['awakenings']['Exe'][-1]['description'] += '\n当前排轴不模拟斩杀。'
    mechanisms=[dict(id='character_exe_talent1',character_id=CID,talent=1,name='额外得分',description=plain(tips['GA_Exe_Passive_1']['AbilityDescription'][0]['Description'])),
        dict(id='character_exe_talent2',character_id=CID,talent=2,name='临时超频',description=plain(tips['GA_Exe_Passive_2']['AbilityDescription'][0]['Description']))]
    output['mechanisms']=[m for m in output['mechanisms']if m.get('character_id')!=CID]+mechanisms
    return output,damage,curves


def add_effects(output, damage, curves, directory):
    buffs = [b for b in output['buffs'] if not b['id'].startswith(('character_exe_', 'arc_exe_'))]
    def rule(key, name, event='passive', effects=None, source=None, target=None, conditions=None, duration=None, stacking=None, provider=None, **extra):
        b=dict(id=key,name=name,providers=[provider or dict(kind='character',id=CID)],priority=50,
            trigger=dict(event=event,source=source or dict(scope='registrar'),conditions=conditions or []),
            target=target or dict(scope='registrar'),duration=duration or dict(type='permanent',loop_carry=True),
            stacking=dict(key=key,mode='refresh',max_stacks=1,**(stacking or {})) if not stacking else dict(key=key,**stacking),
            effects=effects or {},display=dict(line=event!='passive'),**extra)
        buffs.append(b);return b
    node=lambda n:dict(type='awakening_min',min=n)
    resonance=lambda n:dict(type='awakening_count_min',min=n)
    bands=lambda field,steps:dict(personal_resource_steps=dict(resource='评分',effect_key=field,steps=[dict(min=t,value=v)for t,v in steps]))
    rule('character_exe_equation','方程构建','action_start',source=dict(scope='registrar',action_ids=['action_exe_q']),duration=dict(type='time',ticks=150,loop_carry=True))
    rule('character_exe_a','print 你好，我的世界',conditions=[node(1)],dynamic_effects=bands('element_dmg_暗',[(0,0),(500,.2)]))
    rule('character_exe_b','define X/Y',conditions=[node(2)],dynamic_effects=bands('atk_pct',[(0,.1),(100,.12),(300,.15),(500,.2),(3000,.3)]))
    rule('character_exe_b_omega','define X/Y·Ω评分',conditions=[node(2),resonance(6)],dynamic_effects=bands('atk_pct',[(0,0),(5000,.2)]))
    rule('character_exe_c','for 永远一起玩','action_hit',effects={'res_down_暗':.15},conditions=[node(3)],source=dict(scope='registrar',action_ids=['action_exe_e']),target=dict(scope='team'),duration=dict(type='time',ticks=120,loop_carry=True))
    rule('character_exe_d_team','kill 杀死进程·召唤物',effects={'action_dmg':.1},conditions=[node(4)],target=dict(scope='team',tags=['召唤物']))
    rule('character_exe_d_clone','kill 杀死进程·数字残像',effects={'action_dmg':.3},conditions=[node(4)],target=dict(scope='registrar',tags=['数字残像']))
    rule('character_exe_resonance3','save 不曾想象的未来',effects={'energy_recharge':.25},conditions=[resonance(3)])
    rule('character_exe_resonance6_interrupt','load 不可证明的宝物·抗打断',effects={'interrupt_resistance':.5},conditions=[resonance(6)])
    overclock=rule('character_exe_overclock','临时超频','action_hit',effects={'all_dmg':curves['ExeDmgUpPas2']['Keys'][0]['Value']},source=dict(scope='team',tags=['召唤物']),
         stacking=dict(mode='add_stack',max_stacks=4,max_stacks_per_source=2,stack_gain_from='hit_count',seed_from_arc_names=['茶花会','海特洛的安宁']))
    overclock['trigger']['periodic_damage']=True
    rule('character_exe_cooldown','数字残像·冷却缩减','action_hit',source=dict(scope='registrar',action_ids=['action_exe_clone']),
         activation=dict(reduce_action_cooldowns={'action_exe_e':10},cooldown_floor_ticks=20))
    hit=damage['GE_Player_Exe_CloneBoom_Damage']
    rule('character_exe_e','null 空指针','action_hit',source=dict(scope='registrar',action_ids=['action_exe_clone']),
         conditions=[node(5),dict(type='active_buff_key',key='character_exe_equation')],
         triggered_damage=dict(action=dict(id='action_exe_clone_boom',name='数字残像·冲击',action_type='普攻',damage_type='普攻',
            skill_level_category='basic',hit_count=1,tags=['召唤物','数字残像'],multipliers={'atk':hit['AtkRateBaseArray'][0]},
            atk_multipliers_by_level=hit['AtkRateBaseArray'],ranked_resource_hits=[dict(frames=0,score=10)])))
    for buff in buffs:
        if buff['id'] in ['character_exe_cooldown','character_exe_e']:buff['display']['line']=False
    fork=table(directory,'DT_ForkItemData'); raw=next(v for k,v in fork.items()if k.casefold()=='fork_exe')
    arc=dict(id=ARC,name=plain(raw['ItemName']),level=80,rarity='S',adaptation='等离子',base_atk=570,
             detail='暴击伤害提高28%–40%，攻击力提高28%–56%；装备者每造成一次暗属性伤害，召唤物伤害提高2%–4%，最多10层，持续20秒，重复获得刷新持续时间。',
             modifiers={'crit_dmg':.48+.28,'atk_pct':.28},element_dmg={})
    arcs=[a for a in output['arcs']if a['id']!=ARC];index=next(i for i,a in enumerate(arcs)if a['adaptation']=='等离子');arcs.insert(index,arc);output['arcs']=arcs
    curve=table(directory,'CT_ForkBuff')
    def cv(key,level):
        keys=curve[key]['Keys']
        for left,right in zip(keys,keys[1:]):
            if left['Time']<=level<=right['Time']:
                return round(left['Value']+(right['Value']-left['Value'])*(level-left['Time'])/(right['Time']-left['Time']),9)
        if len(keys)==1:return keys[0]['Value']
        raise ValueError('Unresolved refinement curve')
    b=rule('arc_exe_summon','缥缈的执手同行','action_hit',effects={'action_dmg':.02},source=dict(scope='registrar',elements=['暗']),target=dict(scope='registrar',tags=['召唤物']),
           stacking=dict(mode='add_stack',max_stacks=10,stack_gain_from='hit_count'),duration=dict(type='time',ticks=200,loop_carry=True),provider=dict(kind='arc',id=ARC))
    output['arc_refinements']['arcs'][ARC]=dict(name=arc['name'],default_level=1,available_levels=[1,2,3,4,5],effect_name='缥缈的执手同行',effect_description=arc['detail'],
        levels={str(i):dict(panel_modifiers={'crit_dmg':round(.48+cv('buff_Exe_CritDamageUp',i),9),'atk_pct':cv('buff_Exe_SummonDmgUp',i)},element_dmg={},
        effect_values=[f"{cv(key,i)*100:g}%" for key in ['buff_Exe_CritDamageUp','buff_Exe_SummonDmgUp','buff_Exe_StackDmgUp']]+['20'],buff_effects={'arc_exe_summon':{'action_dmg':cv('buff_Exe_StackDmgUp',i)}})for i in range(1,6)})
    for mechanism in output['mechanisms']:
        if mechanism['id']=='character_exe_talent2':mechanism['replaces_rule_ids']=['character_exe_overclock','character_exe_cooldown']
    character_names={c['id']:c['name'] for c in output['characters']}
    for buff in buffs:
        if not buff.get('periodic_damage'):continue
        if any(p['kind']=='character' and character_names.get(p['id'])=='阿德勒' for p in buff.get('providers',[])):
            buff['periodic_damage']['tags']=list(dict.fromkeys([*buff['periodic_damage'].get('tags',[]),'召唤物']))
    output['buffs']=buffs
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--damage-workbook',type=Path,required=True)
    args=parser.parse_args();output,damage,curves=build(args.data_dir,args.damage_workbook)
    add_effects(output,damage,curves,args.data_dir)
    # Buff and arc configuration is maintained alongside the imported character.
    for name,value in output.items():
        target=DATA/(name+'.json')
        if json.loads(target.read_text()) == value:
            continue
        temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(target)


if __name__=='__main__':main()
