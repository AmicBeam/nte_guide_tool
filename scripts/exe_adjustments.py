"""Current Exe damage, passive and resonance adjustments for offline imports."""
from copy import deepcopy
from decimal import Decimal

CLONE_EFFECTS={'GE_Player_Exe_CloneAtk_Damage','GE_Player_Exe_CloneBoom_Damage'}
SUPPORT_EFFECTS=('GE_Player_Exe_QTE_Damage','GE_Player_Exe_QTE1_Damage')


def adjust_tables(tables):
    result={name:dict(rows) for name,rows in tables.items()}
    damage=result['DT_SkillDamageData']
    baseline=damage['GE_Player_Exe_Melee1_Damage']['AtkRateBaseArray'][0]
    scales={.537:Decimal('0.756'),.4833:Decimal('0.84'),.405972:Decimal('1')}
    if baseline not in scales:raise ValueError('Unexpected Exe baseline multiplier')
    scale=scales[baseline]
    for key,raw in list(damage.items()):
        if scale==1 or not key.startswith('GE_Player_Exe_') or key in CLONE_EFFECTS or key in SUPPORT_EFFECTS:continue
        row=deepcopy(raw)
        for field in ['AtkRateBaseArray','DefRateBaseArray','HPRateBaseArray']:
            row[field]=[float(Decimal(str(v))*scale) for v in row.get(field,[])]
        damage[key]=row
    # Preserve each damage row's level growth and the two-small/one-large hit ratio.
    support_base=sum(Decimal(str(damage[key]['AtkRateBaseArray'][0]))*count
                     for key,count in zip(SUPPORT_EFFECTS,(2,1)))
    if support_base<=0:raise ValueError('Invalid Exe support multiplier')
    for key in SUPPORT_EFFECTS:
        row=deepcopy(damage[key])
        row['AtkRateBaseArray']=[round(float(Decimal(str(v))*2/support_base),9)
                                 for v in row['AtkRateBaseArray']]
        damage[key]=row
    def text_update(value,old,new):
        if isinstance(value,dict):return {k:text_update(v,old,new)for k,v in value.items()}
        if isinstance(value,list):return [text_update(v,old,new)for v in value]
        return value.replace(old,new) if isinstance(value,str) else value
    tips=result['DT_GameplayAbilityTipsData']
    tips['GA_Exe_Passive_2']=text_update(tips['GA_Exe_Passive_2'],'10%','2.5%')
    tips['GA_Exe_Passive_2']=text_update(tips['GA_Exe_Passive_2'],'当前冷却时间低于2秒时不再触发','冷却时间无法减至2秒以下')
    curves=result['DT_ExeEffectFigure']
    curves['ExeDmgUpPas2']=deepcopy(curves['ExeDmgUpPas2'])
    for key in curves['ExeDmgUpPas2']['Keys']:key['Value']=.025
    awaken=result['1074AwakenEffect']
    awaken['1074']=deepcopy(awaken['1074'])
    row=awaken['1074']['AwakenEffectStructList'][-1]
    row['Desc']=text_update(row['Desc'],'10%','5%')
    for field in ['SourceString','LocalizedString']:
        if field in row['Desc'] and '抗打断' not in row['Desc'][field]:
            row['Desc'][field]+='\nExe抗打断能力提高50%。'
    return result
