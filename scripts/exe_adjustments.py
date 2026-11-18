"""Current Exe damage, passive and resonance adjustments for offline imports."""
from copy import deepcopy
from decimal import Decimal

CLONE_EFFECTS={'GE_Player_Exe_CloneAtk_Damage','GE_Player_Exe_CloneBoom_Damage'}


def adjust_tables(tables):
    result={name:dict(rows) for name,rows in tables.items()}
    damage=result['DT_SkillDamageData']
    baseline=damage['GE_Player_Exe_Melee1_Damage']['AtkRateBaseArray'][0]
    if baseline not in (.537,.4833):raise ValueError('Unexpected Exe baseline multiplier')
    for key,raw in list(damage.items()):
        if baseline==.4833 or not key.startswith('GE_Player_Exe_') or key in CLONE_EFFECTS:continue
        row=deepcopy(raw)
        for field in ['AtkRateBaseArray','DefRateBaseArray','HPRateBaseArray']:
            row[field]=[float(Decimal(str(v))*Decimal('0.9')) for v in row.get(field,[])]
        damage[key]=row
    def text_update(value,old,new):
        if isinstance(value,dict):return {k:text_update(v,old,new)for k,v in value.items()}
        if isinstance(value,list):return [text_update(v,old,new)for v in value]
        return value.replace(old,new) if isinstance(value,str) else value
    tips=result['DT_GameplayAbilityTipsData']
    tips['GA_Exe_Passive_2']=text_update(tips['GA_Exe_Passive_2'],'10%','2.5%')
    curves=result['DT_ExeEffectFigure']
    curves['ExeDmgUpPas2']=deepcopy(curves['ExeDmgUpPas2'])
    for key in curves['ExeDmgUpPas2']['Keys']:key['Value']=.025
    awaken=result['1074AwakenEffect']
    awaken['1074']=deepcopy(awaken['1074'])
    row=awaken['1074']['AwakenEffectStructList'][-1]
    row['Desc']=text_update(row['Desc'],'10%','5%')
    return result
