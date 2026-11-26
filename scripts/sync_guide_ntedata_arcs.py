#!/usr/bin/env python3
"""Import explicitly selected arcs from local NTEData tables into the Wiki."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
from guide_skill_details import table_rows, display_text
from refresh_guide_data import clean
from sync_guide_arc_cultivation import build as cultivation_data, PATHS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'app/modules/guide/data'


def curve_value(curve, level):
    if curve['InterpMode'] != 'ERichCurveInterpMode::RCIM_Linear':
        raise ValueError('Unsupported arc parameter interpolation')
    keys = sorted((Decimal(str(v['Time'])), Decimal(str(v['Value']))) for v in curve['Keys'])
    if not keys or len({x for x, _ in keys}) != len(keys):
        raise ValueError('Missing or duplicate curve keys')
    level = Decimal(level)
    if not keys[0][0] <= level <= keys[-1][0]:
        if len(keys) != 1:
            raise ValueError('Arc parameter outside curve range')
        return keys[0][1]
    for (left, a), (right, b) in zip(keys, keys[1:]):
        if left <= level <= right:
            return a + (b-a)*(level-left)/(right-left)
    return keys[-1][1]


def assemble(directory, catalog, identifier):
    def table(name):
        return table_rows(json.loads((directory/(name+'.json')).read_text(encoding='utf-8')))
    def lookup(rows, key):
        return next(v for k, v in rows.items() if k.casefold() == key.casefold())
    raw = lookup(table('DT_ForkItemData'), identifier)
    element = raw['ElementData']
    group = element['ApplyGroupType'].rsplit('_', 1)[-1]
    type_id = str(('ONE','TWO','THREE','FOUR','FIVE').index(group)+1)
    types = table('DT_ForkTypeData')
    rarity = {'ORANGE':5,'PURPLE':4,'BLUE':3}[raw['ItemQuality'].rsplit('_',1)[-1]]
    names = {row['id_stats']: row['name'] for arc in catalog['arcs'] for row in arc['stats']}
    items = {k.casefold(): display_text(v['ItemName']) for k,v in table('DT_ItemConfig').items()}
    breakthroughs = table('DT_ForkBreakthroughData')
    stages = [lookup(breakthroughs,f"{element['BreakthroughPackId']}_{i}") for i in range(element['MaxBreakthrough']+1)]
    caps = [row['MaxForkLevel'] for row in stages]
    if caps != sorted(set(caps)) or caps[-1] != 80:
        raise ValueError('Invalid arc level caps')
    upgrades, modifiers = table('DT_ForkUpgradeData'), table('DT_ForkModifyData')
    stats = {}
    for level in range(1, caps[-1]+1):
        # Wiki stat arrays show the breakthrough already applied at each boundary level.
        stage = sum(cap <= level for cap in caps[:-1])
        growth = lookup(upgrades,f"{element['UpgradePackId']}_{level}")['ModifyPack']
        values = {}
        for key in (growth, stages[stage]['ModifyPackID']):
            row = lookup(modifiers,key)
            if row['ConditionArray']: raise ValueError('Conditional arc attribute')
            for mod in row['ModifyData']:
                if not mod['ModifierOp'].endswith('MODIFY_MODOP_ADDITIVE'):
                    raise ValueError('Nonadditive arc attribute')
                prop = mod['PropName']
                values[prop] = values.get(prop,Decimal(0)) + Decimal(str(mod['PropValue']))
        if level>1 and set(values)!=set(stats): raise ValueError('Inconsistent arc attributes')
        for prop,value in values.items():
            percent = prop != 'AtkBase'
            if prop not in names: raise ValueError('Unknown arc attribute: '+prop)
            value = float(value*100) if percent else int(value)
            stats.setdefault(prop,dict(id_stats=prop,name=names[prop],values=[],b_is_percent=percent))['values'].append(value)
    materials = []
    for row in stages[1:]:
        ids, amounts, named = [],[],[]
        for pair in row['NeedItems'].split(','):
            key,amount=pair.split(':');amount=int(amount)
            if amount<=0 or key.casefold() not in items:raise ValueError('Unknown arc cultivation material')
            ids.append(key);amounts.append(amount);named.append(dict(id=key,name=items[key.casefold()],amount=amount))
        currency, gold = row['NeedGolds'].split(':')
        if currency.casefold()!='gold' or int(gold)<0:raise ValueError('Invalid arc gold cost')
        materials.append(dict(level=str(row['MaxForkLevel']),items_id=ids,amount=amounts,items=named,cost_gold=int(gold)))
    stars = [lookup(table('DT_ForkUpgradeStarDataTable'),f"{element['UpgradeStarPackID']}_{i}") for i in range(1,element['MaxUpgradeStar']+1)]
    first=stars[0];bindings=first['DataList'];parameters=[];curves=table('CT_ForkBuff')
    for index,binding in enumerate(bindings):
        values=[]
        for level,star in enumerate(stars,1):
            if star['DataList'] != bindings or star['Description'] != first['Description']:
                raise ValueError('Inconsistent arc effect structure')
            value=curve_value(lookup(curves,binding['NameId']),level)
            if binding['bIsPercent']:value*=100
            values.append(format(value.normalize(),'f')+('%' if binding['bIsPercent'] else ''))
        parameters.append(dict(id_value=index,b_is_percent=binding['bIsPercent'],values=values))
    return dict(id=identifier,name=display_text(raw['ItemName']),rarity=rarity,type_id=type_id,
                type_name=display_text(types[type_id]['TypeName']),description=clean(display_text(raw['Description'])),
                stats=list(stats.values()),effect=dict(name=clean(display_text(first['Title'])),description=clean(display_text(first['Description'])),values=parameters),
                exclusive_character_ids=element['ExclusiveCharacterIDArray'],exclusive_desc=clean(display_text(element['ExclusiveDesc'])),materials=materials,aliases=[])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ntedata-dir',type=Path,required=True)
    parser.add_argument('--arc-id',action='append',required=True)
    args=parser.parse_args()
    def load(name):return json.loads((DATA/name).read_text(encoding='utf-8'))
    catalog=load('catalog.json')
    target=DATA/'ntedata_arcs.json'
    local=json.loads(target.read_text(encoding='utf-8')) if target.exists() else {'arcs':[]}
    for identifier in args.arc_id:
        arc=assemble(args.ntedata_dir,catalog,identifier)
        catalog['arcs']=[a for a in catalog['arcs'] if a['id']!=identifier]+[arc]
        local['arcs']=[a for a in local['arcs'] if a['id']!=identifier]+[arc]
    tables=[json.loads((args.ntedata_dir/Path(path).name).read_text(encoding='utf-8')) for path in PATHS]
    config=cultivation_data(catalog,*tables)
    for name,payload in [('catalog.json',catalog),('ntedata_arcs.json',local),('arc_cultivation.json',config)]:
        path=DATA/name;temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8');temporary.replace(path)
    print('Imported arcs: '+', '.join(args.arc_id))


if __name__=='__main__':main()
