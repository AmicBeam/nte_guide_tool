#!/usr/bin/env python3
"""Explicit offline import of permanent furniture and Puka training bonuses."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from sync_guide_affinity import text

ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    'furniture': 'DataAssets/DataAssetSet/Room/DT_SpecialFurniture.json',
    'furniture_modifiers': 'DataTable/PackData/ModifyData/DT_SpecialFurnitureModifyData.json',
    'training': 'DataAssets/DataAssetSet/Tycoon/DT_TycoonPukalandTraining.json',
    'training_modifiers': 'DataAssets/DataAssetSet/Tycoon/DT_PukalandTrainingModifyData.json',
    'items': 'DataTable/Inventory/DT_ItemConfig.json',
}
ATTRIBUTES = {'AtkAdd': ('攻击力', False), 'DefAdd': ('防御力', False),
              'HPMaxAdd': ('生命值', False), 'CritBase': ('暴击率', True),
              'CritDamageBase': ('暴击伤害', True), 'MagBase': ('环合强度', False),
              'UnbalIntensityBase': ('倾陷强度', False)}
FURNITURE_IDS = ('SF_0011', 'SF_0012', 'SF_0013', 'SF_0015', 'SF_0016', 'SF_0017', 'SF_0009')


def bonuses(modifier):
    if modifier['ConditionArray']:
        raise ValueError('Conditional bonus requires explicit handling')
    result = []
    for row in modifier['ModifyData']:
        if row['ModifierOp'] != 'EModifyModOp::MODIFY_MODOP_ADDITIVE':
            raise ValueError('Unsupported modifier operation')
        name, percent = ATTRIBUTES[row['PropName']]
        value = round(row['PropValue'] * (100 if percent else 1), 8)
        result.append({'name': name, 'value': value, 'percent': percent,
                       'text': f'{name}＋{value:g}{"%" if percent else ""}'})
    return result


def build_records(tables):
    items = tables['items']
    def item_name(identifier):
        name = text(items[identifier]['ItemName'])
        if not name:
            raise ValueError('Missing item name')
        return name
    furniture = []
    for identifier in FURNITURE_IDS:
        row = tables['furniture'][identifier]
        unlock = []
        for condition in row['ActivateData']['Conditions']:
            description = text(condition['ActivateDescription']).replace('{0}', str(condition['Count']))
            unlock.append(description.strip() or f'获得「{item_name(condition["TargetId"])}」×{condition["Count"]}后激活')
        levels = sorted(row['UpgradeLevels'], key=lambda level: level['Level'])
        if [level['Level'] for level in levels] != list(range(1, len(levels) + 1)):
            raise ValueError('Non-contiguous furniture levels')
        output = []
        for index, level in enumerate(levels):
            if level['AttributeID'] != 'None':
                effects = bonuses(tables['furniture_modifiers'][level['AttributeID']])
            elif level['ConsumeReduceValue']:
                value = round(-level['ConsumeReduceValue'] * 100, 8)
                name = {'SF_0015': '冲刺、闪避耐力消耗', 'SF_0016': '攀爬耐力消耗', 'SF_0017': '游泳耐力消耗'}[identifier]
                effects = [{'name': name, 'value': -value, 'percent': True, 'text': f'{name}降低{value:g}%'}]
            elif identifier == 'SF_0009':
                value = level['FixValue']
                effects = [{'name': '本性像素上限', 'value': value, 'percent': False, 'text': f'本性像素上限＋{value:g}'}]
            else:
                raise ValueError('Unsupported furniture bonus')
            costs = []
            # UpgradeMaterials describes the transition FROM this level; max level has no next upgrade.
            if index < len(levels) - 1:
                for material in level['UpgradeMaterials']:
                    if material['MaterialType'].endswith('::Item'):
                        name = item_name(material['MaterialID'])
                    elif identifier == 'SF_0009' and material['MaterialType'].endswith('::Progress'):
                        name = '本性像素消耗进度'
                    else:
                        raise ValueError('Unsupported upgrade requirement')
                    costs.append({'name': name, 'amount': material['Count']})
                if not costs:
                    raise ValueError('Missing upgrade requirement')
            output.append({'level': level['Level'], 'bonuses': effects, 'next_level': level['Level'] + 1 if index < len(levels) - 1 else None, 'upgrade_costs': costs})
        furniture.append({'id': identifier, 'name': text(row['Title']), 'max_level_name': text(row['Title2']),
            'unlock': unlock, 'description': text(row['PostActivateDescription']),
            'max_level': levels[-1]['Level'], 'levels': output})
    training = []
    total = 0
    for row in sorted(tables['training'].values(), key=lambda row: row['Level']):
        if row['Level'] == 0:
            continue
        if row['Locked']:
            raise ValueError('Locked training level requires explicit handling')
        total += row['Fons']
        training.append({'level': row['Level'], 'required_puka_level': row['PukalandLevel'],
            'fons_cost': row['Fons'], 'total_fons': total,
            'bonuses': bonuses(tables['training_modifiers'][row['AttributeID']])})
    if [row['level'] for row in training] != list(range(1, 21)):
        raise ValueError('Expected all 20 training levels')
    return [
        {'id': 'furniture', 'name': '异象家具', 'description': '上阵角色的攻击力、防御力、暴击伤害，以及探索耐力和本性像素上限加成。',
         'notes': ['加成按当前家具等级读取，不把不同等级重复相加。解锁后1级即有对应效果。',
                   '升级表列的是当前等级升至下一级的单次材料或进度要求，满级不再显示升级消耗。',
                   '仅收录属性、耐力及本性像素上限相关家具，不包含产出和便捷服务家具。'], 'entries': furniture},
        {'id': 'puka-training', 'name': '噗咔特训', 'aliases': ['噗卡特训'],
         'description': '20级共需11,000,000方斯；满级提供攻击力＋28、防御力＋30、生命值＋420、暴击率＋2.5%、暴击伤害＋5%、环合强度＋12、倾陷强度＋12。',
         'notes': ['噗卡计划7级解锁首级特训，后续需满足对应噗卡计划等级并支付方斯。噗卡计划等级与特训等级分别记录。',
                   '表中属性是达到该特训等级后的总加成，各级之间不叠加；单次方斯为升入该等级所需，累计方斯为从未特训升至该等级的总花费。',
                   '家具和特训加成属于常驻属性。核对角色面板时先确认是否已包含，避免重复叠加。'], 'levels': training},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ntedata-dir', type=Path, help='本地解包目录或包含同名数据表的目录')
    args = parser.parse_args()
    def load(pair):
        key, path = pair
        if args.ntedata_dir:
            location = args.ntedata_dir / path
            if not location.exists():
                location = args.ntedata_dir / Path(path).name
            raw = json.loads(location.read_text())
        else:
            raw = json.loads(subprocess.run(['gh', 'api', 'repos/mc-ctrl/NTEData/contents/' + path,
                '-H', 'Accept: application/vnd.github.raw+json'], check=True, capture_output=True, text=True).stdout)
        return key, raw[0]['Rows']
    with ThreadPoolExecutor(max_workers=5) as executor:
        tables = dict(executor.map(load, TABLES.items()))
    records = build_records(tables)
    path = ROOT / 'app/modules/guide/data/permanent_bonuses.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    print(f'Synced {len(records)} permanent bonus topics')


if __name__ == '__main__':
    main()
