#!/usr/bin/env python3
"""Explicit offline sync of character affinity levels, preferred gifts and shop offers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
AFFINITY_ROLE_IDS = {'1052': '1029', '1054': '1005', '1055': '1014'}
TABLES = {
    'reward_groups': 'Drop/Client/ClientDropGroupDataTable.json',
    'reward_sequences': 'Drop/DropSequenceDataTable.json',
    'modifiers': 'PackData/ModifyData/DT_LikeabilityModifyData.json',
    'roles': 'LikeabilitySystem/DT_LikeabilityRoleData.json',
    'gifts': 'LikeabilitySystem/DT_LikeabilityGiftData.json',
    'goods': 'Shop/GoodsDataTable.json',
    'shops': 'Shop/ShopDataTable.json',
    'items': 'Inventory/DT_ItemConfig.json',
    'currencies': 'Inventory/DT_CapitalItemConfig.json',
    'foods': 'Food/DT_FoodItemData.json',
}


def text(value):
    if isinstance(value, dict):
        value = value.get('LocalizedString') or value.get('SourceString') or ''
    return re.sub(r'<[^>]*>', '', value or '').strip()


ATTRIBUTE_NAMES = {'AtkUp': '攻击力', 'HPMaxUp': '生命值', 'DefUp': '防御力',
                   'CritBase': '暴击率', 'CritDamageBase': '暴击伤害'}


def level_rewards(award_id, tables, names):
    if not award_id or award_id == 'None':
        return []
    groups = [(key, row) for key, row in tables.get('reward_groups', {}).items()
              if re.fullmatch(re.escape(award_id) + r'_\d+', key)]
    rewards = []
    for _, group in groups:
        sequence_id = group['SequenceId']
        sequences = [row for key, row in tables.get('reward_sequences', {}).items()
                     if key == sequence_id or re.fullmatch(re.escape(sequence_id) + r'_\d+', key)]
        if not sequences:
            raise ValueError(f'Missing reward sequence: {sequence_id}')
        if len(sequences) != 1 or group.get('DropConditions'):
            raise ValueError('Variable or conditional affinity reward requires explicit handling')
        for row in sequences:
            if row['SequenceNumType'] != 'EDropSequenceNumType::DROPSEQUENCENUMTYPE_FIXED':
                raise ValueError('Unsupported variable affinity reward')
            item = names.get(row['ItemID'].casefold(), {})
            name = text(item.get('ItemName'))
            if not name:
                raise ValueError(f'Missing reward name: {row["ItemID"]}')
            rewards.append({'name': name, 'amount': row['Num'],
                            'description': text(item.get('UseContext'))})
    if tables.get('reward_groups') and not groups:
        raise ValueError(f'Missing reward group: {award_id}')
    return rewards


def affinity_bonuses(role, tables):
    bonuses = []
    for binding in role.get('MapModifyData', []):
        level = int(binding['Key'])
        modifier = tables.get('modifiers', {}).get(binding['Value'])
        if modifier is None:
            if 'modifiers' in tables:
                raise ValueError('Missing affinity modifier')
            continue
        if modifier.get('ConditionArray'):
            raise ValueError('Conditional affinity modifier requires explicit handling')
        for row in modifier['ModifyData']:
            if row['PropName'] not in ATTRIBUTE_NAMES or row['ModifierOp'] != 'EModifyModOp::MODIFY_MODOP_ADDITIVE':
                raise ValueError('Unknown affinity attribute')
            value = round(row['PropValue'] * 100, 8)
            name = ATTRIBUTE_NAMES[row['PropName']]
            bonuses.append({'level': level, 'name': name, 'value': value, 'percent': True,
                            'text': f'{name}＋{value:g}%'})
    return bonuses


def build_affinity(ids, tables):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Missing or duplicate character IDs')
    if any(not tables[key] for key in ('roles', 'gifts', 'goods', 'shops', 'items')):
        raise ValueError('Empty affinity table')
    names = {key.casefold(): row for category in ('items', 'currencies', 'foods')
             for key, row in tables.get(category, {}).items()}

    def item_name(identifier):
        return text(names.get(identifier.casefold(), {}).get('ItemName')) or identifier

    offers = {}
    for goods_id, row in tables['goods'].items():
        if row['PriceType'] != 'PriceShowType::Cost':
            continue
        shop_id = row['OwnerShopID']
        currency = row['CurrencyID']
        offer = {'goods_id': goods_id, 'shop_id': shop_id,
                 'shop_name': text(tables['shops'].get(shop_id, {}).get('ShopName')) or shop_id,
                 'currency_id': currency, 'currency_name': item_name(currency),
                 'price': row['Price'], 'discount_price': row['OffPrice'] if row['bUseOff'] else None,
                 'discount_time_limited': bool(row['bUseOffTime']),
                 'availability_limited': bool(row['UseLimitStartTime'] or row['UseLimitEndTime']),
                 'unlock_id': None if row['UnLockID'] == 'None' else row['UnLockID']}
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
               for value in (offer['price'], offer['discount_price']) if value is not None):
            raise ValueError('Invalid gift price')
        offers.setdefault(row['ItemID'].casefold(), []).append(offer)
    # Collapse equivalent branches for reading, retaining every shop/product identity.
    for item_id, entries in offers.items():
        grouped = {}
        for offer in entries:
            shop_id, goods_id = offer.pop('shop_id'), offer.pop('goods_id')
            signature = json.dumps(offer, ensure_ascii=False, sort_keys=True)
            if signature not in grouped:
                grouped[signature] = {**offer, 'shop_ids': [], 'goods_ids': []}
            grouped[signature]['shop_ids'].append(shop_id)
            grouped[signature]['goods_ids'].append(goods_id)
        offers[item_id] = list(grouped.values())
    result = []
    for identifier in ids:
        role_id = AFFINITY_ROLE_IDS.get(identifier, identifier)
        role = tables['roles'].get(role_id)
        gifts = []
        for gift_id, row in tables['gifts'].items():
            gain = next((entry['LikeabilityValue'] for entry in row['SpeicalGiftDatas']
                         if str(entry['RoleId']) == role_id), None)
            exclusive = row['ExclusiveLikeabilityID']
            if gain is None or exclusive not in ('None', role_id, identifier):
                continue
            if gain < 0:
                raise ValueError('Negative affinity gain')
            gifts.append({'id': gift_id, 'name': item_name(gift_id),
                          'affinity_gain': gain, 'default_gain': row['DefaultLikeabilityValue'],
                          'exclusive': exclusive in (role_id, identifier), 'unlimited': bool(row['bUnlimit']),
                          'offers': offers.get(gift_id.casefold(), [])})
        gifts.sort(key=lambda g: (g['affinity_gain'], g['id']))
        bonuses = affinity_bonuses(role, tables) if role else []
        levels = [{'level': index, 'max_value': entry['MaxValue'],
                   'rewards': level_rewards(entry.get('AwardID'), tables, names),
                   'bonuses': [b for b in bonuses if b['level'] == index]}
                  for index, entry in enumerate(role['LevelInfo'], 1)] if role else []
        if role and (not levels or any(isinstance(row['max_value'], bool) or
                not isinstance(row['max_value'], (int, float)) or row['max_value'] < 0 for row in levels)):
            raise ValueError('Invalid affinity levels')
        result.append({'id': identifier, 'affinity': {
            'available': role is not None, 'levels': levels, 'gifts': gifts,
            'full_affinity_bonus': [b for b in bonuses if b['level'] <= len(levels)]}})
    return {'characters': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ntedata-dir', type=Path, help='Directory containing the named data tables')
    parser.add_argument('--output', type=Path, default=ROOT / 'app/modules/guide/data/affinity.json')
    args = parser.parse_args()

    def load(entry):
        key, path = entry
        if args.ntedata_dir:
            payload = json.loads((args.ntedata_dir / path.rsplit('/', 1)[1]).read_text())
        else:
            payload = json.loads(subprocess.run(['gh', 'api',
                'repos/mc-ctrl/NTEData/contents/DataTable/' + path,
                '-H', 'Accept: application/vnd.github.raw+json'], check=True, capture_output=True).stdout)
        return key, next(item['Rows'] for item in payload if 'Rows' in item)
    with ThreadPoolExecutor(max_workers=6) as executor:
        tables = dict(executor.map(load, TABLES.items()))
    ids = [str(c['id']) for c in json.loads((ROOT / 'app/modules/guide/data/catalog.json').read_text())['characters']]
    result = build_affinity(ids, tables)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temp.replace(args.output)
    print(f'Synced affinity for {len(ids)} character records')


if __name__ == '__main__':
    main()
