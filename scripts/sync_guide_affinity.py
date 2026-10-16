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
TABLES = {
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
        role = tables['roles'].get(identifier)
        gifts = []
        for gift_id, row in tables['gifts'].items():
            gain = next((entry['LikeabilityValue'] for entry in row['SpeicalGiftDatas']
                         if str(entry['RoleId']) == identifier), None)
            exclusive = row['ExclusiveLikeabilityID']
            if gain is None or exclusive not in ('None', identifier):
                continue
            if gain < 0:
                raise ValueError('Negative affinity gain')
            gifts.append({'id': gift_id, 'name': item_name(gift_id),
                          'affinity_gain': gain, 'default_gain': row['DefaultLikeabilityValue'],
                          'exclusive': exclusive == identifier, 'unlimited': bool(row['bUnlimit']),
                          'offers': offers.get(gift_id.casefold(), [])})
        gifts.sort(key=lambda g: (g['affinity_gain'], g['id']))
        levels = [{'level': index, 'max_value': entry['MaxValue']}
                  for index, entry in enumerate(role['LevelInfo'], 1)] if role else []
        if role and (not levels or any(isinstance(row['max_value'], bool) or
                not isinstance(row['max_value'], (int, float)) or row['max_value'] < 0 for row in levels)):
            raise ValueError('Invalid affinity levels')
        result.append({'id': identifier, 'affinity': {
            'available': role is not None, 'levels': levels, 'gifts': gifts}})
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
