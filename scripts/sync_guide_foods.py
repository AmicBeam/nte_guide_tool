#!/usr/bin/env python3
"""Explicit offline import of foods grouped by their configured function."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path
import subprocess
from sync_guide_affinity import text
from sync_guide_arc_images import download

ROOT = Path(__file__).resolve().parents[1]
TABLES = {'foods': 'Food/DT_FoodItemData.json', 'types': 'Food/DT_ItemEffectTypeData.json',
          'goods': 'Shop/GoodsDataTable.json', 'shops': 'Shop/ShopDataTable.json',
          'sources': 'DT_OutputSourceRowData.json', 'source_details': 'DT_OutputDetailSource.json'}
BONUS_GROUPS = {'attack': '攻击力提升', 'crit_rate': '暴击率提升', 'crit_damage': '暴击伤害提升',
                'defense': '防御力提升', 'heal': '即时生命恢复', 'regen': '持续生命恢复',
                'revive': '复活与恢复', 'stamina_cost': '体力消耗降低', 'stamina_regen': '体力回复加快'}


def bonus_group(effect, function):
    for phrase, group in [('攻击力增加', 'attack'), ('暴击率增加', 'crit_rate'),
                          ('暴击伤害增加', 'crit_damage'), ('防御力增加', 'defense'),
                          ('消耗的体力降低', 'stamina_cost'), ('体力回复速度提升', 'stamina_regen')]:
        if phrase in effect:
            return group
    group = {'AddHPInstant': 'heal', 'AddHPConstant': 'regen', 'Revive': 'revive'}.get(function)
    if group is None:
        raise ValueError('Unrecognized food bonus')
    return group


def effect_segments(effect):
    result, start = [], 0
    for match in re.finditer(r'\d+(?:\.\d+)?%?', effect):
        if match.start() > start:
            result.append({'text': effect[start:match.start()], 'highlight': False})
        result.append({'text': match.group(), 'highlight': not effect[match.end():].startswith('秒')})
        start = match.end()
    if start < len(effect):
        result.append({'text': effect[start:], 'highlight': False})
    return result


def purchase_info(identifier, row, tables):
    source_rows = {key.casefold(): value for key, value in tables.get('sources', {}).items()}
    source_keys = source_rows.get(row.get('OutputSourceID', '').casefold(), {}).get('SourceArray', [])
    sources = []
    for source_id in source_keys:
        detail = tables.get('source_details', {}).get(source_id.strip())
        if detail is None:
            raise ValueError('Missing food acquisition detail')
        name = text(detail.get('ContentDesc'))
        if name and name not in sources:
            sources.append(name)
    offers = []
    for goods in tables.get('goods', {}).values():
        if goods['ItemID'].casefold() != identifier.casefold() or goods['PriceType'] != 'PriceShowType::Cost':
            continue
        if goods['CurrencyID'].casefold() != 'fons':
            continue
        price = goods['Price']
        if isinstance(price, bool) or not isinstance(price, (int, float)) or price < 0:
            raise ValueError('Invalid food price')
        shop_id = goods['OwnerShopID']
        shop = text(tables.get('source_details', {}).get(shop_id, {}).get('ContentDesc')) or text(
            tables.get('shops', {}).get(shop_id, {}).get('ShopName'))
        if not shop:
            shop = '商店名称未提供'
        offer = {'shop': shop, 'price': price, 'limited': bool(goods.get('UseLimitStartTime') or goods.get('UseLimitEndTime')),
                 'requires_unlock': goods.get('UnLockID', 'None') != 'None'}
        if offer not in offers:
            offers.append(offer)
    return {'sources': sources, 'offers': offers,
            'fons_price': min((offer['price'] for offer in offers), default=None)}

GROUPS = ('Attack', 'Defense', 'AddHPInstant', 'AddHPConstant', 'Revive', 'Explore', 'Special')


def build_foods(tables):
    result = []
    for identifier, row in tables['foods'].items():
        effect_type = row['ItemEffectType'].split('::')[-1]
        if effect_type not in tables['types'] or effect_type not in GROUPS:
            raise ValueError('Unknown food function')
        name, effect = text(row['ItemName']), text(row['UseContext'])
        if not name or not effect:
            raise ValueError('Missing food name or effect')
        group = bonus_group(effect, effect_type)
        card_effect = effect.split('。多人游戏时')[0].removeprefix('队伍中所有角色').removeprefix('为当前角色').removeprefix('当前角色')
        result.append({'bonus_group': group, 'bonus_name': BONUS_GROUPS[group],
                       'effect_segments': effect_segments(effect), 'card_segments': effect_segments(card_effect),
                       **purchase_info(identifier, row, tables), 'id': identifier, 'name': name, 'function': effect_type,
                       'function_name': text(tables['types'][effect_type]['EffectTypeName']),
                       'effect': effect, 'description': text(row['Description']),
                       'quality': row['ItemQuality'].split('_')[-1].lower(),
                       'cooldown': row['ItemCD'],
                       'image': f'/static/guide/foods/{identifier.lower()}.webp'})
    result.sort(key=lambda r: (list(BONUS_GROUPS).index(r['bonus_group']), r['id']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ntedata-dir', type=Path)
    args = parser.parse_args()
    def load(pair):
        key, path = pair
        raw = ((args.ntedata_dir / Path(path).name).read_bytes() if args.ntedata_dir else
               subprocess.check_output(['gh', 'api', 'repos/mc-ctrl/NTEData/contents/DataTable/' + path,
                                        '-H', 'Accept: application/vnd.github.raw+json']))
        return key, next(x['Rows'] for x in json.loads(raw) if 'Rows' in x)
    tables = dict(map(load, TABLES.items()))
    foods = build_foods(tables)
    target = ROOT / 'app/modules/guide/static/foods'
    target.mkdir(parents=True, exist_ok=True)
    def icon(food):
        path = target / (food['id'].lower() + '.webp')
        if path.is_file():
            return
        asset = tables['foods'][food['id']]['ItemIcon']['AssetPathName'].split('.')[0]
        if not asset.startswith('/Game/'):
            raise ValueError('Invalid food icon')
        data = download('https://api.everness.info/data/assets/' + asset.removeprefix('/Game/UI/') + '.webp')
        if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
            raise ValueError('Invalid food image')
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(data)
        temporary.replace(path)
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(icon, foods))
    path = ROOT / 'app/modules/guide/data/foods.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(foods, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)
    print(f'Synced {len(foods)} foods')


if __name__ == '__main__':
    main()
