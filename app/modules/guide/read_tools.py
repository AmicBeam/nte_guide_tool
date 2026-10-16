"""Bounded Wiki reads used by authenticated chat and the standalone CLI APIs."""
from typing import Any

from . import service, cultivation, user_data, arc_cultivation


def axis_characters(reader: Any, player: Any, values: list[str]) -> list[str]:
    if len(values) > 4:
        raise ValueError('最多按4名角色筛选。')
    records = reader.characters(player)
    guides = {str(row['id']): row['name'] for row in service.catalog()['characters']}
    result = []
    for value in values:
        name = '主角' if value in ('1046', '1051', '零', '「零」') else guides.get(value, value)
        character = next((c for c in records if c['id'] == value or c['name'] == name), None)
        if character is None:
            raise ValueError('无法识别筛选角色，请使用攻略角色ID、排轴角色ID或完整中文名。')
        result.append(character['id'])
    return list(dict.fromkeys(result))


def axes(reader: Any, player: Any, *, characters: list[str] | None = None, scope: str = 'public',
         query: str = '', sort: str = 'new', page: int = 1, limit: int = 10) -> dict:
    if not 1 <= page <= 200 or not 1 <= limit <= 20:
        raise ValueError('页码须为1–200，每页须为1–20条。')
    payload = reader.search(player, scope=scope, character_ids=axis_characters(reader, player, characters or []),
                            query=query[:80], sort=sort, page=page, limit=limit)
    return {**payload, 'items': [{**row, 'preview_url': f'/shaft/rotation?preview={row["id"]}',
                                'detail_url': f'/api/guide/axes/{row["id"]}'} for row in payload['items']]}


def axis_detail(reader: Any, player: Any, axis_id: int, section: str = 'all') -> dict:
    if axis_id <= 0 or section not in ('all', 'summary', 'axis', 'result'):
        raise ValueError('无效轴ID或章节。')
    value = reader.get(player, axis_id)
    if section == 'summary':
        value = {key: item for key, item in value.items() if key not in ('axis', 'result')}
    elif section != 'all':
        value = {key: value[key] for key in ('id', 'title', 'team', section) if key in value}
    return {**value, 'preview_url': f'/shaft/rotation?preview={axis_id}'}


def my_profiles(player: Any, role_id: str = '', character: str = '') -> dict:
    if player is None:
        raise PermissionError('读取本人配装需要登录。')
    rows = user_data.profiles(player)
    if role_id:
        rows = [row for row in rows if str(row['role_id']) == role_id]
    if character:
        rows = [{**row, 'characters': [c for c in row['characters'] if str(c['id']) == character]}
                for row in rows]
        rows = [row for row in rows if row['characters']]
    return {'data': rows}


def reference_search(query: str = '', kind: str | None = None, limit: int = 10, offset: int = 0,
                     element: str = '', rarity: int | None = None) -> dict:
    if kind is not None and kind not in service.KINDS:
        raise ValueError('未知资料分类。')
    if not 1 <= limit <= 20 or not 0 <= offset <= 10000:
        raise ValueError('查询范围无效。')
    rows = service.search(kind, query=query[:120], element=element, rarity=rarity)
    return {'total': len(rows), 'items': rows[offset:offset + limit],
            'next_offset': offset + limit if offset + limit < len(rows) else None}


def cultivation_result(character: str, values: dict) -> dict:
    record = service.detail('characters', character)
    config = service.catalog()['cultivation']
    state = cultivation.defaults(record, config)
    state.update(values)
    return arc_cultivation.calculate(record, config, record['experience'], state)
