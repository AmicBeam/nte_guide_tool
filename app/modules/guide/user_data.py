"""Private player snapshots, normalization and calculator prefill."""
import json
from datetime import datetime

from peewee import AutoField, CharField, DateTimeField, ForeignKeyField, TextField
from app.models import BaseModel, Player
from app.db import atomic_transaction
from . import service, cultivation


class GuideGameProfile(BaseModel):
    id = AutoField()
    player = ForeignKeyField(Player, backref='guide_game_profiles', on_delete='CASCADE')
    role_id = CharField(max_length=64)
    role_name = CharField(max_length=80, default='')
    characters_json = TextField(default='[]')
    updated_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        indexes = ((('player', 'role_id'), True),)


def bounded_int(value, low=0, high=100000):
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value))
        return number if low <= number <= high else None
    except (ValueError, TypeError):
        return None


def text(value, limit=100):
    return str(value)[:limit] if isinstance(value, (str, int, float)) else ''


def properties(items):
    if not isinstance(items, list):
        return []
    return [{'id': text(x.get('id')), 'name': text(x.get('name')), 'value': text(x.get('value'))}
            for x in items[:100] if isinstance(x, dict)]


def equipment(raw):
    raw = raw if isinstance(raw, dict) else {}
    fork = raw.get('fork') if isinstance(raw.get('fork'), dict) else {}
    suit = raw.get('suit') if isinstance(raw.get('suit'), dict) else {}
    result = {'fork': {k: text(fork.get(k)) for k in ('id', 'name', 'alev', 'blev', 'slev')},
              'suit': {k: text(suit.get(k)) for k in ('id', 'name')}}
    result['fork']['properties'] = properties(fork.get('properties'))
    for group in ('core', 'pie'):
        entries = suit.get(group) if isinstance(suit.get(group), list) else []
        result['suit'][group] = [dict(id=text(x.get('id')), name=text(x.get('name')),
            level=bounded_int(x.get('lev')), main_properties=properties(x.get('mainProperties')),
            properties=properties(x.get('properties'))) for x in entries[:30] if isinstance(x, dict)]
    return result


def normalize(raw):
    if not isinstance(raw, list) or not raw or len(raw) > 200:
        raise ValueError('没有取得角色资料，本次没有覆盖已保存的数据。')
    result, seen = [], set()
    for item in raw:
        if not isinstance(item, dict) or not item.get('id'):
            raise ValueError('角色资料不完整，本次没有覆盖已保存的数据。')
        identifier = text(item['id'], 64)
        if not identifier or bounded_int(item.get('alev'), 1, 100) is None:
            raise ValueError('角色等级或身份不完整，本次没有覆盖已保存的数据。')
        if identifier in seen:
            raise ValueError('角色资料出现重复，本次没有覆盖已保存的数据。')
        seen.add(identifier)
        skills = []
        for group in ('skills', 'citySkills'):
            entries = item.get(group) if isinstance(item.get(group), list) else []
            skills.extend(dict(id=text(x.get('id')), name=text(x.get('name')),
                level=bounded_int(x.get('level'), 0, 30), group=group)
                for x in entries[:30] if isinstance(x, dict))
        result.append(dict(id=identifier, name=text(item.get('name')), level=bounded_int(item.get('alev'), 1, 100),
            awaken_level=bounded_int(item.get('awakenLev'), 0, 6),
            awaken_effects=[text(x) for x in item.get('awakenEffect', [])[:6]] if isinstance(item.get('awakenEffect'), list) else [],
            resonance_level=bounded_int(item.get('slev'), 0, 6),
            affinity_experience=bounded_int(item.get('likeabilitylev'), 0, 100000000),
            properties=properties(item.get('properties')), skills=skills, equipment=equipment(item)))
    return result


def save_profile(player, role, raw):
    characters = normalize(raw)
    with atomic_transaction():
        row, _ = GuideGameProfile.get_or_create(player=player, role_id=role['role_id'])
        row.role_name = role['role_name']
        row.characters_json = json.dumps(characters, ensure_ascii=False)
        row.updated_at = datetime.utcnow()
        row.save()
    return serialize(row)


def serialize(row):
    return dict(role_id=row.role_id, role_name=row.role_name, updated_at=row.updated_at.isoformat() + 'Z',
                characters=json.loads(row.characters_json))


def profiles(player):
    return [serialize(row) for row in GuideGameProfile.select().where(
        GuideGameProfile.player == player).order_by(GuideGameProfile.updated_at.desc())]


def prefill(character, saved):
    values, warnings = {'current_ascended': False}, []
    config = service.catalog()['cultivation']
    caps = config['level_caps'][str(character['id'])]
    if saved.get('level') and saved['level'] <= caps[-1]:
        values['current_level'] = saved['level']
    else:
        warnings.append('角色等级未能确认，请手动填写。')
    if saved.get('level') in caps[:-1]:
        warnings.append('接口未提供可确认的角色突破状态，请选择是否已突破。')
    # Provider skill IDs use lowercase; casing does not change identity.
    # Never map by name, type or list position.
    abilities = {x['id']: x for x in character['abilities']}
    for i, skill in enumerate(cultivation.skill_options(character)):
        group = 'citySkills' if abilities[skill['id']]['type'] == 'city' else 'skills'
        candidates = [x for x in saved['skills'] if x['group'] == group and x['id'].casefold() == skill['id'].casefold()]
        if len(candidates) == 1 and candidates[0]['level'] is not None and skill['minimum'] <= candidates[0]['level'] <= skill['maximum']:
            values[f'current_skill_{i}'] = candidates[0]['level']
        else:
            values[f'current_skill_{i}'] = skill['minimum']
            warnings.append(f'{skill["label"]}未能确认基础等级，已恢复为初始值，请手动填写。')
    from . import arc_cultivation
    fork = saved.get('equipment', {}).get('fork', {})
    arc = next((a for a in arc_cultivation.choices(character)
                if a['id'].casefold() == str(fork.get('id', '')).casefold()), None)
    if arc:
        caps = arc_cultivation.config()['arcs'][arc['id']]['level_caps']
        level = bounded_int(fork.get('alev'), 1, caps[-1])
        if level is not None:
            values.update(arc_id=arc['id'], arc_current_level=level)
            stage = bounded_int(fork.get('blev'), 0, len(caps) - 1)
            before = sum(cap < level for cap in caps[:-1])
            if level in caps[:-1] and stage in (before, before + 1):
                values['arc_current_ascended'] = stage == before + 1
            elif level in caps[:-1]:
                values['arc_current_ascended'] = False
                warnings.append('请确认弧盘是否已突破。')
    warnings.append('同步技能等级可能包含额外加成，请核对为基础等级后计算。')
    return dict(values=values, warnings=warnings)
