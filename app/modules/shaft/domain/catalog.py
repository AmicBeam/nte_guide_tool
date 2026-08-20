from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.errors import AppError


MODULE_ROOT = Path(__file__).resolve().parents[1]
SHAFT_DATA_DIR = MODULE_ROOT / 'static' / 'data'
LEGACY_ARC_SELECTIONS: dict[str, tuple[str, int]] = {
    'arc_18c6a54ab6': ('arc_7d0ca08e3d', 1),
    'arc_57b6c49b66': ('arc_1a476075cd', 5),
    'arc_b3c0b59d01': ('arc_126824ee1', 5),
    'arc_f36fcedbc8': ('arc_8b67b6e360', 5),
    'arc_5fb06c9645': ('arc_112b3492d8', 5),
    'arc_0a681d8e81': ('arc_b22de80f07', 5),
    'arc_2b6d5881ef': ('arc_27dc4a7281', 5),
    'arc_92353a7626': ('arc_1ddecc32f3', 5),
    'arc_a01731d2ff': ('arc_6e7753edf5', 1),
}
LEGACY_ACTION_MIGRATIONS: dict[str, tuple[str, bool, bool]] = {
    # 旧 a4脱手 -> a4远脱手；旧 a4闪 -> a4远脱手后接通用“闪”。
    'action_97ef5c83d2': ('action_6d2645f71e', True, False),
    'action_182423b934': ('action_6d2645f71e', True, True),
}


def _load_json(filename: str) -> Any:
    path = SHAFT_DATA_DIR / filename
    if not path.exists():
        raise AppError(f'缺少 shaft 数据文件：{filename}')
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _action_has_damage(action: dict[str, Any]) -> bool:
    if str(action.get('damage_type') or '') in {'', '无'}:
        return False
    multipliers = action.get('multipliers') if isinstance(action.get('multipliers'), dict) else {}
    return any(_num(multipliers.get(key)) != 0 for key in ('atk', 'hp', 'def', 'flat'))


def _formula_hit_count(source_formula: Any) -> int:
    formula = str(source_formula or '')
    if not formula:
        return 0
    total = 0
    for match in re.finditer(
        r'\{\d+\}\s*%?\s*(?:攻击力|防御力|生命上限)?\s*(?:\*\s*(\d+(?:\.\d+)?))?',
        formula,
    ):
        total += max(1, int(round(_num(match.group(1), 1))))
    return total


def _normalize_action_tags(action: dict[str, Any]) -> list[str]:
    raw_tags = action.get('tags')
    if isinstance(raw_tags, str):
        tags = {raw_tags}
    elif isinstance(raw_tags, list):
        tags = {str(tag) for tag in raw_tags if str(tag)}
    else:
        tags = set()
    extra_tag = str(action.get('extra_tag') or '')
    if extra_tag:
        tags.add(extra_tag)
    marker = ' '.join(
        str(action.get(key) or '')
        for key in ('name', 'source_action_name', 'extra_tag')
    )
    if '治疗' in marker:
        tags.update({'heal', '治疗'})
    if any(text in marker for text in ('扣血', '降生命', '降低生命')):
        tags.update({'self_hp_loss', 'hp_loss', '扣血', '降低生命'})
    debuff_names = ('延滞', '黯星', '浸染', '覆纹', '浊燃')
    for debuff in debuff_names:
        if debuff in marker:
            tags.update({debuff, f'debuff:{debuff}', f'enemy_debuff:{debuff}'})
    return sorted(tags)


def _normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(action)
    tags = _normalize_action_tags(action)
    if tags:
        normalized['tags'] = tags
    formula_hits = _formula_hit_count(action.get('source_formula'))
    explicit_hit_count = _num(action.get('hit_count'), -1)
    nightmare_stacks = _num(action.get('nightmare_stacks'), -1)
    if explicit_hit_count >= 0:
        hit_count = max(0, int(round(explicit_hit_count)))
    elif formula_hits > 0:
        hit_count = formula_hits
    elif nightmare_stacks >= 0:
        hit_count = max(0, int(round(nightmare_stacks)))
    elif _action_has_damage(action):
        hit_count = 1
    else:
        hit_count = 0
    normalized['hit_count'] = hit_count
    return normalized


def _noop_action(character: dict[str, Any]) -> dict[str, Any]:
    character_id = str(character.get('id') or '')
    return {
        'id': f'action_none_{character_id.removeprefix("char_")}',
        'character_id': character_id,
        'character_name': str(character.get('name') or ''),
        'name': '无',
        'action_type': '无',
        'damage_type': '无',
        'extra_tag': '切人',
        'is_background_damage': False,
        'duration_seconds': 0,
        'duration_ticks': 0,
        'multipliers': {'atk': 0, 'hp': 0, 'def': 0, 'flat': 0},
        'energy_gain': 0,
        'harmony': 0,
        'stagger': 0,
        'energy_return': 0,
        'self_modifiers': {},
        'cooldown_ticks': 0,
        'energy_cost': 0,
        'personal_resource_cost': {},
        'personal_resource_gain': {},
        'source_row': 0,
        'can_background_override': True,
        'hit_count': 0,
        'is_instant_switch': True,
        'tags': ['切人'],
    }


def _dodge_action(character: dict[str, Any]) -> dict[str, Any]:
    character_id = str(character.get('id') or '')
    return {
        'id': f'action_dodge_{character_id.removeprefix("char_")}',
        'character_id': character_id,
        'character_name': str(character.get('name') or ''),
        'name': '闪',
        'action_type': '无',
        'damage_type': '无',
        'extra_tag': '闪避',
        'is_background_damage': False,
        'duration_seconds': 0.5,
        'duration_ticks': 5,
        'multipliers': {'atk': 0, 'hp': 0, 'def': 0, 'flat': 0},
        'energy_gain': 0,
        'harmony': 0,
        'stagger': 0,
        'energy_return': 0,
        'self_modifiers': {},
        'cooldown_ticks': 0,
        'energy_cost': 0,
        'personal_resource_cost': {},
        'personal_resource_gain': {},
        'source_row': 0,
        'can_background_override': False,
        'hit_count': 0,
        'tags': ['闪避'],
    }


@lru_cache(maxsize=1)
def load_shaft_catalog() -> dict[str, Any]:
    characters = _load_json('characters.json')
    actions = [_normalize_action(action) for action in _load_json('actions.json')]
    actions.extend(_dodge_action(character) for character in characters)
    actions.extend(_noop_action(character) for character in characters)
    energy_capacity_by_character: dict[str, float] = {}
    for action in actions:
        if str(action.get('action_type') or '') != 'Q' and str(action.get('damage_type') or '') != 'Q':
            continue
        character_id = str(action.get('character_id') or '')
        energy_capacity_by_character[character_id] = max(
            energy_capacity_by_character.get(character_id, 0),
            _num(action.get('energy_cost')),
        )
    characters = [
        {
            **character,
            'energy_capacity': max(
                0,
                _num(character.get('energy_capacity'), energy_capacity_by_character.get(str(character.get('id') or ''), 0)),
            ),
        }
        for character in characters
    ]
    legacy_arc_ids = LEGACY_ARC_SELECTIONS.keys()
    arcs = [
        arc for arc in _load_json('arcs.json')
        if str(arc.get('id') or '') not in legacy_arc_ids
    ]
    raw_arc_refinements = _load_json('arc_refinements.json')
    arc_refinements = {
        **raw_arc_refinements,
        'arcs': {
            arc_id: record
            for arc_id, record in (raw_arc_refinements.get('arcs') or {}).items()
            if arc_id not in legacy_arc_ids
        },
    }
    awakenings = _load_json('awakenings.json')
    mechanisms = _load_json('mechanisms.json')
    buffs = [
        buff for buff in _load_json('buffs.json')
        if not any(
            provider.get('kind') == 'arc' and provider.get('id') in legacy_arc_ids
            for provider in buff.get('providers') or []
        )
    ]
    cartridges = _load_json('cartridges.json')
    formula_constants = _load_json('formula_constants.json')
    source_meta = _load_json('source_meta.json')
    starter_axis = _load_json('starter_axis.json')
    actions_by_character: dict[str, list[dict[str, Any]]] = {}
    for action in actions:
        actions_by_character.setdefault(str(action.get('character_id') or ''), []).append(action)
    for items in actions_by_character.values():
        items.sort(key=lambda item: (
            str(item.get('name') or '') == '无',
            str(item.get('name') or '') == '闪',
            str(item.get('action_type') or ''),
            str(item.get('name') or ''),
        ))
    return {
        'characters': characters,
        'actions': actions,
        'actions_by_character': actions_by_character,
        'arcs': arcs,
        'arc_refinements': arc_refinements,
        'awakenings': awakenings,
        'mechanisms': mechanisms,
        'buffs': buffs,
        'cartridges': cartridges,
        'formula_constants': formula_constants,
        'source_meta': source_meta,
        'starter_axis': starter_axis,
        'legacy_action_migrations': {
            action_id: {
                'action_id': migration[0],
                'detached': migration[1],
                'append_dodge': migration[2],
            }
            for action_id, migration in LEGACY_ACTION_MIGRATIONS.items()
        },
    }


def get_record_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get('id') or ''): record for record in records}
