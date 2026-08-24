from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from app.db import atomic_transaction
from app.errors import RuleValidationError
from app.models import (
    Player,
    ShaftAxis,
    ShaftAxisCharacter,
    ShaftAxisDislike,
    ShaftAxisFavorite,
    ShaftAxisLike,
    ShaftCharacterPublication,
)
from app.modules.shaft.domain.catalog import (
    LEGACY_ACTION_MIGRATIONS,
    LEGACY_ARC_SELECTIONS,
    REMOVED_ACTION_IDS,
    get_record_map,
    load_shaft_catalog,
)
from app.utils.logger import get_logger


logger = get_logger('nte.shaft')
MAX_AXIS_TITLE_LENGTH = 80
MAX_AXIS_DESCRIPTION_LENGTH = 800
MAX_PRIVATE_AXES_PER_PLAYER = 50
MAX_AXIS_STEPS = 240
MAX_BUFF_RULES = 48
MAX_BACKGROUND_ACTION_MULTIPLIER = 999
VISIBILITIES = frozenset({'private', 'public'})
MARKET_SORTS = frozenset({'dps', 'likes', 'favorites', 'new'})
DEFAULT_UNPUBLISHED_CHARACTERS = {
    'char_0846d632e0': '灵可',
}
HALF_OPEN_CHARACTERS = {}
RELEASED_CHARACTERS = {
    'char_a01c39f576': '伊洛伊',
    'char_076a1f4e53': '残虹',
}
ELEMENTS = ('光', '灵', '咒', '暗', '魂', '相')
ZERO_ACTION_VISUAL_TICKS = 5
SUBSTAT_KEYS = (
    'all_dmg',
    'crit_rate',
    'crit_dmg',
    'harmony_strength',
    'stagger_strength',
    'stagger_multiplier',
    'atk_pct',
    'flat_atk',
    'hp_pct',
    'flat_hp',
    'def_pct',
    'flat_def',
)
MODIFIER_KEYS = (
    'all_dmg',
    'other_dmg',
    'crit_rate',
    'crit_dmg',
    'atk_pct',
    'flat_atk',
    'hp_pct',
    'flat_hp',
    'def_pct',
    'flat_def',
    'def_down',
    'def_ignore',
    'res_down',
    'energy_recharge',
    'harmony_strength',
    'stagger_strength',
    'basic_dmg',
    'dodge_counter_dmg',
    'element_dmg',
    'follow_dmg',
    'mind_dmg',
    'attach_dmg',
    'skill_dmg',
    'ultimate_dmg',
    'final_dmg',
)
TEAM_PANEL_BONUS_DEFAULTS = {
    'version': 3.0,
    'furniture_crit_dmg': 0.04,
    'furniture_flat_atk': 20.0,
    'furniture_flat_def': 30.0,
    'small_flat_atk': 420.0,
    'small_flat_hp': 5200.0,
}
SKILL_LEVEL_KEYS = ('basic', 'skill', 'ultimate', 'support')
SKILL_LEVEL_DEFAULTS = {
    'basic': 10,
    'skill': 10,
    'ultimate': 10,
    'support': 10,
}
CURTAIN_PASSIVE_TYPES = ('type2', 'type3', 'type4')
CANHONG_CHARACTER_ID = 'char_076a1f4e53'
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHAFT_COMPUTE_SCRIPT = PROJECT_ROOT / 'scripts' / 'shaft_compute.js'
SHAFT_SOURCE_VERSION = '异环云配队 1.0.4'


class ShaftAxisNameConflictError(RuleValidationError):
    def __init__(self, title: str, axis_id: int):
        super().__init__(f'已存在名为「{title}」的排轴。')
        self.title = title
        self.axis_id = axis_id


def _ensure_private_axis_capacity(player: Player) -> None:
    private_axis_count = ShaftAxis.select().where(
        (ShaftAxis.owner == player) &
        (ShaftAxis.visibility == 'private')
    ).count()
    if private_axis_count >= MAX_PRIVATE_AXES_PER_PLAYER:
        raise RuleValidationError('每个账号最多创建 50 个排轴，请删除不需要的排轴后再试。')


def initialize_shaft_character_publications() -> None:
    now = datetime.utcnow()
    with atomic_transaction():
        for character_id, character_name in DEFAULT_UNPUBLISHED_CHARACTERS.items():
            publication, created = ShaftCharacterPublication.get_or_create(
                character_id=character_id,
                defaults={
                    'character_name': character_name,
                    'access_level': 'invited' if character_id in HALF_OPEN_CHARACTERS else 'test',
                    'is_published': False,
                    'updated_at': now,
                },
            )
            expected_level = 'invited' if character_id in HALF_OPEN_CHARACTERS else 'test'
            fields_to_update = []
            if publication.character_name != character_name:
                publication.character_name = character_name
                fields_to_update.append(ShaftCharacterPublication.character_name)
            if not publication.is_published and publication.access_level != expected_level:
                publication.access_level = expected_level
                fields_to_update.append(ShaftCharacterPublication.access_level)
            if fields_to_update:
                publication.updated_at = now
                publication.save(only=[
                    *fields_to_update,
                    ShaftCharacterPublication.updated_at,
                ])
        for character_id, character_name in RELEASED_CHARACTERS.items():
            publication, _ = ShaftCharacterPublication.get_or_create(
                character_id=character_id,
                defaults={
                    'character_name': character_name,
                    'access_level': 'public',
                    'is_published': True,
                    'updated_at': now,
                },
            )
            fields_to_update = []
            if publication.character_name != character_name:
                publication.character_name = character_name
                fields_to_update.append(ShaftCharacterPublication.character_name)
            if not publication.is_published:
                publication.is_published = True
                fields_to_update.append(ShaftCharacterPublication.is_published)
            if publication.access_level != 'public':
                publication.access_level = 'public'
                fields_to_update.append(ShaftCharacterPublication.access_level)
            if fields_to_update:
                publication.updated_at = now
                publication.save(only=[
                    *fields_to_update,
                    ShaftCharacterPublication.updated_at,
                ])


def _character_access_levels() -> dict[str, str]:
    if not ShaftCharacterPublication.table_exists():
        return {
            character_id: ('invited' if character_id in HALF_OPEN_CHARACTERS else 'test')
            for character_id in DEFAULT_UNPUBLISHED_CHARACTERS
        }
    existing_columns = {
        column.name for column in ShaftCharacterPublication._meta.database.get_columns(
            ShaftCharacterPublication._meta.table_name
        )
    }
    has_access_level = 'access_level' in existing_columns
    selected_fields = [
        ShaftCharacterPublication.character_id,
        ShaftCharacterPublication.is_published,
    ]
    if has_access_level:
        selected_fields.append(ShaftCharacterPublication.access_level)
    publication_states = {}
    for publication in ShaftCharacterPublication.select(*selected_fields):
        if publication.is_published:
            access_level = 'public'
        elif has_access_level:
            access_level = str(publication.access_level or 'test')
        else:
            access_level = (
                'invited' if publication.character_id in HALF_OPEN_CHARACTERS else 'test'
            )
        publication_states[publication.character_id] = access_level
    for character_id in DEFAULT_UNPUBLISHED_CHARACTERS:
        publication_states.setdefault(
            character_id,
            'invited' if character_id in HALF_OPEN_CHARACTERS else 'test',
        )
    for character_id in RELEASED_CHARACTERS:
        publication_states[character_id] = 'public'
    return publication_states


def _unpublished_character_ids() -> frozenset[str]:
    return frozenset(
        character_id
        for character_id, access_level in _character_access_levels().items()
        if access_level != 'public'
    )


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _safe_json_loads(payload: str, fallback: Any) -> Any:
    try:
        return json.loads(payload)
    except (TypeError, ValueError):
        return fallback


def _clean_text(value: Any, max_length: int) -> str:
    text = re.sub(r'\s+', ' ', str(value or '').strip())
    return text[:max_length]


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    return int(round(_num(value, default)))


def _axis_body(payload: dict[str, Any]) -> dict[str, Any]:
    body = payload.get('axis')
    if isinstance(body, dict):
        return dict(body)
    return dict(payload)


def _normalize_enemy(raw: Any, catalog: dict[str, Any]) -> dict[str, Any]:
    defaults = catalog['formula_constants'].get('default_enemy') or {}
    enemy = raw if isinstance(raw, dict) else {}
    weakness = enemy.get('weakness_elements')
    if not isinstance(weakness, list):
        weakness = defaults.get('weakness_elements') or []
    debuffs = enemy.get('debuffs') if isinstance(enemy.get('debuffs'), dict) else {}
    resistances = enemy.get('resistances') if isinstance(enemy.get('resistances'), dict) else {}
    initial_resistance_raw = enemy.get('initial_resistance')
    initial_resistance = max(
        -1.0,
        min(1.0, _num(initial_resistance_raw, _num(resistances.get('光'), 0.3))),
    )
    hp_ratio = enemy.get('hp_ratio')
    if hp_ratio is None and enemy.get('hp_percent') is not None:
        hp_ratio = _num(enemy.get('hp_percent'), 100) / 100
    return {
        'level': max(1, min(120, _int(enemy.get('level'), _int(defaults.get('level'), 90)))),
        'track_outside': bool(enemy.get('track_outside', defaults.get('track_outside', False))),
        'weakness_elements': [str(item) for item in weakness if str(item) in ELEMENTS],
        'debuffs': {
            str(name): max(0, min(6000, _int(end_tick)))
            for name, end_tick in debuffs.items()
            if str(name) in {'延滞', '黯星', '浸染', '覆纹', '浊燃'}
        },
        'hp_ratio': max(0.0, min(1.0, _num(hp_ratio, 1.0))),
        'initial_resistance': initial_resistance,
        'resistances': {
            element: (
                initial_resistance
                if initial_resistance_raw is not None
                else max(-1.0, min(1.0, _num(resistances.get(element), 0.3)))
            )
            for element in [*ELEMENTS, '心灵']
        },
    }


def _normalize_substat_counts(raw: Any) -> dict[str, int]:
    counts = raw if isinstance(raw, dict) else {}
    normalized: dict[str, int] = {}
    remaining = 120
    for key in SUBSTAT_KEYS:
        value = max(0, min(30, _int(counts.get(key)), remaining))
        normalized[key] = value
        remaining -= value
    return normalized


def _normalize_skill_levels(raw: Any) -> dict[str, int]:
    levels = raw if isinstance(raw, dict) else {}
    return {
        key: max(1, min(10, _int(levels.get(key), SKILL_LEVEL_DEFAULTS[key])))
        for key in SKILL_LEVEL_KEYS
    }


def _default_arc_refinement(arc_id: str, catalog: dict[str, Any]) -> int:
    refinements = catalog.get('arc_refinements') or {}
    arcs = refinements.get('arcs') if isinstance(refinements.get('arcs'), dict) else {}
    record = arcs.get(arc_id) if isinstance(arcs.get(arc_id), dict) else {}
    return max(1, min(5, _int(record.get('default_level'), 1)))


def _normalize_arc_refinement(raw: Any, arc_id: str, catalog: dict[str, Any]) -> int:
    level = _int(raw)
    if 1 <= level <= 5:
        return level
    return _default_arc_refinement(arc_id, catalog)


def _normalize_arc_selection(raw_arc_id: Any, raw_refinement: Any) -> tuple[str, Any]:
    arc_id = str(raw_arc_id or '')
    replacement = LEGACY_ARC_SELECTIONS.get(arc_id)
    if not replacement:
        return arc_id, raw_refinement
    canonical_arc_id, legacy_level = replacement
    return canonical_arc_id, raw_refinement if 1 <= _int(raw_refinement) <= 5 else legacy_level


def _default_build_options(character_id: str, catalog: dict[str, Any]) -> dict[str, Any]:
    constants = catalog.get('formula_constants') or {}
    defaults = constants.get('default_build_options') if isinstance(constants.get('default_build_options'), dict) else {}
    return defaults.get(character_id) if isinstance(defaults.get(character_id), dict) else {}


def _normalize_stat_name(value: Any) -> str:
    text = str(value or '').strip()
    return '精通' if text == '环合强度' else text


def _normalize_cartridge_main_stat(raw: Any, character_id: str, catalog: dict[str, Any]) -> str:
    constants = catalog.get('formula_constants') or {}
    options = constants.get('cartridge_main_stat_options') if isinstance(constants.get('cartridge_main_stat_options'), dict) else {}
    default = _normalize_stat_name(_default_build_options(character_id, catalog).get('cartridge_main_stat')) or next(iter(options), '')
    selected = _normalize_stat_name(raw) or default
    return selected if selected in options else default


def _normalize_curtain_bonus(raw: Any, character_id: str, catalog: dict[str, Any]) -> dict[str, Any]:
    constants = catalog.get('formula_constants') or {}
    stat_options = constants.get('curtain_bonus_stat_options') if isinstance(constants.get('curtain_bonus_stat_options'), dict) else {}
    defaults = _default_build_options(character_id, catalog).get('curtain_bonus')
    default_bonus = defaults if isinstance(defaults, dict) else {}
    source = raw if isinstance(raw, dict) else {}
    default_stat = _normalize_stat_name(default_bonus.get('stat')) or next(iter(stat_options), '')
    stat = (
        default_stat
        if character_id == CANHONG_CHARACTER_ID
        else (_normalize_stat_name(source.get('stat')) or default_stat)
    )
    passive_type = str(source.get('passive_type') or default_bonus.get('passive_type') or 'type3')
    if passive_type not in CURTAIN_PASSIVE_TYPES:
        passive_type = 'type3'
    return {
        'value': max(0.0, min(100.0, _num(default_bonus.get('value')))),
        'stat': stat if stat in stat_options else default_stat,
        'passive_type': passive_type,
    }


def _normalize_awakening_nodes(raw_nodes: Any, legacy_awakening: Any = 0) -> list[int]:
    if isinstance(raw_nodes, list):
        return sorted({
            level
            for value in raw_nodes
            if 1 <= (level := _int(value)) <= 6
        })
    legacy_level = max(0, min(6, _int(legacy_awakening)))
    return list(range(1, legacy_level + 1))


def _normalize_team(raw: Any, catalog: dict[str, Any]) -> list[dict[str, Any]]:
    team = raw if isinstance(raw, list) and raw else catalog['starter_axis']['team']
    characters = get_record_map(catalog['characters'])
    arcs = get_record_map(catalog['arcs'])
    cartridges = get_record_map(catalog['cartridges'])
    normalized: list[dict[str, Any]] = []
    used_slots: set[int] = set()
    for index, member in enumerate(team[:4]):
        if not isinstance(member, dict):
            continue
        slot = max(0, min(3, _int(member.get('slot'), index)))
        if slot in used_slots:
            raise RuleValidationError('队伍中存在重复位置。')
        used_slots.add(slot)
        character_id = str(member.get('character_id') or '')
        if character_id not in characters:
            raise RuleValidationError('队伍中存在未知角色。')
        arc_id, arc_refinement = _normalize_arc_selection(
            member.get('arc_id'),
            member.get('arc_refinement'),
        )
        cartridge_id = str(member.get('cartridge_id') or '')
        if arc_id and arc_id not in arcs:
            raise RuleValidationError('队伍中存在未知弧盘。')
        if cartridge_id and cartridge_id not in cartridges:
            raise RuleValidationError('队伍中存在未知卡带。')
        character = characters[character_id]
        arc = arcs.get(arc_id)
        if arc and str(arc.get('adaptation') or '') != str(character.get('adaptation') or ''):
            raise RuleValidationError('角色与弧盘的适配类型不一致。')
        cartridge = cartridges.get(cartridge_id)
        awakening_nodes = _normalize_awakening_nodes(member.get('awakening_nodes'), member.get('awakening'))
        has_bond_bonus = bool((character.get('bond_bonus') or {}).get('modifiers'))
        normalized.append({
            'slot': slot,
            'character_id': character_id,
            'character_name': character.get('name') or '',
            'arc_id': arc_id,
            'arc_name': (arc or {}).get('name') or '',
            'arc_refinement': _normalize_arc_refinement(arc_refinement, arc_id, catalog),
            'cartridge_id': cartridge_id,
            'cartridge_name': (cartridge or {}).get('name') or '',
            'awakening': len(awakening_nodes),
            'awakening_nodes': awakening_nodes,
            'bond_level': max(0, min(1, _int(member.get('bond_level'), 1 if member.get('bond_full') else 0))) if has_bond_bonus else 0,
            'bond_full': (bool(member.get('bond_full')) or _int(member.get('bond_level')) > 0) if has_bond_bonus else False,
            'skill_levels': _normalize_skill_levels(member.get('skill_levels')),
            'cartridge_main_stat': _normalize_cartridge_main_stat(member.get('cartridge_main_stat'), character_id, catalog),
            'curtain_bonus': _normalize_curtain_bonus(member.get('curtain_bonus'), character_id, catalog),
            'substat_counts': _normalize_substat_counts(member.get('substat_counts')),
        })
    if not normalized:
        raise RuleValidationError('队伍不能为空。')
    normalized.sort(key=lambda item: item['slot'])
    return normalized


def _normalize_character_builds(raw: Any, team: list[dict[str, Any]], catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    characters = get_record_map(catalog['characters'])
    arcs = get_record_map(catalog['arcs'])
    cartridges = get_record_map(catalog['cartridges'])
    source = raw if isinstance(raw, dict) else {}
    normalized: dict[str, dict[str, Any]] = {}

    def normalize_one(character_id: str, value: Any) -> None:
        if character_id not in characters:
            return
        raw_value = value[0] if isinstance(value, list) and value else value
        if isinstance(raw_value, dict) and isinstance(raw_value.get('variants'), list):
            raw_value = raw_value['variants'][0] if raw_value['variants'] else {}
        build = raw_value if isinstance(raw_value, dict) else {}
        arc_id, arc_refinement = _normalize_arc_selection(
            build.get('arc_id'),
            build.get('arc_refinement'),
        )
        arc = arcs.get(arc_id)
        if arc and str(arc.get('adaptation') or '') != str(characters[character_id].get('adaptation') or ''):
            arc_id = ''
        cartridge_id = str(build.get('cartridge_id') or '')
        awakening_nodes = _normalize_awakening_nodes(build.get('awakening_nodes'), build.get('awakening'))
        has_bond_bonus = bool((characters[character_id].get('bond_bonus') or {}).get('modifiers'))
        normalized[character_id] = {
            'character_id': character_id,
            'character_name': characters[character_id].get('name') or '',
            'arc_id': arc_id if arc_id in arcs else '',
            'arc_name': (arcs.get(arc_id) or {}).get('name') or '',
            'arc_refinement': _normalize_arc_refinement(arc_refinement, arc_id, catalog),
            'cartridge_id': cartridge_id if cartridge_id in cartridges else '',
            'cartridge_name': (cartridges.get(cartridge_id) or {}).get('name') or '',
            'awakening': len(awakening_nodes),
            'awakening_nodes': awakening_nodes,
            'bond_level': max(0, min(1, _int(build.get('bond_level'), 1 if build.get('bond_full') else 0))) if has_bond_bonus else 0,
            'bond_full': (bool(build.get('bond_full')) or _int(build.get('bond_level')) > 0) if has_bond_bonus else False,
            'skill_levels': _normalize_skill_levels(build.get('skill_levels')),
            'cartridge_main_stat': _normalize_cartridge_main_stat(build.get('cartridge_main_stat'), character_id, catalog),
            'curtain_bonus': _normalize_curtain_bonus(build.get('curtain_bonus'), character_id, catalog),
            'substat_counts': _normalize_substat_counts(build.get('substat_counts')),
        }

    for character_id, build in source.items():
        normalize_one(str(character_id), build)
    for member in team:
        character_id = str(member.get('character_id') or '')
        if character_id not in normalized:
            normalize_one(character_id, member)
    return normalized


def _apply_character_builds_to_team(team: list[dict[str, Any]], character_builds: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    merged_team: list[dict[str, Any]] = []
    for member in team:
        character_id = str(member.get('character_id') or '')
        build = character_builds.get(character_id)
        if not build:
            merged_team.append(member)
            continue
        merged = dict(member)
        for key in (
            'arc_id',
            'arc_name',
            'arc_refinement',
            'cartridge_id',
            'cartridge_name',
            'awakening',
            'awakening_nodes',
            'bond_level',
            'bond_full',
            'skill_levels',
            'cartridge_main_stat',
            'curtain_bonus',
            'substat_counts',
        ):
            if key in build:
                merged[key] = build[key]
        merged_team.append(merged)
    return merged_team


def _normalize_steps(raw: Any, catalog: dict[str, Any]) -> list[dict[str, Any]]:
    steps = raw if isinstance(raw, list) else catalog['starter_axis']['steps']
    actions = get_record_map(catalog['actions'])
    normalized: list[dict[str, Any]] = []
    for index, step in enumerate(steps[:MAX_AXIS_STEPS]):
        if len(normalized) >= MAX_AXIS_STEPS:
            break
        if not isinstance(step, dict):
            continue
        original_action_id = str(step.get('action_id') or '')
        if original_action_id in REMOVED_ACTION_IDS:
            continue
        migration = LEGACY_ACTION_MIGRATIONS.get(original_action_id)
        action_id = migration[0] if migration else original_action_id
        action = actions.get(action_id)
        if not action:
            raise RuleValidationError('轴中存在未知动作。')
        start_tick = max(0, _int(step.get('start_tick')))
        placement = _normalize_step_placement(step, action)
        repeat = (
            max(1, min(MAX_BACKGROUND_ACTION_MULTIPLIER, _int(step.get('repeat'), 1)))
            if _is_background_action(action)
            else 1
        )
        normalized_step = {
            'id': str(step.get('id') or f'step_{index + 1:03d}')[:40],
            'slot': max(0, min(3, _int(step.get('slot')))),
            'action_id': action_id,
            'action_name': action.get('name') or '',
            'start_tick': start_tick,
            'repeat': repeat,
            'tags': step.get('tags') if isinstance(step.get('tags'), list) else [],
        }
        detached = bool(migration[1]) if migration else bool(step.get('detached'))
        if detached and bool(action.get('can_detach')):
            normalized_step['detached'] = True
        base_duration_ticks = max(0, _int(
            action.get('detached_duration_ticks') if normalized_step.get('detached') else action.get('duration_ticks'),
        ))
        if (
            bool(step.get('interrupted'))
            and base_duration_ticks > 0
            and not _is_support_action(action)
            and not _is_instant_native_background_action(action)
            and action.get('can_interrupt') is not False
        ):
            normalized_step['interrupted'] = True
            normalized_step['interrupt_duration_ticks'] = max(
                1,
                min(base_duration_ticks, _int(step.get('interrupt_duration_ticks'), base_duration_ticks)),
            )
            normalized_step['interrupt_hit_count'] = max(
                0,
                min(max(0, _int(action.get('hit_count'))), _int(step.get('interrupt_hit_count'), _int(action.get('hit_count')))),
            )
        if placement == 'background' and not detached and not _is_background_action(action):
            normalized_step['placement'] = 'background'
        normalized.append(normalized_step)
        if migration and migration[2] and len(normalized) < MAX_AXIS_STEPS:
            dodge_action_id = f'action_dodge_{str(action.get("character_id") or "").removeprefix("char_")}'
            dodge_action = actions.get(dodge_action_id)
            if dodge_action:
                normalized.append({
                    'id': f'{normalized_step["id"][:32]}_dodge',
                    'slot': normalized_step['slot'],
                    'action_id': dodge_action_id,
                    'action_name': '闪',
                    'start_tick': start_tick + max(0, _int(action.get('detached_duration_ticks'), 5)),
                    'repeat': 1,
                    'tags': [],
                })
    normalized.sort(key=lambda item: (item['start_tick'], item['slot'], item['action_id']))
    return normalized


def _axis_duration_ticks(steps: list[dict[str, Any]], catalog: dict[str, Any]) -> int:
    actions = get_record_map(catalog['actions'])
    return calculate_axis_duration_ticks(steps, actions)


def _normalize_options(raw: Any, catalog: dict[str, Any], team: list[dict[str, Any]]) -> dict[str, Any]:
    options = raw if isinstance(raw, dict) else {}
    switch_gap_ticks = max(
        0,
        min(
            20,
            _int(
                options.get('switch_gap_ticks'),
                _int(options.get('switch_loss_ticks'), _int(catalog['formula_constants'].get('switch_loss_ticks'), 2)),
            ),
        ),
    )
    raw_loop_resources = options.get('loop_initial_resources')
    raw_loop_resources = raw_loop_resources if isinstance(raw_loop_resources, dict) else {}
    characters = get_record_map(catalog['characters'])
    formula_constants = catalog.get('formula_constants') if isinstance(catalog.get('formula_constants'), dict) else {}
    personal_resource_caps = formula_constants.get('personal_resource_caps')
    personal_resource_caps = personal_resource_caps if isinstance(personal_resource_caps, dict) else {}
    sustained_reactions = {
        str(option.get('id') or '')
        for option in formula_constants.get('loop_initial_reaction_options', [])
        if isinstance(option, dict)
        and str(option.get('id') or '')
        and _int(option.get('duration_ticks')) > 0
    }
    hidden_personal_resources = {
        str(name)
        for name in formula_constants.get('hidden_personal_resources', [])
        if str(name)
    }
    personal_resource_names: dict[str, set[str]] = {}
    for action in catalog.get('actions', []):
        if not isinstance(action, dict):
            continue
        character_id = str(action.get('character_id') or '')
        names = personal_resource_names.setdefault(character_id, set())
        for field in ('personal_resource_cost', 'personal_resource_gain', 'personal_resource_threshold'):
            values = action.get(field)
            if isinstance(values, dict):
                names.update(str(name) for name in values if str(name) not in hidden_personal_resources)
    loop_initial_resources: dict[str, dict[str, Any]] = {}
    for member in team:
        character_id = str(member.get('character_id') or '')
        configured = raw_loop_resources.get(character_id)
        if not isinstance(configured, dict):
            continue
        energy_capacity = max(0, _num((characters.get(character_id) or {}).get('energy_capacity'), 100))
        configured_personal = configured.get('personal_resources')
        configured_personal = configured_personal if isinstance(configured_personal, dict) else {}
        configured_dot_layers = configured.get('dot_layers')
        configured_dot_layers = configured_dot_layers if isinstance(configured_dot_layers, dict) else {}
        character_caps = personal_resource_caps.get(character_id)
        character_caps = character_caps if isinstance(character_caps, dict) else {}
        normalized_personal = {}
        for name in sorted(personal_resource_names.get(character_id, set())):
            if name not in configured_personal:
                continue
            maximum = max(0, _num(character_caps.get(name), 1_000_000))
            normalized_personal[name] = max(0, min(maximum, _num(configured_personal.get(name))))
        loop_initial_resources[character_id] = {
            'energy': max(0, min(energy_capacity, _num(configured.get('energy')))),
            'harmony': max(0, min(100, _num(configured.get('harmony')))),
            'reaction': str(configured.get('reaction') or '')
            if str(configured.get('reaction') or '') in sustained_reactions else '',
            'personal_resources': normalized_personal,
        }
        if character_id == 'char_076a1f4e53':
            loop_initial_resources[character_id]['dot_layers'] = {
                name: max(0, min(10, _int(configured_dot_layers.get(name))))
                for name in ('蚀心', '鸩火')
            }
    return {
        'switch_gap_ticks': switch_gap_ticks,
        'switch_loss_ticks': switch_gap_ticks,
        'loop_enabled': bool(options.get('loop_enabled')),
        'loop_initial_resources': loop_initial_resources,
    }


def _team_panel_bonus_defaults(catalog: dict[str, Any]) -> dict[str, float]:
    defaults = dict(TEAM_PANEL_BONUS_DEFAULTS)
    starter_bonus = (catalog.get('starter_axis') or {}).get('team_panel_bonus')
    if isinstance(starter_bonus, dict):
        for key in defaults:
            defaults[key] = _num(starter_bonus.get(key), defaults[key])
    defaults['furniture_crit_dmg'] = TEAM_PANEL_BONUS_DEFAULTS['furniture_crit_dmg']
    defaults['furniture_flat_atk'] = TEAM_PANEL_BONUS_DEFAULTS['furniture_flat_atk']
    defaults['furniture_flat_def'] = TEAM_PANEL_BONUS_DEFAULTS['furniture_flat_def']
    return defaults


def _normalize_team_panel_bonus(raw: Any, catalog: dict[str, Any]) -> dict[str, float]:
    defaults = _team_panel_bonus_defaults(catalog)
    source = raw if isinstance(raw, dict) else {}
    version = max(0, _int(source.get('version')))
    legacy_furniture = version < int(TEAM_PANEL_BONUS_DEFAULTS['version'])
    furniture_crit_dmg = defaults['furniture_crit_dmg'] if legacy_furniture else round(max(
        0.0,
        min(defaults['furniture_crit_dmg'], _num(source.get('furniture_crit_dmg'), defaults['furniture_crit_dmg'])),
    ), 3)
    furniture_flat_atk = defaults['furniture_flat_atk'] if legacy_furniture else float(max(
        0,
        min(int(defaults['furniture_flat_atk']), _int(source.get('furniture_flat_atk'), int(defaults['furniture_flat_atk']))),
    ))
    furniture_flat_def = defaults['furniture_flat_def'] if legacy_furniture else float(max(
        0,
        min(int(defaults['furniture_flat_def']), _int(source.get('furniture_flat_def'), int(defaults['furniture_flat_def']))),
    ))
    small_flat_atk = max(0.0, min(5000.0, _num(source.get('small_flat_atk'), defaults['small_flat_atk'])))
    if version < 2 and furniture_flat_atk == 20.0 and small_flat_atk == 440.0:
        small_flat_atk = 420.0
    return {
        'version': 3,
        'furniture_crit_dmg': furniture_crit_dmg,
        'furniture_flat_atk': furniture_flat_atk,
        'furniture_flat_def': furniture_flat_def,
        'small_flat_atk': small_flat_atk,
        'small_flat_hp': max(0.0, min(100000.0, _num(source.get('small_flat_hp'), defaults['small_flat_hp']))),
    }


def _is_background_action(action: dict[str, Any]) -> bool:
    marker_text = f'{action.get("name") or ""} {action.get("extra_tag") or ""}'
    return bool(action.get('is_background_damage')) or '后台' in marker_text


def _is_basic_action(action: dict[str, Any]) -> bool:
    return str(action.get('action_type') or '') == '普攻' or str(action.get('damage_type') or '') == '普攻'


def _is_support_action(action: dict[str, Any]) -> bool:
    return str(action.get('action_type') or '') == '援护'


def _is_instant_switch_action(action: dict[str, Any]) -> bool:
    return bool(action.get('is_instant_switch'))


def _is_instant_native_background_action(action: dict[str, Any]) -> bool:
    return _is_background_action(action) and not _is_support_action(action) and not bool(action.get('pre_input_node'))


def _can_background_override(action: dict[str, Any]) -> bool:
    return bool(action.get('can_background_override')) and (
        _is_basic_action(action) or _is_instant_switch_action(action)
    )


def _normalize_step_placement(step: dict[str, Any], action: dict[str, Any]) -> str:
    if _is_background_action(action):
        return 'background'
    return 'background' if _can_background_override(action) and str(step.get('placement') or '') == 'background' else 'foreground'


def _is_q_action(action: dict[str, Any]) -> bool:
    return str(action.get('action_type') or '') == 'Q' or str(action.get('damage_type') or '') == 'Q'


def _is_zero_foreground_q_step(step: dict[str, Any], action: dict[str, Any]) -> bool:
    return (
        _normalize_step_placement(step, action) != 'background' and
        _is_q_action(action) and
        max(0, _int(action.get('duration_ticks'))) == 0
    )


def _action_duration_ticks(step: dict[str, Any], action: dict[str, Any]) -> int:
    if bool(step.get('detached')) and bool(action.get('can_detach')):
        base_duration_ticks = max(0, _int(action.get('detached_duration_ticks')))
    else:
        base_duration_ticks = max(0, _int(action.get('duration_ticks')))
    if (
        bool(step.get('interrupted'))
        and base_duration_ticks > 0
        and not _is_support_action(action)
        and not _is_instant_native_background_action(action)
        and action.get('can_interrupt') is not False
    ):
        return max(1, min(base_duration_ticks, _int(step.get('interrupt_duration_ticks'), base_duration_ticks)))
    return base_duration_ticks


def calculate_axis_duration_ticks(steps: list[dict[str, Any]], actions_by_id: dict[str, dict[str, Any]]) -> int:
    last_tick = 0
    for step in steps:
        action = actions_by_id.get(str(step.get('action_id') or '')) or {}
        if (
            _normalize_step_placement(step, action) == 'background' and
            not _is_instant_switch_action(action)
        ):
            continue
        start_tick = max(0, _int(step.get('start_tick')))
        duration_ticks = _action_duration_ticks(step, action)
        if _is_zero_foreground_q_step(step, action):
            visual_duration_ticks = ZERO_ACTION_VISUAL_TICKS
        else:
            visual_duration_ticks = duration_ticks
        last_tick = max(last_tick, start_tick + visual_duration_ticks, start_tick)
    return max(0, last_tick)


def _normalize_modifiers(raw: Any) -> dict[str, float]:
    source = raw if isinstance(raw, dict) else {}
    return {
        key: max(-5.0, min(10.0, _num(source.get(key))))
        for key in MODIFIER_KEYS
        if key in source and _num(source.get(key)) != 0
    }


def _normalize_buff_rules(raw: Any, team: list[dict[str, Any]], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    rules = raw if isinstance(raw, list) else []
    actions = get_record_map(catalog['actions'])
    valid_slots = {member['slot'] for member in team}
    normalized: list[dict[str, Any]] = []
    for index, rule in enumerate(rules[:MAX_BUFF_RULES]):
        if not isinstance(rule, dict):
            continue
        trigger = rule.get('trigger') if isinstance(rule.get('trigger'), dict) else {}
        targets = rule.get('targets') if isinstance(rule.get('targets'), dict) else {}
        trigger_action_id = str(trigger.get('action_id') or '')
        target_action_ids = [
            str(action_id)
            for action_id in (targets.get('action_ids') if isinstance(targets.get('action_ids'), list) else [])
            if str(action_id) in actions
        ]
        trigger_slot = trigger.get('slot')
        target_slots = [
            slot for slot in (max(0, min(3, _int(item))) for item in (targets.get('slots') if isinstance(targets.get('slots'), list) else []))
            if slot in valid_slots
        ]
        normalized.append({
            'id': str(rule.get('id') or f'buff_{index + 1:03d}')[:40],
            'name': _clean_text(rule.get('name') or f'增益 {index + 1}', 60),
            'trigger': {
                'slot': max(0, min(3, _int(trigger_slot))) if trigger_slot not in (None, '') else None,
                'action_id': trigger_action_id if trigger_action_id in actions else '',
                'action_type': _clean_text(trigger.get('action_type'), 24),
            },
            'targets': {
                'slots': sorted(set(target_slots)),
                'action_ids': target_action_ids,
                'action_types': [
                    _clean_text(item, 24)
                    for item in (targets.get('action_types') if isinstance(targets.get('action_types'), list) else [])
                    if _clean_text(item, 24)
                ],
            },
            'delay_ticks': max(0, min(600, _int(rule.get('delay_ticks')))),
            'duration_ticks': max(0, min(6000, _int(rule.get('duration_ticks'), 100))),
            'modifiers': _normalize_modifiers(rule.get('modifiers')),
        })
    return [rule for rule in normalized if rule['duration_ticks'] > 0 and rule['modifiers']]


def _validate_step_team_actions(team: list[dict[str, Any]], steps: list[dict[str, Any]], catalog: dict[str, Any]) -> None:
    actions = get_record_map(catalog['actions'])
    character_by_slot = {member['slot']: member['character_id'] for member in team}
    for step in steps:
        if _int(step.get('slot')) not in character_by_slot:
            raise RuleValidationError('轴中存在不在队伍中的角色位置。')
        action = actions.get(str(step.get('action_id') or '')) or {}
        if str(action.get('character_id') or '') != character_by_slot[_int(step.get('slot'))]:
            raise RuleValidationError('轴中存在与角色不匹配的动作。')


def normalize_axis_payload(payload: dict[str, Any]) -> dict[str, Any]:
    catalog = load_shaft_catalog()
    body = _axis_body(payload)
    team = _normalize_team(body.get('team'), catalog)
    character_builds = _normalize_character_builds(body.get('character_builds'), team, catalog)
    team = _apply_character_builds_to_team(team, character_builds)
    steps = _normalize_steps(body.get('steps'), catalog)
    _validate_step_team_actions(team, steps, catalog)
    axis_payload = {
        'team': team,
        'character_builds': character_builds,
        'steps': steps,
        'enemy': _normalize_enemy(body.get('enemy'), catalog),
        'team_panel_bonus': _normalize_team_panel_bonus(body.get('team_panel_bonus'), catalog),
        'options': _normalize_options(body.get('options'), catalog, team),
        'duration_ticks': _axis_duration_ticks(steps, catalog),
        'initial_energy': max(0, min(1000, _num(body.get('initial_energy'), 1000))),
        'buff_rules': _normalize_buff_rules(body.get('buff_rules'), team, catalog),
    }
    return axis_payload


def _is_shaft_test_player(player: Player | None) -> bool:
    return bool(player and getattr(player, 'shaft_test_whitelisted', False))


def shaft_player_access_level(player: Player | None) -> str:
    if _is_shaft_test_player(player):
        return 'test'
    if player and getattr(player, 'shaft_invited', False):
        return 'invited'
    return 'public'


def _character_is_accessible(
    character_id: str,
    player: Player | None,
    access_levels: dict[str, str] | None = None,
) -> bool:
    required_level = (access_levels or _character_access_levels()).get(character_id, 'public')
    player_level = shaft_player_access_level(player)
    access_rank = {'public': 0, 'invited': 1, 'test': 2}
    return access_rank.get(player_level, 0) >= access_rank.get(required_level, 2)


def _team_contains_disabled_character(team: Any, player: Player | None = None) -> bool:
    return isinstance(team, list) and any(
        isinstance(member, dict)
        and not _character_is_accessible(str(member.get('character_id') or ''), player)
        for member in team
    )


def _axis_contains_disabled_character(axis: ShaftAxis, player: Player | None = None) -> bool:
    return _team_contains_disabled_character(_safe_json_loads(axis.team_json, []), player)


def _filter_visible_character_axes(query, player: Player | None):
    access_levels = _character_access_levels()
    inaccessible_character_ids = {
        character_id
        for character_id in access_levels
        if not _character_is_accessible(character_id, player, access_levels)
    }
    if not inaccessible_character_ids:
        return query
    restricted_axis_ids = ShaftAxisCharacter.select(ShaftAxisCharacter.axis).where(
        ShaftAxisCharacter.character_id.in_(inaccessible_character_ids)
    )
    return query.where(ShaftAxis.id.not_in(restricted_axis_ids))


def get_shaft_catalog_payload(player: Player | None = None) -> dict[str, Any]:
    catalog = load_shaft_catalog()
    access_levels = _character_access_levels()
    return {
        'characters': [
            {
                **character,
                'selection_disabled': (
                    not _character_is_accessible(
                        str(character.get('id') or ''), player, access_levels
                    )
                ),
            }
            for character in catalog['characters']
        ],
        'actions': catalog['actions'],
        'actions_by_character': catalog['actions_by_character'],
        'arcs': catalog['arcs'],
        'arc_refinements': catalog['arc_refinements'],
        'awakenings': catalog['awakenings'],
        'mechanisms': catalog['mechanisms'],
        'buffs': catalog['buffs'],
        'cartridges': catalog['cartridges'],
        'formula_constants': catalog['formula_constants'],
        'source_meta': catalog['source_meta'],
        'starter_axis': catalog['starter_axis'],
        'legacy_action_migrations': catalog['legacy_action_migrations'],
    }


def _simulate_axis_with_js(axis_payload: dict[str, Any]) -> dict[str, Any]:
    node_bin = shutil.which('node')
    if not node_bin:
        raise RuleValidationError('服务器未配置 Node.js，无法执行排轴计算。')
    request_payload = _json_dumps({
        'axis': axis_payload,
        'catalog': get_shaft_catalog_payload(),
    })
    try:
        completed = subprocess.run(
            [node_bin, str(SHAFT_COMPUTE_SCRIPT)],
            input=request_payload,
            text=True,
            capture_output=True,
            cwd=str(PROJECT_ROOT),
            timeout=12,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuleValidationError('排轴计算超时，请减少动作数量后重试。') from exc
    except OSError as exc:
        raise RuleValidationError('排轴计算引擎启动失败。') from exc
    output = (completed.stdout or '').strip()
    try:
        payload = json.loads(output) if output else {}
    except ValueError as exc:
        logger.error('shaft js compute invalid output returncode=%s stderr=%s stdout=%s', completed.returncode, completed.stderr, completed.stdout)
        raise RuleValidationError('排轴计算引擎返回了无效结果。') from exc
    if completed.returncode != 0 or not payload.get('ok'):
        error = str(payload.get('error') or (completed.stderr or '').strip() or '排轴计算失败。')
        raise RuleValidationError(error)
    result = payload.get('result')
    if not isinstance(result, dict):
        raise RuleValidationError('排轴计算引擎未返回结果。')
    return result


def simulate_shaft_axis(payload: dict[str, Any]) -> dict[str, Any]:
    axis_payload = normalize_axis_payload(payload)
    result = _simulate_axis_with_js(axis_payload)
    return {
        'axis': axis_payload,
        'result': result,
    }


def _refresh_axis_character_index(axis: ShaftAxis, team: list[dict[str, Any]]) -> None:
    ShaftAxisCharacter.delete().where(ShaftAxisCharacter.axis == axis).execute()
    for member in team:
        ShaftAxisCharacter.create(
            axis=axis,
            character_id=str(member.get('character_id') or ''),
            slot=_int(member.get('slot')),
        )


def _axis_query_for_player(axis_id: int, player: Player) -> ShaftAxis | None:
    return ShaftAxis.select().where(
        (ShaftAxis.id == axis_id) &
        (ShaftAxis.owner == player)
    ).first()


def _private_axis_query_for_player(axis_id: int, player: Player) -> ShaftAxis | None:
    return ShaftAxis.select().where(
        (ShaftAxis.id == axis_id) &
        (ShaftAxis.owner == player) &
        (ShaftAxis.visibility == 'private')
    ).first()


def _visible_axis(axis_id: int, player: Player | None = None) -> ShaftAxis:
    axis = ShaftAxis.select(ShaftAxis, Player).join(Player).where(ShaftAxis.id == axis_id).first()
    if axis is None:
        raise RuleValidationError('排轴不存在。')
    if axis.visibility != 'public' and (player is None or axis.owner_id != player.id):
        raise RuleValidationError('没有查看这个排轴的权限。')
    if (
        axis.visibility == 'public'
        and _axis_contains_disabled_character(axis, player)
    ):
        raise RuleValidationError('排轴不存在。')
    return axis


def _is_liked(axis: ShaftAxis, player: Player | None) -> bool:
    if player is None:
        return False
    return ShaftAxisLike.select().where(
        (ShaftAxisLike.axis == axis) &
        (ShaftAxisLike.player == player)
    ).exists()


def _is_disliked(axis: ShaftAxis, player: Player | None) -> bool:
    if player is None:
        return False
    return ShaftAxisDislike.select().where(
        (ShaftAxisDislike.axis == axis) &
        (ShaftAxisDislike.player == player)
    ).exists()


def _favorite_query(axis: ShaftAxis, player: Player | None, visitor_key: str):
    if player is not None:
        return ShaftAxisFavorite.select().where(
            (ShaftAxisFavorite.axis == axis) &
            (ShaftAxisFavorite.player == player)
        )
    return ShaftAxisFavorite.select().where(
        (ShaftAxisFavorite.axis == axis) &
        (ShaftAxisFavorite.player.is_null(True)) &
        (ShaftAxisFavorite.visitor_key == visitor_key)
    )


def _is_favorited(axis: ShaftAxis, player: Player | None, visitor_key: str) -> bool:
    visitor_key = _clean_visitor_key(visitor_key)
    if player is None and not visitor_key:
        return False
    return _favorite_query(axis, player, visitor_key).exists()


def _clean_visitor_key(value: Any) -> str:
    key = re.sub(r'[^a-zA-Z0-9_.:-]+', '', str(value or '').strip())
    return key[:96]


def _enrich_team(team: list[dict[str, Any]]) -> list[dict[str, Any]]:
    catalog = load_shaft_catalog()
    characters = get_record_map(catalog['characters'])
    arcs = get_record_map(catalog['arcs'])
    cartridges = get_record_map(catalog['cartridges'])
    enriched = []
    for member in team:
        item = dict(member)
        character = characters.get(str(item.get('character_id') or '')) or {}
        arc = arcs.get(str(item.get('arc_id') or '')) or {}
        cartridge = cartridges.get(str(item.get('cartridge_id') or '')) or {}
        item['character_name'] = character.get('name') or item.get('character_name') or ''
        item['character_avatar'] = character.get('avatar') or ''
        item['character_element'] = character.get('element') or ''
        item['arc_name'] = arc.get('name') or item.get('arc_name') or ''
        item['cartridge_name'] = cartridge.get('name') or item.get('cartridge_name') or ''
        enriched.append(item)
    return enriched


def serialize_shaft_axis(
    axis: ShaftAxis,
    *,
    include_axis: bool = False,
    player: Player | None = None,
    visitor_key: str = '',
) -> dict[str, Any]:
    team = _safe_json_loads(axis.team_json, [])
    axis_payload = _safe_json_loads(axis.axis_json, {})
    result = _safe_json_loads(axis.result_json, {})
    summary = result.get('summary') if isinstance(result, dict) and isinstance(result.get('summary'), dict) else {}
    harmony_damage = summary.get('harmony_damage')
    if harmony_damage is None:
        harmony_sources = {'创生', '创生复制体', '覆纹', '浊燃', '黯星'}
        harmony_damage = sum(
            _num(item.get('damage'))
            for item in (result.get('damage_by_source') or [])
            if isinstance(item, dict) and str(item.get('source') or '') in harmony_sources
        ) if isinstance(result, dict) else 0
    duration_seconds = summary.get('duration_seconds')
    if duration_seconds is None:
        duration_seconds = axis.duration_ticks / 10
    payload = {
        'id': axis.id,
        'title': axis.title,
        'description': axis.description,
        'visibility': axis.visibility,
        'owner': {
            'player_uid': axis.owner.player_uid,
            'nickname': axis.owner.nickname or axis.owner.player_uid,
        },
        'team': _enrich_team(team if isinstance(team, list) else []),
        'enemy': _safe_json_loads(axis.enemy_json, {}),
        'summary': summary,
        'duration_ticks': axis.duration_ticks,
        'duration_seconds': _num(duration_seconds),
        'loop_enabled': bool(
            isinstance(axis_payload, dict)
            and isinstance(axis_payload.get('options'), dict)
            and axis_payload['options'].get('loop_enabled')
        ),
        'direct_damage': axis.direct_damage,
        'harmony_damage': _int(harmony_damage),
        'stagger_damage': axis.stagger_damage,
        'total_damage': axis.total_damage,
        'dps': axis.dps_x100 / 100,
        'like_count': axis.like_count,
        'dislike_count': axis.dislike_count,
        'favorite_count': axis.favorite_count,
        'liked': _is_liked(axis, player),
        'disliked': _is_disliked(axis, player),
        'favorited': _is_favorited(axis, player, visitor_key),
        'is_owner': player is not None and axis.owner_id == player.id,
        'dedupe_hash': axis.dedupe_hash,
        'source_version': axis.source_version,
        'created_at': axis.created_at.isoformat(),
        'updated_at': axis.updated_at.isoformat(),
        'published_at': axis.published_at.isoformat() if axis.published_at else '',
        'source_axis_id': axis.forked_from_id,
    }
    if axis.visibility == 'private' and player is not None and axis.owner_id == player.id:
        published_snapshot = ShaftAxis.select(ShaftAxis.id, ShaftAxis.published_at).where(
            (ShaftAxis.owner == player) &
            (ShaftAxis.visibility == 'public') &
            (ShaftAxis.forked_from == axis)
        ).order_by(ShaftAxis.updated_at.desc()).first()
        payload['published_snapshot_id'] = published_snapshot.id if published_snapshot is not None else None
        payload['published_snapshot_at'] = (
            published_snapshot.published_at.isoformat()
            if published_snapshot is not None and published_snapshot.published_at
            else ''
        )
    if include_axis:
        payload['axis'] = axis_payload
        payload['result'] = result
        payload['local_copy_id'] = None
        if player is not None and axis.visibility == 'public' and axis.owner_id != player.id:
            local_copy = ShaftAxis.select().where(
                (ShaftAxis.owner == player) &
                (ShaftAxis.visibility == 'private') &
                (ShaftAxis.forked_from == axis)
            ).order_by(ShaftAxis.updated_at.desc()).first()
            if local_copy is None:
                author = (axis.owner.nickname or axis.owner.player_uid or '作者').strip() or '作者'
                suffix = f' - {author}'
                base_title = (axis.title or '未命名排轴').strip() or '未命名排轴'
                legacy_title = f'{base_title[:max(1, MAX_AXIS_TITLE_LENGTH - len(suffix))]}{suffix}'
                local_copy = ShaftAxis.select().where(
                    (ShaftAxis.owner == player) &
                    (ShaftAxis.visibility == 'private') &
                    (ShaftAxis.title == legacy_title)
                ).order_by(ShaftAxis.updated_at.desc()).first()
            payload['local_copy_id'] = local_copy.id if local_copy is not None else None
    return payload


def normalize_axis_for_hash(axis_payload: dict[str, Any]) -> dict[str, Any]:
    team = []
    for member in axis_payload.get('team') or []:
        team.append({
            'slot': _int(member.get('slot')),
            'character_id': str(member.get('character_id') or ''),
            'arc_id': str(member.get('arc_id') or ''),
            'cartridge_id': str(member.get('cartridge_id') or ''),
        })
    steps = []
    for step in axis_payload.get('steps') or []:
        item = {
            'slot': _int(step.get('slot')),
            'action_id': str(step.get('action_id') or ''),
            'start_tick': _int(step.get('start_tick')),
            'repeat': max(1, _int(step.get('repeat'), 1)),
        }
        if str(step.get('placement') or '') == 'background':
            item['placement'] = 'background'
        if bool(step.get('detached')):
            item['detached'] = True
        if bool(step.get('interrupted')):
            item['interrupted'] = True
            item['interrupt_duration_ticks'] = max(1, _int(step.get('interrupt_duration_ticks'), 1))
            item['interrupt_hit_count'] = max(0, _int(step.get('interrupt_hit_count')))
        steps.append(item)
    return {
        'team': sorted(team, key=lambda item: item['slot']),
        'steps': sorted(steps, key=lambda item: (item['start_tick'], item['slot'], item['action_id'])),
    }


def calculate_axis_hash(axis_payload: dict[str, Any]) -> str:
    normalized = normalize_axis_for_hash(axis_payload)
    payload = _json_dumps(normalized)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _submitted_axis_result(payload: dict[str, Any], axis_payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get('result')
    if not isinstance(result, dict):
        raise RuleValidationError('缺少本地计算结果，请先完成计算后再保存。')
    summary = result.get('summary')
    if not isinstance(summary, dict):
        raise RuleValidationError('本地计算结果缺少伤害汇总。')
    required_summary_fields = (
        'duration_ticks',
        'direct_damage',
        'stagger_damage',
        'total_damage',
        'dps',
    )
    if any(field not in summary for field in required_summary_fields):
        raise RuleValidationError('本地计算结果不完整，请重新计算后再保存。')
    for field in required_summary_fields:
        value = _num(summary.get(field), -1)
        if not math.isfinite(value) or value < 0:
            raise RuleValidationError('本地计算结果包含无效数值，请重新计算后再保存。')
    try:
        normalized_result = json.loads(_json_dumps(result))
    except (TypeError, ValueError):
        raise RuleValidationError('本地计算结果无法保存，请重新计算后再试。') from None
    normalized_result['enemy'] = axis_payload['enemy']
    return normalized_result


def save_shaft_axis(player: Player, payload: dict[str, Any], axis_id: int | None = None) -> dict[str, Any]:
    conflict_action = str(payload.get('conflict_action') or '').strip().lower()
    if conflict_action not in {'', 'overwrite'}:
        raise RuleValidationError('未知的同名排轴处理方式。')
    axis_payload = normalize_axis_payload(payload)
    if _team_contains_disabled_character(axis_payload.get('team'), player):
        raise RuleValidationError('队伍中存在当前账号无权使用的角色。')
    result = _submitted_axis_result(payload, axis_payload)
    axis_payload['duration_ticks'] = _int(result['summary'].get('duration_ticks'), axis_payload['duration_ticks'])
    dedupe_hash = calculate_axis_hash(axis_payload)
    now = datetime.utcnow()
    summary = result['summary']
    title = _clean_text(payload.get('title') or '未命名排轴', MAX_AXIS_TITLE_LENGTH) or '未命名排轴'
    description = _clean_text(payload.get('description'), MAX_AXIS_DESCRIPTION_LENGTH)
    source_axis_id = _int(payload.get('source_axis_id'))

    with atomic_transaction():
        source_axis: ShaftAxis | None = None
        if source_axis_id > 0:
            source_axis = ShaftAxis.select().where(
                (ShaftAxis.id == source_axis_id) &
                (ShaftAxis.visibility == 'public')
            ).first()
            if source_axis is None or source_axis.owner_id == player.id:
                raise RuleValidationError('在线排轴来源无效。')
        record: ShaftAxis | None = None
        if axis_id is not None:
            record = _private_axis_query_for_player(axis_id, player)
            if record is None:
                raise RuleValidationError('没有保存这个排轴的权限。')
        elif source_axis is not None:
            record = ShaftAxis.select().where(
                (ShaftAxis.owner == player) &
                (ShaftAxis.visibility == 'private') &
                (ShaftAxis.forked_from == source_axis)
            ).order_by(ShaftAxis.updated_at.desc()).first()
            if record is None:
                record = ShaftAxis.select().where(
                    (ShaftAxis.owner == player) &
                    (ShaftAxis.visibility == 'private') &
                    (ShaftAxis.title == title)
                ).order_by(ShaftAxis.updated_at.desc()).first()
        duplicate_title_query = ShaftAxis.select().where(
            (ShaftAxis.owner == player) &
            (ShaftAxis.visibility == 'private') &
            (ShaftAxis.title == title)
        )
        if record is not None:
            duplicate_title_query = duplicate_title_query.where(ShaftAxis.id != record.id)
        duplicate_title = duplicate_title_query.first()
        if duplicate_title is not None:
            if conflict_action != 'overwrite':
                raise ShaftAxisNameConflictError(title, duplicate_title.id)
            if record is None:
                record = duplicate_title
            else:
                duplicate_title.delete_instance(recursive=True)
        if record is None:
            _ensure_private_axis_capacity(player)
            record = ShaftAxis.create(owner=player, created_at=now)
        record.title = title
        record.description = description
        record.visibility = 'private'
        record.source_version = SHAFT_SOURCE_VERSION
        record.team_json = _json_dumps(axis_payload['team'])
        record.axis_json = _json_dumps(axis_payload)
        record.enemy_json = _json_dumps(axis_payload['enemy'])
        record.result_json = _json_dumps(result)
        record.duration_ticks = _int(summary.get('duration_ticks'))
        record.direct_damage = _int(summary.get('character_damage', summary.get('direct_damage')))
        record.stagger_damage = _int(summary.get('stagger_damage'))
        record.total_damage = _int(summary.get('total_damage'))
        record.dps_x100 = _int(_num(summary.get('dps')) * 100)
        record.dedupe_hash = dedupe_hash
        if source_axis is not None:
            record.forked_from = source_axis
        record.updated_at = now
        record.published_at = None
        record.save()
        _refresh_axis_character_index(record, axis_payload['team'])

    logger.info('save_shaft_axis player_uid=%s axis_id=%s visibility=private', player.player_uid, record.id)
    return serialize_shaft_axis(record, include_axis=True, player=player)


def get_shaft_axis(axis_id: int, player: Player | None = None, visitor_key: str = '') -> dict[str, Any]:
    axis = _visible_axis(axis_id, player)
    return serialize_shaft_axis(axis, include_axis=True, player=player, visitor_key=visitor_key)


def create_shaft_axis_share(player: Player, axis_id: int) -> dict[str, Any]:
    with atomic_transaction():
        axis = _private_axis_query_for_player(axis_id, player)
        if axis is None:
            raise RuleValidationError('没有分享这个排轴的权限。')
        if not axis.share_token:
            while True:
                share_token = secrets.token_urlsafe(24)
                if not ShaftAxis.select().where(ShaftAxis.share_token == share_token).exists():
                    break
            axis.share_token = share_token
            axis.save(only=[ShaftAxis.share_token])
    logger.info('create_shaft_axis_share player_uid=%s axis_id=%s', player.player_uid, axis.id)
    return {
        'axis_id': axis.id,
        'share_token': axis.share_token,
        'share_path': f'/shaft/rotation?share={axis.share_token}',
    }


def get_shared_shaft_axis(
    share_token: str,
    player: Player | None = None,
    visitor_key: str = '',
) -> dict[str, Any]:
    token = str(share_token or '').strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{20,64}', token):
        raise RuleValidationError('分享链接无效或已失效。')
    axis = ShaftAxis.select(ShaftAxis, Player).join(Player).where(
        (ShaftAxis.share_token == token) &
        (ShaftAxis.visibility == 'private')
    ).first()
    if axis is None:
        raise RuleValidationError('分享链接无效或已失效。')
    payload = serialize_shaft_axis(
        axis,
        include_axis=True,
        player=player,
        visitor_key=visitor_key,
    )
    payload['shared'] = True
    payload['read_only'] = player is None
    return payload


def backup_shaft_axis(player: Player, axis_id: int) -> dict[str, Any]:
    now = datetime.utcnow()
    with atomic_transaction():
        source = _private_axis_query_for_player(axis_id, player)
        if source is None:
            raise RuleValidationError('没有备份这个排轴的权限。')
        source_title = _clean_text(source.title or '未命名排轴', MAX_AXIS_TITLE_LENGTH) or '未命名排轴'
        copy_number = 1
        while True:
            suffix = '-副本' if copy_number == 1 else f'-副本 ({copy_number})'
            title = f'{source_title[:MAX_AXIS_TITLE_LENGTH - len(suffix)]}{suffix}'
            if not ShaftAxis.select().where(
                (ShaftAxis.owner == player) &
                (ShaftAxis.visibility == 'private') &
                (ShaftAxis.title == title)
            ).exists():
                break
            copy_number += 1
        _ensure_private_axis_capacity(player)
        backup = ShaftAxis.create(
            owner=player,
            title=title,
            description=source.description,
            visibility='private',
            source_version=source.source_version,
            team_json=source.team_json,
            axis_json=source.axis_json,
            enemy_json=source.enemy_json,
            result_json=source.result_json,
            duration_ticks=source.duration_ticks,
            direct_damage=source.direct_damage,
            stagger_damage=source.stagger_damage,
            total_damage=source.total_damage,
            dps_x100=source.dps_x100,
            dedupe_hash=source.dedupe_hash,
            created_at=now,
            updated_at=now,
        )
        team = _safe_json_loads(source.team_json, [])
        _refresh_axis_character_index(backup, team if isinstance(team, list) else [])
    logger.info(
        'backup_shaft_axis player_uid=%s source_axis_id=%s backup_axis_id=%s',
        player.player_uid,
        axis_id,
        backup.id,
    )
    return serialize_shaft_axis(backup, include_axis=True, player=player)


def publish_shaft_axis_snapshot(player: Player, axis_id: int) -> dict[str, Any]:
    now = datetime.utcnow()
    with atomic_transaction():
        source = _private_axis_query_for_player(axis_id, player)
        if source is None:
            raise RuleValidationError('没有上传这个排轴的权限。')
        if _axis_contains_disabled_character(source, player):
            raise RuleValidationError('队伍中存在当前账号无权使用的角色。')
        snapshot = ShaftAxis.select().where(
            (ShaftAxis.owner == player) &
            (ShaftAxis.visibility == 'public') &
            (ShaftAxis.forked_from == source)
        ).order_by(ShaftAxis.updated_at.desc()).first()
        duplicate_query = ShaftAxis.select().where(
            (ShaftAxis.visibility == 'public') &
            (ShaftAxis.dedupe_hash == source.dedupe_hash)
        )
        if snapshot is not None:
            duplicate_query = duplicate_query.where(ShaftAxis.id != snapshot.id)
        duplicate = duplicate_query.first()
        if duplicate is not None:
            if snapshot is None and duplicate.owner_id == player.id and duplicate.forked_from_id is None:
                snapshot = duplicate
            else:
                raise RuleValidationError('广场中已经存在相同角色、弧盘、卡带与动作轴的快照。')
        if snapshot is None:
            snapshot = ShaftAxis.create(owner=player, visibility='public', created_at=now)
        snapshot.title = source.title
        snapshot.description = source.description
        snapshot.visibility = 'public'
        snapshot.source_version = source.source_version
        snapshot.team_json = source.team_json
        snapshot.axis_json = source.axis_json
        snapshot.enemy_json = source.enemy_json
        snapshot.result_json = source.result_json
        snapshot.duration_ticks = source.duration_ticks
        snapshot.direct_damage = source.direct_damage
        snapshot.stagger_damage = source.stagger_damage
        snapshot.total_damage = source.total_damage
        snapshot.dps_x100 = source.dps_x100
        snapshot.dedupe_hash = source.dedupe_hash
        snapshot.forked_from = source
        snapshot.updated_at = now
        snapshot.published_at = now
        snapshot.save()
        team = _safe_json_loads(source.team_json, [])
        _refresh_axis_character_index(snapshot, team if isinstance(team, list) else [])
    logger.info(
        'publish_shaft_axis_snapshot player_uid=%s source_axis_id=%s snapshot_axis_id=%s',
        player.player_uid,
        axis_id,
        snapshot.id,
    )
    return serialize_shaft_axis(snapshot, include_axis=True, player=player)


def delete_shaft_axis(player: Player, axis_id: int) -> dict[str, Any]:
    with atomic_transaction():
        axis = _axis_query_for_player(axis_id, player)
        if axis is None:
            raise RuleValidationError('没有删除这个排轴的权限。')
        axis.delete_instance(recursive=True)
    logger.info('delete_shaft_axis player_uid=%s axis_id=%s', player.player_uid, axis_id)
    return {'ok': True}


def list_shaft_market(
    *,
    character_ids: list[str] | None = None,
    sort: str = 'dps',
    query_text: str = '',
    page: int = 1,
    page_size: int = 20,
    player: Player | None = None,
    visitor_key: str = '',
) -> dict[str, Any]:
    page = max(1, min(200, page))
    page_size = max(1, min(60, page_size))
    sort = sort if sort in MARKET_SORTS else 'dps'
    query = ShaftAxis.select(ShaftAxis, Player).join(Player).where(ShaftAxis.visibility == 'public')
    query = _filter_visible_character_axes(query, player)
    selected_character_ids = list(dict.fromkeys(
        str(character_id).strip()
        for character_id in (character_ids or [])
        if str(character_id).strip()
    ))[:4]
    for character_id in selected_character_ids:
        subquery = ShaftAxisCharacter.select(ShaftAxisCharacter.axis).where(
            ShaftAxisCharacter.character_id == character_id
        )
        query = query.where(ShaftAxis.id.in_(subquery))
    cleaned_query = _clean_text(query_text, MAX_AXIS_TITLE_LENGTH)
    if cleaned_query:
        query = query.where(
            ShaftAxis.title.contains(cleaned_query) |
            ShaftAxis.description.contains(cleaned_query)
        )
    if sort == 'likes':
        query = query.order_by(ShaftAxis.like_count.desc(), ShaftAxis.dps_x100.desc(), ShaftAxis.updated_at.desc())
    elif sort == 'favorites':
        query = query.order_by(ShaftAxis.favorite_count.desc(), ShaftAxis.dps_x100.desc(), ShaftAxis.updated_at.desc())
    elif sort == 'new':
        query = query.order_by(ShaftAxis.published_at.desc(), ShaftAxis.updated_at.desc())
    else:
        query = query.order_by(ShaftAxis.dps_x100.desc(), ShaftAxis.like_count.desc(), ShaftAxis.updated_at.desc())
    items = list(query.paginate(page, page_size + 1))
    visible_items = items[:page_size]
    return {
        'items': [
            serialize_shaft_axis(axis, include_axis=False, player=player, visitor_key=visitor_key)
            for axis in visible_items
        ],
        'page': page,
        'page_size': page_size,
        'has_more': len(items) > page_size,
    }


def list_my_shaft_axes(
    player: Player,
    *,
    character_ids: list[str] | None = None,
    sort: str = 'new',
    query_text: str = '',
) -> dict[str, Any]:
    return list_filtered_shaft_axes(
        player=player,
        scope='mine',
        character_ids=character_ids,
        sort=sort,
        query_text=query_text,
    )


def _filter_axis_query_by_characters(query, character_ids: list[str] | None):
    selected_character_ids = list(dict.fromkeys(
        str(character_id).strip()
        for character_id in (character_ids or [])
        if str(character_id).strip()
    ))[:4]
    for character_id in selected_character_ids:
        subquery = ShaftAxisCharacter.select(ShaftAxisCharacter.axis).where(
            ShaftAxisCharacter.character_id == character_id
        )
        query = query.where(ShaftAxis.id.in_(subquery))
    return query


def _order_axis_query(query, sort: str):
    if sort == 'likes':
        return query.order_by(ShaftAxis.like_count.desc(), ShaftAxis.dps_x100.desc(), ShaftAxis.updated_at.desc())
    if sort == 'favorites':
        return query.order_by(ShaftAxis.favorite_count.desc(), ShaftAxis.dps_x100.desc(), ShaftAxis.updated_at.desc())
    if sort == 'dps':
        return query.order_by(ShaftAxis.dps_x100.desc(), ShaftAxis.updated_at.desc())
    return query.order_by(ShaftAxis.updated_at.desc())


def list_filtered_shaft_axes(
    *,
    player: Player,
    scope: str,
    character_ids: list[str] | None = None,
    sort: str = 'new',
    query_text: str = '',
) -> dict[str, Any]:
    sort = sort if sort in MARKET_SORTS else 'new'
    query = ShaftAxis.select(ShaftAxis, Player).join(Player)
    if scope == 'favorites':
        favorite_axis_ids = ShaftAxisFavorite.select(ShaftAxisFavorite.axis).where(
            ShaftAxisFavorite.player == player
        )
        query = query.where(
            (ShaftAxis.visibility == 'public') &
            (ShaftAxis.id.in_(favorite_axis_ids))
        )
        query = _filter_visible_character_axes(query, player)
    else:
        query = query.where(
            (ShaftAxis.owner == player) &
            (ShaftAxis.visibility == 'private')
        )
    total = query.count()
    query = _filter_axis_query_by_characters(query, character_ids)
    cleaned_query = _clean_text(query_text, MAX_AXIS_TITLE_LENGTH)
    if cleaned_query:
        query = query.where(
            ShaftAxis.title.contains(cleaned_query) |
            ShaftAxis.description.contains(cleaned_query)
        )
    axes = list(_order_axis_query(query, sort).limit(100))
    return {
        'items': [serialize_shaft_axis(axis, include_axis=False, player=player) for axis in axes],
        'total': total,
    }


def list_favorite_shaft_axes(
    player: Player,
    *,
    character_ids: list[str] | None = None,
    sort: str = 'new',
    query_text: str = '',
) -> dict[str, Any]:
    return list_filtered_shaft_axes(
        player=player,
        scope='favorites',
        character_ids=character_ids,
        sort=sort,
        query_text=query_text,
    )


def set_shaft_axis_like(player: Player, axis_id: int, liked: bool) -> dict[str, Any]:
    with atomic_transaction():
        axis = _visible_axis(axis_id, player)
        if axis.visibility != 'public':
            raise RuleValidationError('只能点赞公开排轴。')
        existing = ShaftAxisLike.select().where(
            (ShaftAxisLike.axis == axis) &
            (ShaftAxisLike.player == player)
        ).first()
        if liked and existing is None:
            existing_dislike = ShaftAxisDislike.select().where(
                (ShaftAxisDislike.axis == axis) &
                (ShaftAxisDislike.player == player)
            ).first()
            if existing_dislike is not None:
                existing_dislike.delete_instance()
                axis.dislike_count = max(0, axis.dislike_count - 1)
            ShaftAxisLike.create(axis=axis, player=player)
            axis.like_count += 1
            axis.save()
        elif not liked and existing is not None:
            existing.delete_instance()
            axis.like_count = max(0, axis.like_count - 1)
            axis.save()
    return serialize_shaft_axis(axis, include_axis=False, player=player)


def set_shaft_axis_dislike(player: Player, axis_id: int, disliked: bool) -> dict[str, Any]:
    with atomic_transaction():
        axis = _visible_axis(axis_id, player)
        if axis.visibility != 'public':
            raise RuleValidationError('只能踩公开排轴。')
        existing = ShaftAxisDislike.select().where(
            (ShaftAxisDislike.axis == axis) &
            (ShaftAxisDislike.player == player)
        ).first()
        if disliked and existing is None:
            existing_like = ShaftAxisLike.select().where(
                (ShaftAxisLike.axis == axis) &
                (ShaftAxisLike.player == player)
            ).first()
            if existing_like is not None:
                existing_like.delete_instance()
                axis.like_count = max(0, axis.like_count - 1)
            ShaftAxisDislike.create(axis=axis, player=player)
            axis.dislike_count += 1
            axis.save()
        elif not disliked and existing is not None:
            existing.delete_instance()
            axis.dislike_count = max(0, axis.dislike_count - 1)
            axis.save()
    return serialize_shaft_axis(axis, include_axis=False, player=player)


def set_shaft_axis_favorite(
    *,
    axis_id: int,
    favorited: bool,
    player: Player | None = None,
    visitor_key: str = '',
) -> dict[str, Any]:
    visitor_key = _clean_visitor_key(visitor_key)
    if player is None and not visitor_key:
        raise RuleValidationError('缺少收藏标识。')
    with atomic_transaction():
        axis = _visible_axis(axis_id, player)
        if axis.visibility != 'public':
            raise RuleValidationError('只能收藏公开排轴。')
        if favorited and player is not None and axis.owner_id == player.id:
            raise RuleValidationError('不能收藏自己上传的排轴。')
        existing = _favorite_query(axis, player, visitor_key).first()
        if favorited and existing is None:
            ShaftAxisFavorite.create(
                axis=axis,
                player=player,
                visitor_key='' if player is not None else visitor_key,
            )
            axis.favorite_count += 1
            axis.save()
        elif not favorited and existing is not None:
            existing.delete_instance()
            axis.favorite_count = max(0, axis.favorite_count - 1)
            axis.save()
    return serialize_shaft_axis(axis, include_axis=False, player=player, visitor_key=visitor_key)


def optional_player_from_token(token: str) -> Player | None:
    from app.dao import get_player_by_token

    return get_player_by_token(token)
