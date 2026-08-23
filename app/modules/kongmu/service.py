from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from app.errors import AppError


MODULE_ROOT = Path(__file__).resolve().parent
APP_ROOT = MODULE_ROOT.parents[1]
KONGMU_DATA_DIR = MODULE_ROOT / 'static' / 'data'
KONGMU_CHARACTER_DIR = KONGMU_DATA_DIR / 'characters'

STATIC_BASE = 'https://static.nanoka.cc'
GAME_KEY = 'nte'

GEOMETRY_SHAPES: dict[str, list[list[int]]] = {
    'EquipmentGeometry_Hen2': [[0, 0], [1, 0]],
    'EquipmentGeometry_Shu2': [[0, 0], [0, 1]],
    'EquipmentGeometry_Hen3': [[0, 0], [1, 0], [2, 0]],
    'EquipmentGeometry_Shu3': [[0, 0], [0, 1], [0, 2]],
    'EquipmentGeometry_ZhiJiao1': [[0, 0], [0, 1], [1, 1]],
    'EquipmentGeometry_ZhiJiao2': [[0, 0], [1, 0], [0, 1]],
    'EquipmentGeometry_ZhiJiao3': [[0, 0], [1, 0], [1, 1]],
    'EquipmentGeometry_ZhiJiao4': [[1, 0], [0, 1], [1, 1]],
    'EquipmentGeometry_Hen4': [[0, 0], [1, 0], [2, 0], [3, 0]],
    'EquipmentGeometry_Shu4': [[0, 0], [0, 1], [0, 2], [0, 3]],
    'EquipmentGeometry_Z3': [[1, 0], [2, 0], [0, 1], [1, 1]],
    'EquipmentGeometry_Z4': [[1, 0], [0, 1], [1, 1], [0, 2]],
}

GEOMETRY_LABELS: dict[str, str] = {
    'EquipmentGeometry_Hen2': 'II横',
    'EquipmentGeometry_Shu2': 'II竖',
    'EquipmentGeometry_Hen3': 'III横',
    'EquipmentGeometry_Shu3': 'III竖',
    'EquipmentGeometry_ZhiJiao1': 'III角1',
    'EquipmentGeometry_ZhiJiao2': 'III角2',
    'EquipmentGeometry_ZhiJiao3': 'III角3',
    'EquipmentGeometry_ZhiJiao4': 'III角4',
    'EquipmentGeometry_Hen4': 'IV横',
    'EquipmentGeometry_Shu4': 'IV竖',
    'EquipmentGeometry_Z3': 'IVZ横',
    'EquipmentGeometry_Z4': 'IVZ竖',
}

GEOMETRY_SORT_ORDER = list(GEOMETRY_SHAPES)

ELEMENT_LABELS: dict[str, str] = {
    'NATURE': '灵',
    'COSMOS': '光',
    'LAKSHANA': '相',
    'INCANTATION': '咒',
    'CHAOS': '暗',
    'PSYCHE': '魂',
}

CHARACTER_ELEMENT_SORT_ORDER = ['COSMOS', 'NATURE', 'INCANTATION', 'CHAOS', 'PSYCHE', 'LAKSHANA']

OWNER_TYPE_LABELS = {
    2: 'II型',
    3: 'III型',
    4: 'IV型',
}

FIXED_CHARACTER_AVATAR_SOURCE_ICONS = {
    '1046': '/Game/UI/UI_Icon/AvatarImage/256/player_009_256',
    '1051': '/Game/UI/UI_Icon/AvatarImage/256/player_009_256',
}

FIXED_CHARACTER_AVATAR_LOCAL_PATHS = {
    '1046': 'images/characters/avatar/男主.webp',
    '1051': 'images/characters/avatar/男主.webp',
    'char_076a1f4e53': 'images/characters/avatar/残虹.png',
}

TEST_CHARACTER_IDS = {'char_076a1f4e53'}
PUBLIC_KONGMU_CHARACTER_IDS = {'char_076a1f4e53'}


@dataclass(frozen=True)
class Placement:
    geometry: str
    mask: int
    cells: tuple[tuple[int, int], ...]
    grid_count: int


@dataclass
class PlanResult:
    best_owner_drive_count: int
    solutions: list[dict[str, Any]]
    searched_solution_count: int


def normalize_text(value: str | None) -> str:
    text = str(value or '').strip().lower()
    text = text.strip('「」『』"\'` ')
    text = re.sub(r'[\s_\-·・:：,，。/\\]+', '', text)
    return text


def display_name_without_quotes(value: str | None) -> str:
    return str(value or '').strip().strip('「」『』"\'` ')


def load_json(path: Path) -> Any:
    if not path.exists():
        raise AppError(f'缺少空幕数据文件：{path.name}')
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def nanoka_icon_url(icon_path: str | None) -> str:
    path = str(icon_path or '').strip()
    if not path:
        return ''
    return f'{STATIC_BASE}/assets/{GAME_KEY}/{path.lstrip("/")}.webp'


def static_asset_url(path: str | None) -> str:
    value = str(path or '').strip().lstrip('/')
    if not value:
        return ''
    if value.startswith('static/'):
        return f'/{value}'
    return f'/static/{value}'


def geometry_label(geometry: str) -> str:
    return GEOMETRY_LABELS.get(geometry, geometry.replace('EquipmentGeometry_', ''))


def valid_cells_from_slots(slots: list[list[int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for y, row in enumerate(slots):
        for x, value in enumerate(row):
            if value != -1:
                cells.append((x, y))
    return cells


def build_placements(
    slots: list[list[int]],
    drives_by_geometry: dict[str, dict[str, Any]],
) -> tuple[list[Placement], dict[int, list[Placement]], dict[tuple[int, int], int], int]:
    valid_cells = valid_cells_from_slots(slots)
    cell_to_index = {cell: index for index, cell in enumerate(valid_cells)}
    valid_set = set(valid_cells)
    valid_mask = 0
    for index in range(len(valid_cells)):
        valid_mask |= 1 << index

    rows = len(slots)
    cols = max((len(row) for row in slots), default=0)
    placements: list[Placement] = []

    for geometry in GEOMETRY_SORT_ORDER:
        if geometry not in drives_by_geometry:
            continue
        shape = GEOMETRY_SHAPES[geometry]
        grid_count = int(drives_by_geometry[geometry]['grid_count'])
        for y in range(rows):
            for x in range(cols):
                cells = tuple((x + dx, y + dy) for dx, dy in shape)
                if not all(cell in valid_set for cell in cells):
                    continue
                mask = 0
                for cell in cells:
                    mask |= 1 << cell_to_index[cell]
                placements.append(Placement(geometry, mask, cells, grid_count))

    placements_by_bit: dict[int, list[Placement]] = {}
    for placement in placements:
        mask = placement.mask
        bit = 0
        while mask:
            if mask & 1:
                placements_by_bit.setdefault(bit, []).append(placement)
            bit += 1
            mask >>= 1

    for bit_placements in placements_by_bit.values():
        bit_placements.sort(key=lambda item: (item.grid_count != 3, item.grid_count, item.geometry, item.cells))

    return placements, placements_by_bit, cell_to_index, valid_mask


def first_empty_bit(filled_mask: int, valid_mask: int) -> int:
    remaining = valid_mask & ~filled_mask
    return (remaining & -remaining).bit_length() - 1


def area_is_fillable(area: int) -> bool:
    if area < 0:
        return False
    if area == 0:
        return True
    for twos in range(area // 2 + 1):
        for threes in range(area // 3 + 1):
            rest = area - 2 * twos - 3 * threes
            if rest >= 0 and rest % 4 == 0:
                return True
    return False


def plan_layouts(
    slots: list[list[int]],
    drives: list[dict[str, Any]],
    cartridge: dict[str, Any],
    owner_grid_count: int,
    limit: int = 0,
) -> PlanResult:
    drives_by_geometry = {drive['geometry']: drive for drive in drives}
    required = list(dict.fromkeys(cartridge['synergy_geometry']))
    required_area = {
        geometry: int(drives_by_geometry[geometry]['grid_count'])
        for geometry in required
        if geometry in drives_by_geometry
    }

    missing_geometry = [geometry for geometry in required if geometry not in drives_by_geometry]
    if missing_geometry:
        raise AppError(f'缺少驱动块数据：{", ".join(missing_geometry)}')

    placements, placements_by_bit, _cell_to_index, valid_mask = build_placements(slots, drives_by_geometry)
    total_cells = valid_mask.bit_count()
    counts: dict[str, int] = {geometry: 0 for geometry in GEOMETRY_SORT_ORDER}
    chosen: list[Placement] = []
    best_owner_drive_count = -1
    solutions_by_counts: dict[tuple[int, ...], dict[str, Any]] = {}
    searched_solution_count = 0

    def counts_key() -> tuple[int, ...]:
        return tuple(counts.get(geometry, 0) for geometry in GEOMETRY_SORT_ORDER)

    def materialize_solution(owner_drive_count: int) -> dict[str, Any]:
        pieces = []
        for index, placement in enumerate(chosen, start=1):
            pieces.append(
                {
                    'index': index,
                    'geometry': placement.geometry,
                    'label': geometry_label(placement.geometry),
                    'grid_count': placement.grid_count,
                    'cells': [list(cell) for cell in placement.cells],
                    'is_owner_drive': placement.grid_count == owner_grid_count,
                    'is_synergy_drive': placement.geometry in required,
                }
            )
        nonzero_counts = {
            geometry: counts[geometry]
            for geometry in GEOMETRY_SORT_ORDER
            if counts.get(geometry, 0) > 0
        }
        return {
            'owner_drive_count': owner_drive_count,
            'counts': nonzero_counts,
            'pieces': pieces,
        }

    def dfs(filled_mask: int, filled_cells: int, owner_drive_count: int) -> None:
        nonlocal best_owner_drive_count, searched_solution_count, solutions_by_counts

        remaining_cells = total_cells - filled_cells
        if owner_grid_count > 0:
            possible_owner = owner_drive_count + remaining_cells // owner_grid_count
            if possible_owner < best_owner_drive_count:
                return

        missing_required = [geometry for geometry in required if counts.get(geometry, 0) == 0]
        min_missing_area = sum(required_area[geometry] for geometry in missing_required)
        if min_missing_area > remaining_cells:
            return
        if not area_is_fillable(remaining_cells - min_missing_area):
            return

        if filled_mask == valid_mask:
            if missing_required:
                return
            searched_solution_count += 1
            if owner_drive_count > best_owner_drive_count:
                best_owner_drive_count = owner_drive_count
                solutions_by_counts = {}
            if owner_drive_count != best_owner_drive_count:
                return
            key = counts_key()
            if key not in solutions_by_counts:
                solutions_by_counts[key] = materialize_solution(owner_drive_count)
            return

        bit = first_empty_bit(filled_mask, valid_mask)
        for placement in placements_by_bit.get(bit, []):
            if placement.mask & filled_mask:
                continue
            counts[placement.geometry] += 1
            chosen.append(placement)
            dfs(
                filled_mask | placement.mask,
                filled_cells + placement.grid_count,
                owner_drive_count + (1 if placement.grid_count == owner_grid_count else 0),
            )
            chosen.pop()
            counts[placement.geometry] -= 1

            if limit > 0 and len(solutions_by_counts) >= limit:
                return

    if not placements:
        return PlanResult(best_owner_drive_count=0, solutions=[], searched_solution_count=0)

    dfs(0, 0, 0)
    solutions = sorted(
        solutions_by_counts.values(),
        key=lambda solution: tuple(solution['counts'].get(geometry, 0) for geometry in GEOMETRY_SORT_ORDER),
    )
    return PlanResult(
        best_owner_drive_count=max(best_owner_drive_count, 0),
        solutions=solutions,
        searched_solution_count=searched_solution_count,
    )


def resolve_record(records: Iterable[dict[str, Any]], query: str, fields: Iterable[str]) -> dict[str, Any]:
    normalized = normalize_text(query)
    if not normalized:
        raise AppError('请选择角色和卡带。')

    exact: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    for record in records:
        values: list[str] = []
        for field in fields:
            value = record.get(field)
            if isinstance(value, dict):
                values.extend(str(item) for item in value.values())
            elif isinstance(value, list):
                values.extend(str(item) for item in value)
            elif value is not None:
                values.append(str(value))
        values.append(str(record.get('id', '')))
        normalized_values = [normalize_text(value) for value in values]
        if normalized in normalized_values:
            exact.append(record)
        elif any(normalized and normalized in value for value in normalized_values):
            partial.append(record)

    matches = exact or partial
    if not matches:
        raise AppError(f'没有找到匹配记录：{query}')
    if len(matches) > 1:
        names = ', '.join(str(match.get('name') or match.get('id')) for match in matches[:8])
        raise AppError(f'匹配结果不唯一：{query}（{names}）')
    return matches[0]


def common_geometry_counts(solutions: list[dict[str, Any]]) -> dict[str, int]:
    if not solutions:
        return {}
    common = dict(solutions[0]['counts'])
    for solution in solutions[1:]:
        counts = solution['counts']
        for geometry in list(common):
            common[geometry] = min(int(common[geometry]), int(counts.get(geometry, 0)))
            if common[geometry] <= 0:
                del common[geometry]
    return common


def drive_chip(geometry: str, count: int = 0) -> dict[str, Any]:
    drive = get_drives_by_geometry().get(geometry, {})
    return {
        'geometry': geometry,
        'label': geometry_label(geometry),
        'name': drive.get('name') or geometry_label(geometry),
        'grid_count': int(drive.get('grid_count') or 0),
        'icon': static_asset_url(filter_drive_icon_path(geometry)),
        'source_icon': drive.get('icon_url') or '',
        'count': count,
    }


def apply_required_flags(solution: dict[str, Any], required_counts: dict[str, int]) -> None:
    used: dict[str, int] = {}
    optional_counts: dict[str, int] = {}
    for piece in solution['pieces']:
        geometry = str(piece['geometry'])
        used_count = used.get(geometry, 0)
        is_required = used_count < int(required_counts.get(geometry, 0))
        piece['is_required_drive'] = is_required
        used[geometry] = used_count + 1
        if not is_required:
            optional_counts[geometry] = optional_counts.get(geometry, 0) + 1
    solution['optional_counts'] = optional_counts


def optional_filter_rows(solutions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for grid_count in (2, 3, 4):
        max_select = 0
        option_counts: dict[str, int] = {}
        for solution in solutions:
            optional_counts = solution.get('optional_counts') or {}
            size_total = 0
            for geometry, count in optional_counts.items():
                drive = get_drives_by_geometry().get(geometry, {})
                if int(drive.get('grid_count') or 0) != grid_count:
                    continue
                int_count = int(count)
                size_total += int_count
                option_counts[geometry] = max(option_counts.get(geometry, 0), int_count)
            max_select = max(max_select, size_total)
        if max_select <= 0 or not option_counts:
            continue
        rows.append(
            {
                'grid_count': grid_count,
                'label': OWNER_TYPE_LABELS.get(grid_count, f'{grid_count}格'),
                'max_select': max_select,
                'options': [
                    drive_chip(geometry, count)
                    for geometry, count in sorted(
                        option_counts.items(),
                        key=lambda item: GEOMETRY_SORT_ORDER.index(item[0]) if item[0] in GEOMETRY_SORT_ORDER else 999,
                    )
                ],
            }
        )
    return rows


def element_label(element_id: str | None, element: str | None = None) -> str:
    key = str(element_id or '').upper()
    if key in ELEMENT_LABELS:
        return ELEMENT_LABELS[key]
    return str(element or '').strip() or '光'


def element_icon_url(label: str) -> str:
    return static_asset_url(f'preteam/element/{label}.png')


def icon_stem(icon_path: str | None) -> str:
    path = str(icon_path or '').strip().rstrip('/')
    if not path:
        return ''
    return path.rsplit('/', 1)[-1]


def character_avatar_path(character_id: str, name: str | None = None, source_icon: str | None = None) -> str:
    for candidate_name in (name, display_name_without_quotes(name)):
        clean_name = str(candidate_name or '').strip()
        if not clean_name:
            continue
        shared_path = APP_ROOT / 'static' / 'images' / 'characters' / 'avatar' / f'{clean_name}.webp'
        if shared_path.exists():
            return f'images/characters/avatar/{clean_name}.webp'

    source_stem = icon_stem(source_icon)
    if source_stem:
        source_path = MODULE_ROOT / 'static' / 'images' / 'characters' / f'{source_stem}.webp'
        if source_path.exists():
            return f'kongmu/images/characters/{source_stem}.webp'
    id_path = MODULE_ROOT / 'static' / 'images' / 'characters' / f'{character_id}.webp'
    if id_path.exists():
        return f'kongmu/images/characters/{character_id}.webp'
    id_svg_path = MODULE_ROOT / 'static' / 'images' / 'characters' / f'{character_id}.svg'
    if id_svg_path.exists():
        return f'kongmu/images/characters/{character_id}.svg'
    return ''


def cartridge_icon_path(cartridge_id: str) -> str:
    return f'kongmu/images/cartridges/{cartridge_id}.webp'


def drive_icon_path(geometry: str) -> str:
    return f'kongmu/images/drive_icons/{geometry}.png'


def filter_drive_icon_path(geometry: str) -> str:
    return f'kongmu/images/drive_icons/{geometry}.webp'


def format_stat_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:g}'
    return str(value).strip()


def kongmu_passive(equip_slots: dict[str, Any]) -> dict[str, Any]:
    title = display_name_without_quotes(equip_slots.get('special_desc'))
    template = str(equip_slots.get('description_template') or '').strip()
    lines: list[str] = []

    for stat in equip_slots.get('stats') or []:
        if not isinstance(stat, dict):
            continue
        stat_name = str(stat.get('name') or stat.get('id_stats') or '').strip()
        stat_value = format_stat_value(stat.get('value'))
        if template:
            line = template.replace('{0}', stat_name).replace('{1}', stat_value).strip()
        elif stat_name and stat_value:
            suffix = '%' if stat.get('b_show_percent') else ''
            line = f'{stat_name}+{stat_value}{suffix}'
        else:
            line = stat_name or stat_value
        if line:
            lines.append(line)

    text = '；'.join(lines)
    if title and text:
        text = f'{title}：{text}'
    elif title:
        text = title

    return {
        'title': title,
        'lines': lines,
        'text': text,
    }


def compact_character(record: dict[str, Any], detail: dict[str, Any] | None = None) -> dict[str, Any]:
    detail = detail or {}
    character_id = str(record.get('id') or detail.get('id') or '')
    label = element_label(record.get('element_id') or detail.get('element_id'), record.get('element') or detail.get('element'))
    equip_slots = detail.get('equip_slots') or {}
    character_name = record.get('name') or detail.get('name') or character_id
    source_icon = FIXED_CHARACTER_AVATAR_SOURCE_ICONS.get(
        character_id,
        record.get('icon') or detail.get('icon'),
    )
    local_avatar = (
        FIXED_CHARACTER_AVATAR_LOCAL_PATHS.get(character_id)
        or character_avatar_path(character_id, character_name, source_icon)
    )
    source_avatar = nanoka_icon_url(source_icon)
    return {
        'id': character_id,
        'name': character_name,
        'names': record.get('names') or detail.get('names') or {},
        'element': label,
        'element_id': record.get('element_id') or detail.get('element_id') or '',
        'element_icon': element_icon_url(label),
        'rarity': record.get('rarity') or detail.get('rarity'),
        'avatar': static_asset_url(local_avatar) if local_avatar else source_avatar,
        'source_icon': source_avatar,
        'owner_grid_count': int(equip_slots.get('owner_grid_count') or 0),
        'owner_type_label': OWNER_TYPE_LABELS.get(int(equip_slots.get('owner_grid_count') or 0), ''),
        'kongmu_passive': kongmu_passive(equip_slots),
    }


def compact_character_avatar_choice(record: dict[str, Any], detail: dict[str, Any] | None = None) -> dict[str, str]:
    detail = detail or {}
    character_id = str(record.get('id') or detail.get('id') or '')
    character_name = record.get('name') or detail.get('name') or character_id
    source_icon = record.get('icon') or detail.get('icon')
    source_avatar = nanoka_icon_url(source_icon)
    local_avatar = (
        FIXED_CHARACTER_AVATAR_LOCAL_PATHS.get(character_id)
        or character_avatar_path(character_id, character_name, source_icon)
    )
    return {
        'id': character_id,
        'avatar': static_asset_url(local_avatar) if local_avatar else source_avatar,
        'source_icon': source_avatar,
    }


def compact_cartridge(cartridge: dict[str, Any]) -> dict[str, Any]:
    cartridge_id = str(cartridge.get('id') or '')
    return {
        'id': cartridge_id,
        'name': display_name_without_quotes(cartridge.get('name')),
        'raw_name': cartridge.get('name') or cartridge_id,
        'aliases': cartridge.get('aliases') or [],
        'icon': static_asset_url(cartridge_icon_path(cartridge_id)),
        'source_icon': nanoka_icon_url(cartridge.get('icon')),
        'synergy_geometry': cartridge.get('synergy_geometry') or [],
        'synergy': [
            {
                'id': item.get('id'),
                'name': item.get('name'),
                'label': geometry_label(str(item.get('id') or '')),
                'icon': nanoka_icon_url(item.get('icon')),
            }
            for item in cartridge.get('synergy') or []
            if isinstance(item, dict)
        ],
        'conditions': cartridge.get('conditions') or [],
    }


def dedupe_cartridges(cartridges: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for cartridge in cartridges:
        key = normalize_text(display_name_without_quotes(cartridge.get('name')))
        if key and key in seen_names:
            continue
        if key:
            seen_names.add(key)
        deduped.append(cartridge)
    return deduped


def dedupe_characters(characters: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for character in characters:
        name_key = normalize_text(display_name_without_quotes(character['record'].get('name') or character['detail'].get('name')))
        if not name_key:
            deduped.append(character)
            continue
        character_id = str(character['record'].get('id') or character['detail'].get('id') or '')
        existing = by_name.get(name_key)
        if existing is None:
            merged_character = {
                **character,
                'merged_ids': [character_id],
                'avatar_choice_items': [],
            }
            by_name[name_key] = merged_character
            deduped.append(merged_character)
            continue
        existing['merged_ids'].append(character_id)
        existing['avatar_choice_items'].append(character)
    return deduped


def character_sort_key(item: dict[str, Any]) -> tuple[int, int]:
    element_id = str(item['record'].get('element_id') or item['detail'].get('element_id') or '').upper()
    try:
        element_rank = CHARACTER_ELEMENT_SORT_ORDER.index(element_id)
    except ValueError:
        element_rank = len(CHARACTER_ELEMENT_SORT_ORDER)
    return element_rank, int(item.get('source_index') or 0)


def compact_drive(drive: dict[str, Any]) -> dict[str, Any]:
    geometry = str(drive.get('geometry') or '')
    return {
        'id': drive.get('id'),
        'name': drive.get('name') or geometry_label(geometry),
        'geometry': geometry,
        'label': geometry_label(geometry),
        'grid_count': int(drive.get('grid_count') or 0),
        'shape': drive.get('shape') or GEOMETRY_SHAPES.get(geometry, []),
        'icon': drive.get('icon'),
        'icon_url': static_asset_url(drive.get('local_icon') or drive_icon_path(geometry)),
        'source_icon': drive.get('asset_url') or nanoka_icon_url(drive.get('icon')),
    }


@lru_cache(maxsize=1)
def get_kongmu_raw_data() -> dict[str, Any]:
    drives = load_json(KONGMU_DATA_DIR / 'drives.json').get('drives') or []
    cartridges = load_json(KONGMU_DATA_DIR / 'cartridges.json').get('cartridges') or []
    index_data = load_json(KONGMU_DATA_DIR / 'character_index.json')
    source_meta_path = KONGMU_DATA_DIR / 'source_meta.json'
    source_meta = load_json(source_meta_path) if source_meta_path.exists() else {}
    characters: list[dict[str, Any]] = []
    for index, record in enumerate(index_data.get('characters') or []):
        character_id = str(record.get('id') or '')
        detail_path = KONGMU_CHARACTER_DIR / f'{character_id}.json'
        if not detail_path.exists():
            continue
        detail = load_json(detail_path)
        detail['names'] = record.get('names') or {}
        characters.append({'record': record, 'detail': detail, 'source_index': index})

    return {
        'meta': {
            'source': source_meta.get('source') or index_data.get('source') or 'nanoka.cc',
            'game': source_meta.get('game') or index_data.get('game') or GAME_KEY,
            'version': source_meta.get('version') or index_data.get('version') or '',
            'updated_at': source_meta.get('updated_at') or index_data.get('updated_at') or '',
            'character_count': len(characters),
            'cartridge_count': len(cartridges),
            'drive_count': len(drives),
        },
        'characters': characters,
        'cartridges': cartridges,
        'drives': drives,
    }


@lru_cache(maxsize=1)
def get_drives_by_geometry() -> dict[str, dict[str, Any]]:
    raw = get_kongmu_raw_data()
    return {drive['geometry']: compact_drive(drive) for drive in raw['drives']}


def is_test_character(record: dict[str, Any]) -> bool:
    character_id = str(record.get('id') or '')
    if character_id in PUBLIC_KONGMU_CHARACTER_IDS:
        return False
    return bool(record.get('test_character')) or character_id in TEST_CHARACTER_IDS


@lru_cache(maxsize=2)
def get_kongmu_catalog_payload(include_test_characters: bool = False) -> dict[str, Any]:
    raw = get_kongmu_raw_data()
    characters = []
    for item in dedupe_characters(sorted(raw['characters'], key=character_sort_key)):
        if is_test_character(item['record']) and not include_test_characters:
            continue
        compacted = compact_character(item['record'], item['detail'])
        merged_ids = [item_id for item_id in item.get('merged_ids', []) if item_id]
        avatar_items = [item, *(item.get('avatar_choice_items') or [])]
        avatar_choices = [
            compact_character_avatar_choice(avatar_item['record'], avatar_item['detail'])
            for avatar_item in avatar_items
        ]
        compacted['merged_ids'] = merged_ids or [compacted['id']]
        compacted['avatar_choices'] = avatar_choices
        characters.append(compacted)
    cartridges = [compact_cartridge(item) for item in dedupe_cartridges(raw['cartridges'])]
    drives = [compact_drive(item) for item in raw['drives']]
    return {
        'meta': {
            **raw['meta'],
            'character_count': len(characters),
            'cartridge_count': len(cartridges),
        },
        'geometry_sort_order': GEOMETRY_SORT_ORDER,
        'geometry_labels': GEOMETRY_LABELS,
        'characters': characters,
        'cartridges': cartridges,
        'drives': drives,
    }


def get_character_detail(character_id: str, *, include_test_characters: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = get_kongmu_raw_data()
    record = resolve_record(
        [item['record'] for item in raw['characters']],
        character_id,
        ['name', 'names'],
    )
    if is_test_character(record) and not include_test_characters:
        raise AppError('未找到该空幕角色。')
    detail = load_json(KONGMU_CHARACTER_DIR / f'{record["id"]}.json')
    return record, detail


def get_cartridge_detail(cartridge_id: str) -> dict[str, Any]:
    raw = get_kongmu_raw_data()
    return resolve_record(dedupe_cartridges(raw['cartridges']), cartridge_id, ['name', 'aliases'])


def plan_kongmu_layout(character_id: str, cartridge_id: str, *, include_test_characters: bool = False) -> dict[str, Any]:
    raw = get_kongmu_raw_data()
    character_record, character = get_character_detail(
        character_id,
        include_test_characters=include_test_characters,
    )
    cartridge = get_cartridge_detail(cartridge_id)
    equip_slots = character.get('equip_slots') or {}
    slots = equip_slots.get('slots') or []
    if not slots:
        raise AppError('该角色没有可计算的空幕格子。')

    owner_grid_count = int(equip_slots.get('owner_grid_count') or 0)
    drives = [compact_drive(item) for item in raw['drives']]
    result = plan_layouts(slots, drives, cartridge, owner_grid_count)
    required_counts = common_geometry_counts(result.solutions)
    for solution in result.solutions:
        apply_required_flags(solution, required_counts)

    return {
        'character': {
            **compact_character(character_record, character),
            'equip_slots': equip_slots,
        },
        'cartridge': compact_cartridge(cartridge),
        'drives': drives,
        'result': {
            'best_owner_drive_count': result.best_owner_drive_count,
            'searched_solution_count': result.searched_solution_count,
            'total_solution_count': len(result.solutions),
            'solutions': result.solutions,
            'required_drives': [
                drive_chip(geometry, count)
                for geometry, count in sorted(
                    required_counts.items(),
                    key=lambda item: GEOMETRY_SORT_ORDER.index(item[0]) if item[0] in GEOMETRY_SORT_ORDER else 999,
                )
            ],
            'optional_rows': optional_filter_rows(result.solutions),
        },
    }
