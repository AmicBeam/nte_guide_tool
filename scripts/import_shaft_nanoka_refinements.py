#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


NAME_ALIASES = {
    '行进时间外白板': '行进于时间之外',
    '闪耀每一天': '闪耀的每一天',
    '电音狂欢': '「电音」狂欢',
    '银河暂留': '「银河暂留」',
}

# 现有弧盘配置中，下列条目的效果数值对应 Nanoka 精炼 5；
# 其余条目对应精炼 1。旧配装未保存精炼字段时使用这里的映射。
DEFAULT_REFINEMENT_FIVE_ARC_IDS = {
    'arc_57b6c49b66',
    'arc_74578e2ec4',
    'arc_585af155aa',
    'arc_1b828c9173',
    'arc_b3c0b59d01',
    'arc_5fff1d2ae0',
    'arc_e21ec2d386',
    'arc_d24279f95f',
    'arc_edbcee2b65',
    'arc_b99b47cd11',
    'arc_f36fcedbc8',
    'arc_5fb06c9645',
    'arc_4c291f3966',
    'arc_320435ce51',
    'arc_c67cabea48',
    'arc_0a681d8e81',
    'arc_4d19ef5194',
    'arc_7378181f20',
    'arc_2b6d5881ef',
    'arc_92353a7626',
    'arc_32be351505',
    'arc_5fdc5b1847',
    'arc_63736d4ae2',
    'arc_19ee4241da',
    'arc_8591b265fb',
    'arc_18855bc4f1',
}

PANEL_EFFECTS: dict[str, list[tuple[int, str, float]]] = {
    'fork_Arachne': [(0, 'hp_pct', 0.01)],
    'fork_BlackBook': [(0, 'stagger_strength', 1)],
    'fork_BoxingCandy': [(0, 'all_dmg', 0.01)],
    'fork_Butterfly': [(0, 'element:灵', 0.01), (1, 'attach_dmg', 0.01)],
    'fork_Door': [(0, 'atk_pct', 0.01)],
    'fork_GoldWool': [(0, 'element:相', 0.01)],
    'fork_LunarPhase': [(0, 'atk_pct', 0.01)],
    'fork_Nakupeda': [(0, 'hp_pct', 0.01)],
    'fork_PaperPlane': [(0, 'skill_dmg', 0.01), (0, 'ultimate_dmg', 0.01)],
    'fork_PoliceRat': [(0, 'atk_pct', 0.01), (1, 'all_dmg', 0.01)],
    'fork_Prokaryon': [(0, 'basic_dmg', 0.01)],
    'fork_Rose': [(0, 'atk_pct', 0.01)],
    'fork_ThiefCandy': [],
    'fork_TigerTally': [(0, 'atk_pct', 0.01)],
    'fork_Time': [(0, 'atk_pct', 0.01)],
    'fork_Whale': [(0, 'atk_pct', 0.01)],
    'fork_appliance': [(0, 'skill_dmg', 0.01)],
    'fork_jiaojuan': [(0, 'stagger_strength', 1)],
    'fork_koinobori': [
        (0, 'atk_pct', 0.01),
        (1, 'def_pct', 0.01),
        (2, 'hp_pct', 0.01),
    ],
    'fork_mofeikesi': [(0, 'energy_recharge', 0.01)],
    'fork_moon': [(0, 'element:魂', 0.01)],
    'fork_rishi': [(0, 'atk_pct', 0.01)],
    'fork_tuansanlang': [(0, 'stagger_strength', 1)],
    'fork_worldrain': [(0, 'skill_dmg', 0.01), (0, 'ultimate_dmg', 0.01)],
    'fork_wuhuakuang': [(0, 'atk_pct', 0.01), (1, 'def_pct', 0.01)],
    'fork_wushoutieyu': [(3, 'element:相', 0.01)],
    'fork_yuren': [(0, 'hp_pct', 0.01), (1, 'def_pct', 0.01)],
}

BUFF_EFFECTS: dict[str, tuple[int, dict[str, tuple[int, float] | tuple[int, float, int]]]] = {
    'arc_time_outside_recorded_crit_def': (
        0,
        {'crit_dmg': (1, 0.01), 'def_ignore': (3, 0.01)},
    ),
    'arc_rain_e_harmony': (0, {'harmony_strength': (1, 1)}),
    'arc_fearless_q_atk': (0, {'atk_pct': (0, 0.01)}),
    'arc_fearless_full_q_atk': (0, {'atk_pct': (0, 0.01)}),
    # 面板常驻附着增伤已写入弧盘面板；Buff 只记录 Q 后替换值与常驻值的差额。
    'arc_reality_shelter_q_attach': (0, {'attach_dmg': (2, 0.01, 1)}),
    'arc_time_thief_e_stagger': (0, {'stagger_strength': (0, 1)}),
    'arc_first_step_e_atk': (0, {'atk_pct': (0, 0.01)}),
    'arc_wrong_door_heal_spirit': (0, {'element_dmg_灵': (2, 0.01)}),
    'arc_last_rose_crit_stack': (0, {'crit_dmg': (1, 0.01)}),
    'arc_last_rose_e_full_stack': (0, {'crit_dmg': (1, 0.01)}),
    'arc_speed_cotton_front_atk': (0, {'atk_pct': (1, 0.01)}),
    'arc_speed_cotton_full_front_atk': (0, {'atk_pct': (1, 0.01)}),
    'arc_thinking_cat_fons_light': (0, {'element_dmg_光': (1, 0.1)}),
    'arc_thinking_cat_full_fons_light': (0, {'element_dmg_光': (1, 0.1)}),
    'arc_bit_game_background_front_atk': (0, {'atk_pct': (0, 0.01)}),
    'arc_bit_game_background_damage_front_atk_stack': (0, {'atk_pct': (1, 0.01)}),
    'arc_bit_game_foreground_psyche_dmg': (0, {'element_dmg_魂': (3, 0.01)}),
    'arc_bit_game_basic_psyche_stack': (0, {'element_dmg_魂': (4, 0.01)}),
    'arc_good_dog_q_team_atk': (0, {'atk_pct': (2, 0.01)}),
    'arc_good_dog_q_control_extra_atk': (0, {'atk_pct': (3, 0.01)}),
    'arc_dazzle_support_all_dmg': (0, {'all_dmg': (1, 0.01)}),
    'arc_overhead_e_atk': (0, {'atk_pct': (0, 0.01)}),
    'arc_overhead_delay_element': (0, {'element_dmg_相': (1, 0.01)}),
    'arc_ready_eq_basic_dodge': (
        0,
        {'basic_dmg': (1, 0.01), 'dodge_counter_dmg': (1, 0.01)},
    ),
    'arc_galaxy_soul_hit_crit': (0, {'crit_dmg': (1, 0.01)}),
    'arc_ora_basic_stack': (0, {'basic_dmg': (1, 0.01)}),
    'arc_crimson_mirage_q_light_def': (
        0,
        {'element_dmg_光': (1, 0.01), 'def_ignore': (2, 0.01)},
    ),
    'arc_crimson_mirage_full_q_light_def': (
        0,
        {'element_dmg_光': (1, 0.01), 'def_ignore': (2, 0.01)},
    ),
    'arc_searched_eq_crit': (0, {'crit_dmg': (1, 0.01)}),
    'arc_fierce_cotton_crit': (0, {'crit_dmg': (0, 0.01)}),
    'arc_fierce_cotton_full_crit': (0, {'crit_dmg': (0, 0.01)}),
    'arc_camellia_hp_loss_crit': (0, {'crit_dmg': (1, 0.01)}),
    'arc_waltz_q_mind': (0, {'mind_dmg': (1, 0.01)}),
    'arc_unyielding_low_hp_bonus': (0, {'all_dmg': (1, 0.01, 0)}),
    'arc_unyielding_full_low_hp_bonus': (0, {'all_dmg': (1, 0.01, 0)}),
    'arc_spring_shield_atk': (0, {'atk_pct': (0, 0.01)}),
    'arc_bitter_medicine_hit_def': (0, {'def_pct': (0, 0.01)}),
    'arc_danger_game_stagger': (0, {'stagger_strength': (0, 1)}),
}

STAT_KEYS = {
    'AtkUp': 'atk_pct',
    'ChargeGetEfficiencyBase': 'energy_recharge',
    'CritBase': 'crit_rate',
    'CritDamageBase': 'crit_dmg',
    'DefUp': 'def_pct',
    'HPMaxUp': 'hp_pct',
    'UnbalIntensityBase': 'stagger_strength',
}


def clean_name(value: str) -> str:
    name = NAME_ALIASES.get(value, value)
    name = re.sub(r'(?:·满精|满精|满|白板)$', '', name)
    return name.replace('「', '').replace('」', '').replace('之棉', '之绵').strip()


def number(value: Any) -> float:
    text = str(value).strip().replace('%', '')
    return float(text)


def rounded(value: float) -> float:
    return round(value, 10)


def effect_value(weapon: dict[str, Any], effect_index: int, level: int) -> float:
    values = weapon.get('effect', {}).get('values') or []
    return number(values[effect_index]['values'][level - 1])


def base_panel(weapon: dict[str, Any]) -> dict[str, float]:
    panel: dict[str, float] = {}
    for stat in weapon.get('stats') or []:
        key = STAT_KEYS.get(str(stat.get('id_stats') or ''))
        if not key:
            continue
        value = number((stat.get('values') or [0])[-1])
        panel[key] = rounded(value * (0.01 if stat.get('b_is_percent') else 1))
    return panel


def level_panel(weapon: dict[str, Any], level: int) -> tuple[dict[str, float], dict[str, float]]:
    panel = base_panel(weapon)
    elements: dict[str, float] = {}
    for effect_index, key, factor in PANEL_EFFECTS.get(str(weapon.get('id') or ''), []):
        target = elements if key.startswith('element:') else panel
        target_key = key.split(':', 1)[-1]
        target[target_key] = rounded(target.get(target_key, 0) + effect_value(weapon, effect_index, level) * factor)
    return panel, elements


def buff_values(
    weapon: dict[str, Any],
    rule_id: str,
    level: int,
) -> dict[str, float]:
    _, mappings = BUFF_EFFECTS[rule_id]
    effects: dict[str, float] = {}
    for key, mapping in mappings.items():
        effect_index, factor, *subtract_index = mapping
        raw = effect_value(weapon, effect_index, level)
        if subtract_index:
            raw -= effect_value(weapon, subtract_index[0], level)
        effects[key] = rounded(raw * factor)
    return effects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--nanoka-dir', type=Path, required=True)
    parser.add_argument('--arcs', type=Path, required=True)
    parser.add_argument('--buffs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    arcs = json.loads(args.arcs.read_text(encoding='utf-8'))
    buffs = json.loads(args.buffs.read_text(encoding='utf-8'))
    weapons = [
        json.loads(path.read_text(encoding='utf-8'))
        for path in sorted(args.nanoka_dir.glob('*.json'))
    ]
    weapon_by_name = {clean_name(str(item.get('name') or '')): item for item in weapons}
    buff_ids_by_arc: dict[str, list[str]] = {}
    for buff in buffs:
        rule_id = str(buff.get('id') or '')
        if rule_id not in BUFF_EFFECTS:
            continue
        for provider in buff.get('providers') or []:
            if provider.get('kind') == 'arc':
                buff_ids_by_arc.setdefault(str(provider.get('id') or ''), []).append(rule_id)

    output: dict[str, Any] = {
        'source': 'nanoka.cc',
        'version': '1.2',
        'generated_from': 'https://static.nanoka.cc/nte/1.2/zh/weapon/{id}.json',
        'arcs': {},
    }
    unmatched: list[str] = []
    for arc in arcs:
        arc_id = str(arc.get('id') or '')
        weapon = weapon_by_name.get(clean_name(str(arc.get('name') or '')))
        if not weapon:
            unmatched.append(str(arc.get('name') or arc_id))
            continue
        levels: dict[str, Any] = {}
        for level in range(1, 6):
            panel, elements = level_panel(weapon, level)
            raw_values = [
                (effect.get('values') or [None] * 5)[level - 1]
                for effect in weapon.get('effect', {}).get('values') or []
            ]
            levels[str(level)] = {
                'effect_values': raw_values,
                'panel_modifiers': panel,
                'element_dmg': elements,
                'buff_effects': {
                    rule_id: buff_values(weapon, rule_id, level)
                    for rule_id in buff_ids_by_arc.get(arc_id, [])
                },
            }
        output['arcs'][arc_id] = {
            'nanoka_id': weapon['id'],
            'name': weapon['name'],
            'default_level': 5 if arc_id in DEFAULT_REFINEMENT_FIVE_ARC_IDS else 1,
            'source_url': f"https://static.nanoka.cc/nte/1.2/zh/weapon/{weapon['id']}.json",
            'effect_name': weapon.get('effect', {}).get('name') or '',
            'effect_description': weapon.get('effect', {}).get('description') or '',
            'levels': levels,
        }
    if unmatched:
        raise SystemExit(f"未匹配弧盘：{', '.join(unmatched)}")
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
