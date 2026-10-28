#!/usr/bin/env python3
"""Print the configured additive damage groups, including dynamic and refinement variants."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'app/modules/shaft/static/data'
TYPES = {'basic_dmg': '普攻', 'dodge_counter_dmg': '闪反', 'skill_dmg': '变轨',
         'ultimate_dmg': '终结', 'follow_dmg': '追击/追加攻击', 'attach_dmg': '附着'}
SPECIAL = {'character_sagiri_a6_negative_damage', 'character_canhong_f_annihilation_dot_layers',
           'character_lingke_harmonic_joint_damage', 'character_lingke_q_burst_negative_damage'}


def load(name):
    return json.loads((DATA / name).read_text())


def group(key):
    if key == 'all_dmg': return '通用增伤'
    if key == 'action_dmg' or key in TYPES: return '动作增伤'
    if key == 'other_dmg': return '其他增伤'
    if key == 'mind_dmg' or key.startswith('element_dmg_'): return '属性增伤'
    return ''


def scan():
    rows = defaultdict(lambda: {'values': set(), 'locations': set()})
    def walk(obj, file, name, owner, path=''):
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_path = f'{path}.{key}'.strip('.')
                effect = value if key in ('effect_key', 'modifier_key') and isinstance(value, str) else key
                if group(effect) and (key in ('effect_key', 'modifier_key') or isinstance(value, (int, float)) and value != 0
                                      or isinstance(value, dict) and ('source' in value or 'unit_value' in value)):
                    row = rows[(file, name, owner, effect)]
                    row['locations'].add(key_path)
                    row['values'].add(json.dumps(obj if key == 'effect_key' else value, ensure_ascii=False, sort_keys=True))
                if key == 'element_dmg' and isinstance(value, dict):
                    walk({f'element_dmg_{e}': amount for e, amount in value.items()}, file, name, owner, key_path)
                else:
                    walk(value, file, name, owner, key_path)
        elif isinstance(obj, list):
            for i, value in enumerate(obj): walk(value, file, name, owner, f'{path}[{i}]')

    buffs = load('buffs.json')
    names = {b['id']: b['name'] for b in buffs}
    for file in ('buffs.json', 'arcs.json', 'cartridges.json', 'actions.json'):
        for obj in load(file):
            name = ('%s·%s' % (obj['character_name'], obj['name'])) if file == 'actions.json' else obj['name']
            fields = {k: v for k, v in obj.items() if k in ('effects', 'modifiers', 'self_modifiers', 'activation', 'dynamic_effects', 'element_damage_bonus', 'element_dmg')}
            walk(fields, file, name, obj['id'])
    for arc_id, arc in load('arc_refinements.json')['arcs'].items():
        for level, config in arc.get('levels', {}).items():
            walk({'panel_modifiers': config.get('panel_modifiers', {}), 'element_dmg': config.get('element_dmg', {})}, 'arc_refinements.json', arc.get('name', arc_id), arc_id, f'精{level}.modifiers')
            for bid, effects in config.get('buff_effects', {}).items():
                if bid in names:
                    walk(effects, 'arc_refinements.json', names[bid], bid, f'精{level}.buff_effects')
    walk(load('formula_constants.json'), 'formula_constants.json', '配装词条与默认面板', 'build')
    # Runtime-only effects are explicit and checked against engine literals.
    engine = (ROOT / 'app/modules/shaft/static/js/shaft_engine.js').read_text()
    assert 'mergeMods(buffModifiers, { other_dmg: 0.5 });' in engine
    for name, owner, key, value in [('灵可E觉·无法环合增伤', 'lingke_joint', 'other_dmg', '0.5'),
                                    ('增伤赐福', 'blessing', 'all_dmg', '0.15')]:
        rows[('shaft_engine.js', name, owner, key)]['values'].add(value)
    return rows


def report():
    rows = scan()
    out = ['# 排轴增伤分类核对', '',
           '三类是同一个增伤乘区内部的展示拆分，仍按 `1＋通用增伤＋动作增伤＋其他增伤＋本次属性增伤` 相乘；不把三类彼此相乘。', '',
           '| 项目 | 旧统计 | 三类设计 |', '| --- | --- | --- |',
           '| 全伤、通伤词条、增伤赐福及全动作增益 | 通用增伤 | 通用增伤 |',
           '| 限定技能/动作的全伤配置及动作自身强化 | 通用增伤 | 动作增伤 |',
           '| 普攻、闪反、变轨、终结、追击/追加攻击、附着 | 其他增伤 | 动作增伤 |',
           '| 残虹狩 | 其他增伤 | 其他增伤 |',
           '| 负面状态计数、残虹灭末段层蚀、灵可谐频与无法环合加成 | 通用增伤 | 其他增伤 |',
           '| 光、灵、咒、暗、魂、相 | 属性增伤 | 属性增伤，按实际伤害属性选择 |',
           '| 心灵 | 其他增伤 | 第7种属性增伤，保留存档键 mind_dmg |', '',
           '属性是伤害属性，不改变角色所属六属性和环合组合。暴击、暴伤、最终增伤、基础倍率提升、减防、穿防、减抗、环合强度和倾陷专用增伤均不并入这三类。', '',
           '「静候彩铃」按通用光属性减抗处理，角色伤害、周期伤害、对应属性的环合与倾陷均可受益；覆纹与浸染保留既有本体增幅算法。', '',
           '以下为当前有效配置的非零条目及动态输出；精炼档位合并在同一行，数值使用内部小数比例。条件、对象和层数仍由各规则控制，不代表整轴全程生效。旧统计列表示拆分前的归属。历史 characters.modifiers 和纯说明文本不参与计算，未列入。', '',
           '| 配置 | 条目 | 规则/记录ID | 字段 | 数值/动态规则 | 旧统计 | 三类设计 |', '| --- | --- | --- | --- | --- | --- | --- |']
    counts = defaultdict(int)
    for (file, name, owner, key), info in sorted(rows.items()):
        category = group(key)
        old = '其他增伤' if key in TYPES or key == 'mind_dmg' or key == 'other_dmg' and owner not in SPECIAL and owner != 'lingke_joint' else '通用增伤'
        if category == '属性增伤' and key != 'mind_dmg': old = '属性增伤'
        counts[category] += 1
        value = ' / '.join(sorted(info['values'])).replace('|', '\\|')
        out.append(f'| {file} | {name} | `{owner}` | `{key}` | {value} | {old} | {category} |')
    out += ['', '条目数按文件、规则与字段去重（不是独立乘区或可同时叠加数量）：' + '，'.join(f'{k} {v}' for k,v in sorted(counts.items())) + '。', '',
            '重新生成：`python3 scripts/audit_shaft_damage_bonuses.py > docs/shaft-damage-bonus-classification.md`。', '']
    return '\n'.join(out)


if __name__ == '__main__':
    print(report(), end='')
