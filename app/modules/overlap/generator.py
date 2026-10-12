"""Offline route enumeration. No Flask, DAO or sibling-module services."""
from __future__ import annotations

import hashlib
import json
import math
import re
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = Path(__file__).resolve().parent
SOURCE = ROOT / 'modules' / 'shaft' / 'static' / 'data'
BASES = ('atk', 'hp', 'def', 'flat')


def number(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'时间/收益必须是有限非负数：{value}')
    return result


def round_value(value):
    return round(value, 6)


def duration(action):
    value = number(action['duration_seconds'])
    if action.get('duration_pending') or (
        value == 0 and not action.get('is_background_damage')
        and not action.get('is_time_stop_zero') and not action.get('duration_implemented')
    ):
        return None
    return value


def switch_points(action, override, normal_seconds=None, replace_detach=False):
    """Manual background placement alone never grants an early switch."""
    result = list(override.get('switch_points', []))
    if normal_seconds is not None:
        result.append({'id':'normal', 'after_seconds':number(normal_seconds), 'label':'正常切人',
                       'source':'用户2026-10-12：动作开始后0.2秒正常切人，完整伤害与资源继续在后台结算。'})
    if action.get('can_detach') and not replace_detach:
        result.append({
            'id': 'detached', 'after_seconds': action['detached_duration_seconds'],
            'label': '脱手切人', 'source': 'actions.json：can_detach / detached_duration_seconds',
        })
    if action.get('is_background_damage'):
        result.append({'id': 'background', 'after_seconds': 0, 'label': '原生后台',
                       'source': 'actions.json：is_background_damage'})
    if action.get('locks_foreground_switch') and result:
        raise ValueError(f"{action['name']}锁定前台，不能同时配置提前切人")
    ids = set()
    for point in result:
        if not point.get('source') or not point.get('id') or point['id'] in ids:
            raise ValueError(f"{action['name']}合轴点须有唯一id与source")
        ids.add(point['id'])
        point['after_seconds'] = number(point['after_seconds'])
        if duration(action) is not None and point['after_seconds'] > duration(action):
            raise ValueError(f"{action['name']}合轴点不能晚于完整释放结束")
    return result


def build_graph(actions, rule):
    by_name = {a['name']: a for a in actions}
    if len(by_name) != len(actions):
        raise ValueError('同角色动作名须唯一')
    edges = {name: [] for name in by_name}
    # Sequential basic/heavy chains are the catalog's baseline route, never cross-form.
    for name in by_name:
        match = re.fullmatch(r'(.*)([az])(\d+)(.*)', name)
        if match:
            prefix, kind, index, suffix = match.groups()
            next_name = f'{prefix}{kind}{int(index) + 1}{suffix}'
            if next_name in by_name:
                edges[name].append({'from': name, 'to': next_name,
                                    'source': '动作目录连续段；按完整动作顺接'})
    for edge in rule.get('transitions', []):
        if edge.get('from') not in by_name or edge.get('to') not in by_name:
            raise ValueError(f'衔接引用不存在的动作：{edge}')
        if not edge.get('source'):
            raise ValueError('衔接必须记录source')
        if 'after_seconds' in edge:
            number(edge['after_seconds'])
        prefix = edge.get('requires_prefix', [])
        if prefix and (prefix[-1] != edge['from'] or any(n not in by_name for n in prefix)):
            raise ValueError(f'衔接前置路径无效：{prefix}')
        edges[edge['from']] = [e for e in edges[edge['from']] if e['to'] != edge['to']]
        edges[edge['from']].append(dict(edge))
    forbidden = rule.get('preinput_forbidden_actions', {})
    if set(forbidden) - set(by_name) or any(not source for source in forbidden.values()):
        raise ValueError('禁止预输入的动作须存在并记录source')
    for successors in edges.values():
        for edge in successors:
            if edge['to'] in forbidden:
                edge['can_preinput'] = False
                edge['source'] += ' ' + forbidden[edge['to']]
    return by_name, edges


def enumerate_paths(by_name, edges, max_actions, max_uses=None, start_actions=None):
    """All simple directed paths, bounded and deterministic; no repeat/CD invention."""
    def walk(names, links):
        candidates = [e for e in edges[names[-1]] if names.count(e['to']) < (max_uses or {}).get(e['to'], 1) and (
            not e.get('requires_prefix') or names == e['requires_prefix']
        ) and (
            not e.get('requires_previous') or len(names) >= 2 and names[-2] in e['requires_previous']
        )]
        automatic = [edge for edge in candidates if edge.get('automatic')]
        if automatic:
            candidates = automatic
        else:
            yield names, links
        if len(names) >= max_actions:
            if candidates:
                raise ValueError(f'路线超过max_actions，不能静默截断：{names}')
            return
        for edge in candidates:
            yield from walk([*names, edge['to']], [*links, edge])
    for name in start_actions if start_actions is not None else by_name:
        yield from walk([name], [])


def calculate_row(names, links, by_name, points, final_point=None, preinput=False, dodge_indices=(), default_delay=0.3):
    actions = [by_name[name] for name in names]
    for index in dodge_indices:
        actions[index] = {
            'name': f'{names[index]}闪', 'duration_seconds': 0.7, 'dodge_replaces': names[index],
            'multipliers': {}, 'duration_implemented': True,
        }
    durations = [duration(a) for a in actions]
    for index, edge in enumerate(links):
        if edge.get('detach_previous'):
            if not actions[index].get('can_detach'):
                raise ValueError('脱手衔接要求原动作支持can_detach')
            durations[index] = number(actions[index]['detached_duration_seconds'])
    starts = [0.0]
    inputs = [0.0]
    for index, edge in enumerate(links):
        previous = starts[-1]
        offset = edge.get('after_seconds', durations[index])
        delay = number(edge.get('input_delay_seconds', default_delay))
        input_offset = durations[index] if edge.get('can_preinput') is False else delay
        inputs.append(None if edge.get('automatic') or previous is None or input_offset is None else previous + input_offset)
        starts.append(None if previous is None or offset is None else max(previous + offset, inputs[-1] or previous))
    ends = [None if s is None or d is None else s + d for s, d in zip(starts, durations)]
    known = all(e is not None for e in ends)
    release_end = max(ends) if known else None
    foreground = release_end
    constraint_notes = []
    input_index = next((i + 1 for i in range(len(links) - 1, -1, -1) if not links[i].get('automatic')), None)
    if preinput:
        foreground = inputs[input_index]
        extra_delay = number(links[input_index-1].get('switch_delay_seconds', 0))
        if foreground is not None:
            foreground += extra_delay
        if extra_delay:
            constraint_notes.append(f'预输入后延迟{extra_delay:g}秒切人')
    if final_point is not None:
        foreground = None if starts[-1] is None else starts[-1] + final_point['after_seconds']
        # Earlier active actions may postpone the last action's proposed switch point.
        for index, action in enumerate(actions[:-1]):
            if starts[index] is None or ends[index] is None:
                foreground = None
                break
            allowed = points.get(action['name'], [])
            required_until = starts[index] + min(p['after_seconds'] for p in allowed) if allowed else ends[index]
            if foreground is not None and required_until > foreground:
                foreground = required_until
                constraint_notes.append(f"等待{action['name']}允许离场")
    for action, end in zip(actions, ends):
        if action.get('locks_foreground_switch') and foreground is not None:
            foreground = None if end is None else max(foreground, end)
    totals = {base: round_value(sum(float(a.get('multipliers', {}).get(base, 0)) for a in actions)) for base in BASES}
    harmony = sum(float(a.get('harmony', 0)) for a in actions)
    start_harmony = sum(float(a.get('harmony_on_start', 0)) for a in actions)
    energy = sum(float(a.get('energy_gain', 0)) for a in actions)
    returned = sum(float(a.get('energy_return', 0)) for a in actions)
    cost = sum(float(a.get('energy_cost', 0)) for a in actions)
    stagger = sum(float(a.get('stagger', 0)) for a in actions)
    resources = {}
    for a in actions:
        for key, sign in (('personal_resource_gain', 1), ('personal_resource_cost', -1)):
            for resource, value in a.get(key, {}).items():
                resources[resource] = resources.get(resource, 0) + sign * float(value)
    def rate(value):
        return round_value(value / foreground) if foreground is not None and foreground > 0 else None
    row = {
        'id': hashlib.sha256(json.dumps([names, final_point and final_point['id'], preinput, dodge_indices], ensure_ascii=False).encode()).hexdigest()[:16],
        'actions': [a['name'] for a in actions], 'original_actions': names,
        'kind': '合轴' if final_point or preinput else ('连招' if len(names) > 1 else '单动作'),
        'switch_method': 'preinput' if preinput else final_point['id'] if final_point else 'full',
        'has_dodge': bool(dodge_indices),
        'preinput_action_indices': [input_index-1] if preinput else [],
        'operation': ((actions[input_index-1]['name'].removesuffix('远') + '预' + actions[input_index]['name'].removesuffix('远')) +
                      ('（自动接' + '→'.join(a['name'] for a in actions[input_index+1:]) + '）' if input_index < len(actions)-1 else '')) if preinput else ' → '.join(a['name'] for a in actions),
        'switch_label': ('预输入合轴' if preinput else final_point.get('label', final_point['id']) if final_point else '完整释放'),
        'foreground_seconds': None if foreground is None else round_value(foreground),
        'switch_seconds': None if foreground is None else round_value(foreground),
        'foreground_intervals': [] if foreground is None else [[0, round_value(foreground)]],
        'release_end_seconds': None if release_end is None else round_value(release_end),
        'background_seconds': None if foreground is None or release_end is None else round_value(max(0, release_end - foreground)),
        'timeline': [{'action': a['name'], 'input_seconds': i, 'start_seconds': s, 'end_seconds': e,
                      'duration_seconds': d, 'automatic': index > 0 and bool(links[index-1].get('automatic')),
                      'locks_foreground_switch': bool(a.get('locks_foreground_switch')),
                      'dodge_replaces': a.get('dodge_replaces'),
                      'detached': bool(index < len(links) and links[index].get('detach_previous') or
                                       index == len(actions)-1 and final_point and final_point['id']=='detached')}
                     for index, (a, i, s, e, d) in enumerate(zip(actions, inputs, starts, ends, durations))],
        'multipliers': totals, 'multiplier_rates': {b: rate(v) for b, v in totals.items()},
        'harmony': round_value(harmony), 'harmony_on_start': round_value(start_harmony),
        'harmony_rate': rate(harmony + start_harmony),
        'energy_gain': round_value(energy), 'energy_return': round_value(returned),
        'energy_cost': round_value(cost), 'energy_net': round_value(energy + returned - cost),
        'energy_rate': rate(energy + returned), 'stagger': round_value(stagger), 'stagger_rate': rate(stagger),
        'personal_resources': {k: round_value(v) for k, v in resources.items()},
        'status': '可计算' if known and foreground is not None else '时长待补',
        'notes': constraint_notes,
        'sources': list(dict.fromkeys([e['source'] for e in links] + ([final_point['source']] if final_point else []) +
                                     (['用户2026-10-12：动作开始后0.3秒预输入即可切人，完成当前和下一动作。'] if preinput else []) +
                                     (['用户2026-10-12：0.7秒闪替代超过0.7秒的中间动作，跳过动作全部收益归零。'] if dodge_indices else []))),
    }
    return project_placements(row)


def project_placements(row):
    """Presentation uses offline foreground intervals, never browser-derived rules."""
    for index, step in enumerate(row['timeline']):
        start = step['start_seconds']
        step['placement'] = 'pending' if start is None else 'foreground' if any(
            left - 1e-9 <= start < right - 1e-9 for left, right in row['foreground_intervals']
        ) else 'background'
        step['preinput'] = step['input_seconds'] is not None and start is not None and step['input_seconds'] < start - 1e-9
        step['preinput_before_switch'] = index in row['preinput_action_indices']
    row['foreground_segments'] = [
        {'number':index+1,'start_seconds':left,'end_seconds':right,'duration_seconds':round_value(right-left)}
        for index,(left,right) in enumerate(row['foreground_intervals'])
    ]
    row['overlap_windows'] = []
    if row['release_end_seconds'] is None:
        return row
    intervals = row['foreground_intervals']
    for index, (left, right) in enumerate(intervals):
        requires_return = index+1 < len(intervals)
        if not requires_return:
            continue
        end = intervals[index+1][0]
        if end <= right + 1e-9:
            continue
        source_indices = [i for i, step in enumerate(row['timeline']) if step['start_seconds'] is not None
                          and left-1e-9 <= step['start_seconds'] < right-1e-9]
        window = {
            'number': len(row['overlap_windows'])+1,
            'start_seconds': round_value(right), 'end_seconds': round_value(end),
            'duration_seconds': round_value(end-right), 'requires_return': requires_return,
            'after_action_index': source_indices[-1] if source_indices else None,
        }
        if requires_return:
            window['return_start_seconds'] = round_value(right+1.2)
            window['return_end_seconds'] = round_value(end)
        row['overlap_windows'].append(window)
    return row


def advanced_rows(names, links, by_name, points, options, final_point=None):
    """Return at queued tail end, retain combo; no teammate benefits are invented."""
    explicit = [edge for edge in links if not edge.get('automatic')]
    if final_point is None and (not explicit or not explicit[-1].get('can_preinput', True)):
        return
    final = calculate_row(names, links, by_name, points, final_point, preinput=final_point is None, **options)
    if final['status'] != '可计算':
        return
    candidates = []
    for split in range(1, len(names)):
        # The first queued segment must be a basic attack chain; resume its next stage.
        if not re.fullmatch(r'.*a\d.*', names[split - 1]) or not re.fullmatch(r'.*a\d.*', names[split]):
            continue
        if split >= 2 and links[split-2].get('can_preinput', True) and not links[split-2].get('automatic'):
            candidates.append((split, final['timeline'][split-1]['input_seconds'] + number(links[split-2].get('switch_delay_seconds',0)), 'preinput'))
        for point in points[names[split-1]]:
            if point['id']=='normal' and split-1 not in options['dodge_indices']:
                candidates.append((split, final['timeline'][split-1]['start_seconds']+point['after_seconds'], 'normal'))
    for split, first_switch, first_method in candidates:
        first_end = final['timeline'][split - 1]['end_seconds']
        if first_end - first_switch < 1.2 - 1e-9:
            continue
        last_switch = final['switch_seconds']
        if last_switch < first_end:
            continue
        row = dict(final)
        row['timeline'] = [dict(step) for step in final['timeline']]
        row['timeline'][split]['input_seconds'] = first_end
        row['preinput_action_indices'] = sorted(set([*final['preinput_action_indices'], *([split-2] if first_method=='preinput' else [])]))
        row['id'] += f'-return-{split}-{first_method}'
        row['kind'] = '分段合轴'
        row['switch_label'] = '衔接动作合轴'
        clean = lambda name: name.removesuffix('远')
        first_operation = (f'{clean(final["actions"][split-2])}预{clean(final["actions"][split-1])}' if first_method=='preinput'
                           else f'{clean(final["actions"][split-1])}正常切人')
        last_operation = final['operation'] if final_point is None else f'{clean(final["actions"][-1])}正常切人'
        row['operation'] = f'{first_operation} / {last_operation}'
        row['window_1'] = {
            'start_seconds': round_value(first_switch), 'end_seconds': round_value(first_end),
            'duration_seconds': round_value(first_end - first_switch),
            'return_start_seconds': round_value(first_switch + 1.2),
            'return_end_seconds': round_value(first_end),
        }
        occupied = first_switch + last_switch - first_end
        row['foreground_seconds'] = round_value(occupied)
        row['foreground_intervals'] = [[0, round_value(first_switch)], [round_value(first_end), last_switch]]
        if any(step.get('locks_foreground_switch') and not any(
            left <= step['start_seconds'] + 1e-9 and right >= step['end_seconds'] - 1e-9
            for left, right in row['foreground_intervals']
        ) for step in row['timeline']):
            continue
        row['background_seconds'] = round_value(row['release_end_seconds'] - occupied)
        row['multiplier_rates'] = {b: round_value(v / occupied) for b, v in row['multipliers'].items()}
        row['harmony_rate'] = round_value((row['harmony'] + row['harmony_on_start']) / occupied)
        row['energy_rate'] = round_value((row['energy_gain'] + row['energy_return']) / occupied)
        row['stagger_rate'] = round_value(row['stagger'] / occupied)
        row['sources'] = [*row['sources'], '用户2026-10-12：切走后至少1.2秒才能切回，预输入尾段结束后可继续连段。']
        yield project_placements(row)


def generate(characters, actions, rules):
    if rules.get('schema_version') != 1:
        raise ValueError('不支持的规则版本')
    max_actions = int(rules.get('max_actions', 8))
    if not 1 <= max_actions <= 16:
        raise ValueError('max_actions应为1–16')
    result = []
    for character in characters:
        if character['name'] not in rules.get('characters', {}):
            continue
        rule = rules.get('characters', {}).get(character['name'], {})
        current = [dict(a) for a in actions if a['character_id'] == character['id'] and a['name'] in rule['action_names']]
        if set(rule['action_names']) - {a['name'] for a in current}:
            raise ValueError(f"{character['name']}存在未识别的动作")
        override_map = rule.get('actions', {})
        if set(override_map) - {a['name'] for a in current}:
            raise ValueError(f"{character['name']}存在未识别的动作覆写")
        for action in current:
            override = override_map.get(action['name'], {})
            if 'locks_foreground_switch' in override:
                if not isinstance(override['locks_foreground_switch'], bool) or not override.get('source'):
                    raise ValueError('前台锁定覆写须有布尔值与source')
                action['locks_foreground_switch'] = override['locks_foreground_switch']
            if 'duration_seconds' in override:
                if not override.get('source'):
                    raise ValueError('耗时覆写须注明source')
                action.update(duration_seconds=number(override['duration_seconds']), duration_pending=False,
                              duration_implemented=True, duration_source=override['source'])
        by_name, edges = build_graph(current, rule)
        start_actions = rule.get('start_actions', [name for name in by_name if re.fullmatch(r'.*a1.*|[魔幻]?e|q\d*', name)])
        if not start_actions or set(start_actions) - set(by_name):
            raise ValueError(f"{character['name']}起手动作配置无效")
        normal_actions = rule.get('normal_switch_actions', [])
        if set(normal_actions)-set(by_name):
            raise ValueError('正常切人动作必须存在')
        points = {a['name']: switch_points(a, override_map.get(a['name'], {}),
                  rules.get('normal_switch_seconds',0.2) if a['name'] in normal_actions else None,
                  bool(rule.get('normal_replaces_detach'))) for a in current}
        inventory = []
        for action in current:
            gaps = []
            if duration(action) is None:
                gaps.append('完整动作时长')
            if re.search(r'z|蓄力|分支|清明梦', action['name']) and not edges[action['name']]:
                gaps.append('重击后衔接')
            inventory.append({
                'name': action['name'], 'action_id': action['id'], 'type': action['action_type'],
                'duration_seconds': duration(action), 'switch_points': points[action['name']],
                'preinput_seconds': rules.get('default_input_delay_seconds', 0.3) if edges[action['name']] else None,
                'successors': edges[action['name']], 'gaps': gaps,
                'can_background_override': bool(action.get('can_background_override')),
                'locks_foreground_switch': bool(action.get('locks_foreground_switch')),
                'can_start': action['name'] in start_actions,
                'preinput_forbidden': action['name'] in rule.get('preinput_forbidden_actions', {}),
                'source': action.get('source_note') or action.get('duration_source') or '仓库当前动作数据',
                'conditions': {k: v for k, v in action.items() if k.startswith('required_') or k in (
                    'forbidden_buff_key', 'personal_resource_threshold', 'cooldown_ticks', 'cooldown_group',
                    'periodic_damage', 'pre_input_node', 'energy_cost', 'personal_resource_cost',
                )},
            })
        rows = []
        for names, links in enumerate_paths(by_name, edges, max_actions, rule.get('max_uses'), start_actions):
            candidates = [i for i in range(1, len(names) - 1) if rule.get('dodge_preserves_combo')
                          and re.fullmatch(r'.*a\d.*', names[i]) and duration(by_name[names[i]]) is not None
                          and duration(by_name[names[i]]) > 0.7 and not links[i].get('detach_previous')]
            for count in range(len(candidates) + 1):
                for skipped in combinations(candidates, count):
                    options = {'dodge_indices': skipped, 'default_delay': rules.get('default_input_delay_seconds', 0.3)}
                    variant_links = [dict(edge) for edge in links]
                    for index, edge in enumerate(variant_links):
                        if index in skipped or index+1 in skipped:
                            edge['can_preinput']=False
                            if index+1 in skipped:
                                edge['automatic']=False
                            edge['source'] += ' 用户2026-10-12：闪不可被预输入，闪完才能输入下一段。'
                    rows.append(calculate_row(names, variant_links, by_name, points, **options))
                    explicit = [edge for edge in variant_links if not edge.get('automatic')]
                    if explicit and explicit[-1].get('can_preinput', True):
                        rows.append(calculate_row(names, variant_links, by_name, points, preinput=True, **options))
                        rows.extend(advanced_rows(names, variant_links, by_name, points, options))
                    for point in points[names[-1]]:
                        rows.append(calculate_row(names, variant_links, by_name, points, point, **options))
                        if point['id']=='normal':
                            rows.extend(advanced_rows(names, variant_links, by_name, points, options, point))
        rows = [row for row in rows if not any(step['dodge_replaces'] and step['placement']=='background'
                                             for step in row['timeline'])]
        result.append({
            **{k: character[k] for k in ('id', 'name', 'element', 'avatar')},
            'actions': inventory, 'rows': rows,
            'dodge_preserves_combo': bool(rule.get('dodge_preserves_combo')),
            'start_actions': start_actions,
            'notes': rule.get('notes', []),
            'coverage': {'actions': len(inventory), 'rows': len(rows),
                         'switch_points': sum(len(p) for p in points.values()),
                         'missing_duration': sum(duration(a) is None for a in current)},
        })
    unknown = set(rules.get('characters', {})) - {c['name'] for c in characters}
    if unknown:
        raise ValueError(f'未识别的角色规则：{unknown}')
    order = rules.get('character_order', [c['name'] for c in result])
    if len(order) != len(set(order)) or set(order) != {c['name'] for c in result}:
        raise ValueError('角色展示顺序须恰好包含全部生成角色')
    result.sort(key=lambda c: order.index(c['name']))
    return {'schema_version': 1, 'characters': result}


def generate_files(source=SOURCE, rule_path=MODULE / 'data' / 'rules.json', output=MODULE / 'static' / 'data' / 'tables.json'):
    inputs = [source / 'characters.json', source / 'actions.json', rule_path]
    characters, actions, rules = [json.loads(p.read_text(encoding='utf-8')) for p in inputs]
    payload = generate(characters, actions, rules)
    payload['input_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [*inputs, Path(__file__)]}
    payload['basis'] = '1级裸动作收益；按前排占轴归一。未含配装、Buff、资源上限与队友分享；经验耗时用于试算。'
    payload['sources'] = rules.get('sources', [])
    payload['enumeration'] = '已记录衔接图与动作使用次数边界内的全部路线，列完整释放、预输入、脱手及满足1.2秒回场窗口的分段合轴。'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return payload
