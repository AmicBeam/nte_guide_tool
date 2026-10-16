"""Pure character cultivation cost calculation over the offline catalog."""
from __future__ import annotations


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f'{name}须为整数')
    try:
        result = int(value)
    except (ValueError, TypeError):
        raise ValueError(f'{name}须为整数') from None
    if not low <= result <= high:
        raise ValueError(f'{name}须为 {low}–{high}')
    return result


def skill_options(character):
    result = []
    for skill in character['materials']['skills']:
        levels = sorted(int(row['level']) for row in skill['levels'])
        # Level-one entries are unlock costs, not free pre-existing passives.
        minimum = 0 if 1 in levels else 1
        result.append({**skill, 'minimum': minimum, 'maximum': max(levels)})
    return result


def defaults(character, config):
    result = {'current_level': 1, 'target_level': config['level_caps'][str(character['id'])][-1],
              'current_breakthrough': 0, 'target_breakthrough': 6, 'current_exp': 0}
    for i, skill in enumerate(skill_options(character)):
        result[f'current_skill_{i}'] = skill['minimum']
        result[f'target_skill_{i}'] = skill['maximum']
    return result


def calculate(character, config, experience, values):
    caps = config['level_caps'][str(character['id'])]
    options = skill_options(character)
    state = {}
    for side in ('current', 'target'):
        stage = number(values.get(f'{side}_breakthrough'), '突破次数', 0, len(caps) - 1)
        level = number(values.get(f'{side}_level'), '角色等级', 1, caps[-1])
        minimum = 1 if stage == 0 else caps[stage - 1]
        if not minimum <= level <= caps[stage]:
            raise ValueError(f'{"当前" if side == "current" else "目标"}突破 {stage} 次时，角色等级须为 {minimum}–{caps[stage]}')
        state[f'{side}_level'], state[f'{side}_breakthrough'] = level, stage
    for field in ('level', 'breakthrough'):
        if state[f'target_{field}'] < state[f'current_{field}']:
            raise ValueError('目标练度不能低于当前练度')
    xp_rows = {row['level']: row['required_exp'] for row in experience}
    current, target = state['current_level'], state['target_level']
    exp_cap = xp_rows[current + 1] - 1 if current < caps[state['current_breakthrough']] else 0
    current_exp = number(values.get('current_exp', 0), '当前等级已积累经验', 0, exp_cap)
    state['current_exp'] = current_exp
    totals, breakdown = {}, []

    def add(category, entries):
        entries = [dict(item) for item in entries if item['amount']]
        if not entries:
            return
        breakdown.append({'name': category, 'items': entries})
        for item in entries:
            key = str(item['id']).casefold()
            if key not in totals:
                totals[key] = {'id': item['id'], 'name': item['name'], 'amount': 0}
            totals[key]['amount'] += item['amount']

    xp_needed, xp_provided = 0, 0
    book_counts = {book['id']: 0 for book in config['exp_books']}
    # Each breakthrough cap ends a feeding segment; excess XP cannot cross it.
    segment_start = current
    for end in sorted({cap for cap in caps if current < cap < target} | {target}):
        if end <= segment_start:
            continue
        needed = sum(xp_rows[level] for level in range(segment_start + 1, end + 1))
        if segment_start == current:
            needed -= current_exp
        xp_needed += needed
        unit = min(book['exp'] for book in config['exp_books'])
        remaining = ((needed + unit - 1) // unit) * unit
        for book in sorted(config['exp_books'], key=lambda b: b['exp'], reverse=True):
            count, remaining = divmod(remaining, book['exp'])
            book_counts[book['id']] += count
            xp_provided += count * book['exp']
        if remaining:
            raise ValueError('经验书配置无法覆盖所需经验')
        segment_start = end
    books = [{'id': b['id'], 'name': b['name'], 'amount': book_counts[b['id']]}
             for b in config['exp_books']]
    book_gold = sum(book_counts[b['id']] * b['cost_gold'] for b in config['exp_books'])
    add('角色升级', books + [{'id': 'gold', 'name': '甲硬币', 'amount': book_gold}])
    start, end = state['current_breakthrough'], state['target_breakthrough']
    stages = character['materials']['breakthrough']
    if len(stages) != len(caps) - 1:
        raise ValueError('角色突破材料配置不完整')
    for index in range(start, end):
        add(f'第 {index + 1} 次突破', stages[index]['items'])
    for i, skill in enumerate(options):
        before = number(values.get(f'current_skill_{i}'), f'{skill["name"]}当前等级', skill['minimum'], skill['maximum'])
        after = number(values.get(f'target_skill_{i}'), f'{skill["name"]}目标等级', before, skill['maximum'])
        state[f'current_skill_{i}'], state[f'target_skill_{i}'] = before, after
        rows = {int(row['level']): row for row in skill['levels']}
        for level, stage in ((before, start), (after, end)):
            if level in rows and rows[level].get('require_breakthrough_level', 0) > stage:
                raise ValueError(f'{skill["name"]} {level} 级需要突破 {rows[level]["require_breakthrough_level"]} 次')
        for level in range(before + 1, after + 1):
            if level not in rows:
                raise ValueError(f'{skill["name"]}缺少 {level} 级材料配置')
            add(f'{skill["name"]} · {level} 级', rows[level]['items'])
    return {'state': state, 'experience': xp_needed, 'book_experience': xp_provided,
            'overflow_experience': xp_provided - xp_needed, 'level_up_gold': book_gold,
            'total_gold': totals.get('gold', {}).get('amount', 0),
            'items': sorted(totals.values(), key=lambda x: (x['id'].casefold() != 'gold', x['name'])),
            'breakdown': breakdown}
