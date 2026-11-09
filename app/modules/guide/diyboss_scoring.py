"""Calculate configured BossDIY challenge scores without trusting client-side point values."""
from decimal import Decimal


def _ids(values, label):
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(label + '必须是ID列表')
    if len(values) != len(set(values)):
        raise ValueError(label + '不能重复选择')
    return set(values)


def calculate(record, stage_id, difficulty_id, buffs=None, targets=None):
    if type(stage_id) not in (str, int) or type(difficulty_id) not in (str, int):
        raise ValueError('关卡和难度必须是ID')
    stage = next((s for s in record['stages'] if s['id'] == str(stage_id)), None)
    if stage is None:
        raise LookupError('未找到争锋关卡')
    difficulty = next((d for d in stage['difficulties'] if d['id'] == str(difficulty_id)), None)
    if difficulty is None:
        raise LookupError('未找到争锋难度')
    selected_buffs = _ids([] if buffs is None else buffs, '强化')
    selected_targets = _ids([] if targets is None else targets, '目标')
    allowed_buffs = {option['id']: option for group in stage['buff_groups'] for option in group['options']}
    allowed_targets = {target['id']: target for target in stage['targets']}
    if selected_buffs - allowed_buffs.keys() or selected_targets - allowed_targets.keys():
        raise ValueError('包含不属于当前关卡的强化或目标')
    for group in stage['buff_groups']:
        if len(selected_buffs & {option['id'] for option in group['options']}) > 1:
            raise ValueError('每行强化最多选择一个档位')
    if any(allowed_targets[key]['score'] is None for key in selected_targets):
        raise ValueError('所选目标缺少积分配置，无法计算')
    buff_score = sum(allowed_buffs[key]['score'] for key in selected_buffs)
    target_score = sum(allowed_targets[key]['score'] for key in selected_targets)
    if target_score > (stage['target_score_limit'] or 0):
        raise ValueError('目标积分超出配置上限')
    base_score = difficulty['base_score']
    subtotal = base_score + buff_score + target_score
    total = Decimal(str(subtotal)) * Decimal(str(difficulty['score_rate']))
    return {'stage': stage['id'], 'difficulty': difficulty['id'],
            'buffs': sorted(selected_buffs), 'targets': sorted(selected_targets),
            'base_score': base_score, 'buff_score': buff_score, 'target_score': target_score,
            'score_rate': difficulty['score_rate'], 'subtotal': subtotal,
            'total': int(total) if total == int(total) else float(total)}
