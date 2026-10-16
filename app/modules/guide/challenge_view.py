"""HTML-only stage selection; machine-readable detail stays independently filterable."""
import json

from . import service


def stage_selection(record, requested=None):
    stages = sorted(service.detail('challenges', str(record['id']))['stages'],
                    key=lambda stage: int(stage['id']), reverse=True)
    ids = [str(stage['id']) for stage in stages]
    selected = str(requested) if requested else ('12' if '12' in ids else (ids[0] if ids else None))
    return {'stages': stages, 'selected': selected}


def group_monsters(monsters):
    """Combine identical cards in one wave without changing the stored configurations."""
    groups = {}
    for monster in monsters:
        visible = {key: value for key, value in monster.items() if key not in ('id', 'class_id', 'count')}
        key = json.dumps(visible, sort_keys=True, ensure_ascii=False)
        if key not in groups:
            groups[key] = {**monster, 'count': 0}
        groups[key]['count'] += monster['count']
    return list(groups.values())
