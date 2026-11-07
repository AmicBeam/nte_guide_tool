"""HTML-only stage selection; machine-readable detail stays independently filterable."""
from . import service


def stage_selection(record, requested=None, kind="challenges"):
    stages = sorted(service.detail(kind, str(record['id']))['stages'],
                    key=lambda stage: int(stage['id']), reverse=(kind != 'endless'))
    ids = [str(stage['id']) for stage in stages]
    selected = str(requested) if requested else ('12' if kind == 'challenges' and '12' in ids else (ids[0] if ids else None))
    if kind == 'endless' and selected in ids:
        first = ((int(selected) - 1) // 50) * 50 + 1
        stages = [stage for stage in stages if first <= int(stage['id']) < first + 50]
    return {'stages': stages, 'selected': selected}
