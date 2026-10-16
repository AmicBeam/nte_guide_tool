"""HTML-only stage selection; machine-readable detail stays independently filterable."""
from . import service


def stage_selection(record, requested=None):
    stages = sorted(service.detail('challenges', str(record['id']))['stages'],
                    key=lambda stage: int(stage['id']), reverse=True)
    ids = [str(stage['id']) for stage in stages]
    selected = str(requested) if requested else ('12' if '12' in ids else (ids[0] if ids else None))
    return {'stages': stages, 'selected': selected}
