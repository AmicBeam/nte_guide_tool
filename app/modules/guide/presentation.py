"""Shared material presentation for guide pages."""
from functools import lru_cache
import json
import re
from pathlib import Path


@lru_cache(maxsize=1)
def material_metadata():
    return json.loads((Path(__file__).parent / 'data/material_metadata.json').read_text(encoding='utf-8'))


def material_display(item):
    return material_metadata().get(str(item['id']).casefold(), {})


def upgrade_material_preview(record):
    """Unique upgrade materials, keeping the highest quality used per family."""
    qualities = {'gray': 0, 'green': 1, 'blue': 2, 'purple': 3, 'orange': 4, 'red': 5}
    families = {}

    def visit(value):
        if isinstance(value, dict):
            for item in value.get('items', []):
                identifier = str(item['id']).casefold()
                if identifier in {'gold', 'fons', 'cityability_upmaterial'} or identifier.startswith('characterupmaterial_'):
                    continue
                family = re.sub(r'_lv\d+$', '', identifier)
                quality = qualities.get(material_display(item).get('quality'), -1)
                if family not in families or quality > families[family][0]:
                    families[family] = (quality, {'id': item['id'], 'name': item['name']})
            for key, child in value.items():
                if key != 'items':
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(record.get('materials', {}))
    return [item for _, item in families.values()]
