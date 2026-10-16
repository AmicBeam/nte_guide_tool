"""Shared material presentation for guide pages."""
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def material_metadata():
    return json.loads((Path(__file__).parent / 'data/material_metadata.json').read_text(encoding='utf-8'))


def material_display(item):
    return material_metadata().get(str(item['id']).casefold(), {})
