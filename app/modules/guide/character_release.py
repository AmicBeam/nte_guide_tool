"""Request-time publication of character reference sections using server time."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
from pathlib import Path
import math

BEIJING = timezone(timedelta(hours=8))
AFTER_SKILL_FIELDS = {'abilities', 'awaken', 'resonance', 'nouns', 'equip_slots', 'affinity',
                      'aliases', 'experience', 'reference_details'}
PREFIX_FIELDS = {'id', 'name', 'element', 'element_id', 'element_name', 'rarity', 'hp', 'atk', 'def',
                 'stats', 'char_tags', 'arc_type', 'assets', 'rarity_icon', 'element_icon',
                 'status', 'preview_image'}


def current_time():
    return datetime.now(BEIJING)


@lru_cache(maxsize=1)
def configured_rows():
    return json.loads((Path(__file__).parent/'data/character_releases.json').read_text(encoding='utf-8'))['characters']


@lru_cache(maxsize=1)
def windows():
    rows = configured_rows()
    result = {}
    for row in rows:
        release = datetime.fromisoformat(row['release_at'])
        if release.tzinfo is None or release.utcoffset() != timedelta(hours=8):
            raise ValueError('Character release times must include UTC+8')
        for key in [*row['ids'], row['name']]:
            key = str(key).casefold()
            if key in result:
                raise ValueError('Duplicate character release identity')
            result[key] = release
    return result


def release_time(record):
    for key in [str(record['id']), record.get('name', ''), *record.get('aliases', [])]:
        if str(key).casefold() in windows():
            return windows()[str(key).casefold()]
    return None


def metadata(record, now=None):
    release = release_time(record)
    if release is None:
        return {}
    now = current_time() if now is None else now
    if now.tzinfo is None:
        raise ValueError('Server time must be timezone-aware')
    return {'release_at': release.isoformat(), 'content_released': now >= release}


def publication_states():
    now = current_time()
    return [{'name': row['name'], **metadata({'id': row['ids'][0], 'name': row['name']}, now)}
            for row in configured_rows()]


def display_time(value):
    return datetime.fromisoformat(value).astimezone(BEIJING).strftime('%Y-%m-%d %H:%M（UTC+8）')


def release_status(record):
    now = current_time()
    state = metadata(record, now)
    released = state.get('content_released', True)
    delay = 0 if released else max(250, min(30000, math.ceil((release_time(record)-now).total_seconds()*1000)))
    return {'id': record['id'], **state, 'content_released': released, 'check_after_ms': delay}


def project(record, now=None):
    state = metadata(record, now)
    if not state:
        return record
    if state['content_released']:
        return {**record, **state}
    result = {key: deepcopy(value) for key, value in record.items() if key in PREFIX_FIELDS}
    # Keep the pre-skill material icons without skill names, IDs, levels, or costs.
    items = {}
    def collect(value):
        if isinstance(value, dict):
            for item in value.get('items', []):
                items.setdefault(str(item['id']).casefold(), {key: item[key] for key in ('id','name')})
            for key, child in value.items():
                if key != 'items': collect(child)
        elif isinstance(value, list):
            for child in value: collect(child)
    collect(record.get('materials', {}))
    result['materials'] = {'breakthrough': [{'items': list(items.values())}], 'skills': []}
    return {**result, **state}


def project_arc(record, now=None):
    state = metadata(record, now)
    if not state:
        return record
    subject = next((row['ids'][0] for row in configured_rows()
                    if str(record['id']).casefold() in {str(value).casefold() for value in row['ids']}
                    or record.get('name', '').casefold() == row['name'].casefold()), record['id'])
    state['release_subject_id'] = subject
    if state['content_released']:
        return {**record, **state}
    public = {'id', 'name', 'rarity', 'type_id', 'type_name', 'description', 'stats',
              'materials', 'image', 'rarity_icon', 'exclusive_character_ids', 'exclusive_desc'}
    return {**{key: deepcopy(value) for key, value in record.items() if key in public}, **state}
