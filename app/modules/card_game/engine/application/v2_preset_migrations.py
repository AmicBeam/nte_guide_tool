"""Retire untouched official gifts without rewriting player-customized builds."""
from collections import Counter

_OLD_CRIMSON_TEAM = ['zhenhong', 'hathor', 'zero', 'iloy']
_OLD_CRIMSON_CARDS = Counter({key: 2 for key in (
    'Z01', 'Z02', 'Z03', 'Z08', 'R01', 'R02', 'R03', 'R08',
    'H01', 'H03', 'H04', 'H08', 'Y01', 'Y02', 'Y03', 'Y08',
)})


def retire_crimson(store, presets):
    if 'crimson' not in store.get('granted_preset_ids', []):
        return False
    replacement = next((p for p in presets if p['id'] == 'zhenhong'), None)
    if replacement is None:
        return False
    old = [b for b in store['builds'] if b.get('name') == '盈蓄预组'
           and b.get('character_ids') == _OLD_CRIMSON_TEAM
           and Counter(b.get('card_ids', [])) == _OLD_CRIMSON_CARDS]
    if not old:
        return False
    existing = next((b for b in store['builds'] if b.get('name') == replacement['name']
                     and b.get('character_ids') == replacement['character_ids']), None)
    if existing is None:
        existing = old.pop(0)
        existing.update(name=replacement['name'], character_ids=list(replacement['character_ids']),
                        card_ids=list(replacement['card_ids']))
    for obsolete in old:
        store['builds'].remove(obsolete)
        if store.get('active_id') == obsolete['id']:
            store['active_id'] = existing['id']
    if 'zhenhong' not in store['granted_preset_ids']:
        store['granted_preset_ids'].append('zhenhong')
    return True
