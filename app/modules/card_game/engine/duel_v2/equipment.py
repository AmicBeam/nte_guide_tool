"""Weapon base panels and permanent growth, separate from temporary modifiers.

Queries are pure; writes import old aggregate base fields once. Weapon replacement
and the legacy equip-to-full operation are explicit, separate commands.
"""


def _printed(h):
    if 'base_stats' in h:
        return dict(h['base_stats'])
    from app.modules.card_game.content.duel_v2.tutorial import lookup_character
    try:
        definition = lookup_character(h.get('id')) or {}
    except KeyError:
        definition = {}
    return dict(attack=int(definition.get('attack', h.get('base_attack', 0))),
                max_hp=int(definition.get('max_hp', h.get('base_max_hp', h.get('max_hp', 0)))))


def stat_layers(h):
    """Legacy aggregate overrides remain readable for old snapshots/tools."""
    base = _printed(h)
    growth = dict(h.get('permanent_growth') or {'attack': 0, 'max_hp': 0})
    for legacy, key in (('base_attack', 'attack'), ('base_max_hp', 'max_hp')):
        if legacy in h:
            growth[key] = int(h[legacy]) - base[key]
    return dict(base=base, permanent_growth=growth)


def import_stats(h):
    layers = stat_layers(h)
    h['base_stats'] = layers['base']
    h['permanent_growth'] = layers['permanent_growth']
    h.pop('base_attack', None)
    h.pop('base_max_hp', None)


def permanent_attack(h):
    layers = stat_layers(h)
    return layers['base']['attack'] + layers['permanent_growth']['attack']


def unarmed_max_hp(h):
    layers = stat_layers(h)
    return layers['base']['max_hp'] + layers['permanent_growth']['max_hp']


def weapon_id(h):
    # shape is the compatible slot selector; it never stores the final panel.
    return h.get('shape')


def weapon_source(h):
    equipment = h.get('equipment') or {}
    if equipment.get('card_id') == weapon_id(h):
        return equipment.get('equipment_id')
    return f"{h.get('entity_id')}:weapon:legacy:{weapon_id(h)}" if weapon_id(h) else None


def weapon_definition(h):
    from app.modules.card_game.content.duel_v2.tutorial import lookup_card
    return (lookup_card(weapon_id(h)) or {}) if weapon_id(h) else {}


def attack_base(h):
    from app.modules.card_game.content.duel_v2.registry import shape_attack_bonus
    layers = stat_layers(h)
    form = weapon_definition(h)
    if form.get('type') == 'form' and int(form.get('attack') or 0) > 0 and int(form.get('hp') or 0) > 0:
        equipment = int(form['attack'])
        growth = max(0, layers['permanent_growth']['attack'])
        other = 0
    else:
        equipment = layers['base']['attack']
        growth = layers['permanent_growth']['attack']
        other = h.get('growth', 0) + h.get('gaze', 0) + shape_attack_bonus(weapon_id(h))
    return dict(value=equipment + growth + other, printed=layers['base']['attack'],
                weapon=equipment if weapon_id(h) else None, permanent=growth, legacy_other=other,
                weapon_source=weapon_source(h))


def grow_permanently(h, attack=0, max_hp=0):
    import_stats(h)
    growth = h['permanent_growth']
    growth['attack'] += attack
    growth['max_hp'] += max_hp
    # Existing growth raises both current and maximum life by the same amount.
    h['max_hp'] += max_hp
    h['hp'] += max_hp


def remove_weapon(state, h, *, restore_panel=True):
    from .modifiers import remove_source
    old_id, source = weapon_id(h), weapon_source(h)
    if old_id:
        remove_source(state, h.get('entity_id'), old_id)  # legacy effect sources
        remove_source(state, h.get('entity_id'), source)
    h['shape'] = None
    h.pop('equipment', None)
    if restore_panel:
        h['max_hp'] = unarmed_max_hp(h)
        h['hp'] = min(h['hp'], h['max_hp'])


def equip_weapon(state, h, card):
    if card.get('type') != 'form':
        raise ValueError('Only weapon cards can be equipped')
    import_stats(h)
    remove_weapon(state, h, restore_panel=False)
    h['equipment_seq'] = int(h.get('equipment_seq', 0)) + 1
    h['shape'] = card['card_id']
    h['equipment'] = dict(card_id=card['card_id'],
                          equipment_id=f"{h.get('entity_id')}:weapon:{h['equipment_seq']}",
                          source_card_entity_id=card.get('entity_id'))
    panel_hp = int(card.get('hp') or 0)
    if panel_hp:
        h['max_hp'] = panel_hp + max(0, h['permanent_growth']['max_hp'])
        h['hp'] = min(h['hp'], h['max_hp'])
    return bool(panel_hp)


def refill_after_equipping(h):
    """Existing equip rule, deliberately not ordinary healing or a stat query."""
    h['hp'] = h['max_hp']


def public_equipment(h):
    return dict(card_id=weapon_id(h), equipment_id=weapon_source(h)) if weapon_id(h) else None
