"""V2 character kits: passives, shapes, and card effects.

The engine fires named timings. Kits register by character id. This is not the
legacy event bus: there is no replace/append queue and deck cards are not scanned.
"""
from __future__ import annotations

from typing import Any, Callable

KITS: dict[str, Any] = {}


def register(kit: Any) -> Any:
    kit_id = kit.id
    if kit_id in KITS:
        raise ValueError(f'重复的 V2 角色包: {kit_id}')
    KITS[kit_id] = kit
    return kit


def _ensure_kits() -> None:
    if not KITS:
        from app.modules.card_game.content.duel_v2 import characters  # noqa: F401


def copy_owner_id() -> str | None:
    _ensure_kits()
    for kit in KITS.values():
        if getattr(kit, 'copy_owner', False):
            return kit.id
    return None


def text_name(character_id: str | None) -> str | None:
    kit = KITS.get(character_id or '')
    if kit is None:
        return None
    return getattr(kit, 'text_name', None)


def copy_owner_text() -> str:
    owner = copy_owner_id()
    return text_name(owner) or '浔'


def shape_attack_bonus(shape_id: str | None) -> int:
    if not shape_id:
        return 0
    from app.modules.card_game.content.duel_v2 import CARDS
    card = CARDS.get(shape_id) or {}
    if card.get('type') != 'form':
        return 0
    if card.get('stat') == 'attack':
        return int(card.get('stat_value') or 0)
    return int(card.get('attack') or 0)


def shape_hp_bonus(shape_id: str | None) -> int:
    if not shape_id:
        return 0
    from app.modules.card_game.content.duel_v2 import CARDS
    card = CARDS.get(shape_id) or {}
    if card.get('type') != 'form':
        return 0
    if card.get('stat') == 'hp':
        return int(card.get('stat_value') or 0)
    return int(card.get('hp') or 0)


def shape_event(c, event, **data) -> None:
    _ensure_kits()
    shape_id = c.character.get('shape')
    if not shape_id:
        return
    kit = KITS.get(c.cid)
    handler = getattr(kit, 'shape_events', {}).get((shape_id, event)) if kit else None
    if handler:
        handler(c.for_weapon(shape_id), **data)


def _ordered_ids(characters: dict) -> list[str]:
    from app.modules.card_game.content.duel_v2.catalog import CHARACTER_ORDER
    return [cid for cid in CHARACTER_ORDER if cid in characters] + [
        cid for cid in characters if cid not in CHARACTER_ORDER
    ]


def _kit_handlers(c, hook: str, *, actor_only: bool = False, include_down: bool = False):
    _ensure_kits()
    side = c.side
    characters = c.state['sides'][side]['characters']
    ids = [c.cid] if actor_only else _ordered_ids(characters)
    for cid in ids:
        kit = KITS.get(cid)
        if kit is None or cid not in characters:
            continue
        if not include_down and characters[cid]['hp'] <= 0:
            continue
        mimic = (characters[cid].get('flags') or {}).get('mimic') or {}
        passive_kit = kit
        wrapped = False
        if mimic.get('id') and hook not in ('on_ultimate', 'on_ultimate_end'):
            passive_kit = KITS.get(mimic['id'])
            wrapped = passive_kit is not None
        handler = getattr(passive_kit, hook, None) if passive_kit is not None else None
        if handler is not None:
            if wrapped:
                def _mimicked(ctx, _handler=handler, **data):
                    awakened = ctx.character.get('awakened')
                    ctx.character['awakened'] = False
                    try:
                        return _handler(ctx, **data)
                    finally:
                        ctx.character['awakened'] = awakened
                yield cid, _mimicked
            else:
                yield cid, handler
        from app.modules.card_game.engine.duel_v2.equipment import weapon_id, weapon_source
        equipped = weapon_id(characters[cid])
        weapon_handler = getattr(kit, 'weapon_hooks', {}).get(equipped, {}).get(hook)
        if weapon_handler is not None:
            def dispatch(ctx, _handler=weapon_handler, _weapon=equipped, _source=weapon_source(characters[cid]), **data):
                if weapon_id(ctx.character) == _weapon and weapon_source(ctx.character) == _source:
                    return _handler(ctx.for_weapon(_weapon), **data)
                return None
            yield cid, dispatch


def fire(c, hook: str, *, actor_only: bool = False, include_down: bool = False, **data) -> None:
    """Run append-only kit callbacks. Missing hooks are skipped."""
    from app.modules.card_game.engine.duel_v2.state import finished
    from app.modules.card_game.engine.duel_v2.tutorial import tutorial_flags
    if finished(c.state):
        return
    if tutorial_flags(c.state).get('disable_passives'):
        return
    for cid, handler in _kit_handlers(c, hook, actor_only=actor_only, include_down=include_down):
        handler(c.for_character(c.side, cid), **data)


def decide(c, hook: str, default, *, actor_only: bool = False, include_down: bool = False, **data):
    """Replace a decision value. Kits return None to keep the current value.

    A non-None return replaces it; a false value such as False or '' cancels.
    Later kits in catalog order see the updated value and may replace again.
    This is a gate override, not the legacy event-bus replace queue.
    """
    from app.modules.card_game.engine.duel_v2.state import finished
    from app.modules.card_game.engine.duel_v2.tutorial import tutorial_flags
    if finished(c.state):
        return default
    if tutorial_flags(c.state).get('disable_passives'):
        return default
    value = default
    for cid, handler in _kit_handlers(c, hook, actor_only=actor_only, include_down=include_down):
        result = handler(c.for_character(c.side, cid), current=value, **data)
        if result is not None:
            value = result
    return value


def genesis_flowers(c) -> int:
    return tally(c, 'genesis_flowers')


def tally(c, hook: str, **data) -> int:
    """Sum integer returns from living kits. Missing hooks count as 0."""
    from app.modules.card_game.engine.duel_v2.state import finished
    if finished(c.state):
        return 0
    total = 0
    for cid, handler in _kit_handlers(c, hook):
        total += int(handler(c.for_character(c.side, cid), **data) or 0)
    return total


def collect_effects() -> tuple[dict[str, Callable], dict[str, str]]:
    effects: dict[str, Callable] = {}
    policies: dict[str, str] = {}
    for kit in KITS.values():
        effects.update(getattr(kit, 'effects', {}))
        policies.update(getattr(kit, 'target_policies', {}))
    return effects, policies
