"""Structured public presentation for V2 observe(); never infers cards from log text."""
from copy import deepcopy
from .simulation import collecting_patches

from .effect_markers import character_effect_markers, team_effect_markers
from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS
from app.modules.card_game.content.duel_v2.tutorial import lookup_card
from app.modules.card_game.content.duel_v2.catalog import CHARACTER_ORDER
from app.modules.card_game.content.duel_v2.effects import TARGET_POLICIES, interaction_kind
from app.modules.card_game.content.duel_v2.registry import copy_owner_text, text_name
from app.modules.card_game.engine.events import V2_PRESENTATION_SCHEMA

SIDES = ('a', 'b')
ORDER = CHARACTER_ORDER


def other(side):
    return 'b' if side == 'a' else 'a'

CARD_FIELDS = (
    'entity_id', 'instance_id', 'card_id', 'character_id', 'name', 'type', 'cost', 'terminal',
    'description', 'copy', 'expires_turn', 'original_character_id',
    'stat', 'stat_value', 'shield', 'attack', 'hp', 'instant', 'response', 'require', 'redeem',
    'derived', 'attack_mode', 'jingu_mark', 'play_options', 'antique_investment',
    'training_excluded', 'portrait', 'avatar', 'selection_kind', 'hand_face',
)
ENTITY_CHARS = set(CHARACTERS)
PRIVATE_EVENT_KEYS = ('private_card', 'private_side', 'private_hand')
INTERACTION_KINDS = ('attack', 'cast', 'target', 'form', 'awaken', 'ultimate', 'choice', 'other')
SILENT_PRESENT_TYPES = frozenset()


def public_card(card, state=None, side=None):
    result = {key: deepcopy(card[key]) for key in CARD_FIELDS if key in card}
    if state is not None and side in SIDES and card.get('character_id') in state['sides'][side]['characters']:
        from .context import EffectContext
        from app.modules.card_game.content.duel_v2.registry import decide
        result['description'] = decide(EffectContext(state, side, card['character_id'], {'side': side}),
            'card_description', result.get('description', ''), actor_only=True, include_down=True, card=card)
    # Independent of the once-per-turn instant discount. Hidden cards never pass here.
    result['action_point_free'] = int(card.get('cost') or 0) == 0
    if state is not None and side in SIDES:
        from .flow import card_cost
        result['action_point_free'] = card_cost(state, side, card) == 0
    if card.get('original_character_id'):
        original = text_name(card.get('original_character_id'))
        if original:
            owner = text_name(card.get('character_id')) or copy_owner_text()
            result['description'] = result.get('description', '').replace(original, owner)
    physical = card.get('_physical_card')
    if physical is not None:
        # Presentation only: retain the physical face without exposing internal state.
        result['hand_face'] = {key: deepcopy(physical[key]) for key in (
            'card_id', 'character_id', 'name', 'type', 'description', 'stat', 'stat_value',
            'shield', 'attack', 'hp', 'attack_mode', 'derived',
        ) if key in physical}
    return result


def entity_id(side, cid):
    if side not in SIDES:
        return None
    from .state import is_player
    if is_player(cid):
        return f'{side}:player'
    return f'{side}:{cid}'


def capture_public_board(state):
    if not collecting_patches():
        return {}
    from .escalation import public_progress, energy_cap, harmony_cap, ultimate_remaining
    from .combat import harmony_marks
    from .flow import ultimate_spec
    board = {
        'escalation': public_progress(state),
        'phase': state.get('phase'),
        'active_side': state.get('active_side'),
        'turn': state.get('turn'),
        'winner': state.get('winner'),
    }
    sides = {}
    available = state.get('sides') or {}
    for ps in SIDES:
        source = available.get(ps)
        if not source:
            continue
        team = {key: deepcopy(source[key]) for key in ('hp', 'shield', 'ap', 'front', 'last_front', 'front_debuff')}
        if 'entity_id' in source:
            team['entity_id'] = source['entity_id']
        team['life_locked'] = bool(source.get('life_locked'))
        team['genesis_player_hits'] = int(source.get('genesis_player_hits', 0))
        team['genesis_turn'] = source.get('genesis_turn')
        team['ultimate_remaining'] = ultimate_remaining(state, ps)
        marks = harmony_marks(state, ps)
        team['harmony_available'] = marks['available']
        team['effect_markers'] = team_effect_markers(source)
        team['hand_count'] = len(source['hand'])
        team['deck_count'] = len(source['deck'])
        pending = state.get('pending_choice')
        if pending and pending.get('side') == ps and pending.get('kind') == 'inspect_top':
            team['deck_count'] += len(pending.get('cards') or ())
        from . import state as duel_state
        from .modifiers import public_effects
        from .equipment import stat_layers, public_equipment
        characters = []
        for cid in duel_state.team_order(state, ps):
            hero = source['characters'][cid]
            shape_id = hero.get('shape')
            attack_breakdown = duel_state.attack_value(hero, state, ps, explain=True)
            characters.append({
                'id': cid,
                **{k: deepcopy(hero[k]) for k in ('name', 'avatar', 'portrait', 'attribute', 'summoned', 'resource_name', 'faction', 'arid', 'truth_keys', 'beast_fangs', 'intimidation', 'extra_attacks', 'protagonist_aura') if k in hero},
                'entity_id': hero.get('entity_id'),
                'effects': public_effects(hero),
                'stat_layers': stat_layers(hero), 'equipment': public_equipment(hero),
                'effect_markers': character_effect_markers(hero, state, ps),
                'damage_immune': duel_state.damage_immune(state, ps, cid),
                'damage_limit': duel_state.damage_limit(state, ps, cid),
                'hp': hero['hp'],
                'max_hp': hero['max_hp'],
                'shield': hero['shield'],
                'attack': attack_breakdown['value'],
                'attack_breakdown': attack_breakdown,
                'harmony': hero['harmony'],
                'harmony_source': marks['payer'] == cid,
                'harmony_ready': cid in marks['ready'],
                'energy': hero['energy'],
                'energy_max': energy_cap(state, hero),
                'harmony_max': harmony_cap(state),
                'awakened': hero['awakened'],
                'ultimate_turns': hero.get('ultimate_turns', 0),
                'ultimate_expires': ultimate_spec(cid)['expires'],
                'down_turns': hero['down_turns'],
                'growth': hero.get('growth', 0),
                'family': hero.get('family', 0),
                'gaze': hero.get('gaze', 0),
                'dream': hero.get('dream', 0),
                **({'jingu': hero['jingu']} if 'jingu' in hero else {}),
                'shape': (lookup_card(shape_id) or {}).get('name') if shape_id else None,
                'shape_id': shape_id,
            })
        team['characters'] = characters
        sides[ps] = team
    board['sides'] = sides
    return board


def diff_public(before, after):
    if not before:
        return deepcopy(after) if after else {}
    patch = {}
    for key in ('phase', 'active_side', 'turn', 'winner', 'version', 'reason',
                'pending_choice', 'resolving_card', 'escalation'):
        if before.get(key) != after.get(key):
            patch[key] = deepcopy(after.get(key))
    sides = {}
    for ps in SIDES:
        previous_side = (before.get('sides') or {}).get(ps)
        current_side = (after.get('sides') or {}).get(ps)
        if not current_side:
            continue
        if not previous_side:
            sides[ps] = deepcopy(current_side)
            continue
        slot = {}
        for key, value in current_side.items():
            if key == 'characters':
                continue
            if previous_side.get(key) != value:
                slot[key] = deepcopy(value)
        changed = []
        previous_chars = {row['id']: row for row in previous_side['characters']}
        for row in current_side['characters']:
            old = previous_chars.get(row['id'], {})
            delta = {key: deepcopy(value) for key, value in row.items()
                     if key == 'id' or old.get(key) != value}
            if set(delta) - {'id'}:
                changed.append(delta)
        current_ids = {row['id'] for row in current_side['characters']}
        changed.extend({'id': cid, 'removed': True} for cid in previous_chars if cid not in current_ids)
        if changed:
            slot['characters'] = changed
        if slot:
            sides[ps] = slot
    if sides:
        patch['sides'] = sides
    return patch


def merge_patch(base, extra):
    result = deepcopy(base) if base else {}
    extra = extra or {}
    for key, value in extra.items():
        if key == 'sides' and isinstance(value, dict) and isinstance(result.get('sides'), dict):
            for side, payload in value.items():
                slot = result['sides'].setdefault(side, {})
                for field, field_value in payload.items():
                    if field == 'characters' and isinstance(field_value, list) and isinstance(slot.get('characters'), list):
                        by_id = {row['id']: row for row in slot['characters'] if 'id' in row}
                        for row in field_value:
                            if row.get('removed'):
                                slot['characters'] = [h for h in slot['characters'] if h.get('id') != row['id']]
                                continue
                            current = by_id.get(row.get('id'))
                            if current is None:
                                slot['characters'].append(deepcopy(row))
                            else:
                                current.update(deepcopy(row))
                    else:
                        slot[field] = deepcopy(field_value)
        else:
            result[key] = deepcopy(value)
    return result


def replay_public_board(before, events):
    """Apply public presentation patches in seq order onto a captured board."""
    board = deepcopy(before) if before else {}
    for event in events or ():
        patch = event.get('patch') if isinstance(event, dict) else None
        if patch:
            board = merge_patch(board, patch)
    return board


def bump_public_patch(state):
    if not collecting_patches():
        return {}
    current = capture_public_board(state)
    extra = diff_public(state.get('_public_board'), current)
    events = state.get('events') or []
    if extra and events:
        events[-1]['patch'] = merge_patch(events[-1].get('patch'), extra)
    state['_public_board'] = current
    return extra


def _normalize_entity(side, raw):
    if raw in (None, ''):
        return None
    if isinstance(raw, str) and ':' in raw:
        owner, rest = raw.split(':', 1)
        if owner in SIDES and (rest in ENTITY_CHARS or rest == 'player'):
            return raw
        return None
    if raw in ENTITY_CHARS and side in SIDES:
        return f'{side}:{raw}'
    if raw == 'player' and side in SIDES:
        return f'{side}:player'
    return None


def derive_actor(event):
    if event.get('actor') not in (None, ''):
        return _normalize_entity(event.get('side'), event.get('actor'))
    source = event.get('source')
    side = event.get('side')
    normalized = _normalize_entity(side, source)
    if normalized:
        return normalized
    if event.get('type') in ('draw', 'turn', 'finish', 'fatigue', 'initiative', 'mulligan') and side in SIDES:
        return entity_id(side, 'player')
    return None


def derive_target(event):
    if event.get('target') not in (None, ''):
        return _normalize_entity(event.get('side'), event.get('target'))
    return None


def next_group_id(state):
    state['combat_group'] = int(state.get('combat_group') or 0) + 1
    action_id = state.get('current_action_id') or f"v{state.get('version', 0)}"
    return f"{action_id}:g{state['combat_group']}"


def enrich_log(state, event):
    event.setdefault('action_id', state.get('current_action_id') or f"legacy-{event['seq']}")
    event.setdefault('action_version', state.get('current_action_version', state.get('version', 0)))
    if 'actor' not in event or event.get('actor') in (None, ''):
        event['actor'] = derive_actor(event)
    else:
        event['actor'] = _normalize_entity(event.get('side'), event.get('actor')) or event.get('actor')
    for field in ('actor', 'target'):
        address = event.get(field)
        if isinstance(address, str) and ':' in address:
            side, cid = address.split(':', 1)
            team = state.get('sides', {}).get(side, {})
            entity = team if cid == 'player' else team.get('characters', {}).get(cid, {})
            if entity.get('entity_id'):
                event[field + '_entity_id'] = entity['entity_id']
    if event.get('card') is not None and not event.get('private_side'):
        event['card'] = public_card(event['card'])
    if event.get('private_card') is not None:
        event['private_card'] = public_card(event['private_card'])
    if 'present' not in event:
        event['present'] = event.get('type') not in SILENT_PRESENT_TYPES
    if not collecting_patches():
        return event
    previous = state.get('_public_board')
    current = capture_public_board(state)
    event['patch'] = diff_public(previous, current)
    state['_public_board'] = current
    return event


def _visible_card(event, viewer):
    private_side = event.get('private_side')
    if private_side:
        if private_side == viewer and event.get('private_card'):
            return deepcopy(event['private_card'])
        return None
    card = event.get('card')
    return deepcopy(card) if card else None


def project_event(event, viewer):
    item = {
        'seq': event.get('seq'),
        'action_id': event.get('action_id') or f"legacy-{event.get('seq', 0)}",
        'action_version': event.get('action_version', 0),
        'type': event.get('type'),
        'side': event.get('side'),
        'text': event.get('text'),
        'actor': derive_actor(event),
        'target': derive_target(event),
        'source': event.get('source'),
    }
    for key in ('amount', 'attack', 'counter', 'group_id', 'harmony_delta', 'energy_delta',
                'hit_index', 'before', 'after', 'source_side', 'card_ids', 'description', 'present',
                'first_side', 'winner', 'reason', 'counter_immunity', 'actor_entity_id', 'target_entity_id'):
        if key in event:
            item[key] = deepcopy(event[key])
    if event.get('type') == 'mulligan' and event.get('side') != viewer:
        item.pop('card_ids', None)
    if 'attack' not in item and event.get('type') == 'attack' and 'amount' in event:
        item['attack'] = event['amount']
    card = _visible_card(event, viewer)
    if card is not None:
        item['card'] = card
    patch = event.get('patch')
    if isinstance(patch, dict):
        item['patch'] = deepcopy(patch)
    _patch_private_hand(item, event, viewer)
    return item


def _patch_private_hand(item, event, viewer):
    if event.get('private_side') == viewer and isinstance(event.get('private_hand'), list):
        team = item.setdefault('patch', {}).setdefault('sides', {}).setdefault(viewer, {})
        team['hand'] = deepcopy(event['private_hand'])
        team['hand_count'] = len(team['hand'])


def localize_text(event, viewer):
    text = event.get('text') or ''
    text = text.replace('甲方玩家', '我方玩家' if viewer == 'a' else '对手玩家')
    text = text.replace('乙方玩家', '我方玩家' if viewer == 'b' else '对手玩家')
    card = _visible_card(event, viewer)
    if event.get('type') == 'draw' and card and card.get('name') and '抽到「' not in text:
        name = card['name']
        if '抽 1 张牌' in text:
            text = text.replace('抽 1 张牌', f'抽到「{name}」')
        else:
            text = text.rstrip('。') + f"：「{name}」。"
    if event.get('side') in SIDES:
        prefix = '我方 · ' if event['side'] == viewer else '对手 · '
        text = prefix + text
    return text


def sanitize_public_event(event, viewer):
    item = deepcopy(event)
    for key in PRIVATE_EVENT_KEYS + ('_public_board',):
        item.pop(key, None)
    card = _visible_card(event, viewer)
    if card is not None:
        item['card'] = card
    else:
        item.pop('card', None)
        item.pop('private_card', None)
    if event.get('type') == 'mulligan' and event.get('side') != viewer:
        item.pop('card_ids', None)
    _patch_private_hand(item, event, viewer)
    item['text'] = localize_text(event, viewer)
    if 'actor' not in item:
        item['actor'] = derive_actor(event)
    if 'target' not in item:
        item['target'] = derive_target(event)
    return item


def presentation_bundle(state, viewer, after_seq=None):
    raw = list(state.get('events') or [])
    projected = []
    for event in raw:
        item = project_event(event, viewer)
        item['text'] = localize_text(event, viewer)
        projected.append(item)
    window = projected[-200:]
    oldest_seq = window[0]['seq'] if window else 0
    cursor = int(state.get('event_seq') or (window[-1]['seq'] if window else 0))
    events = window
    if after_seq is not None:
        events = [event for event in window if int(event['seq'] or 0) > int(after_seq)]
    return {
        'schema_version': V2_PRESENTATION_SCHEMA,
        'cursor': cursor,
        'oldest_seq': oldest_seq,
        'events': events,
    }


def _enemy_front_targets(state, side):
    from .state import front_target
    enemy, cid = front_target(state, side)
    return [entity_id(enemy, cid)]


def interaction_for_action(state, side, action):
    kind = action.get('type')
    if kind == 'attack':
        return {
            'kind': 'attack',
            'actor_id': entity_id(side, action.get('character_id')),
            'target_ids': _enemy_front_targets(state, side),
        }
    if kind == 'ultimate':
        actor = entity_id(side, action.get('character_id'))
        return {'kind': 'ultimate', 'actor_id': actor, 'target_ids': [actor] if actor else []}
    if kind == 'play_card':
        card = next(c for c in state['sides'][side]['hand'] if c['instance_id'] == action['card_id'])
        from .state import effective_card
        card = effective_card(state, side, card)
        actor = entity_id(side, card['character_id'])
        hint = interaction_kind(card)
        if hint == 'form':
            return {'kind': 'form', 'actor_id': actor, 'target_ids': [actor] if actor else []}
        policy = TARGET_POLICIES.get(card.get('card_id'))
        if policy == 'other_living_ally':
            target = _normalize_entity(None, action.get('target_id'))
            return {'kind': 'target', 'actor_id': actor, 'target_ids': [target] if target else []}
        if policy == 'enemy_front':
            return {'kind': 'target', 'actor_id': actor, 'target_ids': _enemy_front_targets(state, side)}
        if hint == 'target' and action.get('target_id'):
            target = action['target_id']
            # Targets were already validated and expanded by the rules engine.
            return {'kind': 'target', 'actor_id': actor, 'target_ids': [target]}
        if policy in ('enemy_hand', 'own_tactic_discard'):
            return {'kind': 'choice', 'actor_id': actor, 'target_ids': []}
        return {'kind': 'cast', 'actor_id': actor, 'target_ids': []}
    if kind in ('choose', 'choose_cards', 'mulligan', 'cycle'):
        actor = None
        operation = state.get('operation')
        if operation:
            actor = entity_id(operation.get('side'), operation.get('actor'))
        return {'kind': 'choice', 'actor_id': actor, 'target_ids': []}
    return {'kind': 'other', 'actor_id': None, 'target_ids': []}
