"""Structured public presentation for V2 observe(); never infers cards from log text."""
from copy import deepcopy

from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.content.duel_v2.effects import TARGET_POLICIES, interaction_kind
from app.modules.card_game.engine.events import V2_PRESENTATION_SCHEMA

SIDES = ('a', 'b')
ORDER = ('nanali', 'zero', 'jiuyuan', 'xun')


def other(side):
    return 'b' if side == 'a' else 'a'

CARD_FIELDS = (
    'instance_id', 'card_id', 'character_id', 'name', 'type', 'cost', 'terminal',
    'description', 'copy', 'expires_turn', 'original_character_id',
)
ENTITY_CHARS = set(ORDER)
PRIVATE_EVENT_KEYS = ('private_card', 'private_side')
INTERACTION_KINDS = ('attack', 'cast', 'target', 'form', 'awaken', 'choice', 'other')


def public_card(card):
    result = {key: deepcopy(card[key]) for key in CARD_FIELDS if key in card}
    if card.get('copy'):
        original = {'nanali': '娜娜莉', 'zero': '零', 'jiuyuan': '九原'}.get(card.get('original_character_id'))
        if original:
            result['description'] = result.get('description', '').replace(original, '浔')
    return result


def entity_id(side, cid):
    if side not in SIDES:
        return None
    if cid in (None, 'player'):
        return f'{side}:player'
    return f'{side}:{cid}'


def capture_public_board(state):
    board = {
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
        team = {key: deepcopy(source[key]) for key in ('hp', 'shield', 'ap', 'front', 'last_front')}
        team['hand_count'] = len(source['hand'])
        team['deck_count'] = len(source['deck'])
        pending = state.get('pending_choice')
        if pending and pending.get('side') == ps and pending.get('kind') == 'inspect_top':
            team['deck_count'] += len(pending.get('cards') or ())
        from . import state as duel_state
        characters = []
        for cid in ORDER:
            hero = source['characters'][cid]
            shape_id = hero.get('shape')
            characters.append({
                'id': cid,
                'hp': hero['hp'],
                'max_hp': hero['max_hp'],
                'shield': hero['shield'],
                'attack': duel_state.attack_value(hero),
                'harmony': hero['harmony'],
                'energy': hero['energy'],
                'awakened': hero['awakened'],
                'down_turns': hero['down_turns'],
                'growth': hero['growth'],
                'shape': CARDS[shape_id]['name'] if shape_id else None,
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
    for key in ('phase', 'active_side', 'turn', 'winner'):
        if before.get(key) != after.get(key):
            patch[key] = after.get(key)
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
    if event.get('type') in ('draw', 'turn', 'finish', 'fatigue') and side in SIDES:
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
    if event.get('card') is not None and not event.get('private_side'):
        event['card'] = public_card(event['card'])
    if event.get('private_card') is not None:
        event['private_card'] = public_card(event['private_card'])
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
                'hit_index', 'before', 'after'):
        if key in event:
            item[key] = deepcopy(event[key])
    if 'attack' not in item and event.get('type') == 'attack' and 'amount' in event:
        item['attack'] = event['amount']
    card = _visible_card(event, viewer)
    if card is not None:
        item['card'] = card
    patch = event.get('patch')
    if isinstance(patch, dict):
        item['patch'] = deepcopy(patch)
    return item


def localize_text(event, viewer):
    text = event.get('text') or ''
    text = text.replace('甲方玩家', '我方玩家' if viewer == 'a' else '对手玩家')
    text = text.replace('乙方玩家', '我方玩家' if viewer == 'b' else '对手玩家')
    if event.get('side') in SIDES:
        prefix = '我方 · ' if event['side'] == viewer else '对手 · '
        text = prefix + text
    return text


def sanitize_public_event(event, viewer):
    item = deepcopy(event)
    for key in PRIVATE_EVENT_KEYS + ('_public_board',):
        item.pop(key, None)
    if event.get('private_side') and event.get('private_side') != viewer:
        item.pop('card', None)
        item.pop('private_card', None)
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
    enemy = other(side)
    front = state['sides'][enemy]['front']
    return [entity_id(enemy, front if front else 'player')]


def interaction_for_action(state, side, action):
    kind = action.get('type')
    if kind == 'attack':
        return {
            'kind': 'attack',
            'actor_id': entity_id(side, action.get('character_id')),
            'target_ids': _enemy_front_targets(state, side),
        }
    if kind == 'awaken':
        actor = entity_id(side, action.get('character_id'))
        return {'kind': 'awaken', 'actor_id': actor, 'target_ids': [actor] if actor else []}
    if kind == 'play_card':
        card = next(c for c in state['sides'][side]['hand'] if c['instance_id'] == action['card_id'])
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
        if policy in ('enemy_hand', 'own_nonterminal_tactic_discard'):
            return {'kind': 'choice', 'actor_id': actor, 'target_ids': []}
        return {'kind': 'cast', 'actor_id': actor, 'target_ids': []}
    if kind in ('choose', 'mulligan', 'cycle'):
        actor = None
        operation = state.get('operation')
        if operation:
            actor = entity_id(operation.get('side'), operation.get('actor'))
        return {'kind': 'choice', 'actor_id': actor, 'target_ids': []}
    return {'kind': 'other', 'actor_id': None, 'target_ids': []}
