"""Match-local entities with persistent identity and JSON-compatible state.

Template IDs select catalog content. Entity IDs identify objects in one match.
The dict interface keeps the existing rules and JSON transport compatible;
identity comparisons must use entity_id, never dict equality or template IDs.
"""
from copy import deepcopy
from uuid import uuid4


class Entity(dict):
    kind = 'entity'

    def __init__(self, values=(), *, entity_id=None, side=None, **fields):
        super().__init__(values, **fields)
        if 'entity_id' not in self:
            self['entity_id'] = entity_id or 'standalone-' + uuid4().hex
        if side is not None:
            self['side'] = side

    def __deepcopy__(self, memo):
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for key, value in self.items():
            result[deepcopy(key, memo)] = deepcopy(value, memo)
        return result

    @property
    def entity_id(self):
        return self['entity_id']

    @property
    def side(self):
        return self.get('side')

    def is_same_entity(self, other):
        return isinstance(other, Entity) and self.entity_id == other.entity_id


class PlayerEntity(Entity):
    kind = 'player'


class CharacterEntity(Entity):
    kind = 'character'

    @property
    def template_id(self):
        return self['id']


class CardEntity(Entity):
    kind = 'card'

    @property
    def template_id(self):
        return self['card_id']


def allocate_entity_id(state):
    state['next_entity'] = int(state.get('next_entity', 0)) + 1
    return 'e' + str(state['next_entity'])


def hydrate_entities(state):
    """Rebuild classes/aliases in place after JSON loading; legacy IDs assigned once.

    Only authoritative live objects are hydrated. Historical event projections,
    catalog definitions and Xun's records are values, not additional entities.
    """
    slots = []
    for side, team in state['sides'].items():
        slots.append((state['sides'], side, PlayerEntity, side, ('player', side)))
        for cid in team['characters']:
            slots.append((team['characters'], cid, CharacterEntity, side, ('character', side, cid)))
        for zone in ('hand', 'deck', 'discard', 'removed'):
            for i, card in enumerate(team.get(zone, [])):
                if card.get('instance_id'):
                    slots.append((team[zone], i, CardEntity, side, ('card', card['instance_id'])))
    op = state.get('operation') or {}
    if (op.get('card') or {}).get('instance_id'):
        slots.append((op, 'card', CardEntity, op['side'], ('card', op['card']['instance_id'])))
    choice = state.get('pending_choice') or {}
    for i, card in enumerate(choice.get('cards', [])):
        if card.get('instance_id'):
            slots.append((choice['cards'], i, CardEntity, choice['side'], ('card', card['instance_id'])))
    high = int(state.get('next_entity', 0))
    for container, key, _, _, _ in slots:
        eid = container[key].get('entity_id', '')
        if isinstance(eid, str) and eid.startswith('e') and eid[1:].isdigit():
            high = max(high, int(eid[1:]))
    state['next_entity'] = high
    objects, identities = {}, {}
    for container, key, cls, side, logical in slots:
        value = container[key]
        if logical in objects:
            previous = objects[logical]
            if value.get('entity_id') not in (None, previous.entity_id):
                raise ValueError('Conflicting entity IDs for one object')
            container[key] = previous
            continue
        if 'entity_id' in value and not value['entity_id']:
            raise ValueError('Empty entity ID')
        eid = value.get('entity_id') or allocate_entity_id(state)
        if not isinstance(eid, str) or not eid or eid in identities:
            raise ValueError('Invalid or duplicate entity ID')
        identities[eid] = logical
        if not isinstance(value, cls):
            value = cls(value, entity_id=eid, side=side)
        else:
            value['side'] = side
        container[key] = value
        objects[logical] = value
    return state


def clone_state(state):
    return hydrate_entities(deepcopy(state))


def character_entity(state, side, template_id):
    team = state['sides'][side]
    value = team['characters'][template_id]
    if not isinstance(value, CharacterEntity):
        value = CharacterEntity(value, entity_id=value.get('entity_id') or allocate_entity_id(state), side=side)
        team['characters'][template_id] = value
    value['side'] = side
    return value


def resolve_entity(state, entity_id):
    """Resolve a live entity by identity, including cards outside the hand."""
    hydrate_entities(state)
    for team in state['sides'].values():
        if team.entity_id == entity_id:
            return team
        for entity in team['characters'].values():
            if entity.entity_id == entity_id:
                return entity
        for zone in ('hand', 'deck', 'discard', 'removed'):
            for entity in team.get(zone, []):
                if entity.get('entity_id') == entity_id:
                    return entity
    op = state.get('operation') or {}
    card = op.get('card')
    if card and card.get('entity_id') == entity_id:
        return card
    for card in (state.get('pending_choice') or {}).get('cards', []):
        if card.get('entity_id') == entity_id:
            return card
    raise KeyError(entity_id)


def normalize_entity_action(state, side, action):
    """Translate entity-addressed commands to the legacy wire shape for validation."""
    result = deepcopy(action)
    for field, legacy in (('actor_entity_id', 'character_id'), ('card_entity_id', 'card_id'),
                          ('target_entity_id', 'target_id'), ('choice_entity_id', 'choice_id')):
        if field not in result:
            continue
        try:
            entity = resolve_entity(state, result.pop(field))
        except KeyError as exc:
            raise ValueError('Unknown entity ID') from exc
        if field == 'actor_entity_id':
            if not isinstance(entity, CharacterEntity) or entity.side != side:
                raise ValueError('Actor must be your character entity')
            value = entity.template_id
        elif field in ('card_entity_id', 'choice_entity_id'):
            if not isinstance(entity, CardEntity):
                raise ValueError('Expected a card entity')
            value = entity['instance_id']
        elif isinstance(entity, CardEntity):
            value = entity['instance_id']
        else:
            value = entity.side + ':' + ('player' if isinstance(entity, PlayerEntity) else entity.template_id)
        if legacy in result and result[legacy] != value:
            raise ValueError('Conflicting legacy and entity target')
        result[legacy] = value
    if 'card_entity_ids' in result:
        ids = result.pop('card_entity_ids')
        if not isinstance(ids, list):
            raise ValueError('Expected card entity ID list')
        values = []
        for eid in ids:
            try:
                card = resolve_entity(state, eid)
            except KeyError as exc:
                raise ValueError('Unknown card entity ID') from exc
            if not isinstance(card, CardEntity) or card.side != side:
                raise ValueError('Expected your card entity')
            values.append(card['instance_id'])
        if 'card_ids' in result and result['card_ids'] != values:
            raise ValueError('Conflicting mulligan entity IDs')
        result['card_ids'] = values
    return result


def entity_action(state, side, action):
    """Preferred entity-addressed action, alongside the compatible legacy action."""
    result = deepcopy(action)
    if 'character_id' in result:
        result['actor_entity_id'] = state['sides'][side]['characters'][result.pop('character_id')]['entity_id']
    all_cards = [c for team in state['sides'].values() for zone in ('hand', 'deck', 'discard', 'removed')
                 for c in team.get(zone, [])]
    all_cards += (state.get('pending_choice') or {}).get('cards', [])
    cards = {c['instance_id']: c for c in all_cards if c.get('instance_id')}
    for legacy, field in (('card_id', 'card_entity_id'), ('choice_id', 'choice_entity_id')):
        if result.get(legacy) in cards:
            result[field] = cards[result.pop(legacy)]['entity_id']
    target = result.get('target_id')
    if target in cards:
        result['target_entity_id'] = cards[target]['entity_id']
        result.pop('target_id')
    elif isinstance(target, str) and ':' in target:
        owner, cid = target.split(':', 1)
        if owner in state['sides']:
            team = state['sides'][owner]
            obj = team if cid == 'player' else team['characters'].get(cid)
            if obj:
                result['target_entity_id'] = obj['entity_id']
                result.pop('target_id')
    if 'card_ids' in result:
        result['card_entity_ids'] = [cards[cid]['entity_id'] for cid in result.pop('card_ids')]
    return result


def needs_hydration(state):
    for team in state['sides'].values():
        if not isinstance(team, PlayerEntity):
            return True
        if any(not isinstance(h, CharacterEntity) for h in team['characters'].values()):
            return True
        if any(not isinstance(c, CardEntity) for zone in ('hand', 'deck', 'discard', 'removed') for c in team.get(zone, [])):
            return True
    return False
