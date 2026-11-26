"""Batched stored-effect matching, without entity dictionaries or host queries.

Canonical source is engine/duel_v2/modifiers.py:_applies and CONDITIONS.
Kit-provided aura generation and legacy-flag migration must be performed by
their own qualified array stages before matching. This is not a full duel.
"""
from dataclasses import dataclass

INVALID_REFERENCE, UNKNOWN_CONDITION, INVALID_EFFECT = 1, 2, 4
SCOPES = ('self', 'allies', 'other_allies', 'enemies', 'all')
CONDITIONS = ('source_front', 'source_alive', 'source_awakened', 'source_shape',
              'enemy_turn', 'resource_at_least', 'damage_attribute', 'card_owner_other',
              'attribute_is')
supports_full_duel_rules = False


@dataclass(frozen=True)
class EffectEntities:
    present: object       # bool[B,H], players and characters, in formal enumeration order
    identity: object      # int64[B,H], -1 for an absent/null identity
    side: object          # int32[B,H], 0/1, declared source side
    hp: object            # int32[B,H]
    awakened: object      # bool[B,H]
    shape: object         # int32[B,H], -1 for null
    resources: object     # int32[B,H,R], missing resources supplied as zero
    attribute: object = None  # int32[B,H], -1 for missing/null; players may omit it


@dataclass(frozen=True)
class EffectTable:
    present: object       # bool[B,E]
    holder: object        # int32[B,E], entity table index
    source_identity: object  # int64[B,E], -1 for absent/null: source is holder
    property: object      # int32[B,E]
    scope: object         # int32[B,E], SCOPES
    aura: object          # bool[B,E]
    expiry_side: object   # int32[B,E], -1 => holder_side
    expiry_turn: object   # int32[B,E], clock value
    expiry_phase: object  # int32[B,E], -1 no absolute turn, 0 start, 1 end
    condition: object     # int32[B,E,K], -1 unused or CONDITIONS index
    argument: object      # int32[B,E,K], shape/attribute/resource index
    threshold: object     # int32[B,E,K], resource threshold


@dataclass(frozen=True)
class EffectQuery:
    subject: object       # int32[B], entity slot; subject is always a real holder
    side: object          # int32[B], query side
    property: object      # int32[B]
    active_side: object   # int32[B], -1 represents null state active side
    front_identity: object  # int64[B,2], -1 for no front
    turn_count: object    # int32[B,2]
    attribute: object     # int32[B], -1 for missing/null
    active: object        # bool[B]


@dataclass(frozen=True)
class EffectMatches:
    applies: object      # bool[B,E]
    source: object       # int64[B,E], -1 when source absent
    errors: object       # int32[B]


def match_effects(entities, effects, query):
    """Exact stored-effect scope, expiry, source and registered conditions.

    This interface requires a real game state. Source lookup returns the first
    identity match in canonical enumeration order, as formal source_for does.
    Identity-null holder/subject equality still uses slot identity.
    """
    import torch
    if not isinstance(entities, EffectEntities) or not isinstance(effects, EffectTable) or not isinstance(query, EffectQuery):
        raise TypeError('Typed SoA effect tables and query required')
    t = entities.present
    if not isinstance(t, torch.Tensor) or t.dtype != torch.bool or t.ndim != 2 or t.shape[1] < 1:
        raise ValueError('Nonempty resident entity slot table required')
    batch, holders = t.shape
    device = t.device

    def check(value, dtype, shape):
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.device != device or value.shape != shape:
            raise ValueError('Exact effect table shape/dtype/device required')

    for name, dtype in (('identity', torch.int64), ('side', torch.int32), ('hp', torch.int32),
                         ('awakened', torch.bool), ('shape', torch.int32)):
        check(getattr(entities, name), dtype, t.shape)
    resources = entities.resources
    if not isinstance(resources, torch.Tensor) or resources.ndim != 3 or resources.shape[:2] != t.shape or resources.shape[2] < 1:
        raise ValueError('Bounded resource columns required')
    check(resources, torch.int32, resources.shape)
    attributes = entities.attribute
    if attributes is None:
        attributes = torch.full_like(entities.hp, -1)
    check(attributes, torch.int32, t.shape)
    if not isinstance(effects.present, torch.Tensor) or effects.present.ndim != 2 or effects.present.shape[0] != batch:
        raise ValueError('Bounded effect table required')
    shape = effects.present.shape
    check(effects.present, torch.bool, shape)
    for name in ('holder', 'property', 'scope', 'expiry_side', 'expiry_turn', 'expiry_phase'):
        check(getattr(effects, name), torch.int32, shape)
    check(effects.source_identity, torch.int64, shape)
    check(effects.aura, torch.bool, shape)
    if not isinstance(effects.condition, torch.Tensor) or effects.condition.ndim != 3 or effects.condition.shape[:2] != shape:
        raise ValueError('Bounded condition slots required')
    for name in ('condition', 'argument', 'threshold'):
        check(getattr(effects, name), torch.int32, effects.condition.shape)
    for name in ('subject', 'side', 'property', 'active_side', 'attribute'):
        check(getattr(query, name), torch.int32, (batch,))
    check(query.active, torch.bool, (batch,))
    check(query.front_identity, torch.int64, (batch, 2))
    check(query.turn_count, torch.int32, (batch, 2))

    subject = query.subject.long().clamp(0, holders - 1)
    holder = effects.holder.long().clamp(0, holders - 1)

    def entity_field(field, slots):
        return field.gather(1, slots)

    holder_live = entity_field(t, holder)
    holder_side = entity_field(entities.side, holder)
    holder_identity = entity_field(entities.identity, holder)
    subject_identity = entity_field(entities.identity, subject[:, None])
    subject_attribute = entity_field(attributes, subject[:, None])
    relevant = effects.present & (effects.property == query.property[:, None])
    invalid_query = (query.subject < 0) | (query.subject >= holders) | (query.side < 0) | (query.side > 1)
    invalid_query |= ~entity_field(t, subject[:, None]).squeeze(1)
    invalid_ref = (effects.holder < 0) | (effects.holder >= holders) | ~holder_live
    invalid_ref |= (holder_side < 0) | (holder_side > 1)
    invalid = (effects.scope < 0) | (effects.scope >= len(SCOPES))
    invalid |= effects.source_identity < -1
    invalid |= (effects.expiry_side < -1) | (effects.expiry_side > 1)
    invalid |= (effects.expiry_phase < -1) | (effects.expiry_phase > 1)
    cond = effects.condition
    unknown_cond = ((cond < -1) | (cond >= len(CONDITIONS))).any(-1)
    # Unknown conditions are rejected, including ones hidden behind another
    # false condition. No unimplemented handler can silently acquire coverage.
    errors = (query.active & (invalid_query | (relevant & invalid_ref).any(-1))).int() * INVALID_REFERENCE
    errors |= (query.active & (relevant & invalid).any(-1)).int() * INVALID_EFFECT
    errors |= (query.active & (relevant & unknown_cond).any(-1)).int() * UNKNOWN_CONDITION

    same = (holder == subject[:, None]) | ((holder_identity >= 0) & (holder_identity == subject_identity))
    same_side = holder_side == query.side[:, None]
    scope = effects.scope
    scoped = ((scope == 0) & same) | ((scope == 1) & same_side)
    scoped |= (scope == 2) & same_side & ~same
    scoped |= ((scope == 3) & ~same_side) | (scope == 4)
    clock_side = torch.where(effects.expiry_side < 0, holder_side, effects.expiry_side)
    clock = query.turn_count.gather(1, clock_side.long().clamp(0, 1))
    timed_out = (effects.expiry_phase >= 0) & ((effects.expiry_turn < clock)
        | ((effects.expiry_phase == 0) & (effects.expiry_turn == clock)))

    requested = effects.source_identity
    use_holder = (requested < 0) | (requested == holder_identity)
    identity_match = (requested[:, :, None] == entities.identity[:, None, :]) & t[:, None, :]
    found = identity_match.any(-1)
    first = identity_match.int().argmax(-1)
    source = torch.where(use_holder, holder, first)
    source_live = use_holder | found
    source_hp = entity_field(entities.hp, source)
    source_side = entity_field(entities.side, source)
    source_identity = entity_field(entities.identity, source)
    source_shape = entity_field(entities.shape, source)
    source_awakened = entity_field(entities.awakened, source)
    invalid_source = source_live & ((source_side < 0) | (source_side > 1))
    errors |= (query.active & (relevant & invalid_source).any(-1)).int() * INVALID_REFERENCE
    applies = relevant & scoped & ~timed_out & (~effects.aura | (source_live & (source_hp > 0)))
    front = query.front_identity.gather(1, source_side.long().clamp(0, 1))
    selected_resources = resources.gather(1, source[:, :, None].expand(-1, -1, resources.shape[2]))
    for slot in range(cond.shape[2]):
        kind = cond[:, :, slot]
        arg = effects.argument[:, :, slot]
        threshold = effects.threshold[:, :, slot]
        resource_index_bad = (kind == 5) & ((arg < 0) | (arg >= resources.shape[2]))
        errors |= (query.active & (relevant & resource_index_bad).any(-1)).int() * INVALID_REFERENCE
        resource_value = selected_resources.gather(2, arg.long().clamp(0, resources.shape[2]-1)[:, :, None]).squeeze(-1)
        condition = kind == -1
        condition |= (kind == 0) & source_live & (front >= 0) & (front == source_identity)
        condition |= (kind == 1) & source_live & (source_hp > 0)
        condition |= (kind == 2) & source_live & source_awakened
        condition |= (kind == 3) & source_live & (source_shape == arg)
        condition |= (kind == 4) & (query.active_side[:, None] != holder_side)
        condition |= (kind == 5) & source_live & (resource_value >= threshold)
        condition |= (kind == 6) & (query.attribute[:, None] == arg)
        condition |= (kind == 7) & source_live & (source_identity != subject_identity)
        condition |= (kind == 8) & (subject_attribute == arg)
        applies &= condition
    applies &= query.active[:, None] & (errors == 0)[:, None]
    return EffectMatches(applies, torch.where(source_live, source, -1), errors)
