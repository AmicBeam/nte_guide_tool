"""Bounded tensor-only transition orchestration over per-field domain SoA.

Named resident columns, prepared decision masks, explicit coverage. Event and
continuation queues are bounded tensors. Unknown hooks and unsupported
seven-team actions fail closed. This is not a Python-dict stepper, native
graph backend, or CPU rule fallback, and arithmetic stages are not a complete
duel.
"""
from dataclasses import dataclass
from types import MappingProxyType

from .domain_soa_rules import (
    INTEGER_OVERFLOW, INVALID_DOMAIN, UNKNOWN_OPERATION,
    capped_add, compact_slots, damage_fields, expire_effects, reduce_integer_modifiers,
    resource_caps, shield_add, supports_full_duel_rules as _RULES_FULL_DUEL,
)

# Transition bits do not overlap domain_soa_rules 1/2/4.
UNSUPPORTED_ACTION = 8
UNKNOWN_HOOK = 16
INVALID_ROW = 32
QUEUE_OVERFLOW = 64
ILLEGAL_ACTION = 128
EVALUATION_ERROR = 256
supports_full_duel_rules = False
if _RULES_FULL_DUEL:
    raise RuntimeError('domain_soa_rules must not claim full duel coverage')

# IDs 0-8 follow seven_lineup.ACTION_TYPES; concede is the extra physical action.
ACTION_END_TURN = 0
ACTION_ATTACK = 1
ACTION_ULTIMATE = 2
ACTION_PLAY_CARD = 3
ACTION_CHOOSE = 4
ACTION_MULLIGAN = 5
ACTION_CHOOSE_DECK_ORDER = 6
ACTION_EXIT_ULTIMATE = 7
ACTION_CHOOSE_CARDS = 8
ACTION_CONCEDE = 9
SUPPORTED_ACTIONS = (ACTION_CONCEDE, ACTION_ATTACK, ACTION_END_TURN)
REJECTED_ACTIONS = (
    ACTION_ULTIMATE, ACTION_PLAY_CARD, ACTION_CHOOSE, ACTION_MULLIGAN,
    ACTION_CHOOSE_DECK_ORDER, ACTION_EXIT_ULTIMATE, ACTION_CHOOSE_CARDS,
)

PHASE_MULLIGAN, PHASE_PLAYING, PHASE_CHOICE, PHASE_FINISHED = 0, 1, 2, 3
WINNER_ONGOING, WINNER_DRAW = -2, -1
SIDES, MAX_TEAM, MAX_EFFECTS, MAX_QUEUE = 2, 4, 16, 16
MAX_LEGAL, MAX_HOOKS, MAX_PENDING = 256, 8, 4
EXPIRY_TURN_START, EXPIRY_TURN_END = 0, 1
EVENT_NONE, EVENT_DAMAGE, EVENT_KNOCKDOWN, EVENT_FINISH, EVENT_RESOURCE, EVENT_EXPIRE = range(6)
CONT_NONE, CONT_DEFERRED_ATTACK, CONT_CHOICE, CONT_PENDING_SORTIE = range(4)
KNOWN_EVENTS = (EVENT_DAMAGE, EVENT_KNOCKDOWN, EVENT_FINISH, EVENT_RESOURCE, EVENT_EXPIRE)
KNOWN_CONTINUATIONS = (CONT_DEFERRED_ATTACK, CONT_CHOICE, CONT_PENDING_SORTIE)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    tail: tuple = ()


DEFAULT_FIELD_SPECS = (
    FieldSpec('phase', 'int32'), FieldSpec('turn', 'int32'), FieldSpec('active_side', 'int32'),
    FieldSpec('escalation_enabled', 'bool'),
    FieldSpec('player_hp', 'int32', (2,)), FieldSpec('player_shield', 'int32', (2,)),
    FieldSpec('player_ap', 'int32', (2,)), FieldSpec('life_locked', 'bool', (2,)),
    FieldSpec('front', 'int32', (2,)), FieldSpec('turn_count', 'int32', (2,)),
    FieldSpec('normal_attack_available', 'bool', (2,)),
    FieldSpec('char_hp', 'int32', (2, 4)), FieldSpec('char_max_hp', 'int32', (2, 4)),
    FieldSpec('char_shield', 'int32', (2, 4)), FieldSpec('char_harmony', 'int32', (2, 4)),
    FieldSpec('char_energy', 'int32', (2, 4)), FieldSpec('char_energy_cap', 'int32', (2, 4)),
    FieldSpec('char_down_turns', 'int32', (2, 4)), FieldSpec('char_awakened', 'bool', (2, 4)),
    FieldSpec('char_present', 'bool', (2, 4)), FieldSpec('char_growth', 'int32', (2, 4)),
    FieldSpec('char_shape', 'int32', (2, 4)),
    FieldSpec('effect_present', 'bool', (MAX_EFFECTS,)),
    FieldSpec('effect_layer', 'int32', (MAX_EFFECTS,)), FieldSpec('effect_op', 'int32', (MAX_EFFECTS,)),
    FieldSpec('effect_amount', 'int32', (MAX_EFFECTS,)),
    FieldSpec('effect_expiry_side', 'int32', (MAX_EFFECTS,)),
    FieldSpec('effect_expiry_phase', 'int32', (MAX_EFFECTS,)),
    FieldSpec('effect_expiry_turn', 'int32', (MAX_EFFECTS,)),
    FieldSpec('effect_owner_side', 'int32', (MAX_EFFECTS,)),
    FieldSpec('event_present', 'bool', (MAX_QUEUE,)), FieldSpec('event_kind', 'int32', (MAX_QUEUE,)),
    FieldSpec('event_side', 'int32', (MAX_QUEUE,)), FieldSpec('event_actor', 'int32', (MAX_QUEUE,)),
    FieldSpec('cont_present', 'bool', (MAX_QUEUE,)), FieldSpec('cont_kind', 'int32', (MAX_QUEUE,)),
    FieldSpec('cont_side', 'int32', (MAX_QUEUE,)), FieldSpec('cont_actor', 'int32', (MAX_QUEUE,)),
)
EFFECT_VALUE_FIELDS = (
    'effect_layer', 'effect_op', 'effect_amount', 'effect_expiry_side',
    'effect_expiry_phase', 'effect_expiry_turn', 'effect_owner_side',
)
SCHEMA_EXECUTION_ALIASES = (
    ('character_hp', 'char_hp'),
    ('character_shield', 'char_shield'),
    ('character_energy', 'char_energy'),
    ('character_valid', 'char_present'),
    ('player_front_slot', 'front'),
)


@dataclass(frozen=True)
class DomainSoACoverage:
    supported_actions: tuple
    supported_stages: tuple
    rejected_actions: tuple
    remaining_for_complete_search: tuple
    supports_full_duel_rules: bool = False
    schema_bound: bool = False


COVERAGE = DomainSoACoverage(
    supported_actions=SUPPORTED_ACTIONS,
    supported_stages=(
        'concede', 'ordinary_combat_after_prepared_masks',
        'end_turn_clock_expiry_without_zone_or_kit', 'damage_settlement',
        'resource_capped_add', 'shield_add', 'integer_modifier_reduction',
        'effect_expiry_and_compaction', 'knockdown_numeric_fields',
        'player_hp_terminal', 'bounded_event_and_continuation_queues',
        'legal_action_selection', 'true_terminal_fixed_viewer_metadata',
    ),
    rejected_actions=REJECTED_ACTIONS,
    remaining_for_complete_search=(
        'kit fire/decide as resident programs over all seven teams',
        'card play, choices, mulligan, deck order, ultimate, exit ultimate',
        'harmony enter/kind, responses, redirect_overflow, weave/stain/dots',
        'summons, weapons, shapes, collapse, copy/derived cards',
        'draw/shuffle RNG and empty-deck finish',
        'natural front return, zone clocks, and effect-expiry dispatch',
        'information-set sampling and public observation encoding',
        'full seven-team legal-action generation',
        'Gumbel search, teacher continuations, native graph and VM backends',
    ),
)


@dataclass(frozen=True)
class PreparedDecisionMasks:
    """Diagnostic stage inputs, not a complete formal action-validation result.

    A caller must resolve entering/harmony, kit hooks, targets, responses,
    immunity, redirection, and draw/zone blockers before selecting a slice.
    There is currently no qualified seven-team producer of these columns.
    """
    legal: object = None
    legal_types: object = None
    hooks: object = None
    clock_blockers: object = None
    attack_amount: object = None
    counter_amount: object = None
    overflow_amount: object = None
    bypass: object = None
    defender_immune: object = None
    attacker_immune: object = None
    defender_life_locked: object = None
    attacker_life_locked: object = None
    defender_floor: object = None
    attacker_floor: object = None
    defender_limit: object = None
    attacker_limit: object = None
    target_is_player: object = None
    attacker_side: object = None
    attacker_seat: object = None
    defender_seat: object = None
    can_gain_energy: object = None
    ap_cost: object = None
    enqueue_kind: object = None
    enqueue_side: object = None
    enqueue_actor: object = None
    continuation_kind: object = None
    continuation_side: object = None
    continuation_actor: object = None


@dataclass(frozen=True)
class DomainSoAState:
    fields: object
    actors: object
    finished: object
    winners: object
    errors: object


@dataclass(frozen=True)
class DomainSoAEvaluation:
    logits: object
    valid: object
    wdl: object
    errors: object


@dataclass(frozen=True)
class DomainSoARecurrentOutput:
    state: DomainSoAState
    reward: object
    discount: object
    prior_logits: object
    valid: object
    value: object
    errors: object


def _load_schema():
    try:
        from . import domain_soa_schema as schema
        return schema
    except ImportError:
        return None


def coverage(*, schema=None):
    module = schema if schema is not None else _load_schema()
    # Recognizing a few column aliases is not a complete codec/executor ABI.
    bound = bool(module is not None and
                 getattr(module, 'COMPLETE_TRANSITION_ABI', None) == 'domain_soa_v1')
    return DomainSoACoverage(
        COVERAGE.supported_actions, COVERAGE.supported_stages, COVERAGE.rejected_actions,
        COVERAGE.remaining_for_complete_search, False, bound)


def _dtype_name(dtype):
    name = getattr(dtype, 'name', None) or str(dtype)
    if 'bool' in name:
        return 'bool'
    if 'int32' in name:
        return 'int32'
    raise ValueError('SoA columns must be int32 or bool')


def _as_spec(item):
    if isinstance(item, FieldSpec):
        return item
    if isinstance(item, dict):
        return FieldSpec(item['name'], _dtype_name(item.get('dtype', 'int32')),
                         tuple(item.get('tail') or item.get('shape_tail') or ()))
    if isinstance(item, (tuple, list)):
        tail = tuple(item[2]) if len(item) > 2 else ()
        return FieldSpec(item[0], _dtype_name(item[1]), tail)
    return FieldSpec(item.name, _dtype_name(getattr(item, 'dtype', 'int32')),
                     tuple(getattr(item, 'tail', getattr(item, 'shape_tail', ()))))


def resolve_field_specs(specs=None, schema=None):
    if specs is not None:
        return tuple(_as_spec(item) for item in specs)
    module = schema if schema is not None else _load_schema()
    if module is None:
        return DEFAULT_FIELD_SPECS
    raw = getattr(module, 'FIELD_SPECS', None) or getattr(module, 'COLUMNS', None)
    if raw is None and hasattr(module, 'fields'):
        raw = module.fields() if callable(module.fields) else module.fields
    return DEFAULT_FIELD_SPECS if not raw else tuple(_as_spec(item) for item in raw)


def _alias_schema_names(fields):
    import torch
    out = dict(fields)
    for schema_name, exec_name in SCHEMA_EXECUTION_ALIASES:
        if schema_name in out and exec_name not in out:
            tensor = out[schema_name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(schema_name + ' must be a resident tensor')
            if exec_name == 'char_present':
                out[exec_name] = tensor if tensor.dtype == torch.bool else tensor != 0
            elif tensor.dtype == torch.int32:
                out[exec_name] = tensor
            else:
                out[exec_name] = tensor.to(torch.int32)
    return out


def _write_aliases(fields):
    out = dict(fields)
    for schema_name, exec_name in SCHEMA_EXECUTION_ALIASES:
        if schema_name in out and exec_name in out and out[schema_name] is not out[exec_name]:
            out[schema_name] = out[exec_name].to(dtype=out[schema_name].dtype)
    return out


def bind_columns(fields, *, specs=None, schema=None):
    """Validate named resident tensors. A later schema can supply the same names."""
    import torch
    if not isinstance(fields, dict) or not fields:
        raise TypeError('Named SoA field mapping required')
    module = schema if schema is not None else _load_schema()
    binder = getattr(module, 'bind_columns', None) if module is not None else None
    if callable(binder) and specs is None:
        fields = binder(fields)
        if not isinstance(fields, dict):
            raise TypeError('Schema bind_columns must return a named field mapping')
    fields = _alias_schema_names(dict(fields))
    specs = resolve_field_specs(specs, None if specs is not None else schema)
    by_name = {spec.name: spec for spec in specs}
    extra = set(fields) - set(by_name)
    allowed_extra = set(src for src, _ in SCHEMA_EXECUTION_ALIASES) | set(getattr(module, 'ARRAY_NAMES', ()) if module is not None else ())
    extra = extra - allowed_extra
    if extra:
        raise ValueError('Unknown SoA columns: ' + ', '.join(sorted(extra)))
    missing = [spec.name for spec in specs if spec.name not in fields]
    if missing:
        raise ValueError('Missing SoA columns: ' + ', '.join(missing))
    first = fields[specs[0].name]
    if not isinstance(first, torch.Tensor) or first.ndim < 1:
        raise ValueError('Resident batched SoA columns required')
    rows, device = first.shape[0], first.device
    bound = {}
    for spec in specs:
        value = fields[spec.name]
        expected = (rows,) + tuple(spec.tail)
        dtype = torch.bool if spec.dtype == 'bool' else torch.int32
        if (spec.tail and spec.tail[-1] == MAX_TEAM and tuple(value.shape[:-1]) == (rows,) + spec.tail[:-1]
                and value.shape[-1] >= MAX_TEAM):
            expected = tuple(value.shape)
        if (not isinstance(value, torch.Tensor) or value.device != device
                or value.dtype != dtype or tuple(value.shape) != expected):
            raise ValueError(spec.name + ' has an incompatible SoA layout')
        bound[spec.name] = value
    for name, value in fields.items():
        if name not in bound:
            bound[name] = value
    return MappingProxyType(bound)


def make_state(fields, actors, finished, winners, errors, *, specs=None, schema=None):
    import torch
    bound = bind_columns(fields, specs=specs, schema=schema)
    sample = bound[next(iter(bound))]
    rows, device = sample.shape[0], sample.device
    for name, value, dtype in (
            ('actors', actors, torch.int64), ('finished', finished, torch.bool),
            ('winners', winners, torch.int64), ('errors', errors, torch.int32)):
        if (not isinstance(value, torch.Tensor) or value.device != device
                or value.dtype != dtype or tuple(value.shape) != (rows,)):
            raise ValueError(name + ' has an incompatible SoA metadata layout')
    return DomainSoAState(bound, actors, finished, winners, errors)


def empty_fields(rows, *, device='cpu', specs=None, schema=None):
    import torch
    specs = resolve_field_specs(specs, schema)
    device = torch.device(device)
    fields = {}
    for spec in specs:
        dtype = torch.bool if spec.dtype == 'bool' else torch.int32
        fill = False if spec.dtype == 'bool' else 0
        tensor = torch.full((rows,) + tuple(spec.tail), fill, dtype=dtype, device=device)
        if spec.name == 'front':
            tensor.fill_(-1)
        if spec.name == 'char_present':
            tensor.fill_(True)
        if spec.name == 'char_max_hp':
            tensor.fill_(10)
        if spec.name == 'char_energy_cap':
            tensor.fill_(5)
        if spec.name == 'player_hp':
            tensor.fill_(30)
        if spec.name == 'phase':
            tensor.fill_(PHASE_PLAYING)
        fields[spec.name] = tensor
    return fields


def _clone_fields(fields):
    return {name: value.clone() for name, value in fields.items()}


def _broadcast(mask, like):
    return mask.reshape(mask.shape + (1,) * (like.ndim - mask.ndim))


def _row_or(errors):
    while errors.ndim > 1:
        errors = errors.amax(-1)
    return errors.int()


def _hot(index, width, device):
    import torch
    return torch.arange(width, device=device)[None, :] == index[:, None]


def _gather_side(field, side):
    import torch
    rows = torch.arange(field.shape[0], device=field.device)
    return field[rows, side.clamp(0, 1).long()]


def _gather_char(field, side, seat):
    import torch
    rows = torch.arange(field.shape[0], device=field.device)
    seats = field.shape[-1]
    return field[rows, side.clamp(0, 1).long(), seat.clamp(0, seats - 1).long()]


def _scatter_char(field, side, seat, work, values):
    seats = field.shape[-1]
    sel = work[:, None, None] & _hot(side, 2, field.device)[:, :, None] & _hot(seat, seats, field.device)[:, None, :]
    return field.where(~sel, _broadcast(values, field))


def _scatter_side(field, side, work, values):
    sel = work[:, None] & _hot(side, field.shape[1], field.device)
    return field.where(~sel, _broadcast(values, field))


def _require(value, name, like, dtype):
    import torch
    if value is None:
        return torch.zeros(like.shape, dtype=dtype, device=like.device)
    if (not isinstance(value, torch.Tensor) or value.device != like.device
            or value.dtype != dtype or tuple(value.shape) != tuple(like.shape)):
        raise ValueError(name + ' must match the prepared mask layout')
    return value


def select_legal_action(legal, legal_types, slots, active):
    """Map a legal slot to its action type. Illegal or OOB slots fail closed."""
    import torch
    if (not isinstance(slots, torch.Tensor) or slots.dtype != torch.int64
            or slots.ndim != 1 or slots.shape != legal.shape[:1]
            or slots.device != legal.device):
        raise ValueError('Resident int64 legal slots required')
    if legal.dtype != torch.bool or legal.ndim != 2:
        raise ValueError('Prepared bool legal mask [B,M] required')
    if (not isinstance(legal_types, torch.Tensor) or legal_types.dtype != torch.int32
            or tuple(legal_types.shape) != tuple(legal.shape) or legal_types.device != legal.device):
        raise ValueError('Prepared int32 legal types required')
    width = legal.shape[1]
    in_range = (slots >= 0) & (slots < width)
    safe = slots.clamp(0, max(width - 1, 0))
    picked = legal.gather(1, safe[:, None]).squeeze(1) & in_range
    types = legal_types.gather(1, safe[:, None]).squeeze(1)
    errors = (active & ~picked).int() * ILLEGAL_ACTION
    actions = torch.where(active & (errors == 0), types.long(), torch.zeros_like(slots))
    return actions, errors


def enqueue_queue(present, kind, side, actor, new_kind, new_side, new_actor, work, known):
    """Write one bounded queue slot. Unknown kinds and full queues fail closed."""
    import torch
    count = present.sum(-1).int()
    width = present.shape[1]
    unknown = work & (new_kind != 0)
    for code in known:
        unknown = unknown & (new_kind != code)
    errors = unknown.int() * UNKNOWN_HOOK
    overflow = work & (errors == 0) & (new_kind != 0) & (count >= width)
    errors = errors | overflow.int() * QUEUE_OVERFLOW
    accept = work & (errors == 0) & (new_kind != 0)
    index = count.clamp(max=width - 1)
    write = accept[:, None] & (torch.arange(width, device=present.device)[None, :] == index[:, None])
    return (present | write,
            torch.where(write, new_kind[:, None], kind),
            torch.where(write, new_side[:, None], side),
            torch.where(write, new_actor[:, None], actor),
            errors)


def _compact_effects(fields, present, active):
    out = dict(fields)
    _, kept, perm = compact_slots(fields['effect_layer'], present, active)
    out['effect_present'] = kept
    for name in EFFECT_VALUE_FIELDS:
        index = perm.reshape(perm.shape + (1,) * (fields[name].ndim - 2)).expand_as(fields[name])
        out[name] = fields[name].gather(1, index)
    return out


def _pending(rows, device):
    import torch
    z = torch.zeros((rows, MAX_PENDING), dtype=torch.int32, device=device)
    return torch.zeros((rows, MAX_PENDING), dtype=torch.bool, device=device), z, z.clone(), z.clone()


def _write_pending(present, kind, side, actor, slot, mask, event, event_side, event_actor):
    import torch
    write = mask[:, None] & (torch.arange(MAX_PENDING, device=mask.device)[None, :] == slot)
    return (present | write, torch.where(write, event[:, None], kind),
            torch.where(write, event_side[:, None], side),
            torch.where(write, event_actor[:, None], actor))


def _flush_pending(fields, present, kind, side, actor, work, known):
    errors = work.new_zeros(work.shape, dtype=fields['event_kind'].dtype)
    event_present, event_kind = fields['event_present'], fields['event_kind']
    event_side, event_actor = fields['event_side'], fields['event_actor']
    for slot in range(MAX_PENDING):
        event_present, event_kind, event_side, event_actor, step = enqueue_queue(
            event_present, event_kind, event_side, event_actor,
            kind[:, slot], side[:, slot], actor[:, slot], work & present[:, slot], known)
        errors = errors | step
    fields = dict(fields)
    fields['event_present'], fields['event_kind'] = event_present, event_kind
    fields['event_side'], fields['event_actor'] = event_side, event_actor
    return fields, errors


def _victory(fields, finished, winners, work):
    import torch
    dead_a, dead_b = fields['player_hp'][:, 0] <= 0, fields['player_hp'][:, 1] <= 0
    over = work & (dead_a | dead_b)
    draw = dead_a & dead_b
    winner = torch.where(draw, WINNER_DRAW, torch.where(dead_a, 1, torch.where(dead_b, 0, winners)))
    fields = dict(fields)
    fields['phase'] = torch.where(over, PHASE_FINISHED, fields['phase'])
    return fields, finished | over, torch.where(over, winner, winners), over


def _knockdown(fields, side, seat, work):
    import torch
    zero = torch.zeros_like(work, dtype=torch.int32)
    three = torch.full_like(zero, 3)
    fields = dict(fields)
    fields['char_hp'] = _scatter_char(fields['char_hp'], side, seat, work, zero)
    fields['char_shield'] = _scatter_char(fields['char_shield'], side, seat, work, zero)
    fields['char_growth'] = _scatter_char(fields['char_growth'], side, seat, work, zero)
    fields['char_shape'] = _scatter_char(fields['char_shape'], side, seat, work, zero)
    fields['char_down_turns'] = _scatter_char(fields['char_down_turns'], side, seat, work, three)
    fields['char_awakened'] = _scatter_char(
        fields['char_awakened'], side, seat, work, torch.zeros_like(work))
    match = _hot(side, 2, work.device) & (fields['front'] == seat[:, None])
    fields['front'] = torch.where(work[:, None] & match, torch.full_like(fields['front'], -1), fields['front'])
    return fields


def grant_energy(energy, amount, cap, eligible, active):
    return capped_add(energy, amount, cap, eligible, active)


def grant_harmony(harmony, amount, cap, eligible, active):
    return capped_add(harmony, amount, cap, eligible, active)


def grant_shield(shield, amount, eligible, active):
    return shield_add(shield, amount, eligible, active)


def settle_damage(hp, shield, amount, bypass, immune, life_locked, floor, limit, active):
    return damage_fields(hp, shield, amount, bypass, immune, life_locked, floor, limit, active)


def reduce_prepared_modifiers(base, layers, operations, amounts, present, active, **kwargs):
    return reduce_integer_modifiers(base, layers, operations, amounts, present, active, **kwargs)


def expire_and_compact(fields, current_side, current_phase, active):
    kept, expired, errors = expire_effects(
        fields['effect_expiry_side'], fields['effect_expiry_phase'], fields['effect_expiry_turn'],
        fields['effect_present'], fields['effect_owner_side'], current_side, current_phase,
        fields['turn_count'], active)
    out = _compact_effects(fields, kept, active)
    return out, expired, errors


def _ordinary_combat(fields, masks, work, errors):
    import torch
    rows, device = work.shape[0], work.device
    like = work
    if masks.attacker_side is None or masks.attacker_seat is None or masks.attack_amount is None:
        pending = _pending(rows, device)
        none = torch.zeros_like(work)
        ongoing = torch.full((rows,), WINNER_ONGOING, dtype=torch.int64, device=device)
        return fields, errors | work.int() * UNSUPPORTED_ACTION, *pending, none, ongoing
    atk_side = _require(masks.attacker_side, 'attacker_side', like, torch.int32)
    atk_seat = _require(masks.attacker_seat, 'attacker_seat', like, torch.int32)
    def_seat = _require(masks.defender_seat, 'defender_seat', like, torch.int32)
    vs_player = _require(masks.target_is_player, 'target_is_player', like, torch.bool)
    amount = _require(masks.attack_amount, 'attack_amount', like, torch.int32)
    counter = _require(masks.counter_amount, 'counter_amount', like, torch.int32)
    overflow = _require(masks.overflow_amount, 'overflow_amount', like, torch.int32)
    ap_cost = _require(masks.ap_cost, 'ap_cost', like, torch.int32)
    bypass = _require(masks.bypass, 'bypass', like, torch.bool)
    d_imm = _require(masks.defender_immune, 'defender_immune', like, torch.bool)
    a_imm = _require(masks.attacker_immune, 'attacker_immune', like, torch.bool)
    d_lock = _require(masks.defender_life_locked, 'defender_life_locked', like, torch.bool)
    a_lock = _require(masks.attacker_life_locked, 'attacker_life_locked', like, torch.bool)
    d_floor = _require(masks.defender_floor, 'defender_floor', like, torch.int32)
    a_floor = _require(masks.attacker_floor, 'attacker_floor', like, torch.int32)
    neg1 = torch.full_like(amount, -1)
    d_limit = masks.defender_limit if masks.defender_limit is not None else neg1
    a_limit = masks.attacker_limit if masks.attacker_limit is not None else neg1
    d_limit = _require(d_limit, 'defender_limit', like, torch.int32)
    a_limit = _require(a_limit, 'attacker_limit', like, torch.int32)
    def_side = 1 - atk_side
    ap = _gather_side(fields['player_ap'], atk_side)
    seats = fields['char_hp'].shape[-1]
    bad_id = ((atk_side < 0) | (atk_side > 1) | (atk_seat < 0) | (atk_seat >= seats)
              | (atk_side != fields['active_side'])
              | (~vs_player & ((def_seat < 0) | (def_seat >= seats))))
    insufficient = (ap < ap_cost) | (ap_cost < 0)
    errors = errors | (work & bad_id).int() * INVALID_ROW | (work & insufficient).int() * ILLEGAL_ACTION
    combat = work & (errors == 0)
    fields = dict(fields)
    fields['player_ap'] = _scatter_side(fields['player_ap'], atk_side, combat, ap - ap_cost)
    fields['normal_attack_available'] = _scatter_side(
        fields['normal_attack_available'], atk_side, combat, torch.zeros_like(combat))
    def_hp = torch.where(vs_player, _gather_side(fields['player_hp'], def_side),
                         _gather_char(fields['char_hp'], def_side, def_seat))
    def_sh = torch.where(vs_player, _gather_side(fields['player_shield'], def_side),
                         _gather_char(fields['char_shield'], def_side, def_seat))
    atk_hp = _gather_char(fields['char_hp'], atk_side, atk_seat)
    atk_sh = _gather_char(fields['char_shield'], atk_side, atk_seat)
    hit = settle_damage(def_hp, def_sh, amount, bypass, d_imm, d_lock, d_floor, d_limit, combat)
    riposte = settle_damage(atk_hp, atk_sh, torch.where(vs_player, torch.zeros_like(counter), counter),
                            torch.zeros_like(bypass), a_imm, a_lock, a_floor, a_limit, combat)
    errors = errors | hit.errors | riposte.errors
    combat = combat & (errors == 0)
    fields['char_hp'] = _scatter_char(fields['char_hp'], atk_side, atk_seat, combat, riposte.hp)
    fields['char_shield'] = _scatter_char(fields['char_shield'], atk_side, atk_seat, combat, riposte.shield)
    fields['char_hp'] = _scatter_char(fields['char_hp'], def_side, def_seat, combat & ~vs_player, hit.hp)
    fields['char_shield'] = _scatter_char(fields['char_shield'], def_side, def_seat, combat & ~vs_player, hit.shield)
    fields['player_hp'] = _scatter_side(fields['player_hp'], def_side, combat & vs_player, hit.hp)
    fields['player_shield'] = _scatter_side(fields['player_shield'], def_side, combat & vs_player, hit.shield)
    ply = settle_damage(
        _gather_side(fields['player_hp'], def_side), _gather_side(fields['player_shield'], def_side),
        overflow, torch.zeros_like(bypass), torch.zeros_like(d_imm),
        _gather_side(fields['life_locked'], def_side),
        torch.zeros_like(d_floor), neg1, combat & ~vs_player & (overflow > 0))
    errors = errors | ply.errors
    combat = combat & (errors == 0)
    fields['player_hp'] = _scatter_side(fields['player_hp'], def_side, combat & ~vs_player & (overflow > 0), ply.hp)
    fields['player_shield'] = _scatter_side(
        fields['player_shield'], def_side, combat & ~vs_player & (overflow > 0), ply.shield)
    down_def = combat & ~vs_player & hit.lethal
    down_atk = combat & riposte.lethal
    fields = _knockdown(fields, def_side, def_seat, down_def)
    fields = _knockdown(fields, atk_side, atk_seat, down_atk)
    finished = torch.zeros_like(combat)
    winners = torch.full_like(atk_side, WINNER_ONGOING, dtype=torch.int64)
    fields, finished, winners, over = _victory(fields, finished, winners, combat)
    alive = (fields['char_hp'] > 0) & (fields['char_down_turns'] == 0) & fields['char_present']
    attacker_alive = _gather_char(alive, atk_side, atk_seat)
    gain = combat & ~finished & attacker_alive
    can = masks.can_gain_energy
    if can is None:
        can = alive
    else:
        can = _require(can, 'can_gain_energy', alive, torch.bool)
    share = gain[:, None, None] & _hot(atk_side, 2, device)[:, :, None] & can & alive
    ones = torch.ones_like(fields['char_energy'])
    actual_energy_cap, harm_cap = resource_caps(
        fields['turn'], fields['escalation_enabled'], fields['char_energy_cap'])
    energy = grant_energy(fields['char_energy'], ones, actual_energy_cap, share, share)
    extra = gain[:, None, None] & _hot(atk_side, 2, device)[:, :, None] & _hot(atk_seat, seats, device)[:, None, :]
    harmony_extra = extra & alive
    extra = extra & can & alive
    energy2 = grant_energy(energy.value, ones, actual_energy_cap, extra, extra)
    harm_cap = harm_cap.reshape(rows, 1, 1).expand_as(fields['char_harmony'])
    harmony = grant_harmony(fields['char_harmony'], ones, harm_cap, harmony_extra, harmony_extra)
    errors = errors | _row_or(energy.errors) | _row_or(energy2.errors) | _row_or(harmony.errors)
    gain = gain & (errors == 0)
    fields['char_energy'] = torch.where(_broadcast(gain, energy2.value), energy2.value, fields['char_energy'])
    fields['char_harmony'] = torch.where(_broadcast(gain, harmony.value), harmony.value, fields['char_harmony'])
    pending_p, pending_k, pending_s, pending_a = _pending(rows, device)
    actor = atk_side * seats + atk_seat
    pending_p, pending_k, pending_s, pending_a = _write_pending(
        pending_p, pending_k, pending_s, pending_a, 0, combat, torch.full_like(amount, EVENT_DAMAGE), atk_side, actor)
    pending_p, pending_k, pending_s, pending_a = _write_pending(
        pending_p, pending_k, pending_s, pending_a, 1, down_def | down_atk,
        torch.full_like(amount, EVENT_KNOCKDOWN), torch.where(down_def, def_side, atk_side),
        torch.where(down_def, def_side * seats + def_seat, actor))
    pending_p, pending_k, pending_s, pending_a = _write_pending(
        pending_p, pending_k, pending_s, pending_a, 2, gain,
        torch.full_like(amount, EVENT_RESOURCE), atk_side, actor)
    pending_p, pending_k, pending_s, pending_a = _write_pending(
        pending_p, pending_k, pending_s, pending_a, 3, over,
        torch.full_like(amount, EVENT_FINISH), atk_side, actor)
    return fields, errors, pending_p, pending_k, pending_s, pending_a, finished, winners


def _end_turn_clock(fields, masks, work, errors):
    import torch
    blockers = _require(masks.clock_blockers, 'clock_blockers', work, torch.bool)
    nxt = 1 - fields['active_side']
    occupied = _gather_side(fields['front'], nxt) >= 0
    errors = errors | (work & (blockers | occupied)).int() * UNSUPPORTED_ACTION
    next_count = _gather_side(fields['turn_count'], nxt)
    errors |= (work & ((fields['turn'] == 2147483647) |
                       (next_count == 2147483647))).int() * INTEGER_OVERFLOW
    clock = work & (errors == 0)
    current = fields['active_side']
    fields = dict(fields)
    fields['turn'] = torch.where(clock, fields['turn'] + 1, fields['turn'])
    fields['turn_count'] = _scatter_side(
        fields['turn_count'], nxt, clock,
        fields['turn_count'][torch.arange(work.shape[0], device=work.device), nxt.long().clamp(0, 1)] + 1)
    fields['active_side'] = torch.where(clock, nxt, current)
    fields['player_shield'] = _scatter_side(fields['player_shield'], nxt, clock, torch.zeros_like(work, dtype=torch.int32))
    side_sel = clock[:, None, None] & _hot(nxt, 2, work.device)[:, :, None]
    fields['char_shield'] = torch.where(side_sel, torch.zeros_like(fields['char_shield']), fields['char_shield'])
    down = fields['char_down_turns']
    dec = side_sel & (down > 0)
    fields['char_down_turns'] = torch.where(dec, down - 1, down)
    revive = side_sel & (down > 0) & (fields['char_down_turns'] == 0)
    fields['char_hp'] = torch.where(revive, fields['char_max_hp'], fields['char_hp'])
    turn_end = torch.full_like(current, EXPIRY_TURN_END)
    turn_start = torch.full_like(current, EXPIRY_TURN_START)
    fields, expired_end, err_end = expire_and_compact(fields, current, turn_end, clock)
    fields, expired_start, err_start = expire_and_compact(fields, nxt, turn_start, clock)
    errors = errors | err_end | err_start
    clock = clock & (errors == 0)
    energy_cap, harmony_cap = resource_caps(fields['turn'], fields['escalation_enabled'], fields['char_energy_cap'])
    clamped = grant_energy(fields['char_energy'], torch.zeros_like(fields['char_energy']),
                           energy_cap, torch.ones_like(clock)[:, None, None].expand_as(fields['char_energy']),
                           clock[:, None, None].expand_as(fields['char_energy']))
    errors = errors | _row_or(clamped.errors)
    fields['char_energy'] = torch.where(clock[:, None, None], clamped.value, fields['char_energy'])
    harm = grant_harmony(fields['char_harmony'], torch.zeros_like(fields['char_harmony']),
                         harmony_cap.reshape(work.shape[0], 1, 1).expand_as(fields['char_harmony']),
                         torch.ones_like(fields['char_harmony'], dtype=torch.bool),
                         clock[:, None, None].expand_as(fields['char_harmony']))
    errors = errors | _row_or(harm.errors)
    fields['char_harmony'] = torch.where(clock[:, None, None], harm.value, fields['char_harmony'])
    ap = torch.where(fields['turn'] == 1, 1, torch.where(
        fields['escalation_enabled'] & (fields['turn'] >= 20), 3, 2)).int()
    fields['player_ap'] = _scatter_side(fields['player_ap'], nxt, clock, ap)
    fields['normal_attack_available'] = _scatter_side(
        fields['normal_attack_available'], nxt, clock, torch.ones_like(clock))
    heat = clock & fields['escalation_enabled'] & (fields['turn'] >= 6)
    can = masks.can_gain_energy
    if can is None:
        errors = errors | heat.int() * UNSUPPORTED_ACTION
        heat = torch.zeros_like(heat)
        can = fields['char_present']
    else:
        can = _require(can, 'can_gain_energy', fields['char_present'], torch.bool)
    share = heat[:, None, None] & _hot(nxt, 2, work.device)[:, :, None] & can
    gained = grant_energy(fields['char_energy'], torch.ones_like(fields['char_energy']),
                          energy_cap, share, share)
    errors = errors | _row_or(gained.errors)
    fields['char_energy'] = torch.where(heat[:, None, None], gained.value, fields['char_energy'])
    pending_p, pending_k, pending_s, pending_a = _pending(work.shape[0], work.device)
    expired = expired_end.any(-1) | expired_start.any(-1)
    pending_p, pending_k, pending_s, pending_a = _write_pending(
        pending_p, pending_k, pending_s, pending_a, 0, clock & expired,
        torch.full_like(current, EVENT_EXPIRE), nxt, nxt)
    return fields, errors, pending_p, pending_k, pending_s, pending_a


def _concede(fields, actors, work):
    import torch
    winner = 1 - actors
    fields = dict(fields)
    fields['phase'] = torch.where(work, PHASE_FINISHED, fields['phase'])
    pending_p, pending_k, pending_s, pending_a = _pending(work.shape[0], work.device)
    pending_p, pending_k, pending_s, pending_a = _write_pending(
        pending_p, pending_k, pending_s, pending_a, 0, work,
        torch.full_like(fields['phase'], EVENT_FINISH), actors.int(), actors.int())
    return fields, work, winner, pending_p, pending_k, pending_s, pending_a


def _unknown_hooks(hooks, active):
    import torch
    if hooks is None:
        return torch.zeros_like(active, dtype=torch.int32)
    if (not isinstance(hooks, torch.Tensor) or hooks.dtype != torch.int32 or hooks.ndim != 2
            or hooks.shape[0] != active.shape[0] or hooks.device != active.device):
        raise ValueError('Prepared int32 hook ids [B,H] required')
    return (active & (hooks != 0).any(-1)).int() * UNKNOWN_HOOK


def transition(state, actions, masks, active, *, slots=None, specs=None, schema=None):
    """Select diagnostic rule stages; this is not full apply_action semantics.

    Prepared combat exchanges do not perform entering/harmony/responses;
    clock-only end-turn does not draw or run character effects. No production
    game/search factory may use this module as a complete action backend.
    """
    import torch
    if not isinstance(state, DomainSoAState):
        raise TypeError('DomainSoAState required')
    if not isinstance(masks, PreparedDecisionMasks):
        raise TypeError('PreparedDecisionMasks required')
    fields = bind_columns(dict(state.fields), specs=specs, schema=schema)
    rows, device = active.shape[0], active.device
    if (not isinstance(active, torch.Tensor) or active.dtype != torch.bool or active.ndim != 1):
        raise ValueError('Resident bool activity mask required')
    if (not isinstance(actions, torch.Tensor) or actions.dtype != torch.int64
            or tuple(actions.shape) != (rows,) or actions.device != device):
        raise ValueError('Resident int64 action ids required')
    errors = state.errors.clone()
    metadata_bad = (((state.actors < 0) | (state.actors > 1)) & ~state.finished
                    | ((state.winners < -1) | (state.winners > 1)) & state.finished
                    | (state.winners != WINNER_ONGOING) & ~state.finished)
    physical = (actions == ACTION_ATTACK) | (actions == ACTION_END_TURN)
    metadata_bad |= physical & ~state.finished & ((fields['phase'] != PHASE_PLAYING) |
                                                 (state.actors != fields['active_side']))
    errors = errors | (active & metadata_bad).int() * INVALID_ROW
    if slots is not None:
        if masks.legal is None or masks.legal_types is None:
            raise ValueError('Prepared legal mask and types required for slot selection')
        selected, select_errors = select_legal_action(masks.legal, masks.legal_types, slots, active)
        errors = errors | select_errors
        actions = torch.where(select_errors == 0, selected, actions)
    errors = errors | _unknown_hooks(masks.hooks, active & ~state.finished)
    original = _clone_fields(fields)
    orig_finished, orig_winners, orig_actors = state.finished.clone(), state.winners.clone(), state.actors.clone()
    work = active & (errors == 0) & ~state.finished
    supported = torch.zeros_like(active)
    for action_id in SUPPORTED_ACTIONS:
        supported = supported | (actions == action_id)
    errors = errors | (work & ~supported).int() * UNSUPPORTED_ACTION
    work = active & (errors == 0) & ~state.finished
    fields = _clone_fields(fields)
    finished, winners = state.finished.clone(), state.winners.clone()
    pending_p, pending_k, pending_s, pending_a = _pending(rows, device)

    concede = work & (actions == ACTION_CONCEDE)
    fields, con_fin, con_win, p, k, s, a = _concede(fields, state.actors, concede)
    finished, winners = finished | con_fin, torch.where(con_fin, con_win, winners)
    pending_p, pending_k = pending_p | p, torch.where(p, k, pending_k)
    pending_s, pending_a = torch.where(p, s, pending_s), torch.where(p, a, pending_a)

    attack = work & (actions == ACTION_ATTACK)
    fields, errors, p, k, s, a, atk_fin, atk_win = _ordinary_combat(fields, masks, attack, errors)
    finished, winners = finished | atk_fin, torch.where(atk_fin, atk_win, winners)
    pending_p, pending_k = pending_p | p, torch.where(p, k, pending_k)
    pending_s, pending_a = torch.where(p, s, pending_s), torch.where(p, a, pending_a)

    end_turn = work & (actions == ACTION_END_TURN)
    fields, errors, p, k, s, a = _end_turn_clock(fields, masks, end_turn, errors)
    pending_p, pending_k = pending_p | p, torch.where(p, k, pending_k)
    pending_s, pending_a = torch.where(p, s, pending_s), torch.where(p, a, pending_a)

    extra_kind = _require(masks.enqueue_kind, 'enqueue_kind', active, torch.int32)
    extra_side = _require(masks.enqueue_side, 'enqueue_side', active, torch.int32)
    extra_actor = _require(masks.enqueue_actor, 'enqueue_actor', active, torch.int32)
    fields, queue_errors = _flush_pending(fields, pending_p, pending_k, pending_s, pending_a, work, KNOWN_EVENTS)
    errors = errors | queue_errors
    pending_p, pending_k, pending_s, pending_a = _pending(rows, device)
    pending_p, pending_k, pending_s, pending_a = _write_pending(
        pending_p, pending_k, pending_s, pending_a, 0,
        work & (extra_kind != 0), extra_kind, extra_side, extra_actor)
    fields, queue_errors = _flush_pending(fields, pending_p, pending_k, pending_s, pending_a, work, KNOWN_EVENTS)
    errors = errors | queue_errors

    cont_kind = _require(masks.continuation_kind, 'continuation_kind', active, torch.int32)
    cont_side = _require(masks.continuation_side, 'continuation_side', active, torch.int32)
    cont_actor = _require(masks.continuation_actor, 'continuation_actor', active, torch.int32)
    fields['cont_present'], fields['cont_kind'], fields['cont_side'], fields['cont_actor'], cont_err = enqueue_queue(
        fields['cont_present'], fields['cont_kind'], fields['cont_side'], fields['cont_actor'],
        cont_kind, cont_side, cont_actor, work, KNOWN_CONTINUATIONS)
    errors = errors | cont_err

    accepted = active & (errors == 0)
    fields = _write_aliases(fields)
    merged = {name: torch.where(_broadcast(accepted, value), fields[name], original[name])
              if name in fields else original[name]
              for name, value in original.items()}
    merged = _write_aliases(merged)
    finished = torch.where(accepted, finished, orig_finished)
    winners = torch.where(accepted, winners, orig_winners)
    actors = torch.where(accepted & ~finished, merged['active_side'].long(), orig_actors)
    actors = torch.where(finished, orig_actors, actors)
    return DomainSoAState(MappingProxyType(merged), actors, finished, winners, errors)


class DomainSoARecurrentFn:
    """True-terminal, fixed-viewer recurrent step over the domain SoA transition.

    Policy uses the current actor; value uses the supplied fixed viewers.
    Absorbing terminals expose a dummy slot-0 action for tree arithmetic only.
    """
    supports_full_duel_rules = False

    def __init__(self, evaluate, *, max_actions, device='cuda', allow_cpu_reference=False,
                 specs=None, schema=None):
        import torch
        if not callable(evaluate):
            raise TypeError('Batched evaluation required')
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError('Positive complete legal action capacity required')
        self.t = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type != 'cuda' and not (self.device.type == 'cpu' and allow_cpu_reference):
            raise ValueError('CUDA or explicit CPU array reference required')
        self.evaluate, self.max_actions = evaluate, max_actions
        self.specs, self.schema = specs, schema

    def _tensor(self, value, dtype, shape):
        if (not isinstance(value, self.t.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != tuple(shape)):
            raise ValueError('Exact resident recurrent tensor shape/dtype/device required')

    def __call__(self, state, actions, viewers, active, masks):
        t = self.t
        if not isinstance(state, DomainSoAState):
            raise TypeError('DomainSoAState required')
        rows = state.actors.shape[0]
        for value, dtype in ((actions, t.int64), (viewers, t.int64), (active, t.bool)):
            self._tensor(value, dtype, (rows,))
        metadata_bad = (((state.actors < 0) | (state.actors > 1)) & ~state.finished
                        | ((state.winners < -1) | (state.winners > 1)) & state.finished
                        | (state.winners != WINNER_ONGOING) & ~state.finished
                        | (viewers < 0) | (viewers > 1))
        errors = state.errors | (metadata_bad & active).to(t.int32) * INVALID_ROW
        owned = DomainSoAState(
            MappingProxyType(_clone_fields(state.fields)), state.actors.clone(),
            state.finished.clone(), state.winners.clone(), errors.clone())
        advanced = transition(owned, actions, masks, active & (errors == 0) & ~state.finished,
                              specs=self.specs, schema=self.schema)
        errors = errors | t.where(active & ~state.finished, advanced.errors, 0)
        bad_next = (((advanced.actors < 0) | (advanced.actors > 1)) & ~advanced.finished
                    | ((advanced.winners < -1) | (advanced.winners > 1)) & advanced.finished
                    | (advanced.winners != WINNER_ONGOING) & ~advanced.finished)
        errors = errors | ((active & ~state.finished & bad_next).to(t.int32) * INVALID_ROW)
        accepted = active & (errors == 0)
        merged = {name: t.where(_broadcast(accepted, value), advanced.fields[name], state.fields[name])
                  for name, value in state.fields.items()}
        actors = t.where(accepted, advanced.actors, state.actors)
        finished = t.where(accepted, advanced.finished, state.finished)
        winners = t.where(accepted, advanced.winners, state.winners)
        next_state = DomainSoAState(MappingProxyType(merged), actors, finished, winners, errors)
        evaluate_rows = active & (errors == 0) & ~finished
        prediction = self.evaluate(next_state, viewers, evaluate_rows)
        if not isinstance(prediction, DomainSoAEvaluation):
            raise TypeError('DomainSoAEvaluation required')
        self._tensor(prediction.logits, t.float32, (rows, self.max_actions))
        self._tensor(prediction.valid, t.bool, (rows, self.max_actions))
        self._tensor(prediction.wdl, t.float32, (rows, 3))
        self._tensor(prediction.errors, t.int32, (rows,))
        bad_eval = (~t.isfinite(prediction.wdl).all(-1)
                    | ((prediction.wdl < 0) | (prediction.wdl > 1)).any(-1)
                    | ((prediction.wdl.sum(-1) - 1.).abs() > 1e-5)
                    | ~prediction.valid.any(-1)
                    | (prediction.valid & ~t.isfinite(prediction.logits)).any(-1)
                    | (prediction.errors != 0))
        errors = errors | ((evaluate_rows & bad_eval).to(t.int32) * EVALUATION_ERROR)
        good = active & (errors == 0)
        merged = {name: t.where(_broadcast(good, value), merged[name], state.fields[name])
                  for name, value in merged.items()}
        next_state = DomainSoAState(
            MappingProxyType(merged), t.where(good, actors, state.actors),
            t.where(good, finished, state.finished), t.where(good, winners, state.winners), errors)
        reached = good & active & ~state.finished & finished
        reward = t.where(reached, t.where(winners == viewers, 1.,
                         t.where(winners == WINNER_DRAW, 0., -1.)), 0.).to(t.float32)
        live = good & ~finished
        value = t.where(live, prediction.wdl[:, 2] - prediction.wdl[:, 0], 0.)
        valid = prediction.valid & live[:, None]
        dummy = t.arange(self.max_actions, device=self.device)[None, :] == 0
        valid = valid | (good & finished)[:, None] & dummy
        logits = t.where(live[:, None], prediction.logits, 0.).masked_fill(~valid, -float('inf'))
        return DomainSoARecurrentOutput(next_state, reward, live.to(t.float32), logits, valid, value, errors)
