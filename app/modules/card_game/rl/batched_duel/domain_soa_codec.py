"""Host ingress/egress domain SoA codec for public-17 mulligan and playing state.

This is not the graph codec in codec.py, not a VM/native_soa runtime, and not a
full-rule engine. Catalog templates supply static nests; mutable numbers live in
dedicated schema columns. Unknown keys and schema-missing authoritative fields
are rejected explicitly rather than dropped.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

import numpy as np

from app.modules.card_game.content.duel_v2.tutorial import lookup_card, lookup_character
from app.modules.card_game.engine.duel_v2.continuous import DOTS
from app.modules.card_game.engine.duel_v2.entities import CardEntity, CharacterEntity, PlayerEntity
from app.modules.card_game.engine.duel_v2.modifiers import PUBLIC_EFFECT_FIELDS
from app.modules.card_game.engine.duel_v2.state import new_character

from .domain_soa_schema import (
    ARRAY_NAMES,
    CARD_BOOL_FIELDS,
    CARD_INT_FIELDS,
    CARD_KEYS,
    CARD_STR_FIELDS,
    CARD_VOCAB,
    CHARACTER_BOOL_FIELDS,
    CHARACTER_INT_FIELDS,
    CHARACTER_KEYS,
    CHARACTER_RUNTIME_STR_FIELDS,
    CHARACTER_STR_FIELDS,
    CHARACTER_VOCAB,
    CHOICE_BOOL_KEYS,
    CHOICE_INT_KEYS,
    CHOICE_KINDS,
    CHOICE_STR_KEYS,
    COVERAGE,
    DOT_KINDS,
    DomainSoABatch,
    DomainSoACapacityError,
    DomainSoADecodeError,
    DomainSoALayout,
    DomainSoAUnsupportedError,
    EFFECT_CLEAR_EVENTS,
    EFFECT_CONDITION_IDS,
    EFFECT_CONDITION_ARGS,
    EFFECT_KINDS,
    EFFECT_OPS,
    EFFECT_PHASES,
    EFFECT_SCOPES,
    EFFECT_VISIBILITY,
    FLAG_BOOL_FIELDS,
    FLAG_INT_FIELDS,
    FLAG_STR_FIELDS,
    GAME_KEYS,
    GAME_PHASES,
    LINGKE_CHAIN_KEYS,
    NON_AUTHORITATIVE_GAME_KEYS,
    OP_FLAG_KEYS,
    OP_INT_KEYS,
    OP_STR_KEYS,
    PLAYER_BOOL_FIELDS,
    PLAYER_INT_FIELDS,
    PLAYER_KEYS,
    PLAYER_STR_FIELDS,
    PRESENCE_ABSENT,
    PRESENCE_CATALOG,
    PRESENCE_NULL,
    PRESENCE_VALUE,
    SIDES,
    UNSUPPORTED_GAME_KEYS,
    WINNERS,
    ZONE_STATUS_KEYS,
    ZONES,
    allocate_arrays,
    catalog_card,
    catalog_fingerprint,
    side_index,
    vocab_get,
)

supports_full_duel_rules = False
INT32_MIN, INT32_MAX = -2147483648, 2147483647
RNG_MAX = (1 << 63) - 1
_CHAR_FROM_VOCAB = {index: cid for cid, index in CHARACTER_VOCAB.items()}
_CARD_FROM_VOCAB = {index: card_id for card_id, index in CARD_VOCAB.items()}
_MISSING = object()
VALUE_KIND_INT, VALUE_KIND_FLOAT, VALUE_KIND_BOOL = 0, 1, 2
FLAG_BOOL_INTS = frozenset({"energy_next", "allied_hurt", "share_overflow", "pact"})
PLAYER_BOOL_INTS = frozenset({"canhong_collapse_used"})
FLAG_SPECIAL = frozenset({"collapse", "atk_buff_expiring", "timed_attack"})
CHOICE_ALLOWED = (
    frozenset({"side", "kind", "prompt", "cards", "actor", "continuation"})
    | frozenset(CHOICE_INT_KEYS)
    | frozenset(CHOICE_BOOL_KEYS)
    | frozenset(CHOICE_STR_KEYS)
)
OP_ALLOWED = (
    frozenset(
        {
            "side",
            "actor",
            "card",
            "target_id",
            "participants",
            "harmony",
            "first_bonus",
            "redeem_bonus",
            "record_candidate",
            "card_mark",
            "option_id",
            "last_attack_target",
            "choice_cards",
        }
    )
    | frozenset(OP_FLAG_KEYS)
    | frozenset(OP_INT_KEYS)
    | frozenset(OP_STR_KEYS)
)
DEFERRED_ALLOWED = frozenset({"side", "actor", "entity_id", "kind"})
MISSING_SCHEMA_FIELDS = (
    "Player.records payloads",
    "pending_sortie specs beyond optional integer bonus",
    "PendingChoice.choices card list",
    "Effect.clear_on values outside EFFECT_CLEAR_EVENTS",
    "_deferred_card_marks operation list",
)
UNSUPPORTED_REMAINING = (
    "unknown keys at any allowlisted object",
    "presentation events/_public_board unless presentation='drop'",
    "tutorial/scenario/turn_snapshots/time_removed",
    "non-public unsommoned character templates",
    "mutated catalog nests (mechanisms/play_options/source/ultimate)",
    "records with payloads",
    "Python callables, numpy/torch payloads, bool used as int except declared bool-int flags",
    *MISSING_SCHEMA_FIELDS,
)


def coverage():
    return {
        "supports_full_duel_rules": False,
        "complete_soa": False,
        "full_public17": False,
        "initial_mulligan": True,
        "normal_playing": True,
        "seven_official_presets": COVERAGE["seven_official_presets"],
        "phases": GAME_PHASES,
        "authoritative_encoded": (
            "game scalars including rng/version/phase/sides/turn/winner/reason",
            "two PlayerEntity rows with vitals, used, harmonized, front/last_front",
            "CharacterEntity rows including summoned ids interned when not public-17",
            "CardEntity table plus ordered hand/deck/discard/removed",
            "entity class, entity_id, instance_id, aliases across zones/choice/operation",
            "typed effects with expiry clocks and known clear_on events",
            "typed flags including has_harmonized/gulang_item/return_after/temp_revive/free_normal_attack_turn",
            "last_hit_source, lingke_chain, beast_fangs, flavor, disappeared, support_turn, effect_seq",
            "pending_choice including deck_cards/deck_order/snapshot_n/continuation strings",
            "deferred_actions empty-list vs absent via presence, response locks, down-response queue",
            "front_debuff delay/burn/star/weave/stain/dots",
            "equipment, shape, copy, jingu_mark",
            "absent vs null vs zero via presence codes",
        ),
        "missing_schema_fields": MISSING_SCHEMA_FIELDS,
        "unsupported_rejected": UNSUPPORTED_REMAINING,
        "non_authoritative_dropped_or_rejected": (
            "events presentation log and patches",
            "_public_board capture cache",
        ),
    }


def encode_domain_states(states, layout=None, presentation="reject"):
    if presentation not in ("reject", "drop"):
        raise ValueError('presentation must be "reject" or "drop"')
    if layout is None:
        layout = DomainSoALayout()
    if not isinstance(layout, DomainSoALayout):
        raise TypeError("layout must be a DomainSoALayout")
    rows = _as_state_list(states)
    arrays = allocate_arrays(layout, len(rows))
    fingerprint = np.frombuffer(catalog_fingerprint(), dtype=np.uint8)
    for row, state in enumerate(rows):
        _encode_row(state, arrays, row, layout, presentation)
        arrays["row_valid"][row] = True
        arrays["catalog_digest"][row] = fingerprint
    batch = DomainSoABatch(layout=layout, **arrays)
    for row, state in enumerate(rows):
        expected = {key: value for key, value in state.items() if key not in NON_AUTHORITATIVE_GAME_KEYS}
        restored = _decode_row(batch, row)
        if not _same(expected, restored):
            raise DomainSoAUnsupportedError("codec cannot represent every supplied field/type")
        _same_aliases(expected, restored)
    return batch


def decode_domain_states(batch):
    if not isinstance(batch, DomainSoABatch):
        raise TypeError("batch must be a DomainSoABatch")
    games = []
    fingerprint = np.frombuffer(catalog_fingerprint(), dtype=np.uint8)
    for row in range(batch.n_rows):
        if not bool(batch.row_valid[row]):
            raise DomainSoADecodeError(f"row {row} is not valid")
        if batch.row_error[row] or not np.array_equal(batch.catalog_digest[row], fingerprint):
            raise DomainSoADecodeError("Failed row or catalog binding mismatch")
        if not 0 <= batch.string_count[row] <= batch.layout.max_string_slots:
            raise DomainSoADecodeError("String count outside layout")
        if not 0 <= batch.string_bytes_used[row] <= batch.layout.max_string_bytes:
            raise DomainSoADecodeError("String bytes outside layout")
        if not 0 <= batch.mulligan_count[row] <= 2:
            raise DomainSoADecodeError("Mulligan side count outside layout")
        state = _decode_row(batch, row)
        canonical = encode_domain_states([state], batch.layout, presentation="drop")
        if any(
            not np.array_equal(getattr(batch, name)[row], getattr(canonical, name)[0])
            for name in ARRAY_NAMES
        ):
            raise DomainSoADecodeError("Noncanonical or unsupported packed state columns")
        games.append(state)
    return games


def _as_state_list(states):
    if isinstance(states, (str, bytes, bytearray, Mapping)):
        raise TypeError("states must be a sequence of game dicts")
    if not isinstance(states, Sequence):
        raise TypeError("states must be a sequence of game dicts")
    rows = list(states)
    for state in rows:
        if not isinstance(state, Mapping):
            raise TypeError(f"game state must be a dict, got {type(state).__name__}")
    return rows


class _Strings:
    def __init__(self, layout):
        self.max_slots = layout.max_string_slots
        self.max_bytes = layout.max_string_bytes
        self.blobs = []
        self.index = {}
        self.used = 0

    def intern(self, text):
        if type(text) is not str:
            raise TypeError(f"string field must be str, got {type(text).__name__}")
        found = self.index.get(text)
        if found is not None:
            return found
        blob = text.encode("utf-8")
        if len(self.blobs) >= self.max_slots:
            raise DomainSoACapacityError("string table slot capacity exceeded")
        if self.used + len(blob) > self.max_bytes:
            raise DomainSoACapacityError("string table byte capacity exceeded")
        slot = len(self.blobs)
        self.blobs.append(blob)
        self.index[text] = slot
        self.used += len(blob)
        return slot

    def dump(self, arrays, row):
        arrays["string_count"][row] = len(self.blobs)
        arrays["string_bytes_used"][row] = self.used
        offset = 0
        for i, blob in enumerate(self.blobs):
            arrays["string_offset"][row, i] = offset
            arrays["string_length"][row, i] = len(blob)
            if blob:
                arrays["string_bytes"][row, offset : offset + len(blob)] = np.frombuffer(
                    blob, dtype=np.uint8
                )
            offset += len(blob)


class _RejectStrings:
    def intern(self, text):
        raise DomainSoAUnsupportedError("non-catalog string is outside SoA coverage")


def _load_string(batch, row, slot):
    slot = int(slot)
    count = int(batch.string_count[row])
    if slot < 0 or slot >= count:
        raise DomainSoADecodeError(f"string slot {slot} out of range")
    offset = int(batch.string_offset[row, slot])
    length = int(batch.string_length[row, slot])
    used = int(batch.string_bytes_used[row])
    if offset < 0 or length < 0 or offset + length > used:
        raise DomainSoADecodeError("string bytes slice is corrupt")
    return bytes(batch.string_bytes[row, offset : offset + length]).decode("utf-8")


def _as_int(name, value, lo=INT32_MIN, hi=INT32_MAX):
    if type(value) is not int:
        raise TypeError(f"{name} must be int, got {type(value).__name__} (bool is rejected)")
    if value < lo or value > hi:
        raise DomainSoAUnsupportedError(f"{name} out of range [{lo}, {hi}]: {value}")
    return value


def _as_bool(name, value):
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool, got {type(value).__name__}")
    return value


def _same(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _same_aliases(left, right):
    forward, backward = {}, {}

    def walk(a, b):
        if isinstance(a, (dict, list)):
            if forward.setdefault(id(a), id(b)) != id(b) or backward.setdefault(id(b), id(a)) != id(a):
                raise DomainSoAUnsupportedError("codec cannot preserve these mutable aliases")
        if isinstance(a, dict):
            for key in a:
                walk(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            for aa, bb in zip(a, b):
                walk(aa, bb)

    walk(left, right)


def _unknown(obj, allowed, what):
    extra = [key for key in obj if key not in allowed]
    if extra:
        raise DomainSoAUnsupportedError(f"unknown {what} keys {extra}")


def _forbid(container, keys, what):
    present = [key for key in keys if key in container]
    if present:
        raise DomainSoAUnsupportedError(f"unsupported {what} fields {present}")


def _empty_map(value, what):
    if type(value) is not dict:
        raise TypeError(f"{what} must be a dict")
    if value:
        raise DomainSoAUnsupportedError(f"nonempty {what} is outside SoA coverage")


def _empty_list(value, what):
    if type(value) is not list:
        raise TypeError(f"{what} must be a list")
    if value:
        raise DomainSoAUnsupportedError(f"nonempty {what} is outside SoA coverage")


def _presence_of_empty(container, key, what):
    if key not in container:
        return PRESENCE_ABSENT
    value = container[key]
    if value is None:
        return PRESENCE_NULL
    if type(value) is dict:
        _empty_map(value, what)
        return PRESENCE_VALUE
    if type(value) is list:
        _empty_list(value, what)
        return PRESENCE_VALUE
    raise DomainSoAUnsupportedError(f"unsupported {what} type {type(value).__name__}")


def _parse_entity_num(value):
    if type(value) is not str or not value.startswith("e"):
        raise DomainSoAUnsupportedError(f"unsupported entity_id {value!r}")
    body = value[1:]
    if not body.isdigit() or body != str(int(body)):
        raise DomainSoAUnsupportedError(f"unsupported entity_id {value!r}")
    number = int(body)
    if number < 1:
        raise DomainSoAUnsupportedError(f"unsupported entity_id {value!r}")
    return _as_int("entity_id", number, 1, INT32_MAX)


def _parse_instance_num(value, side):
    prefix = side + "-"
    if type(value) is not str or not value.startswith(prefix):
        raise DomainSoAUnsupportedError(f"unsupported instance_id {value!r}")
    body = value[len(prefix) :]
    if not body.isdigit() or body != str(int(body)):
        raise DomainSoAUnsupportedError(f"unsupported instance_id {value!r}")
    return _as_int("instance_id", int(body), 1, INT32_MAX)


def _catalog_nest(obj, catalog, key):
    has_obj = key in obj
    has_cat = key in catalog
    if has_obj != has_cat:
        raise DomainSoAUnsupportedError(f"catalog nest {key} presence mismatch")
    if not has_obj:
        return PRESENCE_ABSENT
    if not _same(obj[key], catalog[key]):
        raise DomainSoAUnsupportedError(f"mutated catalog nest {key}")
    return PRESENCE_CATALOG


def _put_optional_int(container, key, presence, values, index, lo=INT32_MIN, hi=INT32_MAX, catalog=None, bool_keys=()):
    if key not in container:
        presence[index] = PRESENCE_ABSENT
        return
    value = container[key]
    if value is None:
        presence[index] = PRESENCE_NULL
        return
    if key in bool_keys:
        flag = _as_bool(key, value)
        presence[index] = PRESENCE_VALUE
        values[index] = int(flag)
        return
    if catalog is not None and key in catalog and _same(value, catalog[key]):
        presence[index] = PRESENCE_CATALOG
        return
    number = _as_int(key, value, lo, hi)
    presence[index] = PRESENCE_VALUE
    values[index] = number


def _put_optional_bool(container, key, presence, values, index, catalog=None):
    if key not in container:
        presence[index] = PRESENCE_ABSENT
        return
    value = container[key]
    if value is None:
        presence[index] = PRESENCE_NULL
        return
    flag = _as_bool(key, value)
    if catalog is not None and key in catalog and type(catalog[key]) is bool and catalog[key] is flag:
        presence[index] = PRESENCE_CATALOG
        return
    presence[index] = PRESENCE_VALUE
    values[index] = flag


def _put_optional_str(container, key, presence, slots, index, strings, catalog=None, allow_value=False):
    if key not in container:
        presence[index] = PRESENCE_ABSENT
        return
    value = container[key]
    if value is None:
        presence[index] = PRESENCE_NULL
        return
    if type(value) is not str:
        raise TypeError(f"{key} must be str, got {type(value).__name__}")
    if catalog is not None and key in catalog and _same(value, catalog[key]):
        presence[index] = PRESENCE_CATALOG
        return
    if not allow_value:
        raise DomainSoAUnsupportedError(f"non-catalog string {key} is outside SoA coverage")
    presence[index] = PRESENCE_VALUE
    slots[index] = strings.intern(value)


def _take_optional(presence, values, index, caster, catalog=None, key=None, bool_keys=()):
    code = int(presence[index])
    if code == PRESENCE_ABSENT:
        return _MISSING
    if code == PRESENCE_NULL:
        return None
    if code == PRESENCE_CATALOG:
        if catalog is None or key not in catalog:
            raise DomainSoADecodeError(f"catalog presence without template field {key}")
        return catalog[key]
    if code != PRESENCE_VALUE:
        raise DomainSoADecodeError(f"invalid presence {code}")
    if key in bool_keys:
        return bool(int(values[index]))
    return caster(values[index])


def _apply_optional(target, key, value):
    if value is _MISSING:
        target.pop(key, None)
    else:
        target[key] = value


def _enum_index(value, options, what):
    if value not in options:
        raise DomainSoAUnsupportedError(f"unsupported {what} {value!r}")
    return options.index(value)


def _enum_from(index, options, what):
    index = int(index)
    if index < 0 or index >= len(options):
        raise DomainSoADecodeError(f"invalid {what} index {index}")
    return options[index]


def _side_from(value):
    return _enum_from(value, SIDES, "side")


def _char_slot(order, cid, what="character"):
    if type(cid) is not str:
        raise DomainSoAUnsupportedError(f"{what} must be a character id string")
    try:
        return order.index(cid)
    except ValueError as exc:
        raise DomainSoAUnsupportedError(f"{what} {cid!r} is not in order") from exc


def _entity_text(number):
    return "e" + str(int(number))

def _encode_row(state, arrays, row, layout, presentation):
    _unknown(state, GAME_KEYS, "game")
    _forbid(state, UNSUPPORTED_GAME_KEYS, "unsupported")
    _forbid(state, ("_deferred_card_marks",), "deferred-mark")
    for key in NON_AUTHORITATIVE_GAME_KEYS:
        if key in state and presentation == "reject":
            raise DomainSoAUnsupportedError(
                f"presentation key {key!r} requires presentation='drop'"
            )
    phase = state["phase"]
    if phase not in GAME_PHASES:
        raise DomainSoAUnsupportedError(f"unsupported phase {phase!r}")
    strings = _Strings(layout)
    arrays["version"][row] = _as_int("version", state["version"])
    arrays["phase"][row] = GAME_PHASES.index(phase)
    arrays["active_side"][row] = side_index(state["active_side"])
    arrays["first_side"][row] = side_index(state["first_side"])
    arrays["turn"][row] = _as_int("turn", state["turn"])
    _encode_winner(state, arrays, row)
    _encode_reason(state, arrays, row, strings)
    arrays["rng"][row] = _as_int("rng", state["rng"], 0, RNG_MAX)
    arrays["next_instance"][row] = _as_int("next_instance", state["next_instance"], 0, INT32_MAX)
    arrays["next_entity"][row] = _as_int("next_entity", state["next_entity"], 0, INT32_MAX)
    arrays["event_seq"][row] = _as_int("event_seq", state["event_seq"])
    arrays["current_action_id_slot"][row] = strings.intern(state["current_action_id"])
    arrays["current_action_version"][row] = _as_int(
        "current_action_version", state["current_action_version"]
    )
    arrays["combat_group"][row] = _as_int("combat_group", state["combat_group"])
    arrays["escalation_enabled"][row] = _as_bool("escalation_enabled", state["escalation_enabled"])
    arrays["harmony_residue_version"][row] = _as_int(
        "harmony_residue_version", state["harmony_residue_version"]
    )
    arrays["rules_version_slot"][row] = strings.intern(state["rules_version"])
    arrays["design_version_slot"][row] = strings.intern(state["design_version"])
    done = state["mulligan_done"]
    if type(done) is not list:
        raise TypeError("mulligan_done must be a list")
    if len(done) > 2:
        raise DomainSoACapacityError("mulligan_done exceeds two sides")
    arrays["mulligan_count"][row] = len(done)
    for i, item in enumerate(done):
        arrays["mulligan_sides"][row, i] = side_index(item)
    _encode_optional_side(state, "turn_start_pending", arrays, row, "turn_start_pending")
    _encode_optional_bool_field(state, "_response_lock", arrays, row, "response_lock")
    _encode_optional_bool_field(state, "_response_intercept", arrays, row, "response_intercept")
    _encode_optional_bool_field(state, "_draining_actions", arrays, row, "draining")
    sides = state["sides"]
    if not isinstance(sides, Mapping) or set(sides) != set(SIDES):
        raise DomainSoAUnsupportedError("sides must contain exactly 'a' and 'b'")
    identities = {}
    instances = {}
    card_by_obj = {}
    card_n = [0]
    zone_seen = set()

    def register_entity(entity, expected_type, *, instance=None):
        if type(entity) is not expected_type:
            raise DomainSoAUnsupportedError("codec requires hydrated formal entity classes")
        eid = entity.get("entity_id")
        previous = identities.get(eid)
        if previous is not None and previous is not entity:
            raise DomainSoAUnsupportedError("Duplicate entity identity or multiple physical zone occupancy")
        identities[eid] = entity
        if instance is not None:
            prev_i = instances.get(instance)
            if prev_i is not None and prev_i is not entity:
                raise DomainSoAUnsupportedError("Duplicate physical card instance identity")
            instances[instance] = entity

    def register_card(card, side):
        physical = card.get('_physical_card', _MISSING)
        if physical is _MISSING:
            register_entity(card, CardEntity, instance=card.get("instance_id"))
        else:
            if not isinstance(card, CardEntity) or not isinstance(physical, CardEntity):
                raise DomainSoAUnsupportedError('physical card view must reference a CardEntity')
            if physical is card or '_physical_card' in physical:
                raise DomainSoAUnsupportedError('nested or cyclic physical card view')
            if any(card.get(key) != physical.get(key) for key in ('entity_id','instance_id','side')):
                raise DomainSoAUnsupportedError('physical view identity mismatch')
        ident = id(card)
        found = card_by_obj.get(ident)
        if found is not None:
            return found
        if card_n[0] >= layout.max_cards:
            raise DomainSoACapacityError("card table exceeds layout.max_cards")
        idx = card_n[0]
        card_by_obj[ident] = idx
        card_n[0] += 1
        if physical is not _MISSING:
            arrays['card_is_view'][row, idx] = True
            arrays['card_physical_index'][row, idx] = register_card(physical, side)
        _encode_card(card, side, arrays, row, idx, strings)
        return idx

    def register_zone_card(card, side):
        if '_physical_card' in card:
            raise DomainSoAUnsupportedError('zone cannot contain an effective physical-card view')
        ident = id(card)
        if ident in zone_seen:
            raise DomainSoAUnsupportedError("Duplicate entity identity or multiple physical zone occupancy")
        zone_seen.add(ident)
        return register_card(card, side)

    for s, side in enumerate(SIDES):
        team = sides[side]
        register_entity(team, PlayerEntity)
        for hero in team["characters"].values():
            register_entity(hero, CharacterEntity)
        _encode_player(team, arrays, row, s, layout, strings, register_zone_card)
    _encode_choice(state, arrays, row, layout, strings, register_card)
    _encode_operation(state, arrays, row, layout, strings, register_card)
    _encode_deferred(state, arrays, row, layout, strings)
    _encode_down_responses(state, arrays, row, layout, strings)
    strings.dump(arrays, row)


def _encode_winner(state, arrays, row):
    if "winner" not in state:
        arrays["winner_presence"][row] = PRESENCE_ABSENT
        return
    value = state["winner"]
    if value is None:
        arrays["winner_presence"][row] = PRESENCE_NULL
        return
    arrays["winner_presence"][row] = PRESENCE_VALUE
    arrays["winner"][row] = _enum_index(value, WINNERS, "winner")


def _encode_reason(state, arrays, row, strings):
    if "reason" not in state:
        arrays["reason_presence"][row] = PRESENCE_ABSENT
        return
    value = state["reason"]
    if value is None:
        arrays["reason_presence"][row] = PRESENCE_NULL
        return
    if type(value) is not str:
        raise TypeError(f"reason must be str, got {type(value).__name__}")
    arrays["reason_presence"][row] = PRESENCE_VALUE
    arrays["reason_slot"][row] = strings.intern(value)


def _encode_optional_side(state, key, arrays, row, prefix):
    if key not in state:
        arrays[prefix + "_presence"][row] = PRESENCE_ABSENT
        return
    value = state[key]
    if value is None:
        arrays[prefix + "_presence"][row] = PRESENCE_NULL
        return
    arrays[prefix + "_presence"][row] = PRESENCE_VALUE
    arrays[prefix + "_side"][row] = side_index(value)


def _encode_optional_bool_field(state, key, arrays, row, prefix):
    if key not in state:
        arrays[prefix + "_presence"][row] = PRESENCE_ABSENT
        return
    value = state[key]
    if value is None:
        arrays[prefix + "_presence"][row] = PRESENCE_NULL
        return
    arrays[prefix + "_presence"][row] = PRESENCE_VALUE
    arrays[prefix][row] = _as_bool(key, value)


def _encode_player(team, arrays, row, s, layout, strings, register_card):
    _unknown(team, PLAYER_KEYS, "player")
    arrays["player_valid"][row, s] = True
    arrays["player_entity_slot"][row, s] = _parse_entity_num(team["entity_id"])
    if team.get("side") not in (None, SIDES[s]):
        raise DomainSoAUnsupportedError("player side does not match slot")
    arrays["player_name_slot"][row, s] = strings.intern(team["name"])
    arrays["player_hp"][row, s] = _as_int("hp", team["hp"])
    arrays["player_max_hp"][row, s] = _as_int("max_hp", team["max_hp"])
    arrays["player_shield"][row, s] = _as_int("shield", team["shield"])
    arrays["player_ap"][row, s] = _as_int("ap", team["ap"])
    declared_order = team["order"]
    characters = team["characters"]
    if type(declared_order) is not list or not isinstance(characters, Mapping):
        raise TypeError("player order/characters must be list/dict")
    order = list(characters)  # Stored identities retain formal dict iteration order.
    if len(order) > layout.max_characters:
        raise DomainSoACapacityError("character storage exceeds layout.max_characters")
    if len(set(declared_order)) != len(declared_order) or any(cid not in characters for cid in declared_order):
        raise DomainSoAUnsupportedError("lineup order must contain distinct stored character ids")
    arrays["player_character_count"][row, s] = len(order)
    arrays["player_order_count"][row, s] = len(declared_order)
    for i, cid in enumerate(declared_order):
        arrays["player_order_slot"][row, s, i] = order.index(cid)
    _encode_front_ref(team, order, arrays, row, s, "front", "player_front")
    _encode_front_ref(team, order, arrays, row, s, "last_front", "player_last_front")
    for i, key in enumerate(PLAYER_INT_FIELDS):
        _put_optional_int(
            team, key, arrays["player_int_presence"][row, s], arrays["player_int_value"][row, s], i,
            bool_keys=PLAYER_BOOL_INTS,
        )
    for i, key in enumerate(PLAYER_BOOL_FIELDS):
        _put_optional_bool(
            team, key, arrays["player_bool_presence"][row, s], arrays["player_bool_value"][row, s], i
        )
    for i, key in enumerate(PLAYER_STR_FIELDS):
        _put_optional_str(
            team, key, arrays["player_str_presence"][row, s], arrays["player_str_slot"][row, s], i,
            strings, allow_value=True,
        )
    used = team["used"]
    if type(used) is not dict:
        raise TypeError("used must be a dict")
    if len(used) > layout.max_used:
        raise DomainSoACapacityError("used exceeds layout.max_used")
    arrays["player_used_count"][row, s] = len(used)
    for i, (key, value) in enumerate(used.items()):
        arrays["player_used_key_slot"][row, s, i] = strings.intern(key)
        if type(value)is bool:
            arrays['player_used_value_kind'][row,s,i]=0
            arrays['player_used_value'][row,s,i]=value
        elif type(value)is int:
            arrays['player_used_value_kind'][row,s,i]=1
            arrays['player_used_int'][row,s,i]=_as_int('used value',value)
            arrays['player_used_value'][row,s,i]=bool(value)
        else:
            raise DomainSoAUnsupportedError('used values must be bool or int')
    harmonized = team["harmonized"]
    if type(harmonized) is not dict:
        raise TypeError("harmonized must be a dict")
    extra = [key for key in harmonized if key not in order]
    if extra:
        raise DomainSoAUnsupportedError(f"harmonized keys outside order {extra}")
    for i, cid in enumerate(order):
        if cid not in harmonized:
            continue
        if harmonized[cid] is not True:
            raise DomainSoAUnsupportedError("harmonized values must be True when present")
        arrays["player_harmonized"][row, s, i] = True
    _encode_attacked(team, order, arrays, row, s)
    records = team["records"]
    if type(records) is not list:
        raise TypeError("records must be a list")
    if records:
        raise DomainSoAUnsupportedError("records payloads have no dedicated SoA arrays")
    arrays["player_records_count"][row, s] = 0
    _encode_effects(team, arrays, row, layout.holder_index(s, None), layout, strings, "player effects")
    arrays["player_effects_presence"][row, s] = _effects_presence(team)
    _encode_flags(team, arrays, row, layout.holder_index(s, None), layout, strings)
    arrays["player_flags_presence"][row, s] = _flags_presence(team)
    if 'revealed_ids' in team:
        revealed=team['revealed_ids']
        if revealed is None:
            arrays['player_revealed_presence'][row,s]=PRESENCE_NULL
        else:
            if type(revealed) is not list:
                raise TypeError('revealed_ids must be a list')
            if len(revealed)>layout.max_cards:
                raise DomainSoACapacityError('revealed_ids exceeds max_cards')
            arrays['player_revealed_presence'][row,s]=PRESENCE_VALUE
            arrays['player_revealed_count'][row,s]=len(revealed)
            for ri,instance in enumerate(revealed):
                if type(instance)is not str or instance[:2]not in ('a-','b-'):
                    raise DomainSoAUnsupportedError('noncanonical revealed instance id')
                rs=instance[0]
                arrays['player_revealed_side'][row,s,ri]=side_index(rs)
                arrays['player_revealed_instance'][row,s,ri]=_parse_instance_num(instance,rs)
    _encode_front_debuff(team.get("front_debuff", {}), arrays, row, s, layout, strings)
    for i, cid in enumerate(order):
        _encode_character(characters[cid], cid, arrays, row, s, i, layout, strings)
    for z, zone in enumerate(ZONES):
        cards = team[zone]
        if type(cards) is not list:
            raise TypeError(f"{zone} must be a list")
        if len(cards) > layout.max_zone:
            raise DomainSoACapacityError(f"{zone} exceeds layout.max_zone")
        arrays["zone_count"][row, s, z] = len(cards)
        for pos, card in enumerate(cards):
            arrays["zone_card_index"][row, s, z, pos] = register_card(card, SIDES[s])


def _encode_front_ref(team, order, arrays, row, s, key, prefix):
    if key not in team:
        arrays[prefix + "_presence"][row, s] = PRESENCE_ABSENT
        return
    value = team[key]
    if value is None:
        arrays[prefix + "_presence"][row, s] = PRESENCE_NULL
        return
    arrays[prefix + "_presence"][row, s] = PRESENCE_VALUE
    arrays[prefix + "_slot"][row, s] = _char_slot(order, value, key)


def _encode_attacked(team, order, arrays, row, s):
    if "attacked_this_turn" not in team:
        arrays["player_attacked_this_turn_presence"][row, s] = PRESENCE_ABSENT
        return
    attacked = team["attacked_this_turn"]
    if type(attacked) is not list:
        raise TypeError("attacked_this_turn must be a list")
    eid_to_slot = {team["characters"][cid]["entity_id"]: i for i, cid in enumerate(order)}
    if any(eid not in eid_to_slot for eid in attacked):
        raise DomainSoAUnsupportedError("attacked_this_turn entity is not in order")
    if len(attacked) > len(order):
        raise DomainSoAUnsupportedError("attacked_this_turn exceeds lineup size")
    arrays["player_attacked_this_turn_presence"][row, s] = PRESENCE_VALUE
    arrays["player_attacked_count"][row, s] = len(attacked)
    for i, eid in enumerate(attacked):
        slot = eid_to_slot[eid]
        arrays["player_attacked_slot"][row, s, i] = slot
        arrays["player_attacked_this_turn"][row, s, slot] = True


def _effects_presence(holder):
    if "effects" not in holder:
        return PRESENCE_ABSENT
    value = holder["effects"]
    if value is None:
        return PRESENCE_NULL
    if type(value) is not list:
        raise TypeError("effects must be a list")
    return PRESENCE_VALUE


def _flags_presence(holder):
    if "flags" not in holder:
        return PRESENCE_ABSENT
    value = holder["flags"]
    if value is None:
        return PRESENCE_NULL
    if type(value) is not dict:
        raise TypeError("flags must be a dict")
    return PRESENCE_VALUE


def _encode_character(hero, cid, arrays, row, s, slot, layout, strings):
    _unknown(hero, CHARACTER_KEYS, "character")
    if hero.get("id", cid) != cid:
        raise DomainSoAUnsupportedError("character id does not match lineup")
    _forbid(hero, ("turn_snapshots",), "character")
    summoned = bool(hero.get("summoned"))
    catalog = {}
    if cid in CHARACTER_VOCAB:
        catalog = lookup_character(cid)
        arrays["character_template"][row, s, slot] = CHARACTER_VOCAB[cid]
    elif summoned:
        arrays["character_template"][row, s, slot] = 0
        arrays["character_id_slot"][row, s, slot] = strings.intern(cid)
    else:
        raise DomainSoAUnsupportedError(f"Unknown character {cid!r} is outside public-17 SoA coverage")
    arrays["character_valid"][row, s, slot] = True
    arrays["character_entity_slot"][row, s, slot] = _parse_entity_num(hero["entity_id"])
    arrays["character_hp"][row, s, slot] = _as_int("hp", hero["hp"])
    arrays["character_shield"][row, s, slot] = _as_int("shield", hero["shield"])
    arrays["character_energy"][row, s, slot] = _as_int("energy", hero["energy"])
    arrays["character_hp_presence"][row, s, slot] = PRESENCE_VALUE
    arrays["character_shield_presence"][row, s, slot] = PRESENCE_VALUE
    arrays["character_energy_presence"][row, s, slot] = PRESENCE_VALUE
    for i, key in enumerate(CHARACTER_INT_FIELDS):
        _put_optional_int(
            hero, key, arrays["character_int_presence"][row, s, slot],
            arrays["character_int_value"][row, s, slot], i, catalog=catalog or None,
        )
    for i, key in enumerate(CHARACTER_BOOL_FIELDS):
        _put_optional_bool(
            hero, key, arrays["character_bool_presence"][row, s, slot],
            arrays["character_bool_value"][row, s, slot], i, catalog=catalog or None,
        )
    for i, key in enumerate(CHARACTER_STR_FIELDS):
        allow = summoned or key == "shape" or key in CHARACTER_RUNTIME_STR_FIELDS
        _put_optional_str(
            hero, key, arrays["character_str_presence"][row, s, slot],
            arrays["character_str_slot"][row, s, slot], i,
            strings if allow else _RejectStrings(), catalog=catalog or None, allow_value=allow,
        )
    if "collapse_reductions" in hero:
        reductions = hero["collapse_reductions"]
        if type(reductions) is not dict:
            raise TypeError("collapse_reductions must be a dict")
        if len(reductions) > 2 * layout.max_characters:
            raise DomainSoACapacityError("collapse_reductions exceeds identity bound")
        arrays["character_reductions_presence"][row, s, slot] = PRESENCE_VALUE
        arrays["character_reductions_count"][row, s, slot] = len(reductions)
        for ri, (eid, value) in enumerate(reductions.items()):
            arrays["character_reductions_entity"][row, s, slot, ri] = _parse_entity_num(eid)
            arrays["character_reductions_value"][row, s, slot, ri] = _as_int("collapse_reductions value", value)
    _encode_lingke_chain(hero, arrays, row, s, slot, strings)
    arrays["character_effects_presence"][row, s, slot] = _effects_presence(hero)
    _encode_effects(hero, arrays, row, layout.holder_index(s, slot), layout, strings, "character effects")
    arrays["character_flags_presence"][row, s, slot] = _flags_presence(hero)
    _encode_flags(hero, arrays, row, layout.holder_index(s, slot), layout, strings)
    if catalog:
        arrays["character_ultimate_presence"][row, s, slot] = _catalog_nest(hero, catalog, "ultimate")
        arrays["character_mechanisms_presence"][row, s, slot] = _catalog_nest(hero, catalog, "mechanisms")
    else:
        if "ultimate" in hero or "mechanisms" in hero:
            raise DomainSoAUnsupportedError("non-catalog ultimate/mechanisms is outside SoA coverage")
        arrays["character_ultimate_presence"][row, s, slot] = PRESENCE_ABSENT
        arrays["character_mechanisms_presence"][row, s, slot] = PRESENCE_ABSENT
    _encode_equipment(hero, arrays, row, s, slot, strings)
    _encode_stat_layer(hero, catalog, arrays, row, s, slot)
    if hero.get("side") not in (None, SIDES[s]):
        raise DomainSoAUnsupportedError("character side does not match slot")
    if "resource_name" in hero:
        raise DomainSoAUnsupportedError("resource_name has no dedicated SoA array")



def _encode_lingke_chain(hero, arrays, row, s, slot, strings):
    if "lingke_chain" not in hero:
        arrays["character_lingke_presence"][row, s, slot] = PRESENCE_ABSENT
        return
    chain = hero["lingke_chain"]
    if chain is None:
        arrays["character_lingke_presence"][row, s, slot] = PRESENCE_NULL
        return
    if type(chain) is not dict:
        raise TypeError("lingke_chain must be a dict")
    _unknown(chain, frozenset(LINGKE_CHAIN_KEYS), "lingke_chain")
    for required in LINGKE_CHAIN_KEYS:
        if required not in chain:
            raise DomainSoAUnsupportedError(f"lingke_chain missing {required}")
    arrays["character_lingke_presence"][row, s, slot] = PRESENCE_VALUE
    arrays["character_lingke_entity_slot"][row, s, slot] = _parse_entity_num(chain["entity_id"])
    if type(chain["character_id"]) is not str:
        raise TypeError("lingke_chain.character_id must be str")
    arrays["character_lingke_id_slot"][row, s, slot] = strings.intern(chain["character_id"])
    arrays["character_lingke_count"][row, s, slot] = _as_int("lingke_chain.count", chain["count"])


def _encode_stat_layer(hero, catalog, arrays, row, s, slot):
    stats = hero.get("base_stats", _MISSING)
    if stats is _MISSING:
        arrays["character_base_attack_presence"][row, s, slot] = PRESENCE_ABSENT
        arrays["character_base_max_hp_presence"][row, s, slot] = PRESENCE_ABSENT
    else:
        if type(stats) is not dict or set(stats) != {"attack", "max_hp"}:
            raise DomainSoAUnsupportedError("base_stats must be {attack, max_hp}")
        attack = _as_int("base_stats.attack", stats["attack"])
        max_hp = _as_int("base_stats.max_hp", stats["max_hp"])
        if catalog.get("attack") == attack and catalog.get("max_hp") == max_hp:
            arrays["character_base_attack_presence"][row, s, slot] = PRESENCE_CATALOG
            arrays["character_base_max_hp_presence"][row, s, slot] = PRESENCE_CATALOG
        else:
            arrays["character_base_attack_presence"][row, s, slot] = PRESENCE_VALUE
            arrays["character_base_max_hp_presence"][row, s, slot] = PRESENCE_VALUE
            arrays["character_base_attack"][row, s, slot] = attack
            arrays["character_base_max_hp"][row, s, slot] = max_hp
    growth = hero.get("permanent_growth", _MISSING)
    if growth is _MISSING:
        arrays["character_growth_attack_presence"][row, s, slot] = PRESENCE_ABSENT
        arrays["character_growth_max_hp_presence"][row, s, slot] = PRESENCE_ABSENT
        return
    if type(growth) is not dict or set(growth) != {"attack", "max_hp"}:
        raise DomainSoAUnsupportedError("permanent_growth must be {attack, max_hp}")
    arrays["character_growth_attack_presence"][row, s, slot] = PRESENCE_VALUE
    arrays["character_growth_max_hp_presence"][row, s, slot] = PRESENCE_VALUE
    arrays["character_growth_attack"][row, s, slot] = _as_int("permanent_growth.attack", growth["attack"])
    arrays["character_growth_max_hp"][row, s, slot] = _as_int("permanent_growth.max_hp", growth["max_hp"])


def _encode_equipment(hero, arrays, row, s, slot, strings):
    if "equipment" not in hero:
        arrays["character_equipment_presence"][row, s, slot] = PRESENCE_ABSENT
        return
    equipment = hero["equipment"]
    if equipment is None:
        arrays["character_equipment_presence"][row, s, slot] = PRESENCE_NULL
        return
    if type(equipment) is not dict:
        raise TypeError("equipment must be a dict")
    _unknown(equipment, frozenset({"card_id", "equipment_id", "source_card_entity_id"}), "equipment")
    arrays["character_equipment_presence"][row, s, slot] = PRESENCE_VALUE
    arrays["character_equipment_card"][row, s, slot] = vocab_get(CARD_VOCAB, equipment["card_id"], what="card")
    arrays["character_equipment_id_slot"][row, s, slot] = strings.intern(equipment["equipment_id"])
    arrays["character_equipment_source_slot"][row, s, slot] = _parse_entity_num(equipment["source_card_entity_id"])


def _encode_card(card, side, arrays, row, index, strings):
    _unknown(card, CARD_KEYS, "card")
    card_id = card["card_id"]
    catalog = lookup_card(card_id) or catalog_card(card_id)
    vocab_get(CARD_VOCAB, card_id, what="card")
    arrays["card_valid"][row, index] = True
    arrays["card_entity_slot"][row, index] = _parse_entity_num(card["entity_id"])
    arrays["card_instance_slot"][row, index] = _parse_instance_num(card["instance_id"], side)
    arrays["card_template"][row, index] = CARD_VOCAB[card_id]
    arrays["card_side"][row, index] = side_index(side)
    if card.get("side") not in (None, side):
        raise DomainSoAUnsupportedError("card side does not match zone")
    copied = bool(card.get("copy"))
    for i, key in enumerate(CARD_INT_FIELDS):
        _put_optional_int(
            card, key, arrays["card_int_presence"][row, index],
            arrays["card_int_value"][row, index], i, catalog=catalog,
        )
    for i, key in enumerate(CARD_BOOL_FIELDS):
        _put_optional_bool(
            card, key, arrays["card_bool_presence"][row, index],
            arrays["card_bool_value"][row, index], i, catalog=catalog,
        )
    for i, key in enumerate(CARD_STR_FIELDS):
        _put_optional_str(
            card, key, arrays["card_str_presence"][row, index],
            arrays["card_str_slot"][row, index], i,
            strings if copied else _RejectStrings(), catalog=catalog, allow_value=copied,
        )
    arrays["card_source_presence"][row, index] = _catalog_nest(card, catalog, "source")
    arrays["card_play_options_presence"][row, index] = _catalog_nest(card, catalog, "play_options")
    _encode_jingu(card, arrays, row, index)


def _encode_jingu(card, arrays, row, index):
    if "jingu_mark" not in card:
        arrays["card_jingu_mark_presence"][row, index] = PRESENCE_ABSENT
        return
    mark = card["jingu_mark"]
    if mark is None:
        arrays["card_jingu_mark_presence"][row, index] = PRESENCE_NULL
        return
    if type(mark) is not dict:
        raise TypeError("jingu_mark must be a dict")
    _unknown(mark, frozenset({"delta", "source_entity_id"}), "jingu_mark")
    arrays["card_jingu_mark_presence"][row, index] = PRESENCE_VALUE
    arrays["card_jingu_delta"][row, index] = _as_int("jingu_mark.delta", mark["delta"])
    arrays["card_jingu_source_slot"][row, index] = _parse_entity_num(mark["source_entity_id"])

def _encode_effects(holder, arrays, row, holder_index, layout, strings, what):
    effects = holder.get("effects", _MISSING)
    if effects is _MISSING or effects is None:
        return
    if type(effects) is not list:
        raise TypeError(f"{what} must be a list")
    if len(effects) > layout.max_effects:
        raise DomainSoACapacityError(f"{what} exceeds layout.max_effects")
    arrays["holder_effect_count"][row, holder_index] = len(effects)
    for i, effect in enumerate(effects):
        if type(effect) is not dict:
            raise TypeError("effect must be a dict")
        _unknown(effect, frozenset(PUBLIC_EFFECT_FIELDS), "effect")
        arrays["effect_valid"][row, holder_index, i] = True
        arrays["effect_id_slot"][row, holder_index, i] = strings.intern(effect["effect_id"])
        arrays["effect_definition_slot"][row, holder_index, i] = strings.intern(effect["definition_id"])
        arrays["effect_property_slot"][row, holder_index, i] = strings.intern(effect["property"])
        arrays["effect_op"][row, holder_index, i] = _enum_index(effect["op"], EFFECT_OPS, "effect op")
        _encode_effect_value(effect["value"], arrays, row, holder_index, i)
        if "source_entity_id" not in effect:
            arrays["effect_source_entity_presence"][row, holder_index, i] = PRESENCE_ABSENT
        elif effect["source_entity_id"] is None:
            arrays["effect_source_entity_presence"][row, holder_index, i] = PRESENCE_NULL
        else:
            arrays["effect_source_entity_presence"][row, holder_index, i] = PRESENCE_VALUE
            arrays["effect_source_entity_slot"][row, holder_index, i] = _parse_entity_num(effect["source_entity_id"])
        arrays["effect_source_key_slot"][row, holder_index, i] = strings.intern(effect.get("source_key") or "")
        arrays["effect_scope"][row, holder_index, i] = _enum_index(effect.get("scope", "self"), EFFECT_SCOPES, "effect scope")
        arrays["effect_kind"][row, holder_index, i] = _enum_index(effect.get("kind", "persistent"), EFFECT_KINDS, "effect kind")
        arrays["effect_layer"][row, holder_index, i] = _as_int("layer", effect.get("layer", 0))
        arrays["effect_label_slot"][row, holder_index, i] = strings.intern(effect.get("label") or "")
        arrays["effect_visibility"][row, holder_index, i] = _enum_index(
            effect.get("visibility", "public"), EFFECT_VISIBILITY, "effect visibility"
        )
        if "stack_key" not in effect:
            arrays["effect_stack_key_presence"][row, holder_index, i] = PRESENCE_ABSENT
        elif effect["stack_key"] is None:
            arrays["effect_stack_key_presence"][row, holder_index, i] = PRESENCE_NULL
        else:
            arrays["effect_stack_key_presence"][row, holder_index, i] = PRESENCE_VALUE
            arrays["effect_stack_key_slot"][row, holder_index, i] = strings.intern(effect["stack_key"])
        if "stack_limit" not in effect:
            arrays["effect_stack_limit_presence"][row, holder_index, i] = PRESENCE_ABSENT
        elif effect["stack_limit"] is None:
            arrays["effect_stack_limit_presence"][row, holder_index, i] = PRESENCE_NULL
        else:
            arrays["effect_stack_limit_presence"][row, holder_index, i] = PRESENCE_VALUE
            arrays["effect_stack_limit"][row, holder_index, i] = _as_int("stack_limit", effect["stack_limit"])
        if "show_marker" not in effect:
            arrays["effect_show_marker_presence"][row, holder_index, i] = PRESENCE_ABSENT
        else:
            arrays["effect_show_marker_presence"][row, holder_index, i] = PRESENCE_VALUE
            arrays["effect_show_marker"][row, holder_index, i] = _as_bool("show_marker", effect["show_marker"])
        if "consume_on" not in effect:
            arrays["effect_consume_on_presence"][row, holder_index, i] = PRESENCE_ABSENT
        elif effect["consume_on"] is None:
            arrays["effect_consume_on_presence"][row, holder_index, i] = PRESENCE_NULL
        else:
            arrays["effect_consume_on_presence"][row, holder_index, i] = PRESENCE_VALUE
            arrays["effect_consume_on_slot"][row, holder_index, i] = strings.intern(effect["consume_on"])
        clear_on = effect.get("clear_on", _MISSING)
        arrays["effect_clear_presence"][row, holder_index, i] = PRESENCE_ABSENT if clear_on is _MISSING else PRESENCE_VALUE
        if clear_on is _MISSING:
            arrays["effect_clear_count"][row, holder_index, i] = 0
        else:
            if type(clear_on) is not list:
                raise TypeError("clear_on must be a list")
            if len(clear_on) > layout.max_clear_on:
                raise DomainSoACapacityError("clear_on exceeds layout.max_clear_on")
            arrays["effect_clear_count"][row, holder_index, i] = len(clear_on)
            for c, event in enumerate(clear_on):
                arrays["effect_clear_event"][row, holder_index, i, c] = _enum_index(
                    event, EFFECT_CLEAR_EVENTS, "clear_on"
                )
        _encode_expiry(effect.get("expiry", _MISSING), arrays, row, holder_index, i)
        conditions = effect.get("conditions", _MISSING)
        if conditions is not _MISSING:
            if type(conditions) is not list:
                raise TypeError("effect.conditions must be a list")
            if len(conditions) > layout.max_conditions:
                raise DomainSoACapacityError("effect.conditions exceeds layout.max_conditions")
            arrays["effect_conditions_presence"][row, holder_index, i] = PRESENCE_VALUE
            arrays["effect_conditions_count"][row, holder_index, i] = len(conditions)
            for ci, condition in enumerate(conditions):
                if type(condition) is not dict:
                    raise TypeError("effect condition must be a dict")
                _unknown(condition, frozenset({"id", "args"}), "condition")
                at = (row, holder_index, i, ci)
                arrays["effect_condition_id"][at] = _enum_index(condition["id"], EFFECT_CONDITION_IDS, "condition id")
                if "args" not in condition:
                    continue
                args = condition["args"]
                if type(args) is not dict:
                    raise TypeError("condition.args must be a dict")
                _unknown(args, frozenset(EFFECT_CONDITION_ARGS), "condition args")
                arrays["effect_condition_args_presence"][at] = PRESENCE_VALUE
                for ai, key in enumerate(EFFECT_CONDITION_ARGS):
                    if key not in args:
                        continue
                    index = at + (ai,)
                    value = args[key]
                    arrays["effect_condition_arg_presence"][index] = PRESENCE_VALUE
                    if type(value) is bool:
                        kind, number = VALUE_KIND_BOOL, int(value)
                        arrays["effect_condition_arg_int"][index] = number
                    elif type(value) is int:
                        kind = VALUE_KIND_INT
                        arrays["effect_condition_arg_int"][index] = _as_int(key, value)
                    elif type(value) is float and np.isfinite(value):
                        kind = VALUE_KIND_FLOAT
                        arrays["effect_condition_arg_float"][index] = value
                    elif type(value) is str:
                        kind = 3
                        arrays["effect_condition_arg_string"][index] = strings.intern(value)
                    else:
                        raise DomainSoAUnsupportedError("unsupported condition argument value")
                    arrays["effect_condition_arg_kind"][index] = kind


def _encode_effect_value(value, arrays, row, holder_index, i):
    if type(value) is bool:
        arrays["effect_value_kind"][row, holder_index, i] = VALUE_KIND_BOOL
        arrays["effect_value_int"][row, holder_index, i] = int(value)
        return
    if type(value) is int:
        arrays["effect_value_kind"][row, holder_index, i] = VALUE_KIND_INT
        arrays["effect_value_int"][row, holder_index, i] = _as_int("effect.value", value)
        return
    if type(value) is float:
        arrays["effect_value_kind"][row, holder_index, i] = VALUE_KIND_FLOAT
        arrays["effect_value_float"][row, holder_index, i] = value
        return
    raise DomainSoAUnsupportedError(f"unsupported effect value type {type(value).__name__}")


def _encode_expiry(expiry, arrays, row, holder_index, i):
    if expiry is _MISSING:
        arrays["effect_expiry_presence"][row, holder_index, i] = PRESENCE_ABSENT
        return
    if expiry is None:
        arrays["effect_expiry_presence"][row, holder_index, i] = PRESENCE_NULL
        return
    if type(expiry) is not dict:
        raise TypeError("expiry must be a dict")
    _unknown(expiry, frozenset({"side", "turn", "remaining", "phase"}), "expiry")
    arrays["effect_expiry_presence"][row, holder_index, i] = PRESENCE_VALUE
    if "side" not in expiry:
        arrays["effect_expiry_side_presence"][row, holder_index, i] = PRESENCE_ABSENT
    elif expiry["side"] is None:
        arrays["effect_expiry_side_presence"][row, holder_index, i] = PRESENCE_NULL
    else:
        arrays["effect_expiry_side_presence"][row, holder_index, i] = PRESENCE_VALUE
        arrays["effect_expiry_side"][row, holder_index, i] = side_index(expiry["side"])
    arrays["effect_expiry_phase"][row, holder_index, i] = _enum_index(expiry["phase"], EFFECT_PHASES, "expiry phase")
    if "turn" not in expiry:
        arrays["effect_expiry_turn_presence"][row, holder_index, i] = PRESENCE_ABSENT
    elif expiry["turn"] is None:
        arrays["effect_expiry_turn_presence"][row, holder_index, i] = PRESENCE_NULL
    else:
        arrays["effect_expiry_turn_presence"][row, holder_index, i] = PRESENCE_VALUE
        arrays["effect_expiry_turn"][row, holder_index, i] = _as_int("expiry.turn", expiry["turn"])
    if "remaining" not in expiry:
        arrays["effect_expiry_remaining_presence"][row, holder_index, i] = PRESENCE_ABSENT
    elif expiry["remaining"] is None:
        arrays["effect_expiry_remaining_presence"][row, holder_index, i] = PRESENCE_NULL
    else:
        arrays["effect_expiry_remaining_presence"][row, holder_index, i] = PRESENCE_VALUE
        arrays["effect_expiry_remaining"][row, holder_index, i] = _as_int("expiry.remaining", expiry["remaining"])


def _encode_flags(holder, arrays, row, holder_index, layout, strings):
    flags = holder.get("flags", _MISSING)
    if flags is _MISSING or flags is None:
        return
    if type(flags) is not dict:
        raise TypeError("flags must be a dict")
    _unknown(
        flags,
        frozenset(FLAG_INT_FIELDS) | frozenset(FLAG_BOOL_FIELDS) | frozenset(FLAG_STR_FIELDS) | FLAG_SPECIAL,
        "flags",
    )
    for i, key in enumerate(FLAG_INT_FIELDS):
        if key=='mimic'and type(flags.get(key))is dict:
            continue
        _put_optional_int(
            flags, key, arrays["flag_int_presence"][row, holder_index],
            arrays["flag_int_value"][row, holder_index], i, bool_keys=FLAG_BOOL_INTS,
        )
    for i, key in enumerate(FLAG_BOOL_FIELDS):
        _put_optional_bool(
            flags, key, arrays["flag_bool_presence"][row, holder_index],
            arrays["flag_bool_value"][row, holder_index], i,
        )
    for i, key in enumerate(FLAG_STR_FIELDS):
        _put_optional_str(
            flags, key, arrays["flag_str_presence"][row, holder_index],
            arrays["flag_str_slot"][row, holder_index], i, strings, allow_value=True,
        )
    mimic=flags.get('mimic',_MISSING)
    if type(mimic)is dict:
        _unknown(mimic,frozenset({'id'}),'mimic')
        arrays['flag_mimic_presence'][row,holder_index]=PRESENCE_VALUE
        if 'id'in mimic:
            if mimic['id']is None:
                arrays['flag_mimic_id_presence'][row,holder_index]=PRESENCE_NULL
            else:
                cid=mimic['id']
                if type(cid)is not str or (cid and cid not in CHARACTER_VOCAB):
                    raise DomainSoAUnsupportedError('unsupported mimicked character')
                arrays['flag_mimic_id_presence'][row,holder_index]=PRESENCE_VALUE
                arrays['flag_mimic_template'][row,holder_index]=CHARACTER_VOCAB[cid]if cid else 0
    collapse = flags.get("collapse", _MISSING)
    if collapse is _MISSING:
        arrays["flag_collapse_presence"][row, holder_index] = PRESENCE_ABSENT
    elif collapse is None:
        arrays["flag_collapse_presence"][row, holder_index] = PRESENCE_NULL
    else:
        if type(collapse) is not dict:
            raise TypeError("collapse must be a dict")
        _unknown(collapse, frozenset({"side", "until"}), "collapse")
        arrays["flag_collapse_presence"][row, holder_index] = PRESENCE_VALUE
        arrays["flag_collapse_side"][row, holder_index] = side_index(collapse["side"])
        arrays["flag_collapse_until"][row, holder_index] = _as_int("collapse.until", collapse["until"])
    expiring = flags.get("atk_buff_expiring", _MISSING)
    if expiring is _MISSING:
        arrays["flag_expiring_presence"][row, holder_index] = PRESENCE_ABSENT
    elif expiring is None:
        arrays["flag_expiring_presence"][row, holder_index] = PRESENCE_NULL
    else:
        if type(expiring) is not dict:
            raise TypeError("atk_buff_expiring must be a dict")
        _unknown(expiring, frozenset({"amount", "until"}), "atk_buff_expiring")
        arrays["flag_expiring_presence"][row, holder_index] = PRESENCE_VALUE
        arrays["flag_expiring_amount"][row, holder_index] = _as_int("atk_buff_expiring.amount", expiring["amount"])
        arrays["flag_expiring_until"][row, holder_index] = _as_int("atk_buff_expiring.until", expiring["until"])
    timed = flags.get("timed_attack", _MISSING)
    if timed is _MISSING:
        arrays["flag_timed_count"][row, holder_index] = 0
    else:
        if type(timed) is not list:
            raise TypeError("timed_attack must be a list")
        if len(timed) > layout.max_timed_attack:
            raise DomainSoACapacityError("timed_attack exceeds layout.max_timed_attack")
        arrays["flag_timed_count"][row, holder_index] = len(timed)
        for i, item in enumerate(timed):
            if type(item) is not dict:
                raise TypeError("timed_attack item must be a dict")
            _unknown(item, frozenset({"amount", "left"}), "timed_attack")
            arrays["flag_timed_amount"][row, holder_index, i] = _as_int("timed_attack.amount", item["amount"])
            arrays["flag_timed_left"][row, holder_index, i] = _as_int("timed_attack.left", item["left"])


def _encode_front_debuff(zone, arrays, row, s, layout, strings):
    if type(zone) is not dict:
        raise TypeError("front_debuff must be a dict")
    _unknown(zone, frozenset(ZONE_STATUS_KEYS) | frozenset({"dots"}), "front_debuff")
    delay_keys = ("left", "tick_side", "slow", "slow_turn", "by", "expires_side")
    burn_keys = ("left", "by", "stacks", "source", "applied_turn", "infinite")
    star_keys = ("left", "by", "source_cid", "renewals")
    weave_keys = ("by", "expires_side")
    stain_keys = ("by", "expires_side", "ends_left")
    specs = (delay_keys, burn_keys, star_keys, weave_keys, stain_keys)
    for z, key in enumerate(ZONE_STATUS_KEYS):
        if key not in zone:
            arrays["zone_status_presence"][row, s, z] = PRESENCE_ABSENT
            continue
        status = zone[key]
        if status is None:
            arrays["zone_status_presence"][row, s, z] = PRESENCE_NULL
            continue
        if type(status) is not dict:
            raise TypeError(f"{key} must be a dict")
        allowed = specs[z]
        _unknown(status, frozenset(allowed), key)
        arrays["zone_status_presence"][row, s, z] = PRESENCE_VALUE
        mask = 0
        ints, sides, bools, slots = [], [], [], []
        for name in allowed:
            if name not in status:
                continue
            value = status[name]
            bit = allowed.index(name)
            mask |= 1 << bit
            if name in ("left", "slow", "slow_turn", "stacks", "applied_turn", "renewals", "ends_left"):
                if value is None:
                    raise DomainSoAUnsupportedError(f"{key}.{name} null must use key absence")
                ints.append(_as_int(f"{key}.{name}", value))
            elif name in ("tick_side", "by", "expires_side"):
                sides.append(side_index(value))
            elif name == "infinite":
                bools.append(_as_bool(f"{key}.{name}", value))
            else:
                slots.append(strings.intern(value))
        arrays["zone_status_i32"][row, s, z, 7] = mask
        for i, number in enumerate(ints):
            arrays["zone_status_i32"][row, s, z, i] = number
        for i, number in enumerate(sides):
            arrays["zone_status_side"][row, s, z, i] = number
        for i, flag in enumerate(bools):
            arrays["zone_status_bool"][row, s, z, i] = flag
        for i, slot in enumerate(slots):
            arrays["zone_status_slot"][row, s, z, i] = slot
    dots = zone.get("dots", _MISSING)
    if dots is _MISSING:
        return
    if type(dots) is not dict:
        raise TypeError("dots must be a dict")
    arrays["zone_dots_presence"][row, s] = True
    extra = [key for key in dots if key not in DOT_KINDS]
    if extra:
        raise DomainSoAUnsupportedError(f"unsupported dot kinds {extra}")
    n = 0
    for kind in DOT_KINDS:
        if kind not in dots:
            continue
        spec = dots[kind]
        if type(spec) is not dict:
            raise TypeError("dot must be a dict")
        catalog = DOTS[kind]
        reconstructed = {
            "name": catalog["name"],
            "stacks": spec["stacks"],
            "by": spec["by"],
            "source": spec["source"],
            "mode": catalog["mode"],
            "halve": bool(catalog.get("halve")),
            "ignore_shield": bool(catalog.get("ignore_shield")),
            "max_stacks": catalog["max_stacks"],
        }
        if "left" in spec:
            reconstructed["left"] = spec["left"]
        if "applied_turn" in spec:
            reconstructed["applied_turn"] = spec["applied_turn"]
        if not _same(spec, reconstructed):
            raise DomainSoAUnsupportedError(f"dot {kind} diverges from catalog reconstructable fields")
        if n >= layout.max_dots:
            raise DomainSoACapacityError("dots exceed layout.max_dots")
        arrays["zone_dot_valid"][row, s, n] = True
        arrays["zone_dot_kind"][row, s, n] = DOT_KINDS.index(kind)
        arrays["zone_dot_stacks"][row, s, n] = _as_int("dot.stacks", spec["stacks"])
        arrays["zone_dot_by"][row, s, n] = side_index(spec["by"])
        arrays["zone_dot_source_slot"][row, s, n] = strings.intern(spec["source"])
        if "left" not in spec:
            arrays["zone_dot_left_presence"][row, s, n] = PRESENCE_ABSENT
        elif spec["left"] is None:
            arrays["zone_dot_left_presence"][row, s, n] = PRESENCE_NULL
        else:
            arrays["zone_dot_left_presence"][row, s, n] = PRESENCE_VALUE
            arrays["zone_dot_left"][row, s, n] = _as_int("dot.left", spec["left"])
        if "applied_turn" not in spec:
            arrays["zone_dot_applied_turn_presence"][row, s, n] = PRESENCE_ABSENT
        elif spec["applied_turn"] is None:
            arrays["zone_dot_applied_turn_presence"][row, s, n] = PRESENCE_NULL
        else:
            arrays["zone_dot_applied_turn_presence"][row, s, n] = PRESENCE_VALUE
            arrays["zone_dot_applied_turn"][row, s, n] = _as_int("dot.applied_turn", spec["applied_turn"])
        n += 1

def _encode_choice(state, arrays, row, layout, strings, register_card):
    pending = state.get("pending_choice", _MISSING)
    if pending is _MISSING:
        arrays["pending_choice_present"][row] = False
        return
    if pending is None:
        arrays["pending_choice_present"][row] = False
        return
    if type(pending) is not dict:
        raise TypeError("pending_choice must be a dict")
    _unknown(pending, CHOICE_ALLOWED, "pending_choice")
    for required in ("kind", "side", "prompt"):
        if required not in pending:
            raise DomainSoAUnsupportedError(f"pending_choice missing {required}")
    arrays["pending_choice_present"][row] = True
    arrays["choice_kind"][row] = _enum_index(pending["kind"], CHOICE_KINDS, "choice kind")
    arrays["choice_side"][row] = side_index(pending["side"])
    arrays["choice_prompt_slot"][row] = strings.intern(pending["prompt"])
    cards = pending.get("cards", [])
    if type(cards) is not list:
        raise TypeError("pending_choice.cards must be a list")
    if len(cards) > layout.max_choice_cards:
        raise DomainSoACapacityError("pending_choice.cards exceeds layout.max_choice_cards")
    arrays["choice_card_count"][row] = len(cards)
    for i, card in enumerate(cards):
        arrays["choice_card_index"][row, i] = register_card(card, pending["side"])
    for i, key in enumerate(CHOICE_INT_KEYS):
        _put_optional_int(pending, key, arrays["choice_int_presence"][row], arrays["choice_int_value"][row], i)
    for i, key in enumerate(CHOICE_BOOL_KEYS):
        _put_optional_bool(pending, key, arrays["choice_bool_presence"][row], arrays["choice_bool_value"][row], i)
    for i, key in enumerate(CHOICE_STR_KEYS):
        _put_optional_str(
            pending, key, arrays["choice_str_presence"][row], arrays["choice_str_slot"][row], i,
            strings, allow_value=True,
        )
    if "actor" not in pending:
        arrays["choice_actor_presence"][row] = PRESENCE_ABSENT
    elif pending["actor"] is None:
        arrays["choice_actor_presence"][row] = PRESENCE_NULL
    else:
        arrays["choice_actor_presence"][row] = PRESENCE_VALUE
        arrays["choice_actor_side"][row] = side_index(pending["side"])
        arrays["choice_actor_slot"][row] = _char_slot(list(state["sides"][pending["side"]]["characters"]), pending["actor"], "choice actor")
    if "continuation" not in pending:
        arrays["choice_continuation_presence"][row] = PRESENCE_ABSENT
    elif pending["continuation"] is None:
        arrays["choice_continuation_presence"][row] = PRESENCE_NULL
    else:
        arrays["choice_continuation_presence"][row] = PRESENCE_VALUE
        arrays["choice_continuation_slot"][row] = strings.intern(pending["continuation"])


def _encode_operation(state, arrays, row, layout, strings, register_card):
    operation = state.get("operation", _MISSING)
    if operation is _MISSING or operation is None:
        arrays["operation_present"][row] = False
        return
    if type(operation) is not dict:
        raise TypeError("operation must be a dict")
    _unknown(operation, OP_ALLOWED, "operation")
    for required in ("side", "actor"):
        if required not in operation:
            raise DomainSoAUnsupportedError(f"operation missing {required}")
    arrays["operation_present"][row] = True
    side = operation["side"]
    arrays["op_side"][row] = side_index(side)
    arrays["op_actor_slot"][row] = _char_slot(list(state["sides"][side]["characters"]), operation["actor"], "operation actor")
    card = operation.get("card", _MISSING)
    if card is _MISSING or card is None:
        arrays["op_card_index"][row] = -1
    else:
        arrays["op_card_index"][row] = register_card(card, side)
    if "target_id" not in operation:
        arrays["op_target_presence"][row] = PRESENCE_ABSENT
    elif operation["target_id"] is None:
        arrays["op_target_presence"][row] = PRESENCE_NULL
    else:
        arrays["op_target_presence"][row] = PRESENCE_VALUE
        arrays["op_target_slot"][row] = strings.intern(operation["target_id"])
    if "harmony" not in operation:
        arrays["op_harmony_presence"][row] = PRESENCE_ABSENT
    elif operation["harmony"] is None:
        arrays["op_harmony_presence"][row] = PRESENCE_NULL
    else:
        arrays["op_harmony_presence"][row] = PRESENCE_VALUE
        arrays["op_harmony_slot"][row] = strings.intern(operation["harmony"])
    arrays["op_first_bonus"][row] = _as_int("first_bonus", operation.get("first_bonus", 0))
    arrays["op_redeem_bonus"][row] = _as_int("redeem_bonus", operation.get("redeem_bonus", 0))
    arrays["op_record_candidate"][row] = _as_bool("record_candidate", operation.get("record_candidate", False))
    for i, key in enumerate(OP_FLAG_KEYS):
        if key == 'pending_sortie' and type(operation.get(key)) is dict:
            continue
        _put_optional_bool(operation, key, arrays["op_flag_presence"][row], arrays["op_flag_value"][row], i)
    for i, key in enumerate(OP_INT_KEYS):
        _put_optional_int(operation, key, arrays["op_int_presence"][row], arrays["op_int_value"][row], i)
    pending_sortie = operation.get('pending_sortie')
    if type(pending_sortie) is dict:
        _unknown(pending_sortie, frozenset({'spec', 'stay_bench'}), 'pending_sortie')
        if set(pending_sortie) != {'spec', 'stay_bench'}:
            raise DomainSoAUnsupportedError('incomplete pending_sortie')
        spec = pending_sortie['spec']
        if type(spec) is not dict:
            raise TypeError('pending_sortie.spec must be a dict')
        _unknown(spec, frozenset({'bonus'}), 'pending_sortie.spec')
        arrays['op_pending_sortie_presence'][row] = PRESENCE_VALUE
        arrays['op_pending_stay_bench'][row] = _as_bool('pending_sortie.stay_bench', pending_sortie['stay_bench'])
        if 'bonus' in spec:
            arrays['op_pending_bonus_presence'][row] = PRESENCE_VALUE
            arrays['op_pending_bonus'][row] = _as_int('pending_sortie.spec.bonus', spec['bonus'])
    for i, key in enumerate(OP_STR_KEYS):
        _put_optional_str(operation, key, arrays["op_str_presence"][row], arrays["op_str_slot"][row], i, strings, allow_value=True)
    if "option_id" not in operation:
        arrays["op_option_id_presence"][row] = PRESENCE_ABSENT
    elif operation["option_id"] is None:
        arrays["op_option_id_presence"][row] = PRESENCE_NULL
    else:
        arrays["op_option_id_presence"][row] = PRESENCE_VALUE
        arrays["op_option_id_slot"][row] = strings.intern(operation["option_id"])
    _encode_last_attack_target(operation, state, arrays, row)
    _encode_card_mark(operation, arrays, row)
    participants = operation.get("participants", [])
    if type(participants) is not list:
        raise TypeError("participants must be a list")
    if len(participants) > layout.max_participants:
        raise DomainSoACapacityError("participants exceed layout.max_participants")
    arrays["op_participant_count"][row] = len(participants)
    for i, item in enumerate(participants):
        if type(item) not in (list, tuple) or len(item) != 2:
            raise DomainSoAUnsupportedError("participant must be (side, cid)")
        arrays["op_participant_side"][row, i] = side_index(item[0])
        arrays["op_participant_slot"][row, i] = _char_slot(list(state["sides"][item[0]]["characters"]), item[1], "participant")
    choice_cards = operation.get("choice_cards", _MISSING)
    if choice_cards is _MISSING:
        arrays["op_choice_card_count"][row] = 0
    else:
        if type(choice_cards) is not list:
            raise TypeError("choice_cards must be a list")
        if len(choice_cards) > layout.max_choice_cards:
            raise DomainSoACapacityError("operation choice_cards exceed layout.max_choice_cards")
        arrays["op_choice_card_count"][row] = len(choice_cards)
        for i, card in enumerate(choice_cards):
            arrays["op_choice_card_index"][row, i] = register_card(card, side)


def _encode_last_attack_target(operation, state, arrays, row):
    if "last_attack_target" not in operation:
        arrays["op_last_attack_target_presence"][row] = PRESENCE_ABSENT
        return
    target = operation["last_attack_target"]
    if target is None:
        arrays["op_last_attack_target_presence"][row] = PRESENCE_NULL
        return
    if type(target) not in (list, tuple) or len(target) != 2:
        raise DomainSoAUnsupportedError("last_attack_target must be (side, cid)")
    arrays["op_last_attack_target_presence"][row] = PRESENCE_VALUE
    arrays["op_last_attack_target_slot"][row] = (
        side_index(target[0]) * 256 + (255 if target[1] == 'player' else _char_slot(list(state["sides"][target[0]]["characters"]), target[1], "last_attack_target"))
    )


def _encode_card_mark(operation, arrays, row):
    if "card_mark" not in operation:
        arrays["op_card_mark_presence"][row] = PRESENCE_ABSENT
        return
    mark = operation["card_mark"]
    if mark is None:
        arrays["op_card_mark_presence"][row] = PRESENCE_NULL
        return
    if type(mark) is not dict:
        raise TypeError("card_mark must be a dict")
    _unknown(mark, frozenset({"delta", "source_entity_id"}), "card_mark")
    arrays["op_card_mark_presence"][row] = PRESENCE_VALUE
    arrays["op_card_mark_delta"][row] = _as_int("card_mark.delta", mark["delta"])
    arrays["op_card_mark_source_slot"][row] = _parse_entity_num(mark["source_entity_id"])


def _encode_deferred(state, arrays, row, layout, strings):
    deferred = state.get("deferred_actions", _MISSING)
    if deferred is _MISSING:
        arrays["deferred_presence"][row] = PRESENCE_ABSENT
        arrays["deferred_count"][row] = 0
        return
    if deferred is None:
        arrays["deferred_presence"][row] = PRESENCE_NULL
        arrays["deferred_count"][row] = 0
        return
    if type(deferred) is not list:
        raise TypeError("deferred_actions must be a list")
    if len(deferred) > layout.max_deferred:
        raise DomainSoACapacityError("deferred_actions exceed layout.max_deferred")
    arrays["deferred_presence"][row] = PRESENCE_VALUE
    arrays["deferred_count"][row] = len(deferred)
    for i, item in enumerate(deferred):
        if type(item) is not dict:
            raise DomainSoAUnsupportedError("deferred action must be a dict")
        _unknown(item, DEFERRED_ALLOWED, "deferred")
        arrays["deferred_side"][row, i] = side_index(item["side"])
        arrays["deferred_actor_slot"][row, i] = _char_slot(
            list(state["sides"][item["side"]]["characters"]), item["actor"], "deferred actor"
        )
        arrays["deferred_entity_slot"][row, i] = _parse_entity_num(item["entity_id"])
        arrays["deferred_kind_slot"][row, i] = strings.intern(item["kind"])


def _encode_down_responses(state, arrays, row, layout, strings):
    items = state.get("_pending_down_responses", _MISSING)
    if items is _MISSING:
        arrays["down_response_count"][row] = 0
        return
    if type(items) is not list:
        raise TypeError("_pending_down_responses must be a list")
    if not items:
        raise DomainSoAUnsupportedError("empty _pending_down_responses cannot be distinguished from absence")
    if len(items) > layout.max_down_responses:
        raise DomainSoACapacityError("down responses exceed layout.max_down_responses")
    arrays["down_response_count"][row] = len(items)
    for i, item in enumerate(items):
        if type(item) not in (list, tuple) or len(item) != 2:
            raise DomainSoAUnsupportedError("down response must be (side, target)")
        arrays["down_response_side"][row, i] = side_index(item[0])
        arrays["down_response_target_slot"][row, i] = strings.intern(item[1])

def _decode_row(batch, row):
    phase = _enum_from(batch.phase[row], GAME_PHASES, "phase")
    rng = int(batch.rng[row])
    if rng < 0 or rng > RNG_MAX:
        raise DomainSoADecodeError("rng is outside the 63-bit stream")
    state = {
        "rules_version": _load_string(batch, row, batch.rules_version_slot[row]),
        "design_version": _load_string(batch, row, batch.design_version_slot[row]),
        "version": int(batch.version[row]),
        "phase": phase,
        "active_side": _side_from(batch.active_side[row]),
        "first_side": _side_from(batch.first_side[row]),
        "turn": int(batch.turn[row]),
        "rng": rng,
        "next_instance": int(batch.next_instance[row]),
        "event_seq": int(batch.event_seq[row]),
        "pending_choice": None,
        "operation": None,
        "mulligan_done": [_side_from(batch.mulligan_sides[row, i]) for i in range(int(batch.mulligan_count[row]))],
        "current_action_id": _load_string(batch, row, batch.current_action_id_slot[row]),
        "current_action_version": int(batch.current_action_version[row]),
        "combat_group": int(batch.combat_group[row]),
        "sides": {},
        "escalation_enabled": bool(batch.escalation_enabled[row]),
        "harmony_residue_version": int(batch.harmony_residue_version[row]),
        "next_entity": int(batch.next_entity[row]),
    }
    _apply_optional(state, "winner", _decode_winner(batch, row))
    _apply_optional(state, "reason", _decode_reason(batch, row))
    _apply_optional(state, "turn_start_pending", _decode_optional_side(batch, row, "turn_start_pending"))
    _apply_optional(state, "_response_lock", _decode_optional_bool(batch, row, "response_lock"))
    _apply_optional(state, "_response_intercept", _decode_optional_bool(batch, row, "response_intercept"))
    _apply_optional(state, "_draining_actions", _decode_optional_bool(batch, row, "draining"))
    cards = _decode_cards(batch, row)
    for s, side in enumerate(SIDES):
        if not bool(batch.player_valid[row, s]):
            raise DomainSoADecodeError(f"missing player {side}")
        state["sides"][side] = _decode_player(batch, row, s, side, cards)
    if bool(batch.pending_choice_present[row]):
        state["pending_choice"] = _decode_choice(batch, row, state, cards)
    if bool(batch.operation_present[row]):
        state["operation"] = _decode_operation(batch, row, state, cards)
    deferred_p = int(batch.deferred_presence[row])
    if deferred_p == PRESENCE_NULL:
        state["deferred_actions"] = None
    elif deferred_p == PRESENCE_VALUE:
        state["deferred_actions"] = _decode_deferred(batch, row, state, int(batch.deferred_count[row]))
    elif deferred_p != PRESENCE_ABSENT:
        raise DomainSoADecodeError("invalid deferred_actions presence")
    down_n = int(batch.down_response_count[row])
    if down_n:
        state["_pending_down_responses"] = [
            (_side_from(batch.down_response_side[row, i]), _load_string(batch, row, batch.down_response_target_slot[row, i]))
            for i in range(down_n)
        ]
    return state


def _decode_winner(batch, row):
    code = int(batch.winner_presence[row])
    if code == PRESENCE_ABSENT:
        return _MISSING
    if code == PRESENCE_NULL:
        return None
    if code != PRESENCE_VALUE:
        raise DomainSoADecodeError("invalid winner presence")
    return _enum_from(batch.winner[row], WINNERS, "winner")


def _decode_reason(batch, row):
    code = int(batch.reason_presence[row])
    if code == PRESENCE_ABSENT:
        return _MISSING
    if code == PRESENCE_NULL:
        return None
    if code != PRESENCE_VALUE:
        raise DomainSoADecodeError("invalid reason presence")
    return _load_string(batch, row, batch.reason_slot[row])


def _decode_optional_side(batch, row, prefix):
    code = int(getattr(batch, prefix + "_presence")[row])
    if code == PRESENCE_ABSENT:
        return _MISSING
    if code == PRESENCE_NULL:
        return None
    if code != PRESENCE_VALUE:
        raise DomainSoADecodeError(f"invalid {prefix} presence")
    return _side_from(getattr(batch, prefix + "_side")[row])


def _decode_optional_bool(batch, row, prefix):
    code = int(getattr(batch, prefix + "_presence")[row])
    if code == PRESENCE_ABSENT:
        return _MISSING
    if code == PRESENCE_NULL:
        return None
    if code != PRESENCE_VALUE:
        raise DomainSoADecodeError(f"invalid {prefix} presence")
    return bool(getattr(batch, prefix)[row])


def _decode_cards(batch, row):
    cards = {}
    for index in range(batch.layout.max_cards):
        if not bool(batch.card_valid[row, index]):
            continue
        template = int(batch.card_template[row, index])
        card_id = _CARD_FROM_VOCAB.get(template)
        if card_id is None:
            raise DomainSoADecodeError(f"unknown card template {template}")
        catalog = lookup_card(card_id) or catalog_card(card_id)
        payload = deepcopy(catalog)
        side = _side_from(batch.card_side[row, index])
        payload["card_id"] = card_id
        payload["instance_id"] = f"{side}-{int(batch.card_instance_slot[row, index])}"
        for i, key in enumerate(CARD_INT_FIELDS):
            _apply_optional(payload, key, _take_optional(
                batch.card_int_presence[row, index], batch.card_int_value[row, index], i, int, catalog, key))
        for i, key in enumerate(CARD_BOOL_FIELDS):
            _apply_optional(payload, key, _take_optional(
                batch.card_bool_presence[row, index], batch.card_bool_value[row, index], i, lambda v: bool(v), catalog, key))
        for i, key in enumerate(CARD_STR_FIELDS):
            _apply_optional(payload, key, _take_optional(
                batch.card_str_presence[row, index], batch.card_str_slot[row, index], i,
                lambda v, _row=row: _load_string(batch, _row, v), catalog, key))
        source_p = int(batch.card_source_presence[row, index])
        options_p = int(batch.card_play_options_presence[row, index])
        if source_p == PRESENCE_ABSENT:
            payload.pop("source", None)
        elif source_p != PRESENCE_CATALOG:
            raise DomainSoAUnsupportedError("card source must remain a catalog nest")
        if options_p == PRESENCE_ABSENT:
            payload.pop("play_options", None)
        elif options_p != PRESENCE_CATALOG:
            raise DomainSoAUnsupportedError("play_options must remain a catalog nest")
        mark_p = int(batch.card_jingu_mark_presence[row, index])
        if mark_p == PRESENCE_ABSENT:
            payload.pop("jingu_mark", None)
        elif mark_p == PRESENCE_NULL:
            payload["jingu_mark"] = None
        elif mark_p == PRESENCE_VALUE:
            payload["jingu_mark"] = {
                "delta": int(batch.card_jingu_delta[row, index]),
                "source_entity_id": _entity_text(batch.card_jingu_source_slot[row, index]),
            }
        else:
            raise DomainSoADecodeError("invalid jingu_mark presence")
        cards[index] = CardEntity(payload, entity_id=_entity_text(batch.card_entity_slot[row, index]), side=side)
    for index, card in cards.items():
        if not bool(batch.card_is_view[row, index]):
            continue
        physical_index = int(batch.card_physical_index[row, index])
        if physical_index not in cards or physical_index == index or bool(batch.card_is_view[row, physical_index]):
            raise DomainSoADecodeError('invalid physical view reference')
        physical = cards[physical_index]
        if any(card.get(key) != physical.get(key) for key in ('entity_id','instance_id','side')):
            raise DomainSoADecodeError('physical view identity mismatch')
        card['_physical_card'] = physical
    return cards


def _decode_player(batch, row, s, side, cards):
    order_n = int(batch.player_order_count[row, s])
    stored_n = int(batch.player_character_count[row, s])
    if not 1 <= stored_n <= batch.layout.max_characters or not 0 <= order_n <= stored_n:
        raise DomainSoADecodeError("invalid character storage or lineup count")
    stored, characters = [], {}
    for slot in range(stored_n):
        if not bool(batch.character_valid[row, s, slot]):
            raise DomainSoADecodeError("missing stored character")
        hero, cid = _decode_character(batch, row, s, slot, side)
        if cid in characters:
            raise DomainSoADecodeError("duplicate stored character id")
        stored.append(cid)
        characters[cid] = hero
    order = []
    for i in range(order_n):
        slot = int(batch.player_order_slot[row, s, i])
        if not 0 <= slot < stored_n or stored[slot] in order:
            raise DomainSoADecodeError("invalid or duplicate lineup slot")
        order.append(stored[slot])
    payload = {
        "name": _load_string(batch, row, batch.player_name_slot[row, s]),
        "hp": int(batch.player_hp[row, s]),
        "max_hp": int(batch.player_max_hp[row, s]),
        "shield": int(batch.player_shield[row, s]),
        "ap": int(batch.player_ap[row, s]),
        "order": order,
        "characters": characters,
    }
    _apply_optional(payload, "front", _decode_front_ref(batch.player_front_presence[row, s], batch.player_front_slot[row, s], stored, "front"))
    _apply_optional(payload, "last_front", _decode_front_ref(batch.player_last_front_presence[row, s], batch.player_last_front_slot[row, s], stored, "last_front"))
    for z, zone in enumerate(ZONES):
        count = int(batch.zone_count[row, s, z])
        if count < 0 or count > batch.layout.max_zone:
            raise DomainSoADecodeError(f"invalid {zone} count")
        packed = []
        for pos in range(count):
            index = int(batch.zone_card_index[row, s, z, pos])
            if index not in cards:
                raise DomainSoADecodeError(f"{zone} card index {index} is missing")
            packed.append(cards[index])
        payload[zone] = packed
    rp=int(batch.player_revealed_presence[row,s]);rn=int(batch.player_revealed_count[row,s])
    if not 0<=rn<=batch.layout.max_cards:
        raise DomainSoADecodeError('revealed count outside layout')
    if rp==PRESENCE_NULL:
        payload['revealed_ids']=None
    elif rp==PRESENCE_VALUE:
        payload['revealed_ids']=[_side_from(batch.player_revealed_side[row,s,j])+'-'+str(int(batch.player_revealed_instance[row,s,j]))for j in range(rn)]
    elif rp!=PRESENCE_ABSENT:
        raise DomainSoADecodeError('invalid revealed presence')
    payload["records"] = []
    for i, key in enumerate(PLAYER_INT_FIELDS):
        _apply_optional(payload, key, _take_optional(
            batch.player_int_presence[row, s], batch.player_int_value[row, s], i, int, bool_keys=PLAYER_BOOL_INTS, key=key))
    for i, key in enumerate(PLAYER_BOOL_FIELDS):
        _apply_optional(payload, key, _take_optional(
            batch.player_bool_presence[row, s], batch.player_bool_value[row, s], i, lambda v: bool(v)))
    for i, key in enumerate(PLAYER_STR_FIELDS):
        _apply_optional(payload, key, _take_optional(
            batch.player_str_presence[row, s], batch.player_str_slot[row, s], i,
            lambda v, _row=row: _load_string(batch, _row, v)))
    used = {}
    used_n = int(batch.player_used_count[row, s])
    for i in range(used_n):
        kind=int(batch.player_used_value_kind[row,s,i])
        if kind not in (0,1):raise DomainSoADecodeError('invalid used value kind')
        used[_load_string(batch, row, batch.player_used_key_slot[row, s, i])] = (bool(batch.player_used_value[row,s,i]) if kind==0 else int(batch.player_used_int[row,s,i]))
    payload["used"] = used
    payload["harmonized"] = {
        cid: True for i, cid in enumerate(stored) if bool(batch.player_harmonized[row, s, i])
    }
    payload["front_debuff"] = _decode_front_debuff(batch, row, s)
    attacked_p = int(batch.player_attacked_this_turn_presence[row, s])
    if attacked_p == PRESENCE_VALUE:
        count = int(batch.player_attacked_count[row, s])
        if count < 0 or count > len(stored):
            raise DomainSoADecodeError("invalid attacked_this_turn count")
        attacked = []
        for i in range(count):
            index = int(batch.player_attacked_slot[row, s, i])
            if index < 0 or index >= len(stored):
                raise DomainSoADecodeError("attacked_this_turn slot outside order")
            attacked.append(characters[stored[index]]["entity_id"])
        payload["attacked_this_turn"] = attacked
    elif attacked_p not in (PRESENCE_ABSENT,):
        raise DomainSoADecodeError("invalid attacked_this_turn presence")
    if int(batch.player_records_count[row, s]):
        raise DomainSoAUnsupportedError("records payloads have no dedicated SoA arrays")
    holder = batch.layout.holder_index(s, None)
    _apply_optional(payload, "effects", _decode_effects_field(batch, row, holder, batch.player_effects_presence[row, s]))
    _apply_optional(payload, "flags", _decode_flags_field(batch, row, holder, batch.player_flags_presence[row, s]))
    return PlayerEntity(payload, entity_id=_entity_text(batch.player_entity_slot[row, s]), side=side)


def _decode_front_ref(presence, slot, order, what):
    code = int(presence)
    if code == PRESENCE_ABSENT:
        return _MISSING
    if code == PRESENCE_NULL:
        return None
    if code != PRESENCE_VALUE:
        raise DomainSoADecodeError(f"invalid {what} presence")
    index = int(slot)
    if index < 0 or index >= len(order):
        raise DomainSoADecodeError(f"{what} slot outside order")
    return order[index]


def _decode_character(batch, row, s, slot, side):
    template = int(batch.character_template[row, s, slot])
    if template == 0:
        cid = _load_string(batch, row, batch.character_id_slot[row, s, slot])
        catalog = {}
        entity_id = _entity_text(batch.character_entity_slot[row, s, slot])
        hero = CharacterEntity({"id": cid}, entity_id=entity_id, side=side)
    else:
        cid = _CHAR_FROM_VOCAB.get(template)
        if cid is None:
            raise DomainSoADecodeError(f"unknown character template {template}")
        catalog = lookup_character(cid)
        entity_id = _entity_text(batch.character_entity_slot[row, s, slot])
        hero = new_character(cid, entity_id=entity_id, side=side)
    if int(batch.character_hp_presence[row, s, slot]) != PRESENCE_VALUE:
        raise DomainSoADecodeError("character hp must be a dedicated value")
    hero["hp"] = int(batch.character_hp[row, s, slot])
    hero["shield"] = int(batch.character_shield[row, s, slot])
    hero["energy"] = int(batch.character_energy[row, s, slot])
    for i, key in enumerate(CHARACTER_INT_FIELDS):
        _apply_optional(hero, key, _take_optional(
            batch.character_int_presence[row, s, slot], batch.character_int_value[row, s, slot], i, int, catalog or None, key))
    for i, key in enumerate(CHARACTER_BOOL_FIELDS):
        _apply_optional(hero, key, _take_optional(
            batch.character_bool_presence[row, s, slot], batch.character_bool_value[row, s, slot], i,
            lambda v: bool(v), catalog or None, key))
    for i, key in enumerate(CHARACTER_STR_FIELDS):
        _apply_optional(hero, key, _take_optional(
            batch.character_str_presence[row, s, slot], batch.character_str_slot[row, s, slot], i,
            lambda v, _row=row: _load_string(batch, _row, v), catalog or None, key))
    holder = batch.layout.holder_index(s, slot)
    _apply_optional(hero, "effects", _decode_effects_field(batch, row, holder, batch.character_effects_presence[row, s, slot]))
    _apply_optional(hero, "flags", _decode_flags_field(batch, row, holder, batch.character_flags_presence[row, s, slot]))
    _decode_stat_layer(batch, row, s, slot, hero, catalog)
    ult = int(batch.character_ultimate_presence[row, s, slot])
    mech = int(batch.character_mechanisms_presence[row, s, slot])
    if ult == PRESENCE_ABSENT:
        hero.pop("ultimate", None)
    elif ult != PRESENCE_CATALOG:
        raise DomainSoAUnsupportedError("ultimate must remain a catalog nest")
    if mech == PRESENCE_ABSENT:
        hero.pop("mechanisms", None)
    elif mech != PRESENCE_CATALOG:
        raise DomainSoAUnsupportedError("mechanisms must remain a catalog nest")
    equip_p = int(batch.character_equipment_presence[row, s, slot])
    if equip_p == PRESENCE_ABSENT:
        hero.pop("equipment", None)
    elif equip_p == PRESENCE_NULL:
        hero["equipment"] = None
    elif equip_p == PRESENCE_VALUE:
        card_id = _CARD_FROM_VOCAB.get(int(batch.character_equipment_card[row, s, slot]))
        if card_id is None:
            raise DomainSoADecodeError("unknown equipment card")
        hero["equipment"] = {
            "card_id": card_id,
            "equipment_id": _load_string(batch, row, batch.character_equipment_id_slot[row, s, slot]),
            "source_card_entity_id": _entity_text(batch.character_equipment_source_slot[row, s, slot]),
        }
    else:
        raise DomainSoADecodeError("invalid equipment presence")
    rp = int(batch.character_reductions_presence[row, s, slot])
    rn = int(batch.character_reductions_count[row, s, slot])
    if not 0 <= rn <= 2 * batch.layout.max_characters:
        raise DomainSoADecodeError("collapse_reductions count outside layout")
    if rp == PRESENCE_VALUE:
        reductions = {}
        for ri in range(rn):
            eid = _entity_text(batch.character_reductions_entity[row, s, slot, ri])
            if eid in reductions:
                raise DomainSoADecodeError("duplicate collapse_reductions identity")
            reductions[eid] = int(batch.character_reductions_value[row, s, slot, ri])
        hero["collapse_reductions"] = reductions
    elif rp != PRESENCE_ABSENT:
        raise DomainSoADecodeError("invalid collapse_reductions presence")
    _apply_optional(hero, "lingke_chain", _decode_lingke_chain(batch, row, s, slot))
    return hero, cid


def _decode_lingke_chain(batch, row, s, slot):
    presence = int(batch.character_lingke_presence[row, s, slot])
    if presence == PRESENCE_ABSENT:
        return _MISSING
    if presence == PRESENCE_NULL:
        return None
    if presence != PRESENCE_VALUE:
        raise DomainSoADecodeError('invalid lingke_chain presence')
    return dict(entity_id=_entity_text(batch.character_lingke_entity_slot[row, s, slot]),
                character_id=_load_string(batch, row, batch.character_lingke_id_slot[row, s, slot]),
                count=int(batch.character_lingke_count[row, s, slot]))


def _decode_stat_layer(batch, row, s, slot, hero, catalog):
    base_a = int(batch.character_base_attack_presence[row, s, slot])
    base_h = int(batch.character_base_max_hp_presence[row, s, slot])
    if base_a != base_h:
        raise DomainSoADecodeError("base_stats presence mismatch")
    if base_a == PRESENCE_ABSENT:
        hero.pop("base_stats", None)
    elif base_a == PRESENCE_CATALOG:
        hero["base_stats"] = {"attack": catalog["attack"], "max_hp": catalog["max_hp"]}
    elif base_a == PRESENCE_VALUE:
        hero["base_stats"] = {
            "attack": int(batch.character_base_attack[row, s, slot]),
            "max_hp": int(batch.character_base_max_hp[row, s, slot]),
        }
    else:
        raise DomainSoADecodeError("invalid base_stats presence")
    grow_a = int(batch.character_growth_attack_presence[row, s, slot])
    grow_h = int(batch.character_growth_max_hp_presence[row, s, slot])
    if grow_a != grow_h:
        raise DomainSoADecodeError("permanent_growth presence mismatch")
    if grow_a == PRESENCE_ABSENT:
        hero.pop("permanent_growth", None)
    elif grow_a == PRESENCE_VALUE:
        hero["permanent_growth"] = {
            "attack": int(batch.character_growth_attack[row, s, slot]),
            "max_hp": int(batch.character_growth_max_hp[row, s, slot]),
        }
    else:
        raise DomainSoADecodeError("invalid permanent_growth presence")

def _decode_effects_field(batch, row, holder, presence):
    code = int(presence)
    if code == PRESENCE_ABSENT:
        return _MISSING
    if code == PRESENCE_NULL:
        return None
    if code != PRESENCE_VALUE:
        raise DomainSoADecodeError("invalid effects presence")
    count = int(batch.holder_effect_count[row, holder])
    effects = []
    for i in range(count):
        if not bool(batch.effect_valid[row, holder, i]):
            raise DomainSoADecodeError("missing packed effect")
        effect = {
            "effect_id": _load_string(batch, row, batch.effect_id_slot[row, holder, i]),
            "definition_id": _load_string(batch, row, batch.effect_definition_slot[row, holder, i]),
            "property": _load_string(batch, row, batch.effect_property_slot[row, holder, i]),
            "op": _enum_from(batch.effect_op[row, holder, i], EFFECT_OPS, "effect op"),
            "value": _decode_effect_value(batch, row, holder, i),
            "source_key": _load_string(batch, row, batch.effect_source_key_slot[row, holder, i]),
            "scope": _enum_from(batch.effect_scope[row, holder, i], EFFECT_SCOPES, "effect scope"),
            "kind": _enum_from(batch.effect_kind[row, holder, i], EFFECT_KINDS, "effect kind"),
            "layer": int(batch.effect_layer[row, holder, i]),
            "label": _load_string(batch, row, batch.effect_label_slot[row, holder, i]),
            "visibility": _enum_from(batch.effect_visibility[row, holder, i], EFFECT_VISIBILITY, "effect visibility"),
        }
        src_p = int(batch.effect_source_entity_presence[row, holder, i])
        if src_p == PRESENCE_NULL:
            effect["source_entity_id"] = None
        elif src_p == PRESENCE_VALUE:
            effect["source_entity_id"] = _entity_text(batch.effect_source_entity_slot[row, holder, i])
        elif src_p != PRESENCE_ABSENT:
            raise DomainSoADecodeError("invalid effect source presence")
        stack_p = int(batch.effect_stack_key_presence[row, holder, i])
        if stack_p == PRESENCE_NULL:
            effect["stack_key"] = None
        elif stack_p == PRESENCE_VALUE:
            effect["stack_key"] = _load_string(batch, row, batch.effect_stack_key_slot[row, holder, i])
        elif stack_p != PRESENCE_ABSENT:
            raise DomainSoADecodeError("invalid stack_key presence")
        limit_p = int(batch.effect_stack_limit_presence[row, holder, i])
        if limit_p == PRESENCE_NULL:
            effect["stack_limit"] = None
        elif limit_p == PRESENCE_VALUE:
            effect["stack_limit"] = int(batch.effect_stack_limit[row, holder, i])
        elif limit_p != PRESENCE_ABSENT:
            raise DomainSoADecodeError("invalid stack_limit presence")
        show_p = int(batch.effect_show_marker_presence[row, holder, i])
        if show_p == PRESENCE_VALUE:
            effect["show_marker"] = bool(batch.effect_show_marker[row, holder, i])
        elif show_p != PRESENCE_ABSENT:
            raise DomainSoADecodeError("invalid show_marker presence")
        consume_p = int(batch.effect_consume_on_presence[row, holder, i])
        if consume_p == PRESENCE_NULL:
            effect["consume_on"] = None
        elif consume_p == PRESENCE_VALUE:
            effect["consume_on"] = _load_string(batch, row, batch.effect_consume_on_slot[row, holder, i])
        elif consume_p != PRESENCE_ABSENT:
            raise DomainSoADecodeError("invalid consume_on presence")
        clear_n = int(batch.effect_clear_count[row, holder, i])
        if int(batch.effect_clear_presence[row, holder, i]) == PRESENCE_VALUE:
            effect["clear_on"] = [
                _enum_from(batch.effect_clear_event[row, holder, i, c], EFFECT_CLEAR_EVENTS, "clear_on")
                for c in range(clear_n)
            ]
        expiry = _decode_expiry(batch, row, holder, i)
        if expiry is not _MISSING:
            effect["expiry"] = expiry
        condition_n = int(batch.effect_conditions_count[row, holder, i])
        if not 0 <= condition_n <= batch.layout.max_conditions:
            raise DomainSoADecodeError("condition count outside layout")
        if int(batch.effect_conditions_presence[row, holder, i]) == PRESENCE_VALUE:
            conditions = []
            for ci in range(condition_n):
                at = (row, holder, i, ci)
                condition = {"id": _enum_from(batch.effect_condition_id[at], EFFECT_CONDITION_IDS, "condition id")}
                if int(batch.effect_condition_args_presence[at]) == PRESENCE_VALUE:
                    args = {}
                    for ai, key in enumerate(EFFECT_CONDITION_ARGS):
                        index = at + (ai,)
                        if int(batch.effect_condition_arg_presence[index]) != PRESENCE_VALUE:
                            continue
                        kind = int(batch.effect_condition_arg_kind[index])
                        if kind == VALUE_KIND_INT:
                            value = int(batch.effect_condition_arg_int[index])
                        elif kind == VALUE_KIND_FLOAT:
                            value = float(batch.effect_condition_arg_float[index])
                        elif kind == VALUE_KIND_BOOL:
                            value = bool(batch.effect_condition_arg_int[index])
                        elif kind == 3:
                            value = _load_string(batch, row, batch.effect_condition_arg_string[index])
                        else:
                            raise DomainSoADecodeError("unknown condition argument type")
                        args[key] = value
                    condition["args"] = args
                conditions.append(condition)
            effect["conditions"] = conditions
        effects.append(effect)
    return effects


def _decode_effect_value(batch, row, holder, i):
    kind = int(batch.effect_value_kind[row, holder, i])
    if kind == VALUE_KIND_BOOL:
        return bool(int(batch.effect_value_int[row, holder, i]))
    if kind == VALUE_KIND_INT:
        return int(batch.effect_value_int[row, holder, i])
    if kind == VALUE_KIND_FLOAT:
        return float(batch.effect_value_float[row, holder, i])
    raise DomainSoADecodeError(f"invalid effect value kind {kind}")


def _decode_expiry(batch, row, holder, i):
    code = int(batch.effect_expiry_presence[row, holder, i])
    if code == PRESENCE_ABSENT:
        return _MISSING
    if code == PRESENCE_NULL:
        return None
    if code != PRESENCE_VALUE:
        raise DomainSoADecodeError("invalid expiry presence")
    expiry = {"phase": _enum_from(batch.effect_expiry_phase[row, holder, i], EFFECT_PHASES, "expiry phase")}
    side_p = int(batch.effect_expiry_side_presence[row, holder, i])
    if side_p == PRESENCE_NULL:
        expiry["side"] = None
    elif side_p == PRESENCE_VALUE:
        expiry["side"] = _side_from(batch.effect_expiry_side[row, holder, i])
    elif side_p != PRESENCE_ABSENT:
        raise DomainSoADecodeError("invalid expiry side presence")
    turn_p = int(batch.effect_expiry_turn_presence[row, holder, i])
    if turn_p == PRESENCE_NULL:
        expiry["turn"] = None
    elif turn_p == PRESENCE_VALUE:
        expiry["turn"] = int(batch.effect_expiry_turn[row, holder, i])
    elif turn_p != PRESENCE_ABSENT:
        raise DomainSoADecodeError("invalid expiry turn presence")
    remain_p = int(batch.effect_expiry_remaining_presence[row, holder, i])
    if remain_p == PRESENCE_NULL:
        expiry["remaining"] = None
    elif remain_p == PRESENCE_VALUE:
        expiry["remaining"] = int(batch.effect_expiry_remaining[row, holder, i])
    elif remain_p != PRESENCE_ABSENT:
        raise DomainSoADecodeError("invalid expiry remaining presence")
    return expiry


def _decode_flags_field(batch, row, holder, presence):
    code = int(presence)
    if code == PRESENCE_ABSENT:
        return _MISSING
    if code == PRESENCE_NULL:
        return None
    if code != PRESENCE_VALUE:
        raise DomainSoADecodeError("invalid flags presence")
    flags = {}
    for i, key in enumerate(FLAG_INT_FIELDS):
        _apply_optional(flags, key, _take_optional(
            batch.flag_int_presence[row, holder], batch.flag_int_value[row, holder], i, int,
            bool_keys=FLAG_BOOL_INTS, key=key))
    for i, key in enumerate(FLAG_BOOL_FIELDS):
        _apply_optional(flags, key, _take_optional(
            batch.flag_bool_presence[row, holder], batch.flag_bool_value[row, holder], i, lambda v: bool(v)))
    for i, key in enumerate(FLAG_STR_FIELDS):
        _apply_optional(flags, key, _take_optional(
            batch.flag_str_presence[row, holder], batch.flag_str_slot[row, holder], i,
            lambda v, _row=row: _load_string(batch, _row, v)))
    collapse_p = int(batch.flag_collapse_presence[row, holder])
    if collapse_p == PRESENCE_NULL:
        flags["collapse"] = None
    elif collapse_p == PRESENCE_VALUE:
        flags["collapse"] = {
            "side": _side_from(batch.flag_collapse_side[row, holder]),
            "until": int(batch.flag_collapse_until[row, holder]),
        }
    elif collapse_p != PRESENCE_ABSENT:
        raise DomainSoADecodeError("invalid collapse presence")
    exp_p = int(batch.flag_expiring_presence[row, holder])
    if exp_p == PRESENCE_NULL:
        flags["atk_buff_expiring"] = None
    elif exp_p == PRESENCE_VALUE:
        flags["atk_buff_expiring"] = {
            "amount": int(batch.flag_expiring_amount[row, holder]),
            "until": int(batch.flag_expiring_until[row, holder]),
        }
    elif exp_p != PRESENCE_ABSENT:
        raise DomainSoADecodeError("invalid atk_buff_expiring presence")
    timed_n = int(batch.flag_timed_count[row, holder])
    if timed_n:
        flags["timed_attack"] = [
            {"amount": int(batch.flag_timed_amount[row, holder, i]), "left": int(batch.flag_timed_left[row, holder, i])}
            for i in range(timed_n)
        ]
    mp=int(batch.flag_mimic_presence[row,holder])
    if mp==PRESENCE_VALUE:
        mimic={};ip=int(batch.flag_mimic_id_presence[row,holder])
        if ip==PRESENCE_NULL:mimic['id']=None
        elif ip==PRESENCE_VALUE:
            value=int(batch.flag_mimic_template[row,holder])
            if value==0:mimic['id']=''
            else:
                cid=next((cid for cid,index in CHARACTER_VOCAB.items()if index==value),None)
                if cid is None:raise DomainSoADecodeError('invalid mimicked character')
                mimic['id']=cid
        elif ip!=PRESENCE_ABSENT:raise DomainSoADecodeError('invalid mimic id presence')
        flags['mimic']=mimic
    elif mp!=PRESENCE_ABSENT:raise DomainSoADecodeError('invalid mimic presence')
    return flags


def _decode_front_debuff(batch, row, s):
    zone = {}
    delay_keys = ("left", "tick_side", "slow", "slow_turn", "by", "expires_side")
    burn_keys = ("left", "by", "stacks", "source", "applied_turn", "infinite")
    star_keys = ("left", "by", "source_cid", "renewals")
    weave_keys = ("by", "expires_side")
    stain_keys = ("by", "expires_side", "ends_left")
    specs = (delay_keys, burn_keys, star_keys, weave_keys, stain_keys)
    int_names = {"left", "slow", "slow_turn", "stacks", "applied_turn", "renewals", "ends_left"}
    side_names = {"tick_side", "by", "expires_side"}
    for z, key in enumerate(ZONE_STATUS_KEYS):
        code = int(batch.zone_status_presence[row, s, z])
        if code == PRESENCE_ABSENT:
            continue
        if code == PRESENCE_NULL:
            zone[key] = None
            continue
        if code != PRESENCE_VALUE:
            raise DomainSoADecodeError(f"invalid {key} presence")
        allowed = specs[z]
        mask = int(batch.zone_status_i32[row, s, z, 7])
        status = {}
        ints = iter(batch.zone_status_i32[row, s, z, :7])
        sides = iter(batch.zone_status_side[row, s, z])
        bools = iter(batch.zone_status_bool[row, s, z])
        slots = iter(batch.zone_status_slot[row, s, z])
        for name in allowed:
            if not (mask & (1 << allowed.index(name))):
                continue
            if name in int_names:
                status[name] = int(next(ints))
            elif name in side_names:
                status[name] = _side_from(next(sides))
            elif name == "infinite":
                status[name] = bool(next(bools))
            else:
                status[name] = _load_string(batch, row, next(slots))
        zone[key] = status
    dots = {}
    for i in range(batch.layout.max_dots):
        if not bool(batch.zone_dot_valid[row, s, i]):
            continue
        kind = _enum_from(batch.zone_dot_kind[row, s, i], DOT_KINDS, "dot kind")
        catalog = DOTS[kind]
        spec = {
            "name": catalog["name"],
            "stacks": int(batch.zone_dot_stacks[row, s, i]),
            "by": _side_from(batch.zone_dot_by[row, s, i]),
            "source": _load_string(batch, row, batch.zone_dot_source_slot[row, s, i]),
            "mode": catalog["mode"],
            "halve": bool(catalog.get("halve")),
            "ignore_shield": bool(catalog.get("ignore_shield")),
            "max_stacks": catalog["max_stacks"],
        }
        left_p = int(batch.zone_dot_left_presence[row, s, i])
        if left_p == PRESENCE_NULL:
            spec["left"] = None
        elif left_p == PRESENCE_VALUE:
            spec["left"] = int(batch.zone_dot_left[row, s, i])
        elif left_p != PRESENCE_ABSENT:
            raise DomainSoADecodeError("invalid dot left presence")
        turn_p = int(batch.zone_dot_applied_turn_presence[row, s, i])
        if turn_p == PRESENCE_NULL:
            spec["applied_turn"] = None
        elif turn_p == PRESENCE_VALUE:
            spec["applied_turn"] = int(batch.zone_dot_applied_turn[row, s, i])
        elif turn_p != PRESENCE_ABSENT:
            raise DomainSoADecodeError("invalid dot applied_turn presence")
        dots[kind] = spec
    if dots or bool(batch.zone_dots_presence[row, s]):
        zone["dots"] = dots
    return zone


def _card_from(cards, index, what):
    index = int(index)
    if index not in cards:
        raise DomainSoADecodeError(f"{what} card index {index} is missing")
    return cards[index]


def _decode_choice(batch, row, state, cards):
    side = _side_from(batch.choice_side[row])
    pending = {
        "side": side,
        "kind": _enum_from(batch.choice_kind[row], CHOICE_KINDS, "choice kind"),
        "prompt": _load_string(batch, row, batch.choice_prompt_slot[row]),
        "cards": [_card_from(cards, batch.choice_card_index[row, i], "choice") for i in range(int(batch.choice_card_count[row]))],
    }
    for i, key in enumerate(CHOICE_INT_KEYS):
        _apply_optional(pending, key, _take_optional(batch.choice_int_presence[row], batch.choice_int_value[row], i, int))
    for i, key in enumerate(CHOICE_BOOL_KEYS):
        _apply_optional(pending, key, _take_optional(batch.choice_bool_presence[row], batch.choice_bool_value[row], i, lambda v: bool(v)))
    for i, key in enumerate(CHOICE_STR_KEYS):
        _apply_optional(pending, key, _take_optional(
            batch.choice_str_presence[row], batch.choice_str_slot[row], i,
            lambda v, _row=row: _load_string(batch, _row, v)))
    actor_p = int(batch.choice_actor_presence[row])
    if actor_p == PRESENCE_NULL:
        pending["actor"] = None
    elif actor_p == PRESENCE_VALUE:
        pending["actor"] = list(state["sides"][side]["characters"])[int(batch.choice_actor_slot[row])]
    elif actor_p != PRESENCE_ABSENT:
        raise DomainSoADecodeError("invalid choice actor presence")
    cont_p = int(batch.choice_continuation_presence[row])
    if cont_p == PRESENCE_NULL:
        pending["continuation"] = None
    elif cont_p == PRESENCE_VALUE:
        pending["continuation"] = _load_string(batch, row, batch.choice_continuation_slot[row])
    elif cont_p != PRESENCE_ABSENT:
        raise DomainSoADecodeError("invalid choice continuation presence")
    return pending


def _decode_operation(batch, row, state, cards):
    side = _side_from(batch.op_side[row])
    order = list(state["sides"][side]["characters"])
    operation = {
        "side": side,
        "actor": order[int(batch.op_actor_slot[row])],
        "first_bonus": int(batch.op_first_bonus[row]),
        "redeem_bonus": int(batch.op_redeem_bonus[row]),
        "record_candidate": bool(batch.op_record_candidate[row]),
        "participants": [
            [
                _side_from(batch.op_participant_side[row, i]),
                list(state["sides"][_side_from(batch.op_participant_side[row, i])]["characters"])[int(batch.op_participant_slot[row, i])],
            ]
            for i in range(int(batch.op_participant_count[row]))
        ],
    }
    card_index = int(batch.op_card_index[row])
    operation["card"] = None if card_index < 0 else _card_from(cards, card_index, "operation")
    target_p = int(batch.op_target_presence[row])
    if target_p == PRESENCE_NULL:
        operation["target_id"] = None
    elif target_p == PRESENCE_VALUE:
        operation["target_id"] = _load_string(batch, row, batch.op_target_slot[row])
    elif target_p != PRESENCE_ABSENT:
        raise DomainSoADecodeError("invalid operation target presence")
    harm_p = int(batch.op_harmony_presence[row])
    if harm_p == PRESENCE_NULL:
        operation["harmony"] = None
    elif harm_p == PRESENCE_VALUE:
        operation["harmony"] = _load_string(batch, row, batch.op_harmony_slot[row])
    elif harm_p != PRESENCE_ABSENT:
        raise DomainSoADecodeError("invalid operation harmony presence")
    for i, key in enumerate(OP_FLAG_KEYS):
        _apply_optional(operation, key, _take_optional(batch.op_flag_presence[row], batch.op_flag_value[row], i, lambda v: bool(v)))
    for i, key in enumerate(OP_INT_KEYS):
        _apply_optional(operation, key, _take_optional(batch.op_int_presence[row], batch.op_int_value[row], i, int))
    for i, key in enumerate(OP_STR_KEYS):
        _apply_optional(operation, key, _take_optional(batch.op_str_presence[row], batch.op_str_slot[row], i, lambda slot: _load_string(batch, row, slot)))
    pending_p = int(batch.op_pending_sortie_presence[row])
    if pending_p == PRESENCE_VALUE:
        spec = {}
        bonus_p = int(batch.op_pending_bonus_presence[row])
        if bonus_p == PRESENCE_VALUE:
            spec['bonus'] = int(batch.op_pending_bonus[row])
        elif bonus_p != PRESENCE_ABSENT:
            raise DomainSoADecodeError('invalid pending_sortie bonus presence')
        operation['pending_sortie'] = {'spec':spec,'stay_bench':bool(batch.op_pending_stay_bench[row])}
    elif pending_p != PRESENCE_ABSENT:
        raise DomainSoADecodeError('invalid pending_sortie presence')
    opt_p = int(batch.op_option_id_presence[row])
    if opt_p == PRESENCE_NULL:
        operation["option_id"] = None
    elif opt_p == PRESENCE_VALUE:
        operation["option_id"] = _load_string(batch, row, batch.op_option_id_slot[row])
    elif opt_p != PRESENCE_ABSENT:
        raise DomainSoADecodeError("invalid option_id presence")
    last_p = int(batch.op_last_attack_target_presence[row])
    if last_p == PRESENCE_NULL:
        operation["last_attack_target"] = None
    elif last_p == PRESENCE_VALUE:
        packed = int(batch.op_last_attack_target_slot[row])
        last_side = _side_from(packed // 256)
        operation["last_attack_target"] = [last_side, "player" if packed % 256 == 255 else list(state["sides"][last_side]["characters"])[packed % 256]]
    elif last_p != PRESENCE_ABSENT:
        raise DomainSoADecodeError("invalid last_attack_target presence")
    mark_p = int(batch.op_card_mark_presence[row])
    if mark_p == PRESENCE_NULL:
        operation["card_mark"] = None
    elif mark_p == PRESENCE_VALUE:
        operation["card_mark"] = {
            "delta": int(batch.op_card_mark_delta[row]),
            "source_entity_id": _entity_text(batch.op_card_mark_source_slot[row]),
        }
    elif mark_p != PRESENCE_ABSENT:
        raise DomainSoADecodeError("invalid card_mark presence")
    choice_n = int(batch.op_choice_card_count[row])
    if choice_n:
        operation["choice_cards"] = [_card_from(cards, batch.op_choice_card_index[row, i], "op choice") for i in range(choice_n)]
    return operation


def _decode_deferred(batch, row, state, count):
    items = []
    for i in range(count):
        side = _side_from(batch.deferred_side[row, i])
        items.append({
            "side": side,
            "actor": list(state["sides"][side]["characters"])[int(batch.deferred_actor_slot[row, i])],
            "entity_id": _entity_text(batch.deferred_entity_slot[row, i]),
            "kind": _load_string(batch, row, batch.deferred_kind_slot[row, i]),
        })
    return items
