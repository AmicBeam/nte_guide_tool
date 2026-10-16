"""Strict initial-state domain SoA codec for public-17 mulligan snapshots.

This is not the graph codec in codec.py, not a complete public-17 encoder, and
not a full duel SoA. Catalog templates supply static nests; mutable numbers live
in dedicated schema columns. Choice, operation, deferred, effects, and nonempty
flags are rejected explicitly.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

import numpy as np

from app.modules.card_game.content.duel_v2.tutorial import lookup_card, lookup_character
from app.modules.card_game.engine.duel_v2.entities import CardEntity, CharacterEntity, PlayerEntity
from app.modules.card_game.engine.duel_v2.state import new_character

from .domain_soa_schema import (
    CARD_BOOL_FIELDS,
    CARD_INT_FIELDS,
    CARD_KEYS,
    CARD_STR_FIELDS,
    CARD_VOCAB,
    CHARACTER_BOOL_FIELDS,
    CHARACTER_INT_FIELDS,
    CHARACTER_KEYS,
    CHARACTER_STR_FIELDS,
    CHARACTER_VOCAB,
    COVERAGE,
    DomainSoABatch,
    DomainSoACapacityError,
    DomainSoADecodeError,
    DomainSoALayout,
    DomainSoAUnsupportedError,
    GAME_KEYS,
    GAME_PHASES,
    NON_AUTHORITATIVE_GAME_KEYS,
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
    ZONES,
    allocate_arrays,
    ARRAY_NAMES,
    catalog_fingerprint,
    catalog_card,
    side_index,
    vocab_get,
)

supports_full_duel_rules = False
INT32_MIN, INT32_MAX = -2147483648, 2147483647
RNG_MAX = (1 << 63) - 1
_CHAR_FROM_VOCAB = {index: cid for cid, index in CHARACTER_VOCAB.items()}
_CARD_FROM_VOCAB = {index: card_id for card_id, index in CARD_VOCAB.items()}
_MISSING = object()
_INITIAL_FORBIDDEN = (
    "deferred_actions",
    "turn_start_pending",
    "_response_lock",
    "_response_intercept",
    "_draining_actions",
    "_pending_down_responses",
    "_deferred_card_marks",
)


def coverage():
    return {
        "supports_full_duel_rules": False,
        "complete_soa": False,
        "full_public17": False,
        "initial_mulligan": True,
        "seven_official_presets": COVERAGE["seven_official_presets"],
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
        arrays['catalog_digest'][row] = fingerprint
    batch = DomainSoABatch(layout=layout, **arrays)
    # Initial-state coverage is deliberately narrow. Never accept a known but
    # unimplemented field and silently discard it during packing.
    for row, state in enumerate(rows):
        expected = {key: value for key, value in state.items()
                    if key not in NON_AUTHORITATIVE_GAME_KEYS}
        restored = _decode_row(batch, row)
        if not _same(expected, restored):
            raise DomainSoAUnsupportedError('Initial codec cannot represent every supplied field/type')
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
            raise DomainSoADecodeError('Failed row or catalog binding mismatch')
        if not 0 <= batch.string_count[row] <= batch.layout.max_string_slots:
            raise DomainSoADecodeError('String count outside layout')
        if not 0 <= batch.string_bytes_used[row] <= batch.layout.max_string_bytes:
            raise DomainSoADecodeError('String bytes outside layout')
        if not 0 <= batch.mulligan_count[row] <= 2:
            raise DomainSoADecodeError('Mulligan side count outside layout')
        state = _decode_row(batch, row)
        canonical = encode_domain_states([state], batch.layout, presentation='drop')
        if any(not np.array_equal(getattr(batch, name)[row], getattr(canonical, name)[0])
               for name in ARRAY_NAMES):
            raise DomainSoADecodeError('Noncanonical or unsupported packed state columns')
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
        raise DomainSoAUnsupportedError("non-catalog string is outside initial SoA coverage")


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
                raise DomainSoAUnsupportedError('Initial codec cannot preserve these mutable aliases')
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
        raise DomainSoAUnsupportedError(f"non-initial {what} fields {present}")


def _empty_map(value, what):
    if type(value) is not dict:
        raise TypeError(f"{what} must be a dict")
    if value:
        raise DomainSoAUnsupportedError(f"non-initial nonempty {what}")


def _empty_list(value, what):
    if type(value) is not list:
        raise TypeError(f"{what} must be a list")
    if value:
        raise DomainSoAUnsupportedError(f"non-initial nonempty {what}")


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


def _put_optional_int(container, key, presence, values, index, lo=INT32_MIN, hi=INT32_MAX, catalog=None):
    if key not in container:
        presence[index] = PRESENCE_ABSENT
        return
    value = container[key]
    if value is None:
        presence[index] = PRESENCE_NULL
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
    if catalog is not None and key in catalog:
        if not _same(value, catalog[key]):
            raise DomainSoAUnsupportedError(f"mutated catalog field {key}")
        presence[index] = PRESENCE_CATALOG
        return
    if not allow_value:
        raise DomainSoAUnsupportedError(f"non-catalog string {key} is outside initial SoA coverage")
    presence[index] = PRESENCE_VALUE
    slots[index] = strings.intern(value)


def _take_optional(presence, values, index, caster, catalog=None, key=None):
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
    return caster(values[index])


def _apply_optional(target, key, value):
    if value is _MISSING:
        target.pop(key, None)
    else:
        target[key] = value


def _null_or_absent(state, key):
    if key not in state:
        return PRESENCE_ABSENT
    if state[key] is None:
        return PRESENCE_NULL
    raise DomainSoAUnsupportedError(f"non-initial {key} is outside initial SoA coverage")


def _side_from(value):
    index = int(value)
    if index not in (0, 1):
        raise DomainSoADecodeError(f"invalid side index {index}")
    return SIDES[index]


def _encode_row(state, arrays, row, layout, presentation):
    _unknown(state, GAME_KEYS, "game")
    _forbid(state, UNSUPPORTED_GAME_KEYS, "unsupported")
    _forbid(state, _INITIAL_FORBIDDEN, "deferred/lock")
    for key in NON_AUTHORITATIVE_GAME_KEYS:
        if key in state and presentation == "reject":
            raise DomainSoAUnsupportedError(
                f"presentation key {key!r} requires presentation='drop'"
            )
    if state.get("phase") != "mulligan":
        raise DomainSoAUnsupportedError(
            f"non-initial phase {state.get('phase')!r}; full SoA is incomplete"
        )
    if "pending_choice" not in state or state["pending_choice"] is not None:
        raise DomainSoAUnsupportedError("pending_choice is outside initial SoA coverage")
    if "operation" not in state or state["operation"] is not None:
        raise DomainSoAUnsupportedError("operation is outside initial SoA coverage")
    strings = _Strings(layout)
    arrays["version"][row] = _as_int("version", state["version"])
    arrays["phase"][row] = GAME_PHASES.index("mulligan")
    arrays["active_side"][row] = side_index(state["active_side"])
    arrays["first_side"][row] = side_index(state["first_side"])
    arrays["turn"][row] = _as_int("turn", state["turn"])
    arrays["winner_presence"][row] = _null_or_absent(state, "winner")
    arrays["reason_presence"][row] = _null_or_absent(state, "reason")
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
    arrays["pending_choice_present"][row] = False
    arrays["operation_present"][row] = False
    done = state["mulligan_done"]
    if type(done) is not list:
        raise TypeError("mulligan_done must be a list")
    if len(done) > 2:
        raise DomainSoACapacityError("mulligan_done exceeds two sides")
    arrays["mulligan_count"][row] = len(done)
    for i, item in enumerate(done):
        arrays["mulligan_sides"][row, i] = side_index(item)
    arrays["turn_start_pending_presence"][row] = PRESENCE_ABSENT
    arrays["response_lock_presence"][row] = PRESENCE_ABSENT
    arrays["response_intercept_presence"][row] = PRESENCE_ABSENT
    arrays["draining_presence"][row] = PRESENCE_ABSENT
    sides = state["sides"]
    if not isinstance(sides, Mapping) or set(sides) != set(SIDES):
        raise DomainSoAUnsupportedError("sides must contain exactly 'a' and 'b'")
    identities = set()
    instances = set()
    for team in sides.values():
        entities = [(team, PlayerEntity)] + [(h, CharacterEntity) for h in team['characters'].values()]
        entities += [(card, CardEntity) for zone in ZONES for card in team[zone]]
        for entity, expected_type in entities:
            if type(entity) is not expected_type:
                raise DomainSoAUnsupportedError('Initial codec requires hydrated formal entity classes')
            eid = entity.get('entity_id')
            if eid in identities:
                raise DomainSoAUnsupportedError('Duplicate entity identity or multiple physical zone occupancy')
            identities.add(eid)
            if expected_type is CardEntity:
                instance = entity.get('instance_id')
                if instance in instances:
                    raise DomainSoAUnsupportedError('Duplicate physical card instance identity')
                instances.add(instance)
    card_n = 0
    for s, side in enumerate(SIDES):
        card_n = _encode_player(sides[side], arrays, row, s, layout, strings, card_n)
    strings.dump(arrays, row)


def _encode_player(team, arrays, row, s, layout, strings, card_n):
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
    arrays["player_front_presence"][row, s] = _null_or_absent(team, "front")
    arrays["player_last_front_presence"][row, s] = _null_or_absent(team, "last_front")
    order = team["order"]
    characters = team["characters"]
    if type(order) is not list or not isinstance(characters, Mapping):
        raise TypeError("player order/characters must be list/dict")
    if len(order) > layout.max_characters:
        raise DomainSoACapacityError("character order exceeds layout.max_characters")
    if set(order) != set(characters) or len(order) != len(characters):
        raise DomainSoAUnsupportedError("initial codec requires lineup characters to match order")
    arrays["player_order_count"][row, s] = len(order)
    for i, key in enumerate(PLAYER_INT_FIELDS):
        _put_optional_int(
            team, key, arrays["player_int_presence"][row, s], arrays["player_int_value"][row, s], i
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
    arrays["player_used_count"][row, s] = 0
    _empty_map(team["used"], "used")
    _empty_map(team["harmonized"], "harmonized")
    if "attacked_this_turn" in team:
        raise DomainSoAUnsupportedError("attacked_this_turn is outside initial SoA coverage")
    arrays["player_attacked_this_turn_presence"][row, s] = PRESENCE_ABSENT
    _empty_list(team["records"], "records")
    arrays["player_records_count"][row, s] = 0
    arrays["player_effects_presence"][row, s] = _presence_of_empty(team, "effects", "player effects")
    arrays["player_flags_presence"][row, s] = _presence_of_empty(team, "flags", "player flags")
    _empty_map(team["front_debuff"], "front_debuff")
    for i, cid in enumerate(order):
        _encode_character(characters[cid], cid, arrays, row, s, i)
    for z, zone in enumerate(ZONES):
        cards = team[zone]
        if type(cards) is not list:
            raise TypeError(f"{zone} must be a list")
        if len(cards) > layout.max_zone:
            raise DomainSoACapacityError(f"{zone} exceeds layout.max_zone")
        arrays["zone_count"][row, s, z] = len(cards)
        for pos, card in enumerate(cards):
            if card_n >= layout.max_cards:
                raise DomainSoACapacityError("card table exceeds layout.max_cards")
            _encode_card(card, SIDES[s], arrays, row, card_n)
            arrays["zone_card_index"][row, s, z, pos] = card_n
            card_n += 1
    return card_n


def _encode_character(hero, cid, arrays, row, s, slot):
    _unknown(hero, CHARACTER_KEYS, "character")
    if hero.get("id", cid) != cid:
        raise DomainSoAUnsupportedError("character id does not match lineup")
    if hero.get("summoned") is True:
        raise DomainSoAUnsupportedError("summons are outside initial SoA coverage")
    if hero.get("shape") not in (None, _MISSING) and "shape" in hero and hero["shape"] is not None:
        raise DomainSoAUnsupportedError("equipped shape is outside initial SoA coverage")
    _forbid(hero, ("equipment", "turn_snapshots"), "character")
    catalog = lookup_character(cid)
    vocab_get(CHARACTER_VOCAB, cid, what="character")
    arrays["character_valid"][row, s, slot] = True
    arrays["character_entity_slot"][row, s, slot] = _parse_entity_num(hero["entity_id"])
    arrays["character_template"][row, s, slot] = CHARACTER_VOCAB[cid]
    arrays["character_hp"][row, s, slot] = _as_int("hp", hero["hp"])
    arrays["character_shield"][row, s, slot] = _as_int("shield", hero["shield"])
    arrays["character_energy"][row, s, slot] = _as_int("energy", hero["energy"])
    arrays["character_hp_presence"][row, s, slot] = PRESENCE_VALUE
    arrays["character_shield_presence"][row, s, slot] = PRESENCE_VALUE
    arrays["character_energy_presence"][row, s, slot] = PRESENCE_VALUE
    for i, key in enumerate(CHARACTER_INT_FIELDS):
        _put_optional_int(
            hero, key, arrays["character_int_presence"][row, s, slot],
            arrays["character_int_value"][row, s, slot], i, catalog=catalog,
        )
    for i, key in enumerate(CHARACTER_BOOL_FIELDS):
        _put_optional_bool(
            hero, key, arrays["character_bool_presence"][row, s, slot],
            arrays["character_bool_value"][row, s, slot], i, catalog=catalog,
        )
    for i, key in enumerate(CHARACTER_STR_FIELDS):
        _put_optional_str(
            hero, key, arrays["character_str_presence"][row, s, slot],
            arrays["character_str_slot"][row, s, slot], i, _RejectStrings(), catalog=catalog,
        )
    arrays["character_effects_presence"][row, s, slot] = _presence_of_empty(hero, "effects", "character effects")
    arrays["character_flags_presence"][row, s, slot] = _presence_of_empty(hero, "flags", "character flags")
    arrays["character_ultimate_presence"][row, s, slot] = _catalog_nest(hero, catalog, "ultimate")
    arrays["character_mechanisms_presence"][row, s, slot] = _catalog_nest(hero, catalog, "mechanisms")
    arrays["character_equipment_presence"][row, s, slot] = PRESENCE_ABSENT
    _encode_stat_layer(hero, catalog, arrays, row, s, slot)
    if hero.get("side") not in (None, SIDES[s]):
        raise DomainSoAUnsupportedError("character side does not match slot")


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


def _encode_card(card, side, arrays, row, index):
    _unknown(card, CARD_KEYS, "card")
    card_id = card["card_id"]
    catalog = lookup_card(card_id) or catalog_card(card_id)
    vocab_get(CARD_VOCAB, card_id, what="card")
    if card.get("copy") is True:
        raise DomainSoAUnsupportedError("copied cards are outside initial SoA coverage")
    if "jingu_mark" in card:
        raise DomainSoAUnsupportedError("jingu_mark is outside initial SoA coverage")
    arrays["card_valid"][row, index] = True
    arrays["card_entity_slot"][row, index] = _parse_entity_num(card["entity_id"])
    arrays["card_instance_slot"][row, index] = _parse_instance_num(card["instance_id"], side)
    arrays["card_template"][row, index] = CARD_VOCAB[card_id]
    arrays["card_side"][row, index] = side_index(side)
    if card.get("side") not in (None, side):
        raise DomainSoAUnsupportedError("card side does not match zone")
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
            arrays["card_str_slot"][row, index], i, _RejectStrings(), catalog=catalog,
        )
    arrays["card_source_presence"][row, index] = _catalog_nest(card, catalog, "source")
    arrays["card_play_options_presence"][row, index] = _catalog_nest(card, catalog, "play_options")
    arrays["card_jingu_mark_presence"][row, index] = PRESENCE_ABSENT


def _decode_row(batch, row):
    if int(batch.pending_choice_present[row]) or int(batch.operation_present[row]):
        raise DomainSoAUnsupportedError("packed choice/operation is outside initial SoA coverage")
    if int(batch.deferred_count[row]) or int(batch.down_response_count[row]):
        raise DomainSoAUnsupportedError("packed deferred fields are outside initial SoA coverage")
    phase_i = int(batch.phase[row])
    if phase_i < 0 or phase_i >= len(GAME_PHASES) or GAME_PHASES[phase_i] != "mulligan":
        raise DomainSoAUnsupportedError("packed non-initial phase; full SoA is incomplete")
    if int(batch.winner_presence[row]) not in (PRESENCE_ABSENT, PRESENCE_NULL):
        raise DomainSoAUnsupportedError("packed winner is outside initial SoA coverage")
    if int(batch.reason_presence[row]) not in (PRESENCE_ABSENT, PRESENCE_NULL):
        raise DomainSoAUnsupportedError("packed reason is outside initial SoA coverage")
    rng = int(batch.rng[row])
    if rng < 0 or rng > RNG_MAX:
        raise DomainSoADecodeError("rng is outside the 63-bit stream")
    state = {
        "rules_version": _load_string(batch, row, batch.rules_version_slot[row]),
        "design_version": _load_string(batch, row, batch.design_version_slot[row]),
        "version": int(batch.version[row]),
        "phase": "mulligan",
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
    if int(batch.winner_presence[row]) == PRESENCE_NULL:
        state["winner"] = None
    if int(batch.reason_presence[row]) == PRESENCE_NULL:
        state["reason"] = None
    cards = _decode_cards(batch, row)
    for s, side in enumerate(SIDES):
        if not bool(batch.player_valid[row, s]):
            raise DomainSoADecodeError(f"missing player {side}")
        state["sides"][side] = _decode_player(batch, row, s, side, cards)
    return state


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
                lambda v: _load_string(batch, row, v), catalog, key))
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
        if int(batch.card_jingu_mark_presence[row, index]) != PRESENCE_ABSENT:
            raise DomainSoAUnsupportedError("jingu_mark is outside initial SoA coverage")
        cards[index] = CardEntity(payload, entity_id="e" + str(int(batch.card_entity_slot[row, index])), side=side)
    return cards


def _decode_player(batch, row, s, side, cards):
    order_n = int(batch.player_order_count[row, s])
    if order_n < 1 or order_n > batch.layout.max_characters:
        raise DomainSoADecodeError("invalid player order count")
    order, characters = [], {}
    for slot in range(order_n):
        if not bool(batch.character_valid[row, s, slot]):
            raise DomainSoADecodeError("missing character in order")
        hero, cid = _decode_character(batch, row, s, slot, side)
        order.append(cid)
        characters[cid] = hero
    payload = {
        "name": _load_string(batch, row, batch.player_name_slot[row, s]),
        "hp": int(batch.player_hp[row, s]),
        "max_hp": int(batch.player_max_hp[row, s]),
        "shield": int(batch.player_shield[row, s]),
        "ap": int(batch.player_ap[row, s]),
        "order": order,
        "characters": characters,
    }
    front = _decode_nullable_ref(batch.player_front_presence[row, s], "front")
    last_front = _decode_nullable_ref(batch.player_last_front_presence[row, s], "last_front")
    if front is not _MISSING:
        payload["front"] = front
    if last_front is not _MISSING:
        payload["last_front"] = last_front
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
    payload["records"] = []
    for i, key in enumerate(PLAYER_INT_FIELDS):
        _apply_optional(payload, key, _take_optional(
            batch.player_int_presence[row, s], batch.player_int_value[row, s], i, int))
    for i, key in enumerate(PLAYER_BOOL_FIELDS):
        _apply_optional(payload, key, _take_optional(
            batch.player_bool_presence[row, s], batch.player_bool_value[row, s], i, lambda v: bool(v)))
    for i, key in enumerate(PLAYER_STR_FIELDS):
        _apply_optional(payload, key, _take_optional(
            batch.player_str_presence[row, s], batch.player_str_slot[row, s], i,
            lambda v: _load_string(batch, row, v)))
    payload["used"] = {}
    payload["harmonized"] = {}
    payload["front_debuff"] = {}
    if int(batch.player_used_count[row, s]) or int(batch.player_records_count[row, s]):
        raise DomainSoAUnsupportedError("used/records are outside initial SoA coverage")
    if int(batch.player_effects_presence[row, s]) not in (PRESENCE_ABSENT, PRESENCE_VALUE):
        raise DomainSoAUnsupportedError("player effects are outside initial SoA coverage")
    if int(batch.player_flags_presence[row, s]) not in (PRESENCE_ABSENT, PRESENCE_VALUE):
        raise DomainSoAUnsupportedError("player flags are outside initial SoA coverage")
    if int(batch.player_attacked_this_turn_presence[row, s]) != PRESENCE_ABSENT:
        raise DomainSoAUnsupportedError("attacked_this_turn is outside initial SoA coverage")
    return PlayerEntity(payload, entity_id="e" + str(int(batch.player_entity_slot[row, s])), side=side)


def _decode_nullable_ref(presence, what):
    code = int(presence)
    if code == PRESENCE_ABSENT:
        return _MISSING
    if code == PRESENCE_NULL:
        return None
    raise DomainSoAUnsupportedError(f"packed {what} occupant is outside initial SoA coverage")


def _decode_character(batch, row, s, slot, side):
    template = int(batch.character_template[row, s, slot])
    cid = _CHAR_FROM_VOCAB.get(template)
    if cid is None:
        raise DomainSoADecodeError(f"unknown character template {template}")
    catalog = lookup_character(cid)
    entity_id = "e" + str(int(batch.character_entity_slot[row, s, slot]))
    hero = new_character(cid, entity_id=entity_id, side=side)
    if int(batch.character_hp_presence[row, s, slot]) != PRESENCE_VALUE:
        raise DomainSoADecodeError("character hp must be a dedicated value")
    hero["hp"] = int(batch.character_hp[row, s, slot])
    hero["shield"] = int(batch.character_shield[row, s, slot])
    hero["energy"] = int(batch.character_energy[row, s, slot])
    for i, key in enumerate(CHARACTER_INT_FIELDS):
        _apply_optional(hero, key, _take_optional(
            batch.character_int_presence[row, s, slot], batch.character_int_value[row, s, slot], i, int, catalog, key))
    for i, key in enumerate(CHARACTER_BOOL_FIELDS):
        _apply_optional(hero, key, _take_optional(
            batch.character_bool_presence[row, s, slot], batch.character_bool_value[row, s, slot], i,
            lambda v: bool(v), catalog, key))
    for i, key in enumerate(CHARACTER_STR_FIELDS):
        _apply_optional(hero, key, _take_optional(
            batch.character_str_presence[row, s, slot], batch.character_str_slot[row, s, slot], i,
            lambda v: _load_string(batch, row, v), catalog, key))
    if int(batch.character_effects_presence[row, s, slot]) not in (PRESENCE_ABSENT, PRESENCE_VALUE):
        raise DomainSoAUnsupportedError("character effects are outside initial SoA coverage")
    flags_p = int(batch.character_flags_presence[row, s, slot])
    if flags_p == PRESENCE_ABSENT:
        hero.pop("flags", None)
    elif flags_p == PRESENCE_VALUE:
        hero["flags"] = {}
    else:
        raise DomainSoAUnsupportedError("character flags are outside initial SoA coverage")
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
    if int(batch.character_equipment_presence[row, s, slot]) != PRESENCE_ABSENT:
        raise DomainSoAUnsupportedError("equipment is outside initial SoA coverage")
    if int(batch.holder_effect_count[row, batch.layout.holder_index(s, slot)]):
        raise DomainSoAUnsupportedError("packed effects are outside initial SoA coverage")
    return hero, cid


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
