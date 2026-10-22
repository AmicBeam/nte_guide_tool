"""Typed per-field domain SoA layout for public-17 duel_v2 rule state.

This is not the graph codec in schema.py / codec.py. Each authoritative field
is a dedicated NumPy array with an exact dtype, a presence code where
absent / null / zero must be distinguished, and a validity mask. Nested Python
dicts are never stored as blobs inside tensors.

Coverage is explicit: supports_full_duel_rules is False. Presentation logs,
time-rewind snapshots, tutorial, non-public characters, mutated catalog nests,
and unknown future keys are out of scope and must be rejected (or dropped only
when the caller names a non-authoritative class).

Offline host usage:

    layout = DomainSoALayout()
    batch = encode_domain_states(states, layout, presentation="drop")
    restored = decode_domain_states(batch)
    fork = batch.clone_rows([0, 2])
    # Torch is imported only by DomainSoABatch.to_torch; CPU NumPy is the
    # ingress/egress form and is not a device rule engine.

Dedicated vitals are always character_hp/shield/energy with shape [B, 2, C]
and dtype int32. Player vitals are [B, 2].
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Sequence

import numpy as np

from app.modules.card_game.content.duel_v2.catalog import (
    CARDS,
    CHARACTERS,
    PUBLIC_CHARACTER_IDS,
)


class DomainSoAError(Exception):
    """Base error for the public-17 domain SoA codec."""


class DomainSoACapacityError(DomainSoAError):
    """A typed table exceeded its frozen layout bound. Never truncated."""


class DomainSoAUnsupportedError(DomainSoAError, TypeError):
    """State is outside the declared public-17 SoA coverage."""


class DomainSoADecodeError(DomainSoAError, ValueError):
    """Packed arrays are corrupt, out of range, or disagree with the layout."""


supports_full_duel_rules = False

PRESENCE_ABSENT = 0
PRESENCE_NULL = 1
PRESENCE_VALUE = 2
PRESENCE_CATALOG = 3

SIDES: tuple[str, str] = ("a", "b")
GAME_PHASES: tuple[str, ...] = ("mulligan", "playing", "choice", "finished")
WINNERS: tuple[str, ...] = ("a", "b", "draw")
ZONES: tuple[str, ...] = ("hand", "deck", "discard", "removed")
CHOICE_KINDS: tuple[str, ...] = (
    "discard",
    "inspect_top",
    "enemy_hand",
    "kit",
    "kit_multi",
    "hand_redraw",
    "deck_order",
    "inspect_deck",
    "heiyu_divination",
    "hand_deck_placement",
)
CHOICE_INT_KEYS: tuple[str, ...] = (
    "max_select",
    "min_select",
    "max_bottom",
    "min_bottom",
    "snapshot_n",
)
CHOICE_BOOL_KEYS: tuple[str, ...] = (
    "multi_bottom",
    "hand_placement",
    "execute_divination",
    "deck_cards",
    "deck_order",
)
CHOICE_STR_KEYS: tuple[str, ...] = (
    "location_id",
    "source_instance_id",
    "scope",
)
OP_FLAG_KEYS: tuple[str, ...] = (
    "release_only",
    "additional_attack",
    "normal_attack",
    "skip_combat_harmony",
    "support",
    "support_resolved",
    "pending_sortie",
    "record_candidate_unused",
)
OP_INT_KEYS: tuple[str, ...] = ("ap_paid", "card_bonus", "card_set_attack", "last_attack_amount")
ZONE_STATUS_KEYS: tuple[str, ...] = ("delay", "burn", "star", "weave", "stain")
LINGKE_CHAIN_KEYS: tuple[str, ...] = ("entity_id", "character_id", "count")
EFFECT_OPS: tuple[str, ...] = ("add", "multiply", "set", "allow", "forbid")
EFFECT_SCOPES: tuple[str, ...] = ("self", "allies", "other_allies", "enemies", "all")
EFFECT_KINDS: tuple[str, ...] = ("persistent", "aura")
EFFECT_VISIBILITY: tuple[str, ...] = ("public", "private")
EFFECT_PHASES: tuple[str, ...] = ("turn_start", "turn_end")
DOT_KINDS: tuple[str, ...] = ("nightmare", "etch", "venom", "guard")
EFFECT_CLEAR_EVENTS: tuple[str, ...] = (
    "down",
    "operation_end",
    "turn_start",
    "turn_end",
    "attack",
    "card_use",
)

PUBLIC_CHARACTER_ORDER: tuple[str, ...] = tuple(sorted(PUBLIC_CHARACTER_IDS))
PUBLIC_CARD_ORDER: tuple[str, ...] = tuple(
    sorted(
        card_id
        for card_id, spec in CARDS.items()
        if spec.get("character_id") in PUBLIC_CHARACTER_IDS
    )
)
CHARACTER_VOCAB: MappingProxyType[str, int] = MappingProxyType(
    {cid: i + 1 for i, cid in enumerate(PUBLIC_CHARACTER_ORDER)}
)
CARD_VOCAB: MappingProxyType[str, int] = MappingProxyType(
    {cid: i + 1 for i, cid in enumerate(PUBLIC_CARD_ORDER)}
)

PLAYER_INT_FIELDS: tuple[str, ...] = (
    "fatigue",
    "turn_count",
    "extra_ap",
    "delay_turns",
    "delay_counted_turn",
    "genesis_player_hits",
    "genesis_turn",
    "genesis_damage_bonus",
    "nanali_family",
    "nanali_family_played",
    "ultimates_used",
    "heiyu_divinations",
    "dark_star_count",
    "insight_bonus",
    "insight_damage",
    "insight_played",
    "support_count",
    "canhong_collapse_used",
    "support_turn",
    "effect_seq",
)
PLAYER_BOOL_FIELDS: tuple[str, ...] = (
    "normal_attack_available",
    "ultimate_available",
    "extra_flower",
    "extra_delay",
    "surplus",
    "harmony_damage",
    "surplus_ever",
    "empty_draw_wins",
    "skip_normal_attack",
    "extra_genesis_pending",
    "life_locked",
    "harmony_available",
)
PLAYER_STR_FIELDS: tuple[str, ...] = ("extra_genesis_actor",)

CHARACTER_INT_FIELDS: tuple[str, ...] = (
    "attack",
    "max_hp",
    "energy_max",
    "harmony",
    "ultimate_turns",
    "growth",
    "family",
    "gaze",
    "dream",
    "down_turns",
    "jingu",
    "effect_seq",
    "equipment_seq",
    "ultimate_used_turn",
    "insight_used_turn",
    "heal_draw_turn",
    "everness_id",
    "energy_max_override",
    "beast_fangs",
    "base_attack",
    "base_max_hp",
)
CHARACTER_BOOL_FIELDS: tuple[str, ...] = ("awakened", "summoned", "jingu_loan", "disappeared")
CHARACTER_STR_FIELDS: tuple[str, ...] = (
    "name",
    "attribute",
    "avatar",
    "portrait",
    "passive",
    "awakened_passive",
    "faction",
    "shape",
    "last_hit_source",
    "flavor",
)

CARD_INT_FIELDS: tuple[str, ...] = (
    "cost",
    "attack",
    "shield",
    "hp",
    "starter_copies",
    "expires_turn",
    "ephemeral_turn",
    "spend_all_ap",
    "additional_ap_cost",
    "enemy_turn_attack",
)
CARD_BOOL_FIELDS: tuple[str, ...] = (
    "copy",
    "terminal",
    "instant",
    "derived",
    "retain",
    "playable_downed",
    "training_excluded",
)
CARD_STR_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "type",
    "description",
    "effect_id",
    "character_id",
    "original_character_id",
    "response",
    "response_faction",
    "require",
    "attack_mode",
    "tooltip",
)

FLAG_INT_FIELDS: tuple[str, ...] = (
    "next_bonus",
    "next_shield",
    "next_followup",
    "energy_next",
    "allied_hurt",
    "pending_atk",
    "share_overflow",
    "hp_floor",
    "collapse_count",
    "collapse_max",
    "pact",
    "mimic",
    "atk_buff",
    "next_attack",
    "form_attack",
    "atk_debuff",
    "counter_bonus",
    "counter_set_attack",
    "brand",
    "nightmare",
    "step_index",
    "scripted_for",
    "damage_immunity_until",
    "free_normal_attack_turn",
)
FLAG_BOOL_FIELDS: tuple[str, ...] = (
    "has_harmonized",
    "return_after",
    "temp_revive",
)
FLAG_STR_FIELDS: tuple[str, ...] = ("gulang_item",)
CHARACTER_RUNTIME_STR_FIELDS: frozenset[str] = frozenset(("last_hit_source", "flavor"))

NON_AUTHORITATIVE_GAME_KEYS: frozenset[str] = frozenset(("_public_board", "events"))
UNSUPPORTED_GAME_KEYS: frozenset[str] = frozenset(
    (
        "turn_snapshots",
        "time_removed",
        "requests",
        "scenario",
        "tutorial",
        "tutorial_id",
        "lesson",
    )
)
GAME_KEYS: frozenset[str] = (
    frozenset(
        (
            "rules_version",
            "design_version",
            "version",
            "phase",
            "active_side",
            "first_side",
            "turn",
            "winner",
            "reason",
            "rng",
            "next_instance",
            "next_entity",
            "event_seq",
            "pending_choice",
            "operation",
            "mulligan_done",
            "current_action_id",
            "current_action_version",
            "combat_group",
            "sides",
            "escalation_enabled",
            "harmony_residue_version",
            "deferred_actions",
            "turn_start_pending",
            "_response_lock",
            "_response_intercept",
            "_draining_actions",
            "_pending_down_responses",
            "_deferred_card_marks",
        )
    )
    | NON_AUTHORITATIVE_GAME_KEYS
    | UNSUPPORTED_GAME_KEYS
)
PLAYER_KEYS: frozenset[str] = (
    frozenset(
        (
            "entity_id",
            "side",
            "name",
            "hp",
            "max_hp",
            "shield",
            "ap",
            "front",
            "last_front",
            "order",
            "characters",
            "hand",
            "deck",
            "discard",
            "removed",
            "records",
            "used",
            "harmonized",
            "front_debuff",
            "attacked_this_turn",
            "effects",
            "effect_seq",
            "flags",
        )
    )
    | frozenset(PLAYER_INT_FIELDS)
    | frozenset(PLAYER_BOOL_FIELDS)
    | frozenset(PLAYER_STR_FIELDS)
)
CHARACTER_KEYS: frozenset[str] = (
    frozenset(
        (
            "id",
            "entity_id",
            "side",
            "hp",
            "shield",
            "energy",
            "flags",
            "effects",
            "base_stats",
            "permanent_growth",
            "equipment",
            "ultimate",
            "mechanisms",
            "resource_name",
            "turn_snapshots",
            "lingke_chain",
        )
    )
    | frozenset(CHARACTER_INT_FIELDS)
    | frozenset(CHARACTER_BOOL_FIELDS)
    | frozenset(CHARACTER_STR_FIELDS)
)
CARD_KEYS: frozenset[str] = (
    frozenset(("entity_id", "instance_id", "side", "card_id", "copy", "jingu_mark", "source", "play_options"))
    | frozenset(CARD_INT_FIELDS)
    | frozenset(CARD_BOOL_FIELDS)
    | frozenset(CARD_STR_FIELDS)
)

COVERAGE: MappingProxyType[str, object] = MappingProxyType(
    {
        "supports_full_duel_rules": False,
        "roster": "public17",
        "public_character_ids": PUBLIC_CHARACTER_ORDER,
        "seven_official_presets": (
            "starter",
            "weave-rush",
            "quick-rush",
            "zhenhong",
            "murk",
            "collapse",
            "lingke",
        ),
        "declared_field_layout": (
            "game scalars including rng/version/phase/sides/turn/winner",
            "two PlayerEntity rows",
            "variable CharacterEntity rows including summons, ordered per side",
            "CardEntity table plus ordered hand/deck/discard/removed",
            "entity class, entity_id, instance_id and ordered zone references",
            "typed modifiers/effects with expiry clocks",
            "typed last_hit_source/lingke_chain/beast_fangs/support_turn/effect_seq",
            "typed flags including collapse/timed_attack/immunity/has_harmonized/gulang_item",
            "pending_choice including deck_cards/deck_order/snapshot_n/continuation",
            "deferred_actions presence distinguishing empty list from absence",
            "front_debuff delay/burn/star/weave/stain/dots",
            "absent vs null vs zero via presence codes",
            "int32/int64 exact range, no truncation",
            "dedicated character_hp/shield/energy tensors [B,2,C]",
        ),
        "non_authoritative_dropped_or_rejected": (
            "events presentation log and patches",
            "_public_board capture cache",
        ),
        "unsupported_rejected": (
            "unknown keys at any allowlisted object",
            "non-public character templates except summoned=True",
            "tutorial / scenario state",
            "turn_snapshots nested games and time_removed ledger",
            "mutated catalog nests (mechanisms/play_options/source/ultimate)",
            "Python callables, numpy/torch payloads, bool used as int",
            "non-finite floats, integers outside declared dtype",
            "records with payloads, visible_deck, graph blobs in tensors",
        ),
    }
)


def _positive_int(name, value, *, lo=1, hi=0x7FFFFFFF):
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__} (bool is rejected)")
    if value < lo or value > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")
    return value


@dataclass(frozen=True)
class DomainSoALayout:
    """Frozen capacities for one domain SoA batch.

    game_capacity is the maximum row count, not the resident count. Resident
    arrays use the actual packed row count n_rows <= game_capacity.
    """

    game_capacity: int = 16
    max_characters: int = 8
    max_cards: int = 192
    max_zone: int = 96
    max_effects: int = 24
    max_used: int = 16
    max_choice_cards: int = 32
    max_deferred: int = 16
    max_string_bytes: int = 65536
    max_string_slots: int = 512
    max_dots: int = 4
    max_timed_attack: int = 8
    max_clear_on: int = 8
    max_participants: int = 8
    max_down_responses: int = 8
    max_deferred_marks: int = 8

    def __post_init__(self):
        for field in self.__dataclass_fields__:
            _positive_int(field, getattr(self, field))
        if self.max_characters < 4:
            raise ValueError("max_characters must be at least 4 for a public lineup")
        if self.max_dots < len(DOT_KINDS):
            raise ValueError("max_dots must cover every known DOT kind")

    @property
    def n_sides(self):
        return 2

    @property
    def holder_count(self):
        return 2 * (1 + self.max_characters)

    def holder_index(self, side, character_slot):
        if side not in (0, 1):
            raise DomainSoADecodeError("side slot must be 0 or 1")
        if character_slot is None:
            return side * (1 + self.max_characters)
        if character_slot < 0 or character_slot >= self.max_characters:
            raise DomainSoADecodeError("character slot outside layout")
        return side * (1 + self.max_characters) + 1 + character_slot


def array_specs(layout):
    """Trailing dimension specs; the batch row axis is prepended at allocate time."""
    C = layout.max_characters
    K = layout.max_cards
    Z = layout.max_zone
    E = layout.max_effects
    U = layout.max_used
    H = layout.holder_count
    D = layout.max_dots
    T = layout.max_timed_attack
    P = layout.max_participants
    CH = layout.max_choice_cards
    CL = layout.max_clear_on
    DF = layout.max_deferred
    DR = layout.max_down_responses
    i8, i16, i32, i64 = np.int8, np.int16, np.int32, np.int64
    u8 = np.uint8
    b = np.bool_
    f64 = np.float64
    n_pint = len(PLAYER_INT_FIELDS)
    n_pbool = len(PLAYER_BOOL_FIELDS)
    n_pstr = len(PLAYER_STR_FIELDS)
    n_cint = len(CHARACTER_INT_FIELDS)
    n_cbool = len(CHARACTER_BOOL_FIELDS)
    n_cstr = len(CHARACTER_STR_FIELDS)
    n_kint = len(CARD_INT_FIELDS)
    n_kbool = len(CARD_BOOL_FIELDS)
    n_kstr = len(CARD_STR_FIELDS)
    n_fint = len(FLAG_INT_FIELDS)
    n_fbool = len(FLAG_BOOL_FIELDS)
    n_fstr = len(FLAG_STR_FIELDS)
    n_chint = len(CHOICE_INT_KEYS)
    n_chbool = len(CHOICE_BOOL_KEYS)
    n_chstr = len(CHOICE_STR_KEYS)
    n_opflag = len(OP_FLAG_KEYS)
    n_opint = len(OP_INT_KEYS)
    return {
        "catalog_digest": (u8, (32,)),
        "row_valid": (b, ()),
        "row_error": (i32, ()),
        "version": (i32, ()),
        "phase": (i16, ()),
        "active_side": (i8, ()),
        "first_side": (i8, ()),
        "turn": (i32, ()),
        "winner_presence": (u8, ()),
        "winner": (i8, ()),
        "reason_presence": (u8, ()),
        "reason_slot": (i32, ()),
        "rng": (i64, ()),
        "next_instance": (i32, ()),
        "next_entity": (i32, ()),
        "event_seq": (i32, ()),
        "current_action_id_slot": (i32, ()),
        "current_action_version": (i32, ()),
        "combat_group": (i32, ()),
        "escalation_enabled": (b, ()),
        "harmony_residue_version": (i32, ()),
        "rules_version_slot": (i32, ()),
        "design_version_slot": (i32, ()),
        "mulligan_count": (i8, ()),
        "mulligan_sides": (i8, (2,)),
        "pending_choice_present": (b, ()),
        "operation_present": (b, ()),
        "turn_start_pending_presence": (u8, ()),
        "turn_start_pending_side": (i8, ()),
        "response_lock_presence": (u8, ()),
        "response_lock": (b, ()),
        "response_intercept_presence": (u8, ()),
        "response_intercept": (b, ()),
        "draining_presence": (u8, ()),
        "draining": (b, ()),
        "deferred_presence": (u8, ()),
        "deferred_count": (i16, ()),
        "down_response_count": (i16, ()),
        "choice_kind": (i16, ()),
        "choice_side": (i8, ()),
        "choice_prompt_slot": (i32, ()),
        "choice_card_count": (i16, ()),
        "choice_card_index": (i32, (CH,)),
        "choice_int_presence": (u8, (n_chint,)),
        "choice_int_value": (i32, (n_chint,)),
        "choice_bool_presence": (u8, (n_chbool,)),
        "choice_bool_value": (b, (n_chbool,)),
        "choice_str_presence": (u8, (n_chstr,)),
        "choice_str_slot": (i32, (n_chstr,)),
        "choice_actor_presence": (u8, ()),
        "choice_actor_side": (i8, ()),
        "choice_actor_slot": (i16, ()),
        "choice_continuation_presence": (u8, ()),
        "choice_continuation_slot": (i32, ()),
        "op_side": (i8, ()),
        "op_actor_slot": (i16, ()),
        "op_card_index": (i32, ()),
        "op_target_presence": (u8, ()),
        "op_target_slot": (i32, ()),
        "op_harmony_presence": (u8, ()),
        "op_harmony_slot": (i32, ()),
        "op_first_bonus": (i32, ()),
        "op_redeem_bonus": (i32, ()),
        "op_record_candidate": (b, ()),
        "op_flag_presence": (u8, (n_opflag,)),
        "op_flag_value": (b, (n_opflag,)),
        "op_int_presence": (u8, (n_opint,)),
        "op_int_value": (i32, (n_opint,)),
        "op_option_id_presence": (u8, ()),
        "op_option_id_slot": (i32, ()),
        "op_last_attack_target_presence": (u8, ()),
        "op_last_attack_target_slot": (i32, ()),
        "op_card_mark_presence": (u8, ()),
        "op_card_mark_delta": (i32, ()),
        "op_card_mark_source_slot": (i32, ()),
        "op_participant_count": (i16, ()),
        "op_participant_side": (i8, (P,)),
        "op_participant_slot": (i16, (P,)),
        "op_choice_card_count": (i16, ()),
        "op_choice_card_index": (i32, (CH,)),
        "deferred_side": (i8, (DF,)),
        "deferred_actor_slot": (i16, (DF,)),
        "deferred_entity_slot": (i32, (DF,)),
        "deferred_kind_slot": (i32, (DF,)),
        "down_response_side": (i8, (DR,)),
        "down_response_target_slot": (i32, (DR,)),
        "player_valid": (b, (2,)),
        "player_entity_slot": (i32, (2,)),
        "player_name_slot": (i32, (2,)),
        "player_hp": (i32, (2,)),
        "player_max_hp": (i32, (2,)),
        "player_shield": (i32, (2,)),
        "player_ap": (i32, (2,)),
        "player_front_presence": (u8, (2,)),
        "player_front_slot": (i16, (2,)),
        "player_last_front_presence": (u8, (2,)),
        "player_last_front_slot": (i16, (2,)),
        "player_order_count": (i16, (2,)),
        "player_character_count": (i16, (2,)),
        "player_order_slot": (i16, (2, C)),
        "player_int_presence": (u8, (2, n_pint)),
        "player_int_value": (i32, (2, n_pint)),
        "player_bool_presence": (u8, (2, n_pbool)),
        "player_bool_value": (b, (2, n_pbool)),
        "player_str_presence": (u8, (2, n_pstr)),
        "player_str_slot": (i32, (2, n_pstr)),
        "player_used_count": (i16, (2,)),
        "player_used_key_slot": (i32, (2, U)),
        "player_used_value": (b, (2, U)),
        "player_harmonized": (b, (2, C)),
        "player_attacked_this_turn_presence": (u8, (2,)),
        "player_attacked_this_turn": (b, (2, C)),
        "player_attacked_count": (i16, (2,)),
        "player_attacked_slot": (i16, (2, C)),
        "player_records_count": (i16, (2,)),
        "player_effects_presence": (u8, (2,)),
        "player_flags_presence": (u8, (2,)),
        "zone_count": (i16, (2, 4)),
        "zone_card_index": (i32, (2, 4, Z)),
        "character_valid": (b, (2, C)),
        "character_entity_slot": (i32, (2, C)),
        "character_template": (i16, (2, C)),
        "character_id_slot": (i32, (2, C)),
        "character_hp": (i32, (2, C)),
        "character_shield": (i32, (2, C)),
        "character_energy": (i32, (2, C)),
        "character_hp_presence": (u8, (2, C)),
        "character_shield_presence": (u8, (2, C)),
        "character_energy_presence": (u8, (2, C)),
        "character_int_presence": (u8, (2, C, n_cint)),
        "character_int_value": (i32, (2, C, n_cint)),
        "character_bool_presence": (u8, (2, C, n_cbool)),
        "character_bool_value": (b, (2, C, n_cbool)),
        "character_str_presence": (u8, (2, C, n_cstr)),
        "character_str_slot": (i32, (2, C, n_cstr)),
        "character_base_attack_presence": (u8, (2, C)),
        "character_base_attack": (i32, (2, C)),
        "character_base_max_hp_presence": (u8, (2, C)),
        "character_base_max_hp": (i32, (2, C)),
        "character_growth_attack_presence": (u8, (2, C)),
        "character_growth_attack": (i32, (2, C)),
        "character_growth_max_hp_presence": (u8, (2, C)),
        "character_growth_max_hp": (i32, (2, C)),
        "character_ultimate_presence": (u8, (2, C)),
        "character_mechanisms_presence": (u8, (2, C)),
        "character_equipment_presence": (u8, (2, C)),
        "character_equipment_card": (i16, (2, C)),
        "character_equipment_id_slot": (i32, (2, C)),
        "character_equipment_source_slot": (i32, (2, C)),
        "character_effects_presence": (u8, (2, C)),
        "character_flags_presence": (u8, (2, C)),
        "character_lingke_presence": (u8, (2, C)),
        "character_lingke_entity_slot": (i32, (2, C)),
        "character_lingke_id_slot": (i32, (2, C)),
        "character_lingke_count": (i32, (2, C)),
        "card_valid": (b, (K,)),
        "card_entity_slot": (i32, (K,)),
        "card_instance_slot": (i32, (K,)),
        "card_template": (i16, (K,)),
        "card_side": (i8, (K,)),
        "card_int_presence": (u8, (K, n_kint)),
        "card_int_value": (i32, (K, n_kint)),
        "card_bool_presence": (u8, (K, n_kbool)),
        "card_bool_value": (b, (K, n_kbool)),
        "card_str_presence": (u8, (K, n_kstr)),
        "card_str_slot": (i32, (K, n_kstr)),
        "card_source_presence": (u8, (K,)),
        "card_play_options_presence": (u8, (K,)),
        "card_jingu_mark_presence": (u8, (K,)),
        "card_jingu_delta": (i32, (K,)),
        "card_jingu_source_slot": (i32, (K,)),
        "holder_effect_count": (i16, (H,)),
        "effect_valid": (b, (H, E)),
        "effect_id_slot": (i32, (H, E)),
        "effect_definition_slot": (i32, (H, E)),
        "effect_property_slot": (i32, (H, E)),
        "effect_op": (i8, (H, E)),
        "effect_value_kind": (i8, (H, E)),
        "effect_value_int": (i32, (H, E)),
        "effect_value_float": (f64, (H, E)),
        "effect_source_entity_presence": (u8, (H, E)),
        "effect_source_entity_slot": (i32, (H, E)),
        "effect_source_key_slot": (i32, (H, E)),
        "effect_scope": (i8, (H, E)),
        "effect_kind": (i8, (H, E)),
        "effect_layer": (i32, (H, E)),
        "effect_label_slot": (i32, (H, E)),
        "effect_visibility": (i8, (H, E)),
        "effect_stack_key_presence": (u8, (H, E)),
        "effect_stack_key_slot": (i32, (H, E)),
        "effect_stack_limit_presence": (u8, (H, E)),
        "effect_stack_limit": (i32, (H, E)),
        "effect_show_marker_presence": (u8, (H, E)),
        "effect_show_marker": (b, (H, E)),
        "effect_consume_on_presence": (u8, (H, E)),
        "effect_consume_on_slot": (i32, (H, E)),
        "effect_clear_count": (i16, (H, E)),
        "effect_clear_event": (i8, (H, E, CL)),
        "effect_expiry_presence": (u8, (H, E)),
        "effect_expiry_side_presence": (u8, (H, E)),
        "effect_expiry_side": (i8, (H, E)),
        "effect_expiry_phase": (i8, (H, E)),
        "effect_expiry_turn_presence": (u8, (H, E)),
        "effect_expiry_turn": (i32, (H, E)),
        "effect_expiry_remaining_presence": (u8, (H, E)),
        "effect_expiry_remaining": (i32, (H, E)),
        "effect_conditions_count": (i16, (H, E)),
        "flag_int_presence": (u8, (H, n_fint)),
        "flag_int_value": (i32, (H, n_fint)),
        "flag_bool_presence": (u8, (H, n_fbool)),
        "flag_bool_value": (b, (H, n_fbool)),
        "flag_str_presence": (u8, (H, n_fstr)),
        "flag_str_slot": (i32, (H, n_fstr)),
        "flag_collapse_presence": (u8, (H,)),
        "flag_collapse_side": (i8, (H,)),
        "flag_collapse_until": (i32, (H,)),
        "flag_expiring_presence": (u8, (H,)),
        "flag_expiring_amount": (i32, (H,)),
        "flag_expiring_until": (i32, (H,)),
        "flag_timed_count": (i16, (H,)),
        "flag_timed_amount": (i32, (H, T)),
        "flag_timed_left": (i32, (H, T)),
        "zone_status_presence": (u8, (2, 5)),
        "zone_status_i32": (i32, (2, 5, 8)),
        "zone_status_side": (i8, (2, 5, 4)),
        "zone_status_bool": (b, (2, 5, 2)),
        "zone_status_slot": (i32, (2, 5, 2)),
        "zone_dot_valid": (b, (2, D)),
        "zone_dot_kind": (i8, (2, D)),
        "zone_dot_stacks": (i32, (2, D)),
        "zone_dot_by": (i8, (2, D)),
        "zone_dot_source_slot": (i32, (2, D)),
        "zone_dot_left_presence": (u8, (2, D)),
        "zone_dot_left": (i32, (2, D)),
        "zone_dot_applied_turn_presence": (u8, (2, D)),
        "zone_dot_applied_turn": (i32, (2, D)),
        "string_count": (i32, ()),
        "string_bytes_used": (i32, ()),
        "string_offset": (i32, (layout.max_string_slots,)),
        "string_length": (i32, (layout.max_string_slots,)),
        "string_bytes": (u8, (layout.max_string_bytes,)),
    }


ARRAY_NAMES: tuple[str, ...] = tuple(array_specs(DomainSoALayout()).keys())


def catalog_fingerprint():
    """Runtime binding for catalog-backed values, not a rule/GPU certificate."""
    from hashlib import sha256
    import json
    payload = dict(
        characters={key: CHARACTERS[key] for key in sorted(PUBLIC_CHARACTER_IDS)},
        cards={key: value for key, value in sorted(CARDS.items())
               if value.get('character_id') in PUBLIC_CHARACTER_IDS},
    )
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).digest()


def allocate_arrays(layout, n_rows):
    if type(n_rows) is not int or isinstance(n_rows, bool) or n_rows < 0:
        raise TypeError("n_rows must be a nonnegative int")
    if n_rows > layout.game_capacity:
        raise DomainSoACapacityError(f"n_rows {n_rows} exceeds game_capacity {layout.game_capacity}")
    arrays = {}
    for name, (dtype, tail) in array_specs(layout).items():
        arrays[name] = np.zeros((n_rows,) + tail, dtype=dtype, order="C")
    return arrays


def clone_rows(batch, row_indices):
    """Copy selected rows into an independent C-contiguous batch."""
    if not isinstance(batch, DomainSoABatch):
        raise TypeError("clone_rows requires a DomainSoABatch")
    indices = list(row_indices)
    for idx in indices:
        if isinstance(idx, (bool, np.bool_)) or not isinstance(idx, (int, np.integer)):
            raise TypeError(f"Row index must be an integer, got {type(idx).__name__}")
        if idx < 0 or idx >= batch.n_rows:
            raise IndexError(f"Row index {idx} out of range [0, {batch.n_rows})")
    if len(indices) > batch.layout.game_capacity:
        raise DomainSoACapacityError("cloned batch exceeds game_capacity")
    if not indices:
        arrays = allocate_arrays(batch.layout, 0)
    else:
        idx_arr = np.array(indices, dtype=np.intp)
        arrays = {
            name: np.ascontiguousarray(getattr(batch, name)[idx_arr].copy()) for name in ARRAY_NAMES
        }
    return DomainSoABatch(layout=batch.layout, **arrays)


class DomainSoABatch:
    """Resident typed SoA batch. Arrays are C-contiguous NumPy with exact dtypes.

    Inputs passed to encode are not mutated. clone_rows / select copy.
    to_torch is an explicit offline export and imports torch lazily.
    """

    def __init__(self, layout, **arrays):
        if not isinstance(layout, DomainSoALayout):
            raise TypeError("layout must be a DomainSoALayout")
        missing = [name for name in ARRAY_NAMES if name not in arrays]
        extra = [name for name in arrays if name not in ARRAY_NAMES]
        if missing or extra:
            raise TypeError(f"DomainSoABatch fields mismatch missing={missing} extra={extra}")
        n_rows = None
        specs = array_specs(layout)
        for name in ARRAY_NAMES:
            arr = arrays[name]
            dtype, tail = specs[name]
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray")
            if arr.dtype != np.dtype(dtype):
                raise TypeError(f"{name} dtype must be {np.dtype(dtype)}, got {arr.dtype}")
            if not arr.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")
            if arr.shape[1:] != tail:
                raise ValueError(f"{name} trailing shape {arr.shape[1:]} != {tail}")
            if n_rows is None:
                n_rows = arr.shape[0]
            elif arr.shape[0] != n_rows:
                raise ValueError(f"{name} row count {arr.shape[0]} != {n_rows}")
            setattr(self, name, arr)
        if n_rows > layout.game_capacity:
            raise DomainSoACapacityError(
                f"Batch row count {n_rows} exceeds game_capacity {layout.game_capacity}"
            )
        self.layout = layout

    @property
    def n_rows(self):
        return int(self.row_valid.shape[0])

    @property
    def allocated_bytes(self):
        return int(sum(getattr(self, name).nbytes for name in ARRAY_NAMES))

    def clone_rows(self, row_indices):
        return clone_rows(self, row_indices)

    def select(self, row_indices):
        return clone_rows(self, row_indices)

    def to_torch(self, device="cpu"):
        """Offline tensor export. Not an ingress path and not a rule engine."""
        import torch

        tensors = {
            name: torch.as_tensor(getattr(self, name).copy(), device=torch.device(device))
            for name in ARRAY_NAMES
        }
        return DomainSoATorchBatch(layout=self.layout, device=str(torch.device(device)), arrays=tensors)

    @classmethod
    def from_torch(cls, packed):
        """Offline host egress from a previous to_torch export."""
        arrays = {
            name: np.array(packed.arrays[name].detach().cpu().numpy(), copy=True, order='C')
            for name in ARRAY_NAMES
        }
        return cls(layout=packed.layout, **arrays)


@dataclass(frozen=True)
class DomainSoATorchBatch:
    """Lazy-torch view of a domain SoA batch. CPU NumPy remains canonical."""

    layout: DomainSoALayout
    device: str
    arrays: dict

    def clone_rows(self, row_indices):
        import torch

        if isinstance(row_indices, torch.Tensor):
            if row_indices.dtype not in (torch.int32, torch.int64):
                raise TypeError("Exact integer row indices required; no float/bool coercion")
            if row_indices.device != self.arrays["row_valid"].device:
                raise ValueError("Row indices must reside on the batch device")
            index = row_indices.to(dtype=torch.int64)
            if index.ndim != 1:
                raise ValueError("Row indices must be a 1-D integer tensor")
        else:
            values = list(row_indices)
            if any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                for v in values
            ):
                raise TypeError("Integer row indices required, not bool/float")
            index = torch.tensor(
                values, dtype=torch.int64, device=self.arrays["row_valid"].device
            )
        n = int(index.numel())
        n_rows = int(self.arrays["row_valid"].shape[0])
        # This is an explicit offline export helper, not a capture-safe device
        # pool. GPU-resident search requires its own masked lease/row API.
        if n and bool(((index < 0) | (index >= n_rows)).any()):
            raise IndexError("Row index outside batch")
        if n > self.layout.game_capacity:
            raise DomainSoACapacityError("cloned batch exceeds game_capacity")
        arrays = {
            name: value.index_select(0, index).contiguous() for name, value in self.arrays.items()
        }
        return DomainSoATorchBatch(layout=self.layout, device=self.device, arrays=arrays)


def side_index(side):
    if side not in SIDES:
        raise DomainSoAUnsupportedError(f"Unsupported side {side!r}")
    return 0 if side == "a" else 1


def vocab_get(table, key, *, what):
    if key not in table:
        raise DomainSoAUnsupportedError(
            f"Unknown {what} {key!r} is outside public-17 SoA coverage"
        )
    return table[key]


def catalog_character(cid):
    spec = CHARACTERS.get(cid)
    if spec is None:
        raise DomainSoAUnsupportedError(f"Unknown character template {cid!r}")
    return spec


def catalog_card(card_id):
    spec = CARDS.get(card_id)
    if spec is None:
        raise DomainSoAUnsupportedError(f"Unknown card template {card_id!r}")
    return spec
