"""Declarative RECORD-POOL schema for canonical public-17 duel_v2 state.

This is not the initial-only tensor codec and not a rule engine. Every
authoritative Game/Player/Character/Card field, plus flags, effects, zone
status, dots, choices, operations, deferred/resumption, clean events and
`_public_board` presentation, is a closed typed pool. Composite values are
references to those pools so mutable aliases survive. Absent and None are
automatic for every field.

Catalog source / mechanisms / ultimate / play_options are expanded into
strict records from the live public-17 catalog, never stored opaque.
"""
from __future__ import annotations

from types import MappingProxyType

from app.modules.card_game.rl.batched_duel.vm_heap import (
    TAG_CARD_ENTITY,
    TAG_CHARACTER_ENTITY,
    TAG_DICT,
    TAG_LIST,
    TAG_PLAYER_ENTITY,
    TAG_TUPLE,
)


supports_full_duel_rules = False
ROOT = 'Game'
LIST_TAGS = (TAG_LIST,)
SEQUENCE_TAGS = (TAG_LIST, TAG_TUPLE)
DICT_TAGS = (TAG_DICT,)
PLAYER_TAGS = (TAG_PLAYER_ENTITY, TAG_DICT)
CHARACTER_TAGS = (TAG_CHARACTER_ENTITY, TAG_DICT)
CARD_TAGS = (TAG_CARD_ENTITY, TAG_DICT)
TUPLE_TAGS = (TAG_TUPLE,)

# Conservative per-row pool sizes; the primary may override at compile time.
CAP_GAME = 8
CAP_PLAYER = 16
CAP_CHARACTER = 32
CAP_CARD = 256
CAP_EFFECT = 384
CAP_EVENT = 512
CAP_SMALL = 64
CAP_TINY = 32
CAP_MAP = 16
MAX_ZONE = 96
MAX_CHARS = 16
MAX_EFFECTS = 32
MAX_EVENTS = 512
MAX_USED = 24
MAX_CHOICE = 32
MAX_DEFERRED = 16
MAX_DOTS = 8
MAX_MARKERS = 32
MAX_OPTIONS = 8
MAX_MECHANISMS = 8
MAX_CONDITIONS = 8
MAX_CLEAR = 8
MAX_PARTICIPANTS = 8
MAX_STRING_MAP = 24
MAX_SNAPSHOTS = 4


def _record(capacity, tags, fields):
    return {'kind': 'record', 'capacity': int(capacity), 'tags': tuple(tags), 'fields': dict(fields)}


def _list(capacity, max_items, value, tags=LIST_TAGS):
    return {
        'kind': 'list',
        'capacity': int(capacity),
        'max_items': int(max_items),
        'tags': tuple(tags),
        'value': value,
    }


def _map(capacity, max_items, value, key='str'):
    return {
        'kind': 'map',
        'capacity': int(capacity),
        'max_items': int(max_items),
        'key': key,
        'value': value,
    }


def _tuple(capacity, items, tags=TUPLE_TAGS):
    return {'kind': 'tuple', 'capacity': int(capacity), 'tags': tuple(tags), 'items': list(items)}


def _one_of(*choices):
    return {'one_of': list(choices)}

def _leaf_types():
    return {
        'SideCid': _tuple(CAP_SMALL, ['str', 'str'], SEQUENCE_TAGS),
        'Participant': _tuple(CAP_SMALL, ['str', 'str'], SEQUENCE_TAGS),
        'DownResponse': _tuple(CAP_TINY, ['str', 'str'], SEQUENCE_TAGS),
        'StringList': _list(CAP_SMALL, MAX_MARKERS, 'str', SEQUENCE_TAGS),
        'SideList': _list(CAP_TINY, 2, 'str', SEQUENCE_TAGS),
        'EntityIdList': _list(CAP_TINY, MAX_CHARS, 'str', SEQUENCE_TAGS),
        'IntList': _list(CAP_TINY, MAX_CHOICE, 'int', SEQUENCE_TAGS),
        'StringIntMap': _map(CAP_MAP, MAX_STRING_MAP, 'int'),
        'StringBoolMap': _map(CAP_MAP, MAX_USED, 'bool'),
        'StringScalarMap': _map(CAP_SMALL, MAX_STRING_MAP, 'scalar'),
        'PlayOptionCosts': _map(CAP_TINY, MAX_OPTIONS, 'int'),
        'SupportActor': _record(CAP_TINY, DICT_TAGS, {
            'character_id': 'str',
            'entity_id': 'str',
        }),
        'CollapseStatus': _record(CAP_CHARACTER, DICT_TAGS, {
            'side': 'str',
            'until': 'int',
        }),
        'MimicStatus': _record(CAP_TINY, DICT_TAGS, {
            'id': 'str',
        }),
        'AtkBuffExpiring': _record(CAP_CHARACTER, DICT_TAGS, {
            'amount': 'int',
            'until': 'int',
        }),
        'TimedAttack': _record(CAP_SMALL, DICT_TAGS, {
            'amount': 'int',
            'left': 'int',
        }),
        'JinguMark': _record(CAP_CARD, DICT_TAGS, {
            'delta': 'int',
            'source_entity_id': 'str',
        }),
        'LingkeChain': _record(CAP_TINY, DICT_TAGS, {
            'entity_id': 'str',
            'character_id': 'str',
            'count': 'int',
        }),
        'BaseStats': _record(CAP_CHARACTER, DICT_TAGS, {
            'attack': 'int',
            'max_hp': 'int',
        }),
        'PermanentGrowth': _record(CAP_CHARACTER, DICT_TAGS, {
            'attack': 'int',
            'max_hp': 'int',
        }),
        'Equipment': _record(CAP_CHARACTER, DICT_TAGS, {
            'card_id': 'str',
            'equipment_id': 'str',
            'source_card_entity_id': 'str',
        }),
        'PublicEquipment': _record(CAP_CHARACTER, DICT_TAGS, {
            'card_id': 'str',
            'equipment_id': 'str',
        }),
        'UltimateSpec': _record(CAP_CHARACTER, DICT_TAGS, {
            'kind': 'str',
            'turns': 'int',
            'expires': 'str',
        }),
        'Mechanism': _record(CAP_TINY, DICT_TAGS, {
            'id': 'str',
            'name': 'str',
            'description': 'str',
            'reference_names': '@StringList',
            'character_id': 'str',
            'type': 'str',
        }),
        'CardSource': _record(CAP_CARD, DICT_TAGS, {
            'character_id': _one_of('int', 'str'),
            'type': 'str',
            'index': 'int',
            'fashion_id': 'str',
            'fashion_name': 'str',
            'repository': 'str',
            'commit': 'str',
            'path': 'str',
            'previous_name': 'str',
            'previous_source': '@CardSource',
            'verified_at': 'str',
            'rules_source': 'str',
            'naming_note': 'str',
            'naming': 'str',
            'name_source': 'str',
        }),
        'PlayOption': _record(CAP_TINY, DICT_TAGS, {
            'id': 'str',
            'label': 'str',
            'description': 'str',
            'costs': '@PlayOptionCosts',
            'cost_label': 'str',
        }),
        'EffectExpiry': _record(CAP_EFFECT, DICT_TAGS, {
            'side': 'str',
            'turn': 'int',
            'remaining': 'int',
            'phase': 'str',
        }),
        'EffectConditionArgs': _record(CAP_EFFECT, DICT_TAGS, {
            'card_id': 'str',
            'resource': 'str',
            'amount': 'int',
            'attribute': 'str',
            'value': 'str',
        }),
        'EffectCondition': _record(CAP_EFFECT, DICT_TAGS, {
            'id': 'str',
            'args': '@EffectConditionArgs',
        }),
        'Effect': _record(CAP_EFFECT, DICT_TAGS, {
            'effect_id': 'str',
            'definition_id': 'str',
            'property': 'str',
            'op': 'str',
            'value': _one_of('int', 'float', 'bool'),
            'source_entity_id': 'str',
            'source_key': 'str',
            'scope': 'str',
            'kind': 'str',
            'layer': 'int',
            'label': 'str',
            'clear_on': '@StringList',
            'expiry': '@EffectExpiry',
            'conditions': '@EffectConditionList',
            'consume_on': 'str',
            'visibility': 'str',
            'stack_key': 'str',
            'show_marker': 'bool',
            'stack_limit': 'int',
        }),
        'TimedAttackList': _list(CAP_TINY, 8, '@TimedAttack'),
        'MechanismList': _list(CAP_TINY, MAX_MECHANISMS, '@Mechanism'),
        'PlayOptionList': _list(CAP_TINY, MAX_OPTIONS, '@PlayOption'),
        'EffectList': _list(CAP_SMALL, MAX_EFFECTS, '@Effect'),
        'EffectConditionList': _list(CAP_EFFECT, MAX_CONDITIONS, '@EffectCondition'),
        'ParticipantList': _list(CAP_TINY, MAX_PARTICIPANTS, '@Participant'),
        'DownResponseList': _list(CAP_TINY, MAX_DEFERRED, '@DownResponse'),
        'RecordEntry': _record(CAP_TINY, DICT_TAGS, {
            'card_id': 'str',
            'name': 'str',
            'text': 'str',
            'amount': 'int',
        }),
        'RecordList': _list(CAP_TINY, MAX_USED, '@RecordEntry'),
    }


def _flag_and_zone_types():
    return {
        'Flags': _record(CAP_CHARACTER, DICT_TAGS, {
            'next_bonus': 'int',
            'next_shield': 'int',
            'next_followup': 'int',
            'energy_next': 'bool',
            'allied_hurt': 'bool',
            'pending_atk': 'int',
            'share_overflow': 'bool',
            'hp_floor': 'int',
            'collapse': '@CollapseStatus',
            'collapse_count': 'int',
            'collapse_max': 'int',
            'pact': 'bool',
            'mimic': '@MimicStatus',
            'atk_buff': 'int',
            'atk_buff_expiring': '@AtkBuffExpiring',
            'atk_debuff': 'int',
            'form_attack': 'int',
            'timed_attack': '@TimedAttackList',
            'next_attack': 'int',
            'damage_immunity_until': 'int',
            'brand': 'int',
            'nightmare': 'int',
            'counter_bonus': 'int',
            'counter_set_attack': 'int',
            'return_after': 'bool',
            'free_normal_attack_turn': 'int',
            'has_harmonized': 'bool',
            'gulang_item': 'str',
            'temp_revive': 'bool',
            'burn': 'bool',
            'star': 'bool',
            'slow': 'bool',
        }),
        'DelayStatus': _record(CAP_TINY, DICT_TAGS, {
            'left': 'int',
            'tick_side': 'str',
            'slow': 'int',
            'slow_turn': 'int',
            'by': 'str',
            'expires_side': 'str',
        }),
        'BurnStatus': _record(CAP_TINY, DICT_TAGS, {
            'left': 'int',
            'by': 'str',
            'stacks': 'int',
            'source': 'str',
            'applied_turn': 'int',
            'infinite': 'bool',
        }),
        'StarStatus': _record(CAP_TINY, DICT_TAGS, {
            'left': 'int',
            'by': 'str',
            'source_cid': 'str',
            'renewals': 'int',
        }),
        'WeaveStatus': _record(CAP_TINY, DICT_TAGS, {
            'by': 'str',
            'expires_side': 'str',
        }),
        'StainStatus': _record(CAP_TINY, DICT_TAGS, {
            'by': 'str',
            'expires_side': 'str',
            'ends_left': 'int',
        }),
        'DotStatus': _record(CAP_TINY, DICT_TAGS, {
            'name': 'str',
            'stacks': 'int',
            'by': 'str',
            'source': 'str',
            'left': 'int',
            'applied_turn': 'int',
            'mode': 'str',
            'halve': 'bool',
            'ignore_shield': 'bool',
            'max_stacks': 'int',
        }),
        'DotMap': _map(CAP_TINY, MAX_DOTS, '@DotStatus'),
        'FrontDebuff': _record(CAP_TINY, DICT_TAGS, {
            'delay': '@DelayStatus',
            'burn': '@BurnStatus',
            'star': '@StarStatus',
            'weave': '@WeaveStatus',
            'stain': '@StainStatus',
            'dots': '@DotMap',
        }),
    }

def _flow_types():
    return {
        'SortieSpec': _record(CAP_TINY, DICT_TAGS, {
            'target': '@SideCid',
            'stay_bench': 'bool',
            'no_counter': 'bool',
            'front_only': 'bool',
            'additional_attack': 'bool',
            'sync_attack': 'bool',
            'support_allows_counter': 'bool',
            'no_counter_harmony': 'bool',
            'bypass': 'bool',
            'penetrate_harmony': 'bool',
            'penetrate_low_hp': 'bool',
            'ignore_shield': 'bool',
            'no_combat_resources': 'bool',
            'hits': 'int',
            'shield_after_shape': 'int',
            'additional_damage_bonus': 'int',
            'penetrate_brand': 'int',
            'bonus': 'int',
            'attribute': 'str',
            'support_actor': '@SupportActor',
            'awake_player_bonus': 'int',
            'low_hp_max': 'int',
            'low_hp_bonus': 'int',
            'harmony_bonus': 'int',
        }),
        'PendingSortie': _record(CAP_TINY, DICT_TAGS, {
            'spec': '@SortieSpec',
            'stay_bench': 'bool',
        }),
        'PendingChoice': _record(CAP_TINY, DICT_TAGS, {
            'side': 'str',
            'kind': 'str',
            'prompt': 'str',
            'cards': '@CardList',
            'actor': 'str',
            'continuation': 'str',
            'max_select': 'int',
            'min_select': 'int',
            'max_bottom': 'int',
            'min_bottom': 'int',
            'snapshot_n': 'int',
            'multi_bottom': 'bool',
            'hand_placement': 'bool',
            'execute_divination': 'bool',
            'deck_cards': 'bool',
            'deck_order': 'bool',
            'location_id': 'str',
            'source_instance_id': 'str',
            'scope': 'str',
            'choices': '@CardList',
        }),
        'Operation': _record(CAP_TINY, DICT_TAGS, {
            'side': 'str',
            'actor': 'str',
            'card': '@Card',
            'target_id': 'str',
            'participants': '@ParticipantList',
            'harmony': 'str',
            'first_bonus': 'int',
            'redeem_bonus': 'int',
            'record_candidate': 'bool',
            'card_mark': '@JinguMark',
            'option_id': 'str',
            'ap_paid': 'int',
            'card_bonus': 'int',
            'card_set_attack': 'int',
            'choice_cards': '@CardList',
            'release_only': 'bool',
            'additional_attack': 'bool',
            'normal_attack': 'bool',
            'skip_combat_harmony': 'bool',
            'skip_nightmare': 'bool',
            'support': 'bool',
            'support_resolved': 'bool',
            'support_notified': 'bool',
            'pending_sortie': '@PendingSortie',
            'no_counter': 'bool',
            'last_attack_target': '@SideCid',
            'last_attack_amount': 'int',
            'collapse_bonus': 'int',
            'sortie_attack_bonus': 'int',
            'return_after_resources': 'str',
            'heiyu_after_divination': 'str',
            'heiyu_after_placement': 'bool',
        }),
        'DeferredAction': _record(CAP_TINY, DICT_TAGS, {
            'side': 'str',
            'actor': 'str',
            'entity_id': 'str',
            'kind': 'str',
            'card': '@Card',
            'attribute': 'str',
            'support_actor': '@SupportActor',
        }),
        'DeferredActionList': _list(CAP_TINY, MAX_DEFERRED, '@DeferredAction'),
        'OperationList': _list(CAP_TINY, MAX_DEFERRED, '@Operation'),
    }

def _presentation_types():
    return {
        'EffectMarker': _record(CAP_SMALL, DICT_TAGS, {
            'id': 'str',
            'name': 'str',
            'timing': 'str',
            'description': 'str',
            'clears_on_down': 'bool',
            'kind': 'str',
            'attribute': 'str',
        }),
        'EffectMarkerList': _list(CAP_SMALL, MAX_MARKERS, '@EffectMarker'),
        'EscalationProgress': _record(CAP_TINY, DICT_TAGS, {
            'stage': 'int',
            'active': '@StringList',
            'hint': 'str',
            'description': 'str',
        }),
        'AttackLayers': _record(CAP_CHARACTER, DICT_TAGS, {
            'value': 'int',
            'printed': 'int',
            'weapon': 'int',
            'permanent': 'int',
            'legacy_other': 'int',
            'weapon_source': 'str',
        }),
        'AttackContribution': _record(CAP_EFFECT, DICT_TAGS, {
            'effect_id': 'str',
            'definition_id': 'str',
            'source_entity_id': 'str',
            'source_key': 'str',
            'label': 'str',
            'property': 'str',
            'op': 'str',
            'value': _one_of('int', 'float', 'bool'),
            'visibility': 'str',
            'before': _one_of('int', 'float', 'bool'),
            'after': _one_of('int', 'float', 'bool'),
        }),
        'AttackContributionList': _list(CAP_SMALL, MAX_EFFECTS, '@AttackContribution'),
        'AttackBreakdown': _record(CAP_CHARACTER, DICT_TAGS, {
            'base': 'int',
            'layers': '@AttackLayers',
            'value': 'int',
            'contributions': '@AttackContributionList',
        }),
        'StatLayers': _record(CAP_CHARACTER, DICT_TAGS, {
            'base': '@BaseStats',
            'permanent_growth': '@PermanentGrowth',
        }),
        'PublicCardFace': _record(CAP_CARD, DICT_TAGS, {
            'card_id': 'str',
            'character_id': 'str',
            'name': 'str',
            'type': 'str',
            'description': 'str',
            'stat': 'str',
            'stat_value': 'int',
            'shield': 'int',
            'attack': 'int',
            'hp': 'int',
            'attack_mode': 'str',
            'derived': 'bool',
        }),
        'PublicCard': _record(CAP_CARD, DICT_TAGS, {
            'entity_id': 'str',
            'instance_id': 'str',
            'card_id': 'str',
            'character_id': 'str',
            'name': 'str',
            'type': 'str',
            'cost': 'int',
            'terminal': 'bool',
            'additional_ap_cost': 'int',
            'description': 'str',
            'copy': 'bool',
            'expires_turn': 'int',
            'original_character_id': 'str',
            'stat': 'str',
            'stat_value': 'int',
            'shield': 'int',
            'attack': 'int',
            'hp': 'int',
            'instant': 'bool',
            'response': 'str',
            'require': 'str',
            'redeem': 'str',
            'derived': 'bool',
            'attack_mode': 'str',
            'jingu_mark': '@JinguMark',
            'play_options': '@PlayOptionList',
            'antique_investment': 'int',
            'training_excluded': 'bool',
            'portrait': 'str',
            'avatar': 'str',
            'selection_kind': 'str',
            'hand_face': '@PublicCardFace',
            'action_point_free': 'bool',
        }),
        'PublicCardList': _list(CAP_SMALL, MAX_ZONE, '@PublicCard'),
        'PublicCharacter': _record(CAP_CHARACTER, DICT_TAGS, {
            'id': 'str',
            'entity_id': 'str',
            'name': 'str',
            'avatar': 'str',
            'portrait': 'str',
            'attribute': 'str',
            'lingke_chain': '@LingkeChain',
            'summoned': 'bool',
            'resource_name': 'str',
            'faction': 'str',
            'arid': 'int',
            'truth_keys': 'int',
            'beast_fangs': 'int',
            'intimidation': 'int',
            'extra_attacks': 'int',
            'protagonist_aura': 'int',
            'passive': 'str',
            'awakened_passive': 'str',
            'flavor': 'str',
            'effects': '@EffectList',
            'stat_layers': '@StatLayers',
            'equipment': '@PublicEquipment',
            'effect_markers': '@EffectMarkerList',
            'damage_immune': 'bool',
            'damage_limit': 'int',
            'collapse_count': 'int',
            'collapse_max': 'int',
            'collapse': 'bool',
            'hp': 'int',
            'max_hp': 'int',
            'shield': 'int',
            'attack': 'int',
            'attack_breakdown': '@AttackBreakdown',
            'harmony': 'int',
            'harmony_source': 'bool',
            'harmony_ready': 'bool',
            'energy': 'int',
            'energy_max': 'int',
            'harmony_max': 'int',
            'awakened': 'bool',
            'ultimate_turns': 'int',
            'ultimate_expires': 'str',
            'down_turns': 'int',
            'growth': 'int',
            'family': 'int',
            'gaze': 'int',
            'dream': 'int',
            'jingu': 'int',
            'shape': 'str',
            'shape_id': 'str',
            'shape_description': 'str',
            'removed': 'bool',
        }),
        'PublicCharacterList': _list(CAP_TINY, MAX_CHARS, '@PublicCharacter'),
        'PublicSide': _record(CAP_PLAYER, DICT_TAGS, {
            'hp': 'int',
            'shield': 'int',
            'ap': 'int',
            'front': 'str',
            'last_front': 'str',
            'front_debuff': '@FrontDebuff',
            'max_hp': 'int',
            'entity_id': 'str',
            'life_locked': 'bool',
            'genesis_player_hits': 'int',
            'genesis_turn': 'int',
            'insight_damage': 'int',
            'insight_played': 'int',
            'support_count': 'int',
            'ultimate_remaining': 'int',
            'harmony_available': 'bool',
            'effect_markers': '@EffectMarkerList',
            'hand_count': 'int',
            'deck_count': 'int',
            'characters': '@PublicCharacterList',
            'hand': '@PublicCardList',
        }),
        'PublicSides': _record(CAP_TINY, DICT_TAGS, {
            'a': '@PublicSide',
            'b': '@PublicSide',
        }),
        'PublicBoard': _record(CAP_TINY, DICT_TAGS, {
            'escalation': '@EscalationProgress',
            'phase': 'str',
            'active_side': 'str',
            'turn': 'int',
            'winner': 'str',
            'version': 'int',
            'reason': 'str',
            'pending_choice': '@PendingChoice',
            'resolving_card': '@PublicCard',
            'sides': '@PublicSides',
        }),
        'Event': _record(CAP_EVENT, DICT_TAGS, {
            'seq': 'int',
            'type': 'str',
            'text': 'str',
            'side': 'str',
            'source': 'str',
            'actor': 'str',
            'target': 'str',
            'present': 'bool',
            'card': '@PublicCard',
            'private_side': 'str',
            'private_card': '@PublicCard',
            'private_hand': '@PublicCardList',
            'patch': '@PublicBoard',
            'action_id': 'str',
            'action_version': 'int',
            'actor_entity_id': 'str',
            'target_entity_id': 'str',
            'amount': 'int',
            'attack': 'int',
            'counter': 'int',
            'group_id': 'str',
            'harmony_delta': 'int',
            'energy_delta': 'int',
            'hit_index': 'int',
            'before': '@StringScalarMap',
            'after': '@StringScalarMap',
            'source_side': 'str',
            'card_ids': '@StringList',
            'description': 'str',
            'first_side': 'str',
            'winner': 'str',
            'reason': 'str',
            'counter_immunity': 'str',
            'response': 'bool',
            'card_source': 'str',
            'attribute': 'str',
            'additional_attack': 'bool',
            'release_only': 'bool',
            'effect_source_key': 'str',
            'attack_modifiers': '@AttackContributionList',
            'effect_id': 'str',
            'definition_id': 'str',
            'shield_removed': 'int',
        }),
        'EventList': _list(CAP_TINY, MAX_EVENTS, '@Event'),
    }

def _entity_types():
    return {
        'CardList': _list(CAP_SMALL, MAX_ZONE, '@Card'),
        'CharacterMap': _map(CAP_PLAYER, MAX_CHARS, '@Character'),
        'GameList': _list(CAP_TINY, MAX_SNAPSHOTS, '@Game'),
        'TurnSnapshotMap': _map(CAP_TINY, 2, '@GameList'),
        'TimeRemovedMap': _map(CAP_TINY, MAX_ZONE, '@Card'),
        'Card': _record(CAP_CARD, CARD_TAGS, {
            'entity_id': 'str',
            'instance_id': 'str',
            'side': 'str',
            'card_id': 'str',
            'id': 'str',
            'character_id': 'str',
            'original_character_id': 'str',
            'name': 'str',
            'type': 'str',
            'cost': 'int',
            'terminal': 'bool',
            'instant': 'bool',
            'description': 'str',
            'effect_id': 'str',
            'source': '@CardSource',
            'starter_copies': 'int',
            'shield': 'int',
            'attack': 'int',
            'hp': 'int',
            'response': 'str',
            'response_faction': 'str',
            'require': 'str',
            'playable_downed': 'bool',
            'derived': 'bool',
            'retain': 'bool',
            'copy': 'bool',
            'additional_ap_cost': 'int',
            'enemy_turn_attack': 'int',
            'attack_mode': 'str',
            'play_options': '@PlayOptionList',
            'tooltip': 'str',
            'training_excluded': 'bool',
            'spend_all_ap': 'int',
            'free_if_surplus': 'bool',
            'redeem': 'str',
            'expires_turn': 'int',
            'ephemeral_turn': 'int',
            'jingu_mark': '@JinguMark',
            '_physical_card': '@Card',
            '_k03_recovery': 'bool',
            'virtual': 'bool',
            'antique_investment': 'int',
            'stat': 'str',
            'stat_value': 'int',
            'portrait': 'str',
            'avatar': 'str',
        }),
        'Character': _record(CAP_CHARACTER, CHARACTER_TAGS, {
            'id': 'str',
            'entity_id': 'str',
            'side': 'str',
            'name': 'str',
            'attribute': 'str',
            'attack': 'int',
            'max_hp': 'int',
            'hp': 'int',
            'shield': 'int',
            'energy': 'int',
            'energy_max': 'int',
            'energy_max_override': 'int',
            'harmony': 'int',
            'awakened': 'bool',
            'ultimate_turns': 'int',
            'growth': 'int',
            'family': 'int',
            'gaze': 'int',
            'dream': 'int',
            'shape': 'str',
            'down_turns': 'int',
            'flags': '@Flags',
            'effects': '@EffectList',
            'effect_seq': 'int',
            'avatar': 'str',
            'portrait': 'str',
            'passive': 'str',
            'awakened_passive': 'str',
            'faction': 'str',
            'role': 'str',
            'everness_id': _one_of('int', 'str'),
            'ultimate': '@UltimateSpec',
            'mechanisms': '@MechanismList',
            'resource_name': 'str',
            'turn_snapshots': 'bool',
            'base_stats': '@BaseStats',
            'permanent_growth': '@PermanentGrowth',
            'base_attack': 'int',
            'base_max_hp': 'int',
            'equipment': '@Equipment',
            'equipment_seq': 'int',
            'summoned': 'bool',
            'disappeared': 'bool',
            'flavor': 'str',
            'jingu': 'int',
            'jingu_loan': 'bool',
            'lingke_chain': '@LingkeChain',
            'beast_fangs': 'int',
            'intimidation': 'int',
            'extra_attacks': 'int',
            'surplus_passive_triggers': 'int',
            'dissonance_layers': 'int',
            'insight_used_turn': 'int',
            'heal_draw_turn': 'int',
            'ultimate_used_turn': 'int',
            'last_hit_source': 'str',
            'next_damage_resistance': '@StringIntMap',
            'collapse_reductions': '@StringIntMap',
            'arid': 'int',
            'arid_contributors': '@StringList',
            'truth_keys': 'int',
            'protagonist_aura': 'int',
        }),
        'Player': _record(CAP_PLAYER, PLAYER_TAGS, {
            'entity_id': 'str',
            'side': 'str',
            'name': 'str',
            'hp': 'int',
            'max_hp': 'int',
            'shield': 'int',
            'ap': 'int',
            'front': 'str',
            'last_front': 'str',
            'order': '@StringList',
            'characters': '@CharacterMap',
            'hand': '@CardList',
            'deck': '@CardList',
            'discard': '@CardList',
            'removed': '@CardList',
            'records': '@RecordList',
            'used': '@StringBoolMap',
            'harmonized': '@StringBoolMap',
            'front_debuff': '@FrontDebuff',
            'attacked_this_turn': '@EntityIdList',
            'effects': '@EffectList',
            'effect_seq': 'int',
            'flags': '@Flags',
            'fatigue': 'int',
            'turn_count': 'int',
            'extra_ap': 'int',
            'delay_turns': 'int',
            'delay_counted_turn': 'int',
            'genesis_player_hits': 'int',
            'genesis_turn': 'int',
            'genesis_damage_bonus': 'int',
            'nanali_family': 'int',
            'nanali_family_played': 'int',
            'ultimates_used': 'int',
            'heiyu_divinations': 'int',
            'dark_star_count': 'int',
            'insight_bonus': 'int',
            'insight_played': 'int',
            'support_count': 'int',
            'support_turn': 'int',
            'canhong_collapse_used': 'bool',
            'normal_attack_available': 'bool',
            'ultimate_available': 'bool',
            'extra_flower': 'bool',
            'extra_delay': 'bool',
            'surplus': 'bool',
            'harmony_damage': 'bool',
            'surplus_ever': 'bool',
            'empty_draw_wins': 'bool',
            'skip_normal_attack': 'bool',
            'extra_genesis_pending': 'bool',
            'life_locked': 'bool',
            'harmony_available': 'bool',
            'extra_genesis_actor': 'str',
            'seal_normal_attack': 'str',
            'next_damage_resistance': '@StringIntMap',
            'revealed_ids': '@StringList',
        }),
        'Sides': _record(CAP_TINY, DICT_TAGS, {
            'a': '@Player',
            'b': '@Player',
        }),
        'Game': _record(CAP_GAME, DICT_TAGS, {
            'rules_version': 'str',
            'design_version': 'str',
            'version': 'int',
            'phase': 'str',
            'active_side': 'str',
            'first_side': 'str',
            'turn': 'int',
            'winner': 'str',
            'reason': 'str',
            'rng': 'int',
            'next_instance': 'int',
            'next_entity': 'int',
            'event_seq': 'int',
            'events': '@EventList',
            'pending_choice': '@PendingChoice',
            'operation': '@Operation',
            'mulligan_done': '@SideList',
            'current_action_id': 'str',
            'current_action_version': 'int',
            'combat_group': 'int',
            'sides': '@Sides',
            'escalation_enabled': 'bool',
            'harmony_residue_version': 'int',
            'deferred_actions': '@DeferredActionList',
            'turn_start_pending': 'str',
            '_response_lock': 'bool',
            '_response_intercept': 'bool',
            '_draining_actions': 'bool',
            '_pending_down_responses': '@DownResponseList',
            '_deferred_card_marks': '@OperationList',
            '_public_board': '@PublicBoard',
            'turn_snapshots': '@TurnSnapshotMap',
            'time_removed': '@TimeRemovedMap',
        }),
    }

_SCALAR_NAMES = frozenset({'int', 'bool', 'float', 'str', 'number', 'scalar'})
_CONTAINER_KINDS = frozenset({'record', 'list', 'map', 'tuple'})


def _merge_types():
    merged = {}
    for part in (
        _leaf_types(),
        _flag_and_zone_types(),
        _flow_types(),
        _presentation_types(),
        _entity_types(),
    ):
        overlap = set(merged) & set(part)
        if overlap:
            raise RuntimeError('Duplicate domain type names: ' + ', '.join(sorted(overlap)))
        merged.update(part)
    if ROOT not in merged:
        raise RuntimeError('Root type Game is missing')
    return merged


def domain_types():
    """Frozen mapping of type-name -> record/list/map/tuple definition."""
    return MappingProxyType(_merge_types())


def _iter_type_exprs(expression, path):
    if isinstance(expression, str):
        yield path, expression
        return
    if isinstance(expression, dict) and set(expression) == {'one_of'}:
        for index, choice in enumerate(expression['one_of']):
            yield from _iter_type_exprs(choice, path + (f'one_of[{index}]',))
        return
    raise TypeError(f'Illegal type expression at {".".join(path)}: {expression!r}')


def unresolved_references(types=None):
    types = dict(domain_types() if types is None else types)
    missing = []
    for name, definition in types.items():
        kind = definition.get('kind')
        if kind == 'record':
            items = definition.get('fields') or {}
            for field, expr in items.items():
                for path, token in _iter_type_exprs(expr, (name, field)):
                    if token.startswith('@') and token[1:] not in types:
                        missing.append('.'.join(path) + '->' + token)
        elif kind in ('list', 'map'):
            for path, token in _iter_type_exprs(definition.get('value'), (name, 'value')):
                if token.startswith('@') and token[1:] not in types:
                    missing.append('.'.join(path) + '->' + token)
        elif kind == 'tuple':
            for index, expr in enumerate(definition.get('items') or ()):
                for path, token in _iter_type_exprs(expr, (name, f'[{index}]')):
                    if token.startswith('@') and token[1:] not in types:
                        missing.append('.'.join(path) + '->' + token)
    return tuple(missing)


def scalar_container_escapes(types=None):
    """Reject scalars used as containers, and containers hidden in scalar names."""
    types = dict(domain_types() if types is None else types)
    escapes = []
    for name, definition in types.items():
        kind = definition.get('kind')
        if kind not in _CONTAINER_KINDS:
            escapes.append(f'{name}: unknown kind {kind!r}')
            continue
        if kind == 'record':
            for field, expr in (definition.get('fields') or {}).items():
                for path, token in _iter_type_exprs(expr, (name, field)):
                    if token in _SCALAR_NAMES:
                        continue
                    if token.startswith('@'):
                        continue
                    escapes.append('.'.join(path) + ': ' + repr(token))
        elif kind in ('list', 'map'):
            for path, token in _iter_type_exprs(definition.get('value'), (name, 'value')):
                if token in _SCALAR_NAMES or token.startswith('@'):
                    continue
                escapes.append('.'.join(path) + ': ' + repr(token))
            if kind == 'map' and definition.get('key') != 'str':
                escapes.append(f'{name}.key: maps must use str keys, got {definition.get("key")!r}')
        elif kind == 'tuple':
            for index, expr in enumerate(definition.get('items') or ()):
                for path, token in _iter_type_exprs(expr, (name, f'[{index}]')):
                    if token in _SCALAR_NAMES or token.startswith('@'):
                        continue
                    escapes.append('.'.join(path) + ': ' + repr(token))
    return tuple(escapes)


def compile_domain_schema(**options):
    from app.modules.card_game.rl.batched_duel.record_soa import compile_schema
    return compile_schema(dict(domain_types()), root=ROOT, **options)


def compile_execution_schema(*, pool_scale=8, capacities=None, **options):
    """Bounded rule workspace, including temporary canonical event snapshots.

    Persistent slots are collected between calls. Each declared pool has spare
    records during a call for deepcopy, events and public projection; list
    lengths and all field/type contracts remain exact and unchanged.
    """
    if type(pool_scale) is not int or not 1 <= pool_scale <= 64:
        raise ValueError('Explicit bounded rule pool scale required')
    types=dict(domain_types())
    sizes={name:definition['capacity']*pool_scale for name,definition in types.items()}
    # A chained attack can generate many public snapshots before the call's
    # events are drained. This pool needs more headroom than live characters.
    for name in ('PublicCharacter','BaseStats','PermanentGrowth','StatLayers',
                 'AttackBreakdown','AttackLayers','AttackContributionList',
                 'EffectList','EffectMarkerList','PublicEquipment'):
        sizes[name]=max(sizes[name],1024)
    sizes.update(capacities or {})
    return compile_domain_schema(capacities=sizes,**options)


def coverage():
    types = domain_types()
    return {
        'supports_full_duel_rules': False,
        'complete_soa': False,
        'full_public17': False,
        'schema_root': ROOT,
        'type_count': len(types),
        'declared_types': tuple(types),
        'unresolved_references': unresolved_references(types),
        'scalar_container_escapes': scalar_container_escapes(types),
    }

_ENTITY_TYPES = {
    'PlayerEntity': 'Player',
    'CharacterEntity': 'Character',
    'CardEntity': 'Card',
}


def _expr_tokens(expression):
    if isinstance(expression, str):
        return (expression,)
    if isinstance(expression, dict) and set(expression) == {'one_of'}:
        tokens = []
        for choice in expression['one_of']:
            tokens.extend(_expr_tokens(choice))
        return tuple(tokens)
    raise TypeError(f'Illegal type expression {expression!r}')


def _is_scalar(token, value):
    if token == 'int':
        return type(value) is int
    if token == 'bool':
        return type(value) is bool
    if token == 'float':
        return type(value) is float
    if token == 'str':
        return type(value) is str
    if token == 'number':
        return type(value) is int or type(value) is float
    if token == 'scalar':
        return type(value) in (bool, int, float, str)
    return False


def undeclared_fields(value, type_name=ROOT, *, types=None, path=None):
    """Return closed-vocab misses in a live object graph. None/absent are legal."""
    types = dict(domain_types() if types is None else types)
    found = []

    def add(current, message):
        found.append((current if current else ROOT) + ': ' + message)

    def walk(current_value, expression, current):
        if current_value is None:
            return
        tokens = _expr_tokens(expression)
        refs = tuple(token[1:] for token in tokens if token.startswith('@'))
        scalars = tuple(token for token in tokens if token in _SCALAR_NAMES)
        cls_name = type(current_value).__name__
        if cls_name in _ENTITY_TYPES:
            expected = _ENTITY_TYPES[cls_name]
            if expected not in refs:
                add(current, f'{cls_name} is not declared by {tokens}')
            walk_record(current_value, expected, current)
            return
        if isinstance(current_value, dict):
            for name in refs:
                definition = types[name]
                if definition['kind'] == 'record':
                    walk_record(current_value, name, current)
                    return
                if definition['kind'] == 'map':
                    if any(type(key) is not str for key in current_value):
                        add(current, 'map keys must be str')
                    for key, nested in current_value.items():
                        walk(nested, definition['value'], current + f'[{key}]')
                    return
            add(current, f'untyped dict tokens={tokens}')
            return
        if isinstance(current_value, (list, tuple)):
            for name in refs:
                definition = types[name]
                if definition['kind'] == 'list':
                    for index, nested in enumerate(current_value):
                        walk(nested, definition['value'], current + f'[{index}]')
                    return
                if definition['kind'] == 'tuple':
                    items = definition['items']
                    if len(current_value) != len(items):
                        add(current, f'tuple length {len(current_value)} != {len(items)}')
                        return
                    for index, (nested, expr) in enumerate(zip(current_value, items)):
                        walk(nested, expr, current + f'[{index}]')
                    return
            add(current, f'untyped sequence tokens={tokens}')
            return
        if any(_is_scalar(token, current_value) for token in scalars):
            return
        add(current, f'{cls_name} is not declared by {tokens}')

    def walk_record(current_value, name, current):
        fields = types[name]['fields']
        for key in current_value:
            if key not in fields:
                add(current, f'undeclared {name}.{key}')
            else:
                walk(current_value[key], fields[key], current + '.' + str(key))

    walk(value, '@' + type_name, path or type_name)
    return tuple(found)
