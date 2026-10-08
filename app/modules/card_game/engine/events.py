from enum import Enum

V2_PRESENTATION_SCHEMA = 1


class GameEvent(str, Enum):
    # 这里统一声明“引擎感知到的规则时机”。
    # 业务层和内容层都应尽量引用这些常量，而不是在各处手写裸字符串。
    # 回合与牌桌推进
    TURN_BEGIN = 'turn_begin'
    TURN_END = 'turn_end'
    CARD_PLAYED = 'card_played'
    CARD_REVEALED = 'card_revealed'
    ESPER_RESONATED = 'esper_resonated'
    MATERIAL_CONSUMED = 'material_consumed'
    LOCATION_REVEALED = 'location_revealed'
    LOCATION_SCORED = 'location_scored'
    HARMONY_MARK_ADDED = 'harmony_mark_added'
    # 对局内数值包
    DAMAGE_PACKET = 'damage_packet'
    # 对局结果
    RUN_VICTORY = 'run_victory'
    RUN_DEFEAT = 'run_defeat'
    # V2 content callback timings. Kept separate from public log categories.
    V2_GROWTH_GAINED = 'v2_growth_gained'
    V2_SORTIE_FINISHED = 'v2_sortie_finished'
    V2_OPERATION_RESOURCES_RESOLVED = 'after_operation_resources'
    V2_COMBAT_HP_LOST = 'v2_combat_hp_lost'
    V2_HARMONY_RESOLVED = 'v2_harmony_resolved'
    V2_FLOWER_RESOLVED = 'v2_flower_resolved'
    V2_COPY_FINISHED = 'v2_copy_finished'
    V2_RECORD_REDEEMED = 'v2_record_redeemed'
    V2_GAME_START = 'v2_game_start'
    V2_ROUT = 'v2_rout'
    V2_SURPLUS = 'on_surplus'
    V2_DEFERRED_ACTION = 'on_deferred_action'
    V2_ULTIMATE_EXPIRING = 'on_ultimate_expiring'  # Legacy name; no longer dispatched.
    V2_EFFECT_EXPIRED = 'on_effect_expired'
    V2_LEAVE_FRONT = 'on_leave_front'
    V2_TURN_COUNTDOWN = 'on_turn_countdown'



REPLACEABLE_EVENTS = frozenset({
    GameEvent.TURN_BEGIN.value,
})
