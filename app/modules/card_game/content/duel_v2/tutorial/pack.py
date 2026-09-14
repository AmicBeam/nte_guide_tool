"""Tutorial-only character/card tables and campaign scenarios."""
from copy import deepcopy

from app.modules.card_game.content.duel_v2.catalog import CHARACTERS, energy_max_for_attribute

_ART = {
    'protagonist': {
        'avatar': '/static/images/characters/avatar/鉴定师.webp',
        'portrait': '/static/images/characters/portrait/鉴定师.webp',
    },
    'bohe': {
        'avatar': '/static/images/characters/avatar/薄荷.webp',
        'portrait': '/static/images/characters/portrait/薄荷.webp',
    },
    'nanali': {
        'avatar': '/static/images/characters/avatar/娜娜莉.webp',
        'portrait': '/static/images/characters/portrait/娜娜莉.webp',
    },
    'zaowu': {
        'avatar': '/static/images/characters/avatar/早雾.webp',
        'portrait': '/static/images/characters/portrait/早雾.webp',
    },
    'iloy': {
        'avatar': '/static/kongmu/images/characters/player_yiluoyi_256.webp',
        'portrait': '/static/images/characters/portrait/伊洛伊.webp',
    },
    'jiuyuan': {
        'avatar': '/static/images/characters/avatar/九原.webp',
        'portrait': '/static/images/characters/portrait/九原.webp',
    },
}


def _esper(cid, name, attribute, attack, max_hp, art, passive='', awakened_passive=''):
    return {
        'id': cid,
        'name': name,
        'attribute': attribute,
        'attack': attack,
        'max_hp': max_hp,
        'energy_max': energy_max_for_attribute(attribute),
        'avatar': _ART[art]['avatar'],
        'portrait': _ART[art]['portrait'],
        'passive': passive,
        'awakened_passive': awakened_passive,
        'ultimate': {'kind': 'status', 'turns': 2},
    }


TUTORIAL_CHARACTERS = {
    'tutorial_protagonist': _esper('tutorial_protagonist', '主角', '光', 2, 5, 'protagonist'),
    'tutorial_bohe': _esper(
        'tutorial_bohe', '薄荷', '灵', 3, 4, 'bohe',
        passive=CHARACTERS['bohe']['passive'],
        awakened_passive=CHARACTERS['bohe']['awakened_passive'],
    ),
    'tutorial_nanali': _esper('tutorial_nanali', '娜娜莉', '灵', 2, 5, 'nanali'),
    'tutorial_zaowu': _esper('tutorial_zaowu', '早雾', '咒', 3, 4, 'zaowu'),
    'tutorial_iloy': _esper('tutorial_iloy', '伊洛伊', '灵', 1, 6, 'iloy'),
    'tutorial_jiuyuan': _esper('tutorial_jiuyuan', '九原', '灵', 3, 4, 'jiuyuan'),
}

TUTORIAL_CARDS = {
    'tutorial_z03': {
        'id': 'tutorial_z03', 'card_id': 'tutorial_z03', 'character_id': 'tutorial_protagonist',
        'name': '奇异记叙', 'type': 'battle', 'cost': 1, 'terminal': False,
        'description': '', 'effect_id': 'tutorial_z03', 'attack': 0, 'shield': 0,
        'source': {'character_id': 'tutorial_protagonist', 'type': 'tutorial', 'index': 3},
        'tutorial_only': True,
    },
    'tutorial_z_heal': {
        'id': 'tutorial_z_heal', 'card_id': 'tutorial_z_heal', 'character_id': 'tutorial_protagonist',
        'name': '初明凝视', 'type': 'tactic', 'cost': 1, 'terminal': False,
        'description': '选择另一名己方存活异能者。回复其 2 点生命。',
        'effect_id': 'tutorial_z_heal',
        'source': {'character_id': 'tutorial_protagonist', 'type': 'tutorial', 'index': 1},
        'tutorial_only': True,
    },
    'tutorial_z08': {
        'id': 'tutorial_z08', 'card_id': 'tutorial_z08', 'character_id': 'tutorial_protagonist',
        'name': '倾世之雨', 'type': 'form', 'cost': 1, 'terminal': False,
        'description': '', 'effect_id': 'tutorial_z08', 'attack': 3, 'hp': 6,
        'source': {'character_id': 'tutorial_protagonist', 'type': 'tutorial', 'index': 8},
        'tutorial_only': True,
    },
    'tutorial_z_instant': {
        'id': 'tutorial_z_instant', 'card_id': 'tutorial_z_instant',
        'character_id': 'tutorial_protagonist', 'name': '探悉天职', 'type': 'tactic',
        'cost': 1, 'terminal': False, 'description': '瞬发。对对方玩家造成 1 点伤害。',
        'effect_id': 'tutorial_z_instant', 'instant': True,
        'source': {'character_id': 'tutorial_protagonist', 'type': 'tutorial', 'index': 2},
        'tutorial_only': True,
    },
    'tutorial_m_response': {
        'id': 'tutorial_m_response', 'card_id': 'tutorial_m_response',
        'character_id': 'tutorial_bohe', 'name': '二次补考', 'type': 'battle',
        'cost': 1, 'terminal': False,
        'description': '响应。己方其他异能者被攻击时：薄荷进入战斗区，作为反击方。',
        'effect_id': 'tutorial_m_response', 'response': 'ally_attacked', 'attack': 0, 'shield': 0,
        'source': {'character_id': 'tutorial_bohe', 'type': 'tutorial', 'index': 2},
        'tutorial_only': True,
    },
    'tutorial_m_strike': {
        'id': 'tutorial_m_strike', 'card_id': 'tutorial_m_strike', 'character_id': 'tutorial_bohe',
        'name': '特遣行动', 'type': 'battle', 'cost': 1, 'terminal': False,
        'description': '', 'effect_id': 'tutorial_m_strike', 'attack': 0, 'shield': 0,
        'source': {'character_id': 'tutorial_bohe', 'type': 'tutorial', 'index': 1},
        'tutorial_only': True,
    },
}


def lookup_character(cid):
    if cid in TUTORIAL_CHARACTERS:
        return TUTORIAL_CHARACTERS[cid]
    return CHARACTERS[cid]


def lookup_card(card_id):
    if not card_id:
        return None
    if card_id in TUTORIAL_CARDS:
        return TUTORIAL_CARDS[card_id]
    from app.modules.card_game.content.duel_v2.catalog import CARDS
    return CARDS.get(card_id)


CAMPAIGN_LEVELS = (
    'tutorial_l01_board',
    'tutorial_l02_ap',
    'tutorial_l03_hand',
    'tutorial_l04_tactic',
    'tutorial_l05_down',
    'tutorial_l06_form',
    'tutorial_l07_harmony',
    'tutorial_l08_energy',
    'tutorial_l09_instant',
    'tutorial_l10_response',
    'tutorial_l11_pierce',
    'tutorial_l12_roster4',
)

LEVEL_META = (
    {'id': 'tutorial_l01_board', 'title': '第一关 牌桌与回合', 'short': '牌桌与回合'},
    {'id': 'tutorial_l02_ap', 'title': '第二关 行动力', 'short': '行动力'},
    {'id': 'tutorial_l03_hand', 'title': '第三关 手牌', 'short': '手牌'},
    {'id': 'tutorial_l04_tactic', 'title': '第四关 战术', 'short': '战术'},
    {'id': 'tutorial_l05_down', 'title': '第五关 倒地', 'short': '倒地'},
    {'id': 'tutorial_l06_form', 'title': '第六关 武备', 'short': '武备'},
    {'id': 'tutorial_l07_harmony', 'title': '第七关 环合', 'short': '环合'},
    {'id': 'tutorial_l08_energy', 'title': '第八关 终结', 'short': '终结'},
    {'id': 'tutorial_l09_instant', 'title': '第九关 瞬发', 'short': '瞬发'},
    {'id': 'tutorial_l10_response', 'title': '第十关 响应', 'short': '响应'},
    {'id': 'tutorial_l11_pierce', 'title': '第十一关 贯穿', 'short': '贯穿'},
    {'id': 'tutorial_l12_roster4', 'title': '第十二关 四人编队', 'short': '四人编队'},
)

_L01_HIDE = [
    'passive', 'harmony', 'energy', 'ultimate', 'hand', 'deck', 'mulligan', 'form',
]

_L01_FLAGS = {
    'disable_hand': True,
    'disable_draw': True,
    'disable_mulligan': True,
    'disable_passives': True,
    'disable_ultimate': True,
    'disable_harmony_gain': True,
    'disable_energy_gain': True,
    'disable_harmony_trigger': True,
    'disable_collapse': True,
    'disable_overflow': True,
    'enabled_kits': [],
    'ap_per_turn': 1,
    'player_hp': 4,
    'player_shield': 0,
    'character_slots': 2,
    'skip_mulligan': True,
    'fixed_rng': True,
    'hide': list(_L01_HIDE),
}

_L01_STEPS = [
    {
        'id': 'l01_intro',
        'ack_required': True,
        'mask': True,
        'placement': 'center',
        'spotlights': ['bench:a', 'bench:b'],
        'modal': {
            'title': '欢迎来到异象对决',
            'body': '异象对决是一款回合制卡牌对战。双方轮流行动，把异能者送进战斗区出击，把对方玩家生命打到 0 就赢。先认识牌桌，这一关还不用马上开打。',
            'primary': '看看战场',
        },
        'allowed': [{'type': 'tutorial_ack'}],
        'blocked_reason': '先看看双方备战区。',
    },
    {
        'id': 'l01_bench',
        'ack_required': True,
        'mask': True,
        'placement': 'bottom-right',
        'spotlights': ['bench:a', 'bench:b'],
        'modal': {
            'title': '备战区',
            'body': '开局 4 名异能者都在备战区。这一关每方只有 2 人，方便看清位置。稍后从这里把人拖进战斗区。',
            'primary': '下一处',
        },
        'allowed': [{'type': 'tutorial_ack'}],
        'blocked_reason': '先看备战区。',
    },
    {
        'id': 'l01_front',
        'ack_required': True,
        'mask': True,
        'placement': 'bottom-right',
        'spotlights': ['zone:a:front', 'zone:b:front'],
        'modal': {
            'title': '战斗区',
            'body': '每方只有 1 个战斗区。站在里面的人叫前排。现在是空的，所以没有前排。',
            'primary': '下一处',
        },
        'allowed': [{'type': 'tutorial_ack'}],
        'blocked_reason': '先看双方战斗区。',
    },
    {
        'id': 'l01_life',
        'ack_required': True,
        'mask': True,
        'placement': 'bottom-right',
        'spotlights': ['player_hp:a', 'player_hp:b'],
        'modal': {
            'title': '玩家生命',
            'body': '胜负看左下角和左上角的玩家生命，不是把角色攻击加在一起。把对方玩家生命打到 0 就赢。',
            'primary': '下一处',
        },
        'allowed': [{'type': 'tutorial_ack'}],
        'blocked_reason': '先看双方玩家生命。',
    },
    {
        'id': 'l01_ap',
        'ack_required': True,
        'mask': True,
        'placement': 'bottom-right',
        'spotlights': ['ap:a'],
        'modal': {
            'title': '行动力',
            'body': '右下角是你的行动力。出击和出牌都会消耗它。这一关每回合只有 1 点，刚好够行动一次。',
            'primary': '下一处',
        },
        'allowed': [{'type': 'tutorial_ack'}],
        'blocked_reason': '先看行动力。',
    },
    {
        'id': 'l01_turn',
        'ack_required': True,
        'mask': True,
        'placement': 'bottom-right',
        'spotlights': ['turn', 'end_turn'],
        'modal': {
            'title': '回合',
            'body': '现在是你的回合。做完这一回合的行动后，点「结束回合」交给对方。双方轮流，不会同时出招。',
            'primary': '开始出击',
        },
        'allowed': [{'type': 'tutorial_ack'}],
        'blocked_reason': '先看回合和结束回合。',
    },
    {
        'id': 'l01_p1_attack',
        'ack_required': False,
        'mask': True,
        'placement': 'bottom-right',
        'spotlights': [
            'character:a:tutorial_protagonist', 'zone:b:front', 'player_hp:b',
        ],
        'drag_arrow': {
            'from': 'character:a:tutorial_protagonist',
            'to': 'zone:b:front',
        },
        'modal': {
            'title': '把异能者拖进战斗区',
            'body': '把「主角」拖进战斗区就是出击。对方没有前排时，伤害打在对方玩家身上。这一回合还能行动 1 次。',
            'primary': '拖动主角',
        },
        'allowed': [{'type': 'attack', 'character_id': 'tutorial_protagonist'}],
        'blocked_reason': '先让主角出击。',
    },
    {
        'id': 'l01_p1_end',
        'ack_required': False,
        'mask': True,
        'placement': 'bottom-right',
        'spotlights': ['end_turn'],
        'click_cue': 'end_turn',
        'modal': {
            'title': '结束回合',
            'body': '行动做完就结束回合，交给对方。主角会留在前排防守，直到你的下一回合开始才返回备战区。',
            'primary': '结束回合',
        },
        'allowed': [{'type': 'end_turn'}],
        'opponent_then': [
            {'type': 'attack', 'character_id': 'tutorial_nanali'},
            {'type': 'end_turn'},
        ],
        'blocked_reason': '先结束回合。',
    },
    {
        'id': 'l01_o1_watch',
        'ack_required': True,
        'mask': True,
        'placement': 'bottom-right',
        'spotlights': [
            'character:a:tutorial_protagonist', 'character:b:tutorial_nanali', 'log',
        ],
        'modal': {
            'title': '有前排就打前排',
            'body': '对方把娜娜莉拖进战斗区，先打你的前排。双方同时受伤。玩家生命没变。',
            'primary': '继续',
        },
        'allowed': [{'type': 'tutorial_ack'}],
        'blocked_reason': '先看刚才的交战。',
    },
    {
        'id': 'l01_p2_return',
        'ack_required': True,
        'mask': True,
        'placement': 'bottom-right',
        'spotlights': ['bench:a', 'character:b:tutorial_nanali'],
        'modal': {
            'title': '回合开始，前排返回',
            'body': '到你的回合开始，你的前排会自动返回备战区。对方前排还留着防守。',
            'primary': '再出击',
        },
        'allowed': [{'type': 'tutorial_ack'}],
        'blocked_reason': '先看主角回到备战区。',
    },
    {
        'id': 'l01_p2_attack',
        'ack_required': False,
        'mask': True,
        'placement': 'bottom-right',
        'spotlights': ['character:a:tutorial_protagonist', 'character:b:tutorial_nanali'],
        'drag_arrow': {
            'from': 'character:a:tutorial_protagonist',
            'to': 'character:b:tutorial_nanali',
        },
        'modal': {
            'title': '打对方前排',
            'body': '再让主角出击。有前排时先打前排，双方同时造成攻击伤害。',
            'primary': '拖动主角',
        },
        'allowed': [{'type': 'attack', 'character_id': 'tutorial_protagonist'}],
        'blocked_reason': '这一步请让主角出击。',
    },
    {
        'id': 'l01_p2_end',
        'ack_required': False,
        'mask': True,
        'placement': 'bottom-right',
        'spotlights': ['end_turn'],
        'click_cue': 'end_turn',
        'modal': {
            'title': '结束回合',
            'body': '对方这一回合不会再出击。注意对方战斗区空了。',
            'primary': '结束回合',
        },
        'allowed': [{'type': 'end_turn'}],
        'opponent_then': [{'type': 'end_turn'}],
        'blocked_reason': '先结束回合。',
    },
    {
        'id': 'l01_p3_win',
        'ack_required': False,
        'mask': True,
        'placement': 'bottom-right',
        'spotlights': ['character:a:tutorial_bohe', 'zone:b:front', 'player_hp:b'],
        'drag_arrow': {
            'from': 'character:a:tutorial_bohe',
            'to': 'zone:b:front',
        },
        'modal': {
            'title': '空前排打玩家生命',
            'body': '对方没有前排。让攻击更高的薄荷出击，伤害会打在对方玩家身上。',
            'primary': '拖动薄荷',
        },
        'allowed': [{'type': 'attack', 'character_id': 'tutorial_bohe'}],
        'blocked_reason': '这一步请让薄荷出击。',
    },
]

SCENARIOS = {
    'tutorial_l01_board': {
        'id': 'tutorial_l01_board',
        'title': '第一关 牌桌与回合',
        'level_index': 1,
        'seed': 101,
        'first_side': 'a',
        'order_a': ['tutorial_protagonist', 'tutorial_bohe'],
        'order_b': ['tutorial_nanali', 'tutorial_zaowu'],
        'flags': _L01_FLAGS,
        'steps': _L01_STEPS,
        'playable': True,
    },
}


def load_scenario(scenario_id):
    scenario = SCENARIOS.get(scenario_id)
    if scenario is None:
        raise ValueError('未知的教学关卡。')
    if not scenario.get('playable'):
        raise ValueError('这一关尚未开放。')
    return deepcopy(scenario)


def next_level(completed_levels):
    done = set(completed_levels or [])
    for level in CAMPAIGN_LEVELS:
        if level not in done:
            return level
    return CAMPAIGN_LEVELS[-1]
