"""Tutorial-only character/card tables and campaign scenarios."""
from copy import deepcopy

from app.modules.card_game.content.duel_v2.catalog import CHARACTERS, energy_max_for_attribute

_ART = {
    'protagonist': {
        'avatar': '/static/images/characters/avatar/鉴定师.webp',
        'portrait': '/static/images/characters/portrait/鉴定师.webp',
    },
    'haiyue': {
        'avatar': '/static/images/characters/avatar/海月.webp',
        'portrait': '/static/images/characters/portrait/海月.webp',
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
    'baicang': {
        'avatar': '/static/images/characters/avatar/白藏.webp',
        'portrait': '/static/images/characters/portrait/白藏.webp',
    },
    'canhong': {
        'avatar': '/static/images/characters/avatar/残虹.png',
        'portrait': '/static/images/characters/portrait/残虹.webp',
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
    'tutorial_protagonist': _esper(
        'tutorial_protagonist', '主角', '光', 2, 5, 'protagonist',
        passive='主角使用战斗牌后，立刻充满自己的环合值。',
        awakened_passive=CHARACTERS['zero']['awakened_passive'],
    ),
    'tutorial_bohe': _esper(
        'tutorial_bohe', '薄荷', '灵', 3, 4, 'bohe',
        passive=CHARACTERS['bohe']['passive'],
        awakened_passive=CHARACTERS['bohe']['awakened_passive'],
    ),
    'tutorial_haiyue': {
        **_esper('tutorial_haiyue', '海月', '魂', 3, 4, 'haiyue',
                 awakened_passive=CHARACTERS['haiyue']['awakened_passive']),
        'ultimate': {'kind': 'status', 'turns': 1, 'expires': 'turn_end'},
    },
    'tutorial_nanali': _esper('tutorial_nanali', '娜娜莉', '灵', 2, 5, 'nanali'),
    'tutorial_zaowu': _esper('tutorial_zaowu', '早雾', '咒', 3, 4, 'zaowu'),
    'tutorial_iloy': _esper('tutorial_iloy', '伊洛伊', '灵', 1, 6, 'iloy'),
    'tutorial_jiuyuan': _esper('tutorial_jiuyuan', '九原', '灵', 3, 4, 'jiuyuan'),
    'tutorial_baicang': _esper('tutorial_baicang', '白藏', '光', 2, 5, 'baicang'),
    'tutorial_canhong': _esper('tutorial_canhong', '残虹', '暗', 3, 4, 'canhong'),
}

TUTORIAL_CARDS = {
    'tutorial_z03': {
        'id': 'tutorial_z03', 'card_id': 'tutorial_z03', 'character_id': 'tutorial_protagonist',
        'name': '奇异记叙', 'type': 'battle', 'cost': 1, 'terminal': False,
        'description': '这一张战斗牌。', 'effect_id': 'tutorial_z03', 'attack': 1, 'shield': 0,
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
        'description': '这一张武备牌。', 'effect_id': 'tutorial_z08', 'attack': 3, 'hp': 6,
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
    'tutorial_l02_hand',
    'tutorial_l04_tactic',
    'tutorial_l05_down',
    'tutorial_l06_form',
    'tutorial_l07_harmony',
    'tutorial_l08_energy',
    'tutorial_l09_instant',
    'tutorial_l10_response',
    'tutorial_l11_pierce',
    'tutorial_l12_roster4',
    'tutorial_l13_escalation',
    'tutorial_l14_ranged',
)

LEVEL_META = (
    {'id': 'tutorial_l01_board', 'title': '第一关 牌桌与回合', 'short': '牌桌与回合'},
    {'id': 'tutorial_l02_hand', 'title': '第二关 手牌', 'short': '手牌'},
    {'id': 'tutorial_l04_tactic', 'title': '第三关 战术', 'short': '战术'},
    {'id': 'tutorial_l05_down', 'title': '第四关 倒地', 'short': '倒地'},
    {'id': 'tutorial_l06_form', 'title': '第五关 武备', 'short': '武备'},
    {'id': 'tutorial_l07_harmony', 'title': '第六关 环合', 'short': '环合'},
    {'id': 'tutorial_l08_energy', 'title': '第七关 终结', 'short': '终结'},
    {'id': 'tutorial_l09_instant', 'title': '第八关 瞬发', 'short': '瞬发'},
    {'id': 'tutorial_l10_response', 'title': '第九关 响应', 'short': '响应'},
    {'id': 'tutorial_l11_pierce', 'title': '第十关 穿透', 'short': '穿透'},
    {'id': 'tutorial_l12_roster4', 'title': '第十一关 四人编队', 'short': '四人编队'},
    {'id': 'tutorial_l13_escalation', 'title': '第十二关 白热化', 'short': '白热化'},
    {'id': 'tutorial_l14_ranged', 'title': '第十三关 远程', 'short': '远程'},
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
            'title': '欢迎来到异能对决',
            'body': '异能对决是一款回合制卡牌对战。双方轮流行动，把异能者送进战斗区出击，把对方玩家生命打到 0 就赢。先认识牌桌，这一关还不用马上开打。',
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
            'body': '开局角色都在备战区。新手教学中，每方玩家为2名角色，常规对战中为4名。稍后从这里把人拖进战斗区。',
            'primary': '下一处',
        },
        'allowed': [{'type': 'tutorial_ack'}],
        'blocked_reason': '先看备战区。',
    },
    {
        'id': 'l01_opponent_front',
        'ack_required': True,
        'mask': True,
        'placement': 'right',
        'spotlights': ['zone:b:front'],
        'modal': {
            'title': '对手战斗区',
            'body': '上方是对手的战斗区，最多容纳 1 名存活异能者，站在这里的角色叫对手前排。你的出击优先攻击对手前排；这里为空时，直接攻击对手玩家。',
            'primary': '下一处',
        },
        'allowed': [{'type': 'tutorial_ack'}],
        'blocked_reason': '先看对手战斗区。',
    },
    {
        'id': 'l01_player_front',
        'ack_required': True,
        'mask': True,
        'placement': 'right',
        'spotlights': ['zone:a:front'],
        'modal': {
            'title': '我方战斗区',
            'body': '下方是你的战斗区，同样最多容纳 1 名存活异能者。从备战区拖入角色即可出击。出击后，角色留在这里作为我方前排，抵挡对手接下来的进攻。',
            'primary': '下一处',
        },
        'allowed': [{'type': 'tutorial_ack'}],
        'blocked_reason': '先看我方战斗区。',
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
            'body': '右侧面板会显示你的剩余行动力，也就是费用。先攻第一回合有1点，其余回合开始时会变为2点。',
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
            'character:a:tutorial_protagonist', 'play_area',
        ],
        'drag_arrow': {
            'from': 'character:a:tutorial_protagonist',
            'to': 'play_area',
        },
        'modal': {
            'title': '把异能者拖进战斗区',
            'body': '把备战区的异能者拖到战斗区就是通常出击。这会花费1点行动力，且每回合限1次。对方没有前排时，伤害会直接打在对方玩家身上。',
            'primary': '拖动主角',
        },
        'cue': '按住主角，并拖动到对战区',
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
        'cue': '点击「结束回合」',
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
            'body': '战斗时，如果对方有前排角色，就会与其互相攻击，不会伤害到玩家。',
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
            'title': '回合开始',
            'body': '到你的回合开始时，自己的前排会自动返回备战区。对方前排仍留在战斗区防守。',
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
            'body': '再让主角出击。对方有前排时，会与其互相攻击，不会打到玩家。',
            'primary': '拖动主角',
        },
        'cue': '按住主角，并拖动到对战区',
        'allowed': [{'type': 'attack', 'character_id': 'tutorial_protagonist'}],
        'blocked_reason': '这一步请让主角出击。',
        'finish_level': True,
    },
]

_BOARD = ['tutorial_protagonist', 'tutorial_bohe']
_FOES = ['tutorial_nanali', 'tutorial_zaowu']


def _flags(*, hide=None, **kw):
    flags = dict(_L01_FLAGS)
    flags['player_hp'] = 12
    if hide is not None:
        flags['hide'] = list(hide)
    flags.update(kw)
    return flags


def _ack(sid, title, body, spots, primary='继续', finish=False, blocked='请先看说明。'):
    step = {
        'id': sid,
        'ack_required': True,
        'mask': True,
        'placement': 'bottom-right',
        'spotlights': spots,
        'modal': {'title': title, 'body': body, 'primary': primary},
        'allowed': [{'type': 'tutorial_ack'}],
        'blocked_reason': blocked,
    }
    if finish:
        step['finish_level'] = True
    return step


def _act(sid, title, body, spots, allowed, cue='', primary='继续', opponent=None,
         click=None, arrow=None, finish=False, blocked='请按提示操作。'):
    step = {
        'id': sid,
        'ack_required': False,
        'mask': True,
        'placement': 'bottom-right',
        'spotlights': spots,
        'modal': {'title': title, 'body': body, 'primary': primary},
        'allowed': allowed,
        'blocked_reason': blocked,
    }
    if cue:
        step['cue'] = cue
    if click:
        step['click_cue'] = click
    if arrow:
        step['drag_arrow'] = arrow
    if opponent:
        step['opponent_then'] = opponent
    if finish:
        step['finish_level'] = True
    return step


def _scene(sid, title, index, steps, flags=None, **extra):
    payload = {
        'id': sid,
        'title': title,
        'level_index': index,
        'seed': 100 + index,
        'first_side': extra.pop('first_side', 'a'),
        'order_a': extra.pop('order_a', list(_BOARD)),
        'order_b': extra.pop('order_b', list(_FOES)),
        'flags': flags or _flags(),
        'steps': steps,
        'playable': True,
    }
    payload.update(extra)
    return payload


_HIDE_HAND = ['passive', 'harmony', 'energy', 'ultimate', 'deck', 'mulligan', 'form']
_HIDE_FORM = ['passive', 'harmony', 'energy', 'ultimate', 'deck', 'mulligan']
_HIDE_RES = ['passive', 'ultimate', 'deck', 'mulligan', 'form']
_HIDE_FULL = ['passive', 'deck']

_L02_STEPS = [
    _ack('l02_intro', '手牌',
         '出牌花费 1 点行动力。战斗牌打出后，所属异能者会额外出击一次，不占用通常出击次数。这一回合有 2 点，可以先出牌，再通常出击。',
         ['hand:a-1', 'ap:a']),
    _act('l02_play', '打出战斗牌',
         '把这一张战斗牌拖到对战区。这会花费 1 点行动力，并让主角额外出击，本次攻击 +1。',
         ['hand:a-1', 'play_area'],
         [{'type': 'play_card'}],
         cue='把战斗牌拖到对战区', primary='拖动卡牌',
         arrow={'from': 'hand:a-1', 'to': 'play_area'}),
    _act('l02_attack', '再通常出击',
         '战斗牌的出击是额外的。还剩 1 点行动力，可以再做一次通常出击。',
         ['character:a:tutorial_protagonist', 'play_area'],
         [{'type': 'attack', 'character_id': 'tutorial_protagonist'}],
         cue='按住主角，并拖动到对战区', primary='拖动主角',
         arrow={'from': 'character:a:tutorial_protagonist', 'to': 'play_area'}, finish=True),
]

_L04_STEPS = [
    _ack('l04_intro', '战术牌',
         '战术牌同样花费 1 点行动力，但不会让角色出击。手机上先拖到对战区，松手后再点击目标；电脑开启拖动选择目标时，可以直接拖到目标身上。',
         ['hand:a-1']),
    _act('l04_play', '治疗薄荷',
         '用这张战术牌为薄荷回复 2 点生命。手机上先拖到对战区，松手后点击薄荷；电脑开启拖动选择目标时，可以直接拖给薄荷。',
         ['hand:a-1', 'character:a:tutorial_bohe'],
         [{'type': 'play_card'}],
         cue='打出战术牌，选择薄荷', primary='拖动卡牌',
         arrow={'from': 'hand:a-1', 'to': 'character:a:tutorial_bohe'}, finish=True),
]

_L05_STEPS = [
    _ack('l05_intro', '倒地',
         '异能者生命降到 0 会倒地，保留环合和能量，卸下武备，并在 3 个己方回合开始后恢复。倒地角色不能提供环合，也不能再获得资源。',
         ['character:b:tutorial_nanali']),
    _act('l05_hit', '击倒前排',
         '薄荷攻击 3 点。对方前排只剩 3 点生命，这次出击会把她打到倒地。',
         ['character:a:tutorial_bohe', 'character:b:tutorial_nanali'],
         [{'type': 'attack', 'character_id': 'tutorial_bohe'}],
         cue='按住薄荷，并拖动到对战区', primary='拖动薄荷',
         arrow={'from': 'character:a:tutorial_bohe', 'to': 'character:b:tutorial_nanali'}),
    _ack('l05_empty', '空前排打玩家',
         '前排倒地后，战斗区空了。再出击就会直接打在对方玩家身上。',
         ['zone:b:front', 'player_hp:b'], finish=True),
]

_L06_STEPS = [
    _ack('l06_intro', '武备',
         '武备牌花费 1 点行动力，给所属异能者换上一套新的攻击和生命。先打一仗掉血，再装备，就能看出生命被刷新了。',
         ['hand:a-1', 'character:a:tutorial_protagonist']),
    _act('l06_enter', '先出击',
         '这一回合有 2 点行动力。先花 1 点让主角进入战斗区，和对方前排互打掉血。',
         ['character:a:tutorial_protagonist', 'play_area'],
         [{'type': 'attack', 'character_id': 'tutorial_protagonist'}],
         cue='按住主角，并拖动到对战区', primary='拖动主角',
         arrow={'from': 'character:a:tutorial_protagonist', 'to': 'play_area'}),
    _act('l06_play', '装备武备',
         '还剩 1 点行动力。把这一张武备牌拖到对战区，主角就会装备它。攻击和生命会刷新成武备上的数值，掉的血也会被拉满。',
         ['hand:a-1', 'character:a:tutorial_protagonist'],
         [{'type': 'play_card'}],
         cue='把这一张武备牌拖到对战区', primary='拖动卡牌',
         arrow={'from': 'hand:a-1', 'to': 'character:a:tutorial_protagonist'}),
    _act('l06_end', '结束回合',
         '结束回合。对方会打你的前排；生命降到 0 时，武备会卸下。',
         ['end_turn'], [{'type': 'end_turn'}],
         cue='点击「结束回合」', primary='结束回合', click='end_turn',
         opponent=[{'type': 'attack', 'character_id': 'tutorial_zaowu'}, {'type': 'end_turn'}]),
    _ack('l06_down', '倒地脱落',
         '主角倒地后，武备被卸下，攻击和生命回到基础面板。终结状态也会一并结束。',
         ['character:a:tutorial_protagonist'], finish=True, primary='完成本关'),
]

_L07_STEPS = [
    _ack('l07_intro', '环合',
         '主角的异能：使用战斗牌后，立刻充满自己的环合值。环合满 2 后，下一名出战的队友可以消耗它，发动入场环合。光+灵是创生。',
         ['character:a:tutorial_protagonist', 'hand:a-1']),
    _act('l07_play', '打出战斗牌',
         '这一回合有 2 点行动力。先打出主角的战斗牌：她会额外出击，并且环合加满。',
         ['hand:a-1', 'play_area'],
         [{'type': 'play_card'}],
         cue='把这一张战斗牌拖到对战区', primary='拖动卡牌',
         arrow={'from': 'hand:a-1', 'to': 'play_area'}),
    _act('l07_mint', '切换薄荷',
         '还剩 1 点。把薄荷拖进战斗区，消耗主角的满环合。光+灵会创生：先造成 1 点伤害，再由薄荷出击。',
         ['character:a:tutorial_bohe', 'play_area'],
         [{'type': 'attack', 'character_id': 'tutorial_bohe'}],
         cue='按住薄荷，并拖动到对战区', primary='拖动薄荷',
         arrow={'from': 'character:a:tutorial_bohe', 'to': 'play_area'}),
    _ack('l07_full', '薄荷环合满了',
         '薄荷开局就有 1 点环合。主动进攻后再 +1，这一回合结束时她已经满环合。',
         ['character:a:tutorial_bohe']),
    _act('l07_end', '结束回合',
         '结束回合。下一回合开始时，看满环合会怎么显示。',
         ['end_turn'], [{'type': 'end_turn'}],
         cue='点击「结束回合」', primary='结束回合', click='end_turn',
         opponent=[{'type': 'end_turn'}]),
    _ack('l07_style', '满环合',
         '回合开始时，满环合的异能者会这样显示。薄荷还在备战区，满环合仍保留。',
         ['character:a:tutorial_bohe', 'bench:a']),
    _act('l07_harvest', '环合收割',
         '再让主角出场，消耗薄荷的满环合，再次触发创生。',
         ['character:a:tutorial_protagonist', 'play_area'],
         [{'type': 'attack', 'character_id': 'tutorial_protagonist'}],
         cue='按住主角，并拖动到对战区', primary='拖动主角',
         arrow={'from': 'character:a:tutorial_protagonist', 'to': 'play_area'}, finish=True),
]

_L08_STEPS = [
    _ack('l08_intro', '终结',
         '能量满槽后，可以发动终结，不消耗行动力，每回合限 1 次，备战区也可以用。终结是持续若干己方回合的状态，或一次性效果。',
         ['character:a:tutorial_protagonist']),
    _act('l08_ult', '发动终结',
         '主角能量已满。点击主角的「终结」按钮。这不花费行动力。',
         ['ultimate:a:tutorial_protagonist'],
         [{'type': 'ultimate', 'character_id': 'tutorial_protagonist'}],
         cue='点击主角的「终结」按钮', primary='继续',
         click='ultimate:a:tutorial_protagonist', finish=True),
]

_L09_STEPS = [
    _ack('l09_intro', '瞬发',
         '牌面写着瞬发的牌，本回合第一次打出时不消耗行动力。之后再打瞬发，仍要付费。',
         ['hand:a-1', 'ap:a']),
    _act('l09_play', '打出瞬发',
         '现在行动力是 1 点。打出这张瞬发，不会扣行动力。',
         ['hand:a-1', 'play_area'],
         [{'type': 'play_card'}],
         cue='把瞬发牌拖到对战区', primary='拖动卡牌',
         arrow={'from': 'hand:a-1', 'to': 'play_area'}, finish=True),
]

_L10_STEPS = [
    _ack('l10_intro', '响应',
         '牌面写着响应的牌，会在写明的时机自动打出，仍要支付行动力。响应战斗牌是作为反击方进入，不会重新出击，也不拿基础环合和能量。',
         ['hand:a-1']),
    _act('l10_setup', '先占前排',
         '先让主角出击占住前排，并留下 1 点行动力给响应。先攻第一回合只有 1 点，所以这一关你是后手，开局有 2 点。',
         ['character:a:tutorial_protagonist', 'play_area'],
         [{'type': 'attack', 'character_id': 'tutorial_protagonist'}],
         cue='按住主角，并拖动到对战区', primary='拖动主角',
         arrow={'from': 'character:a:tutorial_protagonist', 'to': 'play_area'}),
    _act('l10_end', '保留 1 点',
         '还剩 1 点行动力。结束回合，对手攻击主角时，薄荷的响应会自动打出。',
         ['end_turn'], [{'type': 'end_turn'}],
         cue='点击「结束回合」', primary='结束回合', click='end_turn',
         opponent=[{'type': 'attack', 'character_id': 'tutorial_nanali'}, {'type': 'end_turn'}]),
    _ack('l10_done', '反击方',
         '薄荷作为反击方进入战斗区，和进攻者互相攻击。响应不会当成一次新的出击。',
         ['character:a:tutorial_bohe', 'character:b:tutorial_nanali'], finish=True, primary='完成本关'),
]

_L11_STEPS = [
    _ack('l11_intro', '穿透',
         '薄荷战斗造成的溢出伤害穿透至玩家。同一击先打前排，多出来的伤害打到对方玩家，不是重新选目标。',
         ['character:a:tutorial_bohe']),
    _act('l11_hit', '打出溢出',
         '薄荷攻击 3 点，对方前排只剩 1 点生命。溢出的 2 点会穿透到对方玩家。',
         ['character:a:tutorial_bohe', 'character:b:tutorial_nanali', 'player_hp:b'],
         [{'type': 'attack', 'character_id': 'tutorial_bohe'}],
         cue='按住薄荷，并拖动到对战区', primary='拖动薄荷',
         arrow={'from': 'character:a:tutorial_bohe', 'to': 'character:b:tutorial_nanali'}, finish=True),
]

_L12_STEPS = [
    _act('l12_mulligan', '起手换牌',
         '开局抽 5 张。可以换掉最多 3 张，只能换一次，换掉的牌会洗回牌库。不换也可以直接确认。',
         ['mulligan'], [{'type': 'mulligan'}],
         cue='选择要换的手牌，然后确认', primary='去换牌'),
    _ack('l12_roster', '四人编队',
         '常规对战每方 4 名不同角色，牌库共 32 张。新手教学前面用 2 人，是为了看清位置。这一关使用四人编队，牌库仍为教学专用。',
         ['bench:a', 'bench:b']),
    _ack('l12_empty', '空牌库',
         '自己的回合开始时，抽 1 张牌。抽牌时若牌库已空，会立刻失败。留意剩余牌数，合理使用抽牌效果，避免空库时再次抽牌。',
         ['deck:a'], finish=True, primary='完成本关'),
]


_L13_STEPS = [
    _ack('l13_intro', '对局白热化',
         '这一关从后手的第 4 回合局面开始。回合数按双方轮流行动累计：你行动后是第 5 回合，再轮到你就是第 6 回合。左侧提示下一阶段还要多久。',
         ['round', 'escalation'], primary='看看能量'),
    _ack('l13_energy', '准备迎接第 6 回合',
         '主角现在有 4/5 能量，薄荷已有 5/5。第 6 回合起，每次回合开始，会为当前玩家能量最低的存活角色补 1 点能量，同值时随机选择；同时，每回合可发动两次终结。',
         ['character:a:tutorial_protagonist', 'character:a:tutorial_bohe', 'ultimate_count']),
    _act('l13_end', '进入下一次己方回合',
         '结束第 4 回合。对手完成第 5 回合后，就会进入你的第 6 回合。',
         ['end_turn'], [{'type': 'end_turn'}], click='end_turn',
         cue='点击「结束回合」', opponent=[{'type': 'end_turn'}]),
    _ack('l13_refill', '补能生效',
         '第 6 回合开始，主角是能量最低的存活角色，获得 1 点能量，达到 5/5。右侧终结次数也变为 2。现在两人都可以发动终结。',
         ['round', 'character:a:tutorial_protagonist', 'ultimate_count']),
    _act('l13_first', '第一次终结',
         '先点击主角的终结按钮。消耗她的满槽能量，不消耗行动力。',
         ['ultimate:a:tutorial_protagonist', 'ultimate_count'],
         [{'type': 'ultimate', 'character_id': 'tutorial_protagonist'}], cue='点击主角的终结按钮',
         click='ultimate:a:tutorial_protagonist'),
    _act('l13_second', '同回合再用一次',
         '本回合还剩 1 次终结机会。再点击薄荷的终结按钮，消耗薄荷自己的满槽能量。两次机会由全队共享，不是每人两次。',
         ['ultimate:a:tutorial_bohe', 'ultimate_count'],
         [{'type': 'ultimate', 'character_id': 'tutorial_bohe'}], cue='点击薄荷的终结按钮',
         click='ultimate:a:tutorial_bohe'),
    _ack('l13_done', '后续阶段',
         '两次终结已用完，行动力仍为 2。白热化效果会叠加：第 13 回合起能量上限降低 1 点，第 20 回合起环合上限降为 1 点；已有资源超过新上限时也会降到新上限。记得留意左侧预告。本关到这里结束。',
         ['ultimate_count', 'ap:a', 'escalation'], primary='完成本关', finish=True),
]


_L14_STEPS = [
    _ack('l14_intro', '远程',
         '远程让异能者在备战区出击，不进入战斗区，也不受反击。海月现在只剩 1 点生命，先让薄荷站到前排，再试试海月的终结。',
         ['character:a:tutorial_haiyue', 'character:b:tutorial_nanali', 'hand:a-1']),
    _act('l14_front', '保留前排防守',
         '打出「特遣行动」，让薄荷进入战斗区。战斗牌不消耗普通出击机会，剩下的 1 点行动力可以留给海月。',
         ['hand:a-1', 'play_area'], [{'type': 'play_card', 'card_id': 'a-1'}],
         cue='把「特遣行动」拖到对战区',
         arrow={'from': 'hand:a-1', 'to': 'play_area'}),
    _act('l14_ultimate', '发动海月终结',
         '本回合，海月可以在备战区远程出击。点击海月的终结按钮；终结不消耗行动力。',
         ['ultimate:a:tutorial_haiyue'], [{'type': 'ultimate', 'character_id': 'tutorial_haiyue'}],
         cue='点击海月的终结按钮', click='ultimate:a:tutorial_haiyue'),
    _act('l14_attack', '从备战区远程出击',
         '把海月拖到对战区发出出击指令。她会留在备战区攻击娜娜莉，不换下薄荷，也不会受到娜娜莉的反击。远程仍需支付普通出击的行动力和次数。',
         ['character:a:tutorial_haiyue', 'character:a:tutorial_bohe', 'play_area', 'ap:a'],
         [{'type': 'attack', 'character_id': 'tutorial_haiyue'}],
         cue='按住海月，并拖动到对战区',
         arrow={'from': 'character:a:tutorial_haiyue', 'to': 'play_area'}),
    _ack('l14_done', '出击后仍在备战区',
         '海月仍有 1 点生命，薄荷也仍在前排。远程不触发入场环合，仍优先攻击对方前排；没有对方前排才攻击玩家。它不免疫其他效果伤害；己方战斗区为空时，也不能替玩家挡住攻击。',
         ['character:a:tutorial_haiyue', 'character:a:tutorial_bohe', 'zone:a:front'],
         primary='完成教学', finish=True),
]


SCENARIOS = {
    'tutorial_l14_ranged': _scene(
        'tutorial_l14_ranged', '第十三关 远程', 13, _L14_STEPS,
        flags=_flags(hide=['harmony', 'deck', 'mulligan', 'form'],
                     disable_hand=False, disable_ultimate=False, disable_passives=False),
        order_a=['tutorial_haiyue', 'tutorial_bohe'], first_side='b',
        opening_b=[{'type': 'attack', 'character_id': 'tutorial_nanali'}, {'type': 'end_turn'}],
        hand_a=['tutorial_m_strike'],
        stats={'tutorial_haiyue': {'hp': 1, 'energy': 6}}),
    'tutorial_l13_escalation': _scene(
        'tutorial_l13_escalation', '第十二关 白热化', 12, _L13_STEPS,
        flags=_flags(hide=['passive', 'harmony', 'hand', 'deck', 'mulligan', 'form'],
                     disable_ultimate=False, escalation_enabled=True, player_hp=30),
        first_side='b', starting_turn=4,
        stats={'tutorial_protagonist': {'energy': 4}, 'tutorial_bohe': {'energy': 5}}),
    'tutorial_l01_board': _scene(
        'tutorial_l01_board', '第一关 牌桌与回合', 1, _L01_STEPS, flags=_L01_FLAGS),
    'tutorial_l02_hand': _scene(
        'tutorial_l02_hand', '第二关 手牌', 2, _L02_STEPS,
        flags=_flags(hide=_HIDE_HAND, disable_hand=False),
        first_side='b',
        opening_b=[{'type': 'end_turn'}],
        hand_a=['tutorial_z03']),
    'tutorial_l04_tactic': _scene(
        'tutorial_l04_tactic', '第三关 战术', 3, _L04_STEPS,
        flags=_flags(hide=_HIDE_HAND, disable_hand=False),
        hand_a=['tutorial_z_heal'],
        stats={'tutorial_bohe': {'hp': 2}}),
    'tutorial_l05_down': _scene(
        'tutorial_l05_down', '第四关 倒地', 4, _L05_STEPS,
        stats={'tutorial_nanali': {'hp': 3, 'max_hp': 3}},
        front_b='tutorial_nanali'),
    'tutorial_l06_form': _scene(
        'tutorial_l06_form', '第五关 武备', 5, _L06_STEPS,
        flags=_flags(hide=_HIDE_FORM, disable_hand=False),
        hand_a=['tutorial_z08'],
        extra_ap={'a': 1},
        front_b='tutorial_nanali',
        stats={'tutorial_zaowu': {'attack': 6}}),
    'tutorial_l07_harmony': _scene(
        'tutorial_l07_harmony', '第六关 环合', 6, _L07_STEPS,
        flags=_flags(
            hide=['energy', 'ultimate', 'deck', 'mulligan', 'form'],
            disable_hand=False, disable_harmony_gain=False,
            disable_harmony_trigger=False, disable_passives=False),
        extra_ap={'a': 1},
        hand_a=['tutorial_z03'],
        stats={'tutorial_bohe': {'harmony': 1}}),
    'tutorial_l08_energy': _scene(
        'tutorial_l08_energy', '第七关 终结', 7, _L08_STEPS,
        flags=_flags(
            hide=['passive', 'harmony', 'hand', 'deck', 'mulligan', 'form'],
            disable_ultimate=False),
        stats={'tutorial_protagonist': {'energy': 5}}),
    'tutorial_l09_instant': _scene(
        'tutorial_l09_instant', '第八关 瞬发', 8, _L09_STEPS,
        flags=_flags(hide=_HIDE_HAND, disable_hand=False),
        hand_a=['tutorial_z_instant']),
    'tutorial_l10_response': _scene(
        'tutorial_l10_response', '第九关 响应', 9, _L10_STEPS,
        flags=_flags(hide=_HIDE_HAND, disable_hand=False),
        first_side='b',
        opening_b=[{'type': 'end_turn'}],
        hand_a=['tutorial_m_response']),
    'tutorial_l11_pierce': _scene(
        'tutorial_l11_pierce', '第十关 穿透', 10, _L11_STEPS,
        flags=_flags(disable_overflow=False, disable_passives=False),
        stats={'tutorial_nanali': {'hp': 1, 'max_hp': 1}},
        front_b='tutorial_nanali'),
    'tutorial_l12_roster4': _scene(
        'tutorial_l12_roster4', '第十一关 四人编队', 11, _L12_STEPS,
        flags=_flags(
            hide=_HIDE_FULL, disable_hand=False, disable_draw=False,
            disable_mulligan=False, skip_mulligan=False, opening_draw=5,
            character_slots=4, player_hp=30),
        order_a=['tutorial_protagonist', 'tutorial_bohe', 'tutorial_iloy', 'tutorial_jiuyuan'],
        order_b=['tutorial_nanali', 'tutorial_zaowu', 'tutorial_baicang', 'tutorial_canhong'],
        deck_a=['tutorial_z03', 'tutorial_z_heal', 'tutorial_z08', 'tutorial_z_instant',
                'tutorial_m_strike', 'tutorial_m_response'] * 2,
        deck_b=['tutorial_z03'] * 8),
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
