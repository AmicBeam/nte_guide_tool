"""黑羽：环合占卜与救世魔女桌游规则。"""
from copy import deepcopy
from random import Random

from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.content.duel_v2.characters.common import revive_character, set_shape
from app.modules.card_game.content.duel_v2.registry import register, choice_continuation
from app.modules.card_game.engine.duel_v2.deferred import enqueue
from app.modules.card_game.engine.duel_v2.modifiers import add_effect, deadline
from app.modules.card_game.engine.duel_v2.state import (
    add_hand, alive, card_instance, damage, discard, finished, hero, log, other, team_order,
)


DIVINATION_CARDS = frozenset(f'P{i:02d}' for i in range(1, 9))


def living_espers(state, side):
    return [cid for cid in team_order(state, side)
            if alive(state, side, cid) and not hero(state, side, cid).get('summoned')]


def in_witch(c):
    return c.cid == 'heiyu' and bool(c.character.get('awakened'))


def _rng(state):
    rng = Random(state['rng'])
    state['rng'] = rng.getrandbits(63)
    return rng


def _detach(c, card):
    """Move the actual entity, including when a revealed bottom card releases itself."""
    team = c.state['sides'][c.side]
    for zone in ('deck', 'hand', 'discard'):
        team[zone] = [item for item in team[zone] if item['entity_id'] != card['entity_id']]


def _reveal(c, card):
    from app.modules.card_game.engine.duel_v2.presentation import public_card
    log(c.state, f"黑羽公开「{card['name']}」并发动占卜效果。", 'reveal',
        side=c.side, source=c.cid, actor=f'{c.side}:{c.cid}', card=public_card(card, c.state, c.side),
        response=False, card_source='deck')


def _release(c, card, target=None, *, preserve_parent=True):
    """Release an effect without use triggers; preserve the enclosing operation."""
    from app.modules.card_game.engine.duel_v2.flow import resolve_release
    parent = c.state.get('operation')
    draining = c.state.get('_draining_actions')
    _detach(c, card)
    c.state['_draining_actions'] = True
    try:
        # The preceding reveal already presents this card to both players.
        resolve_release(c.state, c.side, Heiyu.hand_card(c, card, card), target, present=False)
    finally:
        if preserve_parent:
            c.state['operation'] = parent
        if draining is None:
            c.state.pop('_draining_actions', None)
        else:
            c.state['_draining_actions'] = draining


def _immune(c):
    add_effect(c.character, 'heiyu.combat_immune', 'rule.combat_immune', True, op='allow',
               source_entity_id=c.entity_id, source_key='P01', stacking='replace',
               clear_on=('battle', 'operation_end', 'down'), show_marker=False, label='战斗免伤')


def _advance_owned(c):
    deck = c.state['sides'][c.side]['deck']
    choices = [index for index, card in enumerate(deck) if card.get('character_id') == c.cid]
    if choices:
        card = deck.pop(_rng(c.state).choice(choices))
        deck.insert(0, card)
        c.history('「不凋羽」：「占卜」检视牌库前，随机将 1 张黑羽的牌置于牌库顶。')


def start_inspection(c, *, after=None, min_bottom=0):
    if finished(c.state) or not alive(c.state, c.side, c.cid):
        return
    team = c.state['sides'][c.side]
    team['heiyu_divinations'] = int(team.get('heiyu_divinations') or 0) + 1
    if after:
        c.operation['heiyu_after_divination'] = after
    if c.character.get('shape') == 'P08':
        _advance_owned(c)
    count = max(2, len(living_espers(c.state, c.side)))
    cards, team['deck'] = team['deck'][:count], team['deck'][count:]
    if cards:
        c.choose('heiyu_divination', cards,
                 f'检视牌库顶 {len(cards)} 张牌：' + ('至少置底 1 张，其余自由排序。' if min_bottom else '可将任意数量置底，其余自由排序。'),
                 deck_cards=True, deck_order=True, execute_divination=True,
                 actor=c.cid, snapshot_n=count, min_bottom=min_bottom, continuation='heiyu.divination',
                 max_bottom=len(cards), multi_bottom=True)
    else:
        _resume_inspection(c)


def _resume_inspection(c):
    after = c.operation.pop('heiyu_after_divination', None)
    if finished(c.state) or not alive(c.state, c.side, c.cid):
        return
    if after == 'draw':
        c.draw(1)
    elif after == 'sortie':
        c.sortie()
    elif c.operation.get('pending_sortie') is not None:
        from app.modules.card_game.engine.duel_v2.combat import resume_sortie
        resume_sortie(c)


def apply_divination_benefit(c, card, *, witch=False):
    if finished(c.state) or not card or card.get('card_id') not in DIVINATION_CARDS:
        return
    _reveal(c, card)
    card_id = card['card_id']
    if card_id == 'P01':
        if witch:
            _detach(c, card)
            discard(c.state, c.side, card)
            add_effect(c.character, 'heiyu.divination_attack', 'attack.hit', 1,
                       source_entity_id=c.entity_id, source_key=card['entity_id'],
                       clear_on=('battle', 'operation_end', 'down'), consume_on='attack',
                       show_marker=False, label='叛逃者')
            _immune(c)
        else:
            _release(c, card)
    elif card_id == 'P02':
        targets = living_espers(c.state, other(c.side))
        if targets:
            _release(c, card, f'{other(c.side)}:{_rng(c.state).choice(targets)}')
    elif card_id == 'P03':
        c.draw(1)
    elif card_id == 'P04':
        if witch:
            _detach(c, card)
            discard(c.state, c.side, card)
            enqueue(c, 'heiyu_release_after_attack')
            c.state['deferred_actions'][-1]['card'] = card
        else:
            _release(c, card)
    elif card_id == 'P05':
        _detach(c, card)
        add_hand(c.state, c.side, card, reveal=True)
    elif card_id == 'P06':
        downed = [cid for cid in team_order(c.state, c.side)
                  if not hero(c.state, c.side, cid).get('summoned')
                  and int(hero(c.state, c.side, cid).get('down_turns') or 0) > 0]
        if downed:
            cid = _rng(c.state).choice(downed)
            h = hero(c.state, c.side, cid)
            before = int(h['down_turns'])
            h['down_turns'] = before - 1
            if not h['down_turns']:
                h['hp'] = h['max_hp']
                log(c.state, f"{h['name']}在备战区恢复。", 'revive', side=c.side, source=c.cid,
                    actor=f'{c.side}:{c.cid}', target=f'{c.side}:{cid}',
                    before={'hp': 0, 'down_turns': before}, after={'hp': h['hp'], 'down_turns': 0})
            else:
                c.history(f"{h['name']}的倒地倒计时 -1。", target_id=cid)
    elif card_id == 'P07':
        add_effect(c.state['sides'][c.side], 'heiyu.instant_sortie', 'rule.normal_attack_instant', True,
                   op='allow', scope='allies', stacking='replace', clear_on=(),
                   source_entity_id=c.entity_id, source_key='P07',
                   expiry=deadline(c.state, c.side, 0, 'turn_end'), label='普通出击可消耗瞬发机会')
        c.history('「牧星者」：本回合普通出击可以消耗瞬发机会。')
    elif card_id == 'P08':
        c.state['sides'][c.side]['used']['instant'] = False
        c.history('「不凋羽」：瞬发机会恢复为 1 次。')


def _p01(c):
    _immune(c)
    c.sortie()


def _p02(c):
    side, cid = c.operation['target_id'].split(':', 1)
    damage(c.state, side, cid, 3, source=f'{c.side}:{c.cid}', reason='「漫游记」')


def _p03(c):
    c.draw(1)
    cards = list(c.state['sides'][c.side]['hand'])
    if cards and not finished(c.state):
        c.operation['heiyu_after_placement'] = True
        c.choose('hand_deck_placement', cards, '将 1 张手牌置于牌库顶或底',
                 deck_order=True, hand_placement=True, multi_bottom=True,
                 min_bottom=0, max_bottom=1, actor=c.cid, execute_divination=False)
    else:
        c.sortie()


def _p04(c):
    c.sortie()


def _p05(c):
    count = int(c.state['sides'][c.side].get('heiyu_divinations') or 0)
    foe = other(c.side)
    candidates = living_espers(c.state, foe)
    for cid in _rng(c.state).sample(candidates, min(count, len(candidates))):
        if finished(c.state):
            break
        if alive(c.state, foe, cid):
            damage(c.state, foe, cid, 2, source=f'{c.side}:{c.cid}', reason='「烂柯人」')
    start_inspection(c)


def _p06(c):
    side, cid = c.operation['target_id'].split(':', 1)
    revive_character(c, side, cid)
    start_inspection(c)


def _penetrate(c, current=0, attacker_id=None, raw_overflow=0, **_data):
    return int(raw_overflow or 0) if attacker_id == c.cid else None


@register
class Heiyu:
    id = 'heiyu'
    text_name = '黑羽'
    effects = {'P01': _p01, 'P02': _p02, 'P03': _p03, 'P04': _p04,
               'P05': _p05, 'P06': _p06, 'P07': set_shape, 'P08': set_shape}
    target_policies = {'P02': 'living_enemy_esper', 'P06': 'downed_ally'}
    weapon_hooks = {'P07': {'combat_overflow': _penetrate}}

    @staticmethod
    def on_harmony_applied(c, kind=None, **_data):
        if kind in ('浸染', '黯星'):
            enqueue(c, 'heiyu_harmony_divination')

    @staticmethod
    def after_card(c, *, actor_id, card):
        if not in_witch(c) and actor_id == c.cid and card and card.get('type') == 'battle':
            from app.modules.card_game.engine.duel_v2.escalation import harmony_cap
            c.grant_harmony(c.cid, harmony_cap(c.state), reason='黑羽使用战斗牌，触发异能')

    @staticmethod
    def stain_duration(c, current=1, **_data):
        return int(current) * 2

    @staticmethod
    def dark_star_renewals(c, current=0, **_data):
        return max(1, int(current))

    @staticmethod
    def on_ultimate(c):
        for cid in living_espers(c.state, c.side):
            add_effect(hero(c.state, c.side, cid), 'heiyu.ultimate_atk', 'attack.panel', 1,
                       stacking='replace', source_entity_id=c.entity_id, source_key='P-ultimate',
                       label='黑羽终结', expiry=deadline(c.state, c.side, 2, 'turn_start'))
        c.history('全体己方异能者攻击 +1，持续 2 个己方回合。黑羽进入「救世魔女」。')

    @staticmethod
    def can_exit_ultimate(c, current=False, **_data):
        return True if in_witch(c) else current

    @staticmethod
    def exit_ultimate_label(c, current=None, **_data):
        return '退出救世魔女'

    @staticmethod
    def on_deferred_action(c, action):
        if action.get('kind') == 'heiyu_harmony_divination':
            start_inspection(c)
            return
        if action.get('kind') == 'heiyu_release_after_attack' and alive(c.state, c.side, c.cid):
            _release(c, action['card'], preserve_parent=False)

    @staticmethod
    def on_choice(c, selected):
        # Finish a choice already saved under the previous P03 rules.
        if selected and c.operation.pop('heiyu_hand_to_top', False):
            _detach(c, selected)
            c.state['sides'][c.side]['deck'].insert(0, selected)
            c.history(f"将「{selected['name']}」置于牌库顶。")
            start_inspection(c, after='sortie')

    @staticmethod
    def deck_order_spec(c, current, **_data):
        if current.get('kind') != 'heiyu_divination':
            return current
        return dict(current, multi_bottom=True, max_bottom=len(current['cards']),
                    min_bottom=0,
                    execute_divination=True)

    @staticmethod
    def on_deck_reorder(c, bottom=None, bottoms=None, execute_divination=None, **_data):
        if c.operation.pop('heiyu_after_placement', False):
            c.sortie()
            return
        if execute_divination is not False:
            for card in (bottoms if bottoms is not None else ([bottom] if bottom else [])):
                if finished(c.state) or not alive(c.state, c.side, c.cid):
                    break
                apply_divination_benefit(c, card, witch=in_witch(c))
        _resume_inspection(c)

    @staticmethod
    def incoming_damage(c, current=0, kind=None, **_data):
        from app.modules.card_game.engine.duel_v2.modifiers import evaluate
        if kind == 'combat' and evaluate(c.state, c.side, c.character, 'rule.combat_immune', False)['value']:
            return 0
        return None

    @staticmethod
    def hand_card(c, current, card):
        spec = CARDS.get(current.get('card_id'))
        if current.get('character_id') != c.cid or not spec or spec.get('character_id') != c.cid:
            return current
        result = deepcopy(current)
        for field in ('instant', 'attack', 'shield', 'hp'):
            result.pop(field, None)
        for field in ('name', 'type', 'description', 'cost', 'effect_id', 'instant', 'attack', 'shield', 'hp'):
            if field in spec:
                result[field] = spec[field]
        return result

    @staticmethod
    def card_description(c, current, card):
        spec = CARDS.get(card.get('card_id'))
        if not spec or spec.get('character_id') != c.cid:
            return current
        text = spec['description']
        if card.get('card_id') == 'P05':
            count = int(c.state['sides'][c.side].get('heiyu_divinations') or 0)
            text = text.replace('当前为0', f'当前为{count}')
        return text


@choice_continuation('heiyu.divination')
def _continue_divination(c, **data):
    # The card effect remains bound after a copied passive has expired.
    Heiyu.on_deck_reorder(c, **data)
