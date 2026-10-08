"""黑羽：用户指定的占卜与救世魔女试玩规则，不是原作数值。"""
from random import Random

from app.modules.card_game.content.duel_v2.characters.common import revive_character, set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.deferred import enqueue
from app.modules.card_game.engine.duel_v2.modifiers import add_effect, deadline
from app.modules.card_game.engine.duel_v2.state import (
    add_hand, alive, card_instance, damage, discard, finished, heal, hero, log, other, shield, shuffle, team_order,
)


def living_espers(state, side):
    return [cid for cid in team_order(state, side)
            if alive(state, side, cid) and not hero(state, side, cid).get('summoned')]


def in_witch(c):
    return bool(c.character.get('awakened'))


def _rng(state):
    rng = Random(state['rng'])
    state['rng'] = rng.getrandbits(63)
    return rng


def insert_omen(c):
    if in_witch(c) or not alive(c.state, c.side, c.cid) or finished(c.state):
        return
    deck = c.state['sides'][c.side]['deck']
    card = card_instance(c.state, 'PF01', c.side)
    deck.insert(_rng(c.state).randrange(len(deck) + 1), card)
    c.history('异能：将 1 张「谕石」随机插入己方牌库。')


def mark_divined(c):
    c.state['sides'][c.side]['heiyu_divined_turn'] = c.state['turn']


def grant_combat_immune(c):
    add_effect(c.character, 'heiyu.combat_immune', 'rule.combat_immune', True, op='allow',
               source_entity_id=c.entity_id, source_key='P03', stacking='replace',
               clear_on=('battle', 'operation_end', 'down'), show_marker=False, label='战斗免伤')


def grant_magic_attack(c, card, amount):
    add_effect(c.character, 'heiyu.magic_attack', 'attack.hit', amount,
               source_entity_id=c.entity_id, source_key=card['entity_id'],
               clear_on=('battle', 'operation_end', 'down'), consume_on='attack',
               show_marker=False, label='魔女攻击强化')


def start_inspection(c, *, execute_divination):
    if finished(c.state) or not alive(c.state, c.side, c.cid):
        return
    count = len(living_espers(c.state, c.side))
    team = c.state['sides'][c.side]
    cards, team['deck'] = team['deck'][:count], team['deck'][count:]
    if not cards:
        mark_divined(c)
        return
    prompt = ('检视牌库顶 %d 张牌：可将至多 1 张置于牌库底，其余按任意顺序放回牌库顶。'
              % len(cards))
    c.choose('heiyu_divination', cards, prompt, deck_cards=True, deck_order=True,
             execute_divination=bool(execute_divination), actor=c.cid, snapshot_n=count,
             max_bottom=1)


def apply_divination_benefit(c, card):
    if finished(c.state) or not card:
        return
    from app.modules.card_game.engine.duel_v2.presentation import public_card
    log(c.state, f"{c.character['name']}公开占卜「{card['name']}」。", 'reveal',
        side=c.side, source=c.cid, actor=f'{c.side}:{c.cid}', card=public_card(card))
    card_id = card.get('card_id')
    if card_id == 'P01':
        c.state['sides'][c.side]['heiyu_next_other_attack'] = c.state['turn']
        c.history('占卜效果：下一名其他己方编队异能者的主动攻击 +1。')
    elif card_id == 'P02':
        c.draw(1)
    elif card_id == 'P03':
        c.state['sides'][c.side]['heiyu_next_other_shield'] = c.state['turn']
        c.history('占卜效果：下一名其他己方编队异能者主动攻击前获得 2 点护盾。')
    elif card_id == 'P04':
        heal(c.state, c.side, 'player', 2, source=f'{c.side}:{c.cid}', reason='占卜效果')
    elif card_id == 'P05':
        foe = other(c.side)
        enemies = [cid for cid in living_espers(c.state, foe)]
        if enemies:
            target = _rng(c.state).choice(enemies)
            damage(c.state, foe, target, 2, source=f'{c.side}:{c.cid}', reason='占卜效果')
    elif card_id == 'P06':
        downed = [(hero(c.state, c.side, cid).get('down_turns') or 0, cid)
                  for cid in team_order(c.state, c.side)
                  if cid != c.cid and not hero(c.state, c.side, cid).get('summoned')
                  and hero(c.state, c.side, cid).get('down_turns')]
        if downed:
            downed.sort(key=lambda item: (item[0], team_order(c.state, c.side).index(item[1])))
            cid = downed[0][1]
            h = hero(c.state, c.side, cid)
            h['down_turns'] = max(0, int(h['down_turns']) - 1)
            if not h['down_turns']:
                h['hp'] = h['max_hp']
                log(c.state, f"{h['name']}在备战区恢复。", 'revive', side=c.side, source=c.cid,
                    actor=f'{c.side}:{c.cid}', target=f'{c.side}:{cid}',
                    before={'hp': 0, 'down_turns': 1}, after={'hp': h['hp'], 'down_turns': 0})
            else:
                c.history(f"占卜效果：{h['name']}的倒地倒计时变为 {h['down_turns']}。", target_id=cid)
    elif card_id == 'P07':
        team = c.state['sides'][c.side]
        others = {cid for cid in team_order(c.state, c.side)
                  if cid != c.cid and not hero(c.state, c.side, cid).get('summoned')}
        candidates = [index for index, item in enumerate(team['deck'])
                      if item.get('character_id') in others]
        if candidates:
            index = _rng(c.state).choice(candidates)
            taken = team['deck'].pop(index)
            add_hand(c.state, c.side, taken)
    elif card_id == 'P08':
        team = c.state['sides'][c.side]
        found = next((item for item in reversed(team['discard'])
                      if item.get('character_id') == c.cid and item.get('type') == 'battle'), None)
        if found:
            found_id = found.get('entity_id')
            team['discard'] = [item for item in team['discard'] if item.get('entity_id') != found_id]
            team['deck'].insert(0, found)
            c.history(f"占卜效果：「{found['name']}」回到牌库顶。")
    _maybe_weapon_draw(c)


def _maybe_weapon_draw(c):
    if (c.character.get('shape') == 'P07' and not in_witch(c)
            and c.character.get('heiyu_p07_draw_turn') != c.state['turn']
            and c.state['active_side'] == c.side and not finished(c.state)):
        c.character['heiyu_p07_draw_turn'] = c.state['turn']
        c.draw(1)


def magic_trigger(card_id):
    return {
        'P01': 'after_active_attack',
        'P02': 'after_active_attack',
        'P03': 'before_combat',
        'P04': 'before_active_attack',
        'P05': 'before_active_attack',
        'P06': 'after_knockdown',
        'P07': 'before_active_attack',
        'P08': 'before_active_attack',
    }.get(card_id)


def trigger_matches(card_id, trigger):
    owned = magic_trigger(card_id)
    if owned == trigger:
        return True
    return owned == 'before_combat' and trigger in ('before_active_attack', 'before_attacked')


def consume_magic(c, trigger, *, seen=None):
    if finished(c.state) or not alive(c.state, c.side, c.cid) or not in_witch(c):
        return
    team = c.state['sides'][c.side]
    seen = list(seen or [])
    while team['deck'] and not finished(c.state) and c.state['phase'] != 'choice':
        top = team['deck'][0]
        if top.get('character_id') != c.cid:
            break
        entity_id = top.get('entity_id')
        if entity_id in seen:
            break
        card_id = top.get('card_id')
        if not trigger_matches(card_id, trigger):
            break
        if card_id == 'P02':
            if c.state['active_side'] != c.side:
                break
            if not any(item.get('character_id') == c.cid for item in team['hand']):
                break
        card = team['deck'].pop(0)
        from app.modules.card_game.engine.duel_v2.presentation import public_card
        log(c.state, f"「救世魔女」公开「{card['name']}」。", 'reveal',
            side=c.side, source=c.cid, actor=f'{c.side}:{c.cid}', card=public_card(card))
        discard(c.state, c.side, card)
        seen.append(entity_id)
        _execute_magic(c, card, trigger)
        if c.state['phase'] == 'choice' or finished(c.state):
            return


def _execute_magic(c, card, trigger):
    card_id = card.get('card_id')
    if card_id == 'P01' and trigger == 'after_active_attack':
        if alive(c.state, c.side, c.cid):
            enqueue(c, 'heiyu_extra_attack')
            c.history('魔女效果：黑羽将额外主动攻击 1 次。')
    elif card_id == 'P02' and trigger == 'after_active_attack':
        cards = [item for item in c.state['sides'][c.side]['hand'] if item.get('character_id') == c.cid]
        if cards:
            c.choose('kit', cards, '选择 1 张黑羽的牌置于牌库顶')
            c.operation['heiyu_resume_magic'] = trigger
    elif card_id == 'P03':
        grant_combat_immune(c)
        c.history('魔女效果：本次战斗中黑羽不会受到战斗伤害。')
    elif card_id == 'P04' and trigger == 'before_active_attack':
        grant_magic_attack(c, card, 1)
        shield(c.state, c.side, c.cid, 1, reason='魔女效果')
    elif card_id == 'P05' and trigger == 'before_active_attack':
        bonus = 2 + min(2, int(c.state['sides'][c.side].get('heiyu_p05_magic_bonus') or 0))
        grant_magic_attack(c, card, bonus)
        c.operation.setdefault('heiyu_p05_recycle', []).append(card.get('entity_id'))
    elif card_id == 'P06' and trigger == 'after_knockdown':
        if alive(c.state, c.side, c.cid):
            c.heal(c.character['max_hp'], reason='魔女效果')
    elif card_id == 'P07' and trigger == 'before_active_attack':
        add_effect(c.character, 'heiyu.double', 'attack.hit', 2, op='multiply', layer=2000,
                   source_entity_id=c.entity_id, source_key=card['entity_id'],
                   clear_on=('battle', 'operation_end', 'down'),
                   consume_on='attack', show_marker=False, label='攻击翻倍')
    elif card_id == 'P08' and trigger == 'before_active_attack':
        generated = card_instance(c.state, 'P01', c.side)
        c.state['sides'][c.side]['deck'].insert(0, generated)
        c.history('魔女效果：在牌库顶生成 1 张「叛逃者」。')


def recycle_p05(c):
    ids = list(c.operation.get('heiyu_p05_recycle') or [])
    c.operation['heiyu_p05_recycle'] = []
    team = c.state['sides'][c.side]
    for entity_id in ids:
        card = next((item for item in team['discard'] if item.get('entity_id') == entity_id), None)
        if card is None:
            continue
        team['discard'] = [item for item in team['discard'] if item.get('entity_id') != entity_id]
        team['deck'].append(card)
        shuffle(c.state, team['deck'])
        extra = min(2, int(team.get('heiyu_p05_magic_bonus') or 0) + 1)
        team['heiyu_p05_magic_bonus'] = extra
        c.history(f"魔女效果：「{card['name']}」洗入牌库，同名魔女攻击加成变为 +{2 + extra}。")


def _p01(c):
    c.sortie()


def _p02(c):
    start_inspection(c, execute_divination=not in_witch(c))
    mark_divined(c)


def _pf01(c):
    c.operation['heiyu_draw_after_inspection'] = True
    start_inspection(c, execute_divination=not in_witch(c))
    mark_divined(c)
    if c.state['phase'] != 'choice':
        c.operation.pop('heiyu_draw_after_inspection', None)
        c.draw(1)


def _p03(c):
    grant_combat_immune(c)
    c.sortie()


def _p04(c):
    c.sortie()


def _p05(c):
    side, cid = c.operation['target_id'].split(':', 1)
    damage(c.state, side, cid, 3, source=f'{c.side}:{c.cid}', reason='「烂柯人」')


def _p06(c):
    side, cid = c.operation['target_id'].split(':', 1)
    revive_character(c, side, cid)


def _p08_penetrate(c, current=0, attacker_id=None, raw_overflow=0, **_data):
    if not in_witch(c) or attacker_id != c.cid:
        return None
    return int(raw_overflow or 0)


@register
class Heiyu:
    id = 'heiyu'
    text_name = '黑羽'
    effects = {
        'P01': _p01, 'P02': _p02, 'P03': _p03, 'P04': _p04,
        'P05': _p05, 'P06': _p06, 'P07': set_shape, 'P08': set_shape, 'PF01': _pf01,
    }
    target_policies = {'P05': 'living_enemy_esper', 'P06': 'other_downed_ally'}
    weapon_hooks = {'P08': {'combat_overflow': _p08_penetrate}}

    @staticmethod
    def on_turn_begin(c):
        insert_omen(c)

    @staticmethod
    def stain_duration(c, current=1, **_data):
        return int(current) * 2

    @staticmethod
    def dark_star_renewals(c, current=0, **_data):
        return max(1, int(current))

    @staticmethod
    def makes_instant(state, side, card):
        return bool(card and card.get('card_id') == 'P04'
                    and state['sides'][side].get('heiyu_divined_turn') == state['turn'])

    @staticmethod
    def on_ultimate(c):
        for cid in living_espers(c.state, c.side):
            h = hero(c.state, c.side, cid)
            add_effect(h, 'heiyu.ultimate_atk', 'attack.panel', 1, stacking='replace',
                       source_entity_id=c.entity_id, source_key='P-ultimate', label='黑羽终结',
                       expiry=deadline(c.state, c.side, 2, 'turn_start'))
        c.history('发动终结：全体己方异能者攻击 +1，持续 2 个己方回合。黑羽进入「救世魔女」。')

    @staticmethod
    def can_exit_ultimate(c, current=False, **_data):
        return True if in_witch(c) else current

    @staticmethod
    def exit_ultimate_label(c, current=None, **_data):
        return '退出救世魔女'

    @staticmethod
    def before_attack(c, **_data):
        if c.operation.get('actor') == c.cid:
            insert_omen(c)
        if in_witch(c) and c.operation.get('actor') == c.cid:
            consume_magic(c, 'before_active_attack')

    @staticmethod
    def on_opponent_attack(c, attacker_id=None, **_data):
        if in_witch(c) and _data.get('target_id') == c.cid:
            consume_magic(c, 'before_attacked')

    @staticmethod
    def before_attack_damage(c, attacker_id=None, **_data):
        team = c.state['sides'][c.side]
        if attacker_id and attacker_id != c.cid and not hero(c.state, c.side, attacker_id).get('summoned'):
            if team.get('heiyu_next_other_shield') == c.state['turn']:
                team.pop('heiyu_next_other_shield', None)
                shield(c.state, c.side, attacker_id, 2, reason='占卜效果')
            if team.get('heiyu_next_other_attack') == c.state['turn']:
                team.pop('heiyu_next_other_attack', None)
                add_effect(hero(c.state, c.side, attacker_id), 'heiyu.divination_attack', 'attack.hit', 1,
                           source_entity_id=c.entity_id, source_key='P01',
                           clear_on=('battle', 'operation_end', 'down'), consume_on='attack',
                           show_marker=False, label='占卜攻击强化')

    @staticmethod
    def on_turn_end(c):
        team = c.state['sides'][c.side]
        if team.get('heiyu_next_other_attack') == c.state['turn']:
            team.pop('heiyu_next_other_attack', None)
        if team.get('heiyu_next_other_shield') == c.state['turn']:
            team.pop('heiyu_next_other_shield', None)

    @staticmethod
    def after_sortie(c, **_data):
        if c.operation.get('actor') != c.cid:
            return
        recycle_p05(c)
        if in_witch(c) and alive(c.state, c.side, c.cid) and not finished(c.state):
            consume_magic(c, 'after_active_attack')

    @staticmethod
    def on_rout(c, **_data):
        if in_witch(c) and alive(c.state, c.side, c.cid) and not finished(c.state):
            consume_magic(c, 'after_knockdown')

    @staticmethod
    def on_deferred_action(c, action):
        if action.get('kind') != 'heiyu_extra_attack':
            return
        if not alive(c.state, c.side, c.cid) or finished(c.state):
            return
        c.operation['skip_combat_harmony'] = True
        c.history('魔女效果：黑羽额外主动攻击。')
        c.sortie(no_combat_resources=True)

    @staticmethod
    def on_choice(c, selected):
        if selected and (c.operation or {}).get('heiyu_resume_magic'):
            team = c.state['sides'][c.side]
            selected_id = selected.get('entity_id')
            team['hand'] = [item for item in team['hand'] if item.get('entity_id') != selected_id]
            team['deck'].insert(0, selected)
            c.history(f"魔女效果：「{selected['name']}」置于牌库顶。")
            trigger = c.operation.pop('heiyu_resume_magic', None)
            if trigger:
                consume_magic(c, trigger)

    @staticmethod
    def on_deck_reorder(c, bottom=None, execute_divination=None, **_data):
        execute = True if execute_divination is None else bool(execute_divination)
        if bottom and bottom.get('character_id') == c.cid and execute and not in_witch(c):
            apply_divination_benefit(c, bottom)
        if c.operation.pop('heiyu_draw_after_inspection', False) and not finished(c.state):
            c.draw(1)

    @staticmethod
    def incoming_damage(c, current=0, kind=None, **_data):
        from app.modules.card_game.engine.duel_v2.modifiers import evaluate
        if kind == 'combat' and evaluate(c.state, c.side, c.character, 'rule.combat_immune', False)['value']:
            return 0
        return None

    @staticmethod
    def card_description(c, current, card):
        if card.get('card_id') not in ('P02', 'PF01') or not current:
            return current
        count = len(living_espers(c.state, c.side))
        usual = (f'【通常】瞬发。检视牌库顶 {count} 张牌（当前存活己方编队异能者数量，含黑羽，不含召唤物）。'
                 '可将其中至多 1 张置于牌库底，其余以任意顺序放回牌库顶。'
                 '基础形态下，若置于牌库底的是黑羽的牌，公开该牌并结算其占卜效果；'
                 '「救世魔女」期间只整理牌库。牌库不足时检视实际张数，不因此判负。')
        if card.get('card_id') == 'PF01':
            return usual.removeprefix('【通常】') + '完成后抽 1 张牌。'
        tail = current.split('\n', 1)[1] if '\n' in current else ''
        return usual + (('\n' + tail) if tail else '')
