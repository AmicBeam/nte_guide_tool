"""JSON state primitives; no persistence, HTTP, or hidden-information policy."""
from copy import deepcopy
from random import Random

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS
from app.modules.card_game.content.duel_v2.effects import shape_attack_bonus

SIDES = ('a', 'b')
ORDER = ('nanali', 'zero', 'jiuyuan', 'xun')


def other(side):
    return 'b' if side == 'a' else 'a'


def hero(state, side, cid):
    return state['sides'][side]['characters'][cid]


def alive(state, side, cid):
    return hero(state, side, cid)['hp'] > 0


def log(state, text, kind='effect', **fields):
    from .presentation import enrich_log
    seq = state['event_seq'] + 1
    state['event_seq'] = seq
    event = {'seq': seq, 'type': kind, 'text': text, **fields}
    state['events'].append(event)
    enrich_log(state, event)


def name(state, side, cid=None):
    return hero(state, side, cid)['name'] if cid else ('甲方玩家' if side == 'a' else '乙方玩家')


def attack_value(character):
    return character['base_attack'] + character['growth'] + shape_attack_bonus(character['shape'])


def new_character(cid):
    c = CHARACTERS[cid]
    return {**deepcopy(c), 'base_attack': c['attack'], 'hp': c['max_hp'], 'shield': 0,
            'harmony': 0, 'energy': 0, 'awakened': False, 'growth': 0,
            'shape': None, 'down_turns': 0, 'flags': {}}


def card_instance(state, card_id, side, *, copy=False):
    state['next_instance'] += 1
    c = deepcopy(CARDS[card_id])
    c.update(card_id=card_id, instance_id=f"{side}-{state['next_instance']}", copy=copy)
    if copy:
        c['original_character_id'] = c['character_id']
        c['character_id'] = 'xun'
        c['expires_turn'] = state['sides'][side]['turn_count'] + 1
    return c


def shuffle(state, cards):
    # Store a small integer stream state, rather than non-JSON Random tuples.
    rng = Random(state['rng'])
    rng.shuffle(cards)
    state['rng'] = rng.getrandbits(63)


def finished(state):
    return state['phase'] == 'finished'


def victory(state):
    dead = [s for s in SIDES if state['sides'][s]['hp'] <= 0]
    if not dead:
        return
    state['winner'] = 'draw' if len(dead) == 2 else other(dead[0])
    state['phase'] = 'finished'
    state['reason'] = 'player_hp'
    log(state, '双方生命归零，平局。' if len(dead) == 2 else f"{name(state, state['winner'])}获胜。", 'finish')


def move_out(state, side, cid):
    team = state['sides'][side]
    if team['front'] != cid:
        return
    team['last_front'] = cid
    team['front'] = None
    h = hero(state, side, cid)
    before = {'hp': h['hp'], 'shield': h['shield'], 'front': cid}
    h['shield'] = 0
    h['flags'].pop('slow', None)
    log(state, f"{h['name']}返回备战区。", 'move', side=side, source=cid,
        actor=f'{side}:{cid}', target=f'{side}:{cid}',
        before=before, after={'hp': h['hp'], 'shield': h['shield'], 'front': None})


def discard(state, side, card, *, bottom=False):
    from .presentation import public_card
    team = state['sides'][side]
    shown = public_card(card)
    if card.get('copy'):
        team['removed'].append(card)
        log(state, f"临时复制「{card['name']}」移出对局。", 'remove', side=side, card=shown)
    elif bottom:
        team['deck'].append(card)
    else:
        team['discard'].append(card)
        log(state, f"「{card['name']}」进入弃牌堆。", 'discard', side=side, card=shown)


def knockdowns(state):
    for side in SIDES:
        for cid in ORDER:
            h = hero(state, side, cid)
            if h['hp'] > 0 or h['down_turns'] > 0:
                continue
            h['hp'] = 0
            move_out(state, side, cid)
            before = {'hp': 0, 'shield': h['shield'], 'harmony': h['harmony'], 'growth': h['growth'],
                      'shape_id': h['shape'], 'down_turns': h['down_turns']}
            h.update(down_turns=3, harmony=0, shield=0, growth=0, shape=None, flags={})
            log(state, f"{h['name']}倒地，3 次己方回合开始后恢复。", 'down', side=side,
                source=cid, target=f'{side}:{cid}', actor=f'{side}:{cid}',
                before=before, after={'hp': 0, 'shield': 0, 'harmony': 0, 'growth': 0,
                                      'shape_id': None, 'down_turns': 3})
            if cid == 'xun':
                team = state['sides'][side]
                team['records'] = []
                for card in list(team['hand']):
                    if card.get('copy'):
                        team['hand'].remove(card)
                        discard(state, side, card)


def damage(state, side, cid, amount, *, bypass=False, source=None, kind='damage', settle=True, group_id=None):
    if finished(state):
        return 0
    target = hero(state, side, cid) if cid else state['sides'][side]
    if cid and target['hp'] <= 0:
        return 0
    amount = max(0, int(amount))
    before = {'hp': target['hp'], 'shield': target['shield']}
    absorbed = 0 if bypass else min(target['shield'], amount)
    target['shield'] -= absorbed
    loss = amount - absorbed
    target['hp'] = max(0, target['hp'] - loss)
    after = {'hp': target['hp'], 'shield': target['shield']}
    extra = {}
    if group_id:
        extra['group_id'] = group_id
    log(state, f"{name(state, side, cid)}受到 {amount} 点伤害（护盾吸收 {absorbed}）。",
        kind, side=side, source=source, actor=source, target=f'{side}:{cid or "player"}',
        amount=amount, before=before, after=after, **extra)
    if settle:
        knockdowns(state)
        victory(state)
    return loss


def heal(state, side, cid, amount):
    if finished(state) or not alive(state, side, cid):
        return
    h = hero(state, side, cid)
    actual = min(amount, h['max_hp'] - h['hp'])
    before = {'hp': h['hp'], 'shield': h['shield']}
    h['hp'] += actual
    if actual:
        log(state, f"{h['name']}回复 {actual} 点生命。", 'heal', side=side, source=cid,
            actor=f'{side}:{cid}', target=f'{side}:{cid}', amount=actual,
            before=before, after={'hp': h['hp'], 'shield': h['shield']})


def shield(state, side, cid, amount):
    if finished(state) or not alive(state, side, cid):
        return
    h = hero(state, side, cid)
    before = {'hp': h['hp'], 'shield': h['shield']}
    h['shield'] = max(h['shield'], amount)
    log(state, f"{h['name']}获得 {amount} 点护盾（取较大值）。", 'shield', side=side, source=cid,
        actor=f'{side}:{cid}', target=f'{side}:{cid}', amount=amount,
        before=before, after={'hp': h['hp'], 'shield': h['shield']})


def energy(state, side, cid, amount=1):
    h = hero(state, side, cid)
    if not finished(state) and h['hp'] > 0 and not h['awakened']:
        h['energy'] = min(4, h['energy'] + amount)


def add_hand(state, side, card, *, reveal=False):
    from .presentation import public_card
    if finished(state):
        return
    team = state['sides'][side]
    if len(team['hand']) >= 10:
        discard(state, side, card)
    else:
        team['hand'].append(card)
        if reveal or card.get('copy'):
            log(state, f"{name(state, side)}获得{'临时复制' if card.get('copy') else ''}「{card['name']}」。",
                'gain', side=side, card=public_card(card), actor=f'{side}:player')


def draw(state, side, count=1):
    from .presentation import public_card
    team = state['sides'][side]
    for _ in range(count):
        if finished(state):
            break
        if team['deck']:
            card = team['deck'].pop(0)
            add_hand(state, side, card)
            log(state, f"{name(state, side)}抽 1 张牌。", 'draw', side=side,
                actor=f'{side}:player', private_side=side, private_card=public_card(card))
        else:
            team['fatigue'] += 1
            damage(state, side, None, team['fatigue'], bypass=True, kind='fatigue')


def front_target(state, side):
    enemy = other(side)
    return enemy, state['sides'][enemy]['front']


def used(state, side, key):
    return state['sides'][side]['used'].get(key, False)


def consume_once(state, side, key):
    if used(state, side, key):
        return False
    state['sides'][side]['used'][key] = True
    return True
