"""User-authored September 18 rework; numerical rules are not Everness values."""
from random import Random
from app.modules.card_game.content.duel_v2.registry import register
from .common import set_shape

# Unspecified family effects deliberately remain unavailable until designed.
FAMILY_CARDS = {'nanali': 'N03', 'zero': 'Z02', 'adler': None,
                'daffodill': None, 'edgar': None, 'zaowu': None}
ANTIQUE_WEIGHTS = {1: (10, 80, 10, 0), 2: (5, 10, 80, 5), 3: (0, 10, 10, 80)}


def family_options(state, side, *, available=True):
    from app.modules.card_game.engine.duel_v2.state import alive, collapsed
    team = state['sides'][side]
    result = []
    for cid, card in FAMILY_CARDS.items():
        h = team['characters'].get(cid)
        if card and h and (not available or (alive(state, side, cid) and not collapsed(h))):
            result.append((cid, card))
    if not available or any(h['down_turns'] > 0 for h in team['characters'].values() if not h.get('summoned')):
        result.append(('summon', None))
    return result


def family_description(state, side):
    from app.modules.card_game.content.duel_v2 import CARDS
    parts = [f"视为打出一张「{CARDS[card]['name']}」" if card
             else '在战斗区召唤一个 2/4 的塔吉多'
             for _, card in family_options(state, side)]
    if not parts:
        return '当前无可用选项。'
    count = len(parts)
    prefix = ({2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七'}.get(count, str(count))
              + '选一：') if count > 1 else ''
    return prefix + '；或'.join(parts) + '。'


def _rewind(c):
    from app.modules.card_game.engine.duel_v2.temporal import rewind
    rewind(c.state, c.side, c.card)


def _lock(c):
    c.state['sides'][c.side]['life_locked'] = True
    c.history('直到下个己方回合开始时，己方玩家生命不会发生变化。')


def _inspect(c):
    c.state['sides'][c.side]['deck_visible_turn'] = c.state['turn']
    c.history('本回合可以查看己方牌库顺序。')


def _antique(c):
    from app.modules.card_game.engine.duel_v2.state import card_instance, add_hand
    card = card_instance(c.state, 'XA01', c.side)
    card['antique_investment'] = c.operation['ap_paid']
    add_hand(c.state, c.side, card)


def _cash_antique(c):
    from app.modules.card_game.engine.duel_v2.state import alive
    rng = Random(c.state['rng'])
    amount = rng.choices(range(4), weights=ANTIQUE_WEIGHTS[c.card['antique_investment']])[0]
    c.state['rng'] = rng.getrandbits(63)
    if amount == 0 and alive(c.state, c.side, 'zero'):
        amount = 1
    c.state['sides'][c.side]['ap'] += amount
    c.history(f'随机古董：获得 {amount} 点行动力。')


def _retreat(c):
    from app.modules.card_game.engine.duel_v2.state import move_out
    front = c.state['sides'][c.side]['front']
    if front:
        move_out(c.state, c.side, front)


def _gain_family(c, count):
    from app.modules.card_game.engine.duel_v2.state import card_instance, add_hand
    for _ in range(count):
        card = card_instance(c.state, 'XF01', c.side)
        card['description'] = family_description(c.state, c.side)
        add_hand(c.state, c.side, card)


def _heal(c):
    c.heal(c.character['max_hp'])
    _gain_family(c, 1)


def _collapse(c):
    from app.modules.card_game.engine.duel_v2.state import add_collapse
    add_collapse(c.state, c.side, c.cid, 5, by=c.side)
    _gain_family(c, 2)


def _family(c):
    from app.modules.card_game.engine.duel_v2.state import card_instance
    choices = []
    for cid, card_id in family_options(c.state, c.side):
        card = card_instance(c.state, card_id or 'XF01', c.side)
        card['family_option'] = cid
        if cid == 'summon':
            card.update(name='召唤塔吉多', description='在战斗区召唤 2/4 的塔吉多。')
        choices.append(card)
    c.choose('kit', choices, '伊波恩大家庭：选择一项')


def _arid(c, actor_id):
    from app.modules.card_game.engine.duel_v2.state import hero
    if actor_id == c.cid:
        return
    actor = hero(c.state, c.side, actor_id)
    if actor.get('summoned'):
        return
    seen = c.character.setdefault('arid_contributors', [])
    if actor['entity_id'] not in seen:
        seen.append(actor['entity_id'])
        c.character['arid'] = len(seen)
        c.history(f"{actor['name']}发动终结或援护技，浔获得 1 点荒时。")


@register
class Xun:
    id = 'xun'
    text_name = '浔'
    copy_owner = True  # Identity compatibility for pre-rework generated copies; no new copies.
    effects = {'X01': _rewind, 'X02': _lock, 'X03': _inspect, 'X04': _antique,
               'X05': _retreat, 'X06': _heal, 'X07': _collapse, 'X08': set_shape,
               'XF01': _family, 'XA01': _cash_antique}

    @staticmethod
    def on_game_start(c):
        c.character.update(arid=0, arid_contributors=[])

    @staticmethod
    def normal_attack_payment(c, current):
        return False if any(h['hp'] > 0 and h['id'] != c.cid and not h.get('summoned')
                            for h in c.state['sides'][c.side]['characters'].values()) else current

    @staticmethod
    def can_play(c, current, card):
        from app.modules.card_game.engine.duel_v2.temporal import can_rewind
        team = c.state['sides'][c.side]
        if card['card_id'] == 'X01': return can_rewind(c.state, c.side)
        if card['card_id'] == 'XF01': return bool(family_options(c.state, c.side))
        if card['card_id'] == 'X05':
            h = team['characters'].get(team['front'], {})
            return h.get('faction') == '伊波恩古董店' and h.get('hp', 0) > 0
        return current

    @staticmethod
    def card_description(c, current, card):
        return family_description(c.state, c.side) if card['card_id'] == 'XF01' else current

    @staticmethod
    def on_turn_clock(c):
        from app.modules.card_game.engine.duel_v2.escalation import energy_cap
        before = c.character['energy']
        c.character['energy'] = min(energy_cap(c.state, c.character), before + 1)
        if c.character['energy'] > before:
            c.history('己方回合开始，浔获得 1 点时计。')

    @staticmethod
    def on_ally_ultimate(c, actor_id):
        _arid(c, actor_id)

    @staticmethod
    def on_support(c, actor_id):
        _arid(c, actor_id)

    @staticmethod
    def can_ultimate(c, current):
        from app.modules.card_game.engine.duel_v2.state import other
        return any(h['hp'] > 0 and not h.get('summoned')
                   for h in c.state['sides'][other(c.side)]['characters'].values())

    @staticmethod
    def on_ultimate(c):
        from app.modules.card_game.engine.duel_v2.state import other, card_instance
        from app.modules.card_game.engine.duel_v2.flow import operation
        operation(c.state, c.side, c.cid)
        choices = []
        for cid, h in c.state['sides'][other(c.side)]['characters'].items():
            if h['hp'] > 0 and not h.get('summoned'):
                card = card_instance(c.state, 'XF01', c.side)
                card.update(name=h['name'], description='选择为浔的终结目标。',
                            ultimate_target=cid, selection_kind='character', portrait=h.get('portrait'), avatar=h.get('avatar'))
                choices.append(card)
        c.choose('kit', choices, '选择浔的终结目标')

    @staticmethod
    def on_choice(c, selected):
        from app.modules.card_game.engine.duel_v2.state import (other, damage, hero, alive, finished,
                                                               knockdowns, victory)
        target_id = selected.get('ultimate_target')
        if target_id:
            enemy = other(c.side)
            arid = c.character.get('arid', 0)
            c.character.update(arid=0, arid_contributors=[])
            bonus = 3 if c.character.get('shape') == 'X08' else 0
            primary = 4 + (2 if arid >= 1 else 0) + (2 if arid >= 3 else 0) + bonus
            def hit(cid, amount):
                if finished(c.state) or not alive(c.state, enemy, cid): return
                h = hero(c.state, enemy, cid)
                before_hp = h['hp']
                loss = damage(c.state, enemy, cid, amount, source=f'{c.side}:{c.cid}', settle=False)
                if arid >= 3 and loss > before_hp:
                    damage(c.state, enemy, 'player', loss-before_hp, source=f'{c.side}:{c.cid}', kind='penetration', settle=False)
                knockdowns(c.state)
                victory(c.state)
            hit(target_id, primary)
            if arid >= 2:
                for cid, h in list(c.state['sides'][enemy]['characters'].items()):
                    if not h.get('summoned'): hit(cid, 2 + bonus)
            return
        option = selected['family_option']
        if option == 'summon':
            from app.modules.card_game.engine.duel_v2.summons import summon_front
            summon_front(c.state, c.side, name='塔吉多', attack=2, hp=4)
        else:
            from app.modules.card_game.engine.duel_v2.flow import resolve_play
            from app.modules.card_game.engine.duel_v2.state import card_instance
            parent = c.state['operation']
            card = card_instance(c.state, FAMILY_CARDS[option], c.side)
            # Virtual play uses the ally's own kit, but no physical card enters discard.
            card['virtual'] = True
            resolve_play(c.state, c.side, card, pay=False)
            c.state['operation'] = parent
