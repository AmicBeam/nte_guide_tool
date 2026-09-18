"""Edgar: user-authored card rules, Everness 1021 identity/name provenance."""
from app.modules.card_game.content.duel_v2.registry import register
from .common import set_shape


def _keys(c, amount):
    from app.modules.card_game.engine.duel_v2.state import alive, finished
    if finished(c.state) or not alive(c.state, c.side, c.cid):
        return
    c.character['truth_keys'] = c.character.get('truth_keys', 0) + amount
    c.history(f'埃德嘉获得 {amount} 点真理之匙。')


def _e01(c):
    c.state['sides'][c.side]['empty_draw_wins'] = True
    c.history('此后，己方无牌可抽时改为获胜。')


def _e03(c):
    # Remaining hand only: the played card has already left it.
    cards = list(c.state['sides'][c.side]['hand'][:6])
    if not cards:
        return
    c.choose('hand_redraw', cards, '手牌调度：选择至多 3 张替换')
    c.state['pending_choice']['max_select'] = 3


def _e04(c):
    team = c.state['sides'][c.side]
    lost = sum(max(0, h['max_hp'] - h['hp']) for cid, h in team['characters'].items()
               if cid != c.cid and not h.get('summoned'))
    c.energy(lost, reason='其他己方异能者的已损生命')


def _e05(c):
    from app.modules.card_game.content.duel_v2 import CARDS
    from app.modules.card_game.engine.duel_v2.state import card_instance, add_hand, shuffle, team_order, hero, finished, log
    for cid in team_order(c.state, c.side):
        if finished(c.state):
            return
        if cid == c.cid or hero(c.state, c.side, cid).get('summoned'):
            continue
        candidates = [key for key, card in CARDS.items() if card['character_id'] == cid and not card.get('derived')]
        if not candidates:
            continue
        shuffle(c.state, candidates)
        card = card_instance(c.state, candidates[0], c.side)
        card['instant'] = True
        add_hand(c.state, c.side, card)
        if any(held['entity_id'] == card['entity_id'] for held in c.state['sides'][c.side]['hand']):
            from app.modules.card_game.engine.duel_v2.presentation import public_card
            log(c.state, '埃德嘉的效果：随机获得 1 张队友牌，该牌获得瞬发。', 'gain',
                side=c.side, actor=f'{c.side}:{c.cid}', private_side=c.side, private_card=public_card(card))


def _e06(c):
    from app.modules.card_game.engine.duel_v2.state import heal, finished
    team = c.state['sides'][c.side]
    candidates = [(cid, h['hp']) for cid, h in team['characters'].items() if h['hp'] > 0]
    candidates.append(('player', team['hp']))
    target = min(candidates, key=lambda item: item[1])[0]
    heal(c.state, c.side, target, 5, source=f'{c.side}:{c.cid}')
    if not finished(c.state):
        _keys(c, 1)
        c.sortie()


def _ultimate_heal(c):
    from app.modules.card_game.engine.duel_v2.state import heal
    team = c.state['sides'][c.side]
    heal(c.state, c.side, team['front'] or 'player', 2, source=f'{c.side}:{c.cid}', reason='埃德嘉终结')


@register
class Edgar:
    id = 'edgar'
    text_name = '埃德嘉'
    effects = {'E01': _e01, 'E02': lambda c: c.draw(2), 'E03': _e03,
               'E04': _e04, 'E05': _e05, 'E06': _e06, 'E07': set_shape, 'E08': set_shape}

    @staticmethod
    def on_game_start(c):
        c.character['truth_keys'] = 0
        c.draw(1)

    @staticmethod
    def on_support(c, actor_id):
        _keys(c, 1)

    @staticmethod
    def on_knockdown(c):
        c.character['truth_keys'] = 0

    @staticmethod
    def can_play(c, current, card):
        team = c.state['sides'][c.side]
        if card['card_id'] == 'E01':
            return bool(team.get('surplus_ever'))
        if card['card_id'] == 'E03':
            return any(h['instance_id'] != card['instance_id'] for h in team['hand'])
        return current

    @staticmethod
    def ultimate_duration(c, current):
        return 1 + c.character.get('truth_keys', 0)

    @staticmethod
    def on_ultimate(c):
        count = c.character.get('truth_keys', 0)
        c.character['truth_keys'] = 0
        if count:
            c.history(f'埃德嘉消耗 {count} 点真理之匙，终结持续时间延长 {count} 个己方回合。')
        _ultimate_heal(c)

    @staticmethod
    def on_turn_begin(c):
        if c.character.get('awakened'):
            _ultimate_heal(c)

    @staticmethod
    def turn_draw_inspect(c, current):
        if c.character.get('shape') == 'E07':
            return {'character_id': c.cid, 'count': 3}
        return current

    @staticmethod
    def outgoing_heal(c, current):
        return current + (1 if c.character.get('shape') == 'E08' and current > 0 else 0)

    @staticmethod
    def on_ally_healed(c, target_id, amount, source):
        h = c.character
        if (amount <= 0 or h.get('shape') != 'E08'
                or h.get('heal_draw_turn') == c.state['turn']):
            return
        # Mark before drawing: a later heal or re-equipping cannot recurse/refresh.
        h['heal_draw_turn'] = c.state['turn']
        c.history('「扭曲之城的呼唤」触发：己方回复生命后，抽 1 张牌。')
        c.draw(1)
