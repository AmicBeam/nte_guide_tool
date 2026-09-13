from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.events import GameEvent


def _x01(c):
    c.draw(1)
    if c.state['sides'][c.side].get('records'):
        c.draw(1)


def _x02(c):
    c.return_after_sortie()


def _x03(c):
    c.return_after_sortie()


def _x04(c):
    c.sortie()


def _x05(c):
    c.sortie(hits=2, no_counter=True)


def _x06(c):
    c.sortie(bypass=True)


@register
class Xun:
    id = 'xun'
    text_name = '浔'
    copy_owner = True
    effects = {
        'X01': _x01, 'X02': _x02, 'X03': _x03, 'X04': _x04,
        'X05': _x05, 'X06': _x06, 'X07': set_shape, 'X08': set_shape,
    }

    @staticmethod
    def on_operation_start(c):
        team = c.state['sides'][c.side]
        c.operation['record_candidate'] = c.character['hp'] > 0 and team['front'] != c.cid

    @staticmethod
    def before_hits(c):
        from app.modules.card_game.engine.duel_v2.state import add_hand, alive, card_instance, finished, log
        s, side, op, card = c.state, c.side, c.operation, c.card
        team = s['sides'][side]
        if (not card or card.get('copy') or not card.get('redeem') or card['type'] != 'battle'
                or card['character_id'] != c.cid or not team['records'] or team['used'].get('redeem')
                or finished(s) or team['front'] != c.cid or not alive(s, side, c.cid)):
            return
        records = team['records']
        team['records'] = []
        team['used']['redeem'] = True
        op['redeem_bonus'] = 2 if c.character['awakened'] else 1
        log(s, f"浔兑现 {len(records)} 条记录，本次出击所有攻击伤害 +{op['redeem_bonus']}。",
            'redeem', side=side, source=c.cid, actor=f'{side}:{c.cid}', amount=len(records))
        for record in records:
            add_hand(s, side, card_instance(s, record['card_id'], side, copy=True))
        c.on_shape(GameEvent.V2_RECORD_REDEEMED, operation=op, count=len(records))

    @staticmethod
    def after_card(c, *, actor_id, card):
        from app.modules.card_game.engine.duel_v2.state import hero, log
        s, side = c.state, c.side
        team = s['sides'][side]
        op = c.operation
        if (not op.get('record_candidate') or c.character['hp'] <= 0 or team['front'] == c.cid
                or actor_id == c.cid or card.get('copy')
                or card['type'] not in ('battle', 'tactic') or card.get('redeem')
                or team['used'].get('record')):
            return
        team['used']['record'] = True
        record = {key: card[key] for key in ('card_id', 'name', 'type', 'character_id')}
        records = [row for row in team['records'] if row['character_id'] != actor_id]
        team['records'] = (records + [record])[-(2 if c.character['awakened'] else 1):]
        from app.modules.card_game.engine.duel_v2.presentation import public_card
        log(s, f"浔记录了{hero(s, side, actor_id)['name']}的「{card['name']}」。",
            'record', side=side, source=c.cid, actor=f'{side}:{c.cid}',
            target=f'{side}:{actor_id}', card=public_card(card))

    @staticmethod
    def on_ultimate_end(c):
        team = c.state['sides'][c.side]
        team['records'] = team['records'][-1:]

    @staticmethod
    def on_knockdown(c):
        from app.modules.card_game.engine.duel_v2.state import discard
        team = c.state['sides'][c.side]
        team['records'] = []
        for card in list(team['hand']):
            if card.get('copy'):
                team['hand'].remove(card)
                discard(c.state, c.side, card)
