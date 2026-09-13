from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.state import alive, attack_value, damage, hero, other


def _j01(c):
    c.inspect_top_choice(3)


def _j02(c):
    from app.modules.card_game.engine.duel_v2.state import front_target
    side, cid = front_target(c.state, c.side)
    if cid:
        c.add_atk_debuff(side, cid, 2)
        c.attach_pact(side, cid)


def _j03(c):
    side, cid = c.operation['target_id'].split(':', 1)
    hero(c.state, side, cid)['down_turns'] = 3


def _j04(c):
    side, cid = c.operation['target_id'].split(':', 1)
    damage(c.state, side, cid, 2, source=f'{c.side}:{c.cid}')
    c.attach_pact(side, cid)


def _j05(c):
    side, cid = c.operation['target_id'].split(':', 1)
    c.add_atk_debuff(side, cid, 2)


def _j06(c):
    foe = other(c.side)
    front = c.state['sides'][foe].get('front')
    if not front:
        return
    revealed = c.state['sides'][foe].setdefault('revealed_ids', [])
    for card in c.state['sides'][foe]['hand']:
        if card.get('character_id') == front and card['instance_id'] not in revealed:
            revealed.append(card['instance_id'])


@register
class Jiuyuan:
    id = 'jiuyuan'
    text_name = '九原'
    effects = {
        'J01': _j01, 'J02': _j02, 'J03': _j03, 'J04': _j04,
        'J05': _j05, 'J06': _j06, 'J07': set_shape, 'J08': set_shape,
    }
    target_policies = {
        'J02': 'enemy_front',
        'J03': 'downed_enemy',
        'J04': 'living_enemy',
        'J05': 'living_enemy',
    }

    @staticmethod
    def genesis_extra(c) -> int:
        return 1 if c.character['hp'] > 0 else 0

    @staticmethod
    def on_ultimate(c):
        foe = other(c.side)
        for cid in list(c.state['sides'][foe]['characters']):
            if not alive(c.state, foe, cid):
                continue
            amount = 1
            flags = hero(c.state, foe, cid).setdefault('flags', {})
            if flags.get('pact'):
                amount += 2
                flags.pop('pact', None)
            damage(c.state, foe, cid, amount, source=f'{c.side}:{c.cid}')

    @staticmethod
    def on_genesis_damage(c, target=None, **_data):
        if c.character.get('shape') != 'J08' or not target:
            return
        side, cid = target
        if cid and not alive(c.state, side, cid):
            return
        damage(c.state, side, cid, 1, source=f'{c.side}:{c.cid}')
        if cid:
            c.attach_pact(side, cid)

    @staticmethod
    def on_damage_dealt(c, target_side=None, target_id=None, amount=0, **_data):
        if c.character.get('shape') != 'J07' or not target_id or amount <= 0:
            return
        if not alive(c.state, target_side, target_id):
            return
        if attack_value(hero(c.state, target_side, target_id)) == 0:
            hero(c.state, target_side, target_id)['hp'] = 0
