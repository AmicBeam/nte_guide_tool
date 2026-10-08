from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.state import (
    add_atk_buff, alive, attack_value, damage, front_debuff, heal, hero, is_esper, log, other, team_order,
)


def _b01(c):
    damage(c.state, c.side, c.cid, 1, source=f'{c.side}:{c.cid}')
    c.draw(1)


def _b02(c):
    c.sortie()


def _b03(c):
    zone = front_debuff(c.state, other(c.side))
    burn = zone.get('burn')
    if burn and burn.get('by') == c.side:
        burn['infinite'] = True
        log(c.state, '己方施加的浊燃变为无限。', 'harmony', side=c.side, source=c.cid,
            actor=f'{c.side}:{c.cid}')


def _b04(c):
    missing = max(0, int(c.character['max_hp']) - int(c.character['hp']))
    if missing <= 0:
        return
    for side in c.state['sides']:
        for cid in team_order(c.state, side):
            if hero(c.state, side, cid).is_same_entity(c.character):
                continue
            if alive(c.state, side, cid):
                damage(c.state, side, cid, missing, source=f'{c.side}:{c.cid}')


def _b05(c):
    c.character.setdefault('flags', {})['hp_floor'] = 1


def _b06(c):
    extra = 1 if (c.character.get('flags') or {}).get('allied_hurt') else 0
    c.sortie(bonus=extra)


def _b08(c):
    set_shape(c)
    if alive(c.state, c.side, c.cid):
        damage(c.state, c.side, c.cid, 4, source=f'{c.side}:{c.cid}')


@register
class Baicang:
    id = 'baicang'
    text_name = '白藏'
    effects = {
        'B01': _b01, 'B02': _b02, 'B03': _b03, 'B04': _b04,
        'B05': _b05, 'B06': _b06, 'B07': set_shape, 'B08': _b08,
    }

    @staticmethod
    def on_damage_taken(c, amount=0, source=None, **_data):
        if int(amount or 0) <= 0:
            return
        before = attack_value(c.character, c.state, c.side)
        add_atk_buff(c.character, 1, source_entity_id=c.entity_id, source_key='ability')
        src_side = None
        if isinstance(source, str) and ':' in source:
            src_side = source.split(':', 1)[0]
        if src_side == c.side:
            c.character.setdefault('flags', {})['allied_hurt'] = True
        c.history('白藏受到伤害，触发异能：攻击 +1。',
                  before={'attack': before}, after={'attack': attack_value(c.character, c.state, c.side)})

    @staticmethod
    def on_damage_dealt(c, target_side=None, target_id=None, amount=0, **_data):
        if not c.character.get('awakened') or int(amount or 0) <= 0:
            return
        if target_side != other(c.side) or not is_esper(target_id) or not alive(c.state, target_side, target_id):
            return
        target = hero(c.state, target_side, target_id)
        if target.is_same_entity(c.character):
            return
        if 0 < int(target['hp']) <= 2:
            target['hp'] = 0
            log(c.state, f"白藏造成伤害，触发终结：将{target['name']}消灭。", 'effect',
                side=target_side, source=c.cid, actor=f'{c.side}:{c.cid}',
                target=f'{target_side}:{target_id}')

    @staticmethod
    def after_hits(c, target_loss=0, **_data):
        if c.card and c.card.get('card_id') == 'B02' and int(target_loss or 0) > 0:
            heal(c.state, c.side, c.cid, 2, reason='造成伤害，「招魂幡」触发')

    @staticmethod
    def hp_floor(c, current=0, target_id=None, **_data):
        if target_id != c.cid:
            return None
        if (c.character.get('flags') or {}).get('hp_floor'):
            return max(int(current or 0), 1)
        return None

    @staticmethod
    def on_turn_begin(c):
        flags = c.character.setdefault('flags', {})
        flags.pop('allied_hurt', None)
        flags.pop('hp_floor', None)

    @staticmethod
    def on_enemy_turn_begin(c):
        flags = c.character.setdefault('flags', {})
        flags.pop('hp_floor', None)
