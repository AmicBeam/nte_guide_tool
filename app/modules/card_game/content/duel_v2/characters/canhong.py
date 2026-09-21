from app.modules.card_game.content.duel_v2.characters.common import add_brand, negative_kinds, set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.events import GameEvent


def _front_hero(c):
    from app.modules.card_game.engine.duel_v2.state import front_target, hero, is_esper
    enemy, target_id = front_target(c.state, c.side)
    if not is_esper(target_id):
        return None, None, None
    return enemy, target_id, hero(c.state, enemy, target_id)


def _c01(c):
    c.draw(1)
    c.energy()


def _c03(c):
    c.sortie(bonus=1, shield_after_shape=0)
    if c.character.get('shape') and c.character['hp'] > 0:
        c.heal(1)


def _c04(c):
    enemy, _, target = _front_hero(c)
    no_counter = bool(target and negative_kinds(target, c.state, enemy))
    c.sortie(no_counter=no_counter)


def _c06(c):
    _, _, target = _front_hero(c)
    extra = min(3, int((target.get('flags') or {}).get('brand') or 0)) if target else 0
    c.hit_front(1 + extra)


def _c07(c):
    c.sortie(bonus=2, penetrate_brand=2)


def _c08(c):
    _, _, target = _front_hero(c)
    bonus = 4 if target and (target.get('flags') or {}).get('burn') else 2
    c.sortie(bonus=bonus)


def _c02_shape(c, **data):
    if data.get('operation', {}).get('harmony') == '浊燃':
        data['operation']['first_bonus'] += 1
        c.history('触发浊燃，「噬心诡刃」触发：残虹本次第一次攻击伤害 +1。')


def _c05_shape(c, **_data):
    if not c.once('C07'):
        return
    _, _, target = _front_hero(c)
    if target and (target.get('flags') or {}).get('brand'):
        from app.modules.card_game.engine.duel_v2.state import front_target
        c.followup(front_target(c.state, c.side), 1, reason='目标有焚印，「惑心谲影」触发')


@register
class Canhong:
    id = 'canhong'
    text_name = '残虹'
    effects = {
        'C01': _c01, 'C02': _c07, 'C03': _c03, 'C04': _c04,
        'C05': _c08, 'C06': _c06, 'C07': set_shape, 'C08': set_shape,
    }
    target_policies = {'C06': 'enemy_front'}
    shape_events = {
        ('C08', GameEvent.V2_HARMONY_RESOLVED): _c02_shape,
        ('C07', GameEvent.V2_SORTIE_FINISHED): _c05_shape,
    }

    @staticmethod
    def _attack_bonus(c, attacker_id=None, target=None, target_id=None, **_data):
        if attacker_id != c.cid or not target_id:
            return 0
        return int((target.get('flags') or {}).get('brand') or 0)

    @staticmethod
    def attack_bypass(c, current=False, attacker_id=None, target=None, target_id=None, **_data):
        if current or attacker_id != c.cid or not c.character['awakened'] or not target_id:
            return None
        if (target.get('flags') or {}).get('brand'):
            return True
        return None

    @staticmethod
    def after_sortie(c):
        from app.modules.card_game.engine.duel_v2.state import finished
        if finished(c.state) or c.character['hp'] <= 0:
            return
        enemy, target_id, target = _front_hero(c)
        if target:
            gained = add_brand(target, 1)
            if gained:
                c.history(f"出击结束，残虹的异能触发：{target['name']}焚印 +{gained}（现有 {target['flags']['brand']} 层）。",
                          target_side=enemy, target_id=target_id)

    @staticmethod
    def aura_effects(c, property, query):
        if property != 'attack.hit':
            return
        from app.modules.card_game.engine.duel_v2.modifiers import contribution
        amount = Canhong._attack_bonus(c, **{k: v for k, v in query.items() if k != 'operation'})
        if amount:
            yield contribution('canhong.attack_bonus', property, amount, kind='aura', scope='allies',
                               source_entity_id=c.entity_id, source_key='ability', label=c.character['name'])
