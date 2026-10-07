"""阿德勒：护盾即攻击，诛恶护持与浊燃层数。数值是桌游试玩参数。"""
from random import Random

from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.continuous import apply_dot, burn_present, deal_one, living_espers
from app.modules.card_game.engine.duel_v2.modifiers import add_effect, condition, contribution
from app.modules.card_game.engine.duel_v2.state import (
    alive, damage, front_debuff, hero, name, other, shield, team_order,
)


@condition('attribute_is')
def _attribute_is(ctx, args):
    subject = ctx.get('subject') or {}
    return subject.get('attribute') == args.get('value')


def _rng(state):
    rng = Random(state['rng'])
    state['rng'] = rng.getrandbits(63)
    return rng


def _guard(c, turns):
    apply_dot(c.state, other(c.side), 'guard', by=c.side, source_cid=c.cid, stacks=1, duration=turns)


def _d01(c):
    side, cid = c.operation['target_id'].split(':', 1)
    damage(c.state, side, cid, 3, source=f'{c.side}:{c.cid}', reason='「无色」')
    shield(c.state, c.side, c.cid, 3)
    if not burn_present(c.state):
        return
    foe = other(c.side)
    others = [item for item in living_espers(c.state, foe) if item != cid]
    if not others:
        return
    pick = _rng(c.state).choice(others)
    damage(c.state, foe, pick, 3, source=f'{c.side}:{c.cid}', reason='浊燃，「无色」')
    shield(c.state, c.side, c.cid, 3)


def _d02(c):
    shield(c.state, c.side, 'player', 3)
    for cid in team_order(c.state, c.side):
        h = hero(c.state, c.side, cid)
        if h.get('summoned') or h.get('hp', 0) <= 0:
            continue
        shield(c.state, c.side, cid, 3)
    _guard(c, 3)


def _d03(c):
    side, cid = c.operation['target_id'].split(':', 1)
    shield(c.state, side, cid, 4, reason='「无想」')


def _d04(c):
    c.sortie()
    if not burn_present(c.state):
        return
    dealt = int((c.operation or {}).get('last_attack_amount') or 0)
    if dealt:
        from app.modules.card_game.engine.duel_v2.state import heal
        heal(c.state, c.side, 'player', dealt, source=f'{c.side}:{c.cid}', reason='浊燃，「无行」')


def _d05(c):
    c.sortie()
    _guard(c, 3 if burn_present(c.state) else 2)


def _d06(c):
    set_shape(c)
    if alive(c.state, c.side, c.cid):
        shield(c.state, c.side, c.cid, 2, reason='「诸法空相」')


@register
class Adler:
    id = 'adler'
    text_name = '阿德勒'
    effects = {
        'D01': _d01, 'D02': _d02, 'D03': _d03, 'D04': _d04, 'D05': _d05,
        'D06': _d06, 'D07': set_shape, 'D08': set_shape,
    }
    target_policies = {'D01': 'living_enemy', 'D03': 'other_living_ally'}

    @staticmethod
    def panel_attack(c, current=None, **_data):
        return int(c.character.get('shield') or 0)

    @staticmethod
    def on_turn_begin(c):
        amount = 3 if burn_present(c.state) else 2
        shield(c.state, c.side, c.cid, amount, reason='阿德勒的异能')

    @staticmethod
    def on_ultimate(c):
        _guard(c, 2)
        foe = other(c.side)
        for cid in ['player', *team_order(c.state, foe)]:
            target = c.state['sides'][foe] if cid == 'player' else hero(c.state, foe, cid)
            removed = int(target.get('shield') or 0)
            if not removed:
                continue
            before = {'hp': target['hp'], 'shield': removed}
            target['shield'] = 0
            c.history(f"阿德勒的终结：移除{name(c.state, foe, cid)}的 {removed} 点护盾。",
                      target_side=foe, target_id=cid, before=before,
                      after={'hp': target['hp'], 'shield': 0}, shield_removed=removed)
        c.history('发动终结：施加「诛恶护持」，持续 2 个己方回合，并移除所有对方角色的护盾。')

    @staticmethod
    def redirect_overflow(c, current=None, loss=0, absorbed=0, target_id=None, **_data):
        if c.character.get('shape') != 'D06' or target_id == c.cid:
            return None
        if absorbed > 0 and loss > 0 and alive(c.state, c.side, c.cid):
            return c.cid
        return None

    @staticmethod
    def aura_effects(c, property, query):
        if property != 'attack.panel' or c.character.get('shape') != 'D07':
            return
        yield contribution('adler.curse_attack', property, 1, kind='aura', scope='other_allies',
                           source_entity_id=c.entity_id, source_key='D07', label='克己',
                           conditions=[{'id': 'attribute_is', 'args': {'value': '咒'}}])

    @staticmethod
    def on_burn_stacks(c, gained=0, enemy=None, **_data):
        if c.character.get('shape') != 'D08' or not gained or not enemy:
            return
        targets = living_espers(c.state, enemy)
        if not targets:
            return
        pick = _rng(c.state).choice(targets)
        h = hero(c.state, enemy, pick)
        flags = h.setdefault('flags', {})
        options = []
        if not any(effect.get('definition_id') == 'adler.all_resist' for effect in h.get('effects') or []):
            options.append('resist')
        if int(flags.get('collapse_max') or 5) > 4:
            options.append('collapse')
        if not any(effect.get('definition_id') == 'adler.attack_cut' for effect in h.get('effects') or []):
            options.append('attack')
        if not options:
            return
        choice = _rng(c.state).choice(options)
        if choice == 'resist':
            add_effect(h, 'adler.all_resist', 'damage.resistance', -1, stacking='replace',
                       source_entity_id=c.entity_id, source_key='D08', label='全属性抗性 -1',
                       clear_on=('down',))
            c.history(f"浊燃获得层数，「正心」触发：{h['name']}全属性抗性 -1。", target_side=enemy, target_id=pick)
        elif choice == 'collapse':
            flags['collapse_max'] = 4
            c.history(f"浊燃获得层数，「正心」触发：{h['name']}倾陷值上限 -1。", target_side=enemy, target_id=pick)
        else:
            add_effect(h, 'adler.attack_cut', 'attack.panel', -1, stacking='replace',
                       source_entity_id=c.entity_id, source_key='D08', label='攻击 -1',
                       clear_on=('down',))
            c.history(f"浊燃获得层数，「正心」触发：{h['name']}攻击 -1。", target_side=enemy, target_id=pick)
