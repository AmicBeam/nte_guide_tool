from app.modules.card_game.content.duel_v2.characters.common import (
    draw_owned_card, perm_plus, revive_character, set_shape,
)
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.state import (
    add_atk_buff, attack_value, damage, energy, front_target, heal, hero, is_player, log, shield,
)


def _mark_healed(c, side, cid):
    if side != c.side or is_player(cid):
        return
    target = hero(c.state, side, cid)
    before = attack_value(target, c.state, side)
    add_atk_buff(target, 1, source_entity_id=c.entity_id, source_key=(c.card or {}).get('card_id', c.cid))
    c.history(f"{c.character['name']}的治疗效果：{target['name']}攻击 +1。", target_side=side, target_id=cid,
              before={'attack': before}, after={'attack': attack_value(target, c.state, side)})


def _y01(c):
    downed = sum(1 for item in c.state['sides'][c.side]['characters'].values()
                 if item.get('down_turns'))
    if downed:
        side, cid = front_target(c.state, c.side)
        damage(c.state, side, cid, 3 * downed, source=f'{c.side}:{c.cid}')


def _y02(c):
    _side, cid = c.operation['target_id'].split(':', 1)
    draw_owned_card(c, cid, reveal=False)


def _y03(c):
    side, cid = c.operation['target_id'].split(':', 1)
    heal(c.state, side, cid, 5)
    _mark_healed(c, side, cid)


def _y04(c):
    flags = c.character.setdefault('flags', {})
    flags['next_bonus'] = int(flags.get('next_bonus') or 0) + 1
    flags['next_shield'] = int(flags.get('next_shield') or 0) + 2


def _y05(c):
    from app.modules.card_game.engine.duel_v2.state import alive, hero, knockdowns, team_order
    marked = []
    for cid in team_order(c.state, c.side):
        if cid == c.cid or not alive(c.state, c.side, cid):
            continue
        hero(c.state, c.side, cid)['hp'] = 0
        marked.append(cid)
    knockdowns(c.state)
    for cid in marked:
        h = hero(c.state, c.side, cid)
        if h.get('down_turns'):
            h['down_turns'] = 2
    c.draw(2)
    perm_plus(c, atk=1, hp=1)


def _y07_heal(c):
    from app.modules.card_game.engine.duel_v2.state import alive, shuffle, team_order
    candidates = [cid for cid in team_order(c.state, c.side)
                  if alive(c.state, c.side, cid)
                  and hero(c.state, c.side, cid)['hp'] < hero(c.state, c.side, cid)['max_hp']]
    if not candidates:
        return
    if len(candidates) > 1:
        shuffle(c.state, candidates)
    cid = candidates[0]
    heal(c.state, c.side, cid, 2, reason='「想做什么梦？」触发')
    _mark_healed(c, c.side, cid)


def _y07(c):
    set_shape(c)
    _y07_heal(c)


def _y06(c):
    side, cid = c.operation['target_id'].split(':', 1)
    revive_character(c, side, cid)
    _mark_healed(c, side, cid)


@register
class Iloy:
    id = 'iloy'
    text_name = '伊洛伊'
    effects = {
        'Y01': _y01, 'Y02': _y02, 'Y03': _y03, 'Y04': _y04,
        'Y05': _y05, 'Y06': _y06, 'Y07': _y07, 'Y08': set_shape,
    }
    target_policies = {
        'Y02': 'living_ally',
        'Y03': 'injured_role',
        'Y06': 'downed_ally',
    }

    @staticmethod
    def before_hits(c):
        flags = c.character.setdefault('flags', {})
        bonus = int(flags.pop('next_bonus', 0) or 0)
        c.operation['first_bonus'] += bonus
        if bonus:
            c.history(f'「策略特性：可激怒」触发：伊洛伊本次第一次攻击伤害 +{bonus}。')
        extra_shield = int(flags.pop('next_shield', 0) or 0)
        if extra_shield:
            shield(c.state, c.side, c.cid, extra_shield)

    @staticmethod
    def on_ultimate(c):
        c.character.setdefault('flags', {})['energy_next'] = True
        c.history('伊洛伊的终结生效：下个己方回合开始时，己方存活异能者各获得 1 点能量。')

    @staticmethod
    def after_genesis(c, entering=None, **_data):
        if c.character.get('shape') != 'Y08' or c.character['hp'] <= 0:
            return
        team = c.state['sides'][c.side]
        if team.get('extra_genesis_pending'):
            return
        team['extra_genesis_pending'] = True
        team['extra_genesis_actor'] = entering or c.cid
        c.history('「错误的门」触发：下个己方回合开始时追加一次创生。',
                  target_id=entering or c.cid)

    @staticmethod
    def on_turn_begin(c):
        if not c.character.get('flags', {}).pop('energy_next', False):
            return
        for item in c.state['sides'][c.side]['characters'].values():
            if item['hp'] <= 0:
                continue
            cid = item['id']
            before = item['energy']
            energy(c.state, c.side, cid, 1)
            gained = item['energy'] - before
            if gained:
                log(c.state, f"{item['name']}因伊洛伊的终结获得 1 点能量。", 'resource',
                    side=c.side, source=c.cid, actor=f'{c.side}:{c.cid}',
                    target=f'{c.side}:{cid}', energy_delta=gained,
                    before={'energy': before}, after={'energy': item['energy']})

        if c.character.get('shape') == 'Y07':
            _y07_heal(c)

    @staticmethod
    def on_ultimate_end(c):
        c.character.get('flags', {}).pop('energy_next', None)
