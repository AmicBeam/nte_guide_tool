"""User-authored surplus carry rules, 2026-09-21 (not original-game numbers)."""
from copy import deepcopy
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.content.duel_v2.characters.common import draw_owned_card
from app.modules.card_game.engine.duel_v2.state import (
    alive, damage, discard, finished, hero, other, team_order,
)
from app.modules.card_game.engine.duel_v2.deferred import enqueue


def _battle(c):
    card_id = c.card['card_id']
    if card_id == 'RF01':
        c.draw(1)
    elif card_id in ('R02', 'R03', 'R04'):
        c.draw(1, card_type={'R02': 'tactic', 'R03': 'form', 'R04': 'battle'}[card_id])
    if not finished(c.state):
        c.sortie()


def _draw_team(c):
    for cid in team_order(c.state, c.side):
        if cid != c.cid and hero(c.state, c.side, cid)['attribute'] in ('光', '灵', '相'):
            draw_owned_card(c, cid)


def _redraw(c):
    team = c.state['sides'][c.side]
    cards = [card for card in team['hand'] if card['character_id'] == c.cid]
    for card in cards:
        team['hand'].remove(card)
        discard(c.state, c.side, card)
    c.draw(len(cards))


def _shape(c):
    c.character['intimidation'] = 0
    c.set_shape()


def _hand_changed(c):
    from app.modules.card_game.engine.duel_v2.state import effective_card, log
    from app.modules.card_game.engine.duel_v2.presentation import public_card
    log(c.state, '手牌视为效果更新。', 'effect', side=c.side, present=False,
        private_side=c.side, private_hand=[public_card(effective_card(c.state, c.side, card))
                                         for card in c.state['sides'][c.side]['hand']])


@register
class Zhenhong:
    id = 'zhenhong'
    text_name = '真红'
    effects = {'R01': _draw_team, 'R02': _battle, 'R03': _battle, 'R04': _battle,
               'R05': _redraw, 'R06': _battle, 'R07': _shape, 'R08': _shape,
               'RF01': _battle}

    @staticmethod
    def on_game_start(c):
        c.grant_harmony(c.cid, 1, reason='开局异能')

    @staticmethod
    def on_surplus(c, **_data):
        if finished(c.state) or not alive(c.state, c.side, c.cid):
            return
        # User tabletop mechanism 「独行」, 2026-10-07. Count the trigger,
        # even if the queued attack later becomes invalid.
        c.character['surplus_passive_triggers'] = int(c.character.get('surplus_passive_triggers', 0)) + 1
        from app.modules.card_game.engine.duel_v2.modifiers import add_effect, deadline
        add_effect(c.character, 'zhenhong.surplus_attack', 'attack.panel', 1,
                   source_entity_id=c.entity_id, source_key='passive', label='真红异能', show_marker=False,
                   expiry=deadline(c.state, c.side, 1, 'turn_start'))
        # User tabletop change, 2026-10-07: cap each damage event before shields.
        add_effect(c.character, 'zhenhong.surplus_damage_limit', 'damage.limit', 1, op='set',
                   source_entity_id=c.entity_id, source_key='passive', label='真红伤害限制', show_marker=False,
                   stacking='refresh', expiry=deadline(c.state, c.side, 1, 'turn_start'))
        c.history('己方触发盈蓄，真红触发「独行」：直到下个己方回合开始时攻击 +1，单次至多受到 1 点伤害，将发起额外攻击。')
        c.heal(c.character['max_hp'], reason='「独行」触发')
        if finished(c.state):
            return
        enqueue(c, 'surplus')
        if c.character.get('shape') == 'R07':
            foe = other(c.side)
            for cid in team_order(c.state, foe):
                if finished(c.state):
                    break
                if alive(c.state, foe, cid) and not hero(c.state, foe, cid).get('summoned'):
                    damage(c.state, foe, cid, 2, source=f'{c.side}:{c.cid}', reason='「梦的边缘」触发')

    @staticmethod
    def on_deferred_action(c, action):
        if action.get('kind') != 'surplus':
            return
        c.character['extra_attacks'] = int(c.character.get('extra_attacks', 0)) + 1
        c.operation['skip_combat_harmony'] = True
        c.history('「独行」：真红发起额外攻击。')
        c.sortie(support_allows_counter=True)

    @staticmethod
    def aura_effects(c, property, query):
        from app.modules.card_game.engine.duel_v2.modifiers import contribution
        if property == 'rule.gain_energy' and (c.character['awakened'] or c.state['sides'][c.side]['front'] != c.cid):
            yield contribution('zhenhong.energy_gate', property, True, op='forbid', kind='aura',
                               source_entity_id=c.entity_id, source_key='ability', label='真红能量限制')

    @staticmethod
    def natural_return(c, current=True, **_data):
        return False if c.character['awakened'] else current

    @staticmethod
    def hand_card(c, current, card, **_data):
        if not c.character['awakened'] or card.get('card_id') == 'unknown':
            return None
        from app.modules.card_game.content.duel_v2 import CARDS
        result = deepcopy(CARDS['RF01'])
        for key in ('entity_id', 'instance_id', 'side', 'copy', 'jingu_mark', 'expires_turn', 'original_character_id'):
            if key in card:
                result[key] = deepcopy(card[key])
        result.update(card_id='RF01', _physical_card=card)
        from app.modules.card_game.engine.duel_v2.entities import CardEntity
        return CardEntity(result)

    @staticmethod
    def on_ultimate(c):
        _hand_changed(c)
        c.enter_front()

    @staticmethod
    def on_effect_expired(c, effect, **_data):
        if (effect['definition_id'] == 'status.ultimate'
                and effect.get('source_entity_id') == c.entity_id):
            # Natural expiry only; leaving the front or going down never fires
            # this hook. 「角色」 includes the player and front summons.
            foe = other(c.side)
            targets = [cid for cid in c.state['sides'][foe]['characters']
                       if alive(c.state, foe, cid)] + ['player']
            for cid in targets:
                if finished(c.state):
                    break
                damage(c.state, foe, cid, 3, source=f'{c.side}:{c.cid}',
                       reason='真红终结自然结束')

    @staticmethod
    def on_ultimate_end(c):
        _hand_changed(c)

    @staticmethod
    def on_leave_front(c):
        from app.modules.card_game.engine.duel_v2.flow import end_ultimate
        end_ultimate(c.state, c.side, c.cid)

    @staticmethod
    def on_knockdown(c):
        c.character['intimidation'] = 0

    @staticmethod
    def battle_frame(c, attack, shield):
        if c.card and c.card['card_id'] == 'R06':
            count = int(c.character.get('surplus_passive_triggers', 0))
            return attack + count, shield + count
        return attack, shield

    @staticmethod
    def on_turn_end(c):
        team = c.state['sides'][c.side]
        if c.character.get('shape') == 'R08' and c.entity_id not in team.get('attacked_this_turn', []):
            c.character['intimidation'] = min(2, int(c.character.get('intimidation', 0)) + 1)
            c.history(f"「穿过胭红蜃景」触发：威慑凝视 {c.character['intimidation']} 点。")

    @staticmethod
    def on_sortie_ready(c, actor_id, **_data):
        if actor_id != c.cid or c.character.get('shape') != 'R08':
            return
        count = int(c.character.get('intimidation', 0))
        c.character['intimidation'] = 0
        c.operation['sortie_attack_bonus'] += count
        if count:
            c.grant_harmony(c.cid, count, reason='消耗威慑凝视')
