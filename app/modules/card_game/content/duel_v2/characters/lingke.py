"""Lingke's user-authored sync support rules, 2026-10-08."""
from copy import deepcopy
from .common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.state import (
    add_hand, alive, finished, hero, other, shuffle, team_order,
)
from app.modules.card_game.engine.duel_v2.modifiers import add_effect, all_effects, deadline


def _sync(c, attribute, target=None, *, support_actor=None, front_only=False):
    if finished(c.state) or not alive(c.state, c.side, c.cid):
        return
    if front_only and not c.state['sides'][other(c.side)]['front']:
        return
    from app.modules.card_game.engine.duel_v2.combat import additional_attack, resolve_harmony_effect
    c.history(f'{c.character["name"]}发起{attribute}属性的「同频合击」。', present=True)
    if c.character.get('shape') == 'K07':
        if attribute == '光':
            resolve_harmony_effect(c, '创生')
        elif attribute == '咒':
            resolve_harmony_effect(c, '覆纹')
    additional_attack(c, attribute, target=target, sync_attack=True,
                      support_actor=support_actor, front_only=front_only)


def _k01(c):
    owner = c.operation['target_id'].split(':', 1)[1]
    unique = {}
    for card in c.state['sides'][c.side]['deck']:
        if card['character_id'] == owner:
            unique.setdefault(card['card_id'], card)
    if not unique:
        return
    candidates = list(unique.values())
    shuffle(c.state, candidates)
    Lingke.on_choice_cards(c, candidates[:3])


def _k02(c):
    c.draw(1)
    if not finished(c.state):
        add_effect(c.state['sides'][c.side], 'lingke.next_instant', 'card.instant', True,
                   op='allow', scope='allies', source_entity_id=c.entity_id,
                   source_key='K02', stacking='replace', clear_on=(), consume_on='card_use',
                   conditions=[{'id': 'card_owner_other'}], label='下一张非灵可牌瞬发')


def _k04(c):
    side, cid = c.operation['target_id'].split(':', 1)
    add_effect(hero(c.state, side, cid), 'lingke.taunt', 'marker.lingke_taunt', 1,
               source_entity_id=c.entity_id, source_key='K04', stacking='replace',
               label='小鬼打这里', show_marker=True)
    c.history('附着「小鬼打这里」。', target_side=side, target_id=cid)


def _k05(c):
    side, cid = c.operation['target_id'].split(':', 1)
    owner = hero(c.state, side, cid)
    _sync(c, owner['attribute'], support_actor=dict(character_id=cid, entity_id=owner.entity_id))


def _k06(c):
    owners = [(cid, hero(c.state, c.side, cid).entity_id, hero(c.state, c.side, cid)['attribute'])
              for cid in team_order(c.state, c.side)
              if cid != c.cid and alive(c.state, c.side, cid)
              and not hero(c.state, c.side, cid).get('summoned')]
    for cid, entity_id, attribute in owners:
        if alive(c.state, c.side, cid):
            _sync(c, attribute, support_actor=dict(character_id=cid, entity_id=entity_id), front_only=True)
    _sync(c, c.character['attribute'], front_only=True)


def _k03(c):
    if not c.card.get('_k03_recovery'):
        set_shape(c)
        return
    team = c.state['sides'][c.side]
    team['ap'] += 1
    c.history('「星间漫游」：获得 1 点行动力。', before={'ap': team['ap'] - 1}, after={'ap': team['ap']})


@register
class Lingke:
    id = 'lingke'
    text_name = '灵可'
    effects = {'K01': _k01, 'K02': _k02, 'K03': _k03, 'K04': _k04,
               'K05': _k05, 'K06': _k06, 'K07': set_shape, 'K08': set_shape}
    target_policies = {'K01': 'ally_esper', 'K04': 'living_enemy_esper', 'K05': 'living_ally'}

    @staticmethod
    def on_card_used(c, actor_id, card):
        actor = hero(c.state, c.side, actor_id)
        if actor.entity_id != c.entity_id and not actor.get('summoned'):
            chain = c.character.get('lingke_chain') or {}
            count = int(chain.get('count', 0)) + 1 if chain.get('entity_id') == actor.entity_id else 1
            c.character['lingke_chain'] = dict(entity_id=actor.entity_id, character_id=actor_id, count=count % 3)
            if count == 3:
                c.history(f'{actor["name"]}连续使用 3 张牌，触发「同频合击」。')
                _sync(c, actor['attribute'],
                      support_actor=dict(character_id=actor_id, entity_id=actor.entity_id))
        if card.get('type') != 'battle':
            return
        foe = other(c.side)
        for cid in list(team_order(c.state, foe)):
            target = hero(c.state, foe, cid)
            marks = [e for e in all_effects(target) if e['definition_id'] == 'lingke.taunt'
                     and e.get('source_entity_id') == c.entity_id]
            if not marks or not alive(c.state, foe, cid) or finished(c.state):
                continue
            ids = {e['effect_id'] for e in marks}
            target['effects'] = [e for e in target.get('effects', []) if e['effect_id'] not in ids]
            _sync(c, '灵', (foe, cid))
            if not finished(c.state):
                c.energy(1, reason='「小鬼打这里」触发')

    @staticmethod
    def on_deferred_action(c, action):
        if action.get('kind') == 'lingke_sync':
            _sync(c, action['attribute'], support_actor=action.get('support_actor'))

    @staticmethod
    def before_attack_damage(c, target_side, target_id, attribute, support=False, sync_attack=False, **_data):
        if not sync_attack and not (support and c.character.get('awakened')):
            return
        target = hero(c.state, target_side, target_id) if target_id != 'player' else c.state['sides'][target_side]
        add_effect(target, 'lingke.resistance', 'damage.resistance', -1,
                   stacking='replace', stack_key=f'lingke.resistance:{attribute}',
                   source_entity_id=c.entity_id, source_key='sync',
                   expiry=deadline(c.state, c.side, 1), label=f'{attribute}抗性 -1',
                   conditions=[{'id': 'damage_attribute', 'args': {'attribute': attribute}}])
        c.history(f'「同频合击」：{attribute}抗性 -1，持续到下个己方回合开始时。',
                  target_side=target_side, target_id=target_id)

    @staticmethod
    def attack_is_additional(c, current, support=False, **_data):
        return True if support and c.character.get('awakened') else current

    @staticmethod
    def additional_attack_damage_bonus(c, current, **_data):
        return current + 1 if c.character.get('shape') == 'K08' else current

    @staticmethod
    def weave_damage(c, current, **_data):
        return current + 1

    @staticmethod
    def hand_card(c, current, card, **_data):
        team = c.state['sides'][c.side]
        if (c.character.get('shape') != 'K03' or current.get('character_id') != c.cid
                or team['ap'] != 0 or team['used'].get('instant') or card.get('card_id') == 'unknown'):
            return None
        from app.modules.card_game.engine.duel_v2.entities import CardEntity
        # The equipped weapon owns this alternative effect; no derived card is
        # created. K03 has no target policy, regardless of the physical card.
        result = dict(id='K03', card_id='K03', character_id=c.cid, name=card['name'],
                      type='tactic', cost=1, terminal=False, instant=True,
                      description='瞬发。获得 1 点行动力。', effect_id='K03', _k03_recovery=True)
        for key in ('entity_id', 'instance_id', 'side', 'copy', 'jingu_mark', 'expires_turn', 'original_character_id'):
            if key in card:
                result[key] = deepcopy(card[key])
        result['_physical_card'] = card
        return CardEntity(result)

    @staticmethod
    def on_choice_cards(c, selected):
        team = c.state['sides'][c.side]
        ids = {card['instance_id'] for card in selected}
        team['deck'] = [card for card in team['deck'] if card['instance_id'] not in ids]
        c.choose('kit', selected, '选择 1 张加入手牌，其余洗回牌库')
        c.state['pending_choice']['deck_cards'] = True

    @staticmethod
    def on_choice(c, selected):
        team = c.state['sides'][c.side]
        cards = c.operation.pop('choice_cards', [])
        add_hand(c.state, c.side, selected)
        team['deck'].extend(card for card in cards if card['instance_id'] != selected['instance_id'])
        shuffle(c.state, team['deck'])
