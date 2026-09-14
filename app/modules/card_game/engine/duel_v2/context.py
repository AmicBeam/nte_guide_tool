"""Explicit capability context passed down to card content callbacks."""
from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS
from app.modules.card_game.content.duel_v2.registry import shape_event
from . import combat
from .state import (add_hand, alive, consume_once, damage, draw, energy, finished,
                    front_target, heal, hero, is_esper, log, other, shield)


class EffectContext:
    def __init__(self, state, side, cid, operation):
        self.state, self.side, self.cid, self.operation = state, side, cid, operation
        self.card = operation.get('card')

    @property
    def character(self):
        return hero(self.state, self.side, self.cid)

    def for_character(self, side, cid):
        return EffectContext(self.state, side, cid, self.operation)

    def on_shape(self, event, **data):
        if not finished(self.state) and alive(self.state, self.side, self.cid):
            shape_event(self, event, **data)

    def once(self, key):
        return consume_once(self.state, self.side, key)

    def is_enemy_turn(self):
        return self.state['active_side'] != self.side

    def draw(self, count):
        draw(self.state, self.side, count)

    def energy(self):
        energy(self.state, self.side, self.cid)

    def heal(self, amount):
        heal(self.state, self.side, self.cid, amount)

    def shield(self, amount):
        shield(self.state, self.side, self.cid, amount)

    def heal_character(self, cid, amount):
        heal(self.state, self.side, cid, amount)

    def set_shape(self):
        from .presentation import public_card
        previous = self.character['shape']
        before = {'shape_id': previous, 'hp': self.character['hp'], 'max_hp': self.character['max_hp']}
        self.character['shape'] = self.card['card_id']
        panel_hp = int(self.card.get('hp') or 0)
        if panel_hp:
            from app.modules.card_game.content.duel_v2.tutorial import lookup_character
            catalog_hp = int((lookup_character(self.cid) or {}).get('max_hp') or 0)
            perm_hp = int(self.character.get('base_max_hp') or catalog_hp) - catalog_hp
            self.character['max_hp'] = panel_hp + max(0, perm_hp)
            self.character['hp'] = self.character['max_hp']
        log(self.state, f"{self.character['name']}装备武备：{self.card['name']}。",
            'shape', side=self.side, source=self.cid, actor=f'{self.side}:{self.cid}',
            target=f'{self.side}:{self.cid}', card=public_card(self.card),
            before=before,
            after={'shape_id': self.card['card_id'], 'hp': self.character['hp'],
                   'max_hp': self.character['max_hp']})

    def attach_pact(self, side, cid):
        if finished(self.state) or not is_esper(cid) or not alive(self.state, side, cid):
            return
        hero(self.state, side, cid).setdefault('flags', {})['pact'] = True

    def add_atk_debuff(self, side, cid, amount=2):
        if finished(self.state) or not is_esper(cid) or not alive(self.state, side, cid):
            return
        flags = hero(self.state, side, cid).setdefault('flags', {})
        flags['atk_debuff'] = int(flags.get('atk_debuff') or 0) + amount

    def sortie(self, **spec):
        combat.sortie(self, **spec)

    def hit_front(self, amount, bypass=False):
        from .state import reduce_max_hp
        side, cid = front_target(self.state, self.side)
        if bypass:
            reduce_max_hp(self.state, side, cid, amount, source=f'{self.side}:{self.cid}')
        else:
            damage(self.state, side, cid, amount, source=f'{self.side}:{self.cid}')

    def front_wounded(self):
        side, cid = front_target(self.state, self.side)
        if not is_esper(cid):
            return False
        h = hero(self.state, side, cid)
        return h['hp'] < h['max_hp']

    def clear_front_shield(self):
        side, cid = front_target(self.state, self.side)
        if is_esper(cid) and not finished(self.state):
            hero(self.state, side, cid)['shield'] = 0

    def heal_target(self, amount, front_shield=0):
        side, cid = self.operation['target_id'].split(':', 1)
        heal(self.state, side, cid, amount)
        if self.state['sides'][side]['front'] == cid:
            shield(self.state, side, cid, front_shield)

    def followup(self, target, amount):
        side, cid = target
        if is_esper(cid) and (not alive(self.state, side, cid) or self.state['sides'][side]['front'] != cid):
            return
        from .presentation import next_group_id
        damage(self.state, side, cid, amount, source=f'{self.side}:{self.cid}',
               kind='followup', group_id=next_group_id(self.state))

    def choose(self, kind, cards, prompt):
        if finished(self.state) or not cards:
            return
        self.state['phase'] = 'choice'
        self.state['pending_choice'] = {'side': self.side, 'kind': kind, 'prompt': prompt,
                                        'cards': cards}

    def discard_choice(self):
        self.choose('discard', list(self.state['sides'][self.side]['hand']), '选择 1 张手牌弃置')

    def inspect_top_choice(self, count):
        if finished(self.state):
            return
        team = self.state['sides'][self.side]
        cards, team['deck'] = team['deck'][:count], team['deck'][count:]
        if len(cards) == 1:
            add_hand(self.state, self.side, cards[0])
        elif cards:
            self.choose('inspect_top', cards, '检视牌库顶：选择 1 张加入手牌，其余置于牌库底')

    def inspect_enemy_hand(self):
        self.choose('enemy_hand', list(self.state['sides'][other(self.side)]['hand']),
                    '选择对手的 1 张手牌，公开并置于其牌库底')

    def reveal_top_owned(self):
        if finished(self.state):
            return
        team = self.state['sides'][self.side]
        if not team['deck']:
            return
        card = team['deck'].pop(0)
        from .presentation import public_card
        log(self.state, f"{self.character['name']}展示牌库顶「{card['name']}」。",
            'reveal', side=self.side, source=self.cid, actor=f'{self.side}:{self.cid}',
            card=public_card(card))
        if card['character_id'] == self.cid:
            add_hand(self.state, self.side, card, reveal=True)
        else:
            team['deck'].append(card)
            self.energy()

    def team_next_attack(self):
        amount = 2 if self.state['sides'][self.side]['harmonized'].get(self.cid) else 1
        for h in self.state['sides'][self.side]['characters'].values():
            if h['hp'] > 0:
                h['flags']['next_attack'] = max(h['flags'].get('next_attack', 0), amount)

    def extra_flower(self):
        if not finished(self.state):
            self.state['sides'][self.side]['extra_flower'] = True

    def extra_delay(self):
        if not finished(self.state):
            self.state['sides'][self.side]['extra_delay'] = True

    def grant_harmony(self, cid, amount=1):
        if finished(self.state) or not alive(self.state, self.side, cid):
            return
        h = hero(self.state, self.side, cid)
        h['harmony'] = min(2, h['harmony'] + amount)

    def add_gaze(self, amount=1, cap=None):
        if finished(self.state) or self.character['hp'] <= 0:
            return 0
        limit = cap if cap is not None else (4 if self.character['awakened'] else 2)
        before = self.character.get('gaze', 0)
        self.character['gaze'] = min(limit, before + amount)
        return self.character['gaze'] - before

    def recover_target(self):
        team = self.state['sides'][self.side]
        card = next((c for c in team['discard'] if c['instance_id'] == self.operation['target_id']), None)
        if card:
            team['discard'].remove(card)
            add_hand(self.state, self.side, card, reveal=True)

    def return_after_sortie(self):
        self.character['flags']['return_after'] = True
