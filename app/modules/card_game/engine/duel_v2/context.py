"""Explicit capability context passed down to card content callbacks."""
from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS
from app.modules.card_game.content.duel_v2.registry import shape_event
from . import combat
from .state import (add_hand, alive, attack_value, consume_once, damage, draw, energy, finished,
                    front_target, heal, hero, is_esper, log, other, shield)


class EffectContext:
    def __init__(self, state, side, cid, operation):
        self.state, self.side, self.cid, self.operation = state, side, cid, operation
        self.card = operation.get('card')

    @property
    def character(self):
        return hero(self.state, self.side, self.cid)

    @property
    def entity_id(self):
        return self.character.entity_id

    def for_character(self, side, cid):
        return EffectContext(self.state, side, cid, self.operation)

    def for_weapon(self, card_id):
        from .equipment import weapon_id, weapon_source
        if weapon_id(self.character) != card_id:
            raise ValueError('Weapon is not equipped')
        result = EffectContext(self.state, self.side, self.cid, self.operation)
        result.effect_source_key = weapon_source(self.character)
        return result

    def weapon_source(self, card_id):
        from .equipment import weapon_id, weapon_source
        return weapon_source(self.character) if weapon_id(self.character) == card_id else card_id

    def on_shape(self, event, **data):
        if not finished(self.state) and alive(self.state, self.side, self.cid):
            shape_event(self, event, **data)

    def once(self, key):
        return consume_once(self.state, self.side, key)

    def history(self, text, *, target_id=None, target_side=None, present=False, **fields):
        """Record a resolved public effect, silent unless presentation is requested.

        Call at the actual trigger, after its mutation; never from eligibility
        checks. Do not attach the operation card: it may belong to someone else
        (or be a hidden response card).
        """
        if getattr(self, 'effect_source_key', None):
            fields.setdefault('effect_source_key', self.effect_source_key)
        log(self.state, text, 'effect', side=self.side, source=self.cid,
            actor=f'{self.side}:{self.cid}',
            target=f'{target_side or self.side}:{target_id or self.cid}',
            present=present, **fields)

    def is_enemy_turn(self):
        return self.state['active_side'] != self.side

    def draw(self, count, *, card_type=None):
        draw(self.state, self.side, count, card_type=card_type)

    def energy(self, amount=1, *, reason=None):
        before = self.character['energy']
        energy(self.state, self.side, self.cid, amount)
        gained = self.character['energy'] - before
        if gained:
            self.history(f"{reason or self.character['name'] + '的效果'}：{self.character['name']}获得 {gained} 点能量。",
                         before={'energy': before}, after={'energy': self.character['energy']})

    def heal(self, amount, *, reason=None):
        heal(self.state, self.side, self.cid, amount, reason=reason, source=f'{self.side}:{self.cid}')

    def shield(self, amount, *, reason=None):
        shield(self.state, self.side, self.cid, amount, reason=reason)

    def heal_character(self, cid, amount):
        heal(self.state, self.side, cid, amount, source=f'{self.side}:{self.cid}')

    def set_shape(self):
        from .presentation import public_card
        previous = self.character['shape']
        before = {'shape_id': previous, 'hp': self.character['hp'], 'max_hp': self.character['max_hp']}
        from .equipment import equip_weapon, refill_after_equipping
        has_life_panel = equip_weapon(self.state, self.character, self.card)
        if has_life_panel:
            refill_after_equipping(self.character)
        log(self.state, f"{self.character['name']}装备武备：{self.card['name']}。",
            'shape', side=self.side, source=self.cid, actor=f'{self.side}:{self.cid}',
            target=f'{self.side}:{self.cid}', card=public_card(self.card),
            before=before,
            after={'shape_id': self.card['card_id'], 'hp': self.character['hp'],
                   'max_hp': self.character['max_hp']})

    def attach_pact(self, side, cid):
        if finished(self.state) or not is_esper(cid) or not alive(self.state, side, cid):
            return
        h = hero(self.state, side, cid)
        before = bool(h.setdefault('flags', {}).get('pact'))
        h['flags']['pact'] = True
        if not before:
            self.history(f"{self.character['name']}的效果：为{h['name']}附着枚约。",
                         target_side=side, target_id=cid)

    def add_atk_debuff(self, side, cid, amount=2):
        if finished(self.state) or not is_esper(cid) or not alive(self.state, side, cid):
            return
        h = hero(self.state, side, cid)
        before = attack_value(h, self.state, side)
        from .modifiers import add_effect
        add_effect(h, 'attack.debuff', 'attack.panel', -amount,
                   source_entity_id=self.entity_id, source_key=(self.card or {}).get('card_id', 'ability'), label='攻击降低')
        if amount:
            self.history(f"{self.character['name']}的效果：{h['name']}攻击 -{amount}。",
                         target_side=side, target_id=cid,
                         before={'attack': before}, after={'attack': attack_value(h, self.state, side)})

    def sortie(self, **spec):
        combat.sortie(self, **spec)

    def enter_front(self):
        combat.enter(self, as_support=False)

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
        heal(self.state, side, cid, amount, source=f'{self.side}:{self.cid}')
        if self.state['sides'][side]['front'] == cid:
            shield(self.state, side, cid, front_shield)

    def followup(self, target, amount, *, reason=None):
        side, cid = target
        if is_esper(cid) and (not alive(self.state, side, cid) or self.state['sides'][side]['front'] != cid):
            return
        from .presentation import next_group_id
        damage(self.state, side, cid, amount, source=f'{self.side}:{self.cid}',
               kind='followup', group_id=next_group_id(self.state), reason=reason)

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

    def apply_effect(self, definition_id, property, value, *, target=None, **options):
        from .modifiers import add_effect
        options.setdefault('source_entity_id', self.entity_id)
        options.setdefault('source_key', getattr(self, 'effect_source_key', None) or (self.card or {}).get('card_id', 'ability'))
        return add_effect(self.character if target is None else target, definition_id, property, value, **options)

    def grant_next_attack(self, target, amount):
        from .modifiers import add_effect
        # Existing next-attack effects share a single maximum, not additive stacks.
        add_effect(target, 'attack.next', 'attack.hit', amount, stacking='max',
                   source_entity_id=self.entity_id, source_key=(self.card or {}).get('card_id', self.cid),
                   stack_key='next_attack', label='攻击强化', consume_on='attack')

    def team_next_attack(self):
        amount = 2 if self.state['sides'][self.side]['harmonized'].get(self.cid) else 1
        for h in self.state['sides'][self.side]['characters'].values():
            if h['hp'] > 0:
                self.grant_next_attack(h, amount)

    def extra_flower(self):
        if not finished(self.state):
            self.state['sides'][self.side]['extra_flower'] = True

    def extra_delay(self):
        if not finished(self.state):
            self.state['sides'][self.side]['extra_delay'] = True

    def grant_harmony(self, cid, amount=1, *, reason=None):
        if finished(self.state) or not alive(self.state, self.side, cid):
            return
        h = hero(self.state, self.side, cid)
        from .escalation import harmony_cap
        before = h['harmony']
        h['harmony'] = min(harmony_cap(self.state), h['harmony'] + amount)
        gained = h['harmony'] - before
        if gained:
            self.history(f"{reason or self.character['name'] + '的效果'}：{h['name']}获得 {gained} 点环合值。",
                         target_id=cid, before={'harmony': before}, after={'harmony': h['harmony']})

    def add_gaze(self, amount=1, cap=None):
        if finished(self.state) or self.character['hp'] <= 0:
            return 0
        limit = cap if cap is not None else (4 if self.character['awakened'] else 2)
        before = self.character.get('gaze', 0)
        self.character['gaze'] = min(limit, before + amount)
        gained = self.character['gaze'] - before
        if gained:
            self.history(f"{self.character['name']}获得 {gained} 层凝视（现有 {self.character['gaze']} 层）。",
                         before={'gaze': before}, after={'gaze': self.character['gaze']})
        return gained

    def recover_target(self):
        team = self.state['sides'][self.side]
        card = next((c for c in team['discard'] if c['instance_id'] == self.operation['target_id']), None)
        if card:
            team['discard'].remove(card)
            add_hand(self.state, self.side, card, reveal=True)

    def return_after_sortie(self):
        self.character['flags']['return_after'] = True
