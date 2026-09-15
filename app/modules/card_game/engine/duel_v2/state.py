"""JSON state primitives; no persistence, HTTP, or hidden-information policy."""
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from random import Random

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, energy_max_for_attribute
from app.modules.card_game.content.duel_v2.catalog import CHARACTER_ORDER
from app.modules.card_game.content.duel_v2.registry import copy_owner_id, shape_attack_bonus

SIDES = ('a', 'b')
ORDER = CHARACTER_ORDER
PLAYER_ID = 'player'


def other(side):
    return 'b' if side == 'a' else 'a'


def is_player(cid):
    return cid in (None, PLAYER_ID)


def is_esper(cid):
    return bool(cid) and cid != PLAYER_ID


def source_parts(source):
    if not isinstance(source, str) or not source:
        return None, None
    if ':' in source:
        side, cid = source.split(':', 1)
        if side in SIDES:
            return side, None if is_player(cid) else cid
    return None, None


def team_order(state, side):
    team = state['sides'][side]
    return list(team.get('order') or team['characters'])


def hero(state, side, cid):
    return state['sides'][side]['characters'][cid]


def alive(state, side, cid):
    characters = state['sides'][side]['characters']
    return cid in characters and characters[cid]['hp'] > 0


def log(state, text, kind='effect', **fields):
    from .presentation import enrich_log
    seq = state['event_seq'] + 1
    state['event_seq'] = seq
    event = {'seq': seq, 'type': kind, 'text': text, **fields}
    state['events'].append(event)
    enrich_log(state, event)


def name(state, side, cid=None):
    if is_player(cid):
        return '甲方玩家' if side == 'a' else '乙方玩家'
    return hero(state, side, cid)['name']


def attack_value(character):
    flags = character.get('flags') or {}
    shape_id = character.get('shape')
    from app.modules.card_game.content.duel_v2.tutorial import lookup_card
    form = lookup_card(shape_id) if shape_id else None
    form_attack = int((form or {}).get('attack') or 0)
    form_hp = int((form or {}).get('hp') or 0)
    from app.modules.card_game.content.duel_v2.tutorial import lookup_character
    try:
        catalog_attack = int(lookup_character(character.get('id')).get('attack') or 0)
    except (KeyError, TypeError):
        catalog_attack = int(CHARACTERS.get(character.get('id'), {}).get('attack') or 0)
    perm_atk = int(character.get('base_attack') or 0) - catalog_attack
    if form and form.get('type') == 'form' and form_attack > 0 and form_hp > 0:
        value = form_attack + max(0, perm_atk)
    else:
        value = (character['base_attack'] + character.get('growth', 0)
                 + character.get('gaze', 0)
                 + shape_attack_bonus(character['shape']))
    value += int(flags.get('form_attack') or 0)
    value += int(flags.get('atk_buff') or 0)
    value -= int(flags.get('atk_debuff') or 0)
    return max(0, value)


def add_atk_buff(character, amount=1):
    flags = character.setdefault('flags', {})
    gained = int(amount or 0)
    if gained == 0:
        return 0
    flags['atk_buff'] = int(flags.get('atk_buff') or 0) + gained
    return gained


def energy_max(character):
    return energy_max_for_attribute(character.get('attribute'))


def round_half_up(value):
    return int(Decimal(value).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def collapsed(character):
    return bool((character.get('flags') or {}).get('collapse'))


def front_debuff(state, side):
    return state['sides'][side].setdefault('front_debuff', {})


def source_attribute(state, source):
    if not isinstance(source, str) or ':' not in source:
        return None
    side, cid = source.split(':', 1)
    if side not in SIDES or cid not in state['sides'][side]['characters']:
        return None
    return hero(state, side, cid).get('attribute')


def damage_class_for(kind, damage_class=None):
    if damage_class:
        return damage_class
    if kind in ('genesis', 'flower', 'delay', 'burn', 'star', 'overlay', 'sync'):
        return 'harmony'
    return 'esper'


def new_character(cid):
    from app.modules.card_game.content.duel_v2.tutorial import lookup_character
    c = lookup_character(cid)
    return {**deepcopy(c), 'base_attack': c['attack'], 'hp': c['max_hp'],
            'base_max_hp': c['max_hp'], 'shield': 0,
            'harmony': 0, 'energy': 0, 'awakened': False, 'ultimate_turns': 0,
            'growth': 0, 'family': 0, 'gaze': 0, 'dream': 0, 'shape': None, 'down_turns': 0, 'flags': {}}


def card_instance(state, card_id, side, *, copy=False):
    from app.modules.card_game.content.duel_v2.tutorial import lookup_card
    state['next_instance'] += 1
    c = deepcopy(lookup_card(card_id) or CARDS[card_id])
    c.update(card_id=card_id, instance_id=f"{side}-{state['next_instance']}", copy=copy)
    if copy:
        c['original_character_id'] = c['character_id']
        c['character_id'] = copy_owner_id() or c['character_id']
        c['expires_turn'] = state['sides'][side]['turn_count'] + 1
    return c


def shuffle(state, cards):
    # Store a small integer stream state, rather than non-JSON Random tuples.
    rng = Random(state['rng'])
    rng.shuffle(cards)
    state['rng'] = rng.getrandbits(63)


def finished(state):
    return state['phase'] == 'finished'


def victory(state):
    dead = [s for s in SIDES if state['sides'][s]['hp'] <= 0]
    if not dead:
        return
    state['winner'] = 'draw' if len(dead) == 2 else other(dead[0])
    state['phase'] = 'finished'
    state['reason'] = 'player_hp'
    log(state, '双方生命归零，平局。' if len(dead) == 2 else f"{name(state, state['winner'])}获胜。",
        'finish', winner=state['winner'], reason=state.get('reason'))


def clear_harmony_source(state, side, cid):
    """Death expires a character's previous sortie, without consuming personal resources."""
    team = state['sides'][side]
    if team['last_front'] == cid:
        team['last_front'] = None


def move_out(state, side, cid):
    team = state['sides'][side]
    if team['front'] != cid:
        return
    h = hero(state, side, cid)
    team['last_front'] = cid if alive(state, side, cid) else None
    team['front'] = None
    before = {'hp': h['hp'], 'shield': h['shield'], 'front': cid}
    h['shield'] = 0
    log(state, f"{h['name']}返回备战区。", 'move', side=side, source=cid,
        actor=f'{side}:{cid}', target=f'{side}:{cid}',
        before=before, after={'hp': h['hp'], 'shield': h['shield'], 'front': None})


def discard(state, side, card, *, bottom=False):
    from .presentation import public_card
    team = state['sides'][side]
    shown = public_card(card)
    if card.get('copy'):
        team['removed'].append(card)
        log(state, f"临时复制「{card['name']}」移出对局。", 'remove', side=side, card=shown)
    elif bottom:
        team['deck'].append(card)
    else:
        team['discard'].append(card)
        log(state, f"「{card['name']}」进入弃牌堆。", 'discard', side=side, card=shown)


def knockdowns(state):
    for side in SIDES:
        for cid in team_order(state, side):
            h = hero(state, side, cid)
            if h['hp'] > 0 or h['down_turns'] > 0:
                continue
            h['hp'] = 0
            clear_harmony_source(state, side, cid)
            killer = h.pop('last_hit_source', None)
            move_out(state, side, cid)
            before = {'hp': 0, 'shield': h['shield'], 'harmony': h['harmony'], 'energy': h['energy'],
                      'growth': h['growth'], 'shape_id': h['shape'], 'down_turns': h['down_turns']}
            kept_harmony = int(before['harmony'] or 0)
            kept_energy = int(before['energy'] or 0)
            from app.modules.card_game.content.duel_v2.registry import fire
            from .context import EffectContext
            ctx = EffectContext(state, side, cid, {'side': side, 'downed': cid})
            fire(ctx, 'on_ally_down', include_down=True, downed=cid)
            fire(ctx, 'on_knockdown', actor_only=True, include_down=True)
            killer_side, killer_cid = source_parts(killer)
            if killer_side and killer_cid and killer_side != side and alive(state, killer_side, killer_cid):
                fire(EffectContext(state, killer_side, killer_cid, {'side': killer_side}),
                     'on_rout', actor_only=True, downed=cid)
            if h.get('awakened') or h.get('ultimate_turns'):
                from .flow import end_ultimate
                end_ultimate(state, side, cid)
            from app.modules.card_game.content.duel_v2.tutorial import lookup_character
            base_max = int(h.get('base_max_hp') or lookup_character(cid)['max_hp'])
            h.update(down_turns=3, harmony=kept_harmony, energy=kept_energy, shield=0, growth=0,
                     gaze=0, dream=0, shape=None, awakened=False, ultimate_turns=0,
                     max_hp=base_max, flags={})
            log(state, f"{h['name']}倒地，3 次己方回合开始后恢复。", 'down', side=side,
                source=cid, target=f'{side}:{cid}', actor=f'{side}:{cid}',
                before=before, after={'hp': 0, 'shield': 0, 'harmony': kept_harmony,
                                      'energy': kept_energy, 'growth': 0,
                                      'shape_id': None, 'down_turns': 3})


def damage(state, side, cid, amount, *, bypass=False, source=None, kind='damage', settle=True,
           group_id=None, damage_class=None):
    if finished(state):
        return 0
    esper = is_esper(cid)
    cid = PLAYER_ID if not esper else cid
    target = hero(state, side, cid) if esper else state['sides'][side]
    if esper and target['hp'] <= 0:
        return 0
    amount = max(0, int(amount))
    if esper:
        from app.modules.card_game.content.duel_v2.registry import decide
        from .context import EffectContext
        amount = decide(EffectContext(state, side, cid, {'side': side}), 'incoming_damage',
                        amount, kind=kind, target_id=cid, source=source)
        amount = max(0, int(amount or 0))
    klass = damage_class_for(kind, damage_class)
    attr = source_attribute(state, source)
    zone = front_debuff(state, side)
    if (kind == 'combat' and attr in ('魂', '相') and zone.get('stain')
            and (state['sides'][side]['front'] == cid or (not esper and not state['sides'][side]['front']))):
        amount += 2
    before = {'hp': target['hp'], 'shield': target['shield']}
    absorbed = 0 if bypass else min(target['shield'], amount)
    target['shield'] -= absorbed
    loss = amount - absorbed
    floor = 0
    if esper:
        from app.modules.card_game.content.duel_v2.registry import decide
        from .context import EffectContext
        if target['hp'] - loss <= 0:
            from .flow import trigger_responses
            trigger_responses(state, side, 'self_lethal', target_id=cid)
        floor = int(decide(EffectContext(state, side, cid, {'side': side}), 'hp_floor', 0,
                           target_id=cid) or 0)
    target['hp'] = max(floor, target['hp'] - loss)
    after = {'hp': target['hp'], 'shield': target['shield']}
    extra = {}
    if group_id:
        extra['group_id'] = group_id
    src_side, src_cid = source_parts(source)
    if src_side:
        extra['source_side'] = src_side
    if esper and source:
        target['last_hit_source'] = source
    log(state, f"{name(state, side, cid)}受到 {amount} 点伤害（护盾吸收 {absorbed}）。",
        kind, side=side, source=source, actor=source, target=f'{side}:{cid}',
        amount=amount, before=before, after=after, **extra)
    if klass == 'harmony' and isinstance(source, str) and ':' in source:
        src_side = source.split(':', 1)[0]
        if src_side in SIDES:
            state['sides'][src_side]['harmony_damage'] = True
    if loss and esper:
        from app.modules.card_game.content.duel_v2.registry import fire
        from .context import EffectContext
        fire(EffectContext(state, side, cid, {'side': side}), 'on_damage_taken',
             actor_only=True, amount=loss, source=source)
        if isinstance(source, str) and ':' in source:
            src_side, src_cid = source.split(':', 1)
            if src_side in SIDES and src_cid in state['sides'][src_side]['characters']:
                fire(EffectContext(state, src_side, src_cid, {'side': src_side}),
                     'on_damage_dealt', actor_only=True, target_side=side, target_id=cid, amount=loss)
    if settle:
        knockdowns(state)
        victory(state)
    return loss


def reduce_max_hp(state, side, cid, amount, *, source=None, settle=True, group_id=None):
    if finished(state) or not is_esper(cid):
        return 0
    h = hero(state, side, cid)
    if h['hp'] <= 0:
        return 0
    amount = max(0, int(amount))
    before = {'hp': h['hp'], 'max_hp': h['max_hp'], 'shield': h['shield']}
    cut = 0
    for _ in range(amount):
        old_max = h['max_hp']
        if old_max <= 0:
            break
        new_max = old_max - 1
        h['hp'] = min(round_half_up(Decimal(h['hp']) / Decimal(old_max) * Decimal(new_max)), new_max)
        h['max_hp'] = new_max
        cut += 1
        if h['hp'] <= 0 or h['max_hp'] <= 0:
            h['hp'] = 0
            break
    extra = {}
    if group_id:
        extra['group_id'] = group_id
    log(state, f"{h['name']}生命上限减少 {cut}（{before['hp']}/{before['max_hp']} → {h['hp']}/{h['max_hp']}）。",
        'max_hp', side=side, source=source, actor=source, target=f'{side}:{cid}',
        amount=cut, before=before, after={'hp': h['hp'], 'max_hp': h['max_hp'], 'shield': h['shield']},
        **extra)
    if settle:
        knockdowns(state)
        victory(state)
    return cut


def add_collapse(state, side, cid, amount=1, *, by=None):
    if finished(state) or not is_esper(cid) or not alive(state, side, cid):
        return
    h = hero(state, side, cid)
    flags = h.setdefault('flags', {})
    before = int(flags.get('collapse_count') or 0)
    flags['collapse_count'] = before + max(0, int(amount))
    actor = None
    if by and state['sides'][by].get('front'):
        actor = f"{by}:{state['sides'][by]['front']}"
    log(state, f"{h['name']}倾陷值 +{amount}（{flags['collapse_count']}/5）。", 'collapse',
        side=side, source=cid, actor=actor, target=f'{side}:{cid}', amount=amount,
        before={'collapse_count': before}, after={'collapse_count': flags['collapse_count']})
    if flags['collapse_count'] >= 5 and not flags.get('collapse'):
        flags['collapse_count'] = 0
        applier = by or other(side)
        flags['collapse'] = {'side': applier, 'until': state['sides'][applier]['turn_count'] + 1}
        log(state, f"{h['name']}进入倾陷。", 'collapse', side=side, source=cid,
            target=f'{side}:{cid}', after={'collapse': True})


def heal(state, side, cid, amount):
    if finished(state):
        return
    if is_player(cid):
        team = state['sides'][side]
        actual = min(amount, 30 - team['hp'])
        if actual <= 0:
            return
        before = {'hp': team['hp'], 'shield': team['shield']}
        team['hp'] += actual
        log(state, f"{name(state, side)}回复 {actual} 点生命。", 'heal', side=side,
            actor=f'{side}:player', target=f'{side}:player', amount=actual,
            before=before, after={'hp': team['hp'], 'shield': team['shield']})
        return
    h = hero(state, side, cid)
    if h.get('down_turns'):
        return
    if h['hp'] < 0:
        h['hp'] = 0
    actual = min(amount, h['max_hp'] - h['hp'])
    before = {'hp': h['hp'], 'shield': h['shield']}
    h['hp'] += actual
    if actual:
        log(state, f"{h['name']}回复 {actual} 点生命。", 'heal', side=side, source=cid,
            actor=f'{side}:{cid}', target=f'{side}:{cid}', amount=actual,
            before=before, after={'hp': h['hp'], 'shield': h['shield']})


def shield(state, side, cid, amount):
    if finished(state) or not alive(state, side, cid):
        return
    h = hero(state, side, cid)
    before = {'hp': h['hp'], 'shield': h['shield']}
    h['shield'] = max(h['shield'], amount)
    log(state, f"{h['name']}获得 {amount} 点护盾。", 'shield', side=side, source=cid,
        actor=f'{side}:{cid}', target=f'{side}:{cid}', amount=amount,
        before=before, after={'hp': h['hp'], 'shield': h['shield']})


def energy(state, side, cid, amount=1):
    if finished(state) or not alive(state, side, cid):
        return
    h = hero(state, side, cid)
    from .escalation import energy_cap
    h['energy'] = min(energy_cap(state, h), h['energy'] + amount)


def add_hand(state, side, card, *, reveal=False):
    from .presentation import public_card
    if finished(state):
        return
    team = state['sides'][side]
    if len(team['hand']) >= 10:
        discard(state, side, card)
    else:
        team['hand'].append(card)
        if reveal or card.get('copy') or card.get('derived'):
            log(state, f"{name(state, side)}获得{'临时复制' if card.get('copy') else ''}「{card['name']}」。",
                'gain', side=side, card=public_card(card), actor=f'{side}:player')


def draw(state, side, count=1):
    from .presentation import public_card
    team = state['sides'][side]
    for _ in range(count):
        if finished(state):
            break
        if team['deck']:
            card = team['deck'].pop(0)
            add_hand(state, side, card)
            log(state, f"{name(state, side)}抽 1 张牌。", 'draw', side=side,
                actor=f'{side}:player', private_side=side, private_card=public_card(card))
        else:
            state.update(phase='finished', winner=other(side), reason='empty_draw')
            log(state, f"{name(state, side)}牌库已空，无法抽牌。", 'finish', side=side,
                actor=f'{side}:player', winner=state['winner'], reason=state.get('reason'))
            break


def front_target(state, side):
    """前排优先：当前敌方前排；无人则 cid 为 player，伤害打敌方玩家。"""
    enemy = other(side)
    return enemy, state['sides'][enemy]['front'] or PLAYER_ID


def used(state, side, key):
    return state['sides'][side]['used'].get(key, False)


def consume_once(state, side, key):
    if used(state, side, key):
        return False
    state['sides'][side]['used'][key] = True
    return True
