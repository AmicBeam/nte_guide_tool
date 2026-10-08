"""JSON state primitives; no persistence, HTTP, or hidden-information policy."""
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from random import Random
from .entities import CharacterEntity, CardEntity, allocate_entity_id

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


def attack_value(character, state=None, side=None, *, include_zone=True, explain=False):
    from .equipment import attack_base
    layers = attack_base(character)
    value = layers['value']
    from .modifiers import evaluate
    base = value
    panel = evaluate(state, side or character.get('side'), character, 'attack.panel', value)
    override = evaluate(state, side or character.get('side'), character, 'attack.override', panel['value'])
    value = override['value']
    owner = side if side is not None else character.get('side')
    if state is not None and owner in SIDES and character.get('id') in state['sides'][owner]['characters']:
        from app.modules.card_game.content.duel_v2.registry import decide
        from .context import EffectContext
        replaced = decide(EffectContext(state, owner, character['id'], {'side': owner}),
                          'panel_attack', None, actor_only=True)
        if replaced is not None:
            value = max(0, int(replaced))
            if explain:
                return dict(base=base, layers=layers, value=value, contributions=[
                    e for e in panel['contributions'] + override['contributions'] if e['visibility'] == 'public'])
            return value
    if include_zone and state is not None and owner in SIDES:
        value -= zone_attack_penalty(state, owner, character.get('id'))
    value = max(0, int(value))
    if explain:
        return dict(base=base, layers=layers, value=value, contributions=[e for e in panel['contributions'] + override['contributions']
                                                         if e['visibility'] == 'public'])
    return value


def zone_attack_penalty(state, side, cid):
    if state['sides'][side].get('front') != cid:
        return 0
    status = front_debuff(state, side).get('delay') or {}
    return int(status.get('slow') or 0) if status.get('slow_turn', state['turn']) == state['turn'] else 0


def add_atk_buff(character, amount=1, *, source_entity_id=None, source_key='attack_buff', label=None):
    from .modifiers import add_effect
    gained = int(amount or 0)
    if gained:
        add_effect(character, 'attack.buff', 'attack.panel', gained,
                   source_entity_id=source_entity_id or character.get('entity_id'),
                   source_key=source_key, label=label or (CARDS.get(source_key) or {}).get('name') or '攻击增益')
    return gained


def energy_max(character):
    return energy_max_for_attribute(character.get('attribute'))


def round_half_up(value):
    return int(Decimal(value).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def collapsed(character):
    return bool((character.get('flags') or {}).get('collapse'))


def record_delay_turn(state, side):
    team = state['sides'][side]
    if front_debuff(state, other(side)).get('delay') and team.get('delay_counted_turn') != state['turn']:
        team['delay_counted_turn'] = state['turn']
        team['delay_turns'] = int(team.get('delay_turns', 0)) + 1


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


def new_character(cid, *, entity_id=None, side=None):
    from app.modules.card_game.content.duel_v2.tutorial import lookup_character
    c = lookup_character(cid)
    result = CharacterEntity({**deepcopy(c), 'base_attack': c['attack'], 'hp': c['max_hp'],
            'base_max_hp': c['max_hp'], 'shield': 0,
            'harmony': 0, 'energy': 0, 'awakened': False, 'ultimate_turns': 0,
            'growth': 0, 'family': 0, 'gaze': 0, 'dream': 0, 'shape': None, 'down_turns': 0, 'flags': {}}, entity_id=entity_id, side=side)
    from .equipment import import_stats
    import_stats(result)
    return result


def card_instance(state, card_id, side, *, copy=False):
    from app.modules.card_game.content.duel_v2.tutorial import lookup_card
    state['next_instance'] += 1
    c = deepcopy(lookup_card(card_id) or CARDS[card_id])
    c.update(card_id=card_id, instance_id=f"{side}-{state['next_instance']}", copy=copy)
    if copy:
        c['original_character_id'] = c['character_id']
        c['character_id'] = copy_owner_id() or c['character_id']
        c['expires_turn'] = state['sides'][side]['turn_count'] + 1
    return CardEntity(c, entity_id=allocate_entity_id(state), side=side)


def shuffle(state, cards):
    # Store a small integer stream state, rather than non-JSON Random tuples.
    rng = Random(state['rng'])
    rng.shuffle(cards)
    state['rng'] = rng.getrandbits(63)


def finished(state):
    return state['phase'] == 'finished'


def victory(state):
    if finished(state):
        return
    dead = [s for s in SIDES if state['sides'][s]['hp'] <= 0]
    if not dead:
        return
    state['winner'] = 'draw' if len(dead) == 2 else other(dead[0])
    state['phase'] = 'finished'
    state['reason'] = 'player_hp'
    log(state, '双方生命归零，平局。' if len(dead) == 2 else f"{name(state, state['winner'])}获胜。",
        'finish', winner=state['winner'], reason=state.get('reason'))


def upgrade_harmony_residue(state):
    """Legacy snapshots lack evidence that a benched source was full at departure."""
    if state.get('harmony_residue_version') != 1:
        for team in state['sides'].values():
            team['last_front'] = None
        state['harmony_residue_version'] = 1


def clear_harmony_source(state, side, cid):
    """Death expires a character's previous sortie, without consuming personal resources."""
    team = state['sides'][side]
    if team['last_front'] == cid:
        team['last_front'] = None


def move_out(state, side, cid, *, reason=None):
    upgrade_harmony_residue(state)
    team = state['sides'][side]
    if team['front'] != cid:
        return
    h = hero(state, side, cid)
    if h.get('summoned'):
        from .summons import dismiss
        dismiss(state, side, cid)
        return
    from .escalation import harmony_cap
    # Capture eligibility at departure; later bench resource gains cannot create it.
    team['last_front'] = (cid if alive(state, side, cid)
                          and h['harmony'] >= harmony_cap(state) else None)
    if reason == 'replace' and alive(state, side, cid) and not h.get('summoned'):
        from app.modules.card_game.content.duel_v2.registry import fire
        from .context import EffectContext
        fire(EffectContext(state, side, cid, {'side': side}), 'on_replaced', actor_only=True)
    team['front'] = None
    before = {'hp': h['hp'], 'shield': h['shield'], 'front': cid}
    h['shield'] = 0
    from .modifiers import clear
    clear(h, 'leave')
    from app.modules.card_game.content.duel_v2.registry import fire
    from app.modules.card_game.engine.events import GameEvent
    from .context import EffectContext
    fire(EffectContext(state, side, cid, {'side': side}), GameEvent.V2_LEAVE_FRONT,
         actor_only=True, include_down=True)
    log(state, f"{h['name']}返回备战区。", 'move', side=side, source=cid,
        actor=f'{side}:{cid}', target=f'{side}:{cid}',
        before=before, after={'hp': h['hp'], 'shield': h['shield'], 'front': None})


def discard(state, side, card, *, bottom=False):
    from .presentation import public_card
    if card.get('virtual'):
        return
    card = card.get('_physical_card', card)
    team = state['sides'][side]
    card.pop('jingu_mark', None)
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
    # A downed character's effects can kill a character already visited in this
    # pass. Drain those deaths here, before the damage stage can finish.
    while _knockdown_pass(state):
        pass


def _knockdown_pass(state):
    downed = []
    for side in SIDES:
        for cid in team_order(state, side):
            h = hero(state, side, cid)
            if h['hp'] > 0 or h['down_turns'] > 0:
                continue
            if h.get('summoned'):
                from .summons import dismiss
                dismiss(state, side, cid)
                continue
            downed.append((side, cid))
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
            fire(ctx, 'on_knockdown', actor_only=True, include_down=True)
            killer_side, killer_cid = source_parts(killer)
            if killer_side and killer_cid and killer_side != side and alive(state, killer_side, killer_cid):
                fire(EffectContext(state, killer_side, killer_cid, {'side': killer_side}),
                     'on_rout', actor_only=True, downed=cid)
            if (killer_side and killer_cid and killer_cid in state['sides'].get(killer_side, {}).get('characters', {})
                    and hero(state, killer_side, killer_cid).get('summoned')):
                carrier = next((item for item in team_order(state, killer_side)
                                if not hero(state, killer_side, item).get('summoned')), killer_cid)
                fire(EffectContext(state, killer_side, carrier, {'side': killer_side}),
                     'on_summon_kill', summon_id=killer_cid, downed=cid)
            if h.get('awakened') or h.get('ultimate_turns'):
                from .flow import end_ultimate
                end_ultimate(state, side, cid)
            from app.modules.card_game.content.duel_v2.tutorial import lookup_character
            from .equipment import unarmed_max_hp, remove_weapon
            base_max = unarmed_max_hp(h)
            remove_weapon(state, h)
            from .modifiers import clear, remove_source
            if h.get('shape'):
                remove_source(state, h.get('entity_id'), h['shape'])
            clear(h, 'down')
            h.pop('next_damage_resistance', None)
            h.update(down_turns=3, harmony=kept_harmony, energy=kept_energy, shield=0, growth=0,
                     gaze=0, dream=0, shape=None, awakened=False, ultimate_turns=0,
                     max_hp=base_max, flags={})
            log(state, f"{h['name']}倒地，3 次己方回合开始后恢复。", 'down', side=side,
                source=cid, target=f'{side}:{cid}', actor=f'{side}:{cid}',
                before=before, after={'hp': 0, 'shield': 0, 'harmony': kept_harmony,
                                      'energy': kept_energy, 'growth': 0,
                                      'shape_id': None, 'down_turns': 3})
    # Player defeat takes precedence over post-knockdown draws and effects.
    victory(state)
    # Finish every simultaneous knockdown before effects choose a new front.
    for side, cid in downed:
        ctx = EffectContext(state, side, cid, {'side': side, 'downed': cid})
        fire(ctx, 'on_ally_down', downed=cid, include_down=True)
        from .flow import trigger_responses
        trigger_responses(state, side, 'self_downed', target_id=cid)
    return bool(downed)


def damage_immune(state, side, cid):
    h = hero(state, side, cid)
    from .modifiers import evaluate
    return alive(state, side, cid) and bool(evaluate(state, side, h, 'rule.damage_immune', False)['value'])


def damage_limit(state, side, cid):
    """Public per-event damage cap; None means there is no active limit."""
    if not alive(state, side, cid):
        return None
    from .modifiers import evaluate
    value = evaluate(state, side, hero(state, side, cid), 'damage.limit', None)['value']
    return None if value is None else max(0, int(value))


def damage(state, side, cid, amount, *, bypass=False, source=None, kind='damage', settle=True,
           group_id=None, damage_class=None, reason=None, counter_immunity=None):
    if finished(state):
        return 0
    esper = is_esper(cid)
    cid = PLAYER_ID if not esper else cid
    target = hero(state, side, cid) if esper else state['sides'][side]
    if esper and target['hp'] <= 0:
        return 0
    from .modifiers import evaluate
    amount = max(0, int(evaluate(state, side, target, 'damage.amount', amount,
                               query={'kind': kind, 'source': source})['value']))
    src_side, _ = source_parts(source)
    if damage_class == 'attachment' and src_side:
        from .context import EffectContext
        from app.modules.card_game.content.duel_v2.registry import tally
        amount += tally(EffectContext(state, src_side, team_order(state, src_side)[0],
                                      {'side': src_side}), 'attachment_damage_bonus')
    if esper:
        from app.modules.card_game.content.duel_v2.registry import decide
        from .context import EffectContext
        amount = decide(EffectContext(state, side, cid, {'side': side}), 'incoming_damage',
                        amount, kind=kind, target_id=cid, source=source)
        amount = max(0, int(amount or 0))
    klass = damage_class_for(kind, damage_class)
    attr = source_attribute(state, source)
    zone = front_debuff(state, side)
    immune = ((esper and damage_immune(state, side, cid))
              or (counter_immunity == 'support' and kind == 'combat'))
    if immune:
        amount = 0
    if (not immune and kind == 'combat' and attr in ('魂', '相') and zone.get('stain')
            and (state['sides'][side]['front'] == cid or (not esper and not state['sides'][side]['front']))):
        amount += 2
    resistances = target.get('next_damage_resistance') or {}
    if amount > 0:
        from .modifiers import evaluate, consume_applied
        resistance = evaluate(state, side, target, 'damage.resistance', 0,
                              query={'attribute': attr, 'amount': amount, 'kind': kind, 'source': source})
        amount = max(0, amount - int(resistance['value']) - int(resistances.pop(attr, 0)))
        consume_applied(state, 'damage', [e['effect_id'] for e in resistance['contributions']])
    if esper:
        limit = damage_limit(state, side, cid)
        if limit is not None:
            amount = min(amount, limit)
    before = {'hp': target['hp'], 'shield': target['shield']}
    absorbed = 0 if bypass else min(target['shield'], amount)
    target['shield'] -= absorbed
    loss = amount - absorbed
    redirect = None
    redirect_loss = 0
    if esper and loss > 0 and absorbed > 0:
        from app.modules.card_game.content.duel_v2.registry import decide
        from .context import EffectContext
        redirect = decide(EffectContext(state, side, cid, {'side': side}), 'redirect_overflow', None,
                          loss=loss, absorbed=absorbed, target_id=cid)
        if redirect == cid or not is_esper(redirect) or not alive(state, side, redirect):
            redirect = None
        else:
            redirect_loss = loss
            amount = absorbed
            loss = 0
    floor = 0
    if esper:
        from app.modules.card_game.content.duel_v2.registry import decide
        from .context import EffectContext
        if target['hp'] - loss <= 0:
            from .flow import trigger_responses
            trigger_responses(state, side, 'self_lethal', target_id=cid)
        floor = int(decide(EffectContext(state, side, cid, {'side': side}), 'hp_floor', 0,
                           target_id=cid) or 0)
    if not esper and target.get('life_locked'):
        loss = 0
    target['hp'] = max(floor, target['hp'] - loss)
    after = {'hp': target['hp'], 'shield': target['shield']}
    # Public, team-wide history: count each positive genesis hit on the opposing
    # player that reduces life. Flowers and equipment hits are distinct.
    if (kind == 'genesis' and not esper and src_side in SIDES and src_side != side
            and before['hp'] > after['hp']):
        team = state['sides'][src_side]
        team['genesis_player_hits'] = int(team.get('genesis_player_hits', 0)) + 1
    extra = {}
    if group_id:
        extra['group_id'] = group_id
    if counter_immunity == 'support' and kind == 'combat' and amount == 0:
        extra['counter_immunity'] = 'support'
    src_side, src_cid = source_parts(source)
    if src_side:
        extra['source_side'] = src_side
    if esper and source:
        target['last_hit_source'] = source
    text = (f"{name(state, side, cid)}发动援护技，免疫反击。" if extra.get('counter_immunity') else
            (f'{reason}：' if reason else '') + f"{name(state, side, cid)}受到 {amount} 点伤害（护盾吸收 {absorbed}）。")
    log(state, text,
        kind, side=side, source=source, actor=source, target=f'{side}:{cid}',
        amount=amount, before=before, after=after, **extra)
    if klass == 'harmony' and isinstance(source, str) and ':' in source:
        src_side = source.split(':', 1)[0]
        if src_side in SIDES:
            state['sides'][src_side]['harmony_damage'] = True
    if loss:
        from app.modules.card_game.content.duel_v2.registry import fire
        from .context import EffectContext
        if esper:
            fire(EffectContext(state, side, cid, {'side': side}), 'on_damage_taken',
                 actor_only=True, amount=loss, source=source)
            if target.get('summoned'):
                fire(EffectContext(state, side, team_order(state, side)[0], {'side': side}),
                     'on_ally_summon_hurt', summon_id=cid, amount=loss, source=source)
        if isinstance(source, str) and ':' in source:
            src_side, src_cid = source.split(':', 1)
            if src_side in SIDES and src_cid in state['sides'][src_side]['characters']:
                fire(EffectContext(state, src_side, src_cid, {'side': src_side}),
                     'on_damage_dealt', actor_only=True, target_side=side, target_id=cid, amount=loss,
                     damage_kind=kind)
    if redirect_loss and redirect and not finished(state):
        damage(state, side, redirect, redirect_loss, source=source, kind=kind, settle=settle,
               group_id=group_id, damage_class=damage_class, reason='护盾破碎，溢出伤害转移')
        return loss
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


def force_collapse(state, side, cid, *, by=None):
    """Put a living esper straight into 倾陷. Duration matches a full collapse."""
    if finished(state) or not is_esper(cid) or not alive(state, side, cid):
        return
    h = hero(state, side, cid)
    flags = h.setdefault('flags', {})
    if flags.get('collapse'):
        return
    flags['collapse_count'] = 0
    applier = by or other(side)
    flags['collapse'] = {'side': applier, 'until': state['sides'][applier]['turn_count'] + 1}
    log(state, f"{h['name']}进入倾陷。", 'collapse', side=side, source=cid,
        target=f'{side}:{cid}', after={'collapse': True})


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
    limit = int(flags.get('collapse_max') or 5)
    if flags['collapse_count'] >= limit and not flags.get('collapse'):
        flags['collapse_count'] = 0
        applier = by or other(side)
        flags['collapse'] = {'side': applier, 'until': state['sides'][applier]['turn_count'] + 1}
        log(state, f"{h['name']}进入倾陷。", 'collapse', side=side, source=cid,
            target=f'{side}:{cid}', after={'collapse': True})


def heal(state, side, cid, amount, *, reason=None, source=None):
    if finished(state):
        return
    source_side, source_cid = source_parts(source)
    if source_side and source_cid and alive(state, source_side, source_cid):
        from .context import EffectContext
        from app.modules.card_game.content.duel_v2.registry import decide
        amount = decide(EffectContext(state, source_side, source_cid, {'side': source_side}),
                        'outgoing_heal', amount, actor_only=True)
    if is_player(cid):
        team = state['sides'][side]
        if team.get('life_locked'):
            return
        actual = min(amount, 30 - team['hp'])
        if actual <= 0:
            return
        before = {'hp': team['hp'], 'shield': team['shield']}
        team['hp'] += actual
        log(state, (f'{reason}：' if reason else '') + f"{name(state, side)}回复 {actual} 点生命。", 'heal', side=side,
            actor=source or f'{side}:player', target=f'{side}:player', amount=actual,
            before=before, after={'hp': team['hp'], 'shield': team['shield']})
        _notify_healed(state, side, PLAYER_ID, actual, source)
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
        log(state, (f'{reason}：' if reason else '') + f"{h['name']}回复 {actual} 点生命。", 'heal', side=side, source=source or cid,
            actor=source or f'{side}:{cid}', target=f'{side}:{cid}', amount=actual,
            before=before, after={'hp': h['hp'], 'shield': h['shield']})
        _notify_healed(state, side, cid, actual, source)


def _notify_healed(state, side, cid, amount, source):
    from .context import EffectContext
    from app.modules.card_game.content.duel_v2.registry import fire
    owner, _ = source_parts(source)
    owner = owner or side
    fire(EffectContext(state, owner, team_order(state, owner)[0], {'side': owner}),
         'on_ally_healed', target_id=f'{side}:{cid}', amount=amount, source=source)


def shield(state, side, cid, amount, *, reason=None):
    if finished(state) or (not is_player(cid) and not alive(state, side, cid)):
        return
    h = state['sides'][side] if is_player(cid) else hero(state, side, cid)
    before = {'hp': h['hp'], 'shield': h['shield']}
    h['shield'] += max(0, amount)
    log(state, (f'{reason}：' if reason else '') + f"{name(state, side, cid)}获得 {amount} 点护盾。", 'shield', side=side, source=cid,
        actor=f'{side}:{cid}', target=f'{side}:{cid}', amount=amount,
        before=before, after={'hp': h['hp'], 'shield': h['shield']})


def can_gain_energy(state, side, cid):
    if not alive(state, side, cid):
        return False
    h = hero(state, side, cid)
    if h.get('resource_name') or h.get('summoned'):
        return False
    from .context import EffectContext
    from app.modules.card_game.content.duel_v2.registry import decide
    from .modifiers import evaluate
    base = bool(decide(EffectContext(state, side, cid, {'side': side}),
                       'can_gain_energy', True, actor_only=True))
    return bool(evaluate(state, side, h, 'rule.gain_energy', base)['value'])


def effective_card(state, side, card):
    # Resolve hand substitutions without changing the physical card's identity.
    from .context import EffectContext
    from app.modules.card_game.content.duel_v2.registry import decide
    return decide(EffectContext(state, side, team_order(state, side)[0], {'side': side}),
                  'hand_card', card, card=card)


def energy(state, side, cid, amount=1):
    if finished(state) or not can_gain_energy(state, side, cid):
        return
    h = hero(state, side, cid)
    from .escalation import energy_cap
    h['energy'] = min(energy_cap(state, h), h['energy'] + amount)


def add_hand(state, side, card, *, reveal=False, reason=None):
    from .presentation import public_card
    if finished(state):
        return
    team = state['sides'][side]
    if len(team['hand']) >= 10:
        discard(state, side, card)
    else:
        team['hand'].append(card)
        from app.modules.card_game.content.duel_v2.registry import fire
        from .context import EffectContext
        fire(EffectContext(state, side, team_order(state, side)[0], {'side': side}),
             'on_hand_added', card=card)
        if reveal or card.get('copy') or card.get('derived'):
            log(state, (f'{reason}：' if reason else '') + f"{name(state, side)}获得{'临时复制' if card.get('copy') else ''}「{card['name']}」。",
                'gain', side=side, card=public_card(card), actor=f'{side}:player')


def draw(state, side, count=1, *, card_type=None):
    from .presentation import public_card
    team = state['sides'][side]
    for _ in range(count):
        if finished(state):
            break
        index = 0
        if card_type is not None:
            index = next((i for i, card in enumerate(team['deck']) if card.get('type') == card_type), None)
            if index is None:
                break
        if team['deck']:
            card = team['deck'].pop(index)
            add_hand(state, side, card)
            log(state, f"{name(state, side)}抽 1 张牌。", 'draw', side=side,
                actor=f'{side}:player', private_side=side, private_card=public_card(effective_card(state, side, card)))
        else:
            wins = bool(team.get('empty_draw_wins'))
            state.update(phase='finished', winner=side if wins else other(side),
                         reason='empty_draw_win' if wins else 'empty_draw')
            log(state, f"{name(state, side)}牌库已空，" + ('触发效果并获胜。' if wins else '无法抽牌。'), 'finish', side=side,
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
