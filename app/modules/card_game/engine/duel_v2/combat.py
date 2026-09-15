"""Sortie, harmony, simultaneous damage, and timing hooks for character kits."""
from app.modules.card_game.content.duel_v2.registry import decide, fire, genesis_flowers, tally
from app.modules.card_game.engine.events import GameEvent
from .state import (PLAYER_ID, add_collapse, alive, attack_value, collapsed, damage, energy,
                    finished, front_debuff, front_target, hero, is_esper, is_player, knockdowns,
                    log, move_out, name, other, reduce_max_hp, shield, victory)

from .escalation import harmony_cap

PAIRS = {frozenset(('光', '灵')): '创生', frozenset(('光', '相')): '延滞',
         frozenset(('暗', '咒')): '浊燃', frozenset(('暗', '魂')): '黯星',
         frozenset(('咒', '灵')): '覆纹', frozenset(('魂', '相')): '浸染'}


def harmony_previous(state, side, cid):
    """上一名出战角色：当前前排，或战斗区已空时的最近离场角色。不要求此刻仍占着战斗区。"""
    team = state['sides'][side]
    previous = team['front'] or team['last_front']
    if not previous or previous == cid:
        return None
    return previous


def harmony_payer(state, side, cid):
    previous = harmony_previous(state, side, cid)
    if not previous or not alive(state, side, previous):
        return None
    if hero(state, side, previous)['harmony'] < harmony_cap(state):
        return None
    return previous


def harmony_eligible(state, side, cid):
    team = state['sides'][side]
    if team['front'] == cid:
        return False
    return harmony_payer(state, side, cid) is not None


def harmony_kind(state, side, cid):
    if not harmony_eligible(state, side, cid):
        return None
    previous = harmony_previous(state, side, cid)
    kind = PAIRS.get(frozenset((hero(state, side, previous)['attribute'], hero(state, side, cid)['attribute'])))
    return kind


def resolved_harmony_kind(state, side, cid):
    if not harmony_eligible(state, side, cid):
        return None
    kind = harmony_kind(state, side, cid)
    from .context import EffectContext
    decided = decide(EffectContext(state, side, cid, {'side': side, 'actor': cid}),
                     'harmony_kind', kind, previous=harmony_previous(state, side, cid))
    if decided is False:
        return None
    return decided


def harmony_marks(state, side):
    payer = None
    ready = {}
    for cid in state['sides'][side]['characters']:
        kind = resolved_harmony_kind(state, side, cid)
        if not kind:
            continue
        ready[cid] = kind
        payer = payer or harmony_payer(state, side, cid)
    return {'payer': payer, 'ready': ready, 'available': bool(ready)}


def _pending_harmony(c):
    s, side, cid = c.state, c.side, c.cid
    payer_id = harmony_payer(s, side, cid)
    kind = harmony_kind(s, side, cid)
    if harmony_eligible(s, side, cid):
        kind = decide(c, 'harmony_kind', kind, previous=harmony_previous(s, side, cid))
        if kind is False:
            kind = None
    if not kind or not payer_id:
        return None, None
    return payer_id, kind


def resolve_genesis(c, *, arm_delay=True):
    s, side, cid = c.state, c.side, c.cid
    team = s['sides'][side]
    from .presentation import next_group_id
    repeats = 1 + tally(c, 'genesis_extra')
    for _ in range(repeats):
        if finished(s) or not alive(s, side, cid):
            break
        genesis_target = front_target(s, side)
        delayed = bool(is_esper(genesis_target[1]) and front_debuff(s, genesis_target[0]).get('delay'))
        damage(s, *genesis_target, 1, source=f'{side}:{cid}', kind='genesis',
               group_id=next_group_id(s))
        if delayed and alive(s, side, cid):
            s['sides'][side]['surplus'] = True
            energy(s, side, cid, 1)
            log(s, f"{c.character['name']}触发盈蓄，获得 1 点能量。", 'harmony',
                side=side, source=cid, actor=f'{side}:{cid}', target=f'{side}:{cid}')
        fire(c, 'on_genesis_damage', target=genesis_target)
    flowers = genesis_flowers(c) + int(team['extra_flower'])
    team['extra_flower'] = False
    for _ in range(flowers):
        if finished(s) or not alive(s, side, cid):
            break
        target = front_target(s, side)
        flower_group = next_group_id(s)
        damage(s, *target, 1, source=f'{side}:{cid}', kind='flower', group_id=flower_group)
        fire(c, 'on_flower', target=target, entering=cid)
    if arm_delay:
        fire(c, 'after_genesis', entering=cid)


def apply_enter_harmony(c, payer_id, kind, *, as_support=True):
    if not kind or not payer_id:
        return
    s, side, cid, op = c.state, c.side, c.cid, c.operation
    team = s['sides'][side]
    payer = hero(s, side, payer_id)
    before = payer['harmony']
    payer['harmony'] -= harmony_cap(s)
    c.character['flags']['has_harmonized'] = True
    # This once-per-game fact must survive death.
    team['harmonized'][cid] = True
    op['harmony'] = kind
    if as_support:
        op['support'] = True
    log(s, f"{c.character['name']}入场，消耗{payer['name']}的 {harmony_cap(s)} 点环合值，触发{kind}。", 'harmony',
        side=side, source=cid, actor=f'{side}:{cid}', target=f'{side}:{payer_id}',
        before={'harmony': before}, after={'harmony': payer['harmony']})
    enemy, target = front_target(s, side)
    if kind == '创生':
        resolve_genesis(c)
    elif kind == '延滞':
        duration = int(decide(c, 'delay_duration', 1) or 1)
        slow = int(decide(c, 'delay_slow', 1) or 0)
        front_debuff(s, enemy)['delay'] = {
            'left': duration,
            'expires_side': s['active_side'] if duration == 1 else None,
            'slow': slow,
            'by': side,
        }
        log(s, f"对方战斗区获得延滞（{duration} 回合）。", 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{enemy}:front')
        hits = tally(c, 'delay_hits') + int(team.get('extra_delay'))
        team['extra_delay'] = False
        for _ in range(hits):
            if finished(s):
                break
            delay_target = front_target(s, side)
            from .presentation import next_group_id
            damage(s, *delay_target, 1, source=f'{side}:{cid}', kind='delay',
                   group_id=next_group_id(s))
            fire(c, 'on_delay_hit', target=delay_target)
    elif kind == '浊燃':
        zone = front_debuff(s, enemy)
        had_star = bool(zone.get('star'))
        zone['burn'] = {'left': 2, 'by': side}
        log(s, '对方战斗区获得浊燃。', 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{enemy}:front')
        fire(c, 'on_burn_applied', enemy=enemy, target_id=target)
        if had_star:
            add_collapse(s, enemy, target, 1, by=side)
            log(s, f"{c.character['name']}触发失谐。", 'harmony', side=side, source=cid,
                actor=f'{side}:{cid}', target=f'{enemy}:{target}')
    elif kind == '黯星':
        zone = front_debuff(s, enemy)
        had_burn = bool(zone.get('burn'))
        zone['star'] = {'left': 2, 'by': side}
        log(s, '对方战斗区获得黯星。', 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{enemy}:front')
        if had_burn:
            add_collapse(s, enemy, target, 1, by=side)
            log(s, f"{c.character['name']}触发失谐。", 'harmony', side=side, source=cid,
                actor=f'{side}:{cid}', target=f'{enemy}:{target}')
    elif kind == '覆纹':
        front_debuff(s, enemy)['weave'] = {'by': side, 'expires_side': s['active_side']}
        log(s, '对方战斗区获得覆纹。', 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{enemy}:front')
    elif kind == '浸染':
        front_debuff(s, enemy)['stain'] = {'by': side, 'expires_side': s['active_side']}
        log(s, '对方战斗区获得浸染。', 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{enemy}:front')
    elif kind == '同频':
        team['used']['sync'] = True
        fire(c, 'on_sync', entering=cid)
    if not finished(s) and alive(s, side, cid):
        c.on_shape(GameEvent.V2_HARMONY_RESOLVED, operation=op)


def enter(c):
    s, side, cid = c.state, c.side, c.cid
    team = s['sides'][side]
    if team['front'] == cid:
        return
    payer_id, kind = _pending_harmony(c)
    if team['front']:
        move_out(s, side, team['front'])
        if team['front']:
            return
    team['front'] = cid
    log(s, f"{c.character['name']}进入战斗区。", 'enter', side=side, source=cid,
        actor=f'{side}:{cid}', target=f'{side}:{cid}')
    apply_enter_harmony(c, payer_id, kind, as_support=True)


def _zone_slow(state, side, cid):
    if state['sides'][side].get('front') != cid:
        return 0
    return int((front_debuff(state, side).get('delay') or {}).get('slow') or 0)


def intercept_enter(c):
    """Swap in as 反击方. May trigger 环合, but the current attack is not 援护技."""
    s, side, cid = c.state, c.side, c.cid
    if finished(s) or not alive(s, side, cid):
        return
    team = s['sides'][side]
    if team['front'] == cid:
        return
    payer_id, kind = _pending_harmony(c)
    if team['front']:
        move_out(s, side, team['front'])
        if team['front']:
            return
    team['front'] = cid
    log(s, f"{c.character['name']}进入战斗区，作为反击方。", 'enter',
        side=side, source=cid, actor=f'{side}:{cid}', target=f'{side}:{cid}')
    apply_enter_harmony(c, payer_id, kind, as_support=False)


def attack(c, spec, index):
    s, side, cid, op = c.state, c.side, c.cid, c.operation
    enemy, target_id = front_target(s, side)
    if index == 0 and is_esper(target_id):
        from .flow import trigger_responses
        trigger_responses(s, enemy, 'self_attacked', target_id=target_id)
        trigger_responses(s, enemy, 'ally_attacked', target_id=target_id)
        if finished(s) or not alive(s, side, cid):
            return
        enemy, target_id = front_target(s, side)
    h = c.character
    target = hero(s, enemy, target_id) if is_esper(target_id) else s['sides'][enemy]
    bonus = spec.get('bonus', 0)
    if is_player(target_id) and h['awakened'] and 'awake_player_bonus' in spec:
        bonus = spec['awake_player_bonus']
    low = bool(is_esper(target_id) and target['hp'] <= spec.get('low_hp_max', 3))
    if low:
        bonus += spec.get('low_hp_bonus', 0)
    bonus += op.get('redeem_bonus', 0)
    bonus += int(op.get('card_bonus') or 0)
    if index == 0:
        bonus += op['first_bonus']
    bonus += h['flags'].pop('next_attack', 0)
    if spec.get('harmony_bonus') and op.get('harmony'):
        bonus += spec['harmony_bonus']
    bonus += tally(c, 'attack_bonus', attacker_id=cid, target=target, target_id=target_id,
                   spec=spec, index=index)
    attack_amount = max(0, attack_value(h) + bonus - _zone_slow(s, side, cid))
    attack_amount = max(0, int(decide(c, 'outgoing_attack', attack_amount, attacker_id=cid,
                                      target=target, target_id=target_id, spec=spec, index=index) or 0))
    extra_counter = 0
    if is_esper(target_id):
        extra_counter = int((target.get('flags') or {}).pop('counter_bonus', 0) or 0)
    no_counter = (spec.get('no_counter') or op.get('no_counter') or op.get('support')
                  or (spec.get('no_counter_harmony') and op['harmony'])
                  or (is_esper(target_id) and collapsed(target)))
    if is_esper(target_id) and not no_counter:
        from .context import EffectContext
        counter = max(0, attack_value(target) + extra_counter - _zone_slow(s, enemy, target_id))
        counter = max(0, int(decide(EffectContext(s, enemy, target_id, op), 'outgoing_attack',
                                    counter, attacker_id=target_id, target=h, target_id=cid,
                                    spec=spec, index=index) or 0))
    else:
        counter = 0
    bypass = bool(spec.get('bypass'))
    bypass = bool(decide(c, 'attack_bypass', bypass, attacker_id=cid, target=target,
                         target_id=target_id, spec=spec))
    cut_max = bool(bypass and is_esper(target_id))
    penetrate = (spec.get('penetrate_harmony') and op['harmony']) or (spec.get('penetrate_low_hp') and low)
    if spec.get('penetrate_brand') and is_esper(target_id) and (target.get('flags') or {}).get('brand', 0) >= spec['penetrate_brand']:
        penetrate = True
    overflow = 0
    raw_overflow = 0
    if is_esper(target_id) and not cut_max:
        blocked = 0 if spec.get('ignore_shield') else int(target['shield'] or 0)
        raw_overflow = max(0, attack_amount - blocked - target['hp'])
        overflow = raw_overflow if penetrate else 0
        overflow = int(decide(c, 'combat_overflow', overflow, attacker_id=cid,
                              defender_id=target_id, defender_side=enemy,
                              raw_overflow=raw_overflow, spec=spec, index=index) or 0)
        if is_esper(target_id):
            from .context import EffectContext
            overflow = int(decide(EffectContext(s, enemy, target_id, op), 'combat_overflow', overflow,
                                  attacker_id=cid, defender_id=target_id, defender_side=enemy,
                                  raw_overflow=raw_overflow, spec=spec, index=index) or 0)
        from .tutorial import tutorial_flags
        if tutorial_flags(s).get('disable_overflow'):
            overflow = 0
    weave_targets = []
    if h['attribute'] in ('灵', '咒') and front_debuff(s, enemy).get('weave'):
        weave_targets.append((side, cid, enemy, target_id))
    if is_esper(target_id) and not no_counter and target['attribute'] in ('灵', '咒') and front_debuff(s, side).get('weave'):
        weave_targets.append((enemy, target_id, side, cid))
    # Counters deal simultaneous damage but do not accumulate combat resources.
    attacker = [side, cid]
    if attacker not in op['participants']:
        op['participants'].append(attacker)
    from .presentation import next_group_id
    group_id = next_group_id(s)
    hit_index = index
    log(s, f"{h['name']}攻击{name(s, enemy, target_id)}：攻击 {attack_amount}，反击 {counter}。",
        'attack', side=side, source=cid, actor=f'{side}:{cid}',
        target=f'{enemy}:{target_id}', amount=attack_amount, attack=attack_amount,
        counter=counter, group_id=group_id, hit_index=hit_index)
    if cut_max:
        reduce_max_hp(s, enemy, target_id, attack_amount, source=f'{side}:{cid}',
                      settle=False, group_id=group_id)
        target_loss = 0
    else:
        target_loss = damage(s, enemy, target_id, attack_amount,
                             bypass=bool(spec.get('ignore_shield')), source=f'{side}:{cid}',
                             kind='combat', settle=False, group_id=group_id)
    self_loss = damage(s, side, cid, counter, source=f'{enemy}:{target_id}',
                       kind='combat', settle=False, group_id=group_id,
                       counter_immunity='support' if op.get('support') else None) if is_esper(target_id) else 0
    if overflow:
        damage(s, enemy, PLAYER_ID, overflow, source=f'{side}:{cid}', kind='penetration',
               settle=False, group_id=group_id)
    if op.get('support') and is_esper(target_id) and alive(s, enemy, target_id):
        add_collapse(s, enemy, target_id, 1, by=side)
    fire(c, 'after_hits', actor_only=True, include_down=True, target_loss=target_loss,
         self_loss=self_loss, target_id=target_id, enemy=enemy)
    knockdowns(s)
    victory(s)
    if finished(s):
        return
    for source_side, source_id, target_side, target_cid in weave_targets:
        if not alive(s, source_side, source_id):
            continue
        if is_esper(target_cid) and (not alive(s, target_side, target_cid) or s['sides'][target_side]['front'] != target_cid):
            continue
        damage(s, target_side, target_cid, 2, source=f'{source_side}:{source_id}',
               kind='overlay', group_id=next_group_id(s), damage_class='harmony')
        if finished(s):
            return
    # Attack-after shape effects occur per hit, after simultaneous death handling.
    for ps, pc, loss in [(side, cid, self_loss), (enemy, target_id, target_loss)]:
        if is_esper(pc) and loss > 0 and alive(s, ps, pc) and not finished(s):
            c.for_character(ps, pc).on_shape(GameEvent.V2_COMBAT_HP_LOST)


def sortie(c, **spec):
    s, side, cid, op = c.state, c.side, c.cid, c.operation
    if finished(s) or not alive(s, side, cid):
        return
    if s.get('_response_lock'):
        intercept_enter(c)
        return
    enter(c)
    fire(c, 'before_hits', actor_only=True)
    for index in range(spec.get('hits', 1)):
        if finished(s) or not alive(s, side, cid) or s['sides'][side]['front'] != cid:
            break
        attack(c, spec, index)
    if finished(s) or not alive(s, side, cid):
        return
    if spec.get('shield_after_shape') and c.character['shape']:
        c.shield(spec['shield_after_shape'])
    if s['sides'][side]['front'] == cid:
        fire(c, 'after_sortie', actor_only=True)
        c.on_shape(GameEvent.V2_SORTIE_FINISHED)
        fire(c, 'ally_sortie')
        if not finished(s) and alive(s, side, cid) and c.character['flags'].pop('return_after', False):
            move_out(s, side, cid)
            c.draw(1)
