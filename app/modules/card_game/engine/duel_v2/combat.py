"""Sortie, harmony, simultaneous damage, and timing hooks for character kits."""
from app.modules.card_game.content.duel_v2.registry import decide, fire, genesis_flowers, tally
from app.modules.card_game.engine.events import GameEvent
from .state import (PLAYER_ID, add_collapse, alive, attack_value, collapsed, damage, energy,
                    finished, front_debuff, front_target, hero, is_esper, is_player, knockdowns,
                    log, move_out, name, other, reduce_max_hp, shield, victory, zone_attack_penalty,
                    damage_immune)

from .escalation import harmony_cap

PAIRS = {frozenset(('光', '灵')): '创生', frozenset(('光', '相')): '延滞',
         frozenset(('暗', '咒')): '浊燃', frozenset(('暗', '魂')): '黯星',
         frozenset(('咒', '灵')): '覆纹', frozenset(('魂', '相')): '浸染'}


def harmony_previous(state, side, cid):
    """Current front, or the eligible payer captured when the last front left."""
    team = state['sides'][side]
    previous = team['front'] or (team['last_front'] if state.get('harmony_residue_version') == 1 else None)
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
    if state['active_side'] != side:
        return False
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
    ctx = EffectContext(state, side, cid, {'side': side, 'actor': cid})
    if decide(ctx, 'sortie_from_bench', False, actor_only=True):
        return None
    decided = decide(ctx,
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
    if finished(s) or not alive(s, side, cid):
        return
    team = s['sides'][side]
    team['genesis_turn'] = s['turn']
    from .presentation import next_group_id
    repeats = 1 + tally(c, 'genesis_extra')
    surplus_triggered = False
    for _ in range(repeats):
        if finished(s) or not alive(s, side, cid):
            break
        genesis_target = front_target(s, side)
        delayed = bool(front_debuff(s, genesis_target[0]).get('delay'))
        target_side, target_id = genesis_target
        target = hero(s, target_side, target_id) if is_esper(target_id) else s['sides'][target_side]
        shield_before = target['shield']
        loss = damage(s, *genesis_target, 1 + int(team.get('genesis_damage_bonus') or 0), source=f'{side}:{cid}', kind='genesis',
                      group_id=next_group_id(s))
        damage_dealt = loss > 0 or target['shield'] < shield_before
        if delayed and alive(s, side, cid) and not finished(s):
            surplus_triggered = True
            s['sides'][side]['surplus'] = True
            s['sides'][side]['surplus_ever'] = True
            recipient = team.get('front')
            gained = 0
            if recipient and alive(s, side, recipient):
                before_energy = hero(s, side, recipient)['energy']
                energy(s, side, recipient, 1)
                gained = hero(s, side, recipient)['energy'] - before_energy
            text = f"{c.character['name']}触发盈蓄。"
            if gained:
                text += f"{hero(s, side, recipient)['name']}获得 {gained} 点能量。"
            log(s, text, 'harmony', side=side, source=cid, actor=f'{side}:{cid}',
                target=f'{side}:{recipient or cid}')
        fire(c, 'on_genesis_damage', target=genesis_target, damage_dealt=damage_dealt)
    if surplus_triggered:
        fire(c, GameEvent.V2_SURPLUS, entering=cid)
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


def apply_enter_harmony(c, payer_id, kind, *, as_support=True, pay=True):
    if not kind or (pay and not payer_id):
        return
    s, side, cid, op = c.state, c.side, c.cid, c.operation
    team = s['sides'][side]
    payer = hero(s, side, payer_id) if payer_id else None
    before = payer['harmony'] if payer else 0
    if pay:
        payer['harmony'] -= harmony_cap(s)
        if team.get('last_front') == payer_id:
            team['last_front'] = None
    c.character['flags']['has_harmonized'] = True
    # This once-per-game fact must survive death.
    team['harmonized'][cid] = True
    op['harmony'] = kind
    if as_support:
        op['support'] = True
        if not op.pop('support_notified', False):
            fire(c, 'on_support', actor_id=cid)
    if pay:
        log(s, f"{c.character['name']}入场，消耗{payer['name']}的 {harmony_cap(s)} 点环合值，触发{kind}。", 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{side}:{payer_id}',
            before={'harmony': before}, after={'harmony': payer['harmony']})
    else:
        log(s, f"{c.character['name']}入场，不消耗环合值，触发{kind}。", 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{side}:{payer_id or cid}')
    enemy, target = front_target(s, side)
    if kind == '创生':
        resolve_genesis(c)
    elif kind == '延滞':
        duration = int(decide(c, 'delay_duration', 1) or 1)
        slow = int(decide(c, 'delay_slow', 1) or 0)
        old = front_debuff(s, enemy).get('delay') or {}
        front_debuff(s, enemy)['delay'] = {
            'left': max(int(old.get('left', 0)), duration),
            'tick_side': side, 'slow': slow,
            'slow_turn': old.get('slow_turn', s['turn']) if old else s['turn'],
            'by': side,
        }
        log(s, f"对方战斗区获得延滞（{duration} 回合）。", 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{enemy}:front', present=False)
        from .state import record_delay_turn
        record_delay_turn(s, side)
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
        from .continuous import apply_burn
        apply_burn(s, enemy, by=side, source_cid=cid)
    elif kind == '黯星':
        team['dark_star_count'] = int(team.get('dark_star_count', 0)) + 1
        zone = front_debuff(s, enemy)
        had_burn = bool(zone.get('burn'))
        zone['star'] = {'left': 2, 'by': side}
        log(s, '对方战斗区获得黯星。', 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{enemy}:front', present=False)
        if had_burn:
            add_collapse(s, enemy, target, 1, by=side)
            log(s, f"{c.character['name']}触发失谐。", 'harmony', side=side, source=cid,
                actor=f'{side}:{cid}', target=f'{enemy}:{target}')
    elif kind == '覆纹':
        front_debuff(s, enemy)['weave'] = {'by': side, 'expires_side': s['active_side']}
        log(s, '对方战斗区获得覆纹。', 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{enemy}:front', present=False)
    elif kind == '浸染':
        front_debuff(s, enemy)['stain'] = {'by': side, 'expires_side': s['active_side']}
        log(s, '对方战斗区获得浸染。', 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{enemy}:front', present=False)
    elif kind == '同频':
        team['used']['sync'] = True
        fire(c, 'on_sync', entering=cid)
    if not finished(s) and alive(s, side, cid):
        c.on_shape(GameEvent.V2_HARMONY_RESOLVED, operation=op)


def enter(c, *, as_support=True, resolve_harmony=True):
    s, side, cid = c.state, c.side, c.cid
    if finished(s) or not alive(s, side, cid):
        return
    team = s['sides'][side]
    if team['front'] == cid:
        return
    payer_id, kind = _pending_harmony(c)
    if team['front']:
        move_out(s, side, team['front'], reason='replace')
        if team['front']:
            return
    team['last_front'] = None
    team['front'] = cid
    log(s, f"{c.character['name']}进入战斗区。", 'enter', side=side, source=cid,
        actor=f'{side}:{cid}', target=f'{side}:{cid}')
    if resolve_harmony:
        apply_enter_harmony(c, payer_id, kind, as_support=as_support)


def _zone_slow(state, side, cid):
    return zone_attack_penalty(state, side, cid)


def intercept_enter(c):
    """Swap in as 反击方; only the owner's active turn permits entry harmony."""
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
    team['last_front'] = None
    team['front'] = cid
    log(s, f"{c.character['name']}进入战斗区，作为反击方。", 'enter',
        side=side, source=cid, actor=f'{side}:{cid}', target=f'{side}:{cid}')
    apply_enter_harmony(c, payer_id, kind, as_support=False)


def attack(c, spec, index):
    s, side, cid, op = c.state, c.side, c.cid, c.operation
    enemy, target_id = spec.get('target') or front_target(s, side)
    if is_esper(target_id) and not alive(s, enemy, target_id):
        return
    fire(c, 'before_attack', actor_only=True)
    if index == 0 and is_esper(target_id):
        from .flow import trigger_responses, settle_deferred_card_marks
        if not spec.get('target'):
            trigger_responses(s, enemy, 'faction_front_attacked', target_id=target_id)
            target_id = s['sides'][enemy]['front'] or 'player'
        if is_esper(target_id):
            trigger_responses(s, enemy, 'self_attacked', target_id=target_id)
            trigger_responses(s, enemy, 'ally_attacked', target_id=target_id)
        if finished(s) or not alive(s, side, cid):
            settle_deferred_card_marks(s)
            return
        if not spec.get('target'):
            enemy, target_id = front_target(s, side)
    attacked = s['sides'][side].setdefault('attacked_this_turn', [])
    if c.entity_id not in attacked:
        attacked.append(c.entity_id)
    h = c.character
    target = hero(s, enemy, target_id) if is_esper(target_id) else s['sides'][enemy]
    from .attack_modifiers import attack_result
    from .modifiers import consume_applied, migrate_legacy, entities
    for owner_side, owner in entities(s):
        migrate_legacy(owner, s, owner_side)
    attack_effects = attack_result(c, spec, index, target, enemy, target_id)
    attack_amount = attack_effects['value']
    op['last_attack_amount'] = attack_amount
    low = attack_effects['low_target']
    consume_applied(s, 'attack', [e['effect_id'] for e in attack_effects['contributions']])
    extra_counter = 0
    counter_base = attack_value(target, s, enemy, include_zone=False) if is_esper(target_id) else 0
    if is_esper(target_id):
        counter_base = (target.get('flags') or {}).pop('counter_set_attack', counter_base)
        extra_counter = int((target.get('flags') or {}).pop('counter_bonus', 0) or 0)
    support_immunity = op.get('support') and not spec.get('support_allows_counter')
    no_counter = (spec.get('no_counter') or op.get('no_counter') or support_immunity
                  or (spec.get('no_counter_harmony') and op['harmony'])
                  or (is_esper(target_id) and collapsed(target)))
    if is_esper(target_id) and not no_counter:
        counter = max(0, counter_base + extra_counter - _zone_slow(s, enemy, target_id))
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
    if is_esper(target_id) and not cut_max and not damage_immune(s, enemy, target_id):
        blocked = 0 if spec.get('ignore_shield') else int(target['shield'] or 0)
        # User tabletop rule, 2026-10-08: the front's per-hit damage cap
        # protects that character, without reducing the attack's overkill.
        raw_overflow = max(0, attack_amount - blocked - target['hp'])
        overflow = raw_overflow if penetrate else 0
        overflow = int(decide(c, 'combat_overflow', overflow, attacker_id=cid,
                              defender_id=target_id, defender_side=enemy,
                              attacker_entity_id=h.entity_id, defender_entity_id=target.entity_id,
                              raw_overflow=raw_overflow, spec=spec, index=index) or 0)
        if is_esper(target_id):
            from .context import EffectContext
            overflow = int(decide(EffectContext(s, enemy, target_id, op), 'combat_overflow', overflow,
                                  attacker_id=cid, defender_id=target_id, defender_side=enemy,
                              attacker_entity_id=h.entity_id, defender_entity_id=target.entity_id,
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
    if not spec.get('no_combat_resources') and attacker not in op['participants']:
        op['participants'].append(attacker)
    from .presentation import next_group_id
    group_id = next_group_id(s)
    hit_index = index
    log(s, f"{h['name']}攻击{name(s, enemy, target_id)}：攻击 {attack_amount}，反击 {counter}。",
        'attack', side=side, source=cid, actor=f'{side}:{cid}',
        target=f'{enemy}:{target_id}', amount=attack_amount, attack=attack_amount,
        counter=counter, group_id=group_id, hit_index=hit_index,
        attack_modifiers=[e for e in attack_effects['contributions'] if e['visibility'] == 'public'])
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
                       counter_immunity='support' if support_immunity else None) if is_esper(target_id) else 0
    player_damage = target_loss if is_player(target_id) else 0
    if overflow:
        player_damage += damage(s, enemy, PLAYER_ID, overflow, source=f'{side}:{cid}', kind='penetration',
               settle=False, group_id=group_id)
    if op.get('support') and is_esper(target_id) and alive(s, enemy, target_id):
        add_collapse(s, enemy, target_id, 1, by=side)
    op['last_attack_target'] = [enemy, target_id]
    fire(c, 'after_hits', actor_only=True, include_down=True, target_loss=target_loss,
         self_loss=self_loss, target_id=target_id, enemy=enemy)
    knockdowns(s)
    victory(s)
    from .flow import settle_deferred_card_marks
    if finished(s):
        settle_deferred_card_marks(s)
        return
    fire(c, 'on_ally_attack_resolved', actor_id=cid, player_damage=player_damage)
    if finished(s):
        settle_deferred_card_marks(s)
        return
    for source_side, source_id, target_side, target_cid in weave_targets:
        if not alive(s, source_side, source_id):
            continue
        if is_esper(target_cid) and (not alive(s, target_side, target_cid) or s['sides'][target_side]['front'] != target_cid):
            continue
        damage(s, target_side, target_cid, 2, source=f'{source_side}:{source_id}',
               kind='overlay', group_id=next_group_id(s), damage_class='harmony')
        if finished(s):
            settle_deferred_card_marks(s)
            return
    # Attack-after shape effects occur per hit, after simultaneous death handling.
    for ps, pc, loss in [(side, cid, self_loss), (enemy, target_id, target_loss)]:
        if is_esper(pc) and loss > 0 and alive(s, ps, pc) and not finished(s):
            c.for_character(ps, pc).on_shape(GameEvent.V2_COMBAT_HP_LOST)
    settle_deferred_card_marks(s)


def _operation_target(c):
    raw = (c.operation or {}).get('target_id')
    if not isinstance(raw, str) or raw.count(':') != 1:
        return None
    side, cid = raw.split(':', 1)
    if side in ('a', 'b') and is_esper(cid):
        return side, cid
    return None


def sortie(c, **spec):
    s, side, cid, op = c.state, c.side, c.cid, c.operation
    if finished(s) or not alive(s, side, cid):
        return
    op['sortie_attack_bonus'] = 0
    if s.get('_response_intercept'):
        intercept_enter(c)
        if not finished(s) and alive(s, side, cid):
            fire(c, 'on_sortie_ready', actor_id=cid, response=True)
        return
    chosen = spec.get('target') or _operation_target(c)
    if chosen:
        spec['target'] = chosen
    stay_bench = bool(spec.get('stay_bench') or decide(c, 'sortie_from_bench', False, actor_only=True))
    if stay_bench:
        spec['no_counter'] = True
    else:
        enter(c)
    if not finished(s) and alive(s, side, cid):
        fire(c, 'on_sortie_ready', actor_id=cid, response=False)
    fire(c, 'before_hits', actor_only=True)
    for index in range(spec.get('hits', 1)):
        if finished(s) or not alive(s, side, cid) or (not stay_bench and s['sides'][side]['front'] != cid):
            break
        attack(c, spec, index)
    if finished(s) or not alive(s, side, cid):
        return
    if spec.get('shield_after_shape') and c.character['shape']:
        c.shield(spec['shield_after_shape'])
    if stay_bench or s['sides'][side]['front'] == cid:
        fire(c, 'after_sortie', actor_only=True)
        c.on_shape(GameEvent.V2_SORTIE_FINISHED)
        fire(c, 'ally_sortie')
        if not finished(s) and alive(s, side, cid) and c.character['flags'].pop('return_after', False):
            move_out(s, side, cid)
            c.draw(1)
