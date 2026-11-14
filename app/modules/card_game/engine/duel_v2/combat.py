"""Sortie, harmony, simultaneous damage, and timing hooks for character kits."""
from copy import deepcopy
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
    """Most recent card user / ordinary sortie actor, independent of position."""
    team = state['sides'][side]
    previous = team.get('last_front') if state.get('harmony_residue_version') == 3 else None
    if not previous or previous == cid:
        return None
    if team.get('last_actor_turn') is not None and team['last_actor_turn'] != state['turn']:
        return None
    if (team.get('last_actor_entity_id')
            and hero(state, side, previous).get('entity_id') != team['last_actor_entity_id']):
        return None
    return previous


def harmony_payer(state, side, cid):
    previous = harmony_previous(state, side, cid)
    if not previous or not alive(state, side, previous):
        return None
    payer = hero(state, side, previous)
    if collapsed(payer) or payer['harmony'] < harmony_cap(state):
        return None
    return previous


def harmony_eligible(state, side, cid):
    from .tutorial import tutorial_flags
    if tutorial_flags(state).get('disable_harmony_trigger'):
        return False
    if state['active_side'] != side:
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
    from .harmony import source_id
    payer = source_id(state, side)
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
    from .genesis import add_layers, hit
    add_layers(c, uncapped=not arm_delay)
    hit(c)
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


def notify_support(c, *, exclude_entity_ids=()):
    """Count the actual support once, separately from borrowed callbacks."""
    team = c.state['sides'][c.side]
    if team.get('support_turn') != c.state['turn']:
        team['support_turn'], team['support_count'] = c.state['turn'], 0
    team['support_count'] += 1
    fire(c, 'on_support', actor_id=c.cid, exclude_entity_ids=exclude_entity_ids)


def apply_enter_harmony(c, payer_id, kind, *, as_support=True, pay=True):
    if not kind or (pay and not payer_id):
        return
    s, side, cid, op = c.state, c.side, c.cid, c.operation
    team = s['sides'][side]
    payer = hero(s, side, payer_id) if payer_id else None
    if payer is not None and collapsed(payer):
        return
    before = payer['harmony'] if payer else 0
    if pay:
        payer['harmony'] -= harmony_cap(s)
        if team.get('last_front') == payer_id:
            team['last_front'] = None
            team['last_actor_entity_id'] = None
    c.character['flags']['has_harmonized'] = True
    # This once-per-game fact must survive death.
    team['harmonized'][cid] = True
    op['harmony'] = kind
    if as_support:
        op['support'] = True
        if not op.pop('support_notified', False):
            notify_support(c)
    if pay:
        log(s, f"{c.character['name']}入场，消耗{payer['name']}的 {harmony_cap(s)} 点环合值，触发{kind}。", 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{side}:{payer_id}',
            harmony_kind=kind,
            before={'harmony': before}, after={'harmony': payer['harmony']})
    else:
        log(s, f"{c.character['name']}入场，不消耗环合值，触发{kind}。", 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{side}:{payer_id or cid}', harmony_kind=kind)
    fire(c, GameEvent.V2_HARMONY_TRIGGERED, kind=kind, entering=cid, payer=payer_id)
    resolve_harmony_effect(c, kind)
    if not finished(s) and alive(s, side, cid):
        c.on_shape(GameEvent.V2_HARMONY_RESOLVED, operation=op)


def apply_dark_star(c, *, renewals=None):
    s, side, cid = c.state, c.side, c.cid
    team = s['sides'][side]
    enemy, target = front_target(s, side)
    team['dark_star_count'] = int(team.get('dark_star_count', 0)) + 1
    zone = front_debuff(s, enemy)
    had_burn = bool(zone.get('burn'))
    zone['star'] = {'left': 2, 'by': side, 'source_cid': cid,
                    'renewals': int(decide(c, 'dark_star_renewals', 0) or 0) if renewals is None else int(renewals)}
    fire(c, GameEvent.V2_HARMONY_APPLIED, kind='黯星')
    log(s, '对方战斗区获得黯星。', 'harmony',
        side=side, source=cid, actor=f'{side}:{cid}', target=f'{enemy}:front', present=False)
    if had_burn:
        add_collapse(s, enemy, target, 1, by=side)
        log(s, f"{c.character['name']}触发失谐。", 'harmony', side=side, source=cid,
        actor=f'{side}:{cid}', target=f'{enemy}:{target}')
        fire(c, GameEvent.V2_DISSONANCE, enemy=enemy, target_id=target)


def resolve_harmony_effect(c, kind):
    """Apply an effect without fabricating entry, payment or support events."""
    s, side, cid, op = c.state, c.side, c.cid, c.operation
    team = s['sides'][side]
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
        apply_dark_star(c)
    elif kind == '覆纹':
        front_debuff(s, enemy)['weave'] = {'by': side, 'expires_side': s['active_side']}
        log(s, '对方战斗区获得覆纹。', 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{enemy}:front', present=False)
    elif kind == '浸染':
        front_debuff(s, enemy)['stain'] = {'by': side, 'expires_side': s['active_side'],
                                              'ends_left': max(1, int(decide(c, 'stain_duration', 1) or 1))}
        fire(c, GameEvent.V2_HARMONY_APPLIED, kind='浸染')
        log(s, '对方战斗区获得浸染。', 'harmony',
            side=side, source=cid, actor=f'{side}:{cid}', target=f'{enemy}:front', present=False)
    elif kind == '同频':
        team['used']['sync'] = True
        fire(c, 'on_sync', entering=cid)

def enter(c, *, as_support=True, resolve_harmony=True):
    s, side, cid = c.state, c.side, c.cid
    if finished(s) or not alive(s, side, cid):
        return
    team = s['sides'][side]
    if team['front'] == cid:
        if resolve_harmony:
            payer_id, kind = _pending_harmony(c)
            apply_enter_harmony(c, payer_id, kind, as_support=as_support)
        return
    payer_id, kind = _pending_harmony(c)
    if team['front']:
        move_out(s, side, team['front'], reason='replace')
        if team['front']:
            return
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
    team['front'] = cid
    log(s, f"{c.character['name']}进入战斗区，作为反击方。", 'enter',
        side=side, source=cid, actor=f'{side}:{cid}', target=f'{side}:{cid}')
    apply_enter_harmony(c, payer_id, kind, as_support=False)


def _clear_battle_effects(state):
    from .modifiers import clear, entities
    for _, holder in entities(state):
        clear(holder, 'battle')


def _attack_stopped(c):
    """Recheck the attacker after pre-damage effects, before locking combat."""
    s = c.state
    stopped = finished(s) or not alive(s, c.side, c.cid)
    if not stopped and collapsed(c.character):
        log(s, f"{c.character['name']}进入倾陷，攻击被打断。", 'effect',
            side=c.side, source=c.cid, actor=f'{c.side}:{c.cid}', target=f'{c.side}:{c.cid}')
        stopped = True
    if stopped:
        from .flow import settle_deferred_card_marks
        settle_deferred_card_marks(s)
        _clear_battle_effects(s)
    return stopped


def attack(c, spec, index):
    s, side, cid, op = c.state, c.side, c.cid, c.operation
    if _attack_stopped(c):
        return False
    enemy, target_id = spec.get('target') or front_target(s, side)
    if spec.get('front_only') and not is_esper(target_id):
        return False
    if is_esper(target_id) and not alive(s, enemy, target_id):
        return False
    fire(c, 'before_attack', actor_only=True)
    if _attack_stopped(c):
        return False
    if index == 0 and is_esper(target_id):
        from .flow import trigger_responses, settle_deferred_card_marks
        if s['sides'][enemy].get('front') == target_id:
            trigger_responses(s, enemy, 'front_attacked', target_id=target_id, attacker_id=cid)
            if _attack_stopped(c):
                return False
        if not spec.get('target'):
            trigger_responses(s, enemy, 'faction_front_attacked', target_id=target_id)
            if _attack_stopped(c):
                return False
            target_id = s['sides'][enemy]['front'] or 'player'
        if is_esper(target_id):
            trigger_responses(s, enemy, 'self_attacked', target_id=target_id)
            if _attack_stopped(c):
                return False
            trigger_responses(s, enemy, 'ally_attacked', target_id=target_id)
        if _attack_stopped(c):
            return False
        if not spec.get('target'):
            enemy, target_id = front_target(s, side)
    if spec.get('front_only') and not is_esper(target_id):
        return False
    fire(c.for_character(enemy, target_id if is_esper(target_id) else next(iter(s['sides'][enemy]['characters']))),
         GameEvent.V2_OPPONENT_ATTACK, attacker_side=side, attacker_id=cid, target_id=target_id)
    if _attack_stopped(c):
        return False
    h = c.character
    target = hero(s, enemy, target_id) if is_esper(target_id) else s['sides'][enemy]
    attribute = spec.get('attribute') or h['attribute']
    additional = bool(decide(c, 'attack_is_additional', bool(spec.get('additional_attack') or op.get('additional_attack')),
                             attacker_id=cid, support=bool(op.get('support'))))
    additional = bool(decide(c, 'damage_is_additional', additional, actor_only=True,
                             target_side=enemy, target_id=target_id, damage_kind='combat'))
    fire(c, GameEvent.V2_BEFORE_ATTACK_DAMAGE, attacker_id=cid, target_side=enemy,
         target_id=target_id, attribute=attribute, additional_attack=additional,
         support=bool(op.get('support')), sync_attack=bool(spec.get('sync_attack')))
    if _attack_stopped(c):
        return False
    attacked = s['sides'][side].setdefault('attacked_this_turn', [])
    if c.entity_id not in attacked:
        attacked.append(c.entity_id)
    from .attack_modifiers import attack_result
    from .modifiers import consume_applied, migrate_legacy, entities
    for owner_side, owner in entities(s):
        migrate_legacy(owner, s, owner_side)
    spec['additional_damage_bonus'] = int(decide(c, 'additional_attack_damage_bonus', 0) or 0) if additional else 0
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
    counter_additional = False
    if is_esper(target_id) and not no_counter:
        counter_context = c.for_character(enemy, target_id)
        counter_additional = bool(decide(counter_context, 'damage_is_additional', False, actor_only=True,
                                         target_side=side, target_id=cid, damage_kind='combat'))
        if counter_additional:
            counter = max(0, counter + int(decide(counter_context, 'additional_attack_damage_bonus', 0,
                                                 target_side=side, target_id=cid) or 0))
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
    if attribute in ('灵', '咒') and front_debuff(s, enemy).get('weave'):
        weave_targets.append((side, cid, enemy, target_id, attribute))
    if is_esper(target_id) and not no_counter and target['attribute'] in ('灵', '咒') and front_debuff(s, side).get('weave'):
        weave_targets.append((enemy, target_id, side, cid, target['attribute']))
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
        attribute=attribute, additional_attack=additional,
        attack_modifiers=[e for e in attack_effects['contributions'] if e['visibility'] == 'public'])
    if cut_max:
        reduce_max_hp(s, enemy, target_id, attack_amount, source=f'{side}:{cid}',
                      settle=False, group_id=group_id)
        target_loss = 0
    else:
        target_loss = damage(s, enemy, target_id, attack_amount,
                             bypass=bool(spec.get('ignore_shield')), source=f'{side}:{cid}',
                             kind='combat', settle=False, group_id=group_id,
                             attribute=attribute, additional_attack=additional, _additional_bonus_applied=True)
    self_loss = damage(s, side, cid, counter, source=f'{enemy}:{target_id}',
                       kind='combat', settle=False, group_id=group_id,
                       counter_immunity='support' if support_immunity else None,
                       additional_attack=counter_additional, _additional_bonus_applied=True) if is_esper(target_id) else 0
    player_damage = target_loss if is_player(target_id) else 0
    if overflow:
        player_damage += damage(s, enemy, PLAYER_ID, overflow, source=f'{side}:{cid}', kind='penetration',
               settle=False, group_id=group_id, attribute=attribute,
               additional_attack=additional, _additional_bonus_applied=True)
    if (op.get('support') and is_esper(target_id) and s['sides'][enemy]['front'] == target_id
            and alive(s, enemy, target_id)):
        add_collapse(s, enemy, target_id, 1, by=side)
    op['last_attack_target'] = [enemy, target_id]
    fire(c, 'after_hits', actor_only=True, include_down=True, target_loss=target_loss,
         self_loss=self_loss, target_id=target_id, enemy=enemy)
    knockdowns(s)
    victory(s)
    from .flow import settle_deferred_card_marks
    if finished(s):
        settle_deferred_card_marks(s)
        _clear_battle_effects(s)
        return True
    fire(c, 'on_ally_attack_resolved', actor_id=cid, player_damage=player_damage)
    if finished(s):
        settle_deferred_card_marks(s)
        _clear_battle_effects(s)
        return True
    for source_side, source_id, target_side, target_cid, attack_attribute in weave_targets:
        if not alive(s, source_side, source_id):
            continue
        if is_esper(target_cid) and (not alive(s, target_side, target_cid) or s['sides'][target_side]['front'] != target_cid):
            continue
        from .context import EffectContext
        source_context = EffectContext(s, source_side, source_id, {'side': source_side})
        amount = int(decide(source_context, 'weave_damage', 2) or 0)
        damage(s, target_side, target_cid, amount, source=f'{source_side}:{source_id}',
               kind='overlay', group_id=next_group_id(s), damage_class='harmony',
               attribute=attack_attribute, additional_attack=True)
        if finished(s):
            settle_deferred_card_marks(s)
            _clear_battle_effects(s)
            return True
    if is_esper(target_id) and not no_counter and alive(s, enemy, target_id):
        fire(c.for_character(enemy, target_id), GameEvent.V2_COUNTER_RESOLVED,
             actor_only=True, target=(side, cid))
    # Attack-after shape effects occur per hit, after simultaneous death handling.
    for ps, pc, loss in [(side, cid, self_loss), (enemy, target_id, target_loss)]:
        if is_esper(pc) and loss > 0 and alive(s, ps, pc) and not finished(s):
            c.for_character(ps, pc).on_shape(GameEvent.V2_COMBAT_HP_LOST)
    if op.get('support') and not op.get('support_resolved'):
        op['support_resolved'] = True
        fire(c, GameEvent.V2_SUPPORT_RESOLVED, actor_only=True, actor_id=cid)
    settle_deferred_card_marks(s)
    _clear_battle_effects(s)
    return True


def _operation_target(c):
    raw = (c.operation or {}).get('target_id')
    if not isinstance(raw, str) or raw.count(':') != 1:
        return None
    side, cid = raw.split(':', 1)
    if side in ('a', 'b') and is_esper(cid):
        return side, cid
    return None


def _notify_opponent_sortie_finished(c):
    if finished(c.state) or not alive(c.state, c.side, c.cid):
        return
    enemy = other(c.side)
    carrier = next(iter(c.state['sides'][enemy]['characters']))
    fire(c.for_character(enemy, carrier), GameEvent.V2_OPPONENT_SORTIE_FINISHED,
         attacker_side=c.side, attacker_id=c.cid)


def sortie(c, **spec):
    s, side, cid, op = c.state, c.side, c.cid, c.operation
    if _attack_stopped(c):
        return
    op.setdefault('sortie_attack_bonus', 0)
    if s.get('_response_intercept'):
        intercept_enter(c)
        if not finished(s) and alive(s, side, cid):
            fire(c, 'on_sortie_ready', actor_id=cid, response=True)
        return
    chosen = spec.get('target') or _operation_target(c)
    if chosen:
        spec['target'] = chosen
    stay_bench = bool(spec.get('stay_bench') or decide(c, 'sortie_from_bench', False, actor_only=True))
    op['remote_sortie'] = stay_bench
    if stay_bench:
        spec['no_counter'] = True
    else:
        enter(c)
    if _attack_stopped(c):
        return
    if not finished(s) and alive(s, side, cid):
        fire(c, 'on_sortie_ready', actor_id=cid, response=False)
    if _attack_stopped(c):
        return
    if s['phase'] == 'choice':
        op['pending_sortie'] = dict(spec=deepcopy(spec), stay_bench=stay_bench)
        return
    _finish_sortie(c, spec, stay_bench)


def resume_sortie(c):
    """Continue a serialized sortie without re-entering or repeating ready hooks."""
    pending = c.operation.pop('pending_sortie', None)
    if pending is not None:
        _finish_sortie(c, pending['spec'], pending['stay_bench'])


def _finish_sortie(c, spec, stay_bench):
    s, side, cid = c.state, c.side, c.cid
    if _attack_stopped(c):
        return
    from .genesis import on_sortie
    on_sortie(c)
    if _attack_stopped(c):
        return
    fire(c, 'before_hits', actor_only=True)
    attacked = False
    for index in range(spec.get('hits', 1)):
        if finished(s) or not alive(s, side, cid) or (not stay_bench and s['sides'][side]['front'] != cid):
            break
        if not attack(c, spec, index):
            break
        attacked = True
    if not attacked or finished(s) or not alive(s, side, cid):
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
    _notify_opponent_sortie_finished(c)


def additional_attack(c, attribute, *, target=None, sync_attack=False, support_actor=None, front_only=False):
    """A one-sided extra hit; preserve the enclosing card/response operation."""
    from .context import EffectContext
    if _attack_stopped(c):
        return
    if front_only and not c.state['sides'][other(c.side)]['front']:
        return
    # Borrowed identity affects only that character's own support callbacks.
    # Other observers still see the actual attacker, exactly once.
    borrowed = None
    if sync_attack and support_actor:
        owner_id = support_actor.get('character_id')
        owner = c.state['sides'][c.side]['characters'].get(owner_id)
        if (owner and alive(c.state, c.side, owner_id)
                and owner.entity_id == support_actor.get('entity_id')
                and owner.entity_id != c.entity_id):
            borrowed = owner
    parent = c.state.get('operation')
    op = dict(side=c.side, actor=c.cid, card=None, target_id=None, participants=[],
              harmony=None, first_bonus=0, redeem_bonus=0, sortie_attack_bonus=0)
    if sync_attack:
        op['support'] = True
    c.state['operation'] = op
    try:
        context = EffectContext(c.state, c.side, c.cid, op)
        if sync_attack:
            notify_support(context, exclude_entity_ids=(borrowed.entity_id,) if borrowed else ())
        attacked = attack(context,
               dict(attribute=attribute, target=target, no_counter=True,
                    no_combat_resources=True, additional_attack=True, sync_attack=sync_attack,
                    front_only=front_only), 0)
        if (attacked and borrowed and alive(c.state, c.side, borrowed['id'])
                and hero(c.state, c.side, borrowed['id']).entity_id == borrowed.entity_id):
            fire(context.for_character(c.side, borrowed['id']), 'on_support',
                 actor_only=True, actor_id=borrowed['id'])
            fire(context.for_character(c.side, borrowed['id']), GameEvent.V2_SUPPORT_RESOLVED,
                 actor_only=True, actor_id=borrowed['id'])
        if attacked:
            _notify_opponent_sortie_finished(context)
    finally:
        c.state['operation'] = parent
