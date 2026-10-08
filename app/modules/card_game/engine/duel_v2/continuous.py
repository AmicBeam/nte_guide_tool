"""Continuous damage attached to a battle zone.

浊燃、噩梦、蚀心、鸩火 and 诛恶护持 share this clock. Damage is dealt on the
applier's turn end and split across living opponent espers. The same character
receives one merged hit. Stack halving is a countdown, not a turn-start trigger.
"""
from random import Random

from app.modules.card_game.content.duel_v2.registry import fire
from .context import EffectContext
from .state import (SIDES, alive, damage, finished, front_debuff, hero, knockdowns,
                    log, other, team_order, victory)


DOTS = {
    'nightmare': {'name': '噩梦', 'max_stacks': 10, 'mode': 'stacks', 'halve': True},
    'etch': {'name': '蚀心', 'max_stacks': 10, 'mode': 'every5'},
    'venom': {'name': '鸩火', 'max_stacks': 10, 'mode': 'every5'},
    'guard': {'name': '诛恶护持', 'max_stacks': 1, 'mode': 'single', 'unique': True,
              'ignore_shield': True},
}


def living_espers(state, side):
    found = []
    for cid in team_order(state, side):
        h = hero(state, side, cid)
        if h.get('summoned') or h.get('hp', 0) <= 0:
            continue
        found.append(cid)
    return found


def _zone_dots(state, side):
    return front_debuff(state, side).setdefault('dots', {})


def kind_count(state, by):
    total = 0
    for side in SIDES:
        zone = front_debuff(state, side)
        burn = zone.get('burn') or {}
        if burn.get('by') == by:
            total += 1
        for dot in (zone.get('dots') or {}).values():
            if dot.get('by') == by and int(dot.get('stacks') or 0) > 0:
                total += 1
    return total


def total_stacks(state, by):
    total = 0
    for side in SIDES:
        zone = front_debuff(state, side)
        burn = zone.get('burn') or {}
        if burn.get('by') == by:
            total += int(burn.get('stacks') or 1)
        for dot in (zone.get('dots') or {}).values():
            if dot.get('by') == by:
                total += int(dot.get('stacks') or 0)
    return total


def burn_present(state, by=None):
    for side in SIDES:
        burn = (front_debuff(state, side).get('burn') or {})
        if burn and (by is None or burn.get('by') == by):
            return True
    return False


def _boost(state, target_side, by, amount):
    """早雾：已有浊燃时，每种持续伤害使本次伤害 +25%，各自向下取整。"""
    if amount <= 0 or 'zaowu' not in state['sides'][by]['characters'] or not alive(state, by, 'zaowu'):
        return amount
    zone = front_debuff(state, target_side)
    if not zone.get('burn'):
        return amount
    kinds = 1 + sum(1 for dot in (zone.get('dots') or {}).values() if int(dot.get('stacks') or 0) > 0)
    return amount + (amount // 4) * kinds


def _source_context(state, source):
    if not source or ':' not in source:
        return None
    side, cid = source.split(':', 1)
    if side not in SIDES or cid not in state['sides'][side]['characters']:
        return None
    return EffectContext(state, side, cid, {'side': side})


def distribute(state, target_side, amount, *, source, by, ignore_shield=False, reason=None,
               damage_class='esper', after_hit=None, continuous=True):
    if finished(state):
        return []
    amount = max(0, int(amount))
    if continuous:
        amount = _boost(state, target_side, by, amount)
    targets = living_espers(state, target_side)
    if amount <= 0 or not targets:
        return []
    rng = Random(state['rng'])
    buckets = {cid: 0 for cid in targets}
    for _ in range(amount):
        buckets[rng.choice(targets)] += 1
    state['rng'] = rng.getrandbits(63)
    from .presentation import next_group_id
    group = next_group_id(state)
    first_seq = state['event_seq']
    before = set(living_espers(state, target_side))
    for cid, dealt in buckets.items():
        if not dealt:
            continue
        damage(state, target_side, cid, dealt, bypass=ignore_shield, source=source,
               kind='damage', settle=False, group_id=group, damage_class=damage_class, reason=reason)
        if after_hit and not finished(state):
            after_hit(cid, dealt)
        if finished(state):
            return []
    # Trigger histories (for example 残虹 +1 attack) must not split the simultaneous
    # distribution into separate animation batches. Preserve explicit nested groups.
    for event in state['events']:
        if event['seq'] > first_seq:
            event.setdefault('group_id', group)
    knockdowns(state)
    victory(state)
    if finished(state):
        return [cid for cid in before if cid not in living_espers(state, target_side)]
    return [cid for cid in before if cid not in living_espers(state, target_side)]


def deal_one(state, target_side, amount, *, source, by, ignore_shield=False, reason=None):
    if finished(state):
        return []
    amount = _boost(state, target_side, by, max(0, int(amount)))
    targets = living_espers(state, target_side)
    if amount <= 0 or not targets:
        return []
    rng = Random(state['rng'])
    cid = rng.choice(targets)
    state['rng'] = rng.getrandbits(63)
    from .presentation import next_group_id
    before = set(living_espers(state, target_side))
    damage(state, target_side, cid, amount, bypass=ignore_shield, source=source, kind='damage',
           settle=True, group_id=next_group_id(state), reason=reason)
    return [item for item in before if item not in living_espers(state, target_side)]


def _notify_applied(state, by, source_cid, kind):
    if finished(state) or source_cid not in state['sides'][by]['characters']:
        return
    fire(EffectContext(state, by, source_cid, {'side': by}), 'on_continuous_applied',
         kind=kind, source_id=source_cid)


def apply_dot(state, target_side, kind, *, by, source_cid, stacks=1, duration=None):
    spec = DOTS[kind]
    dots = _zone_dots(state, target_side)
    current = dots.get(kind)
    stacks = max(0, int(stacks))
    if current and current.get('by') == by and spec.get('unique'):
        if duration is not None:
            current['left'] = int(duration)
        current['source'] = f'{by}:{source_cid}'
        log(state, f"对方战斗区的{spec['name']}刷新为 {current['left']} 个己方回合。",
            'countdown', side=by, source=source_cid, actor=f'{by}:{source_cid}',
            target=f'{target_side}:front', present=False)
        _notify_applied(state, by, source_cid, kind)
        return 0
    gained = 0
    if current and current.get('by') == by:
        before = int(current.get('stacks') or 0)
        current['stacks'] = min(spec['max_stacks'], before + stacks)
        gained = current['stacks'] - before
        current['source'] = f'{by}:{source_cid}'
        if duration is not None:
            current['left'] = int(duration)
    else:
        gained = min(spec['max_stacks'], stacks)
        dots[kind] = {
            'name': spec['name'], 'stacks': gained, 'by': by,
            'source': f'{by}:{source_cid}', 'left': duration,
            'applied_turn': state.get('turn'), 'mode': spec['mode'],
            'halve': bool(spec.get('halve')), 'ignore_shield': bool(spec.get('ignore_shield')),
            'max_stacks': spec['max_stacks'],
        }
    if gained or spec.get('unique'):
        shown = dots[kind]['stacks']
        log(state, f"对方战斗区获得{spec['name']}（{shown} 层）。", 'harmony',
            side=by, source=source_cid, actor=f'{by}:{source_cid}',
            target=f'{target_side}:front', present=False)
        _notify_applied(state, by, source_cid, kind)
    return gained


def apply_burn(state, target_side, *, by, source_cid):
    from app.modules.card_game.content.duel_v2.registry import decide
    zone = front_debuff(state, target_side)
    had_star = bool(zone.get('star'))
    burn = zone.get('burn')
    ctx = EffectContext(state, by, source_cid, {'side': by})
    gained = 0
    if burn and burn.get('by') == by:
        if not burn.get('infinite'):
            burn['left'] = 2
        before = int(burn.get('stacks') or 1)
        extra = int(decide(ctx, 'burn_extra_stack', 0) or 0)
        if extra and before < 3:
            burn['stacks'] = min(3, before + extra)
            gained = burn['stacks'] - before
        burn['source'] = f'{by}:{source_cid}'
        burn.setdefault('stacks', before)
        log(state, '对方战斗区的浊燃被刷新。', 'harmony', side=by, source=source_cid,
            actor=f'{by}:{source_cid}', target=f'{target_side}:front', present=False)
    else:
        zone['burn'] = {'left': 2, 'by': by, 'stacks': 1, 'source': f'{by}:{source_cid}',
                        'applied_turn': state.get('turn')}
        gained = 1
        log(state, '对方战斗区获得浊燃。', 'harmony', side=by, source=source_cid,
            actor=f'{by}:{source_cid}', target=f'{target_side}:front', present=False)
    fire(ctx, 'on_burn_applied', enemy=target_side, target_id=zone and state['sides'][target_side].get('front'))
    if gained:
        fire(ctx, 'on_burn_stacks', gained=gained, enemy=target_side)
    _notify_applied(state, by, source_cid, 'burn')
    if had_star:
        from .state import add_collapse, is_esper
        target = state['sides'][target_side].get('front')
        if is_esper(target) and alive(state, target_side, target):
            add_collapse(state, target_side, target, 1, by=by)
            log(state, f"{hero(state, by, source_cid)['name']}触发失谐。", 'harmony',
                side=by, source=source_cid, actor=f'{by}:{source_cid}',
                target=f'{target_side}:{target}')
            fire(ctx, 'on_dissonance', enemy=target_side, target_id=target)
    return gained


def add_stack_all(state, target_side, by, amount=1):
    zone = front_debuff(state, target_side)
    burn = zone.get('burn')
    if burn and burn.get('by') == by:
        before = int(burn.get('stacks') or 1)
        burn['stacks'] = min(3, before + amount)
        if burn['stacks'] > before and burn.get('source'):
            side, cid = burn['source'].split(':', 1)
            _notify_applied(state, side, cid, 'burn')
            fire(EffectContext(state, side, cid, {'side': side}), 'on_burn_stacks',
                 gained=burn['stacks'] - before, enemy=target_side)
    for kind, dot in list((zone.get('dots') or {}).items()):
        if dot.get('by') != by:
            continue
        before = int(dot.get('stacks') or 0)
        dot['stacks'] = min(int(dot.get('max_stacks') or 10), before + amount)
        if dot['stacks'] > before and dot.get('source'):
            side, cid = dot['source'].split(':', 1)
            _notify_applied(state, side, cid, kind)


def halve_dots(state, side):
    """Countdown only. Does not fire turn-start hooks."""
    for target_side in SIDES:
        dots = front_debuff(state, target_side).get('dots') or {}
        for kind, dot in list(dots.items()):
            if not dot.get('halve') or dot.get('by') != side:
                continue
            if dot.get('applied_turn') == state.get('turn'):
                continue
            dot['stacks'] = int(dot.get('stacks') or 0) // 2
            name = dot.get('name') or kind
            if dot['stacks'] <= 0:
                dots.pop(kind, None)
                log(state, f'{name}层数归零，持续伤害结束。', 'countdown', side=target_side,
                    actor=f'{target_side}:front', target=f'{target_side}:front', present=False)
            else:
                log(state, f'{name}减半，剩余 {dot["stacks"]} 层。', 'countdown', side=target_side,
                    actor=f'{target_side}:front', target=f'{target_side}:front', present=False)


def _rose_followup(state, dot, cid, dealt):
    source = dot.get('source')
    ctx = _source_context(state, source)
    if not ctx or ctx.character.get('shape') != 'A08' or dealt <= 0:
        return
    from .state import heal, reduce_max_hp
    target_side = other(ctx.side) if ctx.side in SIDES else None
    if target_side is None:
        return
    h = hero(state, target_side, cid)
    if h.get('hp', 0) > 0:
        reduce_max_hp(state, target_side, cid, dealt, source=source, settle=False)
    if not finished(state) and alive(state, ctx.side, ctx.cid):
        heal(state, ctx.side, ctx.cid, dealt, source=source, reason='「最后一朵玫瑰」')


def _kill_player(state, dot, killed):
    if not killed:
        return
    ctx = _source_context(state, dot.get('source'))
    if not ctx or ctx.character.get('shape') != 'A07' or not alive(state, ctx.side, ctx.cid):
        return
    foe = other(ctx.side)
    for _ in killed:
        if finished(state):
            return
        damage(state, foe, 'player', 2, source=dot.get('source'), kind='damage',
               reason='「噩梦」消灭了对方异能者')


def trigger_dot(state, target_side, kind, dot, *, reason=None):
    """Resolve a zone effect identically for its clock and an explicit card trigger."""
    source = dot.get('source') or f"{dot['by']}:{kind}"
    stacks = int(dot.get('stacks') or 0)
    kwargs = dict(source=source, by=dot['by'], reason=reason or dot.get('name'))
    if dot.get('mode') == 'single':
        return deal_one(state, target_side, 1, ignore_shield=bool(dot.get('ignore_shield')), **kwargs)
    if dot.get('mode') == 'every5':
        return distribute(state, target_side, stacks // 5, **kwargs)
    after = (lambda cid, dealt: _rose_followup(state, dot, cid, dealt)) if kind == 'nightmare' else None
    killed = distribute(state, target_side, stacks, after_hit=after, **kwargs)
    if kind == 'nightmare' and not finished(state):
        _kill_player(state, dot, killed)
    return killed


def tick_owner_end(state, side):
    if finished(state):
        return
    for target_side in SIDES:
        zone = front_debuff(state, target_side)
        burn = zone.get('burn')
        if burn and burn.get('by') == side:
            source = burn.get('source') or f'{side}:burn'
            killed = distribute(state, target_side, int(burn.get('stacks') or 1), source=source, by=side,
                                reason='浊燃', damage_class='harmony')
            if not burn.get('infinite'):
                burn['left'] = int(burn.get('left') or 1) - 1
                if burn['left'] <= 0:
                    zone.pop('burn', None)
            if not finished(state):
                log(state, '浊燃结算。', 'countdown', side=target_side,
                    actor=f'{target_side}:front', target=f'{target_side}:front', present=False)
            if finished(state):
                return
        for kind, dot in list((zone.get('dots') or {}).items()):
            if dot.get('by') != side or finished(state):
                continue
            trigger_dot(state, target_side, kind, dot)
            if finished(state):
                return
            if dot.get('left') is not None:
                dot['left'] = int(dot['left']) - 1
                if dot['left'] <= 0:
                    (zone.get('dots') or {}).pop(kind, None)
                    log(state, f"{dot.get('name')}结束。", 'countdown', side=target_side,
                        actor=f'{target_side}:front', target=f'{target_side}:front', present=False)
