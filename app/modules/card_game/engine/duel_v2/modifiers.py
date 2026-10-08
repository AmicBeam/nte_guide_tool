"""Serializable effects and pure property queries; no Flask, persistence or UI.

Stored effects are the source of truth. Legacy flags are read through an adapter
until the next write/tick imports them. Queries never mutate a snapshot.
"""
from copy import deepcopy
from math import isfinite
from typing import TypedDict


class Effect(TypedDict, total=False):
    effect_id: str
    definition_id: str
    property: str
    op: str
    value: int | float | bool
    source_entity_id: str | None
    source_key: str
    scope: str
    kind: str
    layer: int
    label: str
    clear_on: list[str]
    expiry: dict
    conditions: list[dict]
    consume_on: str
    visibility: str
    stack_key: str
    show_marker: bool
    stack_limit: int


PHASES = ('turn_start', 'turn_end')
OPS = ('add', 'multiply', 'set', 'allow', 'forbid')
SCOPES = ('self', 'allies', 'other_allies', 'enemies', 'all')
CONDITIONS = {}


def condition(name):
    def register(fn):
        if name in CONDITIONS and CONDITIONS[name] is not fn:
            raise ValueError(f'Duplicate condition: {name}')
        CONDITIONS[name] = fn
        return fn
    return register


def contribution(definition_id, property, value, *, op='add', layer=0,
                 source_entity_id=None, source_key='', scope='self', kind='persistent',
                 label='', clear_on=None, expiry=None, conditions=(),
                 consume_on=None, visibility='public', stack_key=None, show_marker=None):
    if op not in OPS or scope not in SCOPES or kind not in ('persistent', 'aura'):
        raise ValueError('Invalid effect operation, scope or kind')
    if visibility not in ('public', 'private'):
        raise ValueError('Invalid effect visibility')
    if not isinstance(value, (int, float, bool)) or not isfinite(value):
        raise ValueError('Effects must store numeric or boolean values')
    if expiry and (expiry.get('phase') not in PHASES or not ({'turn', 'remaining'} & expiry.keys())):
        raise ValueError('Invalid expiry clock')
    for item in conditions:
        if item.get('id') not in CONDITIONS:
            raise ValueError(f"Unknown effect condition: {item.get('id')}")
    result = dict(effect_id=f'{source_entity_id}:query:{definition_id}', definition_id=definition_id, property=property,
                  op=op, value=value, source_entity_id=source_entity_id,
                  source_key=source_key, scope=scope, kind=kind, layer=layer,
                  label=label or definition_id, clear_on=list(clear_on if clear_on is not None else (() if kind == 'aura' else ('down',))),
                  visibility=visibility, show_marker=bool(expiry or consume_on) if show_marker is None else show_marker)
    if stack_key is not None:
        result['stack_key'] = stack_key
    if expiry:
        result['expiry'] = deepcopy(expiry)
    if conditions:
        result['conditions'] = deepcopy(list(conditions))
    if consume_on:
        result['consume_on'] = consume_on
    return result


def add_effect(holder, definition_id, property, value, *, stacking='independent', max_stacks=None, **options):
    """Apply one contribution. Refresh/max/replace match definition AND source."""
    if stacking not in ('independent', 'refresh', 'max', 'replace'):
        raise ValueError('Unknown stacking policy')
    if max_stacks is not None and (not isinstance(max_stacks, int) or max_stacks < 1):
        raise ValueError('Stack limit must be a positive integer')
    effect = contribution(definition_id, property, value, **options)
    if max_stacks is not None:
        effect['stack_limit'] = max_stacks
    migrate_legacy(holder)
    effects = holder.setdefault('effects', [])
    keys = ('stack_key', 'property', 'scope', 'op', 'kind') if effect.get('stack_key') else (
        'definition_id', 'source_entity_id', 'source_key', 'property', 'scope', 'op', 'kind')
    matches = [e for e in effects if all(e.get(k) == effect.get(k) for k in keys)]
    if matches and stacking != 'independent':
        previous = matches[-1]
        effect['effect_id'] = previous['effect_id']
        if stacking == 'refresh':
            effect['value'] = previous['value']
        elif stacking == 'max' and previous['value'] > effect['value']:
            effect['value'] = previous['value']
            for k in ('definition_id', 'source_entity_id', 'source_key', 'label', 'visibility', 'conditions'):
                if k in previous:
                    effect[k] = deepcopy(previous[k])
                else:
                    effect.pop(k, None)
        effects[:] = [e for e in effects if e not in matches]
    else:
        holder['effect_seq'] = int(holder.get('effect_seq', 0)) + 1
        effect['effect_id'] = f"{holder.get('entity_id', 'legacy')}:effect:{holder['effect_seq']}"
    if stacking == 'independent' and max_stacks is not None and len(matches) >= max_stacks:
        replaced = {e['effect_id'] for e in matches[:len(matches) - max_stacks + 1]}
        effects[:] = [e for e in effects if e['effect_id'] not in replaced]
    effects.append(effect)
    return effect


def deadline(state, side, turns, phase='turn_start'):
    if phase not in PHASES or turns < 0 or (phase == 'turn_start' and turns == 0):
        raise ValueError('Invalid duration')
    return dict(side=side, turn=state['sides'][side]['turn_count'] + turns, phase=phase)


def legacy_effects(holder):
    flags = holder.get('flags') or {}
    source = holder.get('entity_id')
    result = []
    def make(key, value, property='attack.panel', **kw):
        if value:
            e = contribution('legacy.' + key, property, value,
                             source_entity_id=source, source_key=key, **kw)
            e['effect_id'] = f'{source}:legacy:{key}:{len(result)}'
            result.append(e)
    expiring = flags.get('atk_buff_expiring') or {}
    temporary = int(expiring.get('amount', 0))
    make('atk_buff', int(flags.get('atk_buff', 0)) - temporary, label='攻击增益')
    if temporary:
        make('atk_buff_expiring', temporary, label='攻击增益', show_marker=False, expiry=dict(
            side=holder.get('side'), turn=expiring['until'], phase='turn_start'))
    make('atk_debuff', -int(flags.get('atk_debuff', 0)), label='攻击降低')
    make('form_attack', int(flags.get('form_attack', 0)), label='武备攻击')
    for modifier in flags.get('timed_attack', []):
        if modifier['left'] > 0:
            make('timed_attack', modifier['amount'], label='攻击增益' if modifier['amount'] > 0 else '攻击降低',
                 expiry=dict(side=holder.get('side'), remaining=modifier['left'], phase='turn_start'))
    make('next_attack', int(flags.get('next_attack', 0)), 'attack.hit',
         label='攻击强化', consume_on='attack')
    if flags.get('damage_immunity_until'):
        e = contribution('legacy.damage_immunity', 'rule.damage_immune', True, op='allow',
                         source_entity_id=source, label='免伤', show_marker=False,
                         expiry=dict(side=holder.get('side'), turn=flags['damage_immunity_until'], phase='turn_start'))
        e['effect_id'] = f'{source}:legacy:damage_immunity'
        result.append(e)
    for e in result:
        if e['definition_id'] == 'legacy.next_attack':
            e['definition_id'] = 'attack.next'
            e['stack_key'] = 'next_attack'
    return result


def migrate_legacy(holder, state=None, side=None):
    effects = legacy_effects(holder)
    if state is not None and side is not None:
        for effect in effects:
            expiry = effect.get('expiry')
            if expiry:
                expiry['side'] = expiry.get('side') or side
                if 'remaining' in expiry:
                    expiry['turn'] = state['sides'][side]['turn_count'] + expiry.pop('remaining')
    if effects:
        for effect in effects:
            holder['effect_seq'] = int(holder.get('effect_seq', 0)) + 1
            effect['effect_id'] = f"{holder.get('entity_id', 'legacy')}:effect:{holder['effect_seq']}"
        holder.setdefault('effects', []).extend(effects)
    flags = holder.get('flags') or {}
    for key in ('atk_buff', 'atk_buff_expiring', 'atk_debuff', 'form_attack', 'timed_attack', 'next_attack', 'damage_immunity_until'):
        flags.pop(key, None)
    return holder.get('effects', [])


def all_effects(holder):
    return list(holder.get('effects', [])) + legacy_effects(holder)


def entities(state):
    for side, team in state['sides'].items():
        yield side, team
        for h in team['characters'].values():
            yield side, h


def source_for(state, effect, holder):
    eid = effect.get('source_entity_id')
    if not eid or eid == holder.get('entity_id'):
        return holder
    return next((h for _, h in entities(state) if h.get('entity_id') == eid), None)


def _applies(effect, holder, holder_side, subject, side, state, query):
    expiry = effect.get('expiry') or {}
    if state is not None and 'turn' in expiry:
        clock = state['sides'][expiry.get('side') or holder_side]['turn_count']
        if expiry['turn'] < clock or (expiry['phase'] == 'turn_start' and expiry['turn'] == clock):
            return False
    scope = effect['scope']
    same = holder is subject or (holder.get('entity_id') is not None and holder.get('entity_id') == subject.get('entity_id'))
    if not {'self': same, 'allies': holder_side == side, 'other_allies': holder_side == side and not same,
            'enemies': holder_side != side, 'all': True}[scope]:
        return False
    source = source_for(state, effect, holder) if state is not None else holder
    if effect['kind'] == 'aura' and (not source or source.get('hp', 0) <= 0):
        return False
    context = dict(state=state, side=side, subject=subject, source=source,
                   holder=holder, holder_side=holder_side, query=query)
    for item in effect.get('conditions', []):
        fn = CONDITIONS.get(item['id'])
        if fn is None:
            raise ValueError(f"Unknown effect condition: {item['id']}")
        if not fn(context, item.get('args', {})):
            return False
    return True


def reduce_modifiers(base, effects):
    """Stable layers; within a layer: set, add, multiply, allow, forbid."""
    order = {'set': 0, 'add': 1, 'multiply': 2, 'allow': 3, 'forbid': 4}
    value = base
    details = []
    for effect in sorted(effects, key=lambda e: (e['layer'], order[e['op']])):
        before = value
        op, amount = effect['op'], effect['value']
        if op == 'set':
            value = amount
        elif op == 'add':
            value += amount
        elif op == 'multiply':
            value *= amount
        elif op == 'allow':
            value = bool(value or amount)
        elif op == 'forbid' and amount:
            value = False
        details.append({k: deepcopy(effect.get(k)) for k in (
            'effect_id', 'definition_id', 'source_entity_id', 'source_key', 'label', 'property', 'op', 'value', 'visibility')})
        details[-1].update(before=before, after=value)
    return dict(value=value, contributions=details)


def evaluate(state, side, subject, property, base, *, query=None, extra=()):
    query = query or {}
    candidates = []
    holders = list(entities(state)) if state is not None else [(side, subject)]
    for holder_side, holder in holders:
        for effect in all_effects(holder):
            if effect['property'] == property and _applies(effect, holder, holder_side, subject, side, state, query):
                candidates.append(effect)
    if state is not None:
        from app.modules.card_game.content.duel_v2.registry import KITS
        from .context import EffectContext
        for holder_side, team in state['sides'].items():
            for cid, holder in team['characters'].items():
                kit = KITS.get(cid)
                provider = getattr(kit, 'aura_effects', None)
                if provider:
                    c = EffectContext(state, holder_side, cid, query.get('operation') or {'side': holder_side})
                    for effect in provider(c, property=property, query=query):
                        if effect['property'] == property and _applies(effect, holder, holder_side, subject, side, state, query):
                            candidates.append(effect)
    candidates.extend(extra)
    return reduce_modifiers(base, candidates)


def consume(holder, event):
    migrate_legacy(holder)
    holder['effects'] = [e for e in holder.get('effects', []) if e.get('consume_on') != event]


def clear(holder, event, *, source_key=None):
    migrate_legacy(holder)
    holder['effects'] = [e for e in holder.get('effects', []) if not (
        event in e.get('clear_on', []) and (source_key is None or e.get('source_key') == source_key))]


def prepare_clocks(state):
    """Import old relative counters before advancing the global turn clock."""
    for side, holder in entities(state):
        migrate_legacy(holder, state, side)
        for effect in holder.get('effects', []):
            expiry = effect.get('expiry')
            if expiry and 'remaining' in expiry:
                expiry['side'] = expiry.get('side') or side
                expiry['turn'] = state['sides'][expiry['side']]['turn_count'] + expiry.pop('remaining')


def tick(state, side, phase):
    if phase not in PHASES:
        raise ValueError('Unknown effect clock phase')
    expired = []
    for holder_side, holder in entities(state):
        kept = []
        for effect in holder.get('effects', []):
            expiry = effect.get('expiry')
            if (expiry and expiry['phase'] == phase and (expiry.get('side') or holder_side) == side
                    and expiry['turn'] <= state['sides'][side]['turn_count']):
                expired.append((holder_side, holder, effect))
            else:
                kept.append(effect)
        holder['effects'] = kept
    return expired


def public_effects(holder):
    allowed = tuple(Effect.__annotations__)
    return [{k: deepcopy(e[k]) for k in allowed if k in e}
            for e in all_effects(holder) if e.get('visibility') == 'public']


@condition('source_front')
def _source_front(c, args):
    s, source = c['state'], c['source']
    if s is None or not source:
        return False
    team = s['sides'][source.get('side', c['holder_side'])]
    front = team['characters'].get(team.get('front'))
    return bool(front and front.get('entity_id') == source.get('entity_id'))


@condition('source_alive')
def _source_alive(c, args):
    return bool(c['source'] and c['source'].get('hp', 0) > 0)


@condition('source_awakened')
def _source_awakened(c, args):
    return bool(c['source'] and c['source'].get('awakened'))


@condition('source_shape')
def _source_shape(c, args):
    return bool(c['source'] and c['source'].get('shape') == args.get('card_id'))


@condition('enemy_turn')
def _enemy_turn(c, args):
    return bool(c['state'] and c['state']['active_side'] != c['holder_side'])


@condition('resource_at_least')
def _resource(c, args):
    return bool(c['source'] and c['source'].get(args['resource'], 0) >= args['amount'])


def remaining(effect, state=None, side=None):
    expiry = effect.get('expiry') or {}
    if 'remaining' in expiry:
        return expiry['remaining']
    if state is not None and expiry.get('turn') is not None:
        clock_side = expiry.get('side') or side
        return max(0, expiry['turn'] - state['sides'][clock_side]['turn_count'])
    return None


def compatibility_flags(holder, state=None, side=None):
    """Derived legacy projection for encoders/UI; never written back to state."""
    flags = deepcopy(holder.get('flags') or {})
    for key in ('atk_buff', 'atk_debuff', 'form_attack', 'next_attack'):
        flags[key] = 0
    flags.pop('atk_buff_expiring', None)
    flags.pop('damage_immunity_until', None)
    flags['timed_attack'] = []
    for effect in all_effects(holder):
        if effect['property'] == 'rule.damage_immune' and effect.get('visibility') == 'public' and effect.get('expiry'):
            flags['damage_immunity_until'] = effect['expiry'].get('turn')
        if effect['scope'] != 'self' or effect['op'] != 'add' or effect.get('visibility') != 'public':
            continue
        prop, value = effect['property'], effect['value']
        expiry = effect.get('expiry') or {}
        if prop == 'attack.hit' and effect.get('consume_on') == 'attack':
            flags['next_attack'] += value
        elif prop == 'attack.panel':
            if effect['definition_id'] in ('legacy.timed_attack', 'yi.attack_debuff'):
                flags['timed_attack'].append(dict(amount=value, left=remaining(effect, state, side)))
            elif effect['definition_id'] == 'legacy.form_attack':
                flags['form_attack'] += value
            elif value < 0:
                flags['atk_debuff'] -= value
            else:
                flags['atk_buff'] += value
                if expiry.get('phase') == 'turn_start':
                    record = flags.setdefault('atk_buff_expiring', dict(amount=0, until=expiry.get('turn')))
                    record['amount'] += value
    return flags


def remove_source(state, entity_id, source_key):
    """Remove only source-linked effects, wherever their holders are."""
    for _, holder in entities(state):
        holder['effects'] = [e for e in holder.get('effects', []) if not (
            e.get('source_entity_id') == entity_id and e.get('source_key') == source_key
            and 'source_removed' in e.get('clear_on', []))]


def consume_applied(state, event, effect_ids):
    ids = set(effect_ids)
    for _, holder in entities(state):
        holder['effects'] = [e for e in holder.get('effects', [])
                             if not (e['effect_id'] in ids and e.get('consume_on') == event)]


@condition('damage_attribute')
def _damage_attribute(c, args):
    return c['query'].get('attribute') == args.get('attribute')


def resistance_values(holder):
    """Compatibility observation only; gameplay resolves damage.resistance."""
    values = dict(holder.get('next_damage_resistance') or {})
    for e in all_effects(holder):
        if e['property'] != 'damage.resistance' or e.get('visibility') != 'public':
            continue
        attribute = next((x.get('args', {}).get('attribute') for x in e.get('conditions', [])
                          if x['id'] == 'damage_attribute'), None)
        if attribute:
            values[attribute] = values.get(attribute, 0) + e['value']
    return values
