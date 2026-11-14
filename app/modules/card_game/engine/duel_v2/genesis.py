"""Immediate genesis hits and independently timed beads with fixed stack counts."""
from app.modules.card_game.content.duel_v2.registry import decide, fire
from app.modules.card_game.engine.events import GameEvent
from .state import alive, damage, energy, finished, front_debuff, front_target, hero, is_esper, log, team_order


def public_beads(state, side):
    """Normalize legacy layer records without mutating an observed snapshot."""
    from .context import EffectContext
    beads = state['sides'][side].get('genesis_layers', [])
    legacy_capacity = 1
    if any('stacks' not in bead for bead in beads):
        actor = team_order(state, side)[0]
        legacy_capacity = max(1, min(2, int(decide(EffectContext(state, side, actor, {'side': side}),
                                                  'genesis_bead_stacks', 1, include_down=True) or 1)))
    groups = []
    last_signature = None
    for bead in beads:
        signature = (bead.get('uncapped', False), bead.get('source_entity_id'),
                     bead.get('left'), bead.get('used_turn', -1))
        if 'stacks' not in bead:
            if last_signature == signature and groups[-1]['stacks'] < legacy_capacity:
                groups[-1]['stacks'] += 1
                continue
            groups.append(dict(bead, stacks=1))
            last_signature = signature
        else:
            groups.append(dict(bead, stacks=max(1, min(2, int(bead['stacks'])))))
            last_signature = None
    result = []
    ordinary = 0
    for bead in groups:
        if not bead.get('uncapped'):
            if ordinary >= 3:
                continue
            ordinary += 1
        result.append(bead)
    return result


def ensure_beads(state, side):
    team = state['sides'][side]
    beads = team.setdefault('genesis_layers', [])
    if (any('stacks' not in bead for bead in beads)
            or sum(not bead.get('uncapped') for bead in beads) > 3):
        team['genesis_layers'] = public_beads(state, side)
    return team['genesis_layers']


def add_layers(c, *, uncapped=False):
    team = c.state['sides'][c.side]
    layers = ensure_beads(c.state, c.side)
    count = max(0, int(decide(c, 'genesis_layer_count', 1) or 0))
    cap = 3
    stacks = max(1, min(2, int(decide(c, 'genesis_bead_stacks', 1) or 1)))
    team['genesis_next_layer'] = max([int(team.get('genesis_next_layer') or 0)] +
                                     [int(bead['id']) for bead in layers])
    if not uncapped:
        count = min(count, max(0, cap - sum(not item.get('uncapped') for item in layers)))
    for _ in range(count):
        team['genesis_next_layer'] = int(team.get('genesis_next_layer') or 0) + 1
        layers.append({'id': team['genesis_next_layer'], 'left': 2, 'used_turn': -1, 'stacks': stacks,
                       'uncapped': bool(uncapped), 'source_entity_id': c.entity_id})
    if count:
        c.history(f'获得 {count} 珠【创生】，每珠 {stacks} 层，独立持续 2 个己方回合。')


def refresh_layers(state, side):
    for layer in ensure_beads(state, side):
        layer['left'] = 2


def tick_layers(state, side):
    team = state['sides'][side]
    layers = ensure_beads(state, side)
    if not layers:
        return
    for layer in layers:
        layer['left'] -= 1
    team['genesis_layers'] = [layer for layer in layers if layer['left'] > 0]
    log(state, '【创生】各珠持续时间更新。', 'countdown', side=side,
        actor=f'{side}:player', target=f'{side}:player', present=False)


def hit(c):
    """One damage segment; each segment independently checks surplus."""
    state, side = c.state, c.side
    if finished(state) or not alive(state, side, c.cid):
        return
    team = state['sides'][side]
    fire(c, GameEvent.V2_GENESIS_SEGMENT)
    targets = decide(c, 'genesis_targets', [front_target(state, side)])
    targets = decide(c, 'genesis_sweep_targets', targets)
    delayed = False
    energy_gained = 0
    for target_side, target_id in targets:
        if finished(state) or not alive(state, side, c.cid):
            break
        if is_esper(target_id) and not alive(state, target_side, target_id):
            continue
        target = hero(state, target_side, target_id) if is_esper(target_id) else state['sides'][target_side]
        delayed = delayed or bool(front_debuff(state, target_side).get('delay'))
        before_shield = target['shield']
        amount = 1 + int(team.get('genesis_damage_bonus') or 0)
        amount = int(decide(c, 'genesis_hit_amount', amount, target_side=target_side, target_id=target_id) or 0)
        from .presentation import next_group_id
        loss = damage(state, target_side, target_id, amount, source=f'{side}:{c.cid}', kind='genesis',
                      group_id=next_group_id(state))
        fire(c, 'on_genesis_damage', target=(target_side, target_id),
             damage_dealt=loss > 0 or target['shield'] < before_shield)
    if delayed and alive(state, side, c.cid) and not finished(state):
        team['surplus'] = team['surplus_ever'] = True
        recipient = team.get('front')
        if recipient and alive(state, side, recipient):
            before = hero(state, side, recipient)['energy']
            energy(state, side, recipient, 1)
            energy_gained = hero(state, side, recipient)['energy'] - before
        log(state, f'{c.character["name"]}触发盈蓄。', 'harmony', side=side, source=c.cid,
            actor=f'{side}:{c.cid}', target=f'{side}:{recipient or c.cid}')
        fire(c, GameEvent.V2_SURPLUS_ENERGY, recipient=recipient, amount=energy_gained)
        fire(c, GameEvent.V2_SURPLUS, entering=c.cid)


def on_sortie(c):
    if c.operation.get('genesis_sortie_resolved'):
        return
    c.operation['genesis_sortie_resolved'] = True
    for layer in list(ensure_beads(c.state, c.side)):
        if finished(c.state):
            break
        if layer.get('used_turn') == c.state['turn']:
            continue
        layer['used_turn'] = c.state['turn']
        for _ in range(layer['stacks']):
            hit(c)
