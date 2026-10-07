"""Bounded, observer-only inference guide for the user's Zhenhong setup plan.

This is a hand-written inference policy, not a training reward or a WDL value.
It evaluates three determinized worlds and never uses true opponent dark cards.
"""
from copy import deepcopy
from hashlib import sha256
import time

VERSION = 'zhenhong_setup_v1'
MAX_PLANS = 24
SECONDS = .75


def _features(state, side):
    from app.modules.card_game.engine.duel_v2.state import front_debuff, attack_value, damage_limit
    from app.modules.card_game.engine.duel_v2.escalation import harmony_cap
    own = state['sides'][side]; foe = state['sides']['b' if side == 'a' else 'a']
    red = own['characters']['zhenhong']
    fangs = int(own['characters'].get('yi', {}).get('beast_fangs') or 0) > 0
    delay = front_debuff(state, 'b' if side == 'a' else 'a').get('delay') or {}
    delay = bool(delay and delay.get('by') == side)
    # Existing delay can be cashed out without requiring fangs first.
    stage = 3 if fangs and delay else 2 if fangs else 1 if delay else 0
    previous = own['front'] or own.get('last_front')
    payer = own['characters'].get(previous) or {}
    attrs = ('光',) if delay else ('光', '相')
    readiness = (min(int(payer.get('harmony') or 0), harmony_cap(state)) / harmony_cap(state)
                 if payer.get('hp', 0) > 0 and payer.get('attribute') in attrs else 0.)
    safe = True
    if red['hp'] > 0 and own['front'] == 'zhenhong' and damage_limit(state, side, 'zhenhong') is None:
        threat = max((attack_value(h, state, 'b' if side == 'a' else 'a')
                      for h in foe['characters'].values() if h['hp'] > 0), default=0)
        safe = red['hp'] + red['shield'] > threat
    return dict(stage=stage, fangs=fangs, delay=delay, readiness=readiness,
                safe=safe, triggers=int(red.get('surplus_passive_triggers') or 0))


def _rank(before, after, side):
    enemy = 'b' if side == 'a' else 'a'
    terminal = 2 if after['phase'] == 'finished' and after.get('winner') == side else (
        0 if after['phase'] == 'finished' and after.get('winner') == enemy else 1)
    old = before['sides'][side]; own = after['sides'][side]; foe = after['sides'][enemy]
    red = own['characters']['zhenhong']; f = _features(after, side)
    delta = f['triggers'] - int(old['characters']['zhenhong'].get('surplus_passive_triggers') or 0)
    # Survival comes before speculative preparation; an actual win comes first.
    return (terminal, int(red['hp'] > 0), int(f['safe']), int(delta > 0), min(delta, 3),
            f['stage'], f['readiness'], own['hp'], -foe['hp'],
            sum(h['hp'] > 0 for h in own['characters'].values()), own['ap'])


def recommend(state, side, baseline, *, seed=None, seconds=SECONDS, max_plans=MAX_PLANS, runtime=None):
    """Return an auditable suggestion; caller explicitly decides whether to enable it.

    Replan each real decision. Never commit actions involving a future unknown
    draw, and never modify the input snapshot or the model/search policy target.
    """
    from app.modules.card_game.engine.duel_v2 import legal_actions, acting_side
    from app.modules.card_game.engine.duel_v2.state import effective_card
    from app.modules.card_game.engine.duel_v2.simulation import simulate_action
    from app.modules.card_game.engine.duel_v2.combat import resolved_harmony_kind
    from app.modules.card_game.rl.information_search import sample_world
    if runtime is None:
        from app.modules.card_game.rl import recovery_runtime as runtime
    result = dict(version=VERSION, selected=deepcopy(baseline), changed=False, plans=0,
                  worlds=0, reason='not_applicable', seconds=0.)
    own = state['sides'][side]; red = own['characters'].get('zhenhong')
    if (state['phase'] != 'playing' or acting_side(state) != side or not red
            or red['hp'] <= 0 or red['awakened'] or 'yi' not in own['characters']):
        return result
    start = time.monotonic(); end = start + max(0., seconds)
    legal = [a for a in legal_actions(state, side) if a['type'] != 'concede']
    if baseline not in legal:
        raise ValueError('Guide baseline must be legal')
    if seconds <= 0 or max_plans < 1:
        result['reason'] = 'budget'; return result
    if seed is None:
        seed = int.from_bytes(sha256(f'{VERSION}:{state["version"]}:{side}'.encode()).digest()[:8], 'big')
    try:
        worlds = [sample_world(state, side, seed + 104729*i, runtime=runtime) for i in range(3)]
    except (ValueError, KeyError, TypeError, NotImplementedError):
        result['reason'] = 'unsupported_information_set'; return result
    result['worlds'] = 3; result['stage_before'] = _features(worlds[0], side)
    options = []

    def evaluate(route, starts):
        if result['plans'] >= max_plans or time.monotonic() >= end:
            return None
        result['plans'] += 1
        action = route[-1]
        try:
            if any(acting_side(w) != side or action not in legal_actions(w, side) for w in starts):
                return None
            after = [simulate_action(w, side, action) for w in starts]
        except (ValueError, KeyError, TypeError, NotImplementedError):
            return None
        ranks = [_rank(w, a, side) for w, a in zip(worlds, after)]
        entry = (min(ranks), route, after)
        options.append(entry)
        return entry

    original = evaluate([baseline], worlds)
    if original is None:
        result['reason'] = 'budget_or_unsupported_baseline'; return result
    # Preserve the existing planner's winning action in every sampled world.
    if original[0][0] == 2:
        result.update(reason='preserve_winning_action', seconds=time.monotonic()-start); return result
    hand = {c['instance_id']: c for c in own['hand']}
    features = result['stage_before']

    def order(action):
        cid = action.get('character_id'); face = None
        if action['type'] == 'play_card':
            face = effective_card(state, side, hand[action['card_id']]); cid = face['character_id']
            if face['card_id'] in ('I03', 'I04') and not features['fangs']:
                return (0, int(face['card_id'] == 'I03'))
        sortie = action['type'] == 'attack' or face and face['type'] == 'battle'
        kind = resolved_harmony_kind(state, side, cid) if sortie else None
        if kind == '创生' and features['delay']: return (0, 2)
        if kind == '延滞': return (1, 0)
        if cid and own['characters'][cid]['attribute'] == '光' and sortie: return (2, 0)
        return (3 if sortie else 4, 0)

    # A small deterministic root shortlist, with baseline always evaluated.
    for action in sorted((a for a in legal if a != baseline), key=order)[:11]:
        evaluate([action], worlds)
    # Complete same-turn setups without pretending that a future draw is known.
    # The second action must be legal in all three independently sampled worlds.
    firsts = sorted(options, key=lambda row: row[0], reverse=True)[:4]
    for _, route, starts in firsts:
        if any(w['phase'] != 'playing' or acting_side(w) != side for w in starts): continue
        for action in legal_actions(starts[0], side):
            if action['type'] not in ('attack', 'play_card', 'ultimate'): continue
            evaluate(route+[action], starts)
    best = max(options, key=lambda row: (row[0], row[1][0] == baseline, -len(row[1])))
    result.update(reason='keep_model', baseline_rank=list(original[0]), best_rank=list(best[0]),
                  route=deepcopy(best[1]), seconds=time.monotonic()-start)
    if best[0] > original[0] and best[1][0] != baseline:
        result.update(selected=deepcopy(best[1][0]), changed=True, reason='setup_progress_or_safety',
                      sampled_checks=[dict(features=_features(w, side), rank=list(_rank(base, w, side)))
                                      for base, w in zip(worlds, best[2])])
    return result
