"""User-defined playtest pacing (2026-09-15), counted in global player turns."""
from .state import alive, can_gain_energy, energy, energy_max, hero, log, shuffle, team_order

STAGES = (
    (6, '每回合可用 2 次终结；回合开始时，能量最低的存活角色能量 +1（同值随机）'),
    (13, '能量上限 -1'),
    (20, '环合值上限降为 1 点'),
)


def enabled(state):
    return bool(state.get('escalation_enabled'))


def reached(state, turn):
    return enabled(state) and state.get('turn', 0) >= turn


def energy_cap(state, character):
    return energy_max(character) - int(reached(state, 13))


def harmony_cap(state):
    return 2 - int(reached(state, 20))


def ultimate_limit(state):
    return 2 if reached(state, 6) else 1


def ultimate_remaining(state, side):
    team = state['sides'][side]
    if not team.get('ultimate_available'):
        return 0
    return max(0, ultimate_limit(state) - int(team.get('ultimates_used', 0)))


def public_progress(state):
    if not enabled(state):
        return None
    turn = int(state.get('turn', 0))
    active = [text for start, text in STAGES if turn >= start]
    upcoming = next(((start, text) for start, text in STAGES if turn < start), None)
    return {
        'stage': len(active), 'active': active,
        'hint': (f'{upcoming[0] - turn}回合后，' + ('每回合终结2次，最低能量角色能量+1' if upcoming[0] == 6 else upcoming[1])) if upcoming else '白热化：全部阶段已生效',
        'description': '；'.join(text for _, text in STAGES),
    }


def clamp_resources(state):
    if not enabled(state):
        return
    for ps in state['sides']:
        for h in state['sides'][ps]['characters'].values():
            h['energy'] = min(h['energy'], energy_cap(state, h))
            h['harmony'] = min(h['harmony'], harmony_cap(state))


def begin_escalation(state, side):
    if not reached(state, 6):
        return
    candidates = [cid for cid in team_order(state, side) if can_gain_energy(state, side, cid)]
    if not candidates:
        return
    lowest = min(hero(state, side, cid)['energy'] for cid in candidates)
    candidates = [cid for cid in candidates if hero(state, side, cid)['energy'] == lowest]
    if len(candidates) > 1:
        shuffle(state, candidates)
    cid = candidates[0]
    h = hero(state, side, cid)
    before = h['energy']
    energy(state, side, cid)
    log(state, f"白热化：{h['name']}能量 +{h['energy'] - before}。", 'resource',
        side=side, source=cid, actor=f'{side}:{cid}', target=f'{side}:{cid}',
        energy_delta=h['energy'] - before, before={'energy': before}, after={'energy': h['energy']})
