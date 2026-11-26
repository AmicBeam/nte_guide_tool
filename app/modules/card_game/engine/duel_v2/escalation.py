"""User-defined playtest pacing (2026-10-12), counted in global player turns."""
from app.modules.card_game.content.duel_v2 import CARDS
from .state import add_hand, can_gain_energy, energy, energy_max, hero, log, shuffle, team_order

STAGES = (
    (6, '每回合可用 2 次终结；回合开始时，全体己方可获能的存活异能者能量 +1'),
    (13, '能量上限 -1'),
    (20, '回合开始时，随机从己方弃牌堆回收 1 张可构筑卡牌；每回合行动力刷新为 3 点'),
)


def enabled(state):
    return bool(state.get('escalation_enabled'))


def reached(state, turn):
    return enabled(state) and state.get('turn', 0) >= turn


def energy_cap(state, character):
    return energy_max(character) - int(reached(state, 13))


def harmony_cap(state):
    return 3


def turn_action_points(state):
    return 1 if state.get('turn') == 1 else 3 if reached(state, 20) else 2


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
    short_effects = ('终结×2 · 回能+1', '能量上限-1', '回收1张 · 行动力3')
    stage = len(active)
    title = '白热化' + ('', 'Ⅰ', 'Ⅱ', 'Ⅲ')[stage]
    short_hint = (f'{title}\n{upcoming[0] - turn}回合后\n{short_effects[stage]}'
                  if upcoming else f'{title}\n全部生效')
    return {
        'stage': len(active), 'active': active,
        'short_hint': short_hint,
        'hint': f'{upcoming[0] - turn}回合后，{upcoming[1]}' if upcoming else '白热化：全部阶段已生效',
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
    for cid in team_order(state, side):
        if not can_gain_energy(state, side, cid):
            continue
        h = hero(state, side, cid)
        before = h['energy']
        energy(state, side, cid)
        if h['energy'] > before:
            log(state, f"白热化：{h['name']}能量 +{h['energy'] - before}。", 'resource',
                side=side, source=cid, actor=f'{side}:{cid}', target=f'{side}:{cid}',
                energy_delta=h['energy'] - before, before={'energy': before}, after={'energy': h['energy']})
    if reached(state, 20):
        team = state['sides'][side]
        candidates = [card for card in team['discard']
                      if card.get('card_id') in CARDS and not card.get('copy')
                      and not card.get('derived') and not CARDS[card['card_id']].get('derived')]
        if candidates:
            if len(candidates) > 1:
                shuffle(state, candidates)
            card = candidates[0]
            team['discard'].remove(card)
            add_hand(state, side, card, reveal=True, reason='白热化回收')
