"""Training-only, bounded tie breakers; terminal rewards remain unchanged."""
from copy import deepcopy
import numpy as np


def certified_lethals(state, viewer, actions):
    """Conservative one-action proof, not an all-particles-win estimate.

    Unknown defending hand, draws/searches and RNG use disqualify the proof.
    Deliberately accepts false negatives rather than guessing hidden responses.
    """
    from app.modules.card_game.engine.duel_v2 import apply_action
    if state.get('phase') != 'playing' or state.get('pending_choice') or state.get('operation'):
        return set()
    foe = 'b' if viewer == 'a' else 'a'
    team = state['sides'][foe]
    revealed = set(team.get('revealed_ids', []))
    if any(card['instance_id'] not in revealed for card in team['hand']):
        return set()
    # Opponent gold marks are private even when a card's identity is public.
    if 'xiaozhi' in team['characters'] and team['hand']:
        return set()
    base = deepcopy(state)
    base['events'] = []
    base.pop('_public_board', None)
    base['rng'] = 1729  # Never consult the actual hidden RNG state.
    # Hidden deck identities/order cannot participate in a certified result.
    for side, t in base['sides'].items():
        t['deck'] = [dict(instance_id=f'proof-{side}-{i}', card_id='unknown',
                          character_id='unknown', type='tactic', cost=999)
                     for i in range(len(t['deck']))]
    from app.modules.card_game.engine.duel_v2.entities import hydrate_entities
    hydrate_entities(base)
    certified = set()
    for i, action in enumerate(actions):
        if action.get('type') not in ('attack', 'play_card', 'ultimate'):
            continue
        try:
            after = apply_action(deepcopy(base), viewer, action)
        except (ValueError, KeyError, TypeError):
            continue  # Unknown deck inspection fails closed.
        if after.get('winner') != viewer or after.get('phase') != 'finished':
            continue
        if after.get('rng') != base['rng']:
            continue
        if any(after['sides'][s]['deck'] != base['sides'][s]['deck'] for s in base['sides']):
            continue
        # Empty-deck draws also disqualify (no deck-length change in that case).
        if any(e.get('type') in ('draw', 'empty_draw', 'shuffle') or
               e.get('reason') in ('empty_draw', 'empty_draw_win') for e in after.get('events', [])):
            continue
        certified.add(i)
    return certified


def auxiliary_pairs(results, candidates, particles, lethal=()):
    """Return local-index (better, worse, weight, reason) pairs.

    Timing requires identical paired outcomes and dominance in EVERY decisive
    particle: earlier wins and later losses, never a trade against win rate.
    """
    if len(results) != candidates * particles or not all(g['complete'] for g in results):
        return []
    rows = [results[i*particles:(i+1)*particles] for i in range(candidates)]
    q = np.array([[g['reward'] for g in row] for row in rows])
    lethal = set(lethal)
    pairs = []
    for i in range(candidates):
        for j in range(candidates):
            if i == j:
                continue
            # Primary outcome ranking handles unequal means; never contradict it.
            if abs(float(q[i].mean()-q[j].mean())) > 1e-6:
                continue
            if i in lethal and j not in lethal and np.all(q[i] == 10):
                pairs.append((i, j, .2, 'certain_lethal'))
                continue
            if j in lethal or not np.array_equal(q[i], q[j]):
                continue
            wins = np.flatnonzero(q[i] == 10)
            losses = np.flatnonzero(q[i] == -10)
            # Defeat is delayed by turns, never by padding actions in one turn.
            win_a = [(rows[i][k]['remaining_turns'], rows[i][k]['own_decisions']) for k in wins]
            win_b = [(rows[j][k]['remaining_turns'], rows[j][k]['own_decisions']) for k in wins]
            loss_a = [rows[i][k]['remaining_turns'] for k in losses]
            loss_b = [rows[j][k]['remaining_turns'] for k in losses]
            if not all(x <= y for x, y in zip(win_a, win_b)) or not all(x >= y for x, y in zip(loss_a, loss_b)):
                continue
            if any(x < y for x, y in zip(win_a, win_b)):
                pairs.append((i, j, .05, 'winning_speed'))
            elif any(x > y for x, y in zip(loss_a, loss_b)):
                pairs.append((i, j, .02, 'losing_delay'))
    return pairs


def validate_pairs(pairs, q):
    pairs = tuple(tuple(p) for p in pairs)
    for i, j, weight, reason in pairs:
        if (not isinstance(i, (int, np.integer)) or not isinstance(j, (int, np.integer))
                or not 0 <= i < len(q) or not 0 <= j < len(q) or i == j
                or reason not in ('certain_lethal', 'winning_speed', 'losing_delay')
                or weight != {'certain_lethal': .2, 'winning_speed': .05, 'losing_delay': .02}.get(reason)
                or abs(float(q[i]-q[j])) > 1e-6):
            raise ValueError('Invalid auxiliary preference')
    return pairs
