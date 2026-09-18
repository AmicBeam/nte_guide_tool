"""Experimental public-information tactical proof and paired branch verification.

No serving hook. Proof only covers deterministic actions within the current turn.
Unproved actions are UNKNOWN, never labelled as losses.
"""
from copy import deepcopy
from math import comb
import time
import numpy as np

from .league_schema import SEATS, rule_hash
from .league_rollout import decision, play

# Audited public-eight rules: during the acting player's non-end-turn actions,
# the defender cannot gain AP before responding. Free responses are checked
# separately against every response template of the public defending roster.
# A rule change disables this shortcut until reviewed again.
AP_ZERO_PROOF_RULE = '42e8840fe013df1d8b9e2b38a9e538e2fbcab0c674ca13979967c911ece9f085'


def proof_state(state, viewer):
    from app.modules.card_game.engine.duel_v2.entities import hydrate_entities
    from app.modules.card_game.content.duel_v2 import CARDS
    from app.modules.card_game.engine.duel_v2.flow import can_afford_card
    if state.get('phase') != 'playing' or state.get('pending_choice') or state.get('operation'):
        return None
    if any(cid not in SEATS for t in state['sides'].values() for cid in t['characters']):
        return None
    foe = 'b' if viewer == 'a' else 'a'
    team = state['sides'][foe]
    revealed = set(team.get('revealed_ids', []))
    unknown = any(c['instance_id'] not in revealed for c in team['hand'])
    private_marks_possible = 'xiaozhi' in team['characters'] and bool(team['hand'])
    if unknown or private_marks_possible:
        if team['ap'] != 0 or rule_hash() != AP_ZERO_PROOF_RULE:
            return None
        possible = [dict(c,card_id=cid) for cid,c in CARDS.items() if c.get('response') and c['character_id'] in team['characters']]
        if any(can_afford_card(state, foe, c) for c in possible):
            return None
    out = deepcopy(state)
    out['events'] = []
    out.pop('_public_board', None)
    out['rng'] = 1729
    # No real hidden identity, mark, order or RNG is consulted by the search.
    for side, t in out['sides'].items():
        t['deck'] = [dict(instance_id=f'proof-{side}-{i}', card_id='unknown',
                          character_id='unknown', type='tactic', cost=999)
                     for i in range(len(t['deck']))]
    if unknown or private_marks_possible:
        out['sides'][foe]['hand'] = [dict(instance_id=f'proof-hand-{i}', card_id='unknown',
                                         character_id='unknown', type='tactic', cost=999)
                                    for i in range(len(team['hand']))]
    hydrate_entities(out)
    return out


def safe_step(state, viewer, action):
    from app.modules.card_game.engine.duel_v2 import apply_action
    if action.get('type') not in ('attack', 'play_card', 'ultimate'):
        return None
    try:
        before = deepcopy(state)
        before['events'] = []
        after = apply_action(before, viewer, action)
    except (ValueError, KeyError, TypeError):
        return None
    if after.get('rng') != state.get('rng') or after.get('turn') != state.get('turn'):
        return None
    if after.get('phase') not in ('playing', 'finished'):
        return None
    for side in state['sides']:
        if after['sides'][side]['deck'] != state['sides'][side]['deck']:
            return None
    if any(e.get('type') in ('draw', 'empty_draw', 'shuffle') or
           e.get('reason') in ('empty_draw', 'empty_draw_win') for e in after.get('events', [])):
        return None
    after['events'] = []
    after.pop('_public_board', None)
    return after


def prove_turn(state, viewer, *, depth=2, node_budget=512, deadline=None):
    """Cover every root before deepening; return certified winning root indices.

    No labels are returned for roots without a proof. Deadline/budget exhaustion
    cannot become a failed-lethal label. Paths use one public information state.
    """
    if depth < 1 or node_budget < 1:
        raise ValueError('Positive search limits required')
    actions, _ = decision(state, viewer)
    base = proof_state(state, viewer)
    report = dict(paths={}, depths={}, nodes=0, root_actions=len(actions),
                  roots_visited=0, exhausted=False, eligible=base is not None)
    if base is None:
        return report
    frontier = []
    def expired():
        return report['nodes'] >= node_budget or (deadline is not None and time.time() >= deadline)
    def visit(parent, action, root, path):
        report['nodes'] += 1
        after = safe_step(parent, viewer, action)
        if after is None:
            return
        path = path + [action]
        if after.get('winner') == viewer:
            report['paths'][root] = path
            report['depths'][root] = len(path)
        elif after.get('phase') == 'playing' and len(path) < depth:
            frontier.append((after, root, path))
    for i, action in enumerate(actions):
        if expired():
            report['exhausted'] = True
            break
        report['roots_visited'] += 1
        visit(base, action, i, [])
    cursor = 0
    while cursor < len(frontier):
        parent, root, path = frontier[cursor]
        cursor += 1
        if root in report['paths']:
            continue
        for action in decision(parent, viewer)[0]:
            if expired():
                report['exhausted'] = True
                return report
            visit(parent, action, root, path)
            if root in report['paths']:
                break
    return report


def paired_evidence(candidate, incumbent, alpha=.05):
    """Fixed-size independent confirmation, not optional stopping on p-values.

    Exact one-sided sign test when nonzero paired reward gaps have equal size.
    Mixed draw/win gap sizes are rejected rather than misusing that test.
    """
    if len(candidate) != len(incumbent) or len(candidate) < 16:
        return {'accepted': False, 'reason': 'insufficient_pairs'}
    if not all(g['complete'] for g in [*candidate, *incumbent]):
        return {'accepted': False, 'reason': 'incomplete'}
    d = np.array([a['reward']-b['reward'] for a,b in zip(candidate, incumbent)])
    sizes = set(abs(float(x)) for x in d if x)
    if len(sizes) != 1:
        return {'accepted': False, 'reason': 'ties_or_unequal_gap_sizes'}
    better = int((d > 0).sum()); worse = int((d < 0).sum()); n = better+worse
    p = sum(comb(n,k) for k in range(better,n+1)) / 2**n
    return dict(accepted=bool(p <= alpha and d.mean() > 0), p=p,
                better=better, worse=worse, pairs=len(d), mean_gap=float(d.mean()))


def compare_root(state, viewer, policies, scores, *, seed, deadline, map_jobs=map):
    """All actions screened on 8 worlds, ONE challenger confirmed on fresh 32.

    Continuations use public two-action lethal search before the frozen policy.
    Samples never see the real opposing hidden hand or deck order.
    """
    actions, (x,c) = decision(state, viewer)
    incumbent = int(np.argmax(scores))
    proof = prove_turn(state, viewer, deadline=deadline)
    if proof['paths']:
        shortest = min(proof['depths'].values())
        targets = [i for i,d in proof['depths'].items() if d == shortest]
        return dict(x=x,c=c,targets=targets,kind='proof',proof=proof,games=0)
    def evaluate(indices, particles, offset):
        jobs=[dict(root=state,viewer=viewer,action=actions[i],policies=policies,
                   seed=seed+offset+j,deadline=deadline,sample=bool(j%2),
                   temperature=1.5 if j%2 else 1.,tactical_search=True)
              for i in indices for j in range(particles)]
        games=list(map_jobs(play,jobs))
        return [games[i*particles:(i+1)*particles] for i in range(len(indices))]
    batches=evaluate(list(range(len(actions))),8,0)
    if not all(g['complete'] for b in batches for g in b):
        return dict(targets=[],kind='incomplete_screen',games=len(actions)*8)
    q=[np.mean([g['reward'] for g in b]) for b in batches]
    challenger=int(np.argmax(q))
    if challenger == incumbent:
        return dict(targets=[],kind='no_disagreement',games=len(actions)*8)
    a,b=evaluate([challenger,incumbent],32,100000)
    evidence=paired_evidence(a,b)
    return dict(x=x,c=c,targets=[challenger] if evidence['accepted'] else [],
                incumbent=incumbent,kind='paired',evidence=evidence,games=len(actions)*8+64)


def distill_step(net, opt, samples):
    """Teach the searched winning set over FULL legal candidates, one update."""
    import torch
    from .league_learning import tensors
    if not samples:
        return 0.
    x,c,m=tensors([(s['x'],s['c']) for s in samples],next(net.parameters()).device)
    logits,_=net(x,c,m)
    loss=torch.stack([torch.logsumexp(logits[i,:len(s['c'])],0)-
                      torch.logsumexp(logits[i,s['targets']],0)
                      for i,s in enumerate(samples)]).mean()
    if not torch.isfinite(loss):
        raise ValueError('Non-finite distillation')
    opt.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),.5,error_if_nonfinite=True)
    opt.step()
    return float(loss.detach())
