"""Isolated Gumbel Q-scale experiment sharing official information-search helpers.

The legacy branch must match information_search.search numerically. Natural
WDL scale corresponds to Mctx rescale_values=False. Existing service and
training source hashes are untouched. No rewards, rule shortcuts or hand
visibility changes. Full-game/strength acceptance is still required.
"""
import time
import numpy as np
from . import information_search as core
from . import search_policy as legacy
from .search_stability import raw_completed_q


def completed_q(node, mode):
    if mode == 'legacy':
        return legacy.completed_q(node.prior, node.visits, node.total, node.raw_value)
    if mode != 'natural_wdl':
        raise ValueError('Unknown recovery Q scale')
    q = raw_completed_q(node.prior, node.visits, node.total, node.raw_value)
    if np.any(np.abs(q) > 1 + 1e-6):
        raise ValueError('Natural WDL search requires values in [-1,1]')
    return q * (.1 * (50 + node.visits.max(initial=0)))


def improved_policy(node, mode):
    return legacy.normalized_exp(np.log(np.maximum(node.prior, 1e-300)) + completed_q(node, mode))


def choose_root(node, noise, visit, mode):
    eligible = node.visits == visit
    if not eligible.any():
        raise ValueError('Invalid sequential-halving visit schedule')
    scores = np.log(np.maximum(node.prior, 1e-300)) + noise + completed_q(node, mode)
    return int(np.argmax(np.where(eligible, scores, -np.inf)))


def search(state, viewer, policies, *, seed, runtime, simulations=32, gumbel_candidates=16,
           terminal_horizon=3, max_depth=10, deadline=None, noise=False, q_scale='natural_wdl',
           fast_simulation=True,teacher_mode='network',value_policies=None,root_trace=False):
    """Bounded version of the existing Gumbel traversal with an explicit planner.

    Keep root worlds, opponent private observations, fixed observer values,
    short terminal proofs and the sequential-halving budget identical to the
    shared implementation. This branch has no optional shaped reward path.
    """
    if q_scale not in ('legacy', 'natural_wdl'):
        raise ValueError('Unknown recovery Q scale')
    if teacher_mode not in ('network','covered_terminal','covered_expert','covered_value') or (teacher_mode!='network' and q_scale=='legacy'):
        raise ValueError('Unknown or incompatible recovery teacher mode')
    if teacher_mode!='network':
        from .covered_rollout import rule_decision,terminal_leaf
    if teacher_mode=='covered_value':
        from .covered_value import expert_leaf
        if not value_policies or viewer not in value_policies:raise ValueError('Qualified covered value policies required')
    if (type(simulations) is not int or simulations < 1 or type(gumbel_candidates) is not int
            or gumbel_candidates < 1 or type(max_depth) is not int or max_depth < 1
            or type(terminal_horizon) is not int or terminal_horizon < 0):
        raise ValueError('Invalid bounded recovery search limits')
    if core.acting_side(state) != viewer:
        raise ValueError('Search must use decision owner')
    actions, (x, c) = runtime.decision(state, viewer)
    logits, root_value = runtime.predict_encoded(policies[viewer], x, c)
    if teacher_mode=='covered_value':root_value=runtime.value_for(state,viewer,value_policies)
    prior = core.search_prior(logits)
    rng = np.random.default_rng(seed)
    root = core.Node.create(prior, root_value); tree = {core.info_key(x, c): root}
    requested = 1 if len(actions) == 1 else simulations
    schedule = legacy.visit_schedule(min(len(actions), gumbel_candidates, requested), requested)
    gumbel = rng.gumbel(size=len(actions)) if noise else np.zeros(len(actions))
    depths = []; worlds = 0; trace = []
    terminal_checks = np.zeros(len(actions), dtype=np.int32)
    terminal_wins = np.zeros(len(actions), dtype=np.int32)
    for simulation in range(requested):
        if deadline is not None and time.time() >= deadline:
            break
        world = core.sample_world(state, viewer, seed + simulation * 104729, runtime=runtime)
        worlds += 1; path = []; value = 0.; finished_sim = True
        root_action = None; root_terminal_win = False
        for depth in range(max_depth):
            if deadline is not None and time.time() >= deadline:
                finished_sim = False; break
            if world['phase'] == 'finished':
                value = core.terminal_value(world, viewer); break
            actor = core.acting_side(world)
            if actor!=viewer and teacher_mode!='network':legal,chosen=rule_decision(world,actor)
            else:legal, (xx, cc) = runtime.decision(world, actor)
            if actor != viewer:
                if teacher_mode=='network':
                    scores = runtime.action_scores(policies[actor], xx, cc)
                    chosen = int(rng.choice(len(legal), p=core.softmax(scores)))
            else:
                key = core.info_key(xx, cc)
                if key not in tree:
                    scores, value = runtime.predict_encoded(policies[actor], xx, cc)
                    if teacher_mode=='covered_value':value=runtime.value_for(world,viewer,value_policies)
                    tree[key] = core.Node.create(core.search_prior(scores), value)
                    remaining = max_depth - depth
                    if teacher_mode=='covered_value':
                        value,finished_sim=expert_leaf(world,viewer,value_policies,runtime,
                            deadline if deadline is not None else float('inf'),horizon=terminal_horizon)
                    elif teacher_mode!='network':
                        value,finished_sim=terminal_leaf(world,viewer,policies,runtime,rng,
                            deadline if deadline is not None else float('inf'),own_rule=teacher_mode=='covered_expert')
                    elif terminal_horizon and remaining > 0:
                        proved = core.short_terminal_value(
                            world, viewer, policies, rng, scores=scores, legal=legal,
                            horizon=min(terminal_horizon, remaining), runtime=runtime, optimized=True,
                            fast_simulation=fast_simulation, deadline=deadline)
                        if proved is not None:
                            value = proved
                            # The proof is for this sampled hidden world. It
                            # is one rollout result, not a reusable belief-
                            # state estimate for every world sharing (x, c).
                            # Keep legacy behavior only for exact reproduction.
                            if q_scale == 'legacy':
                                tree[key].raw_value = proved
                    break
                node = tree[key]
                chosen = (choose_root(node, gumbel, schedule[simulation], q_scale) if depth == 0 else
                          int(np.argmax(improved_policy(node, q_scale) - node.visits / (1 + node.visits.sum()))))
                path.append((node, chosen))
            world = core.clean((core.simulate_action if fast_simulation else core.apply_action)(world, actor, legal[chosen]))
            if depth == 0:
                root_action = chosen
                root_terminal_win = world['phase'] == 'finished' and world.get('winner') == viewer
        else:
            if teacher_mode=='covered_value':
                value,finished_sim=expert_leaf(world,viewer,value_policies,runtime,
                    deadline if deadline is not None else float('inf'),horizon=terminal_horizon)
            elif teacher_mode!='network':
                value,finished_sim=terminal_leaf(world,viewer,policies,runtime,rng,
                    deadline if deadline is not None else float('inf'),own_rule=teacher_mode=='covered_expert')
            else:value = core.leaf_value(world, viewer, policies, runtime=runtime)
        if not finished_sim:
            break
        if not np.isfinite(value):
            raise ValueError('Non-finite recovery search value')
        if root_action is not None:
            terminal_checks[root_action] += 1; terminal_wins[root_action] += int(root_terminal_win)
        for node, chosen in path:
            node.visits[chosen] += 1; node.total[chosen] += value
        if root_trace and root_action is not None:
            trace.append(dict(simulation=simulation, considered_visit=schedule[simulation],
                              chosen=int(root_action), value=float(value),
                              visits=root.visits.copy().tolist(),
                              mean_q=np.divide(root.total, root.visits, out=np.zeros(len(actions)),
                                               where=root.visits > 0).tolist(),
                              completed_q=completed_q(root, q_scale).tolist()))
        depths.append(depth + 1)
    visits = root.visits.copy(); pi = improved_policy(root, q_scale)
    selected = choose_root(root, gumbel, int(visits.max()), q_scale)
    diagnostics = core._root_diagnostics(actions, prior, visits, root, selected, requested,
                                         len(depths), gumbel_candidates, 'gumbel')['root_diagnostics']
    diagnostics.update(completed_q=completed_q(root, q_scale),
                       completed_q_definition='recovery_search:' + q_scale, q_scale=q_scale)
    return dict(actions=actions, x=x, c=c, pi=pi.astype(np.float32), visits=visits,
                mean_values=np.divide(root.total, visits, out=np.zeros(len(visits)), where=visits > 0),
                visit_pi=visits / visits.sum() if visits.sum() else prior.copy(),
                root_prior=prior, root_gumbel=gumbel, value=root_value,
                terminal_checks=terminal_checks, terminal_wins=terminal_wins,
                complete=len(depths) == requested, simulations=len(depths), requested=requested,
                roots_covered=int((visits > 0).sum()), search_choice=selected,
                raw_choice=int(np.argmax(logits)), nodes=len(tree), worlds=worlds,
                max_depth=max(depths, default=0), policy_source='gumbel_q_' + q_scale +
                    (':'+teacher_mode if teacher_mode!='network' else ''),
                root_diagnostics=diagnostics, q_scale=q_scale,teacher_mode=teacher_mode,
                **({'root_trace': trace} if root_trace else {}),
                leaf_source='covered_expert_value_or_terminal' if teacher_mode=='covered_value' else
                    'real_terminal' if teacher_mode!='network' else 'network_or_short_proof')
