"""Offline official-Mctx comparison for DeviceGumbelSearch-comparable numerics.

This harness consumes saved recurrent tables or a complete recorded tree. It
is not a JAX Mctx replacement, not hidden-info information-set search, and
not real-game CUDA throughput. Official Mctx is an optional lazy dependency;
missing JAX/Mctx is an explicit skip, never a pass.

Comparable DeviceGumbelSearch axes, when records are perfect-info and use the
natural-WDL backup contract: root sequential-halving, completed-Q without
min-max rescale, full improved policy labels, and max-visit final selection.
Exact equality requires explicit fixed Gumbel data. A different PRNG is
statistical only. This module never installs packages.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import time
from pathlib import Path

import numpy as np

from ..search_policy import visit_schedule


COMPARABLE_AXES = (
    'root_sequential_halving',
    'completed_q_natural_wdl',
    'improved_policy_labels',
    'final_max_visit_selection',
)
NOT_CLAIMED = (
    'hidden_info_search',
    'perfect_info_equivalence_to_infoset',
    'cuda_throughput',
    'website_effects',
    'invented_mctx_replacement',
)
EXACT_GUMBEL_REQUIRED = 'fixed_gumbel_required'
TIMING_SCOPE = 'host numeric replay and optional JAX CPU; not CUDA search throughput'
REMAINING_DEVICE_INTEGRATION = (
    'Export per-root DeviceGumbelSearch tensors: prior or logits, fixed-viewer value, legal mapping, reward/discount, game/action owner, seed, and Gumbel when exactness is claimed.',
    'Optional device_visits/device_policy/device_action fields are the actual-device hook; this harness does not construct DeviceGumbelSearch or capture CUDA graphs.',
    'Do not treat official Mctx perfect-info replay as information-set world sampling, opponent-policy stepping, or seven-team rule qualification.',
    'Do not treat host or JAX CPU timings as 8/32/128/400/1600 device capacity or real-game CUDA throughput.',
    'Mismatch detection is axis-level (visits, completed-Q, policy, final action). Averaging disagreements into a single score is not a pass.',
)


@dataclass(frozen=True)
class OfficialMctxStatus:
    jax_available: bool
    mctx_available: bool
    available: bool
    jax_version: str | None
    mctx_version: str | None
    skip_reason: str | None
    install_attempted: bool = False


def probe_official_mctx(*, load=True):
    """Lazy availability probe. Never pip-installs JAX or Mctx."""
    if not load:
        from importlib.util import find_spec
        from importlib.metadata import version, PackageNotFoundError
        jax_present, mctx_present = find_spec('jax') is not None, find_spec('mctx') is not None
        def installed_version(name, present):
            if not present:
                return None
            try:
                return version(name)
            except PackageNotFoundError:
                return 'unknown'
        return OfficialMctxStatus(jax_present, mctx_present, jax_present and mctx_present,
            installed_version('jax', jax_present), installed_version('mctx', mctx_present),
            None if jax_present and mctx_present else 'optional_dependencies_not_installed', False)
    try:
        import jax
    except Exception as exc:
        return OfficialMctxStatus(False, False, False, None, None,
                                 'jax_unavailable:' + type(exc).__name__, False)
    jax_version = str(getattr(jax, '__version__', 'unknown'))
    try:
        import mctx
    except Exception as exc:
        return OfficialMctxStatus(True, False, False, jax_version, None,
                                 'mctx_unavailable:' + type(exc).__name__, False)
    mctx_version = str(getattr(mctx, '__version__', 'unknown'))
    required = ('gumbel_muzero_policy', 'RootFnOutput', 'RecurrentFnOutput',
                'qtransform_completed_by_mix_value')
    missing = [name for name in required if not hasattr(mctx, name)]
    if missing:
        return OfficialMctxStatus(True, True, False, jax_version, mctx_version,
                                 'mctx_missing_public_api:' + ','.join(missing), False)
    return OfficialMctxStatus(True, True, True, jax_version, mctx_version, None, False)


def mctx_sequence_of_considered_visits(max_num_considered_actions, num_simulations):
    """Published Mctx Sequential Halving targets (mctx/_src/seq_halving.py)."""
    considered = int(max_num_considered_actions)
    budget = int(num_simulations)
    if considered < 0 or budget < 0:
        raise ValueError('Nonnegative sequential-halving sizes required')
    if considered <= 1:
        return list(range(budget))
    phases = int(math.ceil(math.log2(considered)))
    sequence = []
    visits = [0] * considered
    active = considered
    while len(sequence) < budget:
        extra = max(1, budget // (phases * active))
        for _ in range(extra):
            sequence.extend(visits[:active])
            for index in range(active):
                visits[index] += 1
        active = max(2, active // 2)
    return sequence[:budget]


def sequential_halving_schedule(n_legal, simulations, gumbel_candidates):
    """DeviceGumbelSearch root visit targets: min(legal, cap, sims) considered."""
    considered = min(int(n_legal), int(gumbel_candidates), int(simulations))
    if considered < 1 or int(simulations) < 1:
        raise ValueError('Positive sequential-halving considered set required')
    return list(visit_schedule(considered, int(simulations)))


def _legal_vectors(prior, visits, total, legal):
    prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    visits = np.asarray(visits, dtype=np.int64).reshape(-1)
    total = np.asarray(total, dtype=np.float64).reshape(-1)
    legal = np.asarray(legal, dtype=bool).reshape(-1)
    if prior.shape != visits.shape or total.shape != prior.shape or legal.shape != prior.shape:
        raise ValueError('prior/visits/total/legal must share a complete action axis')
    if not legal.any():
        raise ValueError('At least one legal action required')
    return prior, visits, total, legal


def natural_completed_q(prior, visits, total, value, legal):
    """DeviceStatisticsPool natural_wdl completed-Q, including unvisited mix."""
    prior, visits, total, legal = _legal_vectors(prior, visits, total, legal)
    if (not np.isfinite(prior).all() or (prior < 0).any()
            or not np.isfinite(total).all() or not np.isfinite(value)
            or (visits < 0).any()):
        raise ValueError('Finite nonnegative search statistics required')
    if np.any(prior[~legal] != 0.0):
        raise ValueError('Illegal actions must have zero prior mass')
    seen = (visits > 0) & legal
    q = np.divide(total, np.maximum(visits, 1), dtype=np.float64)
    mass = float(np.sum(prior * seen))
    weighted = float(np.sum(prior * seen * q))
    mean = weighted / mass if mass > 0 else float(value)
    visit_sum = float(np.sum(np.where(legal, visits, 0)))
    mixed = (float(value) + visit_sum * mean) / (1.0 + visit_sum)
    q = np.where(seen, q, mixed)
    if np.any(((np.abs(q) > 1.0 + 1e-6) | ~np.isfinite(q)) & legal):
        raise ValueError('Natural WDL completed-Q requires values in [-1, 1]')
    gain = 0.1 * (50.0 + float(np.max(np.where(legal, visits, 0))))
    return np.where(legal, q * gain, 0.0)


def improved_policy(prior, visits, total, value, legal):
    """Full legal improved policy; no top-k truncation."""
    prior, visits, total, legal = _legal_vectors(prior, visits, total, legal)
    q = natural_completed_q(prior, visits, total, value, legal)
    logits = np.log(np.maximum(prior, 1e-300)) + q
    logits = np.where(legal, logits, -np.inf)
    shift = np.max(logits[legal])
    exp = np.where(legal, np.exp(logits - shift), 0.0)
    denom = float(exp.sum())
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError('Improved policy is not finite')
    return exp / denom


def _argmax_masked(score, eligible):
    masked = np.where(eligible, score, -np.inf)
    if not np.isfinite(masked).any():
        raise ValueError('No eligible action for Gumbel selection')
    return int(np.argmax(masked))


def select_root(prior, visits, total, value, legal, noise, visit_target):
    prior, visits, total, legal = _legal_vectors(prior, visits, total, legal)
    noise = np.asarray(noise, dtype=np.float64).reshape(-1)
    if noise.shape != prior.shape or not np.isfinite(noise).all():
        raise ValueError('Finite Gumbel noise matching the action axis required')
    if int(visit_target) < 0:
        raise ValueError('Nonnegative sequential-halving visit target required')
    q = natural_completed_q(prior, visits, total, value, legal)
    eligible = (visits == int(visit_target)) & legal
    if not eligible.any():
        raise ValueError('Invalid sequential-halving visit schedule')
    score = np.log(np.maximum(prior, 1e-300)) + noise + q
    return _argmax_masked(score, eligible)


def select_interior(prior, visits, total, value, legal):
    prior, visits, total, legal = _legal_vectors(prior, visits, total, legal)
    policy = improved_policy(prior, visits, total, value, legal)
    visit_sum = float(np.sum(np.where(legal, visits, 0)))
    scores = policy - visits.astype(np.float64) / (1.0 + visit_sum)
    return _argmax_masked(scores, legal)


def select_final(prior, visits, total, value, legal, noise):
    prior, visits, total, legal = _legal_vectors(prior, visits, total, legal)
    noise = np.asarray(noise, dtype=np.float64).reshape(-1)
    if noise.shape != prior.shape or not np.isfinite(noise).all():
        raise ValueError('Finite Gumbel noise matching the action axis required')
    q = natural_completed_q(prior, visits, total, value, legal)
    max_visits = int(np.max(np.where(legal, visits, 0)))
    eligible = (visits == max_visits) & legal
    if not eligible.any():
        raise ValueError('Final selection requires a visited legal action')
    score = np.log(np.maximum(prior, 1e-300)) + noise + q
    return _argmax_masked(score, eligible)


def prior_from_logits(logits, legal):
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    legal = np.asarray(legal, dtype=bool).reshape(-1)
    masked = np.where(legal, logits, -np.inf)
    shift = np.max(masked[legal])
    exp = np.where(legal, np.exp(masked - shift), 0.0)
    denom = float(exp.sum())
    if denom <= 0:
        raise ValueError('Legal logits must produce a prior')
    return exp / denom


def logits_from_prior(prior, legal):
    prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    legal = np.asarray(legal, dtype=bool).reshape(-1)
    return np.where(legal, np.log(np.maximum(prior, 1e-300)), -1e9)


def _as_bool_mask(values, width, name):
    mask = np.asarray(values, dtype=bool).reshape(-1)
    if mask.size > width:
        raise ValueError(name + ' exceeds max_actions')
    out = np.zeros(width, dtype=bool)
    out[:mask.size] = mask
    return out


def _pad(values, width, dtype, name, fill=0):
    array = np.asarray(values, dtype=dtype).reshape(-1)
    if array.size > width:
        raise ValueError(name + ' exceeds max_actions')
    out = np.full(width, fill, dtype=dtype)
    out[:array.size] = array
    return out


def exact_claim_precondition(root):
    """Reject exact equality unless Gumbel values are explicitly stored."""
    if not isinstance(root, dict):
        return EXACT_GUMBEL_REQUIRED
    noise = root.get('gumbel')
    if noise is None:
        return EXACT_GUMBEL_REQUIRED
    noise = np.asarray(noise, dtype=np.float64)
    if noise.ndim != 1 or noise.size < 1 or not np.isfinite(noise).all():
        return EXACT_GUMBEL_REQUIRED
    return None


def _reward_contract_reason(reward, discount, value, terminal):
    reward = float(reward)
    discount = float(discount)
    value = float(value)
    if terminal:
        if discount != 0.0:
            return 'terminal_discount_must_be_zero'
        if abs(value) > 1e-12:
            return 'terminal_value_must_be_zero'
        if reward not in (-1.0, 0.0, 1.0):
            return 'terminal_reward_must_be_wdl'
        return None
    if reward != 0.0 or discount != 1.0:
        return 'process_rewards_are_not_device_gumbel_backup'
    if not np.isfinite(value) or abs(value) > 1.0 + 1e-6:
        return 'nonterminal_value_outside_wdl'
    return None


def load_records(source):
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding='utf8'))
    elif isinstance(source, dict):
        payload = json.loads(json.dumps(source))
    else:
        raise TypeError('JSON path or mapping of numeric search records required')
    if not isinstance(payload, dict) or payload.get('kind') not in (
            'recurrent_records', 'recorded_tree'):
        raise ValueError('kind must be recurrent_records or recorded_tree')
    payload.setdefault('information_model', 'perfect_info_numeric_replay')
    payload.setdefault('throughput_claim', 'offline_cpu_numeric')
    if payload['information_model'] != 'perfect_info_numeric_replay':
        raise ValueError('Only perfect_info_numeric_replay records are accepted')
    if payload['throughput_claim'] != 'offline_cpu_numeric':
        raise ValueError('Records must not claim CUDA throughput')
    roots = payload.get('roots')
    if not isinstance(roots, list) or not roots:
        raise ValueError('Nonempty roots required')
    payload['roots'] = [dict(root) for root in roots]
    return payload


def _normalize_root(root, *, simulations, gumbel_candidates, max_depth, max_actions):
    if not isinstance(root, dict):
        raise ValueError('Each root must be a mapping')
    viewer = int(root.get('viewer', 0))
    game_owner = int(root.get('game_owner', viewer))
    action_owner = int(root.get('action_owner', viewer))
    legal = _as_bool_mask(root.get('legal', [True]), max_actions, 'legal')
    if 'prior_logits' in root:
        logits = _pad(root['prior_logits'], max_actions, np.float64, 'prior_logits', -1e9)
        logits = np.where(legal, logits, -1e9)
        prior = prior_from_logits(logits, legal)
    elif 'prior' in root:
        prior = _pad(root['prior'], max_actions, np.float64, 'prior')
        prior = np.where(legal, prior, 0.0)
        mass = float(prior.sum())
        if mass <= 0:
            raise ValueError('Legal prior mass required')
        prior = prior / mass
        logits = logits_from_prior(prior, legal)
    else:
        raise ValueError('Root prior or prior_logits required')
    value = float(root.get('value', 0.0))
    if not np.isfinite(value) or abs(value) > 1.0 + 1e-6:
        raise ValueError('Finite fixed-viewer root value in [-1, 1] required')
    gumbel = root.get('gumbel')
    if gumbel is not None:
        gumbel = _pad(gumbel, max_actions, np.float64, 'gumbel')
        gumbel = np.where(legal, gumbel, 0.0)
    transitions = {}
    raw_transitions = root.get('transitions') or {}
    if not isinstance(raw_transitions, dict):
        raise ValueError('transitions must be a mapping')
    for key, payload in raw_transitions.items():
        if not (isinstance(key, str) and ':' in key):
            raise ValueError('Transition keys must be "node:action"')
        node_s, action_s = key.split(':', 1)
        node_id, action = int(node_s), int(action_s)
        if not isinstance(payload, dict):
            raise ValueError('Each transition must be a mapping')
        child_legal = _as_bool_mask(payload.get('legal', [True]), max_actions, 'child legal')
        if 'prior_logits' in payload:
            child_logits = _pad(payload['prior_logits'], max_actions, np.float64, 'child logits', -1e9)
            child_logits = np.where(child_legal, child_logits, -1e9)
            child_prior = prior_from_logits(child_logits, child_legal)
        elif 'prior' in payload:
            child_prior = _pad(payload.get('prior', [1.0]), max_actions, np.float64, 'child prior')
            child_prior = np.where(child_legal, child_prior, 0.0)
            mass = float(child_prior.sum())
            child_prior = child_prior / mass if mass > 0 else child_prior
            child_logits = logits_from_prior(child_prior, child_legal)
        else:
            child_legal = np.zeros(max_actions, dtype=bool)
            child_legal[0] = True
            child_prior = child_legal.astype(np.float64)
            child_logits = logits_from_prior(child_prior, child_legal)
        terminal = float(payload.get('discount', 1.0)) == 0.0
        child = dict(
            child_id=int(payload.get('child_id', -1)),
            reward=float(payload.get('reward', 0.0)),
            discount=float(payload.get('discount', 1.0)),
            value=float(payload.get('value', 0.0)),
            prior=child_prior, prior_logits=child_logits, legal=child_legal,
            actor=int(payload.get('actor', payload.get('action_owner', action_owner))),
            game_owner=int(payload.get('game_owner', game_owner)),
            terminal=terminal,
        )
        reason = _reward_contract_reason(child['reward'], child['discount'], child['value'], terminal)
        if reason:
            child['unsupported'] = reason
        transitions[(node_id, action)] = child
    visits = None if root.get('visits') is None else _pad(root['visits'], max_actions, np.int64, 'visits')
    total = None if root.get('total') is None else _pad(root['total'], max_actions, np.float64, 'total')
    device_visits = None if root.get('device_visits') is None else _pad(
        root['device_visits'], max_actions, np.int64, 'device_visits')
    device_policy = None if root.get('device_policy') is None else _pad(
        root['device_policy'], max_actions, np.float64, 'device_policy')
    device_action = None if root.get('device_action') is None else int(root['device_action'])
    return dict(
        root_id=int(root.get('root_id', 0)), seed=int(root.get('seed', 0)),
        viewer=viewer, game_owner=game_owner, action_owner=action_owner,
        prior=prior, prior_logits=logits, value=value, legal=legal, gumbel=gumbel,
        transitions=transitions, visits=visits, total=total,
        device_visits=device_visits, device_policy=device_policy, device_action=device_action,
        simulations=int(root.get('simulations', simulations)),
        gumbel_candidates=int(root.get('gumbel_candidates', gumbel_candidates)),
        max_depth=int(root.get('max_depth', max_depth)), max_actions=max_actions,
    )


def _root_unsupported(root):
    reasons = []
    if root['viewer'] != root['action_owner']:
        reasons.append('root_action_owner_is_not_fixed_viewer')
    if root['game_owner'] not in (root['viewer'], root['action_owner']):
        reasons.append('game_owner_mismatch')
    for child in root['transitions'].values():
        if child.get('unsupported'):
            reasons.append(child['unsupported'])
        if child['actor'] != root['viewer']:
            reasons.append('alternating_player_tree_is_not_infoset_search')
    return tuple(dict.fromkeys(reasons))


def replay_device_gumbel_semantics(root, *, gumbel=None):
    """Numpy replay of DeviceGumbelSearch network_value formulas on a table."""
    start = time.monotonic()
    noise = root['gumbel'] if gumbel is None else np.asarray(gumbel, dtype=np.float64)
    if noise is None:
        raise ValueError(EXACT_GUMBEL_REQUIRED)
    if noise.shape != root['prior'].shape:
        noise = _pad(noise, root['max_actions'], np.float64, 'gumbel')
        noise = np.where(root['legal'], noise, 0.0)
    unsupported = _root_unsupported(root)
    if unsupported:
        return dict(status='unsupported', unsupported=list(unsupported),
                    seconds=time.monotonic() - start)
    n_legal = int(root['legal'].sum())
    requested = 1 if n_legal == 1 else int(root['simulations'])
    schedule = sequential_halving_schedule(n_legal, requested, root['gumbel_candidates'])
    nodes = {
        root['root_id']: dict(
            prior=root['prior'].copy(),
            visits=np.zeros(root['max_actions'], dtype=np.int64),
            total=np.zeros(root['max_actions'], dtype=np.float64),
            value=float(root['value']), legal=root['legal'].copy(), terminal=False,
        )
    }
    for sim in range(requested):
        node_id = root['root_id']
        path = []
        depth = 0
        leaf = None
        while True:
            node = nodes[node_id]
            if node['terminal'] or depth >= int(root['max_depth']):
                leaf = float(node['value'])
                break
            if depth == 0:
                action = select_root(node['prior'], node['visits'], node['total'],
                                     node['value'], node['legal'], noise, schedule[sim])
            else:
                action = select_interior(node['prior'], node['visits'], node['total'],
                                         node['value'], node['legal'])
            if (node_id, action) not in root['transitions']:
                raise ValueError('Missing recurrent transition for node %s action %s' % (
                    node_id, action))
            path.append((node_id, action))
            child = root['transitions'][(node_id, action)]
            depth += 1
            if child['terminal']:
                leaf = float(child['reward'])
                break
            child_id = child['child_id']
            if child_id not in nodes:
                nodes[child_id] = dict(
                    prior=child['prior'].copy(),
                    visits=np.zeros(root['max_actions'], dtype=np.int64),
                    total=np.zeros(root['max_actions'], dtype=np.float64),
                    value=float(child['value']), legal=child['legal'].copy(), terminal=False,
                )
                leaf = float(child['value'])
                break
            node_id = child_id
        if leaf is None or not np.isfinite(leaf):
            raise ValueError('Non-finite DeviceGumbelSearch backup')
        for nid, action in path:
            nodes[nid]['visits'][action] += 1
            nodes[nid]['total'][action] += leaf
    root_node = nodes[root['root_id']]
    policy = improved_policy(root_node['prior'], root_node['visits'], root_node['total'],
                             root_node['value'], root_node['legal'])
    completed = natural_completed_q(root_node['prior'], root_node['visits'], root_node['total'],
                                    root_node['value'], root_node['legal'])
    action = select_final(root_node['prior'], root_node['visits'], root_node['total'],
                          root_node['value'], root_node['legal'], noise)
    visit_sum = float(root_node['visits'].sum())
    visit_policy = root_node['visits'] / visit_sum if visit_sum else root_node['prior']
    return dict(
        status='ran', unsupported=[], visits=root_node['visits'], total=root_node['total'],
        policy=policy, visit_policy=visit_policy.astype(np.float64), completed_q=completed,
        action=action, schedule=schedule, requested=requested, completed_simulations=requested,
        seconds=time.monotonic() - start,
    )


def recorded_tree_semantics(root, *, gumbel=None):
    """Apply DeviceGumbelSearch formulas to an already completed root snapshot."""
    start = time.monotonic()
    if root['visits'] is None or root['total'] is None:
        raise ValueError('recorded_tree roots require visits and total')
    noise = root['gumbel'] if gumbel is None else np.asarray(gumbel, dtype=np.float64)
    if noise is None:
        raise ValueError(EXACT_GUMBEL_REQUIRED)
    policy = improved_policy(root['prior'], root['visits'], root['total'],
                             root['value'], root['legal'])
    completed = natural_completed_q(root['prior'], root['visits'], root['total'],
                                    root['value'], root['legal'])
    action = select_final(root['prior'], root['visits'], root['total'],
                          root['value'], root['legal'], noise)
    n_legal = int(root['legal'].sum())
    requested = 1 if n_legal == 1 else int(root['simulations'])
    schedule = sequential_halving_schedule(n_legal, requested, root['gumbel_candidates'])
    visit_sum = float(root['visits'].sum())
    return dict(
        status='ran', unsupported=[], visits=root['visits'].copy(), total=root['total'].copy(),
        policy=policy, visit_policy=root['visits'] / visit_sum if visit_sum else root['prior'],
        completed_q=completed, action=action, schedule=schedule, requested=requested,
        completed_simulations=int(visit_sum), seconds=time.monotonic() - start,
    )


def compare_torch_statistics(root, local):
    """Exercise the REAL statistics class in its explicit CPU oracle mode.

    Reconstruct a completed root via public writes/backups, then compare Q,
    policy and final selection. This does not execute the device search FSM
    or qualify CUDA, but avoids validating only a second NumPy formula.
    """
    import torch
    from .device_statistics import DeviceStatisticsPool
    started = time.monotonic()
    pool = DeviceStatisticsPool(1, root['max_actions'], device='cpu',
        allow_cpu_reference=True, max_batch=1, max_path_len=1)
    slots = torch.tensor([0], dtype=torch.int64)
    owners = torch.tensor([root['viewer']], dtype=torch.int64)
    written = pool.write_nodes(slots, owners,
        torch.tensor(root['prior'][None], dtype=torch.float64),
        torch.tensor([root['value']], dtype=torch.float64),
        torch.tensor(root['legal'][None], dtype=torch.bool))
    if written.error.any():
        raise ValueError('Actual Torch statistics rejected recorded root')
    for action, count in enumerate(local['visits']):
        value = float(local['total'][action] / count) if count else 0.
        for _ in range(int(count)):
            receipt = pool.backup(slots[:, None], written.generations[:, None],
                torch.tensor([[action]], dtype=torch.int64),
                torch.ones((1, 1), dtype=torch.bool), torch.tensor([value], dtype=torch.float64), owners)
            if receipt.error.any():
                raise ValueError('Actual Torch statistics rejected recorded backup')
    q = pool.completed_q(slots, written.generations)
    pi = pool.improved_policy(slots, written.generations)
    choice = pool.select_final(slots, written.generations,
        torch.tensor(root['gumbel'][None], dtype=torch.float64))
    snapshot = pool.snapshot(slots, written.generations)
    if q.error.any() or pi.error.any() or choice.error.any() or snapshot.error.any():
        raise ValueError('Actual Torch statistics rejected recorded query')
    actual = dict(visits=snapshot.visits[0].numpy(), completed_q=q.completed_q[0].numpy(),
                  policy=pi.policy[0].numpy(), action=int(choice.actions[0]))
    mismatch = _compare_outputs(local, actual, atol=1e-12, rtol=1e-12, require_all=True)
    return dict(status='ran', passed=not mismatch, mismatches=mismatch,
                backend='DeviceStatisticsPool_cpu_reference', cuda_qualified=False,
                seconds=time.monotonic()-started)


@contextmanager
def _inject_fixed_gumbel(jax, mctx, values):
    import inspect
    modules = [jax.random]
    host = inspect.getmodule(mctx.gumbel_muzero_policy)
    if host is not None and hasattr(getattr(host, 'jax', None), 'random'):
        modules.append(host.jax.random)
    values = np.asarray(values, dtype=np.float64)

    def fake(key, shape=(), dtype=None, **kwargs):
        dtype = dtype or jax.numpy.float32
        arr = jax.numpy.asarray(values, dtype=dtype)
        shape = tuple(shape)
        if not shape:
            return arr
        if tuple(arr.shape) == shape:
            return arr
        if arr.ndim == 1 and shape[-1] == arr.shape[0]:
            return jax.numpy.broadcast_to(arr, shape)
        raise ValueError('Fixed Gumbel shape %s does not match requested %s' % (
            tuple(arr.shape), shape))

    patched = [(module, module.gumbel) for module in modules]
    for module in modules:
        module.gumbel = fake
    try:
        yield
    finally:
        for module, fn in patched:
            module.gumbel = fn


def _pack_transition_tables(root):
    ids = {root['root_id']}
    for (node_id, _action), child in root['transitions'].items():
        ids.add(node_id)
        if child['child_id'] >= 0:
            ids.add(child['child_id'])
    width = int(root['max_actions'])
    nodes = sorted(ids)
    index_of = {node: i for i, node in enumerate(nodes)}
    count = len(nodes)
    child_index = np.full((count, width), -1, dtype=np.int32)
    reward = np.zeros((count, width), dtype=np.float32)
    discount = np.zeros((count, width), dtype=np.float32)
    child_logits = np.full((count, width, width), -1e9, dtype=np.float32)
    child_value = np.zeros((count, width), dtype=np.float32)
    for (node_id, action), child in root['transitions'].items():
        row = index_of[node_id]
        if not (0 <= action < width):
            continue
        reward[row, action] = child['reward']
        discount[row, action] = child['discount']
        child_value[row, action] = child['value']
        child_logits[row, action] = child['prior_logits'].astype(np.float32)
        if child['child_id'] >= 0:
            child_index[row, action] = index_of[child['child_id']]
    return dict(index_of=index_of, child_index=child_index, reward=reward,
                discount=discount, child_logits=child_logits, child_value=child_value)


def _call_official_qtransform(mctx, tree, n_actions):
    import functools
    import jax
    import jax.numpy as jnp
    qfn = functools.partial(mctx.qtransform_completed_by_mix_value,
                            rescale_values=False, value_scale=0.1, maxvisit_init=50.0)
    # The public qtransform contract takes an UNBATCHED tree and a scalar
    # node index. The public policy output above is a one-row batched tree.
    unbatched = jax.tree_util.tree_map(lambda value: value[0], tree)
    return np.asarray(qfn(unbatched, jnp.asarray(0, dtype=jnp.int32)),
                      dtype=np.float64).reshape(n_actions)


def run_official_mctx(root, *, exact=False, gumbel=None):
    """Run public mctx.gumbel_muzero_policy on a recorded perfect-info table."""
    start = time.monotonic()
    status = probe_official_mctx()
    base = dict(jax_available=status.jax_available, mctx_available=status.mctx_available,
                jax_version=status.jax_version, mctx_version=status.mctx_version,
                install_attempted=False, passed=False)
    if not status.available:
        return dict(base, status='skipped', skip_reason=status.skip_reason,
                    seconds=time.monotonic() - start)
    unsupported = list(_root_unsupported(root))
    if not root['transitions']:
        unsupported.append('recorded_tree_has_no_recurrent_fn')
    noise = root['gumbel'] if gumbel is None else np.asarray(gumbel, dtype=np.float64)
    if exact:
        reject = exact_claim_precondition({'gumbel': noise})
        if reject:
            return dict(base, status='rejected', skip_reason=reject,
                        seconds=time.monotonic() - start)
    if unsupported:
        return dict(base, status='unsupported', skip_reason=','.join(unsupported),
                    unsupported=unsupported, seconds=time.monotonic() - start)
    import functools
    import jax
    import jax.numpy as jnp
    import mctx
    tables = _pack_transition_tables(root)
    child_index = jnp.asarray(tables['child_index'])
    reward = jnp.asarray(tables['reward'])
    discount = jnp.asarray(tables['discount'])
    child_logits = jnp.asarray(tables['child_logits'])
    child_value = jnp.asarray(tables['child_value'])
    width = int(root['max_actions'])

    def recurrent_fn(params, rng_key, action, embedding):
        del params, rng_key
        node = embedding.astype(jnp.int32)
        act = action.astype(jnp.int32)
        out = mctx.RecurrentFnOutput(
            reward=reward[node, act], discount=discount[node, act],
            prior_logits=child_logits[node, act], value=child_value[node, act])
        nxt = child_index[node, act]
        return out, jnp.where(nxt < 0, node, nxt)

    invalid = np.logical_not(root['legal']).astype(np.float32)[None, :]
    qtransform = functools.partial(mctx.qtransform_completed_by_mix_value,
                                   rescale_values=False, value_scale=0.1, maxvisit_init=50.0)
    kwargs = dict(
        params=(),
        rng_key=jax.random.PRNGKey(int(root['seed']) % (2 ** 31 - 1)),
        root=mctx.RootFnOutput(
            prior_logits=jnp.asarray(root['prior_logits'].astype(np.float32)[None, :]),
            value=jnp.asarray([root['value']], dtype=np.float32),
            embedding=jnp.asarray([tables['index_of'][root['root_id']]], dtype=jnp.int32)),
        recurrent_fn=recurrent_fn,
        num_simulations=int(root['simulations'] if int(root['legal'].sum()) > 1 else 1),
        invalid_actions=jnp.asarray(invalid),
        max_depth=int(root['max_depth']),
        qtransform=qtransform,
        max_num_considered_actions=int(root['gumbel_candidates']),
    )
    try:
        if exact and noise is not None and np.allclose(noise[root['legal']], 0.0):
            output = mctx.gumbel_muzero_policy(**kwargs, gumbel_scale=0.0)
        elif exact and noise is not None:
            with _inject_fixed_gumbel(jax, mctx, noise):
                output = mctx.gumbel_muzero_policy(**kwargs, gumbel_scale=1.0)
        else:
            output = mctx.gumbel_muzero_policy(**kwargs, gumbel_scale=1.0)
    except Exception as exc:
        return dict(base, status='unsupported', skip_reason='official_mctx_runtime:' + type(exc).__name__,
                    error=str(exc), seconds=time.monotonic() - start)
    tree = output.search_tree
    try:
        visits = np.asarray(tree.children_visits)[0, 0, :width].astype(np.int64)
    except Exception:
        visits = None
    policy = np.asarray(output.action_weights, dtype=np.float64).reshape(-1, width)[0]
    action = int(np.asarray(output.action).reshape(-1)[0])
    completed = None
    q_reason = None
    try:
        completed = _call_official_qtransform(mctx, tree, width)
    except Exception as exc:
        q_reason = 'official_qtransform:' + type(exc).__name__
    return dict(
        base, status='ran', passed=None, skip_reason=q_reason, visits=visits, policy=policy,
        action=action, completed_q=completed, seconds=time.monotonic() - start,
    )


def _mismatch(axis, left, right, *, atol=0.0, rtol=0.0):
    left_a = np.asarray(left)
    right_a = np.asarray(right)
    if left_a.shape != right_a.shape:
        return dict(axis=axis, detail='shape %s != %s' % (left_a.shape, right_a.shape))
    if np.issubdtype(left_a.dtype, np.integer) and np.issubdtype(right_a.dtype, np.integer):
        if np.array_equal(left_a, right_a):
            return None
        return dict(axis=axis, detail='integer mismatch %s vs %s' % (left_a.tolist(), right_a.tolist()))
    if np.allclose(left_a, right_a, atol=atol, rtol=rtol, equal_nan=False):
        return None
    return dict(axis=axis, detail='max_abs=%.3e' % float(np.max(np.abs(left_a.astype(np.float64) - right_a.astype(np.float64)))))


def _compare_outputs(left, right, *, atol, rtol, require_all=False):
    mismatches = []
    if require_all:
        for field in ('visits', 'policy', 'action', 'completed_q'):
            if left.get(field) is None or right.get(field) is None:
                mismatches.append(dict(axis=field, detail='required comparison output unavailable'))
    if left.get('visits') is not None and right.get('visits') is not None:
        item = _mismatch('visits', left['visits'], right['visits'])
        if item:
            mismatches.append(item)
    if left.get('action') is not None and right.get('action') is not None:
        if int(left['action']) != int(right['action']):
            mismatches.append(dict(axis='final_action',
                                   detail='%s vs %s' % (left['action'], right['action'])))
    if left.get('policy') is not None and right.get('policy') is not None:
        item = _mismatch('improved_policy', left['policy'], right['policy'], atol=atol, rtol=rtol)
        if item:
            mismatches.append(item)
    if left.get('completed_q') is not None and right.get('completed_q') is not None:
        item = _mismatch('completed_q', left['completed_q'], right['completed_q'], atol=atol, rtol=rtol)
        if item:
            mismatches.append(item)
    return mismatches


def _spearman(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        return 1.0 if np.array_equal(left, right) else 0.0
    order_l = np.argsort(left, kind='mergesort')
    order_r = np.argsort(right, kind='mergesort')
    rank_l = np.empty(left.size, dtype=np.float64)
    rank_r = np.empty(right.size, dtype=np.float64)
    rank_l[order_l] = np.arange(left.size)
    rank_r[order_r] = np.arange(right.size)
    rank_l -= rank_l.mean()
    rank_r -= rank_r.mean()
    denom = float(np.sqrt(np.sum(rank_l ** 2) * np.sum(rank_r ** 2)))
    if denom == 0.0:
        return 1.0
    return float(np.sum(rank_l * rank_r) / denom)


def make_bandit_fixture():
    """Deterministic depth-1 same-player table with explicit Gumbel noise."""
    transitions = {
        '0:0': dict(child_id=1, reward=1.0, discount=0.0, value=0.0, actor=0, game_owner=0,
                    legal=[True], prior=[1.0]),
        '0:1': dict(child_id=2, reward=0.0, discount=0.0, value=0.0, actor=0, game_owner=0,
                    legal=[True], prior=[1.0]),
        '0:2': dict(child_id=3, reward=-1.0, discount=0.0, value=0.0, actor=0, game_owner=0,
                    legal=[True], prior=[1.0]),
    }
    return dict(
        kind='recurrent_records', information_model='perfect_info_numeric_replay',
        throughput_claim='offline_cpu_numeric', simulations=8, gumbel_candidates=3,
        max_depth=2, max_actions=3,
        roots=[dict(root_id=0, seed=20261016, viewer=0, game_owner=0, action_owner=0,
                    prior=[0.5, 0.3, 0.2], value=0.0, legal=[True, True, True],
                    gumbel=[0.10, 0.80, -0.20], transitions=transitions)],
    )


def make_recorded_tree_fixture():
    """Completed root snapshot used for formula comparison without search."""
    return dict(
        kind='recorded_tree', information_model='perfect_info_numeric_replay',
        throughput_claim='offline_cpu_numeric', simulations=2, gumbel_candidates=2,
        max_depth=1, max_actions=2,
        roots=[dict(root_id=0, seed=7, viewer=0, game_owner=0, action_owner=0,
                    prior=[0.75, 0.25], value=0.0, legal=[True, True],
                    visits=[2, 0], total=[2.0, 0.0], gumbel=[0.0, 0.0])],
    )


def _prepare_payload(payload):
    payload = load_records(payload)
    simulations = int(payload.get('simulations', 8))
    gumbel_candidates = int(payload.get('gumbel_candidates', 16))
    max_depth = int(payload.get('max_depth', 10))
    max_actions = int(payload.get('max_actions', 8))
    roots = [_normalize_root(root, simulations=simulations, gumbel_candidates=gumbel_candidates,
                             max_depth=max_depth, max_actions=max_actions)
             for root in payload['roots']]
    payload = dict(payload)
    payload.update(simulations=simulations, gumbel_candidates=gumbel_candidates,
                   max_depth=max_depth, max_actions=max_actions, roots=roots)
    return payload


def _tolist(value):
    return None if value is None else np.asarray(value).tolist()


def _public_search_result(result):
    out = dict(status=result.get('status'), seconds=result.get('seconds'),
               unsupported=list(result.get('unsupported') or []))
    for name in ('action', 'requested', 'completed_simulations'):
        if result.get(name) is not None:
            out[name] = int(result[name])
    if result.get('schedule') is not None:
        out['schedule'] = list(result['schedule'])
    for name in ('visits', 'policy', 'visit_policy', 'completed_q', 'total'):
        if result.get(name) is not None:
            out[name] = _tolist(result[name])
    return out


def _public_official(result):
    out = dict(status=result.get('status'), passed=bool(result.get('passed')),
               skip_reason=result.get('skip_reason'), jax_available=bool(result.get('jax_available')),
               mctx_available=bool(result.get('mctx_available')), jax_version=result.get('jax_version'),
               mctx_version=result.get('mctx_version'), install_attempted=False,
               seconds=result.get('seconds'))
    for name in ('unsupported', 'mismatches'):
        if result.get(name):
            out[name] = list(result[name])
    for name in ('visits', 'policy', 'completed_q'):
        if result.get(name) is not None:
            out[name] = _tolist(result[name])
    if result.get('action') is not None:
        out['action'] = int(result['action'])
    if result.get('error'):
        out['error'] = str(result['error'])
    return out


def _statistical_trials(root, *, trials, run_official):
    start = time.monotonic()
    rng_a = np.random.default_rng(int(root['seed']) + 17)
    rng_b = np.random.default_rng(int(root['seed']) + 8011)
    actions_a, actions_b, visits_corr, official_agreement = [], [], [], []
    n_legal = int(root['legal'].sum())
    width = int(root['max_actions'])
    for _trial in range(int(trials)):
        noise_a = np.zeros(width, dtype=np.float64)
        noise_b = np.zeros(width, dtype=np.float64)
        noise_a[root['legal']] = rng_a.gumbel(size=n_legal)
        noise_b[root['legal']] = rng_b.gumbel(size=n_legal)
        left = replay_device_gumbel_semantics(root, gumbel=noise_a)
        right = replay_device_gumbel_semantics(root, gumbel=noise_b)
        actions_a.append(int(left['action']))
        actions_b.append(int(right['action']))
        visits_corr.append(_spearman(left['visits'], right['visits']))
        if run_official:
            official = run_official_mctx(root, exact=False)
            if official.get('status') == 'ran' and official.get('action') is not None:
                official_agreement.append(int(official['action']) == int(left['action']))
    same = replay_device_gumbel_semantics(root)
    same_again = replay_device_gumbel_semantics(root)
    return dict(
        trials=int(trials),
        different_prng_action_agreement=float(np.mean(np.array(actions_a) == np.array(actions_b))),
        different_prng_visit_spearman=float(np.mean(visits_corr)),
        official_prng_action_agreement=(None if not official_agreement else float(np.mean(official_agreement))),
        exact_same_noise_self_match=same['action'] == same_again['action'] and np.array_equal(
            same['visits'], same_again['visits']),
        note='Different PRNG families are statistical only; they never satisfy an exact claim.',
        seconds=time.monotonic() - start,
    )


def compare_records(payload, *, exact=False, run_official=True, statistical_trials=0):
    payload = _prepare_payload(payload)
    started = time.monotonic()
    cases = []
    exact_attempted = bool(exact)
    exact_passed = True if exact else False
    exact_reject = None
    for index, root in enumerate(payload['roots']):
        case = dict(root_id=root['root_id'], index=index, seed=root['seed'],
                    viewer=root['viewer'], game_owner=root['game_owner'],
                    action_owner=root['action_owner'])
        if exact:
            reject = exact_claim_precondition(root)
            if reject:
                exact_reject = reject
                exact_passed = False
                case['exact'] = dict(status='rejected', reason=reject, matched=False)
                cases.append(case)
                continue
        if payload['kind'] == 'recorded_tree' or (root['visits'] is not None and not root['transitions']):
            local = recorded_tree_semantics(root)
        else:
            local = replay_device_gumbel_semantics(root)
        case['device_semantics'] = _public_search_result(local)
        mismatches = []
        if local.get('status') != 'ran':
            exact_passed = False if exact else exact_passed
            mismatches.append(dict(axis='device_semantics', detail=local.get('status')))
        if local.get('status') == 'ran':
            try:
                case['torch_statistics'] = compare_torch_statistics(root, local)
                mismatches.extend(case['torch_statistics']['mismatches'])
            except Exception as exc:
                case['torch_statistics'] = dict(status='failed', passed=False,
                    error=str(exc), type=type(exc).__name__, cuda_qualified=False)
                mismatches.append(dict(axis='torch_statistics', detail=str(exc)))
        if root['device_visits'] is not None and local.get('visits') is not None:
            mismatches.extend(_compare_outputs(
                local, dict(visits=root['device_visits'], policy=root['device_policy'],
                            action=root['device_action']),
                atol=1e-12, rtol=1e-12))
        official = dict(status='skipped', passed=False, skip_reason='official_not_requested',
                        install_attempted=False, jax_available=False, mctx_available=False)
        if run_official:
            official = run_official_mctx(root, exact=exact)
            if official.get('status') == 'ran' and local.get('status') == 'ran':
                official_mismatches = _compare_outputs(local, official, atol=1e-5, rtol=1e-5, require_all=True)
                official['mismatches'] = official_mismatches
                official['passed'] = (not official_mismatches) if exact else False
                if exact:
                    mismatches.extend(official_mismatches)
                    if official_mismatches:
                        exact_passed = False
            elif exact:
                official['passed'] = False
                exact_passed = False
                mismatches.append(dict(axis='official_execution', detail=official.get('status'),
                                       reason=official.get('skip_reason')))
        case['official_mctx'] = _public_official(official)
        stats = None
        if statistical_trials > 0 and local.get('status') == 'ran' and root['transitions']:
            stats = _statistical_trials(root, trials=statistical_trials, run_official=False)
        case['statistical'] = stats
        case['mismatches'] = mismatches
        case['matched'] = (not mismatches) if exact and exact_reject is None else None
        if exact and case['matched'] is False:
            exact_passed = False
        cases.append(case)
    report = dict(
        scope='offline official Mctx vs DeviceGumbelSearch-comparable numerics; not hidden-info; not CUDA throughput',
        kind=payload['kind'], comparable=list(COMPARABLE_AXES), not_claimed=list(NOT_CLAIMED),
        install_attempted=False,
        official_requested=bool(run_official),
        exact=dict(attempted=exact_attempted, passed=bool(exact_attempted and exact_passed and exact_reject is None),
                   reject_reason=exact_reject),
        official_probe=probe_official_mctx(load=run_official).__dict__,
        cases=cases,
        timings=dict(scope=TIMING_SCOPE, seconds=time.monotonic() - started),
        remaining_device_integration=list(REMAINING_DEVICE_INTEGRATION),
    )
    report['passed'] = bool(report['exact']['passed'])
    report['official_comparison_passed'] = bool(run_official and cases and
        all(case.get('official_mctx', {}).get('status') == 'ran' and
            case.get('official_mctx', {}).get('passed') for case in cases))
    report['torch_statistics_passed'] = bool(cases and all(
        case.get('torch_statistics', {}).get('passed') for case in cases))
    report['actual_device_export_compared'] = bool(payload['roots'] and all(
        root.get('device_visits') is not None and root.get('device_policy') is not None
        and root.get('device_action') is not None for root in payload['roots']))
    return report


def plan_report():
    status = probe_official_mctx(load=False)
    official_status = 'available_not_run' if status.available else 'skipped'
    return dict(
        mode='plan',
        scope='offline official Mctx vs DeviceGumbelSearch-comparable numerics; not hidden-info; not CUDA throughput',
        install_attempted=False,
        official_mctx=dict(status=official_status, passed=False, skip_reason=status.skip_reason,
                           jax_available=status.jax_available, mctx_available=status.mctx_available,
                           jax_version=status.jax_version, mctx_version=status.mctx_version,
                           install_attempted=False),
        comparable=list(COMPARABLE_AXES), not_claimed=list(NOT_CLAIMED),
        exact=dict(attempted=False, passed=False, reject_reason='plan_mode_does_not_run_comparison'),
        timings=dict(scope=TIMING_SCOPE, seconds=0.0),
        remaining_device_integration=list(REMAINING_DEVICE_INTEGRATION),
        passed=False,
        next_steps=[
            'Do not pip-install JAX/Mctx from this harness.',
            'Run with --run --exact on records that store Gumbel, or use the built-in bandit fixture.',
            'Use --require-official only when JAX and Mctx are already installed on this CPU runtime.',
        ],
    )
