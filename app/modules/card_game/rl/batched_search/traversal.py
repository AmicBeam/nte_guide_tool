"""Cooperative batched Gumbel search traversal as a Python generator.

Yields InferenceQuery for device batch evaluation and receives (logits, wdl).
Retains exact recovery_search sequential dependencies, formulas, and Q-scales.
Offline batch search component; importing never starts CUDA, PyTorch, or games.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Callable, Generator, List, Optional, Protocol, Tuple, Union
from uuid import uuid4
import numpy as np


class TraversalCancelled(Exception):
    """The scheduler closed this root; retain only earlier complete simulations."""


class _DeadlineReached(TraversalCancelled):
    pass


class SearchBackend(Protocol):
    """Explicit backend interface decoupling search traversal from state representation."""

    def acting_side(self, state: Any) -> str: ...
    def finished(self, state: Any) -> bool: ...
    def terminal_value(self, state: Any, viewer: str) -> float: ...
    def decision(self, state: Any, side: str) -> Tuple[List[Any], Tuple[np.ndarray, np.ndarray]]: ...
    def sample_world(self, state: Any, viewer: str, seed: int) -> Any: ...
    def step(self, state: Any, side: str, action: Any) -> Any: ...
    def clean(self, state: Any) -> Any: ...


@dataclass(frozen=True)
class InferenceQuery:
    """Named neural network evaluation query yielded by search traversal."""
    id: str
    policy_key: str
    model_version: str
    x: np.ndarray
    c: np.ndarray
    value_actor: bool
    observer_side: str
    value_only: bool = False
    deadline: Optional[float] = None


@dataclass(frozen=True)
class InferenceResponse:
    """Optional typed response sent back to the traversal generator."""
    id: str
    logits: np.ndarray
    wdl: np.ndarray
    model_version: Optional[str] = None


@dataclass(frozen=True)
class RootEvaluation:
    """One already completed inference on this exact, unchanged physical root."""
    state: Any
    viewer: str
    actions: Any
    query: InferenceQuery
    response: InferenceResponse


@dataclass(frozen=True)
class PolicyIdentity:
    """Policy name and frozen model hash identity."""
    policy_key: str
    model_version: str


@dataclass
class Node:
    """MCTS node holding prior, visit counts, cumulative value, and raw value."""
    prior: np.ndarray
    visits: np.ndarray
    total: np.ndarray
    raw_value: float = 0.0

    @classmethod
    def create(cls, prior: np.ndarray, raw_value: float = 0.0) -> 'Node':
        prior_arr = np.asarray(prior, dtype=np.float64)
        return cls(
            prior=prior_arr,
            visits=np.zeros(len(prior_arr), dtype=np.int32),
            total=np.zeros(len(prior_arr), dtype=np.float64),
            raw_value=float(raw_value),
        )


def softmax(scores: Any) -> np.ndarray:
    scores_arr = np.asarray(scores, dtype=np.float64)
    p = np.exp(scores_arr - np.max(scores_arr))
    return p / p.sum()


def normalized_exp(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    p = np.exp(arr - arr.max())
    return p / p.sum()


def search_prior(logits: Any, mode: str = 'model', temperature: float = 1.0, uniform_mix: float = 0.0) -> np.ndarray:
    if mode not in ('model', 'uniform', 'tempered') or not np.isfinite(temperature) or temperature <= 0 or not 0 <= uniform_mix <= 1:
        raise ValueError('Invalid search prior settings')
    logits_arr = np.asarray(logits, dtype=np.float64)
    if mode == 'uniform':
        return np.full(len(logits_arr), 1.0 / len(logits_arr), dtype=np.float64)
    prior = softmax(logits_arr / temperature if mode == 'tempered' else logits_arr)
    return (1.0 - uniform_mix) * prior + uniform_mix / len(prior)


def info_key(x: np.ndarray, c: np.ndarray) -> bytes:
    x_bytes = np.asarray(x).tobytes()
    c_bytes = np.asarray(c).tobytes()
    return hashlib.sha256(x_bytes + c_bytes).digest()


def legacy_completed_q(prior: np.ndarray, visits: np.ndarray, total: np.ndarray, raw_value: float) -> np.ndarray:
    prior = np.asarray(prior, dtype=np.float64)
    visits = np.asarray(visits)
    visited = visits > 0
    q = np.divide(total, visits, out=np.zeros(len(visits), dtype=np.float64), where=visited)
    mass = prior[visited].sum()
    mean = float(np.dot(prior[visited], q[visited]) / mass) if mass > 0 else raw_value
    mixed = (raw_value + visits.sum() * mean) / (1 + visits.sum())
    q = np.where(visited, q, mixed)
    return (q - q.min()) / max(float(q.max() - q.min()), 1e-8) * (0.1 * (50 + visits.max(initial=0)))


def raw_completed_q(prior: np.ndarray, visits: np.ndarray, total: np.ndarray, value: float) -> np.ndarray:
    prior = np.asarray(prior, dtype=np.float64)
    visits = np.asarray(visits)
    total = np.asarray(total, dtype=np.float64)
    if (prior.ndim != 1 or not len(prior) or prior.shape != visits.shape or total.shape != prior.shape
            or not np.isfinite(prior).all() or not np.isfinite(total).all() or not np.isfinite(value)
            or (visits < 0).any() or (prior < 0).any() or not np.isclose(prior.sum(), 1.0)):
        raise ValueError('Invalid root search estimates')
    seen = visits > 0
    q = np.divide(total, visits, out=np.zeros_like(total), where=seen)
    mass = prior[seen].sum()
    mean = np.dot(prior[seen], q[seen]) / mass if mass > 0 else value
    mixed = (value + visits.sum() * mean) / (1 + visits.sum())
    return np.where(seen, q, mixed)


def completed_q(node: Node, mode: str) -> np.ndarray:
    if mode == 'legacy':
        return legacy_completed_q(node.prior, node.visits, node.total, node.raw_value)
    if mode != 'natural_wdl':
        raise ValueError('Unknown recovery Q scale')
    q = raw_completed_q(node.prior, node.visits, node.total, node.raw_value)
    if np.any(np.abs(q) > 1.0 + 1e-6):
        raise ValueError('Natural WDL search requires values in [-1,1]')
    return q * (0.1 * (50 + node.visits.max(initial=0)))


def improved_policy(node: Node, mode: str) -> np.ndarray:
    logits = np.log(np.maximum(node.prior, 1e-300)) + completed_q(node, mode)
    return normalized_exp(logits)


def choose_root(node: Node, noise: np.ndarray, visit: int, mode: str) -> int:
    eligible = node.visits == visit
    if not eligible.any():
        raise ValueError('Invalid sequential-halving visit schedule')
    scores = np.log(np.maximum(node.prior, 1e-300)) + noise + completed_q(node, mode)
    return int(np.argmax(np.where(eligible, scores, -np.inf)))


def visit_schedule(actions: int, budget: int) -> List[int]:
    if actions == 1:
        return list(range(budget))
    phases = math.ceil(math.log2(actions))
    counts = [0] * actions
    result: List[int] = []
    active = actions
    while len(result) < budget:
        rounds = max(1, budget // (phases * active))
        for _ in range(rounds):
            result.extend(counts[:active])
            for i in range(active):
                counts[i] += 1
        active = max(2, active // 2)
    return result[:budget]


def action_identity(action: Any) -> str:
    """Stable legal-action identity. Search indexes by legal-list order."""
    return json.dumps(action, sort_keys=True, ensure_ascii=False, default=str)


def _root_diagnostics(actions: List[Any], prior: np.ndarray, visits: np.ndarray, root: Node,
                      selected: int, requested: int, completed: int, gumbel_candidates: int,
                      algorithm: str) -> dict:
    q = completed_q(root, 'legacy') if algorithm == 'gumbel' else np.divide(
        root.total, visits, out=np.full(len(visits), np.nan), where=visits > 0
    )
    cap = min(len(actions), gumbel_candidates, requested) if algorithm == 'gumbel' else len(actions)
    admitted = np.flatnonzero(visits > 0).tolist() if algorithm == 'gumbel' else list(range(len(actions)))
    admitted_set = set(admitted)
    return dict(
        root_diagnostics=dict(
            action_identities=[action_identity(a) for a in actions],
            prior=np.asarray(prior, dtype=np.float64),
            admitted_to_max16=[index in admitted_set for index in range(len(actions))] if algorithm == 'gumbel' else [True] * len(actions),
            admitted_indexes=admitted,
            visits=np.asarray(visits, dtype=np.int32),
            completed_q=np.asarray(q, dtype=np.float64),
            completed_q_definition='search_policy.completed_q: mixed and scaled search score',
            visited_mean_value=[float(root.total[i] / count) if count else None for i, count in enumerate(visits)],
            selected_index=int(selected),
            selected_identity=action_identity(actions[selected]) if actions else None,
            requested_simulations=int(requested),
            completed_simulations=int(completed),
            candidate_cap=int(cap),
        )
    )


def _get_policy_identity(policy_identities: Any, side: str) -> Tuple[str, str]:
    if side not in policy_identities:
        raise ValueError(f'Missing policy identity for side {side}')
    spec = policy_identities[side]
    if isinstance(spec, PolicyIdentity):
        return spec.policy_key, spec.model_version
    if isinstance(spec, tuple) and len(spec) == 2:
        return str(spec[0]), str(spec[1])
    if isinstance(spec, dict):
        key = spec.get('policy_key', spec.get('key'))
        ver = spec.get('model_version', spec.get('version'))
        if key is None or ver is None:
            raise ValueError(f'Invalid policy identity dictionary for side {side}')
        return str(key), str(ver)
    if hasattr(spec, 'policy_key') and hasattr(spec, 'model_version'):
        return str(spec.policy_key), str(spec.model_version)
    if hasattr(spec, 'key') and hasattr(spec, 'version'):
        return str(spec.key), str(spec.version)
    if hasattr(spec, 'manifest') and hasattr(spec, 'version'):
        deck = spec.manifest.get('deck', getattr(spec, 'key', 'model'))
        return str(deck), str(spec.version)
    raise ValueError(f'Explicit policy identity required for side {side}')


def _unpack_response(response: Any, query: InferenceQuery, q_scale: str = 'natural_wdl') -> Tuple[np.ndarray, float]:
    if response is None:
        raise ValueError('Response cannot be None')
    resp_id: Optional[str] = None
    resp_version: Optional[str] = None
    logits_val: Any = None
    wdl_val: Any = None

    if isinstance(response, InferenceResponse):
        if response.id != query.id or response.model_version != query.model_version:
            raise ValueError('Bound response identity/version mismatch')
        resp_id = response.id
        resp_version = response.model_version
        logits_val = response.logits
        wdl_val = response.wdl
    elif isinstance(response, dict):
        if response.get('id') != query.id or response.get('model_version') != query.model_version:
            raise ValueError('Bound response identity/version mismatch')
        resp_id = response.get('id')
        resp_version = response.get('model_version')
        logits_val = response['logits']
        wdl_val = response.get('wdl')
        if wdl_val is None and 'value' in response:
            wdl_val = response['value']
    elif isinstance(response, (tuple, list)):
        if len(response) == 2:
            logits_val, wdl_val = response
        elif len(response) == 3:
            if isinstance(response[0], str):
                resp_id, logits_val, wdl_val = response
            else:
                raise ValueError(f'Unexpected 3-tuple response format: first item must be query id string')
        elif len(response) == 4:
            resp_id, logits_val, wdl_val, resp_version = response
        else:
            raise ValueError(f'Invalid response tuple length: {len(response)}')
    else:
        raise TypeError(f'Unsupported response type: {type(response)}')

    if resp_id is not None and resp_id != query.id:
        raise ValueError(f'Query ID mismatch: expected {query.id!r}, got {resp_id!r}')
    if resp_version is not None and resp_version != query.model_version:
        raise ValueError(f'Model version mismatch: expected {query.model_version!r}, got {resp_version!r}')

    logits = np.asarray(logits_val, dtype=np.float64)
    if logits.ndim != 1 or len(logits) != len(query.c) or not np.isfinite(logits).all():
        raise ValueError(f'Invalid logits: expected 1D finite array of length {len(query.c)}, got shape {logits.shape}')

    if wdl_val is None:
        raise ValueError('Missing WDL or explicitly bound scalar value')
    elif np.ndim(wdl_val) == 0:
        val = float(wdl_val)
        if not np.isfinite(val):
            raise ValueError('Non-finite scalar value in inference response')
    else:
        wdl = np.asarray(wdl_val, dtype=np.float64)
        if wdl.ndim != 1 or len(wdl) != 3 or not np.isfinite(wdl).all():
            raise ValueError(f'Invalid wdl: expected 1D finite array of length 3, got shape {wdl.shape}')
        if (wdl < 0).any() or not np.isclose(wdl.sum(), 1., atol=1e-6):
            raise ValueError('WDL must be a normalized nonnegative probability vector')
        val = float(wdl[2] - wdl[0])

    if q_scale == 'natural_wdl' and abs(val) > 1.0 + 1e-6:
        raise ValueError(f'Natural WDL search requires values in [-1,1], got {val}')

    return logits, val


def _follow_terminal_generator(
    world: Any,
    viewer: str,
    policy_identities_by_side: Any,
    rng: np.random.Generator,
    steps: int,
    *,
    backend: SearchBackend,
    query_factory: Callable[[str, np.ndarray, np.ndarray], InferenceQuery],
    deadline: Optional[float],
    clock: Callable[[], float],
    q_scale: str = 'natural_wdl',
) -> Generator[InferenceQuery, Any, Optional[float]]:
    """Play up to `steps` more decisions. Return terminal value, or None if none occurs."""
    for step_idx in range(steps + 1):
        if deadline is not None and clock() >= deadline:
            raise _DeadlineReached('Decision deadline')
        if backend.finished(world):
            return backend.terminal_value(world, viewer)
        if step_idx == steps:
            return None
        actor = backend.acting_side(world)
        legal, (encoded, encoded_actions) = backend.decision(world, actor)
        query = query_factory(actor, encoded, encoded_actions)
        response = yield query
        if deadline is not None and clock() >= deadline:
            raise _DeadlineReached('Inference completed after decision deadline')
        scores, _ = _unpack_response(response, query, q_scale=q_scale)
        if actor != viewer:
            chosen = int(rng.choice(len(legal), p=softmax(scores)))
        else:
            chosen = int(np.argmax(scores))
        world = backend.clean(backend.step(world, actor, legal[chosen]))
    return None


def _short_terminal_value_generator(
    world: Any,
    viewer: str,
    policy_identities_by_side: Any,
    rng: np.random.Generator,
    *,
    scores: np.ndarray,
    legal: List[Any],
    horizon: int,
    backend: SearchBackend,
    query_factory: Callable[[str, np.ndarray, np.ndarray], InferenceQuery],
    deadline: Optional[float],
    clock: Callable[[], float],
    q_scale: str = 'natural_wdl',
) -> Generator[InferenceQuery, Any, Optional[float]]:
    """Replace a leaf value only when a short real line proves the result."""
    if horizon < 1 or backend.finished(world):
        return backend.terminal_value(world, viewer) if backend.finished(world) else None
    attacks = [i for i, action in enumerate(legal) if action.get('type') == 'attack']
    ranked = [int(i) for i in np.argsort(-np.asarray(scores, dtype=np.float64))]
    chosen: List[int] = []
    for index in attacks + ranked:
        if index not in chosen:
            chosen.append(index)
        if len(chosen) >= max(8, len(attacks)) and index not in attacks:
            break
    outcomes: List[float] = []
    complete = True
    for index in chosen:
        if deadline is not None and clock() >= deadline:
            raise _DeadlineReached('Decision deadline')
        nxt = backend.clean(backend.step(world, viewer, legal[index]))
        outcome = yield from _follow_terminal_generator(
            nxt, viewer, policy_identities_by_side, rng,
            steps=horizon - 1,
            backend=backend, query_factory=query_factory,
            deadline=deadline, clock=clock, q_scale=q_scale,
        )
        if outcome is None:
            complete = False
            continue
        outcomes.append(outcome)
        if outcome > 0:
            return 1.0
    if complete and len(chosen) == len(legal) and outcomes and all(item < 0 for item in outcomes):
        return -1.0
    return None


def search_traversal(
    root_state: Any,
    viewer: str,
    policy_identities_by_side: Any,
    *,
    backend: SearchBackend,
    seed: int,
    simulations: int = 32,
    gumbel_candidates: int = 16,
    terminal_horizon: int = 3,
    max_depth: int = 10,
    q_scale: str = 'natural_wdl',
    noise: bool = False,
    clock: Optional[Callable[[], float]] = None,
    time_limit: Optional[float] = None,
    deadline: Optional[float] = None,
    root_id: Optional[str] = None,
    statistics: Optional[Any] = None,
    collect_diagnostics: bool = True,
    root_evaluation: Optional[RootEvaluation] = None,
) -> Generator[InferenceQuery, Any, dict]:
    """Collaborative Gumbel search traversal yielding InferenceQuery instances.

    Each simulation advances step-by-step with sequential visit-count dependencies.
    Terminates with a result dict matching recovery_search.search upon StopIteration.
    """
    if type(collect_diagnostics) is not bool:raise TypeError('Boolean diagnostics option required')
    if q_scale not in ('legacy', 'natural_wdl'):
        raise ValueError('Unknown recovery Q scale')
    if (type(simulations) is not int or simulations < 1 or type(gumbel_candidates) is not int
            or gumbel_candidates < 1 or type(max_depth) is not int or max_depth < 1
            or type(terminal_horizon) is not int or terminal_horizon < 0):
        raise ValueError('Invalid bounded recovery search limits')
    if backend.acting_side(root_state) != viewer:
        raise ValueError('Search must use decision owner')

    if clock is None:
        clock_fn: Callable[[], float] = time.monotonic
    elif callable(clock):
        clock_fn = clock
    else:
        raise TypeError('Clock must be callable')
    if statistics is not None:
        if statistics.q_scale!=q_scale or statistics.clock is not clock_fn:
            raise ValueError('Statistics scale and monotonic clock must match the traversal')

    calc_deadline: Optional[float] = None
    if time_limit is not None:
        if isinstance(time_limit, bool) or not isinstance(time_limit, (int, float)) or not math.isfinite(time_limit) or time_limit <= 0:
            raise ValueError('Positive finite time_limit required')
        calc_deadline = clock_fn() + float(time_limit)
    elif deadline is not None:
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
            raise ValueError('Finite deadline required')
        calc_deadline = float(deadline)

    if root_id is None:
        root_id = getattr(root_state, 'id', None)
        if root_id is None and isinstance(root_state, dict):
            root_id = root_state.get('game_id') or root_state.get('id')
        if root_id is None:
            root_id = f"root_{seed}_{uuid4().hex[:8]}"

    query_counter = 0
    root_token = uuid4().hex

    def make_query(actor_side: str, enc_x: np.ndarray, enc_c: np.ndarray, *, value_actor=True, value_only=False) -> InferenceQuery:
        nonlocal query_counter
        p_key, m_version = _get_policy_identity(policy_identities_by_side, actor_side)
        qid = f"{root_id}:{root_token}:{query_counter}"
        query_counter += 1
        return InferenceQuery(
            id=qid,
            policy_key=p_key,
            model_version=m_version,
            x=enc_x,
            c=enc_c,
            value_actor=bool(value_actor),
            observer_side=actor_side,
            value_only=value_only,
        )

    if root_evaluation is None:
        actions, (x, c) = backend.decision(root_state, viewer)
        root_query = make_query(viewer, x, c)
        response = yield root_query
    else:
        cached=root_evaluation
        key,version=_get_policy_identity(policy_identities_by_side,viewer)
        if (not isinstance(cached,RootEvaluation) or cached.state is not root_state or cached.viewer!=viewer
                or not isinstance(cached.query,InferenceQuery) or cached.query.policy_key!=key
                or cached.query.model_version!=version or cached.query.observer_side!=viewer
                or not cached.query.value_actor or cached.query.value_only):
            raise ValueError('Cached evaluation is not bound to this physical root/model/observer')
        actions=cached.actions;root_query=cached.query;response=cached.response
        x,c=root_query.x,root_query.c
        if len(actions)!=len(c) or not actions:raise ValueError('Cached root action count mismatch')
    logits, root_value = _unpack_response(response, root_query, q_scale=q_scale)

    prior = search_prior(logits)
    rng = np.random.default_rng(seed)
    owner=statistics.new_owner() if statistics is not None else None
    root=(yield from statistics.create(prior,root_value,owner)) if statistics is not None else Node.create(prior,root_value)
    root_ref=root
    tree: dict[bytes, Any] = {info_key(x, c): root}
    requested = 1 if len(actions) == 1 else simulations
    schedule = visit_schedule(min(len(actions), gumbel_candidates, requested), requested)
    gumbel = rng.gumbel(size=len(actions)) if noise else np.zeros(len(actions))
    depths: List[int] = []
    worlds = 0
    terminal_checks = np.zeros(len(actions), dtype=np.int32)
    terminal_wins = np.zeros(len(actions), dtype=np.int32)

    stop_reason = None
    try:
        for simulation in range(requested):
            if calc_deadline is not None and clock_fn() >= calc_deadline:
                break
            world = backend.sample_world(root_state, viewer, seed + simulation * 104729)
            worlds += 1
            path: List[Tuple[Node, int]] = []
            value = 0.0
            finished_sim = True
            root_action: Optional[int] = None
            root_terminal_win = False

            for depth in range(max_depth):
                if calc_deadline is not None and clock_fn() >= calc_deadline:
                    finished_sim = False
                    break
                if backend.finished(world):
                    value = backend.terminal_value(world, viewer)
                    break
                actor = backend.acting_side(world)
                legal, (xx, cc) = backend.decision(world, actor)
                if actor != viewer:
                    query = make_query(actor, xx, cc)
                    response = yield query
                    if calc_deadline is not None and clock_fn() >= calc_deadline:
                        raise _DeadlineReached('Inference completed after decision deadline')
                    scores, _ = _unpack_response(response, query, q_scale=q_scale)
                    chosen = int(rng.choice(len(legal), p=softmax(scores)))
                else:
                    key = info_key(xx, cc)
                    if key not in tree:
                        query = make_query(actor, xx, cc)
                        response = yield query
                        if calc_deadline is not None and clock_fn() >= calc_deadline:
                            raise _DeadlineReached('Inference completed after decision deadline')
                        scores, value = _unpack_response(response, query, q_scale=q_scale)
                        tree[key] = ((yield from statistics.create(search_prior(scores),value,owner))
                                     if statistics is not None else Node.create(search_prior(scores), value))
                        remaining = max_depth - depth
                        if terminal_horizon and remaining > 0:
                            proved = yield from _short_terminal_value_generator(
                                world, viewer, policy_identities_by_side, rng,
                                scores=scores, legal=legal,
                                horizon=min(terminal_horizon, remaining),
                                backend=backend, query_factory=make_query,
                                deadline=calc_deadline, clock=clock_fn, q_scale=q_scale,
                            )
                            if proved is not None:
                                value = proved
                                if q_scale == 'legacy':
                                    if statistics is None:tree[key].raw_value = proved
                                    else:yield from statistics.set_value(tree[key],proved)
                        break
                    node = tree[key]
                    if statistics is None:
                        chosen = (choose_root(node, gumbel, schedule[simulation], q_scale) if depth == 0 else
                                  int(np.argmax(improved_policy(node, q_scale) - node.visits / (1 + node.visits.sum()))))
                    else:
                        chosen=((yield from statistics.root_choice(node,gumbel,int(schedule[simulation]))) if depth==0
                                else (yield from statistics.interior_choice(node)))
                        if calc_deadline is not None and clock_fn()>=calc_deadline:
                            raise _DeadlineReached('Device selection completed after decision deadline')
                    path.append((node, chosen))
                world = backend.clean(backend.step(world, actor, legal[chosen]))
                if depth == 0:
                    root_action = chosen
                    root_terminal_win = backend.finished(world) and backend.terminal_value(world, viewer) > 0
            else:
                if backend.finished(world):
                    value = backend.terminal_value(world, viewer)
                else:
                    actor = backend.acting_side(world)
                    # A leaf uses the fixed root observer's independent WDL. An
                    # opponent-action policy estimate cannot be negated into it.
                    if hasattr(backend, 'value_observation'):
                        xx, cc = backend.value_observation(world, viewer)
                    else:
                        _, (xx, original_c) = backend.decision(world, viewer)
                        cc = original_c[:0]
                    query = make_query(viewer, xx, cc, value_actor=actor == viewer, value_only=True)
                    response = yield query
                    if calc_deadline is not None and clock_fn() >= calc_deadline:
                        raise _DeadlineReached('Inference completed after decision deadline')
                    _, value = _unpack_response(response, query, q_scale=q_scale)

            if not finished_sim:
                break
            if not np.isfinite(value):
                raise ValueError('Non-finite recovery search value')
            if root_action is not None:
                terminal_checks[root_action] += 1
                terminal_wins[root_action] += int(root_terminal_win)
            if statistics is None:
                for node, chosen in path:
                    node.visits[chosen] += 1
                    node.total[chosen] += value
            else:
                accepted=yield from statistics.backup(path,value,calc_deadline)
                if not accepted:raise _DeadlineReached('Device backup completed after decision deadline')
            depths.append(depth + 1)

    except TraversalCancelled as exc:
        stop_reason = 'deadline' if isinstance(exc, _DeadlineReached) else 'cancelled'

    if not collect_diagnostics:
        if statistics is not None:
            compact=yield from statistics.final_summary(root_ref,gumbel)
            selected=compact['search_choice'];covered=compact['roots_covered']
        else:
            selected=choose_root(root,gumbel,int(root.visits.max(initial=0)),q_scale)
            covered=int((root.visits>0).sum())
        return dict(actions=actions,search_choice=selected,raw_choice=int(np.argmax(logits)),
            simulations=len(depths),requested=requested,complete=len(depths)==requested,
            roots_covered=covered,nodes=len(tree),worlds=worlds,max_depth=max(depths,default=0),
            stop_reason=stop_reason or ('complete' if len(depths)==requested else 'deadline'),
            q_scale=q_scale,policy_source='gumbel_q_'+q_scale)
    if statistics is not None:
        saved=yield from statistics.snapshot(root_ref)
        root=Node.create(saved['prior'],saved['raw_value'])
        root.visits=saved['visits'];root.total=saved['total']
    visits = root.visits.copy()
    pi = improved_policy(root, q_scale)
    selected = choose_root(root, gumbel, int(visits.max(initial=0)), q_scale)
    diagnostics = _root_diagnostics(
        actions, prior, visits, root, selected, requested,
        len(depths), gumbel_candidates, 'gumbel'
    )['root_diagnostics']
    diagnostics.update(
        completed_q=completed_q(root, q_scale),
        completed_q_definition='recovery_search:' + q_scale,
        q_scale=q_scale,
    )

    result = dict(
        actions=actions,
        x=x,
        c=c,
        pi=pi.astype(np.float32),
        visits=visits,
        mean_values=np.divide(root.total, visits, out=np.zeros(len(visits), dtype=np.float64), where=visits > 0),
        visit_pi=visits / visits.sum() if visits.sum() else prior.copy(),
        root_prior=prior,
        rootprior=prior,
        root_gumbel=gumbel,
        gumbel=gumbel,
        value=root_value,
        terminal_checks=terminal_checks,
        terminal_wins=terminal_wins,
        complete=len(depths) == requested,
        simulations=len(depths),
        sims=len(depths),
        requested=requested,
        roots_covered=int((visits > 0).sum()),
        search_choice=selected,
        selected=selected,
        raw_choice=int(np.argmax(logits)),
        rawchoice=int(np.argmax(logits)),
        nodes=len(tree),
        worlds=worlds,
        max_depth=max(depths, default=0),
        depth=max(depths, default=0),
        policy_source='gumbel_q_' + q_scale,
        root_diagnostics=diagnostics,
        q_scale=q_scale,
        teacher_mode='network',
        leaf_source='network_or_short_proof',
        stop_reason=stop_reason or ('complete' if len(depths) == requested else 'deadline'),
    )
    return result
