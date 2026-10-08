"""Root-target stability statistics; no changes to the production search."""
import numpy as np
from .policy_equivalence import representation_groups


def raw_completed_q(prior, visits, total, value):
    prior = np.asarray(prior, dtype=np.float64); visits = np.asarray(visits)
    total = np.asarray(total, dtype=np.float64)
    if (prior.ndim != 1 or not len(prior) or prior.shape != visits.shape or total.shape != prior.shape
            or not np.isfinite(prior).all() or not np.isfinite(total).all() or not np.isfinite(value)
            or (visits < 0).any() or (prior < 0).any() or not np.isclose(prior.sum(), 1.)):
        raise ValueError('Invalid root search estimates')
    seen = visits > 0
    q = np.divide(total, visits, out=np.zeros_like(total), where=seen)
    mass = prior[seen].sum()
    mean = np.dot(prior[seen], q[seen]) / mass if mass > 0 else value
    mixed = (value + visits.sum() * mean) / (1 + visits.sum())
    return np.where(seen, q, mixed)


def policy_without_range_rescaling(result):
    """Counterfactual target using the same tree, not a new search result."""
    visits = result['visits']
    total = np.asarray(result['mean_values']) * visits
    q = raw_completed_q(result['root_prior'], visits, total, result['value'])
    logits = np.log(np.maximum(result['root_prior'], 1e-300)) + .1 * (50 + max(visits)) * q
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()


def jensen_shannon(targets):
    targets = np.asarray(targets, dtype=np.float64)
    if (targets.ndim != 2 or not len(targets) or not np.isfinite(targets).all()
            or (targets < 0).any() or not np.allclose(targets.sum(-1), 1.)):
        raise ValueError('Invalid repeated target distributions')
    center = targets.mean(0)
    return float(np.mean(np.sum(np.where(targets > 0,
        targets * (np.log(np.maximum(targets, 1e-300)) - np.log(np.maximum(center, 1e-300))), 0), -1)))


def summarize(results):
    if not results or any(not r['complete'] for r in results):
        raise ValueError('Only full-budget roots may be compared')
    candidates = results[0]['c']; groups = representation_groups(candidates)
    for result in results[1:]:
        if not np.array_equal(result['c'], candidates) or not np.array_equal(result['x'], results[0]['x']):
            raise ValueError('Repeated search changed the visible root')
    group_of = {i: g for g, indexes in enumerate(groups) for i in indexes}
    targets = [r['pi'] for r in results]
    grouped = [[sum(pi[i] for i in indexes) for indexes in groups] for pi in targets]
    alternate = [policy_without_range_rescaling(r) for r in results]
    q = [raw_completed_q(r['root_prior'], r['visits'], np.asarray(r['mean_values']) * r['visits'], r['value']) for r in results]
    return dict(repeats=len(results), actual_selected_groups=[group_of[r['search_choice']] for r in results],
                target_top_groups=[group_of[int(np.argmax(r['pi']))] for r in results],
                target_js=jensen_shannon(targets), represented_group_js=jensen_shannon(grouped),
                same_tree_unscaled_target_js=jensen_shannon(alternate),
                raw_q_spans=[float(np.ptp(values)) for values in q],
                range_gain=[float(.1 * (50 + max(r['visits'])) / max(np.ptp(values), 1e-8))
                            for r, values in zip(results, q)],
                roots_covered=[r['roots_covered'] for r in results],
                same_tree_counterfactual_not_winrate=True)
