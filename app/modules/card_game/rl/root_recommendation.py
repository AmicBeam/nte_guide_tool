"""Opt-in offline final-action experiments; never a training policy target.

Mctx's max-visit recommendation remains the baseline. A completed sweep is
one equal-visit allocation pass, not an independent sample or a win proof.
The value variant is a research heuristic, with no confidence guarantee for
adaptive tree values or sampled hidden worlds.
"""
import math
import numpy as np

MODES = ('legacy', 'completed_sweep_gumbel', 'completed_sweep_value')


def sweeps(actions, budget):
    """Untruncated passes whose prefix is search_policy.visit_schedule."""
    if actions < 1 or budget < 1:
        raise ValueError('Positive candidate count and budget required')
    phases = max(1, math.ceil(math.log2(actions)))
    active = actions
    visit = used = 0
    result = []
    while used < budget:
        rounds = max(1, budget // (phases * active))
        for _ in range(rounds):
            result.append(dict(visit=visit, start=used, stop=used + active, width=active))
            used += active
            visit += 1
        active = 1 if actions == 1 else max(2, active // 2)
    return result


def recommend(result, mode='legacy'):
    """Return action index and an auditable recommendation, using root data only.

    A partial sweep cannot eliminate its unvisited rivals. Use the most recent
    completed sweep's cohort, with latest *completed* estimates. Before any
    completed sweep, retain the baseline. Value ranking requires at least two
    visits for every member; one root-value pseudo-observation shrinks noisy
    means. Prior+Gumbel+Q breaks exact ties, but does not outweigh value evidence.
    These choices change neither allocation nor pi and are inference-only.
    """
    if mode not in MODES:
        raise ValueError('Unknown root recommendation')
    visits = np.asarray(result['visits'])
    prior = np.asarray(result['root_prior'])
    mean = np.asarray(result['mean_values'])
    cq = np.asarray(result['root_diagnostics']['completed_q'])
    noise = np.asarray(result['root_gumbel'])
    if (visits.ndim != 1 or not len(visits) or not np.isfinite(visits).all()
            or np.any(visits < 0) or np.any(visits != np.floor(visits))
            or any(a.shape != visits.shape for a in (prior, mean, cq, noise))
            or any(not np.isfinite(a).all() for a in (prior, mean, cq, noise))
            or np.any(prior < 0) or not np.isclose(prior.sum(), 1.)):
        raise ValueError('Invalid root statistics')
    score = np.log(np.maximum(prior, 1e-300)) + noise + cq
    eligible = np.flatnonzero(visits == visits.max()).tolist()
    baseline = int(result['search_choice'])
    if not 0 <= baseline < len(visits):
        raise ValueError('Invalid baseline action')
    chosen = baseline
    completed = []
    fallback = None
    ranking = score
    if mode != 'legacy':
        trace = result.get('root_trace')
        if trace is None:
            raise ValueError('Completed-sweep recommendation requires root trace')
        if len(trace) != int(visits.sum()) or len(trace) != result['simulations']:
            raise ValueError('Trace must contain only completed simulations')
        indexes = [row['chosen'] for row in trace]
        if any(type(i) is not int or not 0 <= i < len(visits) for i in indexes):
            raise ValueError('Invalid trace action')
        if not np.array_equal(np.bincount(indexes, minlength=len(visits)), visits):
            raise ValueError('Trace action counts do not match visits')
        candidates = min(len(visits), result['root_diagnostics']['candidate_cap'], result['requested'])
        for sweep in sweeps(candidates, result['requested']):
            if sweep['stop'] > len(trace):
                break
            rows = trace[sweep['start']:sweep['stop']]
            cohort = [int(row['chosen']) for row in rows]
            if (len(set(cohort)) != sweep['width'] or
                    any(row['considered_visit'] != sweep['visit'] for row in rows)):
                raise ValueError('Trace is not a sequential-halving sweep')
            completed.append(dict(sweep, cohort=cohort))
        if completed:
            eligible = completed[-1]['cohort']
            if mode == 'completed_sweep_value' and min(visits[eligible]) >= 2:
                raw = float(result['value'])
                if (result.get('q_scale') != 'natural_wdl' or not np.isfinite(raw)
                        or abs(raw) > 1 + 1e-6 or np.any(np.abs(mean[eligible]) > 1 + 1e-6)):
                    raise ValueError('Value recommendation requires bounded WDL')
                ranking = (mean * visits + raw) / (visits + 1)
            elif mode == 'completed_sweep_value':
                fallback = 'fewer_than_two_visits_per_finalist'
            chosen = max(eligible, key=lambda i: (ranking[i], score[i], -i))
        else:
            fallback = 'no_completed_sweep'
    return int(chosen), dict(mode=mode, baseline=baseline, selected=int(chosen),
                             eligible=eligible, completed_sweeps=completed,
                             scores=ranking.tolist(), combined_scores=score.tolist(), fallback=fallback)
