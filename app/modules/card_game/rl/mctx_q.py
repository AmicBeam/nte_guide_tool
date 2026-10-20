"""NumPy Q profiles shared by offline Mctx comparisons.

This is an algorithm adaptation, not a JAX runtime dependency. Values retain
the fixed-observer terminal-WDL contract; profiles only change search scaling.
"""
import numpy as np

Q_SCALES = ('natural_wdl', 'mctx_default')


def completed_q(prior, visits, total, value, legal, *, q_scale='mctx_default'):
    if q_scale not in Q_SCALES:
        raise ValueError('Unknown Mctx Q profile')
    prior = np.asarray(prior, dtype=np.float64)
    visits = np.asarray(visits)
    total = np.asarray(total, dtype=np.float64)
    legal = np.asarray(legal, dtype=bool)
    if (prior.ndim != 1 or prior.shape != visits.shape or prior.shape != total.shape
            or prior.shape != legal.shape or not legal.any()
            or not np.issubdtype(visits.dtype, np.integer) or (visits < 0).any()
            or not np.isfinite(prior).all() or (prior < 0).any()
            or not np.isfinite(total).all() or not np.isfinite(value)
            or (prior[~legal] != 0).any() or (visits[~legal] != 0).any()
            or not np.isclose(prior.sum(), 1., atol=1e-12, rtol=1e-10)):
        raise ValueError('Finite full legal-action statistics required')
    seen = (visits > 0) & legal
    q = np.divide(total, visits, out=np.zeros_like(total), where=seen)
    mass = prior[seen].sum()
    mean = np.dot(prior[seen], q[seen]) / mass if mass > 0 else value
    mixed = (value + visits[legal].sum() * mean) / (1 + visits[legal].sum())
    q = np.where(seen, q, mixed)
    if abs(value) > 1 + 1e-6 or (np.abs(q[legal]) > 1 + 1e-6).any():
        raise ValueError('Fixed-observer WDL outside [-1,1]')
    if q_scale == 'mctx_default':
        low, high = q[legal].min(), q[legal].max()
        q = (q - low) / max(float(high - low), 1e-8)
    gain = .1 * (50 + visits[legal].max())
    return np.where(legal, q * gain, 0.)
