"""Audit actions the current representation cannot distinguish.

Representation equality is not a proof of rule equivalence. These groups
are useful for diagnostics and irreducible-loss bounds only; do not silently
merge game actions or change inference based on this audit.
"""
import numpy as np


def representation_groups(candidates):
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.ndim != 2 or not len(candidates) or not np.isfinite(candidates).all():
        raise ValueError('Nonempty finite candidate matrix required')
    groups = {}
    for index, vector in enumerate(candidates):
        vector = np.where(vector == 0, np.float32(0), vector)
        groups.setdefault(vector.tobytes(), []).append(index)
    return list(groups.values())


def grouped_target(candidates, pi):
    pi = np.asarray(pi, dtype=np.float64)
    groups = representation_groups(candidates)
    if (pi.shape != (len(candidates),) or not np.isfinite(pi).all()
            or (pi < 0).any() or not np.isclose(pi.sum(), 1.)):
        raise ValueError('Invalid full policy target')
    mass = np.asarray([pi[indexes].sum() for indexes in groups])
    return groups, mass


def irreducible_kl(candidates, pi):
    """Minimum KL to any model giving identical feature rows identical logits."""
    groups, mass = grouped_target(candidates, pi)
    pi = np.asarray(pi, dtype=np.float64)
    projected = np.zeros_like(pi)
    for indexes, value in zip(groups, mass):
        projected[indexes] = value / len(indexes)
    positive = pi > 0
    return float(np.sum(pi[positive] * np.log(pi[positive] / projected[positive])))


def representation_top_hit(candidates, pi, selected):
    groups, _ = grouped_target(candidates, pi)
    if not 0 <= selected < len(candidates):
        raise ValueError('Selected action outside candidate list')
    target = int(np.argmax(pi))
    return any(target in indexes and selected in indexes for indexes in groups)


def canonical_root(x, candidates, pi):
    """Order-independent identity and group target; retains all observed data."""
    from hashlib import sha256
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Invalid visible observation')
    groups, mass = grouped_target(candidates, pi)
    ordered = sorted((np.asarray(candidates[indexes[0]], dtype=np.float32).tobytes(), float(value))
                     for indexes, value in zip(groups, mass))
    return sha256(x.tobytes() + b''.join(vector for vector, _ in ordered)).hexdigest(), np.array([value for _, value in ordered])
