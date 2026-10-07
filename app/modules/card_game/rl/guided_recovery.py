"""Explicit, optional rule-guide distillation for complete recovery episodes.

The original Gumbel distribution stays in the record. The effective policy
target is a declared mixture; environment/value labels remain terminal WDL.
Serving never enables this hook implicitly.
"""
import time
import numpy as np

MODE = 'zhenhong_setup_v3_mix_v1'


def apply_guide(state, actor, actions, chosen, job, runtime, *, pi=None):
    mode = job.get('guided_policy')
    if mode is None:
        return chosen, {}, None
    if mode != MODE:
        raise ValueError('Unknown guided recovery policy')
    sides = job.get('guided_sides', job.get('train_sides', ()))
    if actor not in sides or job['policies'][actor][1] != 'zhenhong':
        return chosen, {}, None
    coefficient = job.get('guide_coefficient', .5)
    if not isinstance(coefficient, (int, float)) or not 0 < coefficient <= 1:
        raise ValueError('Invalid guide policy mixture')
    seconds = min(float(job.get('guide_seconds', .75)), job['deadline'] - time.time())
    if seconds <= .01:
        return chosen, {}, dict(skipped='deadline')
    from ..engine.ai.zhenhong_guide import recommend, VERSION
    recommendation = recommend(state, actor, actions[chosen], runtime=runtime,
                               seconds=seconds, max_plans=24,
                               seed=job['seed'] * 1000003 + int(state['version']))
    selected = recommendation['selected']
    if selected not in actions:
        raise ValueError('Guide returned an action outside the real legal root')
    index = actions.index(selected)
    audit = dict(version=VERSION, reason=recommendation['reason'],
                 changed=index != chosen, baseline=actions[chosen], selected=selected)
    if index == chosen or pi is None:
        return index, {}, audit
    original = np.asarray(pi, dtype=np.float32).copy()
    if (original.shape != (len(actions),) or not np.isfinite(original).all()
            or (original < 0).any() or not np.isclose(original.sum(), 1)):
        raise ValueError('Invalid original Gumbel target')
    guide = np.zeros_like(original); guide[index] = 1
    return index, dict(pi=(1-coefficient)*original + coefficient*guide,
                       original_search_pi=original, guide_pi=guide,
                       guide_coefficient=float(coefficient), guide_version=VERSION,
                       policy_source=MODE), audit
