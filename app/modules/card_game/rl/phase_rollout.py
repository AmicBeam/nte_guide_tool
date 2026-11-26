"""Phase-model validation rollout; historical seven runtime remains immutable.

The hybrid production collector remains TrainingRollout. This adapter supports
pure/value-calibration and bounded CPU reference jobs with the same encodings.
"""
from .policy_runtime import model
from .seven_runtime import decision, encode, value_precheck_ok


def search_game(job):
    """Bounded recovery validation using the shared information-set search.

    Not a formal trainer: returns learning rows only after a complete real
    game. Full Gumbel pi, selected action, per-observer terminal WDL, and
    opponent-turn value context are retained. No shaped rewards or dark cards.
    """
    import hashlib
    import time
    from .information_search import search, terminal_value
    from .league_rollout import clean
    from ..engine.duel_v2 import new_game, apply_action, acting_side, observe
    import sys
    from . import seven_runtime as runtime
    state = new_game(seed=job['seed'], first_side=job['first'], decks=job['decks'])
    policies = {side: model(*spec) for side, spec in job['policies'].items()}
    value_policies=None
    if not job.get('pure',False) and job.get('teacher_mode')=='covered_value':
        from .seven_covered_value import load_scope
        auxiliary,_=load_scope(job['covered_value_source'])
        value_policies={side:auxiliary[spec[1]] for side,spec in job['policies'].items()}
    if not job.get('pure', False):
        # A new value head is not a calibrated search teacher. Explicitly
        # marked diagnosis may inspect it, but cannot become training data.
        if not value_precheck_ok(job.get('value_precheck'), policies):
            if not job.get('uncalibrated_value_diagnostic', False) or job.get('training', False):
                raise ValueError('WDL precheck required before recovery search learning')
    rows = []; values = []; trace = []; roots = []
    preserved_roots = 0
    behavior = {}
    references = {side: model(*spec) for side, spec in job.get('references', {}).items()}
    for step in range(job.get('max_actions', 800)):
        if state['phase'] == 'finished':
            break
        if time.time() >= job['deadline']:
            return dict(complete=False, reason='deadline', rows=[], value_rows=[], episode_actions=[], roots=roots)
        actor = acting_side(state)
        for side in ('a', 'b'):
            x, _ = encode(observe(state, side, include_previews=False), [])
            values.append(dict(x=x, side=side, key=job['policies'][side][1], step=step,
                               value_actor=int(side == actor)))
        if job.get('pure', False):
            actions, (x, c) = decision(state, actor)
            scores = policies[actor].scores(x, c)
            if job.get('require_preservation'):
                import numpy as np
                from .cross_teacher_distillation import project_observation, project_candidates, teacher_scores
                teacher = policies[actor].teacher
                old_c, rejected = project_candidates(c, teacher)
                if rejected.any():
                    raise ValueError('Preservation root contains unsupported action')
                reference = teacher_scores(teacher, project_observation(x, teacher), old_c)
                if not np.array_equal(scores, reference):
                    raise ValueError('Frozen behavior was not preserved')
                preserved_roots += 1
            chosen = int(scores.argmax())
            from .league_schema import ACT_PLAY
            import numpy as np
            b = behavior.setdefault(actor, dict(decisions=0, changed=0, legal_play=0, plays=0,
                                                reference_plays=0, entropy_sum=0., unsupported_legacy=0))
            b['decisions'] += 1
            probabilities = np.exp(scores.astype(np.float64) - np.logaddexp.reduce(scores.astype(np.float64)))
            b['entropy_sum'] += float(-(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum())
            if any(int(candidate[0]) == ACT_PLAY for candidate in c):
                b['legal_play'] += 1
                b['plays'] += int(int(c[chosen][0]) == ACT_PLAY)
            reference_choice = int(references[actor].scores(x, c).argmax()) if actor in references else chosen
            b['changed'] += int(chosen != reference_choice)
            b['reference_plays'] += int(int(c[reference_choice][0]) == ACT_PLAY)
            if getattr(policies[actor], 'teacher', None) is not None:
                b['unsupported_legacy'] += int(any(int(row[0])>=6 or int(row[1])>=10 or int(row[2])>=81 or int(row[4])>=10 for row in c))
            roots.append(dict(step=step, side=actor, chosen=actions[chosen], legal=len(actions),
                              reference_chosen=actions[reference_choice]))
        else:
            seed = (1 << 256) | int(hashlib.sha256(f"recovery:{job['seed']}:{step}:{actor}".encode()).hexdigest(), 16)
            q_scale = job.get('q_scale', 'legacy')
            if q_scale in ('natural_wdl', 'mctx_default'):
                from .recovery_search import search as recovery_search
                result = recovery_search(state, actor, policies, seed=seed, runtime=runtime,
                                         simulations=job.get('simulations', 32),
                                         gumbel_candidates=job.get('gumbel_candidates',16),
                                         terminal_horizon=job.get('terminal_horizon',3), max_depth=job.get('max_depth',10),
                                         world_sampling=job.get('world_sampling','per_simulation'), deadline=job['deadline'],
                                         noise=job.get('exploration',job.get('training', False)), q_scale=q_scale,
                                         teacher_mode=job.get('teacher_mode','network'),value_policies=value_policies)
            elif q_scale == 'legacy':
                result = search(state, actor, policies, seed=seed, simulations=job.get('simulations', 32),
                                algorithm='gumbel', gumbel_candidates=16, terminal_horizon=3,
                                runtime=runtime, deadline=job['deadline'], noise=job.get('exploration',job.get('training', False)),
                                diagnostics=True)
            else:
                raise ValueError('Unknown recovery Q scale')
            if not result['complete']:
                return dict(complete=False, reason='incomplete_search', rows=[], value_rows=[], episode_actions=[], roots=roots)
            actions = result['actions']; chosen = result['search_choice']
            from .guided_recovery import apply_guide
            chosen, guided, guide_audit = apply_guide(state, actor, actions, chosen, job, runtime, pi=result['pi'])
            row = dict(x=result['x'], c=result['c'], pi=result['pi'], visit_pi=result['visit_pi'],
                             selected=chosen, side=actor, key=job['policies'][actor][1], step=step,
                             phase=state['phase'], value_actor=1, search_budget=result['simulations'],
                             policy_source=result['policy_source'], search_algorithm='gumbel',
                             teacher_mode=result.get('teacher_mode','network'), q_scale=q_scale)
            row.update(guided)
            if guide_audit is not None: row['guide'] = guide_audit
            rows.append(row)
            roots.append(dict(step=step, side=actor, chosen=actions[chosen], legal=len(actions),
                              diagnostics=result['root_diagnostics']))
        if job.get('pure', False) and job.get('guided_policy'):
            from .guided_recovery import apply_guide
            chosen, _, audit = apply_guide(state, actor, actions, chosen, job, runtime)
            roots[-1].update(chosen=actions[chosen], guide=audit)
        trace.append(dict(side=actor, action=actions[chosen]))
        state = clean(apply_action(state, actor, actions[chosen]))
    complete = state['phase'] == 'finished'
    if complete:
        for row in rows + values:
            row['z'] = terminal_value(state, row['side'])
    return dict(complete=complete, reason=None if complete else 'action_limit', winner=state.get('winner'),
                rows=rows if complete else [], value_rows=values if complete else [],
                episode_actions=trace if complete else [], roots=roots, decisions=len(trace),
                searched=0 if job.get('pure') else len(rows), turn=state['turn'],
                preserved_roots=preserved_roots, behavior=behavior,
                learning_eligible=bool(job.get('training') and not job.get('pure')
                                       and value_precheck_ok(job.get('value_precheck'), policies)),
                model_versions={side: p.version for side, p in policies.items()})
