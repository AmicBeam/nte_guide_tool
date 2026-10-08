"""Seven-team numerical/search runtime. Historical five-team contracts stay separate."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from . import seven_policy as architecture, seven_grounded as grounded
from .seven_lineup import encode, all_features, candidate_names, identity
from .league_schema import build_hash
from .seven_sampler import determinize
from .cross_runtime import passive_count
from .outcome_runtime import update


def decision(state,side):
    from ..engine.duel_v2 import observe
    view=observe(state,side,include_previews=False)
    actions=[entry['action'] for entry in view['legal_actions'] if entry['action']['type']!='concede']
    return actions,encode(view,actions)


def create_network(key,hidden=128,device='cpu',legacy=False):
    net=architecture.network(hidden,device)
    if legacy:net=architecture.attach_legacy(net,key)
    return net


def export(net,directory,key,build,origin):
    from .recovery_round import write
    from .build_acceptance import canonical_build
    from .seven_lineup import KEYS
    if key not in KEYS or net.policy_schema!=architecture.SCHEMA:raise ValueError('Seven-team policy required')
    build=canonical_build(build)
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    path=root/(key+'.npz');tmp=root/(key+'.tmp.npz')
    np.savez_compressed(tmp,**{name:t.detach().cpu().numpy() for name,t in net.state_dict().items()});tmp.replace(path)
    write(root/(key+'.json'),dict(schema=architecture.SCHEMA,deck=key,hidden=net.hidden,rule_hash=identity(),
        architecture_sha256=architecture.fingerprint(),candidate_transform_sha256=grounded.fingerprint(),
        features=all_features(),candidates=candidate_names(),grounded_features=grounded.transformed_names(),
        source_identity=getattr(net,'source_identity',None),equal_input_scores='canonical_first_v1',sha256=sha256(path.read_bytes()).hexdigest(),
        build=build,build_sha256=build_hash(build),origin=origin,wdl_order=['loss','draw','win'],
        value_context=['viewer_is_actor'],reward_mode='terminal_wdl_only',
        automatic_serving_approval=False,validation=dict(approved=False)))


class SevenModel:
    def __init__(self,directory,key):
        self.directory=str(directory)
        root=Path(directory);m=json.loads((root/(key+'.json')).read_text(encoding='utf8'))
        path=root/(key+'.npz');self.version=sha256(path.read_bytes()).hexdigest()
        if (m.get('schema')!=architecture.SCHEMA or m.get('deck')!=key or m.get('rule_hash')!=identity()
                or m.get('architecture_sha256')!=architecture.fingerprint()
                or m.get('candidate_transform_sha256')!=grounded.fingerprint()
                or m.get('features')!=all_features() or m.get('candidates')!=candidate_names()
                or m.get('grounded_features')!=grounded.transformed_names() or m.get('sha256')!=self.version
                or m.get('reward_mode')!='terminal_wdl_only' or m.get('automatic_serving_approval') is not False
                or m.get('equal_input_scores')!='canonical_first_v1'
                or m.get('wdl_order')!=['loss','draw','win'] or m.get('value_context')!=['viewer_is_actor']):
            raise ValueError('Seven-team numeric identity mismatch')
        self.hidden=m['hidden'];expected=architecture.tensor_shapes(self.hidden);self.teacher=None
        self.legacy_extra_indices=None
        if m.get('source_identity'):
            from .cross_teacher_distillation import verify_old_teacher, REPO_TEACHERS, named_index_map
            from .preserved_policy import encoded_name
            teacher=verify_old_teacher(REPO_TEACHERS,key)
            source={name:teacher[name] for name in m['source_identity'] if name not in ('observation_index','candidate_index')}
            source.update(observation_index=named_index_map(teacher['features'],all_features()),
                          candidate_index=named_index_map(teacher['candidates'],candidate_names()))
            if source!=m['source_identity']:raise ValueError('Seven-team frozen teacher changed')
            expected.update({'legacy.'+encoded_name(name):value.shape for name,value in teacher['weights'].items()})
            self.teacher=dict(teacher,**source)
            supported=set(source['candidates'])
            self.legacy_extra_indices=np.asarray([i for i,name in enumerate(candidate_names()) if name not in supported],dtype=np.int64)
        with np.load(path,allow_pickle=False) as arrays:
            if set(arrays.files)!=set(expected):raise ValueError('Seven-team tensor set mismatch')
            self.weights={n:arrays[n].copy() for n in arrays.files}
        for n,shape in expected.items():
            if self.weights[n].shape!=shape or not np.isfinite(self.weights[n]).all():raise ValueError('Seven-team tensor shape/nonfinite')
            self.weights[n].flags.writeable=False
        for prefix in ('state_net','value_net','cand_net'):
            active=self.weights[prefix+'.0.active']
            if not np.isin(active,[0,1]).all() or not active.any():raise ValueError('Invalid named normalization mask')
        if self.teacher:
            from .preserved_policy import encoded_name
            for n,w in self.teacher['weights'].items():
                if not np.array_equal(w,self.weights['legacy.'+encoded_name(n)]):raise ValueError('Frozen teacher arrays changed')
        self.manifest=m;self.serving_deck=m['build']
        if build_hash(self.serving_deck)!=m['build_sha256']:raise ValueError('Seven-team build mismatch')

    def scores(self,x,c):
        scores=architecture.scores_numpy(x,c,self.weights)
        if self.teacher is not None and len(c):
            from .fixed_lineup import SEATS,CARD_IDS
            from .cross_teacher_distillation import teacher_scores
            extra=self.legacy_extra_indices
            unknown=(c[:,0]>=6)|(c[:,1]>=len(SEATS))|(c[:,2]>=len(CARD_IDS))|(c[:,4]>=len(SEATS))|(np.abs(c[:,extra])>1e-8).any(-1)
            if not unknown.any():
                scores=scores+teacher_scores(self.teacher,x[self.teacher['observation_index']],c[:,self.teacher['candidate_index']])
        return architecture.canonical_scores_numpy(scores,c)
    scores_only=scores

    def wdl(self,x,is_actor):return architecture.wdl_numpy(x,is_actor,self.weights)
    def scores_value(self,x,c):
        p=self.wdl(x,True)
        return self.scores(x,c),float(p[2]-p[0])
    def validate_human(self,build):raise ValueError('Seven-team research model has no website authorization')


@lru_cache(maxsize=32)
def _load(directory,key,manifest_ns,weights_ns):return SevenModel(directory,key)


def model(directory,key):
    root=Path(directory)
    return _load(str(root),key,(root/(key+'.json')).stat().st_mtime_ns,(root/(key+'.npz')).stat().st_mtime_ns)


def restore(directory,key,device='cpu'):
    import torch
    policy=model(directory,key);net=create_network(key,policy.hidden,device,legacy=policy.teacher is not None)
    net.load_state_dict({n:torch.as_tensor(w.copy(),device=device) for n,w in policy.weights.items()})
    return net,policy.serving_deck,policy.manifest['origin']


def value_for(state,viewer,policies):
    from ..engine.duel_v2 import observe,acting_side
    x,_=encode(observe(state,viewer,include_previews=False),[])
    p=policies[viewer].wdl(x,acting_side(state)==viewer)
    return float(p[2]-p[0])


def predict_encoded(policy,x,c):return policy.scores_value(x,c)
def action_scores(policy,x,c):return policy.scores(x,c)


def value_precheck_ok(report, policies):
    """Bind the precheck to actual arrays; a launch boolean is not evidence."""
    import math
    from .seven_lineup import identity
    if (not isinstance(report, dict) or report.get('schema') != 'seven_value_precheck_v1'
            or report.get('complete') is not True or report.get('passed') is not True
            or report.get('heldout') is not True or report.get('rule_hash') != identity()):
        return False
    for policy in policies.values():
        key = policy.manifest['deck']
        entry = (report.get('by_version') or {}).get(policy.version)
        metrics = entry['metrics'] if entry else (report.get('metrics') or {}).get(key, {})
        version = entry.get('version') if entry else report.get('models', {}).get(key)
        if (version != policy.version
                or metrics.get('normal_games', 0) < 8):
            return False
        brier = metrics.get('brier'); logloss = metrics.get('log_loss')
        if (not isinstance(brier, (int, float)) or not isinstance(logloss, (int, float))
                or not math.isfinite(brier) or not math.isfinite(logloss)
                or not 0 <= brier < 2 / 3 or not 0 <= logloss < math.log(3)):
            return False
    return True


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
    runtime = sys.modules[__name__]
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
            if q_scale == 'natural_wdl':
                from .recovery_search import search as recovery_search
                result = recovery_search(state, actor, policies, seed=seed, runtime=runtime,
                                         simulations=job.get('simulations', 32),
                                         gumbel_candidates=job.get('gumbel_candidates',16),
                                         terminal_horizon=job.get('terminal_horizon',3), deadline=job['deadline'],
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
                             teacher_mode=result.get('teacher_mode','network'))
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
