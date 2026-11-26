"""Frozen broad expert-value surrogate; separate from learner and service arrays."""
from hashlib import sha256
import json
from pathlib import Path
import time
import numpy as np

from .seven_auxiliary_jobs import AUXILIARY_DECISION_LIMIT


def identity():
    root=Path(__file__).parent;digest=sha256()
    for name in ('seven_covered_value.py','covered_rollout.py','seven_sampler.py','policies.py','seven_auxiliary_jobs.py'):
        digest.update((root/name).read_bytes())
    return digest.hexdigest()


def computation_identity(root=None):
    """Bind the actual auxiliary target/inference code, separately from world sampling."""
    import ast
    root=Path(root) if root is not None else Path(__file__).parent
    tree=ast.parse((root/'seven_covered_value.py').read_text())
    functions={node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    digest=sha256()
    for name in ('collect_game','expert_leaf'):
        digest.update(ast.dump(functions[name],include_attributes=False).encode())
    for name in ('covered_rollout.py','policies.py','seven_auxiliary_jobs.py','seven_runtime.py',
                 'outcome_runtime.py','seven_policy.py','seven_lineup.py','seven_grounded.py','information_search.py'):
        digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()


def _algorithm_binding_matches(folder,report):
    if report.get('algorithm_sha256')==identity():return True
    path=folder/'value-computation-compatibility.json'
    if not path.is_file():return False
    receipt=json.loads(path.read_text())
    expected=dict(schema='covered_value_unchanged_computation_v1',
        original_report_sha256=sha256((folder/'scope.json').read_bytes()).hexdigest(),
        original_algorithm_sha256=report.get('algorithm_sha256'),rule_hash=report.get('rule_hash'),
        models=report.get('models'),value_computation_sha256=computation_identity(),
        original_value_computation_sha256=computation_identity(),
        accepted_sampler_sha256=sha256((Path(__file__).parent/'seven_sampler.py').read_bytes()).hexdigest())
    return all(receipt.get(key)==value for key,value in expected.items())


def phase_counts(report):
    if report.get('schema')=='seven_expert_value_v1':
        return dict(training=168,selection=56,heldout=56)
    counts=report.get('phase_counts',{})
    if (report.get('schema')!='seven_expert_value_v2'
            or set(counts)!={'training','selection','heldout'}
            or any(type(v) is not int or v%56 for v in counts.values())
            or not 168<=counts['training']<=672
            or not 56<=counts['selection']<=168
            or counts['heldout']!=counts['selection']):
        raise ValueError('Bounded covered value phase counts required')
    return counts


def load_scope(folder):
    from . import seven_runtime as rt
    from .seven_schedule import ROSTER
    from .seven_lineup import identity as rules
    folder=Path(folder);report=json.loads((folder/'scope.json').read_text())
    if (report.get('schema') not in ('seven_expert_value_v1','seven_expert_value_v2') or report.get('passed') is not True
            or report.get('rule_hash')!=rules() or not _algorithm_binding_matches(folder,report)
            or report.get('objective')!='visible_rule_after_six_exploration_actions'
            or set(report.get('models',{}))!=set(ROSTER)):
        raise ValueError('Covered expert value scope is unqualified')
    for name,digest in report.get('files',{}).items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Covered value raw evidence changed')
    if not all(name in report.get('files',{}) for name in ('heldout.json','selection.json','training.json')):
        raise ValueError('Covered value raw evidence missing')
    phases={p:json.loads((folder/(p+'.json')).read_text()) for p in ('training','selection','heldout')}
    for phase,count in phase_counts(report).items():
        entry=phases[phase]
        if entry.get('complete') is not True or entry.get('games')!=count or len(entry.get('seeds',[]))!=count:
            raise ValueError('Covered value phase endpoints incomplete')
        if len([n for n in report['files'] if n.startswith(phase+'-') and n.endswith('-private.json.gz')])!=count:
            raise ValueError('Covered value private trajectories missing')
    sets=[set(v['seeds']) for v in phases.values()]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Covered value seed partitions overlap')
    models={k:rt.model(folder/'models',k) for k in ROSTER}
    for key,model in models.items():
        metric=report['metrics'].get(key,{})
        if (model.manifest['origin'].get('usage')!='covered_expert_value_auxiliary'
                or model.version!=report['models'][key] or metric.get('normal_games',0)<8
                or not np.isfinite(metric.get('brier',np.nan)) or metric['brier']>=.7
                or not np.isfinite(metric.get('log_loss',np.nan)) or metric['log_loss']>=1.2):
            raise ValueError('Covered expert value heldout calibration failed')
    return models,report


def collect_game(job):
    from .covered_rollout import rule_decision
    from .seven_lineup import encode
    from .league_rollout import clean
    from ..engine.duel_v2 import new_game,acting_side,observe
    from ..engine.duel_v2.simulation import simulate_action
    from .information_search import terminal_value
    state=new_game(seed=job['seed'],first_side=job['first'],decks=job['decks'])
    rng=np.random.default_rng(job['seed']);rows=[];trace=[]
    for step in range(AUXILIARY_DECISION_LIMIT):
        if state['phase']=='finished':break
        if time.time()>=job['deadline']:return dict(complete=False,rows=[],value_rows=[])
        actor=acting_side(state);legal,index=rule_decision(state,actor)
        if step>=6:
            for side in ('a','b'):
                x,_=encode(observe(state,side,include_previews=False),[])
                rows.append(dict(x=x,side=side,key=job['keys'][side],value_actor=int(side==actor),step=step))
        elif state['phase']=='playing':index=int(rng.integers(len(legal)))
        trace.append(dict(side=actor,action=legal[index]));state=clean(simulate_action(state,actor,legal[index]))
    complete=state['phase']=='finished'
    if complete:
        for row in rows:row['z']=terminal_value(state,row['side'])
    return dict(complete=complete,rows=[],value_rows=rows if complete else [],actions=trace,
                winner=state.get('winner'),seed=job['seed'],first=job['first'])


def expert_leaf(state,viewer,values,runtime,deadline,*,horizon=3):
    from copy import deepcopy
    from .covered_rollout import rule_decision
    from .league_rollout import clean
    from .information_search import terminal_value
    from ..engine.duel_v2 import acting_side
    from ..engine.duel_v2.simulation import simulate_action
    world=deepcopy(state)
    for _ in range(horizon):
        if world['phase']=='finished':return terminal_value(world,viewer),True
        if time.time()>=deadline:return None,False
        actor=acting_side(world);legal,index=rule_decision(world,actor)
        world=clean(simulate_action(world,actor,legal[index]))
    if world['phase']=='finished':return terminal_value(world,viewer),True
    value=runtime.value_for(world,viewer,values)
    if value is None or not np.isfinite(value) or abs(value)>1+1e-6:raise ValueError('Invalid covered expert value')
    return float(value),True
