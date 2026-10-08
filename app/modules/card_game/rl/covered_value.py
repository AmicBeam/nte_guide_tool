"""Frozen broad expert-value surrogate; separate from learner and service arrays."""
from hashlib import sha256
import json
from pathlib import Path
import time
import numpy as np


def identity():
    root=Path(__file__).parent;digest=sha256()
    for name in ('covered_value.py','covered_rollout.py','recovery_sampler.py','policies.py'):
        digest.update((root/name).read_bytes())
    return digest.hexdigest()


def phase_counts(report):
    if report.get('schema')=='covered_expert_value_v1':
        return dict(training=90,selection=30,heldout=30)
    counts=report.get('phase_counts',{})
    if (report.get('schema')!='covered_expert_value_v2'
            or set(counts)!={'training','selection','heldout'}
            or any(type(v) is not int or v%30 for v in counts.values())
            or not 90<=counts['training']<=360
            or not 30<=counts['selection']<=90
            or counts['heldout']!=counts['selection']):
        raise ValueError('Bounded covered value phase counts required')
    return counts


def load_scope(folder):
    from . import recovery_runtime as rt
    from .cross_schedule import ROSTER
    from .cross_lineup import identity as rules
    folder=Path(folder);report=json.loads((folder/'scope.json').read_text())
    if (report.get('schema') not in ('covered_expert_value_v1','covered_expert_value_v2') or report.get('passed') is not True
            or report.get('rule_hash')!=rules() or report.get('algorithm_sha256')!=identity()
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
    from .cross_lineup import encode
    from .league_rollout import clean
    from ..engine.duel_v2 import new_game,acting_side,observe
    from ..engine.duel_v2.simulation import simulate_action
    from .information_search import terminal_value
    state=new_game(seed=job['seed'],first_side=job['first'],decks=job['decks'])
    rng=np.random.default_rng(job['seed']);rows=[];trace=[]
    for step in range(512):
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
