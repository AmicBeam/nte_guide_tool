"""Historical seven-team equivalence proof; never fresh GPU collection or fitting.

Every declared archived game must pass current rules/teacher/observation replay
before an isolated copied scope can qualify. Existing datasets and models are
immutable. Staged numerical models remain unapproved research artifacts.
"""
from copy import deepcopy
from dataclasses import dataclass
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
import time

import numpy as np

from . import seven_covered_value as cv
from .seven_auxiliary_jobs import auxiliary_jobs, AUXILIARY_DECISION_LIMIT
from .seven_schedule import ROSTER

_PHASES=('training','selection','heldout')
_BASES=dict(training='foundation',selection='selection',heldout='confirmation')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _safe(root,name):
    if not isinstance(name,str) or not name or Path(name).is_absolute():
        raise ValueError('Relative source evidence path required')
    root=Path(root).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Source evidence escapes auxiliary directory or is missing')
    return path


def _deadline(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<=time.time():
        raise TimeoutError('Live bounded reattestation deadline required')


@dataclass(frozen=True)
class ReattestationPlan:
    source: Path
    scope: dict
    config: dict
    starting_decks: dict
    jobs: tuple
    source_scope_sha256: str
    source_config_sha256: str
    model_files: dict
    plan_sha256: str


@dataclass(frozen=True)
class ReattestationShard:
    plan_sha256: str
    rule_hash: str
    algorithm_sha256: str
    verifier_sha256: str
    jobs: tuple
    historical_only: bool=True
    fresh_gpu_collection: bool=False


def _verifier_identity():
    return _digest(__file__)


def prepare_reattestation(source, *, starting_decks):
    """Verify source hashes and reconstruct unchanged broad-build phase plans."""
    from .build_acceptance import canonical_build, SEED_PHASES
    source=Path(source).resolve();scope=_json(source/'scope.json');config=_json(source/'config.json')
    if scope.get('schema')!='seven_expert_value_v2' or scope.get('passed') is not True:
        raise ValueError('Existing passed seven-team v2 auxiliary scope required')
    if (scope.get('objective')!='visible_rule_after_six_exploration_actions'
            or scope.get('real_learner_updates')!=0
            or scope.get('primary_policy_arrays_unchanged') is not True):
        raise ValueError('Unchanged historical auxiliary objective required')
    counts=cv.phase_counts(scope)
    if set(starting_decks)!=set(ROSTER) or set(scope.get('models',{}))!=set(ROSTER):
        raise ValueError('All seven actual starting builds and source models required')
    starts={key:canonical_build(starting_decks[key]) for key in ROSTER}
    if set(config.get('seeds',{}))!=set(SEED_PHASES) or any(
            type(v)is not int or v<0 for v in config['seeds'].values()):
        raise ValueError('Original complete numeric seed reservation required')
    if config.get('phase_counts')!=counts:
        raise ValueError('Source config and scope phase counts differ')
    files=scope.get('files',{})
    if not isinstance(files,dict):raise ValueError('Verified source file manifest required')
    for name,digest in files.items():
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest) or _digest(_safe(source,name))!=digest:
            raise ValueError('Original auxiliary evidence changed')
    jobs=[];partitions=[];expected_files=set()
    for phase in _PHASES:
        name=phase+'.json'
        if name not in files:raise ValueError('Original phase summary missing')
        summary=_json(_safe(source,name))
        plan=auxiliary_jobs(config['seeds'][_BASES[phase]],counts[phase]//56,
                            decks=starts,deadline=config['deadline']-30)
        if (summary.get('complete') is not True or summary.get('games')!=counts[phase]
                or summary.get('seeds')!=[j['seed'] for j in plan]
                or summary.get('first')!=[j['first'] for j in plan]):
            raise ValueError('Original phase endpoints differ from exact reconstructed plan')
        actual={n for n in files if re.fullmatch(re.escape(phase)+r'-\d+-private\.json\.gz',n)}
        expected={f'{phase}-{i}-private.json.gz' for i in range(counts[phase])}
        if actual!=expected:raise ValueError('Exact declared private trajectory ordinals required')
        partitions.append(set(j['seed'] for j in plan))
        expected_files.update(expected);expected_files.add(name)
        jobs.extend((phase,i,job,files[f'{phase}-{i}-private.json.gz']) for i,job in enumerate(plan))
    if set(files)!=expected_files:
        raise ValueError('Only original phase summaries and exact private game files allowed')
    if any(partitions[i]&partitions[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Historical seed partitions overlap')
    model_files={}
    for key in ROSTER:
        for suffix in ('.json','.npz'):
            name='models/'+key+suffix;model_files[name]=_digest(_safe(source,name))
        if model_files['models/'+key+'.npz']!=scope['models'][key]:
            raise ValueError('Original model arrays differ from scope')
    scope_sha,config_sha=_digest(source/'scope.json'),_digest(source/'config.json')
    payload=dict(scope=scope_sha,config=config_sha,models=model_files,starts=starts,jobs=jobs)
    identity=sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return ReattestationPlan(source,deepcopy(scope),deepcopy(config),starts,tuple(jobs),scope_sha,config_sha,model_files,identity)


def _source_unchanged(plan):
    if not isinstance(plan,ReattestationPlan):raise TypeError('Actual reconstructed reattestation plan required')
    if (_digest(plan.source/'scope.json')!=plan.source_scope_sha256
            or _digest(plan.source/'config.json')!=plan.source_config_sha256
            or _json(plan.source/'scope.json')!=plan.scope or _json(plan.source/'config.json')!=plan.config):
        raise ValueError('Source proof or reconstructed plan changed')
    for name,digest in plan.model_files.items():
        if _digest(_safe(plan.source,name))!=digest:raise ValueError('Source models changed')
    payload=dict(scope=plan.source_scope_sha256,config=plan.source_config_sha256,
                 models=plan.model_files,starts=plan.starting_decks,jobs=plan.jobs)
    digest=sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=plan.plan_sha256:raise ValueError('Reattestation plan was altered')


def _oracle():
    from ..engine.duel_v2 import new_game,apply_action,observe,acting_side
    from .league_rollout import clean
    from .seven_lineup import encode
    from .covered_rollout import rule_decision
    from .policies import VisibleEngineRulePolicy
    return new_game,apply_action,observe,acting_side,clean,encode,rule_decision,VisibleEngineRulePolicy()


def _game(plan,phase,ordinal):
    name=f'{phase}-{ordinal}-private.json.gz';path=_safe(plan.source,name)
    if _digest(path)!=plan.scope['files'][name]:raise ValueError('Archived game changed')
    with gzip.open(path,'rt',encoding='utf8')as stream:return json.load(stream)


def _verify_game(game,job,*,deadline):
    new_game,apply_action,observe,acting_side,clean,encode,rule_decision,reference_policy=_oracle()
    if (game.get('complete') is not True or game.get('seed')!=job['seed']
            or game.get('first')!=job['first'] or game.get('rows')!=[]):
        raise ValueError('Historical trajectory identity or auxiliary-only contract differs')
    actions=game.get('actions');values=game.get('value_rows')
    if (not isinstance(actions,list) or not 7<=len(actions)<=AUXILIARY_DECISION_LIMIT
            or not isinstance(values,list)):
        raise ValueError('Natural complete auxiliary trajectory within512 actions required')
    rows={}
    for row in values:
        if not isinstance(row,dict) or type(row.get('step'))is not int or row.get('side') not in ('a','b'):
            raise ValueError('Exact value row coordinates required')
        coordinate=row['step'],row['side']
        if coordinate in rows:raise ValueError('Duplicate historical value row')
        rows[coordinate]=row
    if set(rows)!={(step,side)for step in range(6,len(actions))for side in ('a','b')}:
        raise ValueError('Complete exact dual-viewer post-exploration value rows required')
    state=new_game(seed=job['seed'],first_side=job['first'],decks=deepcopy(job['decks']))
    rng=np.random.default_rng(job['seed'])
    for step,entry in enumerate(actions):
        _deadline(deadline)
        if state.get('phase')=='finished':raise ValueError('Action after natural terminal')
        actor=acting_side(state);legal,index=rule_decision(state,actor)
        reference=observe(state,actor,include_previews=True)
        ref_legal=[v['action']for v in reference['legal_actions']if v['action']['type']!='concede']
        if legal!=ref_legal or index!=reference_policy(reference,tuple(ref_legal)):
            raise ValueError('Current selective teacher differs from full preview rule policy')
        if step<6 and state['phase']=='playing':index=int(rng.integers(len(legal)))
        if entry!=dict(side=actor,action=legal[index]):raise ValueError('Historical action differs from current rule/exploration')
        if step>=6:
            for side in ('a','b'):
                row=rows[step,side];x,_=encode(observe(state,side,include_previews=False),[])
                archived=np.asarray(row.get('x'),dtype=np.float32)
                if (row.get('key')!=job['keys'][side] or type(row.get('value_actor'))not in (int,float)
                        or row.get('value_actor')!=int(actor==side)
                        or archived.shape!=x.shape or not np.isfinite(archived).all()
                        or not np.array_equal(archived,x)):
                    raise ValueError('Current dual-viewer value input/context/key differs')
        state=clean(apply_action(state,actor,deepcopy(entry['action'])))
    winner=state.get('winner')
    if (state.get('phase')!='finished' or winner not in ('a','b','draw') or winner!=game.get('winner')
            or state.get('reason') in (None,'','timeout','deadline','truncated','action_limit')):
        raise ValueError('Current natural terminal winner/reason differs')
    if any(type(r.get('z'))not in (int,float) or r['z']!=(0 if winner=='draw'else 1 if r['side']==winner else -1)
           for r in values):raise ValueError('Historical terminal label differs')
    return len(actions),len(values)


def verify_reattestation_shard(plan, *, start, max_jobs, deadline):
    """Replay at most N ordinal jobs. No models, optimization, or GPU collection."""
    from .seven_lineup import identity
    _deadline(deadline);_source_unchanged(plan)
    if (type(start)is not int or not 0<=start<len(plan.jobs)
            or type(max_jobs)is not int or not 1<=max_jobs<=len(plan.jobs)):
        raise ValueError('Bounded existing ordinal range required')
    evidence=[]
    for absolute in range(start,min(start+max_jobs,len(plan.jobs))):
        _deadline(deadline);phase,ordinal,job,digest=plan.jobs[absolute]
        actions,rows=_verify_game(_game(plan,phase,ordinal),job,deadline=deadline)
        evidence.append(dict(absolute=absolute,phase=phase,ordinal=ordinal,file_sha256=digest,
                             actions=actions,value_rows=rows))
    return ReattestationShard(plan.plan_sha256,identity(),cv.identity(),_verifier_identity(),tuple(evidence))


def stage_reattestation_models(plan, output, *, teacher_directory):
    """Stage exact arrays and tightly checked research metadata in a NEW folder.

    This never creates a scope or approval. Model compatibility and current
    numerical runtime loading are checked before metric evaluation is allowed.
    """
    from . import seven_policy as architecture, seven_grounded as grounded, seven_runtime as rt
    from .seven_lineup import identity,all_features,candidate_names
    from .build_acceptance import canonical_build
    from .league_schema import build_hash
    from .cross_teacher_distillation import verify_old_teacher, named_index_map
    _source_unchanged(plan);output=Path(output)
    if output.exists():raise ValueError('New isolated unapproved staging directory required')
    manifests={}
    for key in ROSTER:
        m=_json(_safe(plan.source,'models/'+key+'.json'))
        if (m.get('schema')!=architecture.SCHEMA or m.get('deck')!=key
                or m.get('architecture_sha256')!=architecture.fingerprint()
                or m.get('candidate_transform_sha256')!=grounded.fingerprint()
                or m.get('features')!=all_features() or m.get('candidates')!=candidate_names()
                or m.get('grounded_features')!=grounded.transformed_names()
                or m.get('sha256')!=plan.scope['models'][key]
                or canonical_build(m.get('build'))!=plan.starting_decks[key]
                or m.get('build_sha256')!=build_hash(plan.starting_decks[key])
                or m.get('wdl_order')!=['loss','draw','win']
                or m.get('value_context')!=['viewer_is_actor'] or m.get('reward_mode')!='terminal_wdl_only'
                or m.get('equal_input_scores')!='canonical_first_v1'
                or m.get('automatic_serving_approval') is not False
                or m.get('origin',{}).get('usage')!='covered_expert_value_auxiliary'):
            raise ValueError('Historical numerical feature/schema/architecture/build contract differs')
        if m.get('source_identity'):
            teacher=verify_old_teacher(teacher_directory,key)
            old=m['source_identity']
            current={n:teacher[n]for n in old if n not in ('observation_index','candidate_index')}
            current.update(observation_index=named_index_map(teacher['features'],all_features()),
                           candidate_index=named_index_map(teacher['candidates'],candidate_names()))
            if current!=old:raise ValueError('Current explicit teacher dependency differs')
        manifests[key]=dict(m,rule_hash=identity(),automatic_serving_approval=False,validation=dict(approved=False))
    output.mkdir(parents=True)
    from .seven_round import write
    write(output/'staging.json',dict(schema='seven_auxiliary_equivalence_staging_v1',
          source_scope_sha256=plan.source_scope_sha256,plan_sha256=plan.plan_sha256,
          historical_reattestation=True,fresh_gpu_collection=False,qualified=False,serving_approved=False))
    (output/'models').mkdir()
    for key,m in manifests.items():
        name='models/'+key+'.npz';shutil.copyfile(_safe(plan.source,name),output/name)
        if _digest(output/name)!=plan.model_files[name]:raise ValueError('Staged arrays changed')
        write(output/'models'/(key+'.json'),m)
    # Actual runtime rechecks shapes, finite tensors, normalization and embedded
    # frozen teachers against its current dependency directory as well.
    return {key:rt.model(output/'models',key)for key in ROSTER}


def finalize_reattestation(plan, shards, output, *, deadline, teacher_directory,
                           metric_backend='numpy', device=None):
    """Qualify only complete current replay plus unchanged-model recalibration.

    Existing epoch/temperature choices are retained; this performs neither new
    selection nor an independent confirmation, and never calls an optimizer.
    """
    from .seven_lineup import identity
    from . import seven_round as rr
    _deadline(deadline);_source_unchanged(plan)
    if metric_backend not in ('numpy','cuda_batched') or (metric_backend=='cuda_batched'and device!='cuda'):
        raise ValueError('Explicit supported metric backend required')
    found={}
    for shard in shards:
        if (not isinstance(shard,ReattestationShard) or shard.plan_sha256!=plan.plan_sha256
                or shard.rule_hash!=identity() or shard.algorithm_sha256!=cv.identity()
                or shard.verifier_sha256!=_verifier_identity() or not shard.historical_only
                or shard.fresh_gpu_collection):raise ValueError('Current historical replay shard binding required')
        for item in shard.jobs:
            index=item['absolute']
            if type(index)is not int or not 0<=index<len(plan.jobs) or index in found:
                raise ValueError('Duplicate/unplanned reattestation job')
            phase,ordinal,job,digest=plan.jobs[index]
            if (item.get('phase')!=phase or item.get('ordinal')!=ordinal or item.get('file_sha256')!=digest
                    or type(item.get('actions'))is not int or not 7<=item['actions']<=512
                    or item.get('value_rows')!=2*(item['actions']-6)):
                raise ValueError('Shard coordinates or complete dual-viewer evidence differs')
            found[index]=item
    if set(found)!=set(range(len(plan.jobs))):raise ValueError('Every declared historical phase game must verify')
    output=Path(output)
    models=stage_reattestation_models(plan,output,teacher_directory=teacher_directory)
    phases={}
    for phase in ('selection','heldout'):
        phases[phase]=[]
        for p,ordinal,job,digest in plan.jobs:
            if p==phase:
                _deadline(deadline);game=_game(plan,p,ordinal)
                for row in game['value_rows']:row['x']=np.asarray(row['x'],dtype=np.float32)
                phases[phase].append(game)
    selection=rr.value_metrics(phases['selection'],models,backend=metric_backend,device=device)
    _deadline(deadline)
    heldout=rr.value_metrics(phases['heldout'],models,backend=metric_backend,device=device)
    passed=all(set(metrics)==set(ROSTER)and all(m.get('normal_games',0)>=8
        and np.isfinite(m.get('brier',np.nan))and m['brier']<.7
        and np.isfinite(m.get('log_loss',np.nan))and m['log_loss']<1.2 for m in metrics.values())
        for metrics in (selection,heldout))
    for name,digest in plan.scope['files'].items():
        destination=output/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(_safe(plan.source,name),destination)
        if _digest(destination)!=digest:raise ValueError('Copied historical evidence differs')
    shutil.copyfile(plan.source/'config.json',output/'config.json')
    report=dict(deepcopy(plan.scope),passed=passed,rule_hash=identity(),algorithm_sha256=cv.identity(),
        metrics=heldout,selection_metrics=selection,
        reattestation=dict(historical_only=True,fresh_gpu_collection=False,fresh_confirmation=False,
            weight_updates=0,epoch_selection=False,temperature_selection=False,
            source_scope_sha256=plan.source_scope_sha256,plan_sha256=plan.plan_sha256,
            counts=cv.phase_counts(plan.scope),actions=sum(v['actions']for v in found.values()),
            verifier_sha256=_verifier_identity(),exact_rule_decisions=True))
    rr.write(output/'scope.json',report)
    _deadline(deadline);_source_unchanged(plan)
    return cv.load_scope(output)
