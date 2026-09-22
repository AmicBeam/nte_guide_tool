"""Private decision-boundary checkpoints; no terminal labels before completion."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np

CONTRACT_FIELDS=('seed','first','decks','policies','runtime','simulations','max_actions','training','collect',
                 'record_episode','collect_value_only','train_sides','raw_sides','search_mix',
                 'zhenhong_passive_reward','setup_rewards','reuse_inference','fast_simulation','search_algorithm','gumbel_candidates')


def contract(job):
    return json.loads(json.dumps({k:job.get(k) for k in CONTRACT_FIELDS},sort_keys=True))


def algorithm_hash():
    root=Path(__file__).parent
    digest=hashlib.sha256()
    for name in ('information_search.py','trajectory_checkpoint.py','outcome_runtime.py','league_rollout.py',
                 'league_policy.py','fixed_lineup.py','ten_search_runtime.py','setup_reward.py','search_policy.py','grounded_candidates.py'):
        digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()


def save(path,job,versions,rule_hash,progress):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=dict(schema='private_search_trajectory_v1',contract=contract(job),models=versions,
              rule_hash=rule_hash,algorithm_hash=algorithm_hash(),hard_deadline=job['deadline'],progress=progress)
    def encode(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    temporary=path.with_suffix('.tmp')
    with gzip.open(temporary,'wt',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,default=encode)
    temporary.replace(path)


def load(path,job,versions,rule_hash):
    with gzip.open(path,'rt',encoding='utf-8') as f:data=json.load(f)
    if (data['schema']!='private_search_trajectory_v1' or data['contract']!=contract(job)
            or data['models']!=versions or data['rule_hash']!=rule_hash or data['algorithm_hash']!=algorithm_hash()):
        raise ValueError('Trajectory checkpoint identity mismatch')
    if job['deadline']>data['hard_deadline']:raise ValueError('Cannot extend trajectory deadline')
    p=data['progress']
    from ..engine.duel_v2.entities import hydrate_entities
    p['state']=hydrate_entities(p['state'])
    for r in p['rows']+p['value_rows']:
        if 'z' in r:raise ValueError('Pending trajectory cannot have terminal labels')
        for k in ('x','c','pi'):
            if k in r:r[k]=np.asarray(r[k],dtype=np.float32)
    if len(p['trace'])!=p['decisions']:raise ValueError('Incomplete action prefix')
    return p
