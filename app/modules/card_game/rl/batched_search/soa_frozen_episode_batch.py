"""Postbatch adapter for the existing seven-team episode consumer.

This reads completed, verified frozen engineering artifacts. It is not a live
GPU executor, does not satisfy the formal launch gate and never enables updates.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import numpy as np


class FrozenSoAEpisodeBatch:
    supports_full_duel_rules=False
    formal_training_qualified=False

    def __init__(self,report_path):
        from ..episode_replay import load_episode
        path=Path(report_path).resolve();root=path.parent
        report=json.loads(path.read_text(encoding='utf8'));count=report.get('rows')
        if (type(count) is not int or count<1 or report.get('passed') is not True
                or report.get('capture_training') is not True or report.get('weights_unchanged') is not True
                or report.get('frozen_collection_complete') is not True or report.get('natural_games')!=count
                or report.get('error_rows')!=0 or report.get('learning_updates')!=0
                or report.get('graph_executed') is not True or report.get('gpu_qualified') is not False):
            raise ValueError('Completed frozen engineering batch required; formal readiness is separate')
        search=report.get('search',{})
        if any(search.get(k)!=v for k,v in dict(simulations=32,gumbel_candidates=16,terminal_horizon=0,
                teacher_mode='network_value',opponent_mode='policy',cpu_fallback=False,world_sampling='per_decision').items()):
            raise ValueError('Frozen SoA search profile changed')
        proof=report.get('postbatch',{})
        if proof.get('verified') is not True or proof.get('information_hashes_verified') is not True:
            raise ValueError('Independent physical replay and information keys required')
        folder=root/'frozen-episodes';manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
        if manifest!=report.get('frozen_collection') or manifest.get('complete') is not True:
            raise ValueError('Collection manifest differs from the completed controller receipt')
        if manifest.get('optimizer_steps')!=0 or manifest.get('formal_training_qualified') is not False:
            raise ValueError('Frozen collection cannot imply optimizer or formal training approval')
        replay=json.loads((root/'raw-replay.json').read_text(encoding='utf8'))
        if len(manifest['episodes'])!=count or len(replay['final'])!=count or len(report['per_root'])!=count:
            raise ValueError('Omitted or duplicated frozen episodes')
        self.jobs=[];self._games={};total_policy=total_value=0
        for owner,(entry,final,bound) in enumerate(zip(manifest['episodes'],replay['final'],report['per_root'])):
            target=(folder/entry['path']).resolve()
            if target.parent!=folder.resolve() or sha256(target.read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError('Frozen episode path or content integrity failed')
            episode=load_episode(target);job=episode['job'];keys=bound['decks'];winner=final.get('winner')
            if (job.get('id')!='soa-frozen-'+str(owner) or job.get('seed')!=bound['seed']
                    or job.get('first')!=bound['first'] or job.get('left')!=keys['a'] or job.get('right')!=keys['b']
                    or final.get('phase')!='finished' or episode.get('winner')!=winner):
                raise ValueError('Frozen root/episode identity or natural terminal changed')
            if any(episode['model_versions'][side]!=report['weight_sha256'][keys[side]] for side in ('a','b')):
                raise ValueError('Frozen model version changed')
            policy=[r for r in episode['rows'] if 'pi' in r];values=[r for r in episode['rows'] if 'pi' not in r]
            n=len(policy)
            if n!=entry['policy_rows'] or len(values)!=entry['value_rows'] or len(values)!=2*n or n!=len(episode['actions']):
                raise ValueError('Full policy and both viewers required at every decision')
            raw=[r for r in replay['records'] if r['game']==owner]
            if episode['actions']!=[dict(side=r['actor'],action=r['raw_action']) for r in raw]:
                raise ValueError('Episode actions differ from the independently verified raw trajectory')
            if {(r['step'],r['side']) for r in values}!={(i,s) for i in range(n) for s in ('a','b')}:
                raise ValueError('Missing or duplicate WDL viewer')
            if len({(r['step'],r['side']) for r in values})!=len(values):
                raise ValueError('Duplicate WDL viewer')
            for step,(row,action) in enumerate(zip(policy,episode['actions'])):
                if (row['step']!=step or row['side']!=action['side'] or len(row['pi'])!=len(row['c'])
                        or row['search_budget']!=(1 if len(row['c'])==1 else 32)
                        or row['value_actor']!=1 or not 0<=row['selected']<len(row['c'])
                        or not np.isfinite(row['c']).all()):
                    raise ValueError('Policy/action coordinate mismatch')
                if not np.isfinite(row['pi']).all() or np.any(row['pi']<0) or not np.isclose(row['pi'].sum(),1,atol=1e-5):
                    raise ValueError('Invalid full policy target')
            for row in policy+values:
                z=0 if winner not in ('a','b') else (1 if winner==row['side'] else -1)
                if row['key']!=keys[row['side']] or row['z']!=z or not np.isfinite(row['x']).all():
                    raise ValueError('Frozen terminal label or public observation changed')
                if row['value_actor']!=int(row['side']==policy[row['step']]['side']):
                    raise ValueError('Fixed-viewer value context changed')
            digest=sha256(json.dumps(final,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
            game=dict(id=job['id'],complete=True,winner=winner,reason=None,rows=policy,value_rows=values,
                episode_actions=episode['actions'],decisions=n,searched=n,turn=final['turn'],
                final_state_sha256=digest,model_versions=episode['model_versions'],learning_eligible=False,
                rule_backend='native_soa_frozen_postbatch',formal_training_qualified=False)
            self.jobs.append(deepcopy(job));self._games[job['id']]=game;total_policy+=n;total_value+=len(values)
        if total_policy!=manifest['policy_rows'] or total_value!=manifest['value_rows']:
            raise ValueError('Frozen batch policy/value counts changed')
        self.receipt=dict(complete=True,games=count,policy_rows=total_policy,value_rows=total_value,
                         optimizer_steps=0,formal_training_qualified=False)

    def collect(self,jobs):
        jobs=list(jobs);ids=[j['id'] for j in jobs]
        if len(set(ids))!=len(ids) or set(ids)!=set(self._games):
            raise ValueError('Exact frozen episode cohort required')
        expected={job['id']:job for job in self.jobs}
        if any(any(job.get(k)!=expected[job['id']].get(k) for k in ('seed','first','left','right')) for job in jobs):
            raise ValueError('Frozen episode consumer identity changed')
        return [deepcopy(self._games[job['id']]) for job in jobs]
