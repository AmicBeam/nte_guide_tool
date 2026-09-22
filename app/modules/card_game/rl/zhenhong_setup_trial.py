"""Bounded, paired setup-reward trial using the shared information-search learner."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import hashlib,json,time,zipfile
import numpy as np
from .fixed_lineup import FixedModel,identity,write
from .ten_search_runtime import load_numeric,export_model
from .information_search import search_game,policy_value_update
from .information_longrun import append
from .information_experiment import compact
from .setup_reward import coefficient
from .full_cycle_experiment import evaluate_job

FOES=('starter','weave-rush','quick-rush')
ARMS=('control','setup')


def paired_rows(results):
    if set(results)!=set(ARMS) or any(not r['complete'] for r in results.values()):return None
    rows={arm:[r for r in results[arm]['rows'] if r['side']=='a' and r['key']=='zhenhong'] for arm in ARMS}
    if not all(rows.values()):return None
    for result in results.values():
        if result['decisions']!=result['searched'] or len(result['rows'])!=result['decisions']:raise ValueError('Unsearched training decision')
    return rows


def run(out,config):
    import torch
    torch.set_num_threads(1);torch.manual_seed(config['seeds']['foundation'])
    if config['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    out=Path(out);source=out/'sources';key='zhenhong'
    if identity()!=config['rule_hash']:raise ValueError('Rule identity mismatch')
    initial={k:FixedModel(source,k) for k in (key,*FOES)}
    originals={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    nets={};opts={};origins={};replay={a:[] for a in ARMS};steps={a:0 for a in ARMS};totals={a:dict(games=0,samples=0,setup_games=0,passive=0) for a in ARMS};current={}
    for arm in ARMS:
        nets[arm],origins[arm]=load_numeric(source,key,config['device']);opts[arm]=torch.optim.Adam(nets[arm].parameters(),lr=3e-5)
        current[arm]=out/arm/'initial';export_model(nets[arm],current[arm],key,initial[key].serving_deck,origin=origins[arm])
    groups=0;paired_updates=0
    def scale(arm):return coefficient(.1,origins[arm].get('setup_reward_updates',0)+steps[arm],64) if arm=='setup' else 0.
    def status(phase,**extra):write(out/'status.json',dict(phase=phase,steps=steps,training=totals,paired_updates=paired_updates,hard_deadline=config['hard_deadline'],**extra))
    with ProcessPoolExecutor(max_workers=config['workers'],mp_context=get_context('spawn')) as pool:
        while time.time()<config['train_until']-90 and not (config.get('smoke') and groups):
            # Four matched cases use eight workers; weights stay frozen for this batch.
            jobs=[];group_ids=list(range(groups,groups+(1 if config.get('smoke') else 4)))
            for group in group_ids:
                foe=FOES[(group//2)%len(FOES)];first=('a','b')[group%2]
                for arm in ARMS:
                    jobs.append(dict(arm=arm,group=group,seed=config['seeds']['foundation']+group,first=first,
                        deadline=min(config['train_until']-60,time.time()+1500),runtime='ten',simulations=config['simulations'],training=True,collect=True,train_sides=['a'],
                        zhenhong_passive_reward=.2,setup_rewards={'a':scale(arm)},failure_dir=str(out/'failure-roots'),
                        decks={'a':initial[key].serving_deck,'b':initial[foe].serving_deck},policies={'a':(str(current[arm]),key),'b':(str(source),foe)}))
            groups+=len(group_ids);pending={};status('searching',scheduled_groups=groups)
            futures={pool.submit(search_game,j):j for j in jobs}
            for future in as_completed(futures):
                j=futures[future];g=future.result();arm=j['arm'];group=j['group']
                append(out,'training-games.jsonl',dict(arm=arm,group=group,seed=j['seed'],first=j['first'],foe=j['policies']['b'][1],model_directory=str(current[arm]),setup_coefficient=j['setup_rewards']['a'],**compact(g)))
                pending.setdefault(group,{})[arm]=g
                if g['complete']:
                    totals[arm]['games']+=1;totals[arm]['setup_games']+=bool(g['setup_achieved']['a']);totals[arm]['passive']+=g['passive_triggers']['a']
                if len(pending[group])!=2:status('searching',scheduled_groups=groups);continue
                fresh=paired_rows(pending.pop(group))
                if fresh is None or time.time()>=config['train_until']:continue
                for a in ARMS:
                    totals[a]['samples']+=len(fresh[a]);replay[a]=(replay[a]+fresh[a])[-4096:]
                    if scale(a)==0:replay[a]=[r for r in replay[a] if not r.get('setup_reward',0)]
                before_steps=dict(steps)
                for update in range(1 if config.get('smoke') else 4):
                    if time.time()>=config['train_until']-2:break
                    if scale('setup')==0 and any(r.get('setup_reward',0)>0 for r in fresh['setup']):
                        replay={a:[] for a in ARMS};break
                    for a in ARMS:
                        rng=np.random.default_rng(config['seeds']['foundation']+group*100+update)
                        new=rng.choice(len(fresh[a]),min(64,len(fresh[a])),replace=False);old=rng.choice(len(replay[a]),min(64,len(replay[a])),replace=False)
                        result=policy_value_update(nets[a],opts[a],[fresh[a][i] for i in new]+[replay[a][i] for i in old],setup_coefficient=scale(a))
                        steps[a]+=1;append(out,'losses.jsonl',dict(arm=a,group=group,step=steps[a],**result))
                    paired_updates+=1
                if steps==before_steps:continue
                for a in ARMS:
                    destination=out/a/'generations'/f'update-{steps[a]:04}'
                    export_model(nets[a],destination,key,initial[key].serving_deck,origin=dict(origins[a],setup_reward_updates=origins[a].get('setup_reward_updates',0)+steps[a],setup_trial_arm=a))
                    # In-flight jobs retain their original immutable model directory.
                torch.save(dict(steps=steps,nets={a:n.state_dict() for a,n in nets.items()},opts={a:o.state_dict() for a,o in opts.items()}),out/'checkpoint.tmp')
                (out/'checkpoint.tmp').replace(out/'checkpoint.pt');status('updating',scheduled_groups=groups)
            for a in ARMS:
                if steps[a]:current[a]=out/a/'generations'/f'update-{steps[a]:04}'
            if not any(steps.values()) and time.time()>=config['train_until']-90:break
        assert steps['control']==steps['setup']
        for a in ARMS:export_model(nets[a],out/a/'final',key,initial[key].serving_deck,origin=dict(origins[a],setup_reward_updates=origins[a].get('setup_reward_updates',0)+steps[a],setup_trial_arm=a))
        write(out/'training-summary.json',dict(steps=steps,totals=totals,paired_updates=paired_updates,groups=groups,finished=time.time()))
        folder=out/'evaluation';(folder/'raw').mkdir(parents=True);(folder/'replays').mkdir();jobs=[]
        for variant,directory in [('initial',source),*[(a,out/a/'final') for a in ARMS]]:
            for fi,foe in enumerate(FOES):
                for i in range(2 if config.get('smoke') else 64):
                    for first in ('a','b'):
                        jobs.append(dict(id=f'{variant}-{foe}-{i}-{first}',cell=f'{variant}/{foe}',variant=variant,foe=foe,seed=config['seeds']['audit']+fi*10000+i,first=first,
                            deadline=config['evaluate_until'],runtime='ten',decks={'a':initial[key].serving_deck,'b':initial[foe].serving_deck},policies={'a':(str(directory),key),'b':(str(source),foe)},output=str(folder),replay=i==0))
        write(folder/'jobs.json',jobs);results=[]
        for r in pool.map(evaluate_job,jobs):results.append(r);append(folder,'index.jsonl',r);status('evaluation',completed=len(results),scheduled=len(jobs))
    metrics={}
    for variant in ('initial',*ARMS):
        selected=[r for r in results if r['variant']==variant];m=dict(games=0,wins=0,setup_games=0,passive_triggers=0,passive_games=0,immune_ally_down=0)
        for r in selected:
            if not r['complete']:continue
            triggers=r['passive_triggers']['a']
            m['games']+=1;m['wins']+=r['winner']=='a';m['setup_games']+=r['setup_achieved']['a'];m['passive_triggers']+=triggers;m['passive_games']+=triggers>0;m['immune_ally_down']+=r['immune_window']['a'].get('allies_down_while_immune',0)
        metrics[variant]=m
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in originals.items())
    summary=dict(complete=all(r['complete'] for r in results),training_updated=steps['setup']>0,equal_updates=steps['setup']==steps['control'],steps=steps,metrics=metrics,source_unchanged=True,automatic_serving_approval=False)
    write(out/'summary.json',summary)
    lines=['# 真红前置奖励一小时对照','', '两组同起点、同构筑、共同训练种子、相同优化次数；仅前置奖开关不同。实际被动奖励均为0.2。独立评估只看实际胜负，不加奖励。','', '| 方案 | 胜/完成 | 前置达成局 | 被动触发局/次数 |','| --- | --- | --- | --- |']
    for a,m in metrics.items():lines.append(f"| {a} | {m['wins']}/{m['games']} | {m['setup_games']} | {m['passive_games']}/{m['passive_triggers']} |")
    lines+=['',f'两组优化次数：{steps}。这是有限样本试验，不自动批准模型。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    members=[p for p in out.rglob('*') if p.is_file() and not any(x in p.relative_to(out).parts for x in ('frozen','generations')) and p.name not in ('delivery.zip','delivery.json','console.log')]
    with zipfile.ZipFile(out/'delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        manifest={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in members}
        for p in members:z.write(p,p.relative_to(out))
        z.writestr('manifest.json',json.dumps(manifest))
    write(out/'delivery.json',dict(sha256=hashlib.sha256((out/'delivery.zip').read_bytes()).hexdigest(),members=len(members)))
    status('finished',complete=summary['complete'])
