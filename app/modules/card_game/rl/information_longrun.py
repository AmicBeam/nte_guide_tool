"""Deadline-bounded search self-play with recoverable numeric replay and selection."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from copy import deepcopy
import hashlib,json,math,shutil,time,zipfile
import numpy as np
from .league_schema import PRESETS,deck,rule_hash,feature_names,CAND_DIM
from .league_policy import LeagueModel,export_model
from .information_search import search_game,policy_value_update
from .information_experiment import write,compact


def load_numeric(directory,key,device):
    import torch
    from .gpu_duel.ppo import CompactScorer
    policy=LeagueModel(directory,key)
    net=CompactScorer(len(feature_names()),CAND_DIM,policy.hidden)
    net.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in policy.weights.items()})
    return net.to(device),dict(source_sha256=policy.version,source_deck=key,source_kind='validated_numeric_v5')


def validate_deadlines(train_until,evaluate_until,hard_deadline,now=None):
    now=time.time() if now is None else now
    if not now<train_until<evaluate_until<hard_deadline:
        raise ValueError('Require now < training cutoff < evaluation cutoff < hard deadline')
    if hard_deadline-now>12*3600:raise ValueError('Maximum authorized long-run span is 12 hours')


def save_replay(path,rows):
    offsets=np.zeros(len(rows)+1,dtype=np.int64)
    for i,row in enumerate(rows):offsets[i+1]=offsets[i]+len(row['c'])
    np.savez_compressed(path,x=np.stack([r['x'] for r in rows]) if rows else np.zeros((0,len(feature_names())),np.float32),
        c=np.concatenate([r['c'] for r in rows]) if rows else np.zeros((0,CAND_DIM),np.float32),
        pi=np.concatenate([r['pi'] for r in rows]) if rows else np.zeros(0,np.float32),
        z=np.array([r['z'] for r in rows],np.float32),offsets=offsets)


def load_replay(path):
    with np.load(path,allow_pickle=False) as data:
        x,c,pi,z,offsets=(data[k].copy() for k in ('x','c','pi','z','offsets'))
    if len(x)>16384 or len(offsets)!=len(x)+1 or offsets[0]!=0 or offsets[-1]!=len(c) or len(pi)!=len(c) or len(z)!=len(x):
        raise ValueError('Invalid replay shape')
    if x.shape[1:]!=(len(feature_names()),) or c.shape[1:]!=(CAND_DIM,) or not all(np.isfinite(a).all() for a in (x,c,pi,z)):
        raise ValueError('Invalid replay values')
    rows=[]
    for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        if b<=a or z[i] not in (-1,0,1) or np.any(pi[a:b]<0) or not np.isclose(pi[a:b].sum(),1):
            raise ValueError('Invalid replay target')
        rows.append(dict(x=x[i],c=c[a:b],pi=pi[a:b],z=float(z[i])))
    return rows


def checkpoint(out,nets,opts,replay,state):
    import torch
    root=Path(out)/'recovery';root.mkdir(exist_ok=True)
    tmp=root/f'pending-{state["cycle"]}-{time.time_ns()}';tmp.mkdir()
    torch.save(dict(nets={k:n.state_dict() for k,n in nets.items()},opts={k:o.state_dict() for k,o in opts.items()},state=state),tmp/'training.pt')
    for k in PRESETS:save_replay(tmp/(k+'.npz'),replay[k])
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp.iterdir()}
    write(tmp/'manifest.json',manifest)
    dest=root/f'checkpoint-{state["cycle"]}-{time.time_ns()}';tmp.rename(dest)
    write(root/'pointer.json',dict(path=dest.name,manifest_sha256=hashlib.sha256((dest/'manifest.json').read_bytes()).hexdigest()))
    # Only this task's managed recovery generations are pruned; policy archives stay.
    old=sorted([p for p in root.glob('checkpoint-*') if p.is_dir()],key=lambda p:p.stat().st_mtime,reverse=True)
    for p in old[2:]:shutil.rmtree(p)


def restore(out,nets,opts):
    import torch
    root=Path(out)/'recovery';pointer=json.loads((root/'pointer.json').read_text())
    name=pointer['path']
    if Path(name).name!=name:raise ValueError('Invalid recovery pointer')
    path=root/name;raw=(path/'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pointer['manifest_sha256']:raise ValueError('Recovery manifest mismatch')
    for name,digest in json.loads(raw).items():
        if Path(name).name!=name or hashlib.sha256((path/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Recovery file mismatch')
    saved=torch.load(path/'training.pt',map_location='cpu',weights_only=True)
    if saved['state']['rule_hash']!=rule_hash():raise ValueError('Recovery rule mismatch')
    for k in PRESETS:nets[k].load_state_dict(saved['nets'][k]);opts[k].load_state_dict(saved['opts'][k])
    return {k:load_replay(path/(k+'.npz')) for k in PRESETS},saved['state']


def selection_decision(raw,retained,screen=None,confirm=None,rollback_confirm=None):
    if not raw['complete']:return 'keep'
    if raw['candidate']<=raw['best']-4 and rollback_confirm and rollback_confirm['complete'] and rollback_confirm['candidate']<=rollback_confirm['best']-4:
        return 'rollback'
    if not retained or raw['candidate']<raw['best']-2:return 'keep'
    if screen and confirm and screen['complete'] and confirm['complete']:
        if screen['candidate']>=screen['best']+2 and confirm['candidate']>confirm['best']:return 'promote'
        # When search already saturates this opponent set, allow independently
        # confirmed raw-policy improvement while both search checks do not regress.
        if raw['candidate']>=raw['best']+2 and rollback_confirm and rollback_confirm['complete'] and rollback_confirm['candidate']>rollback_confirm['best'] and screen['candidate']>=screen['best'] and confirm['candidate']>=confirm['best']:return 'promote'
    return 'keep'


def append(out,name,row):
    with (Path(out)/name).open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def build_probes(out,deadline):
    from .search_pilot import position
    from .league_search import prove_turn
    probes={}
    for ki,key in enumerate(PRESETS):
        probes[key]=[]
        for i in range(1500):
            if time.time()>=deadline:raise TimeoutError('Probe generation deadline')
            s,side=position(key,1003000000+ki*10000+i)
            proof=prove_turn(s,side,depth=1,node_budget=128,deadline=deadline)
            if proof['paths']:
                probes[key].append(dict(state=s,side=side,targets=list(proof['paths'])))
            if len(probes[key])==12:break
        if len(probes[key])!=12:raise ValueError('Insufficient selection probes')
    write(Path(out)/'selection-probes.json',probes)
    return probes


def probe_results(directory,key,probes):
    from .league_rollout import decision
    policy=LeagueModel(directory,key)
    return [int(policy.scores(*decision(p['state'],p['side'])[1]).argmax()) in p['targets'] for p in probes[key]]


def comparison(pool,out,key,candidate,best,reference,seed,deadline,*,searching,pairs):
    jobs=[];variants=[]
    for label,directory in (('candidate',candidate),('best',best)):
        for fi,foe in enumerate(k for k in PRESETS if k!=key):
            for i in range(pairs):
                for first in ('a','b'):
                    jobs.append(dict(seed=seed+fi*100+i,first=first,deadline=deadline,
                        decks={'a':deck(key),'b':deck(foe)},policies={'a':(str(directory),key),'b':(str(reference),foe)},
                        raw_sides=['b'] if searching else ['a','b'],simulations=256))
                    variants.append(label)
    results=list(pool.map(search_game,jobs))
    return dict(complete=all(g['complete'] for g in results),n=len(jobs)//2,
                candidate=sum(g.get('winner')=='a' for g,v in zip(results,variants) if v=='candidate'),
                best=sum(g.get('winner')=='a' for g,v in zip(results,variants) if v=='best'))


def run(out,config,resume=False):
    import torch
    torch.set_num_threads(1);torch.manual_seed(1000000000)
    out=Path(out);initial=out/'initial';reference=out/'reference'
    nets={};opts={};origins={}
    for key in PRESETS:
        nets[key],origins[key]=load_numeric(initial,key,config['device'])
        opts[key]=torch.optim.Adam(nets[key].parameters(),lr=3e-5)
    if resume:
        replay,state=restore(out,nets,opts)
        probes=json.loads((out/'selection-probes.json').read_text())
    else:
        replay={k:[] for k in PRESETS}
        state=dict(cycle=0,rule_hash=rule_hash(),best={k:str(initial) for k in PRESETS},current=str(initial),
                   next_selection=time.time()+config['selection_interval'],selection_round=0,
                   totals={k:dict(samples=0,optimizer_steps=0) for k in PRESETS},games=0,searched_decisions=0)
        probes=build_probes(out,min(config['train_until'],time.time()+180))
        checkpoint(out,nets,opts,replay,state)
    if state['rule_hash']!=config['rule_hash']:raise ValueError('Frozen rule identity changed')
    with ProcessPoolExecutor(max_workers=config['workers'],mp_context=get_context('spawn')) as pool:
        while time.time()<config['train_until'] and (not config.get('max_cycles') or state['cycle']<config['max_cycles']):
            cycle=state['cycle']+1;current=Path(state['current']);jobs=[]
            write(out/'status.json',dict(phase='selfplay',cycle=cycle,totals=state['totals'],games=state['games'],train_until=config['train_until'],deadline=config['hard_deadline']))
            for ki,key in enumerate(PRESETS):
                for fi,foe in enumerate(k for k in PRESETS if k!=key):
                    historical=cycle%4==0
                    enemy=Path(state['best'][foe]) if cycle%8==0 else initial if historical else current
                    for first in ('a','b'):
                        jobs.append(dict(seed=1000100000+cycle*10000+ki*1000+fi*100,first=first,
                            deadline=config['train_until'],decks={'a':deck(key),'b':deck(foe)},
                            policies={'a':(str(current),key),'b':(str(enemy),foe)},simulations=config['simulations'],
                            training=True,collect=True,train_sides=['a'] if historical else ['a','b']))
            games=list(pool.map(search_game,jobs));fresh={k:[] for k in PRESETS}
            for job,g in zip(jobs,games):
                append(out,'training-games.jsonl',dict(cycle=cycle,seed=job['seed'],first=job['first'],decks={s:d['id'] for s,d in job['decks'].items()},**compact(g)))
                if not g['complete']:continue
                if g['decisions']!=g['searched'] or len(g['rows'])!=g['decisions']:raise ValueError('Unsearched training decision')
                state['games']+=1;state['searched_decisions']+=g['searched']
                for row in g['rows']:
                    if row['side'] in job['train_sides']:fresh[row['key']].append(row)
            for key in PRESETS:
                replay[key]=(replay[key]+fresh[key])[-16384:]
                state['totals'][key]['samples']+=len(fresh[key])
            if not all(g['complete'] for g in games) or time.time()>=config['train_until']:
                checkpoint(out,nets,opts,replay,state);break
            rng=np.random.default_rng(1000200000+cycle)
            for key in PRESETS:
                steps=min(48,2*math.ceil(len(fresh[key])/128))
                for step in range(steps):
                    if time.time()>=config['train_until']:break
                    new=rng.choice(len(fresh[key]),min(64,len(fresh[key])),replace=False)
                    old=rng.choice(len(replay[key]),min(64,len(replay[key])),replace=False)
                    batch=[fresh[key][i] for i in new]+[replay[key][i] for i in old]
                    result=policy_value_update(nets[key],opts[key],batch)
                    state['totals'][key]['optimizer_steps']+=1
                    append(out,'losses.jsonl',dict(cycle=cycle,key=key,step=step,**result))
            dest=out/'generations'/f'cycle-{cycle:05}-{time.time_ns()}'
            for key in PRESETS:export_model(nets[key],dest,key,deck(key),origin=origins[key])
            state['current']=str(dest);state['cycle']=cycle
            if time.time()>=state['next_selection'] and time.time()<config['train_until']-900:
                state['selection_round']+=1;round_id=state['selection_round']
                selection_end=min(config['train_until'],time.time()+900)
                write(out/'status.json',dict(phase='selection',round=round_id,cycle=cycle,deadline=config['hard_deadline']))
                for ki,key in enumerate(PRESETS):
                    if time.time()>=selection_end:break
                    best=state['best'][key];seed=1010000000+round_id*10000+ki*1000
                    raw=comparison(pool,out,key,dest,best,reference,seed,selection_end,searching=False,pairs=4)
                    old=probe_results(best,key,probes);new=probe_results(dest,key,probes)
                    retained=all(not a or b for a,b in zip(old,new))
                    confirm_raw=screen=confirm=None
                    if raw['complete'] and raw['candidate']<=raw['best']-4:
                        confirm_raw=comparison(pool,out,key,dest,best,reference,seed+500000,selection_end,searching=False,pairs=4)
                    elif raw['complete'] and raw['candidate']>=raw['best']-2 and retained:
                        screen=comparison(pool,out,key,dest,best,reference,seed+1000000,selection_end,searching=True,pairs=2)
                        if screen['complete'] and (screen['candidate']>=screen['best']+2 or (raw['candidate']>=raw['best']+2 and screen['candidate']>=screen['best'])):
                            if raw['candidate']>=raw['best']+2:
                                confirm_raw=comparison(pool,out,key,dest,best,reference,seed+500000,selection_end,searching=False,pairs=4)
                            confirm=comparison(pool,out,key,dest,best,reference,seed+2000000,selection_end,searching=True,pairs=2)
                    choice=selection_decision(raw,retained,screen,confirm,confirm_raw)
                    if choice=='promote':state['best'][key]=str(dest)
                    elif choice=='rollback':
                        restored,_=load_numeric(best,key,config['device']);nets[key].load_state_dict(restored.state_dict())
                        lr=max(1e-5,opts[key].param_groups[0]['lr']/2)
                        opts[key]=torch.optim.Adam(nets[key].parameters(),lr=lr);replay[key]=[]
                    append(out,'selections.jsonl',dict(round=round_id,cycle=cycle,key=key,raw=raw,retained=retained,
                                                      search=screen,confirmation=confirm,raw_confirmation=confirm_raw,decision=choice))
                # Export rollbacks too; the next games must use these exact weights.
                selected=out/'generations'/f'selected-{cycle:05}-{time.time_ns()}'
                for key in PRESETS:export_model(nets[key],selected,key,deck(key),origin=origins[key])
                state['current']=str(selected);state['next_selection']=time.time()+config['selection_interval']
            checkpoint(out,nets,opts,replay,state)
        write(out/'training-summary.json',state)
        for key in PRESETS:
            export_model(nets[key],out/'latest',key,deck(key),origin=origins[key])
            (out/'best').mkdir(exist_ok=True)
            for ext in ('.npz','.json'):shutil.copyfile(Path(state['best'][key])/(key+ext),out/'best'/(key+ext))
        write(out/'status.json',dict(phase='final_evaluation',training_stopped=time.time(),deadline=config['hard_deadline']))
        evaluation_dir=out/'evaluations';evaluation_dir.mkdir(exist_ok=True)
        records=[];cache={}
        pairs=1 if config.get('smoke') else 4
        for variant in ('initial','best','latest'):
            for searching in (False,True):
                for ki,key in enumerate(PRESETS):
                    for fi,foe in enumerate(k for k in PRESETS if k!=key):
                        label=f'{variant}-{key}-vs-{foe}-'+('search' if searching else 'raw')
                        path=evaluation_dir/(label+'.json')
                        if path.exists():
                            prior=json.loads(path.read_text())
                            if prior['complete']:records.append(prior);continue
                        policy=LeagueModel(out/variant,key)
                        signature=(policy.version,key,foe,searching)
                        if signature in cache:
                            row={**cache[signature],'label':label,'variant':variant,'reused_identical_weights':True}
                        else:
                            jobs=[dict(seed=1030000000+ki*10000+fi*1000+i,first=first,deadline=config['evaluate_until'],
                                  decks={'a':deck(key),'b':deck(foe)},policies={'a':(str(out/variant),key),'b':(str(reference),foe)},
                                  raw_sides=['b'] if searching else ['a','b'],simulations=config['simulations'])
                                  for i in range(pairs) for first in ('a','b')]
                            result=list(pool.map(search_game,jobs))
                            row=dict(label=label,variant=variant,key=key,foe=foe,search=searching,
                                     complete=all(g['complete'] for g in result),wins=sum(g.get('winner')=='a' for g in result),
                                     n=sum(g['complete'] for g in result),scheduled=len(jobs),games=[compact(g) for g in result])
                            if row['complete']:cache[signature]=row
                        write(path,row);records.append(row)
                        write(out/'status.json',dict(phase='final_evaluation',completed_cells=len(records),deadline=config['hard_deadline']))
        write(out/'evaluation-index.json',records)
        from .league_gate import cases as heldout_cases
        from .search_budget_eval import consistent_tactical_fixture,expected_actions,case_job
        fixtures=[]
        for key,name,s,side,goal in heldout_cases():
            s,changes=consistent_tactical_fixture(s)
            foe_side='b' if side=='a' else 'a'
            foe=next((k for k in PRESETS if set(deck(k)['character_ids'])==set(s['sides'][foe_side]['characters'])),'starter')
            fixtures.append(dict(key=key,name=name,state=s,side=side,foe=foe,expected=expected_actions(s,side,goal)))
        if config.get('user_snapshot'):
            s=json.loads(Path(config['user_snapshot']).read_text(encoding='utf-8'))
            fixtures.append(dict(key='starter',name='user-turn3-keep-zero-clear-front',state=s,side='a',foe='weave-rush',expected=expected_actions(s,'a','family')))
        jobs=[];info=[]
        for variant in ('initial','best','latest'):
            for fixture in fixtures:
                side=fixture['side'];foe_side='b' if side=='a' else 'a'
                for budget in (0,config['simulations']):
                    jobs.append(dict(state=fixture['state'],side=side,budget=budget,seed=1040000000,
                                deadline=config['evaluate_until'],expected=fixture['expected'],
                                policies={side:(str(out/variant),fixture['key']),foe_side:(str(reference),fixture['foe'])}))
                    info.append(dict(variant=variant,case=fixture['name'],budget=budget))
        tactical=[dict(meta,**row) for meta,row in zip(info,pool.map(case_job,jobs))]
        write(out/'tactical-evaluation.json',tactical)
    summary=dict(training=state,complete_evaluation=all(r['complete'] for r in records) and all(r['complete'] for r in tactical),tactical_evaluation=tactical,finished=time.time(),
                 hard_deadline=config['hard_deadline'],automatic_serving_approval=False,
                 evaluations=[{k:v for k,v in row.items() if k!='games'} for row in records])
    write(out/'summary.json',summary)
    lines=['# 中午截止训练报告','',f"完整训练局：{state['games']}；逐决策搜索：{state['searched_decisions']}；完成轮次：{state['cycle']}。",'',
           '模型未自动发布。结果仅对应固定对手与本次种子；战术回归与原录像资源判定另见 tactical-evaluation.json；原录像允许N02或普通出击保零清场，不指定唯一动作。','',
           '| 模型 | 策略 | 对手 | 模式 | 胜局 | 完成局 |','| --- | --- | --- | --- | ---: | ---: |']
    for row in records:lines.append(f"| {row['variant']} | {row['key']} | {row['foe']} | {'search' if row['search'] else 'raw'} | {row['wins']} | {row['n']} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'status.json',dict(phase='complete' if summary['complete_evaluation'] else 'incomplete_evaluation',finished=time.time(),deadline=config['hard_deadline']))
    manifest={}
    with zipfile.ZipFile(out/'delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('initial','best','latest','evaluations'):
            for p in sorted((out/folder).rglob('*')):
                if p.is_file() and p.suffix in ('.json','.npz'):
                    name=p.relative_to(out).as_posix();data=p.read_bytes();manifest[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
        for name in ('config.json','summary.json','training-summary.json','evaluation-index.json','tactical-evaluation.json','status.json','report.md','selections.jsonl','losses.jsonl'):
            p=out/name
            if p.exists():data=p.read_bytes();manifest[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    write(out/'delivery.json',dict(sha256=hashlib.sha256((out/'delivery.zip').read_bytes()).hexdigest(),files=len(manifest)))
