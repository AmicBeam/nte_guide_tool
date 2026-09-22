"""One-hour pure-outcome replay experiment on the shared search engine."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import copy
import hashlib
import json
import math
import time
import zipfile
import numpy as np
from . import outcome_runtime as rt
from .fixed_lineup import FixedModel, identity, write
from .information_search import search_game
from .information_longrun import append
from .episode_replay import EpisodePool, dump_episode, reanalyse
from .full_cycle_experiment import evaluate_job

ARMS=('uniform','reanalyse')
FOES=('starter','weave-rush','quick-rush')
KEY='zhenhong'


def game_summary(g):
    return {k:v for k,v in g.items() if k not in ('rows','value_rows','episode_actions','search_stats')}


def calibration(net, rows):
    import torch
    if not rows:
        raise ValueError('No holdout value data')
    with torch.no_grad():
        x=torch.as_tensor(np.stack([r['x'] for r in rows]),device=next(net.parameters()).device)
        actor=torch.tensor([[r['value_actor']] for r in rows],dtype=x.dtype,device=x.device)
        p=net.wdl(torch.cat((net.state_net(x),actor),-1)).softmax(-1).cpu().numpy()
    z=np.array([int(r['z'])+1 for r in rows]);target=np.eye(3)[z]
    return dict(rows=len(rows),log_loss=float(-np.log(p[np.arange(len(z)),z].clip(1e-12)).mean()),
                brier=float(((p-target)**2).sum(-1).mean()),mean_probabilities=p.mean(0).tolist())


def run(out,config):
    import torch
    ARMS=tuple(config.get('arms',('uniform','reanalyse')))
    selfplay=config.get('profile') in ('selfplay_budget','gumbel_low_budget','grounded_actions','continue_outcome','self_imitation')
    torch.set_num_threads(1);torch.manual_seed(config['seeds']['foundation'])
    if config['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    out=Path(out);source=out/'sources'
    if identity()!=config['rule_hash']:raise ValueError('Rule identity changed')
    base={k:rt.model(source,k) for k in (KEY,*FOES)}
    source_hash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    steps={a:0 for a in ARMS};totals={a:dict(games=0,samples=0,wins=0,reanalysed=0) for a in ARMS}
    def status(phase,**kw):write(out/'status.json',dict(phase=phase,steps=steps,training=totals,hard_deadline=config['hard_deadline'],**kw))
    rng=np.random.default_rng(config['seeds']['foundation']);pools={a:EpisodePool() for a in ARMS}
    with ProcessPoolExecutor(max_workers=config['workers'],mp_context=get_context('spawn')) as executor:
        status('value_warmup')
        warm=[];hold=[];jobs=[]
        n=4 if config['smoke'] else 36
        for i in range(n):
            held=i>=int(n*2/3);foe=KEY if i%2==0 else FOES[(i//2)%3]
            jobs.append(dict(id=f'warm-{i}',held=held,seed=config['seeds']['foundation']+50000+i,first=('a','b')[(i//2)%2],
                deadline=min(config['train_until']-90,time.time()+(180 if config['smoke'] else 360)),runtime='outcome',
                training=False,collect=True,collect_value_only=True,record_episode=True,raw_sides=['a','b'],train_sides=['a'],
                decks={'a':base[KEY].serving_deck,'b':base[foe].serving_deck},policies={'a':(str(source),KEY),'b':(str(source),foe)}))
        for j,g in zip(jobs,executor.map(search_game,jobs)):
            append(out,'warmup-games.jsonl',dict(job=j,**game_summary(g)))
            if not g['complete']:raise RuntimeError('Incomplete value warmup game')
            data=dump_episode(out/'private-episodes'/f"{j['id']}.json.gz",j,g)
            (hold if j['held'] else warm).extend(data['rows'])
        net=rt.create_network(source,KEY,config['device']);before=calibration(net,hold)
        opt=torch.optim.Adam(net.wdl.parameters(),lr=1e-3)
        warmup_steps=0 if isinstance(base[KEY],rt.OutcomeModel) else 4 if config['smoke'] else 64
        for i in range(warmup_steps):
            if time.time()>=config['train_until']-90:raise RuntimeError('Warmup exhausted training budget')
            rows=[warm[j] for j in rng.choice(len(warm),min(128,len(warm)),replace=False)]
            result=rt.update(net,opt,rows,warmup=True)
            append(out,'warmup-losses.jsonl',{k:v for k,v in result.items() if k!='priority'})
        after=calibration(net,hold)
        if not math.isfinite(after['log_loss']) or (warmup_steps and after['log_loss']>math.log(3)+.15):
            raise RuntimeError('WDL warmup holdout gate failed')
        origin=dict(source_sha256=base[KEY].version,warmup_games=n,warmup_steps=warmup_steps,auxiliary_rewards=0,
                    optimizer_reset=True,source_outcome_model=isinstance(base[KEY],rt.OutcomeModel))
        initial=out/'common-initial';rt.export(net,initial,KEY,base[KEY].serving_deck,origin)
        numeric=rt.OutcomeModel(initial,KEY)
        for row in hold[:16]:
            with torch.no_grad():
                x=torch.as_tensor(row['x'],device=config['device']);h=net.state_net(x)
                p=net.wdl(torch.cat((h,h.new_tensor([row['value_actor']])))).softmax(-1).cpu().numpy()
            if not np.allclose(p,numeric.wdl(row['x'],row['value_actor']),atol=2e-5):raise ValueError('WDL numeric export mismatch')
        if not all(np.array_equal(v,net.state_dict()[k].detach().cpu().numpy()) for k,v in base[KEY].weights.items()):raise ValueError('Warmup changed source policy')
        write(out/'warmup.json',dict(before=before,after=after,source_policy_unchanged=True,
                                   warmup_steps=warmup_steps,holdout_used_for_gradient=False,calibrated_probability_claim=False))
        nets={a:copy.deepcopy(net) for a in ARMS}
        if config.get('profile')=='grounded_actions':
            config['arm_initial_models']={}
            for arm in ARMS:
                if arm!='inherited':
                    torch.manual_seed(config['seeds']['foundation']+1)
                    nets[arm]=rt.create_grounded_network(initial,KEY,config['device'],categorical=arm=='grounded')
                folder=out/arm/'initial'
                rt.export(nets[arm],folder,KEY,base[KEY].serving_deck,{**origin,'action_initialization':arm})
                config['arm_initial_models'][arm]=str(folder)
            write(out/'arm-initial-models.json',config['arm_initial_models'])
        opts={a:torch.optim.Adam([p for p in nets[a].parameters() if p.requires_grad],lr=3e-5) for a in ARMS}
        current={a:initial for a in ARMS};groups=0;reanalysis_seconds=0.;generation_seconds=0.
        if selfplay:
            from .selfplay_trial import learn
            selfplay_started=time.time()
            current,groups=learn(executor,out,config,base,source,initial,nets,opts,pools,steps,totals,status,origin)
            generation_seconds=time.time()-selfplay_started
        while not selfplay and time.time()<config['train_until']-90 and not (config['smoke'] and groups):
            started=time.time();jobs=[]
            for group in range(groups,groups+(1 if config['smoke'] else 4)):
                foe=FOES[(group//2)%3];first=('a','b')[group%2]
                for a in ARMS:
                    jobs.append(dict(id=f'{a}-{group}',arm=a,group=group,seed=config['seeds']['foundation']+group,first=first,
                        deadline=min(config['train_until']-75,time.time()+1200),runtime='outcome',simulations=config['simulations'],fast_simulation=config.get('fast_simulation',True),
                        training=True,collect=True,record_episode=True,train_sides=['a'],failure_dir=str(out/'failure-roots'),
                        decks={'a':base[KEY].serving_deck,'b':base[foe].serving_deck},policies={'a':(str(current[a]),KEY),'b':(str(source),foe)}))
            groups+=1 if config['smoke'] else 4;pending={};completed=[]
            status('searching',scheduled_groups=groups)
            futures={executor.submit(search_game,j):j for j in jobs}
            for f in as_completed(futures):
                j=futures[f];g=f.result();a=j['arm'];append(out,'training-games.jsonl',dict(job=j,**game_summary(g)))
                if g['complete']:
                    if g['searched']!=g['decisions'] or len(g['rows'])!=g['decisions']:raise ValueError('Training decision not searched')
                    path=out/'private-episodes'/f"{j['id']}.json.gz";data=dump_episode(path,j,g)
                    totals[a]['games']+=1;totals[a]['wins']+=g['winner']=='a';totals[a]['samples']+=len(data['rows'])
                    pending.setdefault(j['group'],{})[a]=(j,data,path)
                status('searching',scheduled_groups=groups)
            elapsed=time.time()-started;generation_seconds+=elapsed
            matched=[d for d in pending.values() if set(d)==set(ARMS)]
            for pair in matched:
                for a,(j,data,path) in pair.items():pools[a].add(j['id'],data)
            for pair in matched:
                for _ in range(1 if config['smoke'] else 2):
                    if time.time()>=config['train_until']-5:break
                    batches={a:pools[a].sample(rng,64,a=='reanalyse') for a in ARMS}
                    if not all(batches[a][0] for a in ARMS):break
                    for a in ARMS:
                        rows,refs,diagnostics=batches[a];result=rt.update(nets[a],opts[a],rows)
                        pools[a].priorities(refs,result.pop('priority'));steps[a]+=1
                        append(out,'losses.jsonl',dict(arm=a,step=steps[a],**diagnostics,**result))
            for a in ARMS:
                if steps[a]:
                    current[a]=out/a/'generations'/f"update-{steps[a]:04}"
                    rt.export(nets[a],current[a],KEY,base[KEY].serving_deck,{**origin,'steps':steps[a],'arm':a})
            # At most 20% of cumulative search+reanalysis wall time; all work inside train cutoff.
            allowed=min(120.,generation_seconds*.25-reanalysis_seconds,config['train_until']-time.time()-60)
            if matched and allowed>2:
                j,data,path=matched[-1]['reanalyse'];available=[r['step'] for r in data['rows'] if 'pi' in r]
                selected=sorted(rng.choice(available,min(2 if config['smoke'] else 8,len(available)),replace=False).tolist())
                task=dict(episode=str(path),episode_id=j['id'],steps=selected,policies={**j['policies'],'a':(str(current['reanalyse']),KEY)},
                          seed=config['seeds']['foundation']+100000+groups*1000,simulations=config['simulations'],deadline=time.time()+allowed)
                begin=time.time();result=executor.submit(reanalyse,task).result();reanalysis_seconds+=time.time()-begin
                pools['reanalyse'].replace_labels(j['id'],result['rows']);totals['reanalyse']['reanalysed']+=len(result['rows'])
                write(out/'reanalysis'/f"{j['id']}.json",dict(episode_id=j['id'],requested=result['requested'],completed=len(result['rows']),
                      steps=[r['step'] for r in result['rows']],model=rt.model(str(current['reanalyse']),KEY).version))
            if matched and totals['reanalyse']['reanalysed'] and time.time()<config['train_until']-5:
                for a in ARMS:
                    rows,refs,diagnostics=pools[a].sample(rng,64,a=='reanalyse')
                    if not rows:raise ValueError('Replay exhausted during paired update')
                    result=rt.update(nets[a],opts[a],rows);pools[a].priorities(refs,result.pop('priority'));steps[a]+=1
                    append(out,'losses.jsonl',dict(arm=a,step=steps[a],after_reanalysis=True,**diagnostics,**result))
                for a in ARMS:
                    current[a]=out/a/'generations'/f"update-{steps[a]:04}"
                    rt.export(nets[a],current[a],KEY,base[KEY].serving_deck,{**origin,'steps':steps[a],'arm':a})
            torch.save(dict(steps=steps,nets={a:n.state_dict() for a,n in nets.items()},opts={a:o.state_dict() for a,o in opts.items()},
                replay={a:p.episodes for a,p in pools.items()},rng=rng.bit_generator.state,groups=groups),out/'checkpoint.tmp')
            (out/'checkpoint.tmp').replace(out/'checkpoint.pt');status('updating',scheduled_groups=groups)
        if not selfplay and steps['uniform']!=steps['reanalyse']:raise ValueError('Unequal updates')
        for a in ARMS:rt.export(nets[a],out/a/'final',KEY,base[KEY].serving_deck,{**origin,'steps':steps[a],'arm':a})
        write(out/'training-summary.json',dict(steps=steps,training=totals,generation_seconds=generation_seconds,reanalysis_seconds=reanalysis_seconds))
        folder=out/'evaluation';(folder/'raw').mkdir(parents=True);(folder/'replays').mkdir();jobs=[]
        variants={'initial':initial,**{a:out/a/'final' for a in ARMS}}
        for v,directory in variants.items():
            for fi,foe in enumerate(FOES):
                for i in range(1 if config['smoke'] else 32):
                    for first in ('a','b'):
                        jobs.append(dict(id=f'{v}-{foe}-{i}-{first}',cell=f'{v}/{foe}',variant=v,kind='reference',foe=foe,
                            seed=config['seeds']['audit']+fi*10000+i,first=first,deadline=config['evaluate_until'],runtime='outcome',
                            decks={'a':base[KEY].serving_deck,'b':base[foe].serving_deck},policies={'a':(str(directory),KEY),'b':(str(source),foe)},output=str(folder),replay=i==0))
        for ai,a in enumerate(ARMS):
            for bi,b in enumerate(ARMS):
                for i in range(2 if config['smoke'] else 128):
                    first=('a','b')[i%2];second='b' if first=='a' else 'a'
                    jobs.append(dict(id=f'matrix-{a}-{b}-{i}',cell=f'{a}/{b}',kind='matrix',row=a,col=b,
                        seed=config['seeds']['audit']+50000+ai*1000+bi*200+i//2,first=first,deadline=config['evaluate_until'],runtime='outcome',
                        decks={'a':base[KEY].serving_deck,'b':base[KEY].serving_deck},policies={first:(str(variants[a]),KEY),second:(str(variants[b]),KEY)},output=str(folder),replay=i==0))
        write(folder/'jobs.json',jobs);results=[];status('evaluation',scheduled=len(jobs))
        for result in executor.map(evaluate_job,jobs):
            results.append(result);append(folder,'index.jsonl',result);status('evaluation',completed=len(results),scheduled=len(jobs))
    metrics={}
    for result in results:
        key=f"{result['kind']}/{result['cell']}";m=metrics.setdefault(key,dict(planned=0,complete=0,wins=0,passive=0,setup=0))
        m['planned']+=1
        if result['complete']:
            owner=result['first'] if result['kind']=='matrix' else 'a'
            m['complete']+=1;m['wins']+=result['winner']==owner;m['passive']+=result['passive_triggers'][owner];m['setup']+=bool(result['setup_achieved'][owner])
    unchanged=all(hashlib.sha256((source/n).read_bytes()).hexdigest()==sha for n,sha in source_hash.items())
    if not unchanged:raise ValueError('Frozen source changed')
    changed={}
    for a in ARMS:
        final=rt.OutcomeModel(out/a/'final',KEY)
        starting=rt.OutcomeModel(config.get('arm_initial_models',{}).get(a,initial),KEY)
        changed[a]=dict(policy=any(not np.array_equal(starting.weights[k],v) for k,v in final.weights.items() if not k.startswith(('value.','wdl.'))),
                        wdl=any(not np.array_equal(starting.weights[k],v) for k,v in final.weights.items() if k.startswith('wdl.')))
    import csv
    with (out/'first-second-winrates.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['row','col','planned','complete','wins','row_model_sha','col_model_sha']);writer.writeheader()
        for a in ARMS:
            for b in ARMS:
                m=metrics[f'matrix/{a}/{b}'];writer.writerow(dict(row=a,col=b,**{k:m[k] for k in ('planned','complete','wins')},
                    row_model_sha=rt.OutcomeModel(out/a/'final',KEY).version,col_model_sha=rt.OutcomeModel(out/b/'final',KEY).version))
    summary=dict(complete=len(results)==len(jobs) and all(r['complete'] for r in results),training_updated=min(steps.values())>0 and all(v['policy'] and v['wdl'] for v in changed.values()),
        weights_changed=changed,steps=steps,reanalysed=totals.get('reanalyse',{}).get('reanalysed',0),metrics=metrics,source_unchanged=True,automatic_serving_approval=False,
        single_training_seed=True,scope='fixed_zhenhong_'+config['profile'] if selfplay else 'fixed_zhenhong_WDL_uniform_vs_replay_reanalysis')
    write(out/'summary.json',summary)
    description='三组共享WDL起点、关闭人工奖励；fixed256固定陪练，self256自博弈混合固定256，selfmixed相同自博弈分布混合32/256。各组相同并发槽与截止；比较同墙钟效果，不要求优化次数相同。' if selfplay else '两组共享WDL预热起点、关闭前置奖及被动奖。仅比较均匀按局回放与优先回放＋重分析；单次随机种子试验，不代表完整算法验收。'
    if config.get('profile')=='gumbel_low_budget':description='同起点、同墙钟、纯胜负的Gumbel16与Gumbel32自博弈对照；策略标签由先验和完成Q生成，训练动作采用Gumbel搜索的最终选择。'
    if config.get('profile')=='grounded_actions':description='同状态与WDL起点、同16次Gumbel预算：继承旧动作头、重置序数动作头、类别及公开角色特征动作头三组对照；纯胜负自博弈，不复用评估数据学习。'
    if config.get('profile')=='continue_outcome':description='从已核验WDL数值权重开始新阶段，保留动作头、状态与价值；优化器和经验池显式重置。使用新的独立训练及评估种子，不复用评估数据。'
    if config.get('profile')=='self_imitation':description='同一已核验WDL起点、相同16次Gumbel与对手分布，对比原目标和附加正优势自模仿；只使用实际完整终局与实际动作，不奖励准备或被动计数。'
    lines=['# 真红纯胜负经验学习试验','',description,'',
           '| 对照 | 胜/正常完成 | 计划局数 | 实际被动次数 |','| --- | --- | --- | --- |']
    for key,m in metrics.items():lines.append(f"| {key} | {m['wins']}/{m['complete']} | {m['planned']} | {m['passive']} |")
    lines+=['','## 新模型变体互搏：行先手、列后手','','| 先手＼后手 | '+' | '.join(ARMS)+' |','| --- | '+' | '.join('---' for _ in ARMS)+' |']
    for a in ARMS:
        cells=[]
        for b in ARMS:
            m=metrics[f'matrix/{a}/{b}'];cells.append(f"{100*m['wins']/m['complete']:.1f}%（{m['wins']}/{m['complete']}）" if m['complete']==m['planned'] else f"未完成（{m['wins']}/{m['complete']}）")
        lines.append('| '+a+' | '+' | '.join(cells)+' |')
    lines+=['','这是同一真红构筑的训练变体，不是四套或五套新模型矩阵。',f'真实更新：{steps}；重分析决策数：{totals.get("reanalyse",{}).get("reanalysed",0)}。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    members=[p for p in out.rglob('*') if p.is_file() and not any(x in p.relative_to(out).parts for x in ('frozen','generations')) and p.name not in ('delivery.zip','delivery.json','console.log')]
    with zipfile.ZipFile(out/'delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        manifest={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in members}
        for p in members:z.write(p,p.relative_to(out))
        z.writestr('manifest.json',json.dumps(manifest))
    write(out/'delivery.json',dict(sha256=hashlib.sha256((out/'delivery.zip').read_bytes()).hexdigest(),members=len(members)))
    status('finished',complete=summary['complete'])
