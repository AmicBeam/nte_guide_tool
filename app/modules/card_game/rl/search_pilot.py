"""Small synthetic tactical curriculum; no serving or default league changes."""
from copy import deepcopy
from pathlib import Path
from random import Random
import hashlib
import json
import time
import numpy as np

from .league_schema import PRESETS, deck, rule_hash
from .league_rollout import decision, play
from .league_search import prove_turn, distill_step, compare_root
from .league_policy import network_from_old, export_model, LeagueModel


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def position(key, seed):
    """Random legal deck/hand with synthetic public tactical state perturbations.

    These are curriculum positions, not claims of naturally observed game rates.
    No named held-out fixtures or user replay are imported.
    """
    from app.modules.card_game.engine.duel_v2 import new_game
    rng=Random(seed);side='ab'[seed%2];foe='b' if side=='a' else 'a'
    other=PRESETS[rng.randrange(3)]
    s=new_game(seed=seed,first_side=side,skip_mulligan=True,
               decks={side:deck(key),foe:deck(other)})
    own=s['sides'][side];enemy=s['sides'][foe]
    own['ap']=rng.choice((1,2));enemy['ap']=0
    generated=[c for c in own['hand'] if c.get('derived')]
    pool=[c for c in own['hand']+own['deck'] if not c.get('derived')]
    rng.shuffle(pool);n=rng.randint(4,8)-len(generated)
    own['hand']=generated+pool[:n];own['deck']=pool[n:]
    enemy['hp']=rng.randint(3,11);enemy['shield']=rng.randint(0,2)
    if rng.random()<.4:
        front=rng.choice(list(enemy['characters']));enemy['front']=front
        enemy['characters'][front]['hp']=rng.randint(1,3)
    if rng.random()<.5:
        prev=rng.choice(list(own['characters']));own['last_front']=prev
        own['characters'][prev]['harmony']=2
    if 'xiaozhi' in own['characters']:
        own['characters']['xiaozhi']['jingu']=rng.choice((0,3,6))
    return s,side


def corpus(key, seed, count, seen, deadline):
    samples=[];attempted=0;nodes=0;uncertain=None
    while len(samples)<count and attempted<count*15 and time.time()<deadline:
        s,side=position(key,seed+attempted);attempted+=1
        actions,(x,c)=decision(s,side)
        fingerprint=hashlib.sha256(x.tobytes()+c.tobytes()).hexdigest()
        if fingerprint in seen:
            continue
        proof=prove_turn(s,side,node_budget=512,deadline=deadline);nodes+=proof['nodes']
        if not proof['paths']:
            if uncertain is None:uncertain=(s,side)
            continue
        if proof['roots_visited']!=len(actions):
            continue
        shortest=min(proof['depths'].values())
        targets=[i for i,d in proof['depths'].items() if d==shortest]
        seen.add(fingerprint)
        samples.append(dict(x=x,c=c,targets=targets,fingerprint=fingerprint,
                            seed=seed+attempted-1,depth=shortest,actions=actions,
                            paths=proof['paths']))
    return samples,dict(attempted=attempted,accepted=len(samples),nodes=nodes,
                        two_action_proofs=sum(s['depth']==2 for s in samples)),uncertain


def choices(net, samples):
    import torch
    from .league_learning import tensors
    if not samples:return []
    with torch.no_grad():
        x,c,m=tensors([(s['x'],s['c']) for s in samples],next(net.parameters()).device)
        return net(x,c,m)[0].argmax(-1).cpu().tolist()


def correct(net, samples):
    return [a in s['targets'] for a,s in zip(choices(net,samples),samples)]


def quality_report(summary, comparisons):
    reasons=[]
    for key in PRESETS:
        data=summary[key]
        if data['scores']['test']['forgotten']:
            reasons.append(key+': held-out tactical forgetting')
        if not data['heldout_gates']['final']['passed']:
            reasons.append(key+': independent tactical gate failed')
        totals={v:[0,0] for v in ('baseline','final')}
        for row in comparisons:
            if row['key']==key:
                totals[row['variant']][0]+=row['wins'];totals[row['variant']][1]+=row['n']
        if totals['baseline'][1]!=16 or totals['final'][1]!=16:
            reasons.append(key+': incomplete game evaluation')
        elif totals['final'][0]<totals['baseline'][0]:
            reasons.append(key+': fewer wins in small paired game sample')
    return dict(passed=not reasons,reasons=reasons,automatic_serving_approval=False,
                scope='pilot gate only; passing is not proof of overall strength')


def run(out, sources, deadline):
    import torch
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    torch.set_num_threads(1);torch.manual_seed(973100000)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    started=time.time();nets={};origins={};summary={};uncertain={}
    write(out/'status.json',dict(phase='started',deadline=deadline))
    for key in PRESETS:
        nets[key],origins[key]=network_from_old(sources['weave-rush' if key=='quick-rush' else key],key,'cpu')
        export_model(nets[key],out/'baseline',key,deck(key),origin=origins[key])
    for ki,key in enumerate(PRESETS):
        if time.time()>=deadline:raise TimeoutError('Pilot budget exhausted')
        write(out/'status.json',dict(phase='corpus:'+key,deadline=deadline))
        seen=set();sets={};coverage={}
        for si,name in enumerate(('train','selection','confirmation','test')):
            n=48 if name=='train' else 24
            sets[name],coverage[name],root=corpus(key,973100000+ki*100000+si*10000,n,seen,min(deadline-90,time.time()+90))
            if root is not None:uncertain[key]=root
            if len(sets[name])<n:raise RuntimeError('Insufficient independent corpus: '+key+'/'+name)
        net=nets[key];initial=deepcopy(net.state_dict());opt=torch.optim.Adam(net.parameters(),lr=1e-4)
        before={name:correct(net,v) for name,v in sets.items()}
        selected=initial;best=sum(before['selection'])+sum(before['confirmation']);selected_step=0
        history=[];rng=Random(973200000+ki);stable={s['fingerprint']:0 for s in sets['train']}
        for step in range(1,161):
            if time.time()>=deadline-90:break
            status=correct(net,sets['train'])
            for s,ok in zip(sets['train'],status):
                stable[s['fingerprint']]=stable[s['fingerprint']]+1 if ok else 0
            hard=[s for s in sets['train'] if stable[s['fingerprint']]<3]
            # Revisit reliable errors until learned, retaining mastered examples.
            batch=rng.sample(hard,min(12,len(hard)))+rng.sample(sets['train'],min(12,len(sets['train'])))
            loss=distill_step(net,opt,batch)
            if step%10:continue
            a=correct(net,sets['selection']);b=correct(net,sets['confirmation'])
            retained=all(now or not old for now,old in zip(a+b,before['selection']+before['confirmation']))
            improved=sum(a)>=sum(before['selection']) and sum(b)>=sum(before['confirmation']) and sum(a+b)>best
            if retained and improved:
                best=sum(a+b);selected=deepcopy(net.state_dict());selected_step=step
            history.append(dict(step=step,loss=loss,selection=sum(a),confirmation=sum(b),retained=retained))
        net.load_state_dict(selected)
        after={name:correct(net,v) for name,v in sets.items()}
        summary[key]=dict(coverage=coverage,selected_step=selected_step,history=history,
                          scores={name:dict(n=len(v),before=sum(before[name]),after=sum(after[name]),
                                            forgotten=sum(old and not new for old,new in zip(before[name],after[name])))
                                  for name,v in sets.items()})
        write(out/(key+'-curriculum.json'),{name:[{k:s[k] for k in ('seed','fingerprint','targets','depth','actions','paths')} for s in v] for name,v in sets.items()})
        export_model(net,out/'final',key,deck(key),origin=origins[key])
        torch.save(dict(schema='search_pilot_v1',rule_hash=rule_hash(),model=net.state_dict(),selected_step=selected_step),out/(key+'.pt'))
        write(out/'summary.json',summary)
    # Held-out old tactical fixtures are evaluated only AFTER checkpoint selection.
    from .league_gate import evaluate
    for key in PRESETS:
        summary[key]['heldout_gates']={v:evaluate(LeagueModel(out/v,key)) for v in ('baseline','final')}
    comparisons=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
        for ki,key in enumerate(PRESETS):
            for fi,foe in enumerate(k for k in PRESETS if k!=key):
                for variant in ('baseline','final'):
                    jobs=[dict(deadline=deadline-10,seed=978000000+ki*10000+fi*100+i,
                               first=first,viewer='a',decks={'a':deck(key),'b':deck(foe)},
                               policies={'a':(str(out/variant),key),'b':(str(out/'baseline'),foe)})
                          for i in range(4) for first in ('a','b')]
                    games=list(pool.map(play,jobs))
                    comparisons.append(dict(key=key,foe=foe,variant=variant,wins=sum(g.get('winner')=='a' for g in games),
                                            n=sum(g['complete'] for g in games),games=games))
        # Exercise discovery+fresh confirmation on one unproved generated root.
        if uncertain and time.time()<deadline-30:
            key='quick-rush' if 'quick-rush' in uncertain else next(iter(uncertain))
            s,side=uncertain[key];foe='b' if side=='a' else 'a';actions,(x,c)=decision(s,side)
            policies={side:(str(out/'baseline'),key),foe:(str(out/'baseline'),'starter')}
            checked=compare_root(s,side,policies,LeagueModel(out/'baseline',key).scores(x,c),
                                 seed=979000000,deadline=min(deadline-10,time.time()+45),map_jobs=pool.map)
            write(out/'paired-discovery.json',{k:v for k,v in checked.items() if k not in ('x','c')})
    write(out/'quality.json',quality_report(summary,comparisons))
    write(out/'comparisons.json',comparisons)
    write(out/'summary.json',summary)
    write(out/'status.json',dict(phase='complete',started=started,finished=time.time(),deadline=deadline,
                                serving_approved=False,experiment='synthetic tactical search-distillation pilot'))
