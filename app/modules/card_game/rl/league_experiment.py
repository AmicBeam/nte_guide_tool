"""Three-strategy experiment orchestration; wall budget is enforced by outer guard."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path
from random import Random
import hashlib,json,time,zipfile
import numpy as np
from .league_schema import *
from .league_policy import export_model,network_from_old,LeagueModel
from .league_rollout import play,decision,determinize
from .league_learning import ordinary_update,branch_update,tensors
from .league_replay import ErrorReplay,OptimizerBudget
from .league_preferences import certified_lethals,auxiliary_pairs

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');p.replace(path)

def metric(out,row):
    with (out/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')

def sample_deck(seed,heroes):
    from app.modules.card_game.content.duel_v2 import CARDS,validate_deck
    rng=Random(seed);cards=[]
    for cid in heroes:
        pool=[c for c in CARD_IDS if CARDS[c]['character_id']==cid and not CARDS[c].get('derived') for _ in range(2)]
        cards+=rng.sample(pool,8)
    return validate_deck({'id':f'league-public-{seed}', 'name':'八人公开留出','character_ids':list(heroes),'card_ids':cards})

def experiment(out,sources,deadline,*,device='cuda',workers=6,phase_seconds=720,evaluation_reserve=900,total_seconds=5400,max_cycles=0,smoke=False):
    import torch
    torch.set_num_threads(1);torch.manual_seed(815000000)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    nets={};optimizers={};origins={};builds={k:deck(k) for k in PRESETS};current={}
    for key in PRESETS:
        source=sources['weave-rush' if key=='quick-rush' else key]
        nets[key],origins[key]=network_from_old(source,key,device)
        optimizers[key]=torch.optim.Adam(nets[key].parameters(),lr=1e-4)
        export_model(nets[key],out/'baseline',key,builds[key],origin=origins[key])
        current[key]=out/'baseline'
    write(out/'config.json',{'schema':SCHEMA,'rule_hash':rule_hash(),'deadline_epoch':deadline,'total_seconds':total_seconds,'phase_seconds':phase_seconds,'evaluation_reserve_seconds':evaluation_reserve,'device':device,'rules_backend':'official_python','source_manifest_sha256':hashlib.sha256((Path(__file__).resolve().parents[4]/'source-manifest.json').read_bytes()).hexdigest(),'workers':workers,'presets':PRESETS,'builds':builds,'origins':origins,'reward':'terminal +/-10, draw 0; unfinished excluded','learning_version':'terminal_tiebreak_v3','minimum_branch_steps_per_ordinary_step':3,'max_ordinary_steps_per_cycle':1,'max_branch_steps_per_cycle':3,'error_replay_capacity':256,'error_replay_ttl_cycles':5,'error_replay_max_uses':3,'minimum_return_gap':2.5,'ordinary_games':4,'candidates':6,'particles':8,'online_search':False,'automatic_serving_approval':False,'smoke':smoke,'selection_seed_segment':830000000,'branch_policy_loss':'pairwise_terminal_ranking_with_weak_tiebreaks','certain_lethal_weight':.2,'winning_speed_weight':.05,'losing_delay_weight':.02,'branch_value_target_temperature':4,'disable_anchor_return_gap':2.5,'branch_anchor_kl':.05,'phase_selection':'keep starting/best model unless at least 2/32 extra validation wins'})
    total={k:{'ordinary_updates':0,'branch_updates':0,'ordinary_games':0,'branch_games':0,'informative_roots':0,'attempted_roots':0,'ordinary_optimizer_steps':0,'branch_optimizer_steps':0} for k in PRESETS}
    from .league_gate import evaluate as evaluate_gate
    for key in PRESETS:write(out/(key+'-gate-before.json'),evaluate_gate(LeagueModel(out/'baseline',key)))
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'))
    try:
        for round_no in ((1,) if smoke else (1,2)):
            pinned=dict(current)
            for key in PRESETS:
                stage=f'{key}-round-{round_no}';dest=out/stage;dest.mkdir(exist_ok=True)
                export_model(nets[key],dest,key,builds[key],origin=origins[key]);current[key]=dest
                phase_end=min(time.time()+phase_seconds,deadline-evaluation_reserve)
                if phase_end<=time.time():raise TimeoutError('No phase budget remains')
                cycle=0;started=time.time()
                peers=[p for p in PRESETS if p!=key]
                anchor_model=LeagueModel(dest,key)
                replay=ErrorReplay();budget=OptimizerBudget()
                def selection_check():
                    jobs=[]
                    for pi,foe in enumerate(peers):
                        for pair in range(8):
                            for first in ('a','b'):
                                jobs.append({'deadline':phase_end,'seed':830000000+round_no*100000+PRESETS.index(key)*10000+pi*1000+pair,'first':first,'viewer':'a','decks':{'a':builds[key],'b':builds[foe]},'policies':{'a':(str(dest),key),'b':(str(pinned[foe]),foe)}})
                    result=list(pool.map(play,jobs))
                    return {'complete':all(g['complete'] for g in result),'wins':sum(g.get('winner')=='a' for g in result),'n':len(result)}
                best_check=selection_check()
                if not best_check['complete']:raise TimeoutError('Phase baseline selection did not complete')
                best_weights=deepcopy(nets[key].state_dict());best_optimizer=deepcopy(optimizers[key].state_dict());selected_cycle=0
                metric(out,{'stage':stage,'kind':'selection-baseline',**best_check})
                while time.time()<phase_end and (not max_cycles or cycle<max_cycles):
                    cycle+=1;seed=815000000+round_no*100000+PRESETS.index(key)*10000+cycle*100
                    write(out/'status.json',{'phase':stage,'cycle':cycle,'totals':total,'deadline_epoch':deadline})
                    jobs=[]
                    for j in range(4):
                        foe=peers[j%2];jobs.append({'deadline':phase_end,'seed':seed+j,'first':'ab'[j//2], 'viewer':'a','decks':{'a':builds[key],'b':builds[foe]},'policies':{'a':(str(dest),key),'b':(str(pinned[foe]),foe)},'sample':True,'collect':True})
                    games=list(pool.map(play,jobs));total[key]['ordinary_games']+=sum(g['complete'] for g in games)
                    stat={'updates':0,'optimizer_steps':0,'samples':0,'reason':'waiting_for_branch_credit'}
                    if budget.can_update_ordinary():
                        stat=ordinary_update(nets[key],optimizers[key],games,seed)
                        budget.record_ordinary(stat.get('optimizer_steps',0))
                    total[key]['ordinary_updates']+=stat['updates']
                    total[key]['ordinary_optimizer_steps']+=stat.get('optimizer_steps',0)
                    metric(out,{'stage':stage,'cycle':cycle,'kind':'ordinary',**stat})
                    roots=[(r,jobs[i]) for i,g in enumerate(games) for r in g.get('roots',[])];rng=Random(seed);rng.shuffle(roots)
                    teacher=LeagueModel(dest,key)
                    for ri,(root,sourcejob) in enumerate(roots[:3]):
                        if time.time()>=phase_end:break
                        total[key]['attempted_roots']+=1
                        actions,(x,c)=decision(root,'a');scores=teacher.scores(x,c)
                        ranked=list(np.argsort(-scores));selected=ranked if len(ranked)<=6 else ranked[:3]+rng.sample(ranked[3:],3)
                        # Precheck root eligibility without scheduling partial candidate sets.
                        try:determinize(root,'a',seed+40+ri*8)
                        except ValueError as exc:
                            metric(out,{'stage':stage,'cycle':cycle,'kind':'skip_root','reason':str(exc)});continue
                        lethal=certified_lethals(root,'a',actions)
                        selected+=sorted(lethal-set(selected))
                        branches=[]
                        for ai in selected:
                            for particle in range(8):
                                branches.append({'deadline':phase_end,'seed':seed+40+ri*8+particle,'root':root,'viewer':'a','action':actions[ai],'policies':sourcejob['policies'],'sample':bool(particle%2),'temperature':1.5 if particle%2 else 1.0})
                        results=list(pool.map(play,branches));total[key]['branch_games']+=sum(g['complete'] for g in results)
                        if not all(g['complete'] for g in results):continue
                        q=np.array([g['reward'] for g in results],dtype=np.float32).reshape(len(selected),8).mean(1)
                        auxiliary=auxiliary_pairs(results,len(selected),8,{i for i,ai in enumerate(selected) if ai in lethal})
                        kept=replay.add(x,c,selected,q,cycle,auxiliary)
                        total[key]['informative_roots']+=int(kept)
                        metric(out,{'stage':stage,'cycle':cycle,'kind':'branch_collect','particles':8,'candidates':len(selected),'terminal_returns':q.tolist(),'replay_admitted':kept,'certified_lethals':len(lethal),'auxiliary_pairs':auxiliary})
                    def current_scores(x,c):
                        with torch.no_grad():
                            xx,cc,mm=tensors([(x,c)],next(nets[key].parameters()).device)
                            return nets[key](xx,cc,mm)[0][0].cpu().numpy()
                    for _ in range(3):
                        if time.time()>=phase_end:break
                        error=replay.take(current_scores,cycle)
                        if error is None:break
                        x,c=error['x'],error['c'];selected=list(error['selected'])
                        result=branch_update(nets[key],optimizers[key],x,c[selected],error['q'],anchor_logits=anchor_model.scores(x,c),all_candidates=c,selected_indices=selected,auxiliary=error['auxiliary'])
                        budget.record_branch(result.get('optimizer_steps',0))
                        total[key]['branch_updates']+=result['updates']
                        total[key]['branch_optimizer_steps']+=result.get('optimizer_steps',0)
                        metric(out,{'stage':stage,'cycle':cycle,'kind':'branch','sample_age_cycles':cycle-error['cycle'],'sample_uses':error['uses'],'error_priority':error['priority'],**result})
                    export_model(nets[key],dest,key,builds[key],origin=origins[key])
                    if (cycle%5==0 or max_cycles) and time.time()<phase_end-5:
                        checked=selection_check();decision_label='keep_best'
                        if checked['complete'] and checked['wins']>=best_check['wins']+2:
                            best_check=checked;best_weights=deepcopy(nets[key].state_dict());best_optimizer=deepcopy(optimizers[key].state_dict());selected_cycle=cycle;decision_label='select_candidate'
                        elif checked['complete'] and checked['wins']<=best_check['wins']-4:
                            replay.clear()
                            prior_lr=min(g['lr'] for g in optimizers[key].param_groups)
                            nets[key].load_state_dict(best_weights);optimizers[key].load_state_dict(best_optimizer)
                            for group in optimizers[key].param_groups:group['lr']=max(1e-5,min(group['lr'],prior_lr)*.5)
                            export_model(nets[key],dest,key,builds[key],origin=origins[key]);decision_label='rollback_and_reduce_lr'
                        metric(out,{'stage':stage,'kind':'selection','cycle':cycle,'decision':decision_label,**checked})
                    torch.save({'schema':SCHEMA,'deck':key,'rule_hash':rule_hash(),'model':nets[key].state_dict(),'optimizer':optimizers[key].state_dict(),'totals':total[key],'origin':origins[key]},dest/'latest.pt')
                nets[key].load_state_dict(best_weights);optimizers[key].load_state_dict(best_optimizer)
                export_model(nets[key],dest,key,builds[key],origin=origins[key])
                torch.save({'schema':SCHEMA,'deck':key,'rule_hash':rule_hash(),'model':best_weights,'optimizer':best_optimizer,'selected_cycle':selected_cycle,'selection':best_check,'totals':total[key],'origin':origins[key]},dest/'selected.pt')
                write(dest/'status.json',{'phase':'complete','selected_cycle':selected_cycle,'selection':best_check,'elapsed':time.time()-started,'cycles':cycle,**total[key]})
        for key in PRESETS:export_model(nets[key],out/'final',key,builds[key],origin=origins[key])
        write(out/'training-summary.json',total)
        for key in PRESETS:write(out/(key+'-gate-after.json'),evaluate_gate(LeagueModel(out/'final',key)))
        # Paired first/second games against the identical frozen baseline strategies.
        evaluations=[];seed=945000000
        def evaluate(label,key,variant,foe,foe_variant,pairs,teams=None):
            nonlocal seed
            write(out/'status.json',{'phase':'evaluate:'+label,'completed_cells':len(evaluations),'totals':total,'deadline_epoch':deadline})
            jobs=[];info=[]
            for pair in range(pairs):
                opponent=sample_deck(seed+pair,teams[pair]) if teams else builds[foe] if foe else builds[PRESETS[pair%3]]
                for first in ('a','b'):
                    jobs.append({'deadline':deadline-60,'seed':seed+pair,'first':first,'viewer':'a','decks':{'a':builds[key],'b':opponent},'policies':{'a':(str(out/variant),key),'b':(str(out/foe_variant),foe) if foe else None}})
                    info.append({'pair':pair,'position':'first' if first=='a' else 'second','opponent_team':opponent['character_ids']})
            results=list(pool.map(play,jobs));rows=[{**meta,'complete':g['complete'],'winner':g.get('winner'),'steps':g['steps'],'remaining_turns':g.get('remaining_turns'),'own_decisions':g.get('own_decisions')} for meta,g in zip(info,results)]
            row={'label':label,'key':key,'variant':variant,'foe':foe,'seed':seed,'games':rows,'complete':all(g['complete'] for g in results),'wins':sum(g.get('winner')=='a' for g in results),'n':sum(g['complete'] for g in results)}
            evaluations.append(row);write(out/'evaluations'/f'{label}.json',row)
            return row
        comparisons={}
        for key in PRESETS:
            for foe in (p for p in PRESETS if p!=key):
                # Do not change seed between baseline and final.
                before=evaluate(f'{key}-baseline-vs-{foe}',key,'baseline',foe,'baseline',1 if smoke else 8)
                after=evaluate(f'{key}-final-vs-{foe}',key,'final',foe,'baseline',1 if smoke else 8)
                comparisons[key+'-vs-'+foe]={'before':before['wins'],'after':after['wins'],'n':after['n'],'complete':before['complete'] and after['complete']}
                seed+=10000
        for a,b in combinations(PRESETS,2):
            evaluate(f'cross-{a}-{b}',a,'final',b,'final',1 if smoke else 16);seed+=10000
        teams=[tuple(builds['quick-rush']['character_ids'])] if smoke else list(combinations(SEATS,4))
        for key in PRESETS:
            label=key+('-public-smoke' if smoke else '-public-70-teams')
            evaluate(label,key,'final',None,None,len(teams),teams);seed+=10000
        write(out/'evaluation-index.json',evaluations)
        acceptance={'experiment_complete':all(e['complete'] for e in evaluations),'same_frozen_comparisons':comparisons,'training':total,'automatic_serving_approval':False,'win_prediction_calibrated':False}
        write(out/'acceptance.json',acceptance)
        write(out/'status.json',{'phase':'complete' if acceptance['experiment_complete'] else 'incomplete_evaluation','finished_epoch':time.time(),'deadline_epoch':deadline,'totals':total})
        lines=['# 三套策略共同训练测试','',f'规则：{rule_hash()}',f'正式 Python 引擎结算；训练设备 {device}。三套合计 {total_seconds/60:g} 分钟上限；未自动发布模型。','', '| 配对 | 初始胜局 | 新模型胜局 | 每组局数 |','| --- | ---: | ---: | ---: |']
        for k,r in comparisons.items():lines.append(f"| {k} | {r['before']} | {r['after']} | {r['n']} |")
        (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
    manifest={}
    with zipfile.ZipFile(out/'delivery.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if not p.is_file() or p.suffix not in ('.json','.jsonl','.npz','.pt','.md'):continue
            name=p.relative_to(out).as_posix();data=p.read_bytes();manifest[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
        source_path=Path(__file__).resolve().parents[4]/'source-manifest.json'
        source_data=source_path.read_bytes();manifest['source-manifest.json']=hashlib.sha256(source_data).hexdigest();z.writestr('source-manifest.json',source_data)
        z.writestr('delivery-manifest.json',json.dumps(manifest,indent=2))
    write(out/'delivery.json',{'sha256':hashlib.sha256((out/'delivery.zip').read_bytes()).hexdigest(),'files':len(manifest)})
