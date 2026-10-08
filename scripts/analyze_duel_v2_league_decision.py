#!/usr/bin/env python3
"""Compare a decision's legal moves by paired hidden-information terminal rollouts."""
import argparse,json,multiprocessing,os,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def wilson(wins,n):
    if not n:return None
    z=1.96;p=wins/n;center=(p+z*z/(2*n))/(1+z*z/n);spread=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
    return [max(0,center-spread),min(1,center+spread)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--side',choices=('a','b'),default='a')
    p.add_argument('--model',type=Path,required=True);p.add_argument('--opponent-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--particles',type=int,default=32);p.add_argument('--seconds',type=int,default=120);p.add_argument('--workers',type=int,default=6);p.add_argument('--seed',type=int,default=955000000)
    a=p.parse_args()
    if not 2<=a.particles<=128 or not 1<=a.seconds<=300 or not 1<=a.workers<=12:p.error('Bounded particles=2..128, seconds=1..300, workers=1..12')
    if a.output.exists():p.error('Choose a new output directory')
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    import hashlib,numpy as np
    from app.modules.card_game.rl.league_rollout import play,decision,determinize
    from app.modules.card_game.rl.league_policy import LeagueModel
    from app.modules.card_game.rl.league_schema import rule_hash
    from app.modules.card_game.engine.duel_v2.entities import hydrate_entities
    from app.modules.card_game.engine.duel_v2.projection import action_label
    raw=a.snapshot.read_bytes();s=json.loads(raw);hydrate_entities(s)
    learner=LeagueModel(a.model.parent,a.model.stem);opponent=LeagueModel(a.opponent_model.parent,a.opponent_model.stem)
    actions,(x,c)=decision(s,a.side)
    if len(actions)>64:p.error('This tool supports up to64 legal candidates; choose a normal playing snapshot')
    determinize(s,a.side,a.seed) # Fail closed for unresolved choices or inconsistent public counts.
    logits,value=learner.scores_value(x,c);probs=np.exp(logits-logits.max());probs/=probs.sum();chosen=int(logits.argmax())
    deadline=time.time()+a.seconds;foe='b' if a.side=='a' else 'a';jobs=[]
    for action in actions:
        for i in range(a.particles):jobs.append({'deadline':deadline,'seed':a.seed+i,'root':s,'action':action,'viewer':a.side,'policies':{a.side:(str(a.model.parent),a.model.stem),foe:(str(a.opponent_model.parent),a.opponent_model.stem)},'sample':bool(i%2),'temperature':1.5 if i%2 else 1.})
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:results=list(pool.map(play,jobs))
    rows=[]
    for index,action in enumerate(actions):
        sample=results[index*a.particles:(index+1)*a.particles];n=sum(g['complete'] for g in sample);wins=sum(g.get('winner')==a.side for g in sample)
        rows.append({'action':action,'label':action_label(s,a.side,action),'policy_probability':float(probs[index]),'selected':index==chosen,'finished':n,'requested':a.particles,'wins':wins,'win_rate':wins/n if n else None,'wilson95':wilson(wins,n),'complete':n==a.particles})
    payload={'snapshot_sha256':hashlib.sha256(raw).hexdigest(),'analysis_rule_hash':rule_hash(),'scope':'This snapshot re-evaluated under current rule hash; not a historical-rule replay proof','seed':a.seed,'learner_model':learner.version,'opponent_model':opponent.version,'particles':a.particles,'continuation':'half frozen greedy, half frozen temperature1.5; paired hidden worlds','complete':all(r['complete'] for r in rows),'value_return_uncalibrated':value,'actions':rows,'warning':'Finite rollout win frequencies and approximate marginal intervals, not calibrated network win probabilities. Do not rank incomplete rows.'}
    a.output.mkdir(parents=True);(a.output/'decision.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 候选操作的终局胜率估计','','同一局面、相同隐藏样本、相同冻结续打策略。按当前规则重新计算，不冒充历史录像原规则复现。','动作概率表示模型愿不愿意选它；模拟胜率表示采样续打结果，两者分开。','','| 操作 | 模型动作概率 | 模拟胜局/完成 | 模拟胜率 | 约95%区间 |','| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        ci=row['wilson95'];rate='未完成' if not row['complete'] else f"{row['win_rate']:.1%}";interval='—' if not row['complete'] else f'{ci[0]:.1%}–{ci[1]:.1%}'
        lines.append(f"| {'→ ' if row['selected'] else ''}{row['label']} | {row['policy_probability']:.1%} | {row['wins']}/{row['finished']} | {rate} | {interval} |")
    lines+=['','→ 为模型当前首选。少量样本的区间较宽，续打策略也会影响结果；不据此声称真实对人胜率或给微小差异定性。']
    (a.output/'decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({'complete':payload['complete'],'candidates':len(rows),'output':str(a.output)}))
if __name__=='__main__':main()
