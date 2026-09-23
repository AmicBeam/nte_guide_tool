"""New-model first/second matrix and reproducible telemetry tables."""
from collections import Counter,defaultdict
from pathlib import Path
import csv,gzip,json


def table(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        if rows:
            fields=list(dict.fromkeys(k for r in rows for k in r))
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def write_matrix(out,results,keys,models,config):
    out=Path(out);cells={};seen=set();identities={}
    names=dict(starter='娜伊零九',**{'weave-rush':'薄白零伊','quick-rush':'吱零薄九'},zhenhong='真红零伊翳',murk='浊燃',midrange='娜白零伊')
    for k in keys:identities[k]=json.loads((Path(models)/(k+'.json')).read_text(encoding='utf-8'))
    for a in keys:
        for b in keys:
            cells[a,b]=dict(row=a,column=b,planned=(2 if a==b else 1) if config['smoke'] else config.get('matrix_games',128),attempts=0,completed=0,first_wins=0,second_wins=0,draws=0,errors=0,truncated=0,row_model_sha256=identities[a]['sha256'],column_model_sha256=identities[b]['sha256'],row_build_sha256=identities[a]['build_sha256'],column_build_sha256=identities[b]['build_sha256'],rule_hash=config['rule_hash'],inference=config.get('inference','argmax'))
    mirrors=[]
    for r in results:
        if not r['cell'].startswith(('mirror/','cross/')):continue
        if r['id'] in seen:raise ValueError('Duplicate matrix game')
        seen.add(r['id']);first=r['first'];second='b' if first=='a' else 'a'
        a=r['policies'][first][1];b=r['policies'][second][1];c=cells[a,b];c['attempts']+=1
        if not r['complete']:
            c['errors']+=bool(r.get('error'));c['truncated']+=not bool(r.get('error'));continue
        c['completed']+=1;c['first_wins']+=r['winner']==first;c['second_wins']+=r['winner']==second;c['draws']+=r['winner'] not in ('a','b')
        if a==b:mirrors.append(r)
    for c in cells.values():c['first_winrate']=c['first_wins']/c['completed'] if c['completed'] else None
    table(out/'first-second-winrates.csv',list(cells.values()))
    lines=['# 本轮新模型互搏报告','','行＝先手，列＝后手，每格为先手胜率（胜局/正常完成局）；对角线为同模型同构筑内战。旧陪练对照仅见附录。','','| 先手＼后手 | '+' | '.join(names.get(k,k) for k in keys)+' |','| --- | '+' | '.join(['---']*len(keys))+' |']
    for a in keys:
        values=[]
        for b in keys:
            c=cells[a,b]
            values.append(f"{c['first_winrate']:.1%}（{c['first_wins']}/{c['completed']}）" if c['completed']==c['planned'] else f"未完成（{c['completed']}/{c['planned']}）")
        lines.append('| '+names.get(a,a)+' | '+' | '.join(values)+' |')
    changed={}
    initial=Path(config.get('initial_models',out/'initial'))
    import numpy as np
    for k in keys:
        before=initial/(k+'.npz');after=Path(models)/(k+'.npz')
        if not before.exists():changed[k]=None;continue
        with np.load(before,allow_pickle=False) as a,np.load(after,allow_pickle=False) as b:
            changed[k]=set(a.files)!=set(b.files) or any(not np.array_equal(a[n],b[n]) for n in a.files)
    verified_new=all(v is True for v in changed.values())
    if not verified_new:
        lines[0]='# 冻结模型互搏报告（不满足全套训练后新模型要求）'
        lines.insert(2,'权重核验：'+json.dumps(changed,ensure_ascii=False)+'；false为数值未变，null为缺少初始权重无法核验。不得把此表标作全套新模型互搏。')
    counts=defaultdict(Counter);denom=Counter();mechanisms=[];turns=[];immune=[];opening=[];contributions=[]
    from .quick_adaptation import contribution_events
    for r in results:
        if not r['complete']:continue
        path=out/'evaluation/raw'/f"{r['id']}.json.gz"
        with gzip.open(path,'rt',encoding='utf-8') as f:raw=json.load(f)
        t=raw['telemetry']
        if r['cell'].startswith('opening/'):
            opening.append(dict(game_id=r['id'],cell=r['cell'],first=r['first'],winner=r['winner'],opening=t['opening'],turns=[x for x in t['turn_states'] if x['turn']<=6]))
        if not r['cell'].startswith(('mirror/','cross/')):continue
        for side in ('a','b'):
            key=r['policies'][side][1];pos='first' if side==r['first'] else 'second';win=r['winner']==side
            immune.append(dict(game_id=r['id'],key=key,position=pos,metrics=raw.get('immune_window',{}).get(side,{}),risks=raw.get('immune_risks',{}).get(side,[])))
            events=raw['events']
            if side=='b':
                events=[dict(e,**{k:('a:'+e[k][2:] if e[k].startswith('b:') else 'b:'+e[k][2:]) for k in ('actor','target') if isinstance(e.get(k),str) and e[k][:2] in ('a:','b:')}) for e in events]
            for hero,m in contribution_events(events,r['decks'][side]['character_ids']).items():
                contributions.append(dict(game_id=r['id'],key=key,position=pos,character=hero,**m))
            if not r['cell'].startswith('mirror/'):continue
            denom[key,pos]+=1
            carried=Counter(r['decks'][side]['card_ids'])
            for cid in set(carried)|set(t['cards'][side]):
                data=t['cards'][side].get(cid,{});bucket=counts[key,pos,cid];sources=data.get('sources',{})
                bucket['carried_copies']=carried.get(cid,0)
                bucket['seen_games']+=bool(sources);bucket['seen_wins']+=bool(sources) and win
                bucket['used_games']+=bool(data.get('plays'));bucket['used_wins']+=bool(data.get('plays')) and win
                bucket['plays']+=data.get('plays',0);bucket['original_plays']+=data.get('original_plays',0);bucket['generated_plays']+=data.get('derived_plays',0);bucket['copy_plays']+=data.get('copy_plays',0)
                bucket['first_play_turn_sum']+=data.get('first_play_turn') or 0
                for source,n in sources.items():bucket['obtained_'+source]+=n
        mechanisms.extend(dict(game_id=r['id'],cell=r['cell'],**m) for m in t['mechanisms'])
        turns.extend(dict(game_id=r['id'],cell=r['cell'],**m) for m in t['turn_states'])
    card_rows=[]
    for (key,pos,cid),m in counts.items():
        card_rows.append(dict(key=key,position=pos,card_id=cid,games=denom[key,pos],**m,seen_winrate=m['seen_wins']/m['seen_games'] if m['seen_games'] else None,used_winrate=m['used_wins']/m['used_games'] if m['used_games'] else None,first_play_mean=m['first_play_turn_sum']/m['used_games'] if m['used_games'] else None))
    table(out/'selfplay-card-metrics.csv',card_rows);table(out/'character-contributions.csv',contributions)
    for name,rows in [('mechanisms',mechanisms),('turn-states',turns),('immune-observations',immune),('opening-analysis',opening)]:
        (out/(name+'.json')).write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    lines+=['','逐卡表仅来自内战，先后手分开；角色贡献按事件实际生命与护盾变化统计。免伤期间仅记录队友倒地，不检查空前排，也不把所有牺牲自动判错。','', '未提供经独立校准的预测胜率；过程奖励不计入胜率。详细动作、事件、开局与回放位于 evaluation/raw 和 evaluation/replays。','', '## 旧陪练及其他对照附录','','见 evaluation/index.jsonl 与 evaluation/summary.json，按 cell 区分旧模型、规则对手、变化构筑和换牌对照。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    complete=all(c['completed']==c['planned'] and not c['errors'] and not c['truncated'] for c in cells.values())
    (out/'matrix-validation.json').write_text(json.dumps(dict(complete=complete,new_model_complete=complete and verified_new,model_changed=changed,cells=len(cells),unique_games=len(seen),identities=identities),ensure_ascii=False,indent=2),encoding='utf-8')
    return complete and verified_new
