"""Render full-chain feasibility evidence from frozen reports and scalar training metrics."""
from collections import Counter,defaultdict
from pathlib import Path
import csv
import hashlib
import json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rate(win,total):
    return f'{win/total:.1%}' if total else '—'


def write_full_report(root):
    root=Path(root)
    index=read(root/'evaluation-index.json');decisions=read(root/'construction-decisions.json')
    selected_path=root/'selection/selected.json'
    serving_selection=read(selected_path) if selected_path.is_file() else None
    if serving_selection is not None and not serving_selection.get('complete'):
        raise ValueError('Final selection validation is incomplete')
    final_variants={key:'selected' if serving_selection else value['selection'] for key,value in decisions.items()}
    if serving_selection:
        screening=read(root/'selection/screen-results.json')
        for key,value in screening.items():
            for candidate in value['candidates']:
                for cell in candidate['cells']:
                    index.append({'label':f"shortlist-{key}-{candidate['id']}-{cell['opponent']}",
                                  'preset':key,'variant':'screening','report':cell['report'].replace('\\','/'),'phase':'screening'})
        for cell in serving_selection['reports']:
            index.append({'label':cell['label'],'preset':cell['preset'],'variant':'selected' if 'selected' in cell['label'] else 'old-website',
                          'report':cell['report'].replace('\\','/'),'phase':'holdout'})
    config=read(root/'config.json')
    comprehensive=config.get('kind')=='comprehensive'
    lines=['# 双预组综合训练报告' if comprehensive else '# 双预组完整链路短测报告','',
           '这是本轮有界综合训练与冻结评估报告，单次结果不证明全局最优或游戏平衡。' if comprehensive else '这是有界短训练的可行性验证，样本不用于证明套牌最优或游戏平衡。',
           '',f"- 规则身份：`{config['rule_hash']}`。",
           '- 训练标量与冻结评估分别列出；报告生成不批准上线。',
           '- 预测胜率：未提供经校准的概率模型；附逐回合局势 CSV，原始价值分数不冒充胜率。',
           '- 终结/环合效果附源事件；同动作内变化可能由多个效果共同造成，未声称因果胜率贡献。',
           '', '## 训练与配牌阶段','']
    for key in ('starter','weave-rush'):
        lines+=['',f'### {key} 阶段结果','']
        base_config=root/key/'base/config.json'
        if base_config.is_file():
            setup=read(base_config)
            origin=setup.get('resume') or setup.get('warm_start') or '从零初始化'
            lines += [f"本轮基础模型起点：{origin}；hidden={setup.get('hidden','未记录')}。训练预算及停止原因见阶段记录。",'']
        lines+=['| 队伍 | 阶段 | 累计更新 | 本轮更新 | 正常终局 | 用时秒 |','| --- | --- | ---: | ---: | ---: | ---: |']
        for stage in ('base','cards','adapted'):
            status=read(root/key/stage/'status.json')
            metrics=[json.loads(x) for x in (root/key/stage/'metrics.jsonl').read_text().splitlines()]
            games=sum(x.get('finished_games',0) for x in metrics) if stage!='cards' else sum(
                json.loads(x)['trials']-json.loads(x)['truncated'] for x in (root/key/stage/'candidate-history.jsonl').read_text().splitlines())
            lines.append(f"| {key} | {stage} | {status['update']} | {len([x for x in metrics if x.get('updated',True)])} | {games} | {status.get('seconds',metrics[-1].get('seconds',0)):.2f} |")
        lines+=['',f'### {key} 训练内先后手统计（出牌阶段）','',
                '| 阶段 | 对手 | 位置 | 胜/负/平 | 胜率 |','| --- | --- | --- | --- | ---: |']
        for stage in ('base','adapted'):
            position_data=read(root/key/stage/'position-results.json')
            groups={**position_data['opponents'],**{'阵容池/'+k:v for k,v in position_data.get('opponent_pools',{}).items()}}
            for foe,positions in groups.items():
                for position,row in positions.items():
                    lines.append(f"| {stage} | {foe} | {position} | {row['win']}/{row['loss']}/{row['draw']} | {rate(row['win'],row['games'])} |")
        lines+=['','每次更新的累计分组统计在对应 `metrics.jsonl`；停止时未完成局另见 `position-results.json`。对手类型与阵容池是同批对局的不同切分，不可相加。']
    lines+=['','## 冻结评估矩阵','','| 测试 | 场数 | 学习方先手胜率 | 学习方后手胜率 | 实际先手方胜率 | 平均全局回合 | 异常/截断 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    generated_rows=[]
    card_rows=[];mechanism_rows=[];curve_rows=[];summary_rows=[];chosen_card_rows=[];chosen_mechanisms=[]
    for entry in index:
        report=read(root/entry['report']);games=report['games_index']
        complete=[g for g in games if g['terminated'] and not g['error'] and not g['truncated']]
        positions=report['by_position']
        first_wins=sum(g['winner']==g['first_side'] for g in complete)
        mean_turn=sum(g['final_turn'] for g in complete)/len(complete) if complete else None
        pos_rate=lambda p:rate(positions[p]['win'],sum(positions[p][k] for k in ('win','loss','draw')))
        mean_text=f'{mean_turn:.2f}' if mean_turn is not None else '—'
        lines.append(f"| {entry['label']} | {len(complete)} | {pos_rate('first')} | {pos_rate('second')} | {rate(first_wins,len(complete))} | {mean_text} | {sum(bool(g['error']) for g in games)}/{sum(bool(g['truncated']) for g in games)} |")
        if entry['label'].startswith('cross-') or entry['label']=='selected-cross':
            first=positions['second'];second=positions['first']
            lines.append(f"| {entry['label']}（覆纹快攻视角，同批对局） | {len(complete)} | {rate(first['loss'],sum(first[k] for k in ('win','loss','draw')))} | {rate(second['loss'],sum(second[k] for k in ('win','loss','draw')))} | {rate(first_wins,len(complete))} | {mean_text} | 同上，勿重复计局 |")
        summary_rows.append({'label':entry['label'],'phase':entry.get('phase','screening' if 'screen' in entry['label'] else 'matrix'),'games':len(complete),'first_wins':first_wins,
            'mean_global_turn':mean_turn,'by_position':positions,'model':report['model'],
            'opponent_model':report['opponent_model'],'complete':report['complete'],
            'opponent_teams':len({tuple(sorted(g['opponent_team'])) for g in games})})
        mirror=entry['label'].endswith('-mirror')
        if mirror and (not report['model'].get('sha256') or report['model']['sha256']!=report['opponent_model'].get('sha256')):
            raise ValueError('Mirror model mismatch')
        by_position=defaultdict(list)
        for game in complete:
            raw=read(root/entry['report'].rsplit('/',1)[0]/game['raw_log'].replace('\\','/'))
            if mirror:
                opponent=raw['opponent_deck']
                if (opponent['character_ids']!=report['learner_build']['character_ids'] or
                        Counter(opponent['card_ids'])!=Counter(report['learner_build']['card_ids'])):
                    raise ValueError('Mirror build mismatch')
            for side in ('a','b') if mirror else ('a',):
                position='first' if game['first_side']==side else 'second'
                by_position[position].append((game,raw['telemetry'],side))
        counts=Counter(report['learner_build']['card_ids'])
        for position,samples in by_position.items():
            all_ids={cid for _,data,side in samples for cid in data['cards'][side]}
            for cid in sorted(all_ids):
                for origin,field,source in (('copy','copy_plays','copy'),('derived','derived_plays','generated')):
                    selected=[(game,data['cards'][side].get(cid,{}),side) for game,data,side in samples]
                    plays=sum(card.get(field,0) for _,card,_ in selected)
                    received=sum(bool(card.get('sources',{}).get(source)) for _,card,_ in selected)
                    used=[(game,side) for game,card,side in selected if card.get(field,0)]
                    if plays or received:
                        generated_rows.append({'test':entry['label'],'position':position,'card_id':cid,'origin':origin,
                            'games':len(samples),'received_games':received,'used_games':len(used),'plays':plays,
                            'wins_when_used':sum(game['winner']==side for game,side in used),
                            'mean_final_turn_when_used':sum(game['final_turn'] for game,_ in used)/len(used) if used else None})
            for cid,copies in counts.items():
                seen=drawn=used=w_seen=w_drawn=w_used=plays=0;first_turns=[];used_lengths=[];sources=Counter()
                for game,data,side in samples:
                    card=data['cards'][side].get(cid,{})
                    source=card.get('sources',{});sources.update(source)
                    has_seen=any(source.get(k,0) for k in ('opening','draw','mulligan','search','recovered','revealed_gain','other_gain'))
                    has_drawn=bool(source.get('draw'))
                    has_used=bool(card.get('original_plays'))
                    win=game['winner']==side
                    seen+=has_seen;drawn+=has_drawn;used+=has_used
                    w_seen+=has_seen and win;w_drawn+=has_drawn and win;w_used+=has_used and win
                    plays+=card.get('original_plays',0)
                    if has_used:
                        first_turns.append(card['first_original_play_turn']);used_lengths.append(data['final_turn'])
                row={'test':entry['label'],'preset':entry['preset'],'variant':entry['variant'],'position':position,
                    'card_id':cid,'copies':copies,'games':len(samples),'seen_games':seen,'drawn_games':drawn,
                    'used_games':used,'seen_rate':seen/len(samples),'draw_rate':drawn/len(samples),'play_rate':used/len(samples),
                    'win_rate_when_seen':w_seen/seen if seen else None,'win_rate_when_drawn':w_drawn/drawn if drawn else None,
                    'win_rate_when_used':w_used/used if used else None,'wins_when_seen':w_seen,'wins_when_drawn':w_drawn,'wins_when_used':w_used,
                    'plays':plays,'mean_plays_per_game':plays/len(samples),'mean_first_play_turn':sum(first_turns)/len(first_turns) if first_turns else None,
                    'mean_final_turn_when_used':sum(used_lengths)/len(used_lengths) if used_lengths else None,
                    'source_counts':json.dumps(sources,ensure_ascii=False)}
                card_rows.append(row)
                if entry['label']==f"{entry['preset']}-{final_variants[entry['preset']]}-mirror":chosen_card_rows.append(row)
            for cid in report['learner_build']['character_ids']:
                for kind in ('ultimate','harmony'):
                    events=[];used_games=win_games=0;opportunities=0;used_opportunities=set();foe_hp_delta=0
                    for game,data,side in samples:
                        selected=[e for e in data['mechanisms'] if e['kind']==kind and e['actor']==f'{side}:{cid}']
                        events.extend(selected);used_games+=bool(selected);win_games+=bool(selected) and game['winner']==side
                        opportunity_kind=kind
                        opportunities+=sum(o['kind']==opportunity_kind and o['side']==side and o['character_id']==cid for o in data['opportunities'])
                        used_opportunities.update((game['game_id'],side,e['turn']) for e in selected)
                        foe_hp_delta+=sum(e['same_action_player_hp_change']['b' if side=='a' else 'a'] for e in selected)
                    row={'test':entry['label'],'position':position,'character':cid,'kind':kind,'games':len(samples),
                        'uses':len(events),'games_used':used_games,'wins_when_used':win_games,'eligible_character_turns':opportunities,
                        'used_character_turns':len(used_opportunities),'first_use_mean':None,
                        'same_action_foe_hp_delta_sum':foe_hp_delta,
                        'attribution':'same-action association, not causal; opportunities count legal ordinary and battle-card attacks'}
                    firsts=[min(e['turn'] for e in data['mechanisms'] if e['kind']==kind and e['actor']==f'{side}:{cid}')
                            for _,data,side in samples if any(e['kind']==kind and e['actor']==f'{side}:{cid}' for e in data['mechanisms'])]
                    row['first_use_mean']=sum(firsts)/len(firsts) if firsts else None
                    mechanism_rows.append(row)
                    if entry['label']==f"{entry['preset']}-{final_variants[entry['preset']]}-mirror":chosen_mechanisms.append(row)
            turns=defaultdict(list)
            for game,data,side in samples:
                for point in data['turn_states']:turns[point['turn']].append(point['sides'][side])
            for turn,points in sorted(turns.items()):
                curve_rows.append({'test':entry['label'],'position':position,'turn':turn,'surviving_games':len(points),
                    **{f'mean_{k}':sum(x[k] for x in points)/len(points) for k in ('hp','ap','energy','harmony')},
                    'predicted_win_probability':None})
    lines+=['','交叉互搏的首行以创生为学习方；第二行换算覆纹快攻视角，来自同批对局。','','## 训练末批候选对照（非最终发布选择）' if serving_selection else '## 构筑调整与理由','',
            '下表只依据独立于最终测试集的筛选样本；小样本用于验证选择流程，不能据此声称最优。候选生成过程是配牌策略采样，不伪造逐张换牌的模型内心理由。']
    construction=[]
    candidate_decisions=[]
    from app.modules.card_game.content.duel_v2 import CARDS
    for key,decision in decisions.items():
        baseline=read(root/key/'original-deck.json')
        selected_deck=root/key/'selected-deck.json'
        candidate=read(selected_deck if selected_deck.is_file() else root/key/'cards/candidate.json')
        a=Counter(baseline['card_ids']);b=Counter(candidate['card_ids'])
        delta=[{'card_id':cid,'name':CARDS[cid]['name'],'before':a[cid],'after':b[cid]} for cid in sorted(a.keys()|b.keys()) if a[cid]!=b[cid]]
        construction.append({'preset':key,**decision,'changes':delta})
        lines+=['',f'### {key}', '',f"- 原始筛选胜率：{decision['original_win_rate']:.1%}；候选：{decision['candidate_win_rate']:.1%}；每种 {decision['sample_games_each']} 场。",
                f"- 选择：{decision['selection']}；理由：{decision['reason']}。",
                f'- 全批候选及原始评分：`{key}/cards/candidate-history.jsonl`。',
                f'- 筛选证据：`{decision.get("evidence", "旧版配对筛选")}`；最差阵容胜率：{decision.get("worst_lineup", "未记录")}；最弱位置胜率：{decision.get("worst_seat", "未记录")}。',
                '- 候选适应训练单独完成；本表筛选使用冻结基础策略，最终矩阵另外展示适应后效果。',
                '', '| 卡牌 | 原始张数 | 候选张数 |','| --- | ---: | ---: |']
        lines += [f"| {x['card_id']} {x['name']} | {x['before']} | {x['after']} |" for x in delta]
        tournament_path=root/key/'candidate-selection.json'
        if tournament_path.is_file():
            tournament=read(tournament_path)
            lines+=['','#### 分级候选比较（筛选集，不是独立留出）','',
                    f"完成：{tournament['complete']}；选择：{tournament['selection']}；{tournament['reason']}。",'']
            if tournament.get('changes'):
                lines+=['','局部搜索换牌过程（得分仅代表同轮共同筛选集）：','',
                        '| 轮次 | 换出 | 换入 | 父配牌得分 | 选中得分 |',
                        '| --- | --- | --- | ---: | ---: |']
                for change in tournament['changes']:
                    lines.append(f"| {change['round']} | {change['removed']} | {change['added']} | {change['parent_score']:.2%} | {change['selected_score']:.2%} |")
                lines+=['','局部初筛仅展示前 8 及基线；完整候选、父配牌和计数见 JSON。','']
            lines+=['| 层级 | 候选 | 对手阵容 | 位置 | 胜/完成 | 截断 |','| --- | --- | --- | --- | --- | ---: |']
            for stage in tournament['stages']:
                shown=stage['results']
                if str(stage['stage']).endswith('-screen'):
                    ranked=sorted(shown,key=lambda r:-(r.get('metrics') or {}).get('utility',-1))[:8]
                    shown=[r for r in shown if r in ranked or r['id']=='original']
                for result in shown:
                    for lineup,positions in result['groups'].items():
                        for position,bucket in positions.items():
                            lines.append(f"| {stage['stage']} | {result['id']} | {lineup} | {position} | {bucket['wins']}/{bucket['games']} | {bucket.get('truncated',0)} |")
            lines+=['','模型身份与对手实际配牌分别记录在 candidate-selection.json；编译配对采用偶/奇种子，不声称牌序完全相同。']


        display=[]
        for text in (root/key/'cards/candidate-history.jsonl').read_text(encoding='utf-8').splitlines():
            history=json.loads(text)
            screened=history['candidate_id']==candidate['id'] and history['valid']
            reason='选入训练末批配对筛选' if screened else '截断，排除更新' if not history['valid'] else '本批未选中；不据此认定构筑更差'
            row={'preset':key,**history,'screened':screened,'reason':reason}
            candidate_decisions.append(row);display.append(row)
        lines += ['',f"本表仅列 {min(12,len(display))} 项代表记录；全部 {len(display)} 项保存在 candidate-decisions.json。",'']
        lines+=['| 采样版本 | 候选 | 先手胜/完成 | 后手胜/完成 | 筛选分 | 处理 |','| --- | --- | --- | --- | ---: | --- |']
        for history in sorted(display,key=lambda row:(row['screened'],row['valid'],row.get('score',-2),row.get('sampling_update',0)),reverse=True)[:12]:
            first=history.get('by_position',{}).get('first',{})
            second=history.get('by_position',{}).get('second',{})
            lines.append(f"| {history.get('sampling_update','—')} | {history.get('candidate_id','—')} | {first.get('wins','—')}/{first.get('games','—')} | {second.get('wins','—')}/{second.get('games','—')} | {history.get('score','—')} | {history['reason']} |")

    if serving_selection:
        shortlist=read(root/'selection/shortlist.json')
        lines+=['','## 最终模型与优选构筑','',
                '从完整搜索记录形成有限短名单，再用独立配对筛选；最终留出集不参与选择。最高分同分时保留原始配牌，不声称穷举全局最优。']
        for key,item in serving_selection['choices'].items():
            lines+=['',f'### {key} 最终选择：{item["id"]}','',
                    f"- 数值模型 SHA：`{item['model_sha256']}`；来源 `{key}/{item['model']}`。",
                    f"- 搜索记录 {shortlist[key]['history_rows']} 条，去重构筑 {shortlist[key]['unique_allocations']} 套，短名单 {len(shortlist[key]['shortlist'])} 项。",
                    '', '| 短名单候选 | 筛选胜/完成 | 胜率 |','| --- | ---: | ---: |']
            for candidate in screening[key]['candidates']:
                lines.append(f"| {candidate['id']} | {candidate['wins']}/{candidate['games']} | {rate(candidate['wins'],candidate['games'])} |")
            original=Counter(read(root/key/'original-deck.json')['card_ids']);chosen=Counter(item['build']['card_ids'])
            lines+=['','| 卡牌 | 原始张数 | 最终张数 |','| --- | ---: | ---: |']
            changes=[cid for cid in sorted(original.keys()|chosen.keys()) if original[cid]!=chosen[cid]]
            for cid in changes:lines.append(f"| {cid} {CARDS[cid]['name']} | {original[cid]} | {chosen[cid]} |")
            if not changes:lines.append('| 配牌保持原始预组 | 32 | 32 |')
    lines+=['','## 本轮选定配牌：逐卡与机制摘要','','下表仅使用最终同模型、同构筑镜像互搏，统计双方并按实际先后手分组；规则 AI 不参与主逐卡统计，交叉互搏另列于诊断 CSV。每套 64 场时各有 64 个先手和后手观测，仍是 64 场（32 个种子配对），不是 128 场独立对局。未携带或未使用的卡没有对应使用胜率，不能据此声称全卡池已有覆盖。抽到仅计 draw 事件，见到还包括起手、换牌、检索与回收等原卡入手。成功使用只计原卡，复制/衍生来源另列。',
            '', '| 套牌 | 位置 | 卡牌 | 张数 | 见到局数/总数 | 打出局数 | 打出时胜率 | 平均首次打出回合 | 使用局平均结束回合 |',
            '| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |']
    if not chosen_card_rows:
        lines+=['','缺少最终镜像逐卡数据；本报告未完成，不以规则 AI 数据回填。','']
    for r in chosen_card_rows:
        val=lambda x:'—' if x is None else f'{x:.2f}'
        lines.append(f"| {r['preset']} | {r['position']} | {r['card_id']} {CARDS[r['card_id']]['name']} | {r['copies']} | {r['seen_games']}/{r['games']} | {r['used_games']} | {rate(r['wins_when_used'],r['used_games'])} | {val(r['mean_first_play_turn'])} | {val(r['mean_final_turn_when_used'])} |")
    lines+=['','### 终结与环合','','| 测试 | 位置 | 角色 | 机制 | 次数 | 使用局数 | 使用局胜率 | 可用角色回合 | 首次使用回合 | 同动作对手生命变化/次 |',
            '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in chosen_mechanisms:
        lines.append(f"| {r['test']} | {r['position']} | {r['character']} | {r['kind']} | {r['uses']} | {r['games_used']} | {rate(r['wins_when_used'],r['games_used'])} | {r['eligible_character_turns']} | {r['first_use_mean'] if r['first_use_mean'] is not None else '—'} | {r['same_action_foe_hp_delta_sum']/r['uses'] if r['uses'] else '—'} |")
    lines+=['','环合类型、提供者、受益者、终结结束及源效果事件保留在每局 telemetry；机会以存在合法发动动作的角色·全局回合去重，包含普通出击和战斗牌；同回合多次触发另计次数。',
            '', '## 曲线与统计限制','',
            '- `turn-states.csv` 给出各回合仍在进行的样本量及平均生命、行动力、能量、环合值；没有概率预测。',
            '- `selfplay-card-metrics.csv` 是主逐卡数据，仅含最终同模型同构筑镜像，双方按实际先后手统计。`card-metrics.csv` 是全矩阵诊断，规则 AI 与交叉互搏不能混入主统计；非镜像行仅统计学习方。包含抽到/见到/使用的分子分母、来源、平均次数和回合数。',
            '- `training-win-rates.csv` 保留各次更新按对手类型和先后手的胜率序列，不能混入冻结评估。',
            '- `generated-card-metrics.csv` 单列复制与衍生牌的获得、使用和胜负关联，不混入主牌组抽到率。',
            '- `mechanism-metrics.csv` 包含按角色/先后手的机会、次数和同动作生命变化。跨动作/跨回合的净因果收益尚未归因，不将相邻效果全算作某机制贡献。',
            '- `sample-replay.json` 是已记录对局的公开可播放录像；私有种子及隐藏牌库不在其中。',
            '- 全部测试可执行不等于本轮样本量足以判定强度；未使用的卡牌/机制不判弱。',
            '- 本轮不提供可靠预测胜率、长期收敛结论或机制的因果胜率贡献。']
    for name,rows in [('selfplay-card-metrics.csv',chosen_card_rows),('selfplay-mechanism-metrics.csv',chosen_mechanisms),('card-metrics.csv',card_rows),('generated-card-metrics.csv',generated_rows),('mechanism-metrics.csv',mechanism_rows),('turn-states.csv',curve_rows)]:
        with (root/name).open('w',encoding='utf-8-sig',newline='') as stream:
            if rows:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    training_rows=[]
    for key in ('starter','weave-rush'):
        for stage in ('base','adapted'):
            for text in (root/key/stage/'metrics.jsonl').read_text(encoding='utf-8').splitlines():
                metric=json.loads(text)
                for opponent,positions in {**metric.get('by_opponent_position',{}),**{'阵容池/'+k:v for k,v in metric.get('by_opponent_pool_position',{}).items()}}.items():
                    for position,row in positions.items():
                        training_rows.append({'preset':key,'stage':stage,'update':metric['update'],
                            'seconds':metric['seconds'],'opponent':opponent,'position':position,**row})
    with (root/'training-win-rates.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        if training_rows:
            writer=csv.DictWriter(stream,fieldnames=list(training_rows[0]));writer.writeheader();writer.writerows(training_rows)
    (root/'candidate-decisions.json').write_text(json.dumps(candidate_decisions,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    opening_path=root/'opening-analysis/opening-summary.json'
    opening=read(opening_path) if opening_path.is_file() else None
    opening_required=any(row['model'].get('schema')=='resident_public_v3' for row in summary_rows)
    from .opening_report import complete_opening_matrix
    opening_complete=bool(opening and opening.get('complete') and complete_opening_matrix(opening.get('cells',[])))
    if opening:
        lines+=['','## 起手与首回合对照','',f"状态：{'完整' if opening_complete else '未完成'}；单元 {len(opening.get('cells',[]))}/8。",'',
                '同一组模型、实际构筑及种子，仅改变学习方是否使用学习换牌；对手策略保持一致。详细留牌与首动作见 opening-analysis/opening-report.md。','',
                '| 学习方 | 对手阵容 | 起手策略 | 位置 | 胜/完成 |', '| --- | --- | --- | --- | --- |']
        for cell in opening['cells']:
            for position,counts in cell['summary']['by_position'].items():
                lines.append(f"| {cell['learner']} | {cell['opponent']} | {cell['mode']} | {position} | {counts['wins']}/{counts['games']} |")
    elif opening_required:
        lines+=['','## 起手与首回合对照','',f"状态：{'完整' if opening_complete else '未完成'}；单元 {len(opening.get('cells',[]))}/8。",'','缺少新模型的学习换牌／全留对照，报告未完成。']
    result={'config':config,'evaluations':summary_rows,'construction':construction,
            'card_metrics_source':{'kind':'final-identical-model-build-mirror','perspectives':['a','b'],
                                   'tests':sorted({r['test'] for r in chosen_card_rows}),
                                   'missing_presets':sorted(set(final_variants)-{r['preset'] for r in chosen_card_rows})},
            'opening_analysis':{'required':opening_required,'complete':opening_complete},
            'prediction_probability':{'available':False,'reason':'No independently calibrated probability model'},
            'serving_selection':serving_selection['choices'] if serving_selection else None,'report_complete':(not opening_required or opening_complete) and all(x['complete'] for x in summary_rows) and set(final_variants)=={r['preset'] for r in chosen_card_rows},
            'limitations':['finite sampled candidate search; not exhaustive optimum','effect events support inspection, not isolated causal contribution']}
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
