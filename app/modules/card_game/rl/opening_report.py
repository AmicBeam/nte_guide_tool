"""Extra frozen opening analysis; input is private offline telemetry, not training logs."""
from collections import Counter,defaultdict
import json
from pathlib import Path


def summarize_opening(directory):
    directory=Path(directory)
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    groups={};opponent_build=None
    for position in ('first','second'):
        games=[g for g in report['games_index'] if g['position']==position and g['terminated'] and not g['error'] and not g['truncated']]
        cards=defaultdict(Counter);actions=defaultdict(Counter);count=Counter();end=[];early=defaultdict(list)
        missing=0
        for game in games:
            raw=json.loads((directory/game['raw_log'].replace('\\','/')).read_text(encoding='utf-8'))
            if opponent_build is None:opponent_build=raw.get('opponent_deck')
            elif opponent_build!=raw.get('opponent_deck'):raise ValueError('Opening comparison must use a fixed opponent build')
            telemetry=raw.get('telemetry') or {}
            opening=(telemetry.get('opening') or {}).get('a')
            if not opening or 'kept' not in opening:
                missing+=1;continue
            win=game['winner']=='a'
            count[len(opening['replaced'])]+=1
            for cid in set(opening['initial_hand']):cards[cid]['seen_games']+=1
            for field in ('kept','replaced'):
                for cid in set(opening[field]):
                    cards[cid][field+'_games']+=1;cards[cid][field+'_wins']+=win
                for cid in opening[field]:cards[cid][field+'_copies']+=1
            first=opening['first_turn_actions'][:1]
            for action in first:
                key=json.dumps(action,sort_keys=True,ensure_ascii=False)
                actions[key]['games']+=1;actions[key]['wins']+=win
            if opening.get('before_first_turn_end'):end.append(opening['before_first_turn_end'])
            for point in telemetry.get('turn_states',[]):
                if point['turn']<=6:early[point['turn']].append(point['sides']['a'])
        groups[position]={'games':len(games),'wins':sum(g['winner']=='a' for g in games),'missing_opening':missing,
            'mulligan_count_games':dict(count),'cards':dict(cards),'first_actions':dict(actions),
            'before_first_turn_end':{'games':len(end),**{f'mean_{k}':sum(p[k] for p in end)/len(end) if end else None for k in ('hp','ap','energy','harmony')}},
            'early_turns':{t:{'games':len(points),**{f'mean_{k}':sum(p[k] for p in points)/len(points) for k in ('hp','ap','energy','harmony')}} for t,points in early.items()}}
    return {'model':report['model'],'opponent_model':report['opponent_model'],'learner_build':report['learner_build'],'opponent_build':opponent_build,
            'complete':report['complete'] and all(not x['missing_opening'] for x in groups.values()),'by_position':groups}


def keep_all_policy(policy):
    def choose(view,actions):
        if view['phase']=='mulligan':return next(i for i,a in enumerate(actions) if a['type']=='mulligan' and not a.get('card_ids'))
        return policy(view,actions)
    choose.model_metadata={**policy.model_metadata,'opening_override':'keep_all'}
    return choose


def complete_opening_matrix(rows):
    from .opening_observation import OPENING_SCHEMA
    try:
        expected={(a,b,m) for a in ('starter','weave-rush') for b in ('starter','weave-rush') for m in ('learned','keep_all')}
        by_key={(r['learner'],r['opponent'],r['mode']):r for r in rows}
        if len(rows)!=8 or set(by_key)!=expected:return False
        for a,b in {(a,b) for a,b,_ in expected}:
            left=by_key[a,b,'learned'];right=by_key[a,b,'keep_all']
            x,y=left['summary'],right['summary']
            if not x['complete'] or not y['complete'] or left['seed_base']!=right['seed_base']:return False
            if x['learner_build']!=y['learner_build'] or x['opponent_build']!=y['opponent_build']:return False
            if x['opponent_model']!=y['opponent_model'] or x['opponent_model'].get('schema')!=OPENING_SCHEMA:return False
            if x['model'].get('schema')!=OPENING_SCHEMA or x['model'].get('opening_override'):return False
            if y['model'].get('opening_override')!='keep_all':return False
            if x['model']!={k:v for k,v in y['model'].items() if k!='opening_override'}:return False
            for position in ('first','second'):
                first=x['by_position'][position];second=y['by_position'][position]
                if first['games']<=0 or first['games']!=second['games'] or first['missing_opening'] or second['missing_opening']:return False
        return True
    except (KeyError,TypeError,AttributeError):
        return False


def write_opening_report(root,rows):
    root=Path(root)
    lines=['# 起手与首回合专项互搏报告','',
           '仅改变学习方换牌策略；同组模型、构筑、配对种子一致。留牌与胜负相关，不是单卡因果收益。未提供预测胜率。','',
           '| 学习方 | 对手阵容 | 起手策略 | 位置 | 胜/完成 |', '| --- | --- | --- | --- | --- |']
    for row in rows:
        for position,data in row['summary']['by_position'].items():
            lines.append(f"| {row['learner']} | {row['opponent']} | {row['mode']} | {position} | {data['wins']}/{data['games']} |")
    from app.modules.card_game.content.duel_v2 import CARDS
    for row in rows:
        if row['mode']!='learned':continue
        for position,data in row['summary']['by_position'].items():
            lines+=['',f"## {row['learner']} 对 {row['opponent']} · {position}",'',
                    '| 起手卡 | 见到局数 | 保留局数 | 换出局数 | 保留时胜/局 |',
                    '| --- | ---: | ---: | ---: | --- |']
            for cid,bucket in sorted(data['cards'].items()):
                kept=bucket.get('kept_games',0)
                lines.append(f"| {cid} {CARDS[cid]['name']} | {bucket.get('seen_games',0)} | {kept} | {bucket.get('replaced_games',0)} | {bucket.get('kept_wins',0)}/{kept} |")
            lines+=['','首动作（按局数排序，最多列 5 种）：','']
            for action,bucket in sorted(data['first_actions'].items(),key=lambda kv:-kv[1]['games'])[:5]:
                lines.append(f"- `{action}`：{bucket['games']} 局，其中获胜 {bucket['wins']} 局。")
    lines+=['','## 留牌、首动作与早期局势','',
            '`opening-summary.json` 按阵容、先后手提供每张牌的见到/保留/换出局数及胜局、换牌张数分布、首动作分布、首回合结束按钮前的生命/行动力/能量/环合及总第 1–6 回合局势。',
            '逐卡分母按局去重，携带多张另计 copies。首动作不是整个首回合的全部操作，完整顺序在 raw telemetry.opening.first_turn_actions。未完成对局不进入胜率，缺少起手记录时报告标为不完整。',
            '公开录像不包含这些私有起手记录。Windows/GPU 训练期间不保存本报告所需的逐动作日志；本报告来自额外冻结评估。']
    complete=complete_opening_matrix(rows)
    lines.insert(2,f"状态：{'完整' if complete else '未完成'}；开局对照单元 {len(rows)}/8。")
    (root/'opening-summary.json').write_text(json.dumps({'complete':complete,'cells':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (root/'opening-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return complete
