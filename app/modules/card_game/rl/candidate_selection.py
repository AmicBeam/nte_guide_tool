"""Bounded, paired candidate tournament. No ML imports or training on import."""
from collections import Counter
import json
from pathlib import Path

PRESETS = ('starter', 'weave-rush')
PROFILES = {
    'smoke': {'pairs_per_lineup': [1, 2, 2], 'survivors': [3, 2, 1]},
    'comprehensive': {'pairs_per_lineup': [8, 32, 128], 'survivors': [4, 2, 1]},
}


def allocation_key(deck):
    # Ordering within the shuffled card list is not a distinct allocation.
    return (tuple(deck['character_ids']), tuple(sorted(Counter(deck['card_ids']).items())))


def candidate_pool(root, key, *, limit=8):
    """Baseline plus deduplicated last/history candidates; invalid evidence excluded."""
    from .capability import fixed_team_serving_deck
    root=Path(root)
    baseline=json.loads((root/'original-deck.json').read_text(encoding='utf-8'))
    choices=[('original', baseline)]
    serving=Path(__file__).resolve().parents[1]/'engine/ai/models'/f'{key}.json'
    if serving.is_file():
        deck=json.loads(serving.read_text(encoding='utf-8')).get('serving_build')
        if deck is not None:choices.append(('serving',deck))
    seeds=root/'seed-candidates.json'
    if seeds.is_file():
        items=json.loads(seeds.read_text(encoding='utf-8'))
        if not isinstance(items,list) or len(items)>2:raise ValueError('At most two seed builds per preset')
        choices.extend((f'seed-{i}',b) for i,b in enumerate(items))
    last=root/'cards/candidate.json'
    if last.is_file():choices.append(('last',json.loads(last.read_text(encoding='utf-8'))))
    history=root/'cards/candidate-history.jsonl'
    best={}
    if history.is_file():
        for line in history.read_text(encoding='utf-8').splitlines():
            item=json.loads(line)
            if not item.get('valid') or item.get('truncated'):continue
            identifier=item['candidate_id']
            if identifier not in best or item['score']>best[identifier]['score']:best[identifier]=item
        for item in sorted(best.values(),key=lambda r:(-r['score'],r['candidate_id'])):
            path=(root/'cards'/item['candidate_file']).resolve()
            if not path.is_relative_to((root/'cards').resolve()):raise ValueError('Candidate path outside run')
            choices.append((item['candidate_id'],json.loads(path.read_text(encoding='utf-8'))))
            # Load only enough ranked candidates to deduplicate, not all raw training records.
            if len(choices)>=limit*4:break
    result=[];seen=set()
    for identifier,deck in choices:
        deck=fixed_team_serving_deck(deck,key)
        signature=allocation_key(deck)
        if signature in seen:continue
        seen.add(signature);result.append({'id':identifier,'build':deck})
        if len(result)==limit:break
    return result


def metrics(groups):
    rates=[];seats=[]
    for key in PRESETS:
        group=groups[key]
        if set(group)!= {'first','second'}:raise ValueError('Missing seat')
        total=sum(g['games'] for g in group.values())
        if not total or any(g['games']<=0 or g.get('truncated',0) for g in group.values()):
            raise ValueError('Incomplete candidate evidence')
        rates.append(sum(g['wins'] for g in group.values())/total)
        seats.extend(g['wins']/g['games'] for g in group.values())
    macro=sum(rates)/len(rates)
    return {'macro':macro,'worst_lineup':min(rates),'worst_seat':min(seats),
            'utility':sum(seats)/len(seats)}


def select_with_baseline(rows):
    """Choose the equal-weight four-scenario win rate; ties retain baseline."""
    baseline=next(r for r in rows if r['id']=='original')
    if not all(r['complete'] for r in rows):return baseline
    # Recompute from counts so saved results cannot retain an old scoring rule.
    for row in rows:
        row['metrics']=metrics(row['groups'])
    candidates=[r for r in rows if r['id']!='original' and
                r['metrics']['utility']>baseline['metrics']['utility']+1e-12]
    return max(candidates,key=lambda r:(r['metrics']['utility'],r['id'])) if candidates else baseline


def run_tournament(candidates, evaluate, *, profile='comprehensive', seed_base=400000000,
                   stop_check=lambda:None, schedule_factory=None):
    """evaluate(builds, schedule) -> per-candidate scalar groups/complete records.

    Stages are selection only. Holdout must use disjoint seeds, is not input here.
    Interrupted/incomplete tournaments never promote candidates.
    """
    if schedule_factory is None:
        from .candidate_opponents import build_opponent_schedule
        schedule_factory=build_opponent_schedule
    plan=PROFILES[profile]
    current=list(candidates)
    baseline=next(c for c in current if c['id']=='original')
    stages=[];chosen=baseline;complete=False
    try:
        for stage,(pairs,keep) in enumerate(zip(plan['pairs_per_lineup'],plan['survivors'])):
            stop_check()
            schedule=schedule_factory(pairs,seed=seed_base+stage*1000000,random_fraction=0,rule_fraction=0,
                                      model_ids=('learner','peer-0'))
            for i,pair in enumerate(schedule):pair['seed']=seed_base+stage*1000000+2*i
            outcomes=evaluate([c['build'] for c in current],schedule)
            if len(outcomes)!=len(current):raise ValueError('Candidate outcome count mismatch')
            rows=[]
            for c,outcome in zip(current,outcomes):
                row={**outcome,'id':c['id'],'build':c['build']}
                # Enforce actual per-seat denominators, not just evaluator's completed flag.
                row['complete']=bool(row['complete']) and all(
                    row['groups'].get(k,{}).get(pos,{}).get('games')==pairs and
                    not row['groups'][k][pos].get('truncated',0)
                    for k in PRESETS for pos in ('first','second'))
                row['metrics']=metrics(row['groups']) if row['complete'] else None
                rows.append(row)
            stages.append({'stage':stage,'pairs_per_lineup':pairs,'schedule':schedule,'results':rows})
            if not all(r['complete'] for r in rows):break
            if stage==2:
                chosen=select_with_baseline(rows);complete=True;break
            ranked=sorted((r for r in rows if r['id']!='original'),
                          key=lambda r:(-r['metrics']['utility'],r['id']))
            current=[baseline,*[{'id':r['id'],'build':r['build']} for r in ranked[:keep-1]]]
    except TimeoutError:
        pass
    return {'complete':complete,'selection':chosen['id'] if complete else 'original',
            'build':chosen['build'] if complete else baseline['build'],'stages':stages,
            'profile':profile,'plan':plan,'holdout_used_for_selection':False,
            'scoring':'equal_four_scenario_win_rate_v1',
            'reason':'Equal-weight win rate across both lineups and both seats; baseline wins ties' if complete
                     else 'Insufficient or interrupted evaluation; baseline retained'}


def evaluate_tensor_candidates(evaluator, builds, schedule, *, stop_check, max_actions=400):
    import torch
    from .gpu_duel.catalog import encode_public_deck_row
    rows=torch.tensor([encode_public_deck_row(b) for b in builds],device=evaluator.device,dtype=torch.int32)
    result=evaluator.evaluate(rows,schedule=schedule,max_actions=max_actions,stop_check=stop_check)
    return [{'complete':bool(result['valid'][i]),'groups':{
        k:{pos:{field:int(value[i]) for field,value in fields.items()} for pos,fields in group.items()}
        for k,group in result['by_lineup'].items()}} for i in range(len(builds))]
