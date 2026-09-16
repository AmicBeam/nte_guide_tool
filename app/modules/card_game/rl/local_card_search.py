"""Paired local card search with exact one/two-copy distance and fixed roles."""
from collections import Counter
from hashlib import sha256
from random import Random

from .capability import assert_public_encoder_deck, public_kit_cards
from .candidate_selection import allocation_key, metrics, select_with_baseline


def replacement_delta(parent, candidate):
    before,after=Counter(parent['card_ids']),Counter(candidate['card_ids'])
    return {'removed':dict(before-after),'added':dict(after-before),
            'copies_changed':sum((before-after).values())}


def local_neighbors(parent, *, seed=0, two_limit=128):
    """All distance-one builds plus a deterministic sample of distance-two builds."""
    parent=assert_public_encoder_deck(parent)
    if type(two_limit) is not int or not 0<=two_limit<=1024:raise ValueError('two_limit must be 0..1024')
    cards=[c for owner in parent['character_ids'] for c in public_kit_cards(owner)]
    counts=Counter(parent['card_ids']);initial=tuple(counts[c] for c in cards)
    def moves(values):
        for owner in range(4):
            for remove in range(owner*8,owner*8+8):
                if not values[remove]:continue
                for add in range(owner*8,owner*8+8):
                    if add==remove or values[add]>=2:continue
                    new=list(values);new[remove]-=1;new[add]+=1
                    yield tuple(new)
    singles=set(moves(initial));doubles=set()
    if two_limit:
        for values in singles:
            for changed in moves(values):
                if sum(max(0,a-b) for a,b in zip(initial,changed))==2:doubles.add(changed)
    chosen=sorted(singles)
    chosen+=Random(seed).sample(sorted(doubles),min(two_limit,len(doubles)))
    result=[]
    for values in chosen:
        identity='local-'+sha256(repr((parent['character_ids'],values)).encode()).hexdigest()[:12]
        build={**parent,'id':identity,'name':'局部配牌候选','card_ids':[c for c,n in zip(cards,values) for _ in range(n)]}
        build=assert_public_encoder_deck(build)
        result.append({'id':identity,'build':build,'parent_id':parent['id'],**replacement_delta(parent,build)})
    return result


def refine_locally(baseline, evaluate, *, initial=None, rounds=4, two_limit=128,
                   screen_pairs=8, final_pairs=128, finalists=8, seed_base=420000000,
                   batch_size=16, stop_check=lambda:None, schedule_factory=None):
    """Repeated paired search, followed by fresh comparison to original/current builds.

    All samples are selection data, never held-out validation. Interrupted output
    is explicitly incomplete and keeps the last completely evaluated incumbent.
    """
    if type(seed_base) is not int or seed_base<0 or seed_base%2 or seed_base+16000000>=2**31:
        raise ValueError('seed_base must be bounded, nonnegative and even')
    if not (1<=rounds<=16 and 1<=screen_pairs<=128 and 1<=final_pairs<=1024 and 1<=finalists<=32 and 1<=batch_size<=32):
        raise ValueError('Invalid local-search budget')
    if schedule_factory is None:
        from .candidate_opponents import build_opponent_schedule
        schedule_factory=build_opponent_schedule
    baseline=assert_public_encoder_deck(baseline)
    current=assert_public_encoder_deck(initial or baseline)
    if current['character_ids']!=baseline['character_ids']:raise ValueError('Cannot change characters')
    stages=[];changes=[];completed=0
    def candidate(build):
        return {'id':'original' if allocation_key(build)==allocation_key(baseline) else build['id'],'build':build}
    def schedule(pairs, offset):
        result=schedule_factory(pairs,seed=seed_base+offset,random_fraction=0,rule_fraction=0,model_ids=('learner','peer-0'))
        for i,row in enumerate(result):row['seed']=seed_base+offset+2*i
        return result
    def compare(candidates,plan,pairs):
        rows=[]
        for start in range(0,len(candidates),batch_size):
            stop_check();batch=candidates[start:start+batch_size]
            outcomes=evaluate([c['build'] for c in batch],plan)
            if len(outcomes)!=len(batch):raise ValueError('Missing evaluation rows')
            for c,o in zip(batch,outcomes):
                complete=bool(o['complete']) and all(o['groups'].get(k,{}).get(pos,{}).get('games')==pairs and not o['groups'][k][pos].get('truncated',0) for k in ('starter','weave-rush') for pos in ('first','second'))
                rows.append({**c,**o,'complete':complete,'metrics':metrics(o['groups']) if complete else None})
        return rows
    def unique(items):
        seen=set();out=[]
        for c in items:
            key=allocation_key(c['build'])
            if key not in seen:seen.add(key);out.append(c)
        return out
    try:
        for turn in range(rounds):
            stop_check();parent=current
            # Revisit original as well as incumbent to avoid losing local branches.
            neighbors=local_neighbors(parent,seed=seed_base+turn,two_limit=two_limit)
            if allocation_key(parent)!=allocation_key(baseline):neighbors+=local_neighbors(baseline,seed=seed_base+turn+100,two_limit=two_limit)
            candidates=unique([candidate(baseline),candidate(current),*neighbors])
            plan=schedule(screen_pairs,turn*1000000)
            rows=compare(candidates,plan,screen_pairs)
            stages.append({'stage':f'local-{turn+1}-screen','pairs_per_lineup':screen_pairs,'schedule':plan,'results':rows})
            if not all(r['complete'] for r in rows):break
            ranked=sorted(rows,key=lambda r:(-r['metrics']['utility'],r['id']))
            finalists_list=unique([candidate(baseline),candidate(current),*[{'id':r['id'],'build':r['build']} for r in ranked[:finalists]]])
            plan=schedule(final_pairs,turn*1000000+500000)
            rows=compare(finalists_list,plan,final_pairs)
            stages.append({'stage':f'local-{turn+1}-final','pairs_per_lineup':final_pairs,'schedule':plan,'results':rows})
            if not all(r['complete'] for r in rows):break
            chosen=select_with_baseline(rows)
            incumbent=next(r for r in rows if allocation_key(r['build'])==allocation_key(current))
            # Keep the incumbent on equal scores; original remains the baseline floor.
            if incumbent['metrics']['utility']>next(r for r in rows if r['id']=='original')['metrics']['utility'] and incumbent['metrics']['utility']>=chosen['metrics']['utility']:
                chosen=incumbent
            current=chosen['build'];completed+=1
            changes.append({'round':turn+1,'parent_id':parent['id'],'selected_id':chosen['id'],**replacement_delta(parent,current),
                            'parent_score':incumbent['metrics']['utility'],'selected_score':chosen['metrics']['utility'],
                            'reason':'Highest equal-weight four-scenario score on common final schedule'})
    except TimeoutError:
        pass
    return {'complete':completed==rounds,'completed_rounds':completed,'requested_rounds':rounds,
            'selection':candidate(current)['id'],'build':current,'stages':stages,'changes':changes,
            'scoring':'equal_four_scenario_win_rate_v1','holdout_used_for_selection':False,
            'reason':'Four-scenario paired local search; one-copy exhaustive, two-copy sampled'}
