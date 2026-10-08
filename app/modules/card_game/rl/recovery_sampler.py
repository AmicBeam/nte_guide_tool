"""Public-count sampler with original/generated identity separation.

No real unexposed enemy card identity, metadata or ordering constrains a draw.
Generation-type allocation is an approximate prior from public card counts,
not a reconstructed action-history posterior. Old sampler stays reproducible.
"""
from collections import Counter
from copy import deepcopy
from random import Random
import numpy as np
from .cross_lineup import CARD_IDS,SEATS,encode
from .league_rollout import clean
from ..content.duel_v2 import CARDS
from ..engine.duel_v2 import observe
from ..engine.duel_v2.state import card_instance


def generated_kind(card):
    cid=card['card_id']
    if cid=='NF01':return 'family'
    if cid=='A03' and card.get('ephemeral_turn') is not None:return 'A03_ephemeral'
    if card.get('derived') or CARDS[cid].get('derived'):return cid
    return None


def _allocation(team,public,surplus,rng,hand_n,hidden_family,temporary_allowed):
    # A03 timing metadata is absent from public_card: do not use its real
    # original/generated flag, even on a revealed/discarded enemy card.
    originals=Counter(c['card_id'] for c in public if c['card_id']!='NF01'
                      and not c.get('derived') and not CARDS[c['card_id']].get('derived'))
    special=Counter(c['card_id'] for c in public if c['card_id']!='NF01'
                    and (c.get('derived') or CARDS[c['card_id']].get('derived')))
    discarded=Counter(c['card_id'] for c in team['discard']);possibilities=[]
    def needed(known,copies,allocated):
        return hidden_family+sum(allocated.values())+sum(max(0,n-known[cid]+
            int(cid=='C02' and discarded['C02']>0)) for cid,n in copies.items())
    def feasible(known,copies,allocated,limits,remaining):
        free=sum(min(limits.get(cid+'_copy',0),max(0,known[cid]-
                 int(cid=='C02' and discarded['C02']>0)-n)) for cid,n in copies.items())
        return needed(known,copies,allocated)+max(0,remaining-free)<=hand_n
    for temporary in range(min(originals['A03'],discarded['AF01'])+1):
        known=originals.copy();known['A03']-=temporary
        if known['A03']>2:continue
        a_known=sum(n for c,n in known.items() if CARDS[c]['character_id']=='anhunqu')
        if a_known>8:continue
        remaining=surplus-sum(special.values())-temporary;copies={};limits={}
        for hero,cid in (('xiaozhi','Q05'),('canhong','C02')):
            if hero not in team['characters']:continue
            count=sum(n for c,n in known.items() if CARDS[c]['character_id']==hero)
            copies[cid]=max(0,known[cid]-2,count-8);remaining-=copies[cid]
            cap=surplus if hero=='xiaozhi' else discarded[cid]
            limits[cid+'_copy']=cap-copies[cid]
        if 'anhunqu' in team['characters']:
            limits['AF01']=discarded['A04']-special['AF01']
            limits['A03_ephemeral']=discarded['AF01']-temporary if temporary_allowed else 0
        if (remaining>=0 and all(v>=0 for v in limits.values()) and remaining<=sum(limits.values())
                and feasible(known,copies,Counter(),limits,remaining)):
            possibilities.append((known,copies,limits,remaining,temporary))
    if not possibilities:raise ValueError('No public-consistent generation allocation')
    known,copies,limits,remaining,temporary=rng.choice(possibilities);allocated=Counter()
    for left in range(remaining,0,-1):
        choices=[]
        for kind,cap in limits.items():
            if not cap:continue
            trial_c=copies.copy();trial_a=allocated.copy();trial_l=limits.copy();trial_l[kind]-=1
            if kind.endswith('_copy'):trial_c[kind[:-5]]+=1
            else:trial_a[kind]+=1
            if feasible(known,trial_c,trial_a,trial_l,left-1):choices.append(kind)
        if not choices:raise ValueError('No hand-consistent generated allocation')
        kind=rng.choice(choices);limits[kind]-=1
        if kind.endswith('_copy'):copies[kind[:-5]]+=1
        else:allocated[kind]+=1
    return known,copies,allocated,temporary


def determinize(state,viewer,seed):
    pending=state.get('pending_choice');choosing=state['phase']=='choice'
    if choosing:
        if (not pending or pending.get('side')!=viewer or pending.get('kind') not in
                ('inspect_top','discard','enemy_hand','hand_redraw')):
            raise ValueError('Unsupported recovery choice root')
        if state.get('operation') and state['operation']['side']!=viewer:
            raise ValueError('Choice operation belongs to another observer')
    elif state['phase']!='playing' or pending or state.get('operation'):
        raise ValueError('Unsupported recovery root')
    out=deepcopy(state);rng=Random(seed);foe='b' if viewer=='a' else 'a';team=out['sides'][foe]
    revealed=set(team.get('revealed_ids',[]))
    if choosing and pending['kind']=='enemy_hand':revealed.update(c['instance_id'] for c in pending['cards'])
    keep=[c for c in team['hand'] if c['instance_id'] in revealed]
    public=list(team['discard'])+keep
    hand_n=len(team['hand'])-len(keep);deck_n=len(team['deck'])
    hidden_family=int(team.get('nanali_family',0))-sum(c['card_id']=='NF01' for c in public)
    if not 0<=hidden_family<=hand_n+deck_n:raise ValueError('Inconsistent public family count')
    surplus=len(team['discard'])+len(team['hand'])+deck_n-int(team.get('nanali_family',0))-32
    if surplus<0:raise ValueError('Negative public generated surplus')
    original,copies,allocated,temporary=_allocation(team,public,surplus,rng,hand_n,hidden_family,
        out['active_side']==foe)
    a03=[c for c in public if c['card_id']=='A03'];rng.shuffle(a03)
    for i,c in enumerate(a03):
        if i<temporary:c['ephemeral_turn']=out['turn']
        else:c.pop('ephemeral_turn',None)
    pool=[]
    for hero,h in team['characters'].items():
        if h.get('summoned'):continue
        if hero not in SEATS:raise ValueError('Unencoded recovery hero')
        kit=[cid for cid in CARD_IDS if CARDS[cid]['character_id']==hero and not CARDS[cid].get('derived')]
        bonus=sum(n for cid,n in copies.items() if CARDS[cid]['character_id']==hero)
        counts={cid:original[cid] for cid in kit};caps={cid:2+copies.get(cid,0) for cid in kit}
        n=8+bonus-sum(counts.values())
        if n<0 or any(counts[c]>caps[c] for c in kit):raise ValueError('Illegal public original-deck counts')
        for cid,number in copies.items():
            if cid not in kit:continue
            forced=max(0,number-counts[cid]+int(cid=='C02' and any(c['card_id']=='C02' for c in team['discard'])))
            pool.extend((cid,False) for _ in range(forced))
            counts[cid]+=forced;n-=forced
        options=[cid for cid in kit for _ in range(caps[cid]-counts[cid])];rng.shuffle(options)
        if not 0<=n<=len(options):raise ValueError('Impossible original allocation')
        pool.extend((cid,False) for cid in options[:n])
    pool.extend(('NF01',False) for _ in range(hidden_family))
    for kind,n in allocated.items():pool.extend(('A03',True) if kind=='A03_ephemeral' else (kind,False) for _ in range(n))
    if len(pool)!=hand_n+deck_n:raise ValueError('Unknown-pool size mismatch')
    rng.shuffle(pool)
    # The supported lineups have no hand-to-deck return: generated cards
    # enter hand, and playing/expiry moves them to discard, never to deck.
    mandatory=[c for c in pool if c[0] in ('NF01','AF01') or c[1]]
    pool=[c for c in pool if c[0] not in ('NF01','AF01') and not c[1]]
    for cid,n in copies.items():
        count=max(0,n-original[cid]+int(cid=='C02' and any(c['card_id']=='C02' for c in team['discard'])))
        for _ in range(count):
            match=next(i for i,c in enumerate(pool) if c[0]==cid);mandatory.append(pool.pop(match))
    if len(mandatory)>hand_n:raise ValueError('Generated cards exceed hidden hand')
    hidden_hand=mandatory+pool[:hand_n-len(mandatory)];rng.shuffle(hidden_hand)
    hidden_deck=pool[hand_n-len(mandatory):]
    # Own deck multiset is known from construction/private history; its order
    # and instance serials are not. Keep timing metadata of known own copies.
    fields=('ephemeral_turn','copy','expires_turn','jingu_mark')
    own=[(c['card_id'],{k:deepcopy(c[k]) for k in fields if k in c}) for c in out['sides'][viewer]['deck']]
    import json
    own.sort(key=lambda c:json.dumps(c,sort_keys=True));rng.shuffle(own)
    visible=[]
    for side,t in out['sides'].items():visible.extend([t,*t['characters'].values(),*t['discard'],*(t['hand'] if side==viewer else keep)])
    if choosing:
        visible.extend(out['pending_choice']['cards'])
        if (out.get('operation') or {}).get('card'):visible.append(out['operation']['card'])
    out['next_entity']=max([int(c['entity_id'][1:]) for c in visible if str(c.get('entity_id','')).startswith('e')]+[0])
    out['next_instance']=max([int(c['instance_id'].rsplit('-',1)[1]) for c in visible if c.get('instance_id')]+[0])
    def instantiate(item,side):
        cid,temporary=item;c=card_instance(out,cid,side)
        if temporary:c['ephemeral_turn']=out['turn']
        return c
    team['hand']=keep+[instantiate(c,foe) for c in hidden_hand]
    team['deck']=[instantiate(c,foe) for c in hidden_deck]
    out['sides'][viewer]['deck']=[]
    for cid,metadata in own:
        c=card_instance(out,cid,viewer);c.update(metadata);out['sides'][viewer]['deck'].append(c)
    h=team['characters'].get('xiaozhi')
    for c in team['hand']:
        if c in keep:continue
        c.pop('jingu_mark',None)
        if h and h.get('hp',0)>0 and h.get('awakened'):
            c['jingu_mark']=dict(delta=rng.choice((-1,0,1)),source_entity_id=h['entity_id'])
    out['rng']=rng.randrange(1,2**30);clean(out)
    before=observe(state,viewer,include_previews=False);actions=[e['action'] for e in before['legal_actions'] if e['action']['type']!='concede']
    x,c=encode(before,actions);after=observe(out,viewer,include_previews=False);xx,cc=encode(after,actions)
    if not np.array_equal(x,xx) or not np.array_equal(c,cc):raise ValueError('Particle changed learner information')
    return out
