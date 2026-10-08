"""Official Python rule rollouts. Workers import NumPy, never Torch or CUDA."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from random import Random
import time
import numpy as np
from .league_schema import CARD_IDS,SEATS,deck
from .league_observation import encode_league
from .league_policy import LeagueModel
from app.modules.card_game.engine.duel_v2 import new_game,apply_action,observe,legal_actions,acting_side
from app.modules.card_game.engine.duel_v2.state import card_instance

_CACHE={}

def model(directory,key):
    p=Path(directory)/(key+'.json');cache=(str(p),p.stat().st_mtime_ns)
    if cache not in _CACHE:
        # Each worker needs at most three current/frozen policies.
        if len(_CACHE)>12:_CACHE.clear()
        _CACHE[cache]=LeagueModel(directory,key)
    return _CACHE[cache]

def decision(s,side):
    v=observe(s,side,include_previews=False)
    a=[entry['action'] for entry in v['legal_actions'] if entry['action']['type']!='concede']
    if not a:raise ValueError('No legal non-concede action')
    return a,encode_league(v,a)

def clean(s):
    s['events']=[]
    s.pop('_public_board',None)
    return s

def determinize(s,viewer,seed, *, encoder=None, seats=None, cards=None, ignore_summons=False, generated_ids=()):
    """Constrain enemy draws by public zones, never its real unexposed card IDs."""
    from app.modules.card_game.content.duel_v2 import CARDS
    encoder = encoder or encode_league
    seats = SEATS if seats is None else seats
    cards = CARD_IDS if cards is None else cards
    pending = s.get('pending_choice')
    choosing = s['phase'] == 'choice'
    if choosing:
        if (not pending or pending.get('side') != viewer
                or pending.get('kind') not in ('inspect_top', 'discard', 'enemy_hand', 'hand_redraw')):
            raise ValueError('Unsupported choice root')
        if s.get('operation') and s['operation']['side'] != viewer:
            raise ValueError('Choice operation belongs to another observer')
    elif s['phase'] != 'playing' or pending or s.get('operation'):
        raise ValueError('Unsupported root')
    out=deepcopy(s);rng=Random(seed);foe='b' if viewer=='a' else 'a';team=out['sides'][foe]
    # Only generated NF01 exists in this scope; its generation count is public.
    known=Counter(c['card_id'] for c in team['discard'] if not c.get('derived') and c['card_id'] not in generated_ids)
    revealed=set(team.get('revealed_ids',[]))
    # These identities are privately visible to the chooser, not public to the foe.
    if choosing and pending['kind'] == 'enemy_hand':
        revealed.update(c['instance_id'] for c in pending['cards'])
    keep=[c for c in team['hand'] if c['instance_id'] in revealed]
    for c in keep:
        if not c.get('derived') and c['card_id'] not in generated_ids:known[c['card_id']]+=1
    hand_n=len(team['hand'])-len(keep);deck_n=len(team['deck'])
    family=int(team.get('nanali_family',0))
    hidden_family=family-sum(c['card_id']=='NF01' for c in team['discard']+keep)
    if not 0<=hidden_family<=hand_n:raise ValueError('Inconsistent public generated-family count')
    extras=len(team['discard'])+len(team['hand'])+deck_n-family-32
    hidden_gen=0
    if generated_ids:
        public_gen=sum(1 for c in list(team['discard'])+keep if c.get('derived') or c['card_id'] in generated_ids)
        hidden_gen=extras-public_gen
        if hidden_gen<0:raise ValueError('Generated cards exceed the public surplus')
        extras=0
    if extras<0 or extras and 'xiaozhi' not in team['characters']:raise ValueError('Unsupported generated-card accounting')
    unknown=[]
    for cid in team['characters']:
        if ignore_summons and team['characters'][cid].get('summoned'):
            continue
        if cid not in seats:raise ValueError('Unencoded hero')
        kit=[c for c in cards if CARDS[c]['character_id']==cid and not CARDS[c].get('derived') and c not in generated_ids]
        counts={c:known[c] for c in kit};caps={c:2+(extras if c=='Q05' else 0) for c in kit}
        n=8+(extras if cid=='xiaozhi' else 0)-sum(counts.values())
        if n<0 or any(counts[c]>caps[c] for c in kit):raise ValueError('Illegal public original-deck counts')
        # Generated copies are all Q05, so a sample must include at least extras Q05 total.
        forced=max(0,extras-counts.get('Q05',0)) if cid=='xiaozhi' else 0
        unknown+=['Q05']*forced
        if forced:counts['Q05']+=forced;n-=forced
        options=[c for c in kit for _ in range(caps[c]-counts[c])];rng.shuffle(options)
        if n<0 or n>len(options):raise ValueError('Impossible public allocation')
        unknown+=options[:n]
    if len(unknown)!=hand_n+deck_n-hidden_family-hidden_gen:raise ValueError('Unknown-pool size mismatch')
    rng.shuffle(unknown)
    own=sorted(c['card_id'] for c in out['sides'][viewer]['deck']);rng.shuffle(own)
    # Opaque hidden IDs and next serials must not carry original shuffle information.
    visible=[]
    for side,t in out['sides'].items():
        visible += [t,*t['characters'].values(),*t['discard'],*(t['hand'] if side==viewer else keep)]
    if choosing:
        visible += out['pending_choice']['cards']
        resolving = (out.get('operation') or {}).get('card')
        if resolving:
            visible.append(resolving)
    out['next_entity']=max([int(x['entity_id'][1:]) for x in visible if str(x.get('entity_id','')).startswith('e')]+[0])
    out['next_instance']=max([int(x['instance_id'].rsplit('-',1)[1]) for x in visible if x.get('instance_id')]+[0])
    if hidden_gen:
        pool=unknown+[generated_ids[i % len(generated_ids)] for i in range(hidden_gen)]+['NF01']*hidden_family
        rng.shuffle(pool)
        hidden_hand=pool[:hand_n]
        deck_cards=pool[hand_n:]
    else:
        hidden_hand=unknown[:hand_n-hidden_family]+['NF01']*hidden_family
        rng.shuffle(hidden_hand)
        deck_cards=unknown[hand_n-hidden_family:]
    team['hand']=keep+[card_instance(out,c,foe) for c in hidden_hand]
    team['deck']=[card_instance(out,c,foe) for c in deck_cards]
    out['sides'][viewer]['deck']=[card_instance(out,c,viewer) for c in own]
    h=team['characters'].get('xiaozhi')
    for c in team['hand']:
        c.pop('jingu_mark',None)
        if h and h.get('hp',0)>0 and h.get('awakened'):
            c['jingu_mark']={'delta':rng.choice((-1,0,1)),'source_entity_id':h['entity_id']}
    out['rng']=rng.randrange(1,2**30)
    clean(out)
    # Verify the learner sees precisely the same encoded decision in every particle.
    before=observe(s,viewer,include_previews=False)
    actions=[e['action'] for e in before['legal_actions'] if e['action']['type']!='concede']
    x,c=encoder(before,actions);after=observe(out,viewer,include_previews=False)
    xx,cc=encoder(after,actions)
    if not np.array_equal(x,xx) or not np.array_equal(c,cc):raise ValueError('Particle changed learner information')
    return out

def play(job):
    """Complete ordinary game or one root/candidate/particle rollout."""
    if job.get('tactical_search') and job.get('collect'):
        raise ValueError('Searched actions cannot be recorded as on-policy PPO samples')
    deadline=job['deadline'];rng=np.random.default_rng(job['seed']);viewer=job.get('viewer','a')
    models={s:model(*p) for s,p in job['policies'].items() if p is not None}
    if 'root' in job:
        s=determinize(job['root'],viewer,job['seed']);start_turn=s['turn'];own_decisions=1
        s=clean(apply_action(s,viewer,job['action']))
    else:
        s=new_game(seed=job['seed'],first_side=job.get('first','a'),decks=job['decks'])
        clean(s);start_turn=s['turn'];own_decisions=0
    trajectory=[];roots=[];sample=job.get('sample',False);steps=0
    for steps in range(job.get('max_actions',400)):
        if s['phase']=='finished':break
        if time.time()>=deadline:return {'complete':False,'reason':'deadline','steps':steps}
        side=acting_side(s)
        actions,(x,c)=decision(s,side)
        if side in models:
            scores,value=models[side].scores_value(x,c);prob=np.exp((scores-scores.max())/job.get('temperature',1.0));prob/=prob.sum()
            ix=int(rng.choice(len(actions),p=prob)) if sample else int(scores.argmax())
        else:
            from app.modules.card_game.engine.duel_v2 import choose_action
            selected=choose_action(s,side);ix=actions.index(selected);value=0.;prob=np.ones(len(actions))/len(actions)
        if job.get('tactical_search') and s['phase']=='playing':
            from .league_search import prove_turn
            proof=prove_turn(s,side,depth=2,node_budget=96,deadline=deadline)
            if proof['paths']:
                ix=min(proof['paths'],key=lambda i:proof['depths'][i])
        if job.get('collect') and side==viewer:
            trajectory.append((x,c,ix,float(np.log(max(1e-20,prob[ix]))),value))
            if s['phase']=='playing' and len(actions)>1:
                critical=any(0<h['hp']<=2 or h['hp']>0 and h['harmony']>=2 for t in s['sides'].values() for h in t['characters'].values())
                priority=float(rng.random())+float(critical)*.5
                roots.append((priority,deepcopy(s)))
                roots=sorted(roots,key=lambda p:-p[0])[:2]
        s=clean(apply_action(s,side,actions[ix]));own_decisions+=int(side==viewer)
    done=s['phase']=='finished';reward=0 if s.get('winner') not in ('a','b') else 10 if s['winner']==viewer else -10
    return {'complete':done,'winner':s.get('winner'),'reward':reward if done else None,'steps':steps+1,'remaining_turns':s['turn']-start_turn,'own_decisions':own_decisions,'trajectory':trajectory if done else [],'roots':[r for _,r in roots] if done else []}
