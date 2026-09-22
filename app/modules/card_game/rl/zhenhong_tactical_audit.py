"""Read-only, replayable Zhenhong behavior audit. Never supplies training labels."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from ..engine.duel_v2 import new_game, apply_action, legal_actions
from ..engine.duel_v2.simulation import simulate_action
from .league_rollout import clean


def flags(state,side):
    team=state['sides'][side];characters=team['characters'];foe='b' if side=='a' else 'a'
    delay=(state['sides'][foe].get('front_debuff') or {}).get('delay') or {}
    red=characters['zhenhong']
    return dict(delay=delay.get('by')==side and delay.get('left',0)>0,
                fangs=characters.get('yi',{}).get('beast_fangs',0)>0,
                alive=red['hp']>0,passive=red.get('surplus_passive_triggers',0))


def audit_file(path,*,counterfactuals=False):
    path=Path(path)
    with gzip.open(path,'rt',encoding='utf-8') as f:data=json.load(f)
    if not data['complete']:raise ValueError('Incomplete evaluation cannot pass a tactical audit')
    state=new_game(seed=data['seed'],first_side=data['first'],decks=data['decks'])
    owners=[s for s in ('a','b') if 'zhenhong' in state['sides'][s]['characters']]
    counts={s:Counter() for s in owners};first_passive={s:None for s in owners};cases=[]
    for step,record in enumerate(data['actions']):
        actor=record['side'];action=record['action'];before={s:flags(state,s) for s in owners}
        after=apply_action(state,actor,action)
        for side in owners:
            b=before[side];a=flags(after,side);c=counts[side]
            c['ever_delay']|=int(a['delay']);c['ever_fangs']|=int(a['fangs'])
            c['ever_ready_alive']|=int(a['delay'] and a['fangs'] and a['alive'])
            delta=a['passive']-b['passive'];c['passive_triggers']+=delta
            c['starts_after_ready']+=int(delta>0 and b['delay'] and b['fangs'] and b['alive'])
            if delta and first_passive[side] is None:first_passive[side]=step
            if first_passive[side] is not None and step>first_passive[side]:
                c['later_zhenhong_attacks']+=sum(e.get('type')=='attack' and e.get('actor')==side+':zhenhong' for e in after.get('events',[]))
        if counterfactuals and actor in owners and state['phase']=='playing':
            c=counts[actor];b=before[actor];winning=[];activating=[]
            for alternative in legal_actions(state,actor):
                if alternative['type']=='concede':continue
                branch=simulate_action(state,actor,alternative)
                if branch.get('winner')==actor:winning.append(alternative)
                if b['delay'] and b['fangs'] and b['alive'] and flags(branch,actor)['passive']>b['passive']:
                    activating.append(alternative)
            if winning:
                c['immediate_win_opportunities']+=1
                c['immediate_wins_taken']+=int(after.get('winner')==actor)
            if activating:
                c['ready_activation_opportunities']+=1
                c['ready_activations_taken']+=int(flags(after,actor)['passive']>b['passive'])
            if (winning and after.get('winner')!=actor) or (activating and flags(after,actor)['passive']==b['passive']):
                cases.append(dict(step=step,turn=state['turn'],side=actor,actual=action,
                                  immediate_wins=winning,possible_activations=activating,
                                  judgment='Activation alternatives are opportunities, not automatically superior sacrifices.'))
        state=clean(after)
        if record.get('state_sha256'):
            digest=hashlib.sha256(json.dumps(state,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            if digest!=record['state_sha256']:raise ValueError('Replay state mismatch')
    if state['phase']!='finished' or state.get('winner')!=data['winner']:raise ValueError('Replay outcome mismatch')
    for side in owners:
        if counts[side]['passive_triggers']!=data['passive_triggers'][side]:raise ValueError('Passive counter mismatch')
    return dict(id=data['id'],variant=data.get('variant'),kind=data.get('kind'),row=data.get('row'),col=data.get('col'),
                winner=data['winner'],first=data['first'],counts={s:dict(c) for s,c in counts.items()},cases=cases,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),replayed=True,counterfactuals=counterfactuals)
