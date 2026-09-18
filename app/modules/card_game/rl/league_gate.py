"""Held-out named tactical decisions. No fixture is consumed by training."""
from copy import deepcopy
import numpy as np
from .league_schema import deck
from .league_rollout import decision
from app.modules.card_game.engine.duel_v2 import new_game,apply_action
from app.modules.card_game.engine.duel_v2.state import hero,card_instance

def cases():
    output=[]
    for side in ('a','b'):
        foe='b' if side=='a' else 'a'
        s=new_game(seed=922000001,first_side=side,skip_mulligan=True,decks={side:deck('quick-rush'),foe:deck('starter')})
        s['sides'][side]['ap']=1;s['sides'][foe]['hp']=7;s['sides'][foe]['shield']=0
        hero(s,side,'xiaozhi')['jingu']=3
        s['sides'][side]['hand']=[card_instance(s,'Q01',side)]
        output.append(('quick-rush','q01_boost_lethal-'+side,s,side,'lethal'))
        s=deepcopy(s);s['sides'][foe]['hp']=3;s['sides'][foe]['front']='zero';s['sides'][side]['hand']=[card_instance(s,'U03',side)]
        output.append(('quick-rush','u03_bypass_lethal-'+side,s,side,'lethal'))
        s=new_game(seed=922000002,first_side=side,skip_mulligan=True,decks={side:deck('starter'),foe:deck('weave-rush')})
        s['sides'][side]['ap']=1;s['sides'][side]['front']='zero';hero(s,side,'zero').update(hp=1,harmony=2)
        hero(s,side,'nanali').update(hp=6,max_hp=6,base_attack=3,attack=3,growth=1)
        s['sides'][foe]['front']='baicang';hero(s,foe,'baicang').update(hp=2)
        s['sides'][side]['hand']=[card_instance(s,c,side) for c in ('Z05','N02')]
        output.append(('starter','safe_genesis_family-'+side,s,side,'family'))
    from .strategy_quality import tactical_cases
    for case in tactical_cases():
        if case['name'].startswith('buff_before'):output.append(('weave-rush',case['name'],case['state'],case['side'],'M02'))
        if case['name'].startswith('revive_then'):output.append(('starter',case['name'],case['state'],case['side'],'N06'))
    return output

def evaluate(model):
    rows=[]
    for key,name,s,side,goal in cases():
        if key!=model.deck_id:continue
        actions,(x,c)=decision(s,side);logits=model.scores(x,c);p=np.exp(logits-logits.max());p/=p.sum();best=int(logits.argmax())
        expected=[];named=[]
        for a in actions:
            cid=next((card['card_id'] for card in s['sides'][side]['hand'] if card['instance_id']==a.get('card_id')),None)
            named.append({'type':a['type'],'actor':a.get('character_id'),'card':cid,'option':a.get('option_id')})
            if goal in ('M02','N06'):good=cid==goal
            else:
                t=apply_action(s,side,a)
                good=t.get('winner')==side if goal=='lethal' else (hero(t,side,'zero')['hp']>0 and hero(t,'b' if side=='a' else 'a','baicang')['hp']==0 and any(card['card_id']=='NF01' for card in t['sides'][side]['hand']))
            expected.append(good)
        rows.append({'case':name,'passed':expected[best],'selected':named[best],'expected_probability':sum(float(prob) for prob,ok in zip(p,expected) if ok),'actions':[dict(a,probability=float(prob),expected=ok) for a,prob,ok in zip(named,p,expected)]})
    return {'passed':bool(rows) and all(r['passed'] for r in rows),'cases':rows,'used_in_training':False,'calibrated_win_probability':False}
