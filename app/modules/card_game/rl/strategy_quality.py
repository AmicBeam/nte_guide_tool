"""Reproducible tactical diagnostics and validation-based checkpoint selection.

No intermediate rewards or automatic training. Values are expected discounted
returns, not calibrated win probabilities.
"""
from copy import deepcopy
import math


class ValidationTracker:
    def __init__(self, patience=6, min_delta=.01):
        self.patience=patience;self.min_delta=min_delta;self.best=None;self.stale=0;self.rows=[]
    def observe(self, update, score, *, complete=True):
        if not complete or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('Only complete finite win rates may select checkpoints')
        improved=self.best is None or score>self.best['score']+1e-12
        meaningful=self.best is None or score>self.best['score']+self.min_delta
        self.stale=0 if meaningful else self.stale+1
        row={'update':update,'score':score,'stale_checks':self.stale}
        self.rows.append(row)
        if improved:self.best=dict(row)
        return improved
    @property
    def plateau(self):return self.stale>=self.patience


def tactical_cases():
    from app.modules.card_game.engine.duel_v2 import new_game
    from app.modules.card_game.engine.duel_v2.state import card_instance
    from .capability import preset_opponent_deck
    result=[]
    for side in ('a','b'):
        foe='b' if side=='a' else 'a'
        for name,deck,cards in [('buff_before_attack','weave-rush',('M01','M02')),
                               ('ally_then_nanali','starter',('N04',)),
                               ('revive_then_attack','starter',('N06',))]:
            s=new_game(seed=730001,first_side=side,skip_mulligan=True,
                       decks={side:preset_opponent_deck(deck),foe:preset_opponent_deck('weave-rush')})
            for t in s['sides'].values():t['hand']=[];t['shield']=0
            s['sides'][side]['ap']=2
            s['sides'][side]['hand']=[card_instance(s,c,side) for c in cards]
            if name=='revive_then_attack':s['sides'][side]['characters']['nanali'].update(hp=0,down_turns=2)
            result.append({'name':name+'-'+side,'side':side,'state':s})
    return result


def diagnose_network(net, *, backend=None):
    """Report every legal action probability/value and optional exact parity."""
    import torch
    import numpy as np
    from app.modules.card_game.engine.duel_v2 import observe,legal_actions,apply_action
    from .opening_observation import encode_opening,opening_observe
    from .rule_ir.compiled_oracle import rebuild,map_python_action,pack_python_row,mechan_from_row
    from .rule_ir.pack_row import bind_gpu_rows
    from .gpu_duel.state import empty_state
    rows=[];device=next(net.parameters()).device
    for case in tactical_cases():
        s=case['state'];side=case['side'];actions=[a for a in legal_actions(s,side) if a['type']!='concede']
        values,candidates=encode_opening(observe(s,side),actions)
        with torch.no_grad():
            logits,value=net(torch.tensor(values,device=device)[None],torch.tensor(candidates,device=device)[None],torch.ones(1,len(actions),device=device,dtype=torch.bool))
        probs=torch.softmax(logits[0],0).cpu().tolist();chosen=int(logits.argmax(1)[0])
        items=[]
        for a,prob in zip(actions,probs):
            card=next((c for c in s['sides'][side]['hand'] if c['instance_id']==a.get('card_id')),None)
            after=apply_action(s,side,a)
            items.append({'action':a,'card_id':card['card_id'] if card else None,'probability':prob,
                          'opponent_hp_after':after['sides']['b' if side=='a' else 'a']['hp']})
        row={'case':case['name'],'value_discounted_return':float(value[0]),'actions':items,'selected':items[chosen],
             'parity_checked':backend is not None}
        sequences = ([['M01','M02'],['M02','M01']] if case['name'].startswith('buff_before') else
                     [['attack:iloy','N04'],['N04','attack:iloy']] if case['name'].startswith('ally_then') else
                     [['N06','attack:nanali'],['attack:iloy','N06']])
        row['branches']=[]
        for sequence in sequences:
            after=s;valid=True
            for move in sequence:
                if move.startswith('attack:'):action={'type':'attack','character_id':move.split(':')[1]}
                else:
                    card=next((c for c in after['sides'][side]['hand'] if c['card_id']==move),None)
                    if card is None:valid=False;break
                    action={'type':'play_card','card_id':card['instance_id']}
                if action not in legal_actions(after,side):valid=False;break
                after=apply_action(after,side,action)
            branch={'sequence':sequence,'legal':valid}
            if valid:
                available=[a for a in legal_actions(after,side) if a['type']!='concede']
                state_values,candidate_values=encode_opening(observe(after,side),available)
                with torch.no_grad():
                    _,future=net(torch.tensor(state_values,device=device)[None],torch.tensor(candidate_values,device=device)[None],torch.ones(1,len(available),device=device,dtype=torch.bool))
                branch.update(value_discounted_return=float(future[0]),ap=after['sides'][side]['ap'],
                    enemy_player_hp=after['sides']['b' if side=='a' else 'a']['hp'],
                    characters={cid:{k:h[k] for k in ('hp','shield','harmony','energy','down_turns')} for cid,h in after['sides'][side]['characters'].items()})
            row['branches'].append(branch)
        if backend is not None:
            packed=rebuild(backend,pack_python_row(s));state=empty_state(1,'cpu');tensor=bind_gpu_rows(state)
            tensor.copy_(torch.tensor([packed],dtype=torch.int32));obs,cands,mask=opening_observe(state)
            np.testing.assert_allclose(obs[0].numpy(),values,atol=1e-6)
            assert int(mask[0].sum())==len(actions), 'Compiled/official legal action counts differ'
            for i,a in enumerate(actions):
                index=map_python_action(packed,s,a)
                assert bool(mask[0,index]), 'Official legal action masked in training'
                np.testing.assert_allclose(cands[0,index].numpy(),candidates[i],atol=1e-6)
                got=backend.step_lists([packed],[index])[0]
                expected=rebuild(backend,pack_python_row(apply_action(s,side,a)))
                assert mechan_from_row(got)==mechan_from_row(expected), (case['name'],a)
            row['parity_actions']=len(actions)
        rows.append(row)
    return {'cases':rows,'value_is_win_probability':False,'reward_changed':False}


def reachable_curriculum(build, foe, *, count=12, seed=710000000):
    """Reachable training-only midgame states; never source them from holdout."""
    from random import Random
    from app.modules.card_game.engine.duel_v2 import new_game,legal_actions,apply_action,acting_side
    from .rule_ir.pack_row import pack_python_row
    result=[];rng=Random(seed)
    for i in range(count*3):
        s=new_game(seed=seed+i,first_side='a' if i%2==0 else 'b',skip_mulligan=True,
                   decks={'a':build,'b':foe},escalation=True)
        target=3+i%5
        for _ in range(100):
            if s['phase']=='finished':break
            if s['turn']>=target and acting_side(s)=='a' and s['phase']=='playing':
                result.append(pack_python_row(s));break
            actions=[a for a in legal_actions(s,acting_side(s)) if a['type']!='concede']
            s=apply_action(s,acting_side(s),rng.choice(actions))
        if len(result)>=count:break
    return result
