"""Detailed analytics for extra official-engine evaluations only; never used by training."""
from copy import deepcopy
import re


class GameTelemetry:
    def __init__(self, state):
        self.cards={'a':{},'b':{}}
        self.opening={side:{'initial_hand':[c['card_id'] for c in state['sides'][side]['hand']],
            'opponent_characters':list(state['sides']['b' if side=='a' else 'a']['order']),
            'position':'first' if side==state['first_side'] else 'second','first_turn_actions':[]} for side in ('a','b')}
        self.mechanisms=[]
        self.opportunities=set()
        self.turns=[]
        self.observed_turns=set()
        self.hand_ids={side:{c['instance_id'] for c in state['sides'][side]['hand']} for side in ('a','b')}
        for side in ('a','b'):
            for card in state['sides'][side]['hand']:
                self.obtain(side,card,'opening',state.get('turn',0))

    def bucket(self,side,cid):
        return self.cards[side].setdefault(cid,{'sources':{},'plays':0,'first_play_turn':None,
                                               'original_plays':0,'copy_plays':0,'derived_plays':0,'response_plays':0,'first_original_play_turn':None})

    def obtain(self,side,card,source,turn):
        cid=card.get('card_id')
        if not cid:return
        if card.get('copy'):source='copy'
        elif card.get('derived'):source='generated'
        bucket=self.bucket(side,cid)
        bucket['sources'][source]=bucket['sources'].get(source,0)+1

    def before(self,state,side,actions):
        turn=state.get('turn',0)
        if turn not in self.observed_turns and state.get('phase') not in ('mulligan','finished'):
            self.observed_turns.add(turn)
            self.turns.append({'turn':turn,'active_side':state.get('active_side'),
                'sides':{s:{'hp':state['sides'][s]['hp'],'ap':state['sides'][s]['ap'],
                    'energy':sum(c['energy'] for c in state['sides'][s]['characters'].values()),
                    'harmony':sum(c['harmony'] for c in state['sides'][s]['characters'].values())} for s in ('a','b')}})
        from app.modules.card_game.engine.duel_v2.combat import resolved_harmony_kind
        hand={c['instance_id']:c for c in state['sides'][side]['hand']}
        for action in actions:
            cid=action.get('character_id')
            card=hand.get(action.get('card_id'),{})
            battle=action['type']=='play_card' and card.get('type')=='battle'
            if battle:cid=card['character_id']
            if action['type']=='ultimate':self.opportunities.add(('ultimate',side,cid,turn))
            if (action['type']=='attack' or battle) and resolved_harmony_kind(state,side,cid):
                self.opportunities.add(('harmony',side,cid,turn))

    def after(self,before,after,action,side):
        new_events=[e for e in after.get('events',[]) if e.get('seq',0)>before.get('event_seq',0)]
        turn=before.get('turn',0)
        obtained=set()
        opening=self.opening[side]
        if action['type']=='mulligan':
            selected=set(action.get('card_ids',[]))
            opening.update(kept=[c['card_id'] for c in before['sides'][side]['hand'] if c['instance_id'] not in selected],
                           replaced=[c['card_id'] for c in before['sides'][side]['hand'] if c['instance_id'] in selected],replacement_draws=[])
        elif before.get('phase')=='playing' and before['sides'][side].get('turn_count')==1:
            hand={c['instance_id']:c['card_id'] for c in before['sides'][side]['hand']}
            opening['first_turn_actions'].append({'type':action['type'],'character_id':action.get('character_id'),
                'card_id':hand.get(action.get('card_id')),'target_id':action.get('target_id')})
            if action['type']=='end_turn':
                team=before['sides'][side]
                opening['before_first_turn_end']={'hp':team['hp'],'ap':team['ap'],'front':team.get('front'),
                    'energy':sum(h['energy'] for h in team['characters'].values()),
                    'harmony':sum(h['harmony'] for h in team['characters'].values())}

        for event in new_events:
            kind=event.get('type');owner=event.get('side')
            turn=(event.get('patch') or {}).get('turn',turn)
            card=event.get('private_card') or event.get('card') or {}
            if owner in self.cards and kind in ('draw','gain'):
                if kind=='draw':
                    source='mulligan' if action['type']=='mulligan' and turn==0 else 'draw'
                    if source=='mulligan' and owner==side:opening['replacement_draws'].append(card.get('card_id'))
                elif any(c.get('instance_id')==card.get('instance_id') for c in before['sides'][owner].get('discard',[])):
                    source='recovered'
                elif any(c.get('instance_id')==card.get('instance_id') for c in before['sides'][owner].get('deck',[])):
                    source='search'
                else:source='revealed_gain'
                self.obtain(owner,card,source,turn)
                obtained.add((owner,card.get('instance_id')))
            if owner in self.cards and kind=='play' and card.get('card_id'):
                bucket=self.bucket(owner,card['card_id']);bucket['plays']+=1
                bucket['first_play_turn']=turn if bucket['first_play_turn'] is None else bucket['first_play_turn']
                bucket['copy_plays' if card.get('copy') else 'derived_plays' if card.get('derived') else 'original_plays']+=1
                if not card.get('copy') and not card.get('derived') and bucket['first_original_play_turn'] is None:
                    bucket['first_original_play_turn']=turn
                if card.get('response') and not (action['type']=='play_card' and action.get('card_id')==card.get('instance_id')):
                    bucket['response_plays']+=1
            ultimate_start=kind=='ultimate' and 'energy' in event.get('before',{})
            ultimate_end=kind=='ultimate' and not ultimate_start
            harmony=kind=='harmony' and 'harmony' in event.get('before',{}) and 'harmony' in event.get('after',{})
            if ultimate_start or ultimate_end or harmony:
                item={'kind':'harmony' if harmony else 'ultimate' if ultimate_start else 'ultimate_end',
                      'side':owner,'actor':event.get('actor'),'target':event.get('target'),
                      'turn':turn,'seq':event['seq'],'text':event.get('text'),
                      'before':deepcopy(event.get('before',{})),'after':deepcopy(event.get('after',{}))}
                if harmony:
                    found=re.search(r'触发([^。]+)',event.get('text',''))
                    item.update(provider=event.get('target'),beneficiary=event.get('actor'),
                                harmony_type=found.group(1) if found else 'unknown',type_source='engine event text')
                # Same-action changes are associations, not isolated causal effects.
                item['same_action_player_hp_change']={s:after['sides'][s]['hp']-before['sides'][s]['hp'] for s in ('a','b')}
                item['effect_events']=[{k:deepcopy(e[k]) for k in ('seq','type','actor','target','source','amount','before','after','text') if k in e}
                    for e in new_events if e.get('seq',0)>event['seq'] and ('amount' in e or 'before' in e)]
                self.mechanisms.append(item)
        for owner in ('a','b'):
            old_ids={c['instance_id'] for c in before['sides'][owner]['hand']}
            for card in after['sides'][owner]['hand']:
                instance=card['instance_id']
                if instance not in old_ids and (owner,instance) not in obtained:
                    source='mulligan' if action['type']=='mulligan' else 'recovered' if any(
                        c.get('instance_id')==instance for c in before['sides'][owner].get('discard',[])) else 'other_gain'
                    self.obtain(owner,card,source,turn)

    def finish(self,state):
        return {'final_turn':state.get('turn',0),'cards':self.cards,'mechanisms':self.mechanisms,
                'opportunities':[dict(kind=k,side=s,character_id=c,turn=t) for k,s,c,t in sorted(self.opportunities)],
                'turn_states':self.turns,'opening':self.opening,'prediction_probability':None,
                'opportunity_unit':'unique character/global turn with a legal ultimate or ordinary or battle-card attack harmony',
                'effect_attribution':'same-action effect events may contain multiple sources; no causal win contribution claimed'}
