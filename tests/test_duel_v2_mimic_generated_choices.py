"""Generated cards keep the copying entity and their own effect continuation."""
from copy import deepcopy
import json
import unittest
from app.modules.card_game.content.duel_v2 import CARDS,validate_deck,STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game,apply_action,observe
from app.modules.card_game.engine.duel_v2.state import add_hand


def game():
    owners=['anhunqu','canhong','zaowu','adler']
    own=validate_deck(dict(id='mimic',name='mimic',character_ids=owners,
        card_ids=[cid for cid,card in CARDS.items() if card['character_id'] in owners and not card.get('derived')]))
    foe=next(d for d in STARTER_DECKS if d['id']=='collapse')
    state=new_game(seed=14,first_side='a',skip_mulligan=True,decks={'a':own,'b':foe})
    state['sides']['a']['ap']=5
    return state


def hold(state,cid):
    team=state['sides']['a']
    card=next((c for c in team['hand'] if c['card_id']==cid),None)
    if card:return card
    card=next(c for c in team['deck'] if c['card_id']==cid);team['deck'].remove(card)
    add_hand(state,'a',card)
    return card


class MimicGeneratedChoicesTest(unittest.TestCase):
    def test_copied_omen_belongs_to_anhunqu_and_works_after_passive_expiry(self):
        state=game();card=hold(state,'A01')
        state=apply_action(state,'a',dict(type='play_card',card_id=card['instance_id'],target_id='b:heiyu'))
        omen=next(c for c in state['sides']['a']['deck'] if c['card_id']=='PF01')
        self.assertEqual(omen['character_id'],'anhunqu')
        self.assertEqual(omen['original_character_id'],'heiyu')
        state['sides']['a']['characters']['anhunqu']['flags'].pop('mimic')
        state['sides']['a']['characters']['anhunqu']['awakened']=True
        omen=hold(state,'PF01')
        observe(state,'a',include_previews=False)
        state=apply_action(state,'a',dict(type='play_card',card_id=omen['instance_id']))
        self.assertTrue(state['pending_choice']['execute_divination'])
        self.assertEqual(state['pending_choice']['continuation'],'heiyu.divination')
        before=len(state['sides']['a']['hand']);pending=state['pending_choice'];cards=pending['cards']
        action=dict(type='choose_deck_order',card_ids=[c['instance_id'] for c in cards[1:]],bottom_card_ids=[cards[0]['instance_id']])
        state=apply_action(json.loads(json.dumps(state)),'a',action)
        self.assertEqual(state['phase'],'playing')
        self.assertEqual(len(state['sides']['a']['hand']),before+1)
        self.assertTrue(state['sides']['a']['characters']['anhunqu']['awakened'])

    def test_public_only_sampler_handles_native_and_copied_derived_cards(self):
        from app.modules.card_game.content.duel_v2.characters.common import generated_card
        from app.modules.card_game.engine.duel_v2.context import EffectContext
        from app.modules.card_game.rl.seven_sampler import determinize
        from app.modules.card_game.rl.seven_runtime import decision
        import numpy as np
        for cid in ('PF01','FF01'):
            state=game();context=EffectContext(state,'a','anhunqu',{})
            generated=generated_card(context,cid);add_hand(state,'a',generated)
            before=observe(state,'b',include_previews=False)
            for seed in (19,31,77):
                sampled=determinize(state,'b',seed)
                after=observe(sampled,'b',include_previews=False)
                self.assertEqual(before['sides']['a']['hand_count'],after['sides']['a']['hand_count'])
                self.assertTrue(all(c['character_id'] in sampled['sides']['a']['characters'] for c in sampled['sides']['a']['hand']+sampled['sides']['a']['deck']))
