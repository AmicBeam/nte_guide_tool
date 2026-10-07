"""User tabletop opening resource; one grant, no source or revival bypass."""
import json
import unittest
from copy import deepcopy
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe
from app.modules.card_game.engine.duel_v2.entities import hydrate_entities
from app.modules.card_game.engine.duel_v2.state import hero, damage, card_instance, front_debuff, damage_limit
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.content.duel_v2.characters.common import revive_character


class ZhenhongInitialHarmonyTest(unittest.TestCase):
    def game(self, first='a', skip=True):
        deck=next(d for d in STARTER_DECKS if d['id']=='zhenhong')
        return new_game(seed=911,first_side=first,skip_mulligan=skip,escalation=False,decks={'a':deck,'b':deck})

    def test_both_positions_start_with_one_public_personal_harmony(self):
        for first in ('a','b'):
            for skip in (False,True):
                with self.subTest(first=first,skip=skip):
                    s=self.game(first,skip)
                    for side in ('a','b'):
                        self.assertEqual(hero(s,side,'zhenhong')['harmony'],1)
                        self.assertTrue(all(h['harmony']==0 for cid,h in s['sides'][side]['characters'].items() if cid!='zhenhong'))
                        self.assertIsNone(s['sides'][side]['front'])
                        self.assertIsNone(s['sides'][side]['last_front'])
                    for viewer in ('a','b'):
                        v=observe(s,viewer,include_previews=False)
                        for side in ('a','b'):
                            self.assertEqual(next(h for h in v['sides'][side]['characters'] if h['id']=='zhenhong')['harmony'],1)

    def test_json_and_revival_do_not_regrant_spent_opening_harmony(self):
        s=self.game();hero(s,'a','zhenhong')['harmony']=0
        s=hydrate_entities(json.loads(json.dumps(s)))
        self.assertEqual(hero(s,'a','zhenhong')['harmony'],0)
        s['sides']['a']['hand']=[]
        damage(s,'a','zhenhong',99)
        revive_character(EffectContext(s,'a','iloy',{'side':'a'}),'a','zhenhong')
        self.assertGreater(hero(s,'a','zhenhong')['hp'],0)
        self.assertEqual(hero(s,'a','zhenhong')['harmony'],0)

    def test_battle_then_iloy_uses_two_ap_and_triggers_surplus(self):
        s=self.game();s['sides']['a'].update(ap=2,hand=[])
        s['sides']['b']['front']='zero';hero(s,'b','zero').update(hp=50,max_hp=50)
        front_debuff(s,'b')['delay']=dict(left=2,by='a',tick_side='a')
        card=card_instance(s,'R02','a');s['sides']['a']['hand'].append(card)
        s=apply_action(s,'a',dict(type='play_card',card_id=card['instance_id']))
        self.assertEqual(hero(s,'a','zhenhong')['harmony'],2)
        self.assertTrue(s['sides']['a']['normal_attack_available'])
        s=apply_action(s,'a',dict(type='attack',character_id='iloy'))
        self.assertEqual(s['sides']['a']['ap'],0)
        self.assertEqual(hero(s,'a','zhenhong')['harmony'],0)
        self.assertEqual(hero(s,'a','zhenhong')['surplus_passive_triggers'],1)
        self.assertEqual(damage_limit(s,'a','zhenhong'),1)

    def test_starting_resource_does_not_allow_benched_payer(self):
        s=self.game();s['sides']['a'].update(ap=1,hand=[])
        # Even an externally full bench is not the most recent sortie source.
        hero(s,'a','zhenhong')['harmony']=2
        s=apply_action(s,'a',dict(type='attack',character_id='iloy'))
        self.assertEqual(hero(s,'a','zhenhong')['harmony'],2)
        self.assertFalse(any(e['type']=='harmony' and '创生' in e.get('text','') for e in s['events']))
