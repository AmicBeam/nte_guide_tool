"""Witch battle frames and divination layered onto the enclosing sortie."""
import json
import unittest
from tests.test_duel_v2_heiyu import game, witch, top, play, confirm
from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.engine.duel_v2 import apply_action, observe
from app.modules.card_game.engine.duel_v2.entities import clone_state
from app.modules.card_game.engine.duel_v2.state import hero,card_instance


class HeiyuLatestTest(unittest.TestCase):
    def test_witch_hand_frame_matches_attack_and_exiting_restores_face(self):
        s=witch(game());card=card_instance(s,'P01','a');s['sides']['a']['hand']=[card]
        face=observe(s,'a',include_previews=False)['sides']['a']['hand'][0]
        self.assertEqual(face['attack'],2)
        s=apply_action(s,'a',dict(type='play_card',card_id=card['instance_id']))
        self.assertEqual(s['sides']['b']['hp'],95)  # 3 panel + 2 card; no duplicate +1.
        self.assertEqual(hero(s,'a','heiyu')['harmony'],1)
        s=apply_action(s,'a',dict(type='exit_ultimate',character_id='heiyu'))
        card=card_instance(s,'P01','a');s['sides']['a']['hand']=[card]
        self.assertEqual(observe(s,'a',include_previews=False)['sides']['a']['hand'][0]['attack'],1)

    def entry(self,bottom,magical=False,actor='heiyu'):
        s=witch(game()) if magical else game()
        top(s,[bottom,'Z01','Z02','Z03'])
        payer=hero(s,'a','jiuyuan');payer['harmony']=3;payer['attribute']='暗'
        s['sides']['a'].update(last_front='jiuyuan',last_actor_entity_id=payer.entity_id,last_actor_turn=s['turn'])
        # Soul + dark creates darkstar and starts divination before the hit.
        s=apply_action(s,'a',dict(type='attack',character_id=actor))
        self.assertEqual(s['phase'],'choice')
        self.assertEqual(s['sides']['b']['hp'],100)
        return s

    def test_p01_layers_on_original_attack_in_both_forms_and_survives_json(self):
        for magical,amount in ((False,3),(True,5)):
            s=self.entry('P01',magical)
            s=confirm(clone_state(json.loads(json.dumps(s))),'P01')
            hits=[e for e in s['events'] if e['type']=='attack']
            self.assertEqual(len(hits),1)
            self.assertEqual(hits[0]['attack'],amount)
            self.assertEqual(s['sides']['b']['hp'],100-amount)
            self.assertEqual(hero(s,'a','heiyu')['harmony'],1 if magical else 3)
            self.assertEqual(len([c for c in s['sides']['a']['discard'] if c['card_id']=='P01']),1)

    def test_p04_waits_until_original_attack_then_releases_in_both_forms(self):
        for magical,amounts in ((False,[2,2]),(True,[3,4])):
            s=confirm(self.entry('P04',magical),'P04')
            hits=[e for e in s['events'] if e['type']=='attack']
            self.assertEqual([e['attack'] for e in hits],amounts)
            release=next(e for e in s['events'] if e['type']=='play' and e.get('release_only'))
            self.assertLess(hits[0]['seq'],release['seq'])
            self.assertEqual(hero(s,'a','heiyu')['harmony'],1 if magical else 3)
            self.assertEqual(len([c for c in s['sides']['a']['discard'] if c['card_id']=='P04']),1)
