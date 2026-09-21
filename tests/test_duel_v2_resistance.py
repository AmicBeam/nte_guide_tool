"""User rules: pre-battle bonuses precede set attack; Q02 resistance lasts this turn."""
import json
import unittest
from copy import deepcopy
from tests import test_duel_v2_xiaozhi_haiyue as fixtures
from app.modules.card_game.engine.duel_v2 import apply_action, observe
from app.modules.card_game.engine.duel_v2.state import hero, damage
from app.modules.card_game.engine.duel_v2.modifiers import resistance_values, add_effect
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.engine.duel_v2.attack_modifiers import attack_result


class ResistanceTest(unittest.TestCase):
    def setup_target(self, character=False):
        helper=fixtures.NewTestCharacters();s=helper.game()
        if character:
            s['sides']['b']['front']='zero';hero(s,'b','zero').update(hp=100,max_hp=100)
        else:
            s['sides']['b']['hp']=100
        return helper.play(s,'Q02')

    def test_unused_resistance_expires_for_player_and_character_on_turn_end(self):
        for cid in ('player','zero'):
            with self.subTest(target=cid):
                s=self.setup_target(cid=='zero')
                target=s['sides']['b'] if cid=='player' else hero(s,'b',cid)
                self.assertEqual(resistance_values(target)['光'],-2)
                s=json.loads(json.dumps(s));s=apply_action(s,'a',{'type':'end_turn'})
                target=s['sides']['b'] if cid=='player' else hero(s,'b',cid)
                self.assertNotIn('光',resistance_values(target))
                hp=target['hp'];damage(s,'b',cid,1,source='a:zero')
                self.assertEqual(target['hp'],hp-1)

    def test_nonmatching_zero_and_immune_damage_do_not_consume_but_shield_does(self):
        s=self.setup_target(True);target=hero(s,'b','zero')
        before=deepcopy(s);observe(s,'a');self.assertEqual(s,before)
        damage(s,'b','zero',0,source='a:zero')
        damage(s,'b','zero',1,source='a:nanali')
        self.assertIn('光',resistance_values(target))
        add_effect(target,'test.immune','rule.damage_immune',True,op='allow')
        damage(s,'b','zero',1,source='a:zero')
        self.assertIn('光',resistance_values(target))
        target['effects']=[e for e in target['effects'] if e['definition_id']!='test.immune']
        target['shield']=9;hp=target['hp']
        damage(s,'b','zero',1,source='a:zero')
        self.assertEqual((target['shield'],target['hp']),(6,hp))
        self.assertNotIn('光',resistance_values(target))

    def test_pre_battle_and_first_hit_bonuses_are_both_overwritten(self):
        s=fixtures.NewTestCharacters().game()
        op={'side':'a','first_bonus':3,'sortie_attack_bonus':4,'card_set_attack':4}
        c=EffectContext(s,'a','xiaozhi',op)
        result=attack_result(c,{},0,s['sides']['b'],'b','player')
        self.assertEqual(result['value'],4)
        del op['card_set_attack']
        result=attack_result(c,{},0,s['sides']['b'],'b','player')
        self.assertEqual(result['value'],10)
