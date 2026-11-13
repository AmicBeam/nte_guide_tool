"""Departure no longer captures harmony; the most recent actual action does."""
import json
import unittest
from app.modules.card_game.engine.duel_v2 import new_game,apply_action,observe
from app.modules.card_game.engine.duel_v2.flow import begin_turn
from app.modules.card_game.engine.duel_v2.state import hero,move_out,card_instance,damage
from app.modules.card_game.engine.duel_v2.combat import harmony_marks
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board


class HarmonyResidueTest(unittest.TestCase):
    def game(self,harmony=1):
        s=new_game(seed=9,first_side='a',skip_mulligan=True,escalation=False)
        for t in s['sides'].values():t['hand']=[]
        s['sides']['a'].update(front='nanali',ap=10)
        hero(s,'a','nanali')['harmony']=harmony
        return s

    def family(self,s):
        c=card_instance(s,'NF01','a');s['sides']['a']['hand']=[c]
        return apply_action(s,'a',dict(type='play_card',card_id=c['instance_id']))

    def test_bench_card_can_establish_full_source_without_departure_evidence(self):
        s=self.game();begin_turn(s,'a');self.assertIsNone(s['sides']['a']['last_front'])
        s=self.family(s)
        self.assertEqual(hero(s,'a','nanali')['harmony'],3)
        self.assertEqual(s['sides']['a']['last_front'],'nanali')
        self.assertTrue(harmony_marks(s,'a')['available'])
        for board in (observe(s,'a'),capture_public_board(s)):
            self.assertTrue(board['sides']['a']['harmony_available'])
        s=apply_action(json.loads(json.dumps(s)),'a',dict(type='attack',character_id='zero'))
        self.assertEqual(hero(s,'a','nanali')['harmony'],0)
        self.assertEqual(s['sides']['a']['last_front'],'zero')

    def test_returning_full_without_acting_does_not_create_source(self):
        for harmony in (1,3):
            s=self.game(harmony);move_out(s,'a','nanali')
            self.assertIsNone(s['sides']['a']['last_front'])
            self.assertFalse(harmony_marks(s,'a')['available'])

    def test_actual_source_survives_return_and_json(self):
        s=self.family(self.game());move_out(s,'a','nanali');begin_turn(s,'a')
        self.assertEqual(s['sides']['a']['last_front'],'nanali')
        self.assertTrue(harmony_marks(s,'a')['available'])
        s=apply_action(json.loads(json.dumps(s)),'a',dict(type='attack',character_id='zero'))
        self.assertEqual(hero(s,'a','nanali')['harmony'],0)

    def test_thirteenth_turn_does_not_lower_harmony_or_invent_source(self):
        s=self.game(2);s['escalation_enabled']=True;s['turn']=12;begin_turn(s,'a')
        self.assertEqual(hero(s,'a','nanali')['harmony'],2)
        self.assertIsNone(s['sides']['a']['last_front'])
        self.assertFalse(harmony_marks(s,'a')['available'])

    def test_source_death_keeps_full_resource_but_expires_identity(self):
        s=self.family(self.game());damage(s,'a','nanali',100)
        self.assertEqual(hero(s,'a','nanali')['harmony'],3)
        self.assertIsNone(s['sides']['a']['last_front'])
        self.assertFalse(harmony_marks(s,'a')['available'])

    def test_legacy_departure_record_is_not_an_action(self):
        s=self.game(3);s['harmony_residue_version']=1;s['sides']['a'].update(front=None,last_front='nanali')
        self.assertFalse(harmony_marks(s,'a')['available'])
        begin_turn(s,'a');self.assertEqual(s['harmony_residue_version'],2)
        self.assertIsNone(s['sides']['a']['last_front'])

    def test_entity_mismatch_never_uses_same_template_as_source(self):
        s=self.family(self.game());s['sides']['a']['last_actor_entity_id']='unrelated-entity'
        self.assertFalse(harmony_marks(s,'a')['available'])

    def test_old_compiled_contract_rejects_missing_current_hooks(self):
        from app.modules.card_game.rl.rule_ir.completeness import validate_compiled_deck,CompletenessError
        with self.assertRaises(CompletenessError):
            validate_compiled_deck('starter')
