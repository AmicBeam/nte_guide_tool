"""Three independently timed genesis beads; displayed numbers are fixed stacks."""
import json
import unittest
from copy import deepcopy

from tests import test_duel_v2_harmony_rework as fixtures
from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.engine.duel_v2 import apply_action, observe
from app.modules.card_game.engine.duel_v2.flow import operation, end_turn, begin_turn
from app.modules.card_game.engine.duel_v2.genesis import add_layers, on_sortie, public_beads, ensure_beads, tick_layers
from app.modules.card_game.engine.duel_v2.state import card_instance, damage, hero
from app.modules.card_game.engine.duel_v2.combat import harmony_marks
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board


class GenesisBeadTest(unittest.TestCase):
    def game(self, jiuyuan=False):
        return fixtures.HarmonyReworkTest().game(('zero','nanali','jiuyuan' if jiuyuan else 'hathor','iloy'))

    def test_one_creation_is_one_two_stack_bead_with_jiuyuan(self):
        state=self.game(True);c=operation(state,'a','zero')
        add_layers(c)
        beads=state['sides']['a']['genesis_layers']
        self.assertEqual(len(beads),1)
        self.assertEqual(beads[0]['stacks'],2)
        on_sortie(c)
        self.assertEqual(state['sides']['b']['hp'],98)
        self.assertEqual([e['amount'] for e in state['events'] if e['type']=='genesis'],[1,1])

    def test_three_beads_give_three_boxes_and_six_independent_hits(self):
        state=self.game(True);c=operation(state,'a','zero')
        for _ in range(4):add_layers(c)
        self.assertEqual(len(state['sides']['a']['genesis_layers']),3)
        markers=observe(state,'a',include_previews=False)['sides']['a']['effect_markers']
        markers=[m for m in markers if m['id'].startswith('genesis_bead_')]
        self.assertEqual([m['name'] for m in markers],['【创生】×2']*3)
        self.assertEqual(len({m['id'] for m in markers}),3)
        on_sortie(c)
        self.assertEqual(state['sides']['b']['hp'],94)
        on_sortie(operation(state,'a','nanali'))
        self.assertEqual(state['sides']['b']['hp'],94)

    def test_base_beads_are_one_stack_and_expire_independently(self):
        state=self.game();c=operation(state,'a','zero')
        add_layers(c);tick_layers(state,'a');add_layers(c)
        beads=state['sides']['a']['genesis_layers']
        self.assertEqual([(b['stacks'],b['left']) for b in beads],[(1,1),(1,2)])
        tick_layers(state,'a')
        self.assertEqual([(b['stacks'],b['left']) for b in state['sides']['a']['genesis_layers']],[(1,1)])

    def test_iloy_exception_is_one_additional_two_stack_bead(self):
        state=self.game(True);c=operation(state,'a','zero')
        for _ in range(3):add_layers(c)
        add_layers(c,uncapped=True)
        beads=state['sides']['a']['genesis_layers']
        self.assertEqual(len(beads),4)
        self.assertEqual(sum(not b['uncapped'] for b in beads),3)
        self.assertEqual([b['stacks'] for b in beads],[2,2,2,2])

    def test_iloy_queues_each_trigger_and_extra_beads_do_not_recurse(self):
        from app.modules.card_game.engine.duel_v2.combat import resolve_genesis
        from app.modules.card_game.engine.duel_v2.flow import finish_operation
        from app.modules.card_game.engine.duel_v2.entities import clone_state
        state=self.game(True)
        for _ in range(2):
            resolve_genesis(operation(state,'a','zero'));finish_operation(state)
        self.assertEqual(len(state['sides']['a']['extra_genesis_queue']),2)
        self.assertEqual(state['sides']['a']['extra_genesis_pending'],2)
        damage(state,'a','iloy',100)
        state=clone_state(json.loads(json.dumps(state)))
        begin_turn(state,'a')
        beads=state['sides']['a']['genesis_layers']
        self.assertEqual(sum(b['uncapped'] for b in beads),2)
        self.assertTrue(all(b['stacks']==2 for b in beads))
        self.assertFalse(state['sides']['a'].get('extra_genesis_pending'))
        self.assertFalse(state['sides']['a'].get('extra_genesis_queue'))

    def test_iloy_pending_entries_check_each_original_actor(self):
        from app.modules.card_game.engine.duel_v2.combat import resolve_genesis
        from app.modules.card_game.engine.duel_v2.flow import finish_operation
        state=self.game(True)
        for actor in ('zero','nanali'):
            resolve_genesis(operation(state,'a',actor));finish_operation(state)
        damage(state,'a','zero',100)
        begin_turn(state,'a')
        self.assertEqual(sum(b['uncapped'] for b in state['sides']['a']['genesis_layers']),1)

    def test_jiuyuan_death_does_not_rewrite_existing_bead_stacks(self):
        state=self.game(True);c=operation(state,'a','zero');add_layers(c)
        damage(state,'a','jiuyuan',100)
        add_layers(c)
        self.assertEqual([b['stacks'] for b in state['sides']['a']['genesis_layers']],[2,1])

    def test_old_doubled_rows_are_read_only_grouped_into_three_beads(self):
        state=self.game(True)
        legacy=[dict(id=i+1,left=2,used_turn=-1,uncapped=False,source_entity_id=hero(state,'a','zero').entity_id) for i in range(6)]
        legacy+= [dict(id=i+7,left=2,used_turn=-1,uncapped=True,source_entity_id=hero(state,'a','zero').entity_id) for i in range(2)]
        state['sides']['a']['genesis_layers']=legacy
        before=deepcopy(state)
        public=public_beads(state,'a')
        self.assertEqual([(b['stacks'],b['uncapped']) for b in public],[(2,False)]*3+[(2,True)])
        self.assertEqual(state,before)
        ensure_beads(state,'a')
        self.assertEqual(state['sides']['a']['genesis_layers'],public)
        restored=json.loads(json.dumps(state))
        self.assertEqual(public_beads(restored,'a'),public)

    def test_end_turn_frame_clear_round_trips_in_public_replay(self):
        state=self.game()
        card=card_instance(state,'Z03','a');state['sides']['a']['hand']=[card]
        state=apply_action(state,'a',dict(type='play_card',card_id=card['instance_id']))
        self.assertTrue(harmony_marks(state,'a')['payer'])
        opening=capture_public_board(state);cursor=state['event_seq']
        state=apply_action(state,'a',dict(type='end_turn'))
        events=[e for e in state['events'] if e['seq']>cursor]
        after=capture_public_board(state)
        self.assertEqual(replay_public_board(opening,events),after)
        self.assertTrue(all(not c['harmony_source'] for c in after['sides']['a']['characters']))
        self.assertEqual(hero(state,'a','zero')['harmony'],3)

    def test_response_in_opponent_turn_does_not_create_next_turn_source(self):
        state=self.game();end_turn(state,'a')
        state['sides']['a']['front']='zero'
        state['sides']['a']['hand']=[card_instance(state,'N02','a')]
        state=apply_action(state,'b',dict(type='attack',character_id='zero'))
        self.assertEqual(hero(state,'a','nanali')['harmony'],1)
        self.assertIsNone(state['sides']['a']['last_front'])
        self.assertFalse(harmony_marks(state,'a')['payer'])
        begin_turn(state,'a')
        self.assertFalse(harmony_marks(state,'a')['available'])

    def test_family_growth_uses_payment_without_extra_resource(self):
        self.assertNotIn('环合值',CARDS['NF01']['description'])
        state=self.game()
        card=card_instance(state,'NF01','a');state['sides']['a']['hand']=[card]
        state=apply_action(state,'a',dict(type='play_card',card_id=card['instance_id']))
        self.assertEqual(hero(state,'a','nanali')['harmony'],1)
        state=self.game();state['turn']=5
        card=card_instance(state,'NF01','a');state['sides']['a']['hand']=[card]
        ap=state['sides']['a']['ap']
        state=apply_action(state,'a',dict(type='play_card',card_id=card['instance_id']))
        self.assertEqual(hero(state,'a','nanali')['harmony'],0)
        self.assertEqual(state['sides']['a']['ap'],ap)
