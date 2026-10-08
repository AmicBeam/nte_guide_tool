"""Official-engine transition matrix for every public kit card; no learning."""
from copy import deepcopy
import importlib.util
import tempfile
import unittest

from app.modules.card_game.engine.duel_v2 import new_game,legal_actions,apply_action,acting_side
from app.modules.card_game.engine.duel_v2.state import card_instance
from app.modules.card_game.rl.capability import PUBLIC_KIT_CARDS,preset_opponent_deck,sample_public_deck
from app.modules.card_game.rl.gpu_duel.compiled_backend import CompiledStarterBackend
from app.modules.card_game.rl.rule_ir.compiled_oracle import rebuild,map_python_action,canonical_row
from app.modules.card_game.rl.rule_ir.pack_row import pack_python_row,legal_entries


class PublicKernelMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.backend=CompiledStarterBackend(cls.tmp.name,deck_id='starter')

    @classmethod
    def tearDownClass(cls):
        cls.backend.close();cls.tmp.cleanup()

    def check(self,state,action):
        row=rebuild(self.backend,pack_python_row(state))
        legal=[a for a in legal_actions(state,acting_side(state)) if a['type']!='concede']
        mapped=[map_python_action(row,state,a) for a in legal]
        self.assertEqual(len(set(mapped)),len(legal_entries(row)), 'Compiled offers extra illegal actions')
        if importlib.util.find_spec('torch'):
            import torch
            import numpy as np
            from app.modules.card_game.engine.duel_v2 import observe
            from app.modules.card_game.rl.gpu_duel.state import empty_state
            from app.modules.card_game.rl.rule_ir.pack_row import bind_gpu_rows
            from app.modules.card_game.rl.public_observation import public_observe,encode_public
            batch=empty_state(1,'cpu')
            bind_gpu_rows(batch).copy_(torch.tensor([row],dtype=torch.int32))
            got_state,got_candidates,_=public_observe(batch)
            expected_state,expected_candidates=encode_public(observe(state,acting_side(state)),legal)
            np.testing.assert_allclose(got_state[0].numpy(),expected_state,atol=1e-7)
            np.testing.assert_allclose(got_candidates[0,mapped].numpy(),expected_candidates,atol=1e-7)
        index=map_python_action(row,state,action)
        got=self.backend.step_lists([row],[index])[0]
        expected=apply_action(state,acting_side(state),action)
        expected_row=rebuild(self.backend,pack_python_row(expected))
        self.assertEqual(canonical_row(got),canonical_row(expected_row))
        return expected

    def fixture(self,card_id):
        from app.modules.card_game.content.duel_v2 import CARDS
        deck=preset_opponent_deck('weave-rush' if card_id[0] in 'MB' else 'starter')
        state=new_game(seed=7,first_side='a',skip_mulligan=True,
                       decks={'a':deck,'b':sample_public_deck(8)})
        team=state['sides']['a'];foe=state['sides']['b']
        owner=CARDS[card_id]['character_id']
        other=next(cid for cid in team['order'] if cid!=owner)
        down=next(cid for cid in team['order'] if cid not in (owner,other))
        team.update(ap=3,hp=25,harmony_damage=True,surplus=True,front=other)
        team['characters'][owner]['hp']=max(1,team['characters'][owner]['max_hp']-2)
        team['characters'][down].update(hp=0,down_turns=2)
        enemy_down=foe['order'][0]
        foe['characters'][enemy_down].update(hp=0,down_turns=1)
        foe['front']=foe['order'][1]
        foe['front_debuff']={'burn':{'left':2,'by':'a'}}
        foe['revealed_ids']=[card['instance_id'] for card in foe['hand'][:2]]
        if card_id=='N06':team['characters'][owner].update(hp=0,down_turns=2)
        card=card_instance(state,card_id,'a');team['hand']=[card]
        team['revealed_ids']=[card['instance_id']]
        actions=[a for a in legal_actions(state,'a') if a.get('card_id')==card['instance_id']]
        self.assertTrue(actions,f'{card_id} fixture must actually exercise its effect')
        return state,actions[0]

    def test_all_public_cards_and_followup_choices(self):
        for card_id in PUBLIC_KIT_CARDS:
            with self.subTest(card=card_id):
                state,action=self.fixture(card_id)
                state=self.check(state,action)
                if state['phase']=='choice':
                    action=next(a for a in legal_actions(state,acting_side(state)) if a['type']=='choose')
                    self.check(state,action)

    def test_escalation_boundaries_and_two_ultimates(self):
        for before in (5,12,19):
            with self.subTest(turn=before):
                state=new_game(seed=8,first_side='a',skip_mulligan=True,
                               decks={'a':preset_opponent_deck('starter'),'b':sample_public_deck(3)})
                state['turn']=before
                for side in ('a','b'):
                    for index,cid in enumerate(state['sides'][side]['order']):
                        h=state['sides'][side]['characters'][cid]
                        h['energy']=index;h['harmony']=2
                # Unique lowest energy makes the exact oracle comparison deterministic.
                self.check(state,{'type':'end_turn'})
        state=new_game(seed=8,first_side='a',skip_mulligan=True,
                       decks={'a':preset_opponent_deck('starter'),'b':sample_public_deck(3)})
        state['turn']=6
        for cid in ('nanali','iloy'):state['sides']['a']['characters'][cid]['energy']=5
        state=self.check(state,{'type':'ultimate','character_id':'nanali'})
        state=self.check(state,{'type':'ultimate','character_id':'iloy'})
        self.assertFalse(any(a['type']=='ultimate' for a in legal_actions(state,'a')))

    def test_tied_escalation_selects_one_lowest_and_is_repeatable(self):
        from app.modules.card_game.rl.rule_ir.layout import OFFSETS
        from app.modules.card_game.rl.gpu_duel.catalog import SEAT_INDEX
        selected=set()
        for seed in range(32):
            state=new_game(seed=seed,first_side='a',skip_mulligan=True,
                           decks={'a':preset_opponent_deck('starter'),'b':preset_opponent_deck('starter')})
            state['turn']=5
            row=rebuild(self.backend,pack_python_row(state))
            index=map_python_action(row,state,{'type':'end_turn'})
            one=self.backend.step_lists([row],[index])[0]
            two=self.backend.step_lists([row],[index])[0]
            self.assertEqual(one,two)
            energies={cid:one[OFFSETS['ch_energy']+6+SEAT_INDEX[cid]] for cid in state['sides']['b']['order']}
            self.assertEqual(sum(energies.values()),1)
            self.assertTrue(all(v in (0,1) for v in energies.values()))
            selected.update(cid for cid,e in energies.items() if e)
        self.assertEqual(len(selected),4)

    def test_public_form_hooks_and_followup_damage(self):
        # M07 zeros the attack panel; per-battle bonuses are applied afterward.
        state=new_game(seed=8,first_side='a',skip_mulligan=True,
                       decks={'a':preset_opponent_deck('starter'),'b':preset_opponent_deck('weave-rush')})
        state['sides']['b']['front']='bohe'
        state['sides']['b']['characters']['bohe'].update(shape='M07',hp=8,max_hp=8)
        self.check(state,{'type':'attack','character_id':'nanali'})
        # J07 refunds energy for every marked target consumed by Jiuyuan's ultimate.
        state=new_game(seed=8,first_side='a',skip_mulligan=True,
                       decks={'a':preset_opponent_deck('starter'),'b':preset_opponent_deck('weave-rush')})
        state['sides']['a']['characters']['jiuyuan'].update(shape='J07',energy=5)
        for cid in ('zero','iloy'):state['sides']['b']['characters'][cid]['flags']['pact']=True
        after=self.check(state,{'type':'ultimate','character_id':'jiuyuan'})
        self.assertEqual(after['sides']['a']['characters']['jiuyuan']['energy'],2)  # Two targets, current J07 grants 1 each.
        # Bohe's ultimate blocks non-battle damage, but explicitly allows followup.
        state=new_game(seed=8,first_side='a',skip_mulligan=True,
                       decks={'a':preset_opponent_deck('starter'),'b':preset_opponent_deck('weave-rush')})
        state['sides']['b']['front']='bohe'
        state['sides']['b']['characters']['bohe']['awakened']=True
        card=card_instance(state,'N03','a');state['sides']['a']['hand']=[card]
        self.check(state,{'type':'play_card','card_id':card['instance_id']})
