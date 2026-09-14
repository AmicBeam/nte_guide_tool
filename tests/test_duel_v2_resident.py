"""Full-rule rush fixtures and resident training contracts."""
import importlib.util
import tempfile
import unittest
from copy import deepcopy

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, acting_side
from app.modules.card_game.engine.duel_v2.state import card_instance
from app.modules.card_game.rl.gpu_duel.compiled_backend import CompiledStarterBackend
from app.modules.card_game.rl.rule_ir.compiled_oracle import canonical_row, rebuild, step_python
from app.modules.card_game.rl.rule_ir.pack_row import pack_python_row


class RushRulesTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.backend=CompiledStarterBackend(self.tmp.name,deck_id='weave-rush')
        self.deck=next(d for d in STARTER_DECKS if d['id']=='weave-rush')
    def tearDown(self):
        self.backend.close();self.tmp.cleanup()
    def game(self):
        s=new_game(seed=13,first_side='a',skip_mulligan=True,decks={'a':self.deck,'b':self.deck})
        for side in ('a','b'):s['sides'][side]['hand']=[];s['sides'][side]['ap']=2
        return s
    def check(self,s,a):
        got,_=step_python(self.backend,s,a)
        expected=apply_action(s,acting_side(s),a)
        self.assertEqual(canonical_row(got),canonical_row(rebuild(self.backend,pack_python_row(expected))))
        return expected
    def play(self,s,cid):
        card=card_instance(s,cid,'a');s['sides']['a']['hand']=[card]
        return self.check(s,dict(type='play_card',card_id=card['instance_id']))
    def test_self_damage_form_and_heal_before_death(self):
        for cid in ('B01','B08'):
            self.play(self.game(),cid)
        s=self.game();s['sides']['a']['characters']['baicang']['hp']=1
        s['sides']['b']['front']='zero'
        s['sides']['b']['characters']['zero']['base_attack']=2
        self.play(s,'B02')
    def test_bypass_player_and_overflow_response(self):
        s=self.game();s['sides']['b']['shield']=5;self.play(s,'M03')
        s=self.game();s['sides']['a']['characters']['bohe']['base_attack']=12
        s['sides']['b']['front']='bohe'
        s['sides']['b']['characters']['bohe']['hp']=2
        s['sides']['b']['hand']=[card_instance(s,'M04','b')]
        self.check(s,dict(type='attack',character_id='bohe'))
    def test_execute_immunity_and_counter_overlay(self):
        s=self.game();s['sides']['a']['characters']['baicang']['awakened']=True
        s['sides']['b']['front']='zero';s['sides']['b']['characters']['zero']['hp']=3
        self.check(s,dict(type='attack',character_id='baicang'))
        s=self.game();s['sides']['b']['front']='bohe'
        s['sides']['b']['characters']['bohe']['awakened']=True
        s['sides']['a']['last_front']='zero';s['sides']['a']['characters']['zero']['harmony']=2
        self.check(s,dict(type='attack',character_id='iloy'))
        s=self.game();s['sides']['a']['front']='bohe';s['sides']['b']['front']='iloy'
        s['sides']['a']['front_debuff']={'weave':{'by':'b','expires_side':'a'}}
        self.check(s,dict(type='attack',character_id='bohe'))
    def test_auto_attack_and_death_hook(self):
        s=self.play(self.game(),'M08')
        s=self.check(s,dict(type='end_turn'))
        self.check(s,dict(type='end_turn'))
        s=self.game();s['sides']['a']['characters']['bohe'].update(hp=0,down_turns=2)
        self.play(s,'Y01')
        s=self.game();s['sides']['a']['characters']['iloy'].update(shape='Y07',hp=1)
        s['sides']['b']['front']='baicang'
        self.check(s,dict(type='attack',character_id='iloy'))
    def test_support_collapse_threshold(self):
        s=self.game();s['sides']['a']['last_front']='zero'
        s['sides']['a']['characters']['zero']['harmony']=2
        s['sides']['b']['front']='baicang'
        s['sides']['b']['characters']['baicang'].update(hp=10,max_hp=10)
        s['sides']['b']['characters']['baicang']['flags']['collapse_count']=4
        self.check(s,dict(type='attack',character_id='iloy'))


@unittest.skipUnless(importlib.util.find_spec('torch'),'torch required')
class ResidentObservationTest(unittest.TestCase):
    def test_hidden_information_does_not_enter_policy(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.env import GpuDuelEnv
        from app.modules.card_game.rl.gpu_duel.resident_obs import resident_observe
        env=GpuDuelEnv(4,device='cpu',deck='starter')
        env.reset(seeds=[0,1,2,3])
        before=resident_observe(env.state)
        for row in range(4):
            foe=1-int(env.state.active[row])
            env.state.hand_kind[row,foe].fill_(0)
        env.state.deck_kind.fill_(0);env.state.rng.add_(9999)
        after=resident_observe(env.state)
        for a,b in zip(before,after):self.assertTrue(torch.equal(a,b))
