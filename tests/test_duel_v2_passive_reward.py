"""Count real Zhenhong callbacks and credit only subsequent triggers to decisions."""
import json
import unittest
from app.modules.card_game.rl.fixed_lineup_training import passive_count, passive_returns
from app.modules.card_game.rl.league_learning import ordinary_rows


class PassiveRewardTest(unittest.TestCase):
    def test_two_genesis_hits_count_one_passive_and_retrigger_adds_one(self):
        from tests.test_duel_v2_surplus_rework import SurplusReworkTest
        from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation
        from app.modules.card_game.engine.duel_v2.combat import resolve_genesis
        helper=SurplusReworkTest();state=helper.game();helper.delay(state)
        state['sides']['a']['front']='zero'
        for expected in (1,2):
            context=operation(state,'a','jiuyuan')
            resolve_genesis(context);finish_operation(state)
            self.assertEqual(passive_count(state,'a'),expected)
            self.assertEqual(passive_count(state,'b'),0)
        self.assertEqual(passive_count(json.loads(json.dumps(state)),'a'),2)

    def test_down_zhenhong_earns_nothing_and_existing_count_survives_down(self):
        from tests.test_duel_v2_surplus_rework import SurplusReworkTest
        from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation
        from app.modules.card_game.engine.duel_v2.combat import resolve_genesis
        from app.modules.card_game.engine.duel_v2.state import damage
        helper=SurplusReworkTest();state=helper.game();helper.delay(state)
        state['sides']['a']['front']='zero'
        context=operation(state,'a','jiuyuan');resolve_genesis(context);finish_operation(state)
        self.assertEqual(passive_count(state,'a'),1)
        damage(state,'a','zhenhong',999)  # immunity from the callback: explicitly end it before knockdown
        from app.modules.card_game.engine.duel_v2.modifiers import clear
        clear(state['sides']['a']['characters']['zhenhong'],'down')
        damage(state,'a','zhenhong',999)
        self.assertEqual(state['sides']['a']['characters']['zhenhong']['hp'],0)
        self.assertEqual(passive_count(state,'a'),1)
        context=operation(state,'a','jiuyuan');resolve_genesis(context);finish_operation(state)
        self.assertTrue(state['sides']['a']['surplus'])
        self.assertEqual(passive_count(state,'a'),1)

    def test_more_triggers_more_reward_but_no_past_credit(self):
        self.assertEqual(passive_returns(10,3,[0,1,3],.2),[10.6,10.4,10])
        self.assertEqual(passive_returns(-10,3,[0,3],.2),[-9.4,-10])
        self.assertEqual(passive_returns(10,3,[0,1,3],0),[10,10,10])
        self.assertGreater(passive_returns(10,100,[0],.2)[0],passive_returns(10,50,[0],.2)[0])

    def test_ppo_rows_use_reward_to_go_not_total_reward(self):
        entry=([],[],0,-.1,0)
        rows=ordinary_rows([dict(complete=True,reward=10.6,trajectory=[entry]*3,
                                trajectory_returns=[10.6,10.4,10])])
        self.assertEqual([r[-1] for r in rows],[10.6,10.4,10])
        self.assertEqual(ordinary_rows([dict(complete=False,reward=None,trajectory=[entry])]),[])
        self.assertEqual(ordinary_rows([dict(complete=True,reward=-10,trajectory=[entry])])[0][-1],-10)
        with self.assertRaises(ValueError):
            ordinary_rows([dict(complete=True,trajectory=[entry],trajectory_returns=[])])

    def test_no_delay_no_passive(self):
        from tests.test_duel_v2_surplus_rework import SurplusReworkTest
        from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation
        from app.modules.card_game.engine.duel_v2.combat import resolve_genesis
        state=SurplusReworkTest().game();context=operation(state,'a','jiuyuan')
        resolve_genesis(context);finish_operation(state)
        self.assertEqual(passive_count(state,'a'),0)
