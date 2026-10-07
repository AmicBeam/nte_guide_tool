"""Search selection, privacy guards and completed partial results at serving."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from app.modules.card_game.engine.ai import advanced_search as planner
from app.modules.card_game.engine.duel_v2 import new_game


class ServingSearchTest(unittest.TestCase):
    def test_completed_partial_search_wins_over_model_argmax(self):
        game=new_game(seed=10,skip_mulligan=True)
        actions=[{'type':'attack','character_id':'nanali'},{'type':'end_turn'}]
        runtime=SimpleNamespace(decision=lambda state,side:(actions,(None,None)))
        model=SimpleNamespace(version='numeric-test')
        with patch.object(planner,'_runtime',return_value=runtime),patch.object(planner,'_public_terminal_allowed',return_value=False),patch('app.modules.card_game.rl.information_search.search',return_value=dict(actions=actions,search_choice=1,simulations=7,complete=False)) as search:
            self.assertEqual(planner.choose(game,game['active_side'],model,actions[0]),actions[1])
            self.assertEqual(search.call_args.kwargs['simulations'],32)
            self.assertIs(search.call_args.kwargs['noise'],False)

    def test_zero_completed_simulations_keep_legal_fallback(self):
        game=new_game(seed=10,skip_mulligan=True);action={'type':'end_turn'}
        runtime=SimpleNamespace(decision=lambda state,side:([action,{'type':'attack'}],(None,None)))
        with patch.object(planner,'_runtime',return_value=runtime),patch.object(planner,'_public_terminal_allowed',return_value=False),patch('app.modules.card_game.rl.information_search.search',return_value={'simulations':0}):
            self.assertEqual(planner.choose(game,game['active_side'],SimpleNamespace(version='test'),action),action)

    def test_response_guard_uses_possible_roster_not_actual_hidden_identity(self):
        game=new_game(seed=10,skip_mulligan=True);side=game['active_side'];foe='b' if side=='a' else 'a'
        # Nanali's response remains possible even when the actual hidden hand
        # happens not to contain it. Do not read dark cards to permit a proof.
        self.assertTrue(game['sides'][foe]['hand'])
        self.assertFalse(planner._public_terminal_allowed(game,side))
        game['sides'][foe]['hand']=[]
        self.assertTrue(planner._public_terminal_allowed(game,side))

    def test_public_immediate_win_has_priority_even_without_a_policy_prior(self):
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        from app.modules.card_game.engine.duel_v2 import apply_action
        decks={d['id']:d for d in STARTER_DECKS}
        game=new_game(seed=33,first_side='a',skip_mulligan=True,decks={'a':decks['zhenhong'],'b':decks['weave-rush']})
        game['sides']['a']['front']='zero'
        game['sides']['a']['characters']['zero']['harmony']=2
        game['sides']['b']['front']=None
        game['sides']['b']['hp']=1
        game['sides']['b']['shield']=0
        for cid,h in game['sides']['b']['characters'].items():
            if cid!='iloy':h['hp']=0
        # No scores method: a verified public terminal result must be chosen
        # before a low/absent network prior can suppress it.
        model=SimpleNamespace(schema='fixed_ten_v1',version='public-win-fixture')
        chosen=planner.choose(game,'a',model,{'type':'end_turn'})
        self.assertNotEqual(chosen,{'type':'end_turn'})
        self.assertEqual(apply_action(game,'a',chosen)['winner'],'a')
