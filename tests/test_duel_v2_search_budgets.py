import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from app.modules.card_game.rl.search_budget_eval import BUDGETS,game_jobs,summarize,freeze_models,hashes,game_job,choose

class BudgetComparisonTest(unittest.TestCase):
    def test_every_budget_gets_identical_openings_opponents_and_seats(self):
        jobs=game_jobs(Path('/models'),123456,2)
        self.assertEqual(len(jobs),96)
        reference=None
        for budget in BUDGETS:
            group=[j for j in jobs if j['budget']==budget]
            signatures={(j['key'],j['foe'],j['seed'],j['first'],str(j['policies'])) for j in group}
            self.assertEqual(len(signatures),24)
            if reference is None:reference=signatures
            self.assertEqual(reference,signatures)
            self.assertEqual({j['first'] for j in group},{'a','b'})

    def test_incomplete_games_and_positions_are_not_losses_or_stable_cases(self):
        base=dict(key='starter',foe='weave-rush',seed=1,first='a',decisions=[])
        games=[dict(base,budget=0,complete=True,winner='b'),dict(base,budget=24,complete=True,winner='a'),
               dict(base,budget=128,complete=False),dict(base,budget=256,complete=True,winner='b')]
        positions=[dict(case='x',kind='natural',budget=128,complete=False,passed=None),
                   dict(case='x',kind='natural',budget=128,complete=True,passed=None,milliseconds=2,selected={'action':{'type':'attack'}})]
        result=summarize(games,positions,4)
        self.assertFalse(result['complete']);self.assertEqual(result['budgets'][24]['paired_gains'],1)
        self.assertEqual(result['budgets'][128]['paired_missing'],1)
        self.assertEqual(result['budgets'][128]['paired_losses'],0)
        self.assertEqual(result['budgets'][128]['assessed_cases'],0)
        self.assertEqual(result['budgets'][128]['partial_cases'],1)
        self.assertFalse(result['automatic_recommendation'])
        self.assertEqual(result['all_budget_complete_pairs'],0)

    def test_freeze_copies_only_valid_numeric_model_files_and_detects_mutation(self):
        from tests.test_duel_v2_league import zero_policy
        from app.modules.card_game.rl.league_schema import PRESETS
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';source.mkdir()
            for k in PRESETS:zero_policy(source,k)
            (source/'unrelated.txt').write_text('not part of model')
            target=Path(d)/'frozen';before=freeze_models(source,source,target)
            self.assertEqual(len(before),12);self.assertEqual(before,hashes(target))
            self.assertFalse(list(target.rglob('unrelated.txt')))
            p=target/'learner/starter.json';p.write_text(p.read_text()+' ')
            self.assertNotEqual(before,hashes(target))

    def test_oracle_audit_never_overrides_selected_action(self):
        state=dict(phase='playing',active_side='a',turn=1,sides={'a':{'hand':[]},'b':{'hand':[]}})
        selected={'type':'attack','character_id':'zero'}
        def commit(s,side,action):
            self.assertEqual(action,selected)
            return dict(s,phase='finished',winner='b')
        metrics=dict(complete=True,index=0,legal=2,milliseconds=1,selected={'action':selected})
        job=dict(seed=1,first='a',decks={},policies={'a':('x','starter'),'b':('x','weave-rush')},deadline=float('inf'),budget=24)
        with patch('app.modules.card_game.rl.search_budget_eval.new_game',return_value=state),patch('app.modules.card_game.rl.search_budget_eval.model'),patch('app.modules.card_game.rl.search_budget_eval.choose',return_value=(selected,metrics)),patch('app.modules.card_game.rl.search_budget_eval.prove_turn',return_value={'eligible':True,'roots_visited':2,'paths':{1:[{}]}}),patch('app.modules.card_game.rl.search_budget_eval.apply_action',side_effect=commit):
            result=game_job(job)
        self.assertTrue(result['complete']);self.assertEqual(result['winner'],'b')
        self.assertTrue(result['decisions'][0]['missed_certified_lethal'])

    def test_latency_and_actual_simulations_are_retained(self):
        a={'type':'attack','character_id':'zero'};s={'sides':{'a':{'hand':[]}}}
        result=dict(actions=[a],search_choice=0,raw_choice=0,complete=True,simulations=1,requested=1,
                    roots_covered=1,nodes=1,visits=np.array([1]),mean_values=np.array([.5]))
        with patch('app.modules.card_game.rl.search_budget_eval.search',return_value=result):
            action,row=choose(s,'a',{},256,1,999)
        self.assertEqual(action,a);self.assertEqual(row['simulations'],1)
        self.assertGreaterEqual(row['milliseconds'],0)

    def test_authored_fixtures_keep_hand_and_goal_with_consistent_decks(self):
        from copy import deepcopy
        from app.modules.card_game.rl.league_gate import cases
        from app.modules.card_game.rl.search_budget_eval import consistent_tactical_fixture,expected_actions
        from app.modules.card_game.rl.information_search import sample_world
        for key,name,state,side,goal in cases():
            original=deepcopy(state)
            repaired,changes=consistent_tactical_fixture(state)
            self.assertEqual(state,original)
            self.assertEqual(repaired['sides'][side]['hand'],state['sides'][side]['hand'])
            self.assertTrue(expected_actions(repaired,side,goal),name)
            sampled=sample_world(repaired,side,44)
            self.assertEqual(sampled['phase'],repaired['phase'])
            for team in repaired['sides'].values():
                self.assertEqual(sum(not c.get('derived') for z in ('hand','deck','discard') for c in team[z]),32)
