import gzip,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.rl.full_cycle_experiment import evaluate_job


class SearchEvaluationTest(unittest.TestCase):
    def test_search_decisions_keep_exact_budget_and_real_replay(self):
        from app.modules.card_game.rl import ten_search_runtime as runtime
        class Policy:
            def scores(self,x,c):return np.zeros(len(c))
            def scores_value(self,x,c):return self.scores(x,c),0.
        build=next(b for b in STARTER_DECKS if b['id']=='zhenhong')
        with tempfile.TemporaryDirectory() as folder,patch.object(runtime,'model',return_value=Policy()):
            out=Path(folder);(out/'raw').mkdir();(out/'replays').mkdir()
            job=dict(id='search-eval',cell='a/b',seed=9812,first='b',deadline=time.time()+60,runtime='ten',
                     decks={'a':build,'b':build},policies={'a':('test','zhenhong'),'b':('test','zhenhong')},
                     output=str(out),replay=True,search_sides=['a'],simulations=2,search_algorithm='gumbel',gumbel_candidates=2)
            result=evaluate_job(job)
            self.assertTrue(result['complete'],result['error']);self.assertTrue(result['replay_verified'])
            with gzip.open(out/'raw/search-eval.json.gz','rt') as f:raw=json.load(f)
            self.assertEqual(len(raw['search_stats']),sum(a['side']=='a' for a in raw['actions']))
            self.assertGreater(result['search_decisions'],0)
            self.assertTrue(all(r['simulations']<=2 and not r['noise'] for r in raw['search_stats']))
            with gzip.open(out/'replays/search-eval.json.gz','rt') as f:public=json.load(f)
            self.assertNotIn('search_stats',public);self.assertNotIn('seed',public)
            with self.assertRaises(ValueError):evaluate_job({**job,'details':False})
