import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from app.modules.card_game.rl.information_search import decision_budget
from app.modules.card_game.rl.selfplay_trial import make_job,ARMS
from app.modules.card_game.rl.episode_replay import dump_episode

class SelfplayTrialTest(unittest.TestCase):
    def job(self,arm,n=0):
        base={k:SimpleNamespace(serving_deck={'id':k}) for k in ('zhenhong','starter','weave-rush','quick-rush')}
        return make_job(arm,n,Path('/tmp/out'),dict(smoke=False,simulations=256,train_until=10**12,seeds={'foundation':100}),base,Path('/tmp/current'),Path('/tmp/source'),Path('/tmp/initial'))

    def test_mirror_learns_both_sides_with_same_model(self):
        j=self.job('self256')
        self.assertEqual(j['train_sides'],['a','b']);self.assertEqual(j['policies']['a'],j['policies']['b'])
        self.assertEqual(j['decks']['a'],j['decks']['b'])

    def test_history_and_cross_freeze_opponent_and_cover_seats(self):
        h=self.job('selfmixed',2);self.assertEqual(h['train_sides'],['a']);self.assertEqual(Path(h['policies']['b'][0]),Path('/tmp/initial'))
        cross=[self.job('self256',i) for i in (3,7,11,15,19,23)]
        self.assertEqual({(j['policies']['b'][1],j['first']) for j in cross},{(f,s) for f in ('starter','weave-rush','quick-rush') for s in ('a','b')})
        self.assertEqual(self.job('fixed256')['matchup'],'cross')

    def test_mixed_budget_is_independent_reproducible_and_labels_only_deep(self):
        j=self.job('selfmixed');r=np.random.default_rng(9)
        decisions=[decision_budget(j,r) for _ in range(100)]
        self.assertEqual(set(decisions),{(32,False),(256,True)})
        r=np.random.default_rng(9);self.assertEqual(decisions,[decision_budget(j,r) for _ in range(100)])
        self.assertEqual(decision_budget(self.job('self256'),r),(256,True))
        with self.assertRaises(ValueError):decision_budget(dict(search_mix=dict(low=0,high=256,deep_probability=.25)),r)

    def test_episode_keeps_both_terminal_labels_and_filters_frozen_sides(self):
        import tempfile
        game=dict(complete=True,decisions=2,episode_actions=[{},{}],winner='a',value_rows=[],rows=[dict(side='a',z=1),dict(side='b',z=-1)])
        with tempfile.TemporaryDirectory() as d:
            both=dump_episode(Path(d)/'both.json.gz',dict(train_sides=['a','b']),game)
            self.assertEqual({r['z'] for r in both['rows']},{-1,1})
            single=dump_episode(Path(d)/'single.json.gz',dict(train_sides=['a']),game)
            self.assertEqual(len(single['rows']),1)
