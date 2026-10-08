import unittest
from app.modules.card_game.rl.progress_stop import ProgressStop

class ProgressStopTest(unittest.TestCase):
    def row(self, **kwargs):
        return dict(value_loss=2.,explained_variance=.7,entropy=.4,approx_kl=.001,clip_fraction=.02,finished_games=10,**kwargs)
    def test_six_windows_after_reference(self):
        p=ProgressStop()
        for n in range(1,7):self.assertFalse(p.observe(n*300,self.row()))
        self.assertTrue(p.observe(2100,self.row()))
        self.assertEqual(p.last_summary['stale_windows'],6)
    def test_change_restarts_patience(self):
        p=ProgressStop(window_seconds=1,patience=2)
        self.assertFalse(p.observe(1,self.row()));self.assertFalse(p.observe(2,self.row()))
        row=self.row();row['explained_variance']=.75
        self.assertFalse(p.observe(3,row));self.assertFalse(p.observe(4,row));self.assertTrue(p.observe(5,row))
    def test_unhealthy_is_not_convergence(self):
        p=ProgressStop(window_seconds=1,patience=2);row=self.row();row['finished_games']=0
        self.assertFalse(p.observe(1,row))
        with self.assertRaises(RuntimeError):p.observe(2,row)
    def test_card_progress(self):
        p=ProgressStop(window_seconds=1,patience=2)
        row=dict(reward_mean=.3,loss=1.,valid_candidates=8)
        self.assertFalse(p.observe(1,row));self.assertFalse(p.observe(2,row));self.assertTrue(p.observe(3,row))
