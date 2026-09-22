import unittest
from app.modules.card_game.rl.zhenhong_setup_trial import paired_rows
class SetupTrialTest(unittest.TestCase):
    def game(self,complete=True):
        return dict(complete=complete,decisions=2,searched=2,rows=[dict(side='a',key='zhenhong'),dict(side='b',key='starter')])
    def test_update_requires_completed_matched_pair_and_filters_frozen_side(self):
        self.assertIsNone(paired_rows({'control':self.game()}))
        self.assertIsNone(paired_rows({'control':self.game(),'setup':self.game(False)}))
        result=paired_rows({'control':self.game(),'setup':self.game()})
        self.assertEqual({k:len(v) for k,v in result.items()},{'control':1,'setup':1})
        self.assertTrue(all(r['side']=='a' for rows in result.values() for r in rows))
    def test_raw_training_labels_cannot_enter_search_update(self):
        bad=self.game();bad['searched']=1
        with self.assertRaises(ValueError):paired_rows({'control':self.game(),'setup':bad})
