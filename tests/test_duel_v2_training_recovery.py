import unittest
from app.modules.card_game.rl.full_cycle_experiment import can_update_completed, interleave_training
class RecoveryTest(unittest.TestCase):
    def test_finished_samples_survive_missing_sibling_before_cutoff(self):
        self.assertTrue(can_update_completed({'a':[{'z':1}],'b':[]},90,100))
        self.assertFalse(can_update_completed({'a':[],'b':[]},90,100))
        self.assertFalse(can_update_completed({'a':[{'z':1}]},100,100))
    def test_interleave_does_not_finish_one_branch_before_starting_other(self):
        seen=[]
        def branch(k,n):
            for i in range(n):seen.append((k,i));yield
            return k+'-finished'
        got=interleave_training({'original':branch('original',2),'candidate':branch('candidate',3)})
        self.assertEqual(seen,[('original',0),('candidate',0),('original',1),('candidate',1),('candidate',2)])
        self.assertEqual(got,{'original':'original-finished','candidate':'candidate-finished'})
    def test_unchanged_weights_cannot_be_labelled_new_models(self):
        import tempfile,json
        from pathlib import Path
        import numpy as np
        from app.modules.card_game.rl.new_model_report import write_matrix
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'initial').mkdir();(out/'models').mkdir()
            for folder in ['initial','models']:np.savez(out/folder/'starter.npz',weight=np.ones(2))
            (out/'models/starter.json').write_text(json.dumps({'sha256':'fixture','build_sha256':'fixture'}))
            self.assertFalse(write_matrix(out,[],['starter'],out/'models',{'smoke':False,'rule_hash':'fixture'}))
            result=json.loads((out/'matrix-validation.json').read_text())
            self.assertEqual(result['model_changed'],{'starter':False})
            self.assertFalse(result['new_model_complete'])
            self.assertIn('不满足全套训练后新模型要求',(out/'report.md').read_text())

    def test_formal_capacity_rejects_smoke_and_underbudget_estimates(self):
        from app.modules.card_game.rl.training_capacity import validate_capacity
        c={'created':1000,'rule_hash':'rule','simulations':256,'workers':8,'deadlines':{'foundation':2000,'search':3000,'candidate_adaptation':4000,'confirmation':5000,'audit':6000,'evaluation':7000}}
        r={'rule_hash':'rule','simulations':256,'workers':8,'complete':True,'measured_at':900,'estimated_stage_seconds':{k:500 for k in ['foundation','search','adaptation','confirmation','audit','evaluation']}}
        self.assertTrue(validate_capacity(r,c,1000))
        with self.assertRaises(ValueError):validate_capacity(r,dict(c,search_algorithm='gumbel'),1000)
        self.assertTrue(validate_capacity(dict(r,search_algorithm='gumbel',gumbel_candidates=16),dict(c,search_algorithm='gumbel',gumbel_candidates=16),1000))
        with self.assertRaises(ValueError):validate_capacity(dict(r,simulations=8),c,1000)
        r['estimated_stage_seconds']['adaptation']=900
        with self.assertRaises(ValueError):validate_capacity(r,c,1000)
