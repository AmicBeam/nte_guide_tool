import importlib.util,json,tempfile,time,unittest
from pathlib import Path
import numpy as np
from app.modules.card_game.rl.information_longrun import validate_deadlines,save_replay,load_replay,selection_decision,append
from app.modules.card_game.rl.league_schema import feature_names,CAND_DIM,PRESETS,rule_hash

class LongrunTest(unittest.TestCase):
    def test_explicit_deadlines_and_no_extension(self):
        validate_deadlines(110,120,130,now=100)
        for times in ((99,120,130),(110,130,120),(110,120,50000)):
            with self.assertRaises(ValueError):validate_deadlines(*times,now=100)
        from scripts.train_duel_v2_until import epoch
        self.assertEqual(epoch('2026-09-19T12:00:00+08:00'),1789790400)
        with self.assertRaises(Exception):epoch('2026-09-19T12:00:00')

    def test_numeric_replay_and_jsonl_roundtrip(self):
        row=dict(x=np.zeros(len(feature_names()),np.float32),c=np.zeros((2,CAND_DIM),np.float32),pi=np.array([.2,.8],np.float32),z=-1.)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows.npz';save_replay(p,[row]);loaded=load_replay(p)
            for k in ('x','c','pi'):np.testing.assert_array_equal(loaded[0][k],row[k])
            self.assertEqual(loaded[0]['z'],-1)
            save_replay(p,[]);self.assertEqual(load_replay(p),[])
            append(d,'events.jsonl',{'a':1});append(d,'events.jsonl',{'a':2})
            self.assertEqual([json.loads(l)['a'] for l in (Path(d)/'events.jsonl').read_text().splitlines()],[1,2])

    def test_promotion_requires_independent_search_confirmation_and_retention(self):
        raw=dict(complete=True,candidate=9,best=10)
        screen=dict(complete=True,candidate=6,best=4)
        confirm=dict(complete=True,candidate=5,best=4)
        self.assertEqual(selection_decision(raw,True,screen,confirm),'promote')
        self.assertEqual(selection_decision(raw,False,screen,confirm),'keep')
        self.assertEqual(selection_decision(raw,True,screen,None),'keep')
        self.assertEqual(selection_decision(raw,True,screen,dict(confirm,complete=False)),'keep')
        bad=dict(complete=True,candidate=5,best=10)
        self.assertEqual(selection_decision(bad,True,rollback_confirm=bad),'rollback')
        self.assertEqual(selection_decision(bad,True),'keep')

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch required')
    def test_checkpoint_roundtrip_and_tamper_rejection(self):
        import torch
        from app.modules.card_game.rl.information_longrun import checkpoint,restore
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
        nets={k:CompactScorer(len(feature_names()),CAND_DIM,4) for k in PRESETS}
        opts={k:torch.optim.Adam(n.parameters()) for k,n in nets.items()}
        replay={k:[] for k in PRESETS}
        with tempfile.TemporaryDirectory() as d:
            original={k:{name:t.clone() for name,t in n.state_dict().items()} for k,n in nets.items()}
            checkpoint(d,nets,opts,replay,dict(cycle=1,rule_hash=rule_hash()))
            with torch.no_grad():
                for n in nets.values():next(n.parameters()).add_(10)
            recovered,state=restore(d,nets,opts)
            self.assertEqual(state['cycle'],1);self.assertTrue(all(not v for v in recovered.values()))
            for k,n in nets.items():
                for name,t in n.state_dict().items():torch.testing.assert_close(t,original[k][name])
            pointer=json.loads((Path(d)/'recovery/pointer.json').read_text())
            target=Path(d)/'recovery'/pointer['path']/'training.pt'
            with target.open('ab') as f:f.write(b'bad')
            with self.assertRaisesRegex(ValueError,'mismatch'):restore(d,nets,opts)

    def test_saturated_search_still_allows_confirmed_raw_improvement(self):
        raw=dict(complete=True,candidate=12,best=10)
        tied=dict(complete=True,candidate=8,best=8)
        confirm=dict(complete=True,candidate=11,best=10)
        self.assertEqual(selection_decision(raw,True,tied,tied,confirm),'promote')
        self.assertEqual(selection_decision(raw,True,tied,tied,None),'keep')
        self.assertEqual(selection_decision(raw,True,tied,dict(tied,candidate=7),confirm),'keep')

    def test_early_finalization_cannot_train_or_extend_deadlines(self):
        from scripts.finalize_duel_v2_until import finalization_config
        original=dict(train_until=110,evaluate_until=155,hard_deadline=160,max_cycles=0)
        effective=finalization_config(original,27,100)
        self.assertEqual(original['train_until'],110)
        self.assertEqual(effective['evaluate_until'],155);self.assertEqual(effective['hard_deadline'],160)
        self.assertEqual(effective['max_cycles'],27)
        # Even a backwards wall-clock correction cannot open another training cycle.
        self.assertFalse(99<effective['train_until'] and (not effective['max_cycles'] or 27<effective['max_cycles']))
        with self.assertRaises(ValueError):finalization_config(original,0,100)
        with self.assertRaises(ValueError):finalization_config(original,27,160)
