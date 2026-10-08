from pathlib import Path
import tempfile
import unittest
import numpy as np
from app.modules.card_game.rl import recovery_round as rr, recovery_runtime as rt
from app.modules.card_game.rl.batched_search.value_metrics import compute
from app.modules.card_game.engine.duel_v2 import new_game


class BatchedValueMetricsTest(unittest.TestCase):
    def test_matches_numpy_with_equal_game_weights_and_both_actor_contexts(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);_,decks,_=rr.initialize(root,16,'cpu',17)
            before=rr.versions(root);policies={key:rt.model(root,key) for key in decks}
            games=[]
            for repeat in (1,3):
                rows=[]
                for i,key in enumerate(decks):
                    s=new_game(seed=70+i,first_side='a',decks={'a':decks[key],'b':decks[key]})
                    _,(x,_)=rt.decision(s,'a')
                    rows.extend(dict(x=x,key=key,z=1 if repeat==1 else -1,value_actor=actor)
                                for _ in range(repeat) for actor in (0,1))
                games.append(dict(value_rows=rows))
            expected=rr.value_metrics(games,policies)
            actual=compute(games,policies,device='cpu')
            for key in policies:
                self.assertEqual(actual[key]['normal_games'],2)
                for name in ('log_loss','brier','win_ece'):
                    self.assertAlmostEqual(actual[key][name],expected[key][name],places=5)
            self.assertEqual(before,rr.versions(root))

    def test_cuda_mode_cannot_fall_back_or_accept_an_unknown_backend(self):
        with self.assertRaisesRegex(ValueError,'silently'):
            rr.value_metrics([],{},backend='cuda_batched',device='cpu')
        with self.assertRaisesRegex(ValueError,'Unknown'):
            rr.value_metrics([],{},backend='other')


if __name__=='__main__':unittest.main()
