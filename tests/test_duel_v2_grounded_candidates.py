import tempfile
import unittest
import numpy as np
from app.modules.card_game.rl import grounded_candidates as gc, outcome_runtime as rt
from app.modules.card_game.rl.fixed_lineup import all_features,CAND_DIM
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game


class GroundedTest(unittest.TestCase):
    def test_actor_fields_are_gathered_without_ordinal_identity(self):
        x=np.zeros(len(all_features()),np.float32);c=np.zeros((2,CAND_DIM),np.float32)
        c[:,1]=[8,9];c[:,2:5]=-1
        x[gc.INDEX[0,8,0]]=3;x[gc.INDEX[0,9,0]]=7
        y=gc.transform(x,c)
        self.assertEqual(y.shape,(2,gc.DIM))
        self.assertEqual(y[0,-2*len(gc.FIELDS)],3)
        self.assertEqual(y[1,-2*len(gc.FIELDS)],7)
        np.testing.assert_array_equal(y[:,-len(gc.FIELDS):],0)
        self.assertEqual(gc.transform(x,c[:0]).shape,(0,gc.DIM))

    def test_torch_numpy_export_gradients_and_reload(self):
        try:import torch
        except ImportError:self.skipTest('Torch required')
        torch.manual_seed(123)
        b=next(b for b in STARTER_DECKS if b['id']=='zhenhong')
        state=new_game(seed=901,first_side='a',skip_mulligan=True,decks={'a':b,'b':b})
        actions,(x,c)=rt.decision(state,'a');net=gc.network(16)
        xx=torch.tensor(x)[None];cc=torch.tensor(c)[None];mask=torch.ones((1,len(c)),dtype=torch.bool)
        np.testing.assert_allclose(net.candidate_features(xx,cc)[0].numpy(),gc.transform(x,c),atol=1e-6)
        row=dict(x=x,c=c,pi=np.eye(len(c),dtype=np.float32)[-1],z=1)
        before=net.cand_net[0].weight.detach().clone()
        rt.update(net,torch.optim.Adam(net.parameters(),lr=.001),[row])
        self.assertFalse(torch.equal(before,net.cand_net[0].weight))
        with tempfile.TemporaryDirectory() as d:
            rt.export(net,d,'zhenhong',b,{})
            m=rt.model(d,'zhenhong')
            with torch.no_grad():scores,_=net(xx,cc,mask)
            np.testing.assert_allclose(m.scores(x,c),scores[0].numpy(),atol=1e-6)
            restored=rt.create_network(d,'zhenhong','cpu')
            with torch.no_grad():again,_=restored(xx,cc,mask)
            torch.testing.assert_close(scores,again)
            self.assertFalse(m.manifest['automatic_serving_approval'])
