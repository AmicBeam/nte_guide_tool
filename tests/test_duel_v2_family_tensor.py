import unittest

class FamilyTensorTest(unittest.TestCase):
    def test_family_harmony_uses_current_cap(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.state import empty_state
        from app.modules.card_game.rl.gpu_duel.rules import _play_effects
        from app.modules.card_game.rl.gpu_duel.catalog import CARD_INDEX,SEAT_INDEX
        t=lambda n:torch.tensor([n],dtype=torch.int32)
        for existing in (0,1,2):
            state=empty_state(1,'cpu');seat=SEAT_INDEX['nanali']
            state.ch_hp[0,0,seat]=5;state.ch_harmony[0,0,seat]=existing
            _play_effects(state,t(0),seat,t(CARD_INDEX['NF01']),torch.tensor([True]),t(0),t(seat))
            self.assertEqual(int(state.ch_harmony[0,0,seat]),min(2,existing+1))
