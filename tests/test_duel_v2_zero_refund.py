"""Z06 user-defined tabletop timing: reserve AP for responses, not next turn."""
import unittest
import importlib.util

from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action
from app.modules.card_game.engine.duel_v2.state import hero, card_instance


class ZeroRefundTest(unittest.TestCase):
    def game(self):
        state = new_game(seed=7, first_side='a', skip_mulligan=True)
        for side in ('a', 'b'):
            state['sides'][side]['hand'] = []
        state['sides']['a']['ap'] = 0
        hero(state, 'a', 'zero')['shape'] = 'Z06'
        return state

    def test_refund_can_pay_for_response_in_opponent_turn(self):
        state = self.game()
        state['sides']['a']['front'] = 'zero'
        response = card_instance(state, 'N02', 'a')
        state['sides']['a']['hand'] = [response]
        ended = apply_action(state, 'a', {'type': 'end_turn'})
        self.assertEqual(ended['sides']['a']['ap'], 1)
        self.assertEqual(ended['sides']['a'].get('extra_ap', 0), 0)
        self.assertEqual(state['sides']['a']['ap'], 0)
        attacked = apply_action(ended, 'b', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(attacked['sides']['a']['ap'], 0)
        self.assertEqual(attacked['sides']['a']['front'], 'nanali')
        self.assertTrue(any(c['instance_id'] == response['instance_id'] for c in attacked['sides']['a']['discard']))
        next_turn = apply_action(attacked, 'b', {'type': 'end_turn'})
        self.assertEqual(next_turn['sides']['a']['ap'], 2)

    def test_unspent_refund_does_not_stack_with_next_turn_budget(self):
        state = self.game()
        state['sides']['a']['ap'] = 2
        ended = apply_action(state, 'a', {'type': 'end_turn'})
        self.assertEqual(ended['sides']['a']['ap'], 3)
        next_turn = apply_action(ended, 'b', {'type': 'end_turn'})
        self.assertEqual(next_turn['sides']['a']['ap'], 2)

    def test_downed_or_replaced_weapon_does_not_refund(self):
        for shape, hp in [('Z06', 0), ('Z07', 5)]:
            with self.subTest(shape=shape, hp=hp):
                state = self.game()
                hero(state, 'a', 'zero').update(shape=shape, hp=hp)
                ended = apply_action(state, 'a', {'type': 'end_turn'})
                self.assertEqual(ended['sides']['a']['ap'], 0)

    def test_weapon_panel_has_six_health(self):
        self.assertEqual((CARDS['Z06']['attack'], CARDS['Z06']['hp']), (2, 6))
        state = self.game()
        hero(state, 'a', 'zero')['shape'] = None
        state['sides']['a']['ap'] = 1
        card = card_instance(state, 'Z06', 'a')
        state['sides']['a']['hand'] = [card]
        result = apply_action(state, 'a', {'type': 'play_card', 'card_id': card['instance_id']})
        self.assertEqual((hero(result, 'a', 'zero')['hp'], hero(result, 'a', 'zero')['max_hp']), (6, 6))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch required')
class TensorZeroRefundTest(unittest.TestCase):
    def test_tensor_refunds_only_live_active_rows(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.pack import pack_python
        from app.modules.card_game.rl.gpu_duel.rules import end_turn
        states=[]
        for hp in (5, 0, 5):
            state=ZeroRefundTest().game()
            hero(state, 'a', 'zero')['hp']=hp
            states.append(state)
        packed=pack_python(states, device='cpu')
        end_turn(packed, torch.tensor([0,0,0], dtype=torch.int32), torch.tensor([True,True,False]))
        self.assertEqual(packed.ap[:,0].tolist(),[1,0,0])
        self.assertEqual(packed.extra_ap[:,0].tolist(),[0,0,0])
