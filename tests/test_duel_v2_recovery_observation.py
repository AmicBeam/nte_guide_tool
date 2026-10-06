import unittest
from copy import deepcopy
import numpy as np

from app.modules.card_game.engine.duel_v2 import new_game, observe
from app.modules.card_game.rl import cross_lineup as encoding, recovery_runtime as runtime
from scripts.train_duel_v2_current_round import preset_decks


class RecoveryObservationTest(unittest.TestCase):
    def setUp(self):
        decks = preset_decks()
        self.state = new_game(seed=91, skip_mulligan=True,
            decks={'a': decks['zhenhong'], 'b': decks['murk']})

    def test_visible_damage_cap_has_its_own_feature_and_no_cap_is_finite(self):
        view = observe(self.state, 'a', include_previews=False)
        x, _ = encoding.encode(view, [])
        offset = encoding.all_features().index('fixed:damage_limit:0:zhenhong')
        self.assertEqual(x[offset], 0)
        hero = next(h for h in view['sides']['a']['characters'] if h['id']=='zhenhong')
        hero['damage_limit'] = 1
        updated, _ = encoding.encode(view, [])
        self.assertAlmostEqual(float(updated[offset]), .1)
        self.assertEqual(np.count_nonzero(x != updated), 1)
        self.assertTrue(np.isfinite(updated).all())

    def test_copy_expiry_is_visible_semantics_and_instance_identity_is_ignored(self):
        actions, _ = runtime.decision(self.state, 'a')
        view = observe(self.state, 'a', include_previews=False)
        hand = view['sides']['a']['hand']
        card = hand[0]
        action = {'type':'play_card', 'card_id':card['instance_id']}
        _, before = encoding.encode(view, [action])
        copied = deepcopy(view)
        item = copied['sides']['a']['hand'][0]
        item.update(copy=True, expires_turn=copied['sides']['a']['turn_count']+1)
        _, after = encoding.encode(copied, [action])
        self.assertEqual(after[0, encoding.candidate_names().index('copy')], 1)
        self.assertAlmostEqual(float(after[0, encoding.candidate_names().index('expires_in')]), .1)
        self.assertFalse(np.array_equal(before, after))
        item['instance_id']='renamed-physical-card'
        _, renamed = encoding.encode(copied, [{**action,'card_id':item['instance_id']}])
        np.testing.assert_array_equal(after, renamed)

    def test_hidden_opponent_copy_expiry_cannot_enter_candidate_context(self):
        view = observe(self.state, 'a', include_previews=False)
        hand = view['sides']['a']['hand']
        action = {'type':'play_card','card_id':hand[0]['instance_id']}
        x, c = encoding.encode(view, [action])
        altered = deepcopy(view)
        for card in altered['sides']['b']['hand']:
            card.update(copy=True, expires_turn=1234, antique_investment=42)
        xx, cc = encoding.encode(altered, [action])
        np.testing.assert_array_equal(x, xx)
        np.testing.assert_array_equal(c, cc)

    def test_transformed_fangs_retain_visible_physical_card_identity_and_discard_semantics(self):
        from app.modules.card_game.engine.duel_v2.state import card_instance, effective_card, discard
        team=self.state['sides']['a']
        team['hand']=[card_instance(self.state,cid,'a') for cid in ('I03','I04')]
        team['characters']['zhenhong']['awakened']=True
        view=observe(self.state,'a',include_previews=False)
        self.assertEqual([c['card_id'] for c in view['sides']['a']['hand']],['RF01','RF01'])
        actions=[dict(type='play_card',card_id=c['instance_id']) for c in team['hand']]
        x,c=encoding.encode(view,actions)
        self.assertEqual(c[0,2],c[1,2])
        self.assertFalse(np.array_equal(c[0],c[1]))
        for index,cid in enumerate(('I03','I04')):
            self.assertEqual(c[index,encoding.candidate_names().index('physical:'+cid)],1)
            self.assertEqual(x[encoding.all_features().index('substituted_hand:'+cid)],.5)
            branch=deepcopy(self.state)
            physical=branch['sides']['a']['hand'][index]
            transformed=effective_card(branch,'a',physical)
            self.assertEqual(transformed['card_id'],'RF01')
            discard(branch,'a',transformed)
            self.assertEqual(branch['sides']['a']['discard'][-1]['card_id'],cid)

    def test_opponent_hidden_original_faces_never_enter_own_inventory(self):
        view=observe(self.state,'a',include_previews=False)
        x,c=encoding.encode(view,[])
        changed=deepcopy(view)
        for item in changed['sides']['b']['hand']:
            item['hand_face']=dict(card_id='I03')
        xx,cc=encoding.encode(changed,[])
        np.testing.assert_array_equal(x,xx)
        np.testing.assert_array_equal(c,cc)


if __name__=='__main__':
    unittest.main()
