from copy import deepcopy
import unittest
import numpy as np
from app.modules.card_game.rl.recovery_sampler import determinize
from app.modules.card_game.rl import recovery_runtime as rt
from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.engine.duel_v2.state import card_instance
from app.modules.card_game.content.duel_v2 import CARDS,validate_deck
from scripts.train_duel_v2_current_round import preset_decks


class RecoverySamplerTest(unittest.TestCase):
    def game(self,all_kit=False):
        decks=preset_decks();build=decks['murk']
        if all_kit:
            ids=[cid for hero in build['character_ids'] for cid,c in CARDS.items()
                 if c['character_id']==hero and not c.get('derived')]
            build=validate_deck(dict(build,card_ids=ids))
        return new_game(seed=2,first_side='a',skip_mulligan=True,
                        decks={'a':decks['starter'],'b':build})

    def discard_card(self,state,cid):
        team=state['sides']['b']
        for zone in ('hand','deck'):
            card=next((c for c in team[zone] if c['card_id']==cid),None)
            if card is not None:team[zone].remove(card);team['discard'].append(card);return card
        self.fail('Missing fixture card '+cid)

    def test_original_a03_is_legal_known_original_not_a_generated_surplus(self):
        state=self.game(True);self.discard_card(state,'A03')
        before=deepcopy(state);out=determinize(state,'a',17)
        self.assertEqual(state,before)
        self.assertEqual(len(out['sides']['b']['hand']),len(state['sides']['b']['hand']))
        self.assertEqual(len(out['sides']['b']['deck']),len(state['sides']['b']['deck']))
        self.assertTrue(any(c['card_id']=='A03' for c in out['sides']['b']['discard']))
        self.assertFalse(any(c.get('ephemeral_turn') is not None for c in out['sides']['b']['hand']+out['sides']['b']['deck']))

    def test_generated_a03_stays_in_hand_and_ignores_unobservable_provenance(self):
        state=self.game();self.discard_card(state,'A04')
        state['active_side']='b'
        state['sides']['b']['discard'].append(card_instance(state,'AF01','b'))
        generated=card_instance(state,'A03','b');generated['ephemeral_turn']=state['turn']
        state['sides']['b']['hand'].append(generated)
        out=determinize(state,'a',5)
        self.assertTrue(any(c['card_id']=='A03' and c.get('ephemeral_turn') is not None for c in out['sides']['b']['hand']))
        self.assertFalse(any(c.get('ephemeral_turn') is not None for c in out['sides']['b']['deck']))
        changed=deepcopy(state)
        for c in changed['sides']['b']['hand']+changed['sides']['b']['deck']:
            c['card_id']='A01';c['ephemeral_turn']=999;c['instance_id']='b-9999';c['entity_id']='e9999'
        changed['rng']=555
        altered=determinize(changed,'a',5)
        self.assertEqual(out,altered)
        from app.modules.card_game.rl.cross_lineup import encode
        from app.modules.card_game.engine.duel_v2 import observe
        x,c=encode(observe(state,'a',include_previews=False),[])
        xx,cc=encode(observe(out,'a',include_previews=False),[])
        np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(c,cc)

    def test_expired_generated_a03_is_publicly_accounted_not_reinvented_in_hand(self):
        state=self.game();self.discard_card(state,'A04')
        state['sides']['b']['discard'].append(card_instance(state,'AF01','b'))
        expired=card_instance(state,'A03','b');expired['ephemeral_turn']=state['turn']-1
        state['sides']['b']['discard'].append(expired)
        for seed in range(4):
            out=determinize(state,'a',seed)
            self.assertFalse(any(c.get('ephemeral_turn') is not None for c in out['sides']['b']['hand']))

    def test_c02_copy_counts_can_exceed_original_limit_and_are_hand_consistent(self):
        state=self.game(True);self.discard_card(state,'C02')
        # Three successive C02 plays can leave two generated copies in the
        # public discard and the third in hand. A single play cannot make 3.
        for _ in range(2):state['sides']['b']['discard'].append(card_instance(state,'C02','b'))
        state['sides']['b']['hand'].append(card_instance(state,'C02','b'))
        for seed in range(4):
            out=determinize(state,'a',seed)
            self.assertGreaterEqual(sum(c['card_id']=='C02' for c in out['sides']['b']['hand']),1)
            self.assertLessEqual(sum(c['card_id']=='C02' for c in out['sides']['b']['deck']),1)

    def test_unexposed_a03_timing_on_public_discard_does_not_condition_belief(self):
        state=self.game(True);self.discard_card(state,'A03')
        changed=deepcopy(state);changed['sides']['b']['discard'][-1]['ephemeral_turn']=999
        self.assertEqual(determinize(state,'a',17),determinize(changed,'a',17))


if __name__=='__main__':unittest.main()
