"""Current public harmony mechanics must remain distinguishable to learners."""
from copy import deepcopy
import unittest
import numpy as np

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game, observe
from app.modules.card_game.rl import seven_lineup as layout


class HarmonyObservationTest(unittest.TestCase):
    def view(self):
        build = next(d for d in STARTER_DECKS if d['id'] == 'starter')
        return observe(new_game(seed=3, decks={'a': build, 'b': build},
                                skip_mulligan=True), 'a', include_previews=False)

    def test_bead_lifetime_stacks_and_used_state_are_distinguishable(self):
        view = self.view()
        meta = view['policy_state']['sides']['a']
        variants = []
        for stacks, left, used, uncapped in ((1,2,-1,False), (2,2,-1,False),
                (2,1,-1,False), (2,2,view['turn'],False), (2,2,-1,True)):
            meta['genesis_layers'] = [dict(id=1, stacks=stacks, left=left,
                                          used_turn=used, uncapped=uncapped)]
            variants.append(np.asarray(layout.encode_harmony(view)).tobytes())
        self.assertEqual(len(set(variants)), 5)
        # Equivalent beads are counted, so an extra bead is never discarded.
        meta['genesis_layers'] *= 2
        self.assertNotIn(np.asarray(layout.encode_harmony(view)).tobytes(), variants)

    def test_continuation_order_and_entity_validity_are_preserved(self):
        view = self.view(); heroes = {h['id']: h for h in view['sides']['a']['characters']}
        meta = view['policy_state']['sides']['a']
        meta['extra_genesis_queue'] = [dict(actor=cid, actor_entity_id=heroes[cid]['entity_id'])
                                       for cid in ('zero', 'jiuyuan')]
        first = layout.encode_harmony(view)
        meta['extra_genesis_queue'].reverse()
        self.assertNotEqual(first, layout.encode_harmony(view))
        meta['extra_genesis_queue'].reverse()
        renamed = deepcopy(view)
        for side in ('a', 'b'):
            for hero in renamed['sides'][side]['characters']:
                hero['entity_id'] = 'renamed-' + hero['entity_id']
            for entry in renamed['policy_state']['sides'][side].get('extra_genesis_queue') or []:
                entry['actor_entity_id'] = 'renamed-' + entry['actor_entity_id']
        self.assertEqual(first, layout.encode_harmony(renamed))
        meta['extra_genesis_queue'][0]['actor_entity_id'] = 'stale-entity'
        self.assertNotEqual(first, layout.encode_harmony(view))

    def test_queue_overflow_fails_instead_of_truncating(self):
        view = self.view()
        view['policy_state']['sides']['a']['extra_genesis_queue'] = [dict(actor='zero')] * 65
        with self.assertRaisesRegex(ValueError, 'capacity'):
            layout.encode_harmony(view)

    def test_numeric_layout_does_not_follow_unrelated_public_releases(self):
        self.assertIn('PF01', layout.CARD_IDS)
        self.assertNotIn('V01', layout.CARD_IDS)
        view = self.view(); actions = [e['action'] for e in view['legal_actions']
                                      if e['action']['type'] != 'concede']
        before = deepcopy(view); x, candidates = layout.encode(view, actions)
        self.assertEqual(view, before)
        self.assertEqual(x.shape, (len(layout.all_features()),))
        self.assertEqual(candidates.shape, (len(actions), layout.CAND_DIM))
        self.assertTrue(np.isfinite(x).all())


if __name__ == '__main__':
    unittest.main()
