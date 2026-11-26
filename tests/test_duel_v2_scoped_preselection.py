"""Candidate enumeration honors per-character locks without implying training readiness."""
from collections import Counter
from copy import deepcopy
import unittest

from app.modules.card_game.content.duel_v2 import CARDS, STARTER_DECKS
from app.modules.card_game.rl.local_card_search import preselection_neighbors, replacement_delta
from app.modules.card_game.rl.build_acceptance import build_identity


class ScopedPreselectionTest(unittest.TestCase):
    def test_single_and_double_replacements_only_change_the_explicit_characters(self):
        for key,mutable in (('starter',['nanali']),('collapse',['heiyu','dafutier']),('lingke',['lingke'])):
            with self.subTest(key=key):
                baseline=deepcopy(next(d for d in STARTER_DECKS if d['id']==key))
                before=deepcopy(baseline)
                candidates=preselection_neighbors(baseline,mutable_characters=mutable,seed=31,two_limit=32)
                self.assertTrue(candidates)
                self.assertEqual(baseline,before)
                self.assertEqual({c['copies_changed'] for c in candidates},{1,2})
                self.assertEqual(len(candidates),len({build_identity(c['build']) for c in candidates}))
                for candidate in candidates:
                    build=candidate['build']
                    self.assertEqual(build['character_ids'],baseline['character_ids'])
                    delta=replacement_delta(baseline,build)
                    self.assertTrue(all(CARDS[c]['character_id'] in mutable for c in
                                        [*delta['removed'],*delta['added']]))
                    self.assertTrue(all(n<=2 for n in Counter(build['card_ids']).values()))
                    self.assertFalse(any(CARDS[c].get('derived') for c in build['card_ids']))
                    for owner in baseline['character_ids']:
                        cards=lambda b:Counter(c for c in b['card_ids'] if CARDS[c]['character_id']==owner)
                        self.assertEqual(sum(cards(build).values()),8)
                        if owner not in mutable:self.assertEqual(cards(build),cards(baseline))

    def test_empty_scope_has_no_candidates_and_invalid_scopes_are_rejected(self):
        deck=next(d for d in STARTER_DECKS if d['id']=='lingke')
        self.assertEqual(preselection_neighbors(deck,mutable_characters=[],two_limit=0),[])
        for invalid in (['zero'],['lingke','lingke']):
            with self.subTest(scope=invalid),self.assertRaises(ValueError):
                preselection_neighbors(deck,mutable_characters=invalid)

    def test_candidate_sampling_is_stable_and_ignores_source_card_row_order(self):
        deck=deepcopy(next(d for d in STARTER_DECKS if d['id']=='collapse'))
        first=preselection_neighbors(deck,mutable_characters=['heiyu'],seed=7,two_limit=8)
        deck['card_ids'].reverse()
        again=preselection_neighbors(deck,mutable_characters=['heiyu'],seed=7,two_limit=8)
        self.assertEqual(first,again)


if __name__=='__main__':
    unittest.main()
