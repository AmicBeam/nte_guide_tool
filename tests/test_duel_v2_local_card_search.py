import unittest
from collections import Counter
from app.modules.card_game.rl.capability import preset_opponent_deck,assert_public_encoder_deck
from app.modules.card_game.rl.local_card_search import local_neighbors,replacement_delta,refine_locally
from tests.test_duel_v2_candidate_selection import groups,schedule


class LocalCardSearchTest(unittest.TestCase):
    def test_one_and_two_copy_neighbors_are_legal_unique_and_deterministic(self):
        deck=preset_opponent_deck('starter')
        rows=local_neighbors(deck,seed=4,two_limit=128)
        self.assertEqual(rows,local_neighbors(deck,seed=4,two_limit=128))
        signatures=set();counts=Counter()
        for r in rows:
            assert_public_encoder_deck(r['build'])
            self.assertEqual(r['build']['character_ids'],deck['character_ids'])
            d=replacement_delta(deck,r['build']);counts[d['copies_changed']]+=1
            self.assertIn(d['copies_changed'],(1,2))
            signatures.add(tuple(sorted(Counter(r['build']['card_ids']).items())))
        self.assertEqual(len(signatures),len(rows))
        self.assertGreater(counts[1],0);self.assertEqual(counts[2],128)
        self.assertEqual(counts[1],len(local_neighbors(deck,two_limit=0)))

    def test_two_copy_synergy_survives_bad_single_changes(self):
        deck=preset_opponent_deck('starter')
        target=next(r['build'] for r in local_neighbors(deck,seed=4,two_limit=16) if r['copies_changed']==2)
        target_key=Counter(target['card_ids'])
        def evaluate(builds,plan):
            pairs=len(plan)//2
            return [{'complete':True,'groups':groups(pairs,pairs if Counter(b['card_ids'])==target_key else pairs//2 if Counter(b['card_ids'])==Counter(deck['card_ids']) else 0)} for b in builds]
        result=refine_locally(deck,evaluate,rounds=1,two_limit=16,screen_pairs=4,final_pairs=8,seed_base=4,schedule_factory=schedule)
        self.assertTrue(result['complete'])
        self.assertEqual(Counter(result['build']['card_ids']),target_key)
        self.assertEqual(result['changes'][0]['copies_changed'],2)
        self.assertEqual(result['changes'][0]['selected_score'],1)

    def test_incomplete_round_does_not_promote(self):
        deck=preset_opponent_deck('starter')
        r=refine_locally(deck,lambda decks,plan:[{'complete':False,'groups':{}} for _ in decks],rounds=1,two_limit=0,schedule_factory=schedule)
        self.assertFalse(r['complete']);self.assertEqual(r['build'],deck)

    def test_rounds_continue_on_ties_and_keep_baseline(self):
        deck=preset_opponent_deck('starter')
        r=refine_locally(deck,lambda decks,plan:[{'complete':True,'groups':groups(len(plan)//2,1)} for _ in decks],rounds=4,two_limit=0,screen_pairs=2,final_pairs=4,schedule_factory=schedule)
        self.assertTrue(r['complete']);self.assertEqual(r['completed_rounds'],4)
        self.assertEqual(r['selection'],'original');self.assertEqual(len(r['stages']),8)


if __name__=='__main__':unittest.main()
