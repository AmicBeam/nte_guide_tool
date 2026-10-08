import unittest
from copy import deepcopy
from app.modules.card_game.engine.duel_v2.replay_log import apply_event_to_hand
from app.modules.card_game.engine.duel_v2.presentation import merge_patch

class ReplayHandCountTest(unittest.TestCase):
    def test_draw_play_discard_counts_are_not_reapplied(self):
        for viewer in ('a','b'):
            board={'viewer_side':viewer,'sides':{'a':{'hand_count':6,'hand':[{'hidden':True} for _ in range(6)]}}}
            for kind,count in [('draw',7),('play',6),('discard',6),('remove',5),('gain',6)]:
                patch={'sides':{'a':{'hand_count':count}}} if kind!='discard' else {}
                event={'side':'a','type':kind,'patch':patch,'card':{'instance_id':'a-1','name':'card'}}
                board=apply_event_to_hand(merge_patch(board,patch),event)
                self.assertEqual(board['sides']['a']['hand_count'],count)
                self.assertEqual(len(board['sides']['a']['hand']),count)

    def test_authoritative_hand_patch_not_removed_again(self):
        board={'viewer_side':'a','sides':{'a':{'hand_count':1,'hand':[{'instance_id':'kept'}]}}}
        event={'side':'a','type':'play','card':{'instance_id':'played'},'patch':{'sides':{'a':deepcopy(board['sides']['a'])}}}
        self.assertEqual(apply_event_to_hand(deepcopy(board),event),board)

    def test_private_mulligan_does_not_keep_stale_card_identities(self):
        from app.modules.card_game.engine.duel_v2.replay_log import normalize_replay_hands
        board={'sides':{'a':{'hand_count':1,'hand':[{'instance_id':'old-1'},{'instance_id':'old-2'}]}}}
        self.assertEqual(normalize_replay_hands(board)['sides']['a']['hand'],[{'hidden':True}])
