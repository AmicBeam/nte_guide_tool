import unittest
from app.modules.card_game.engine.duel_v2.replay_log import leak_markers, strip_hidden
from app.modules.card_game.engine.duel_v2.presentation import merge_patch


class ReplayRemovalTest(unittest.TestCase):
    def test_only_exact_public_character_tombstone_is_allowed(self):
        patch={'sides':{'b':{'characters':[{'id':'summon_1','removed':True}]}}}
        event={'patch':patch}
        self.assertEqual(strip_hidden(event),event)
        self.assertEqual(leak_markers({'views':{'a':{'events':[event]}}}),[])
        board={'sides':{'b':{'characters':[{'id':'summon_1','hp':0},{'id':'zaowu','hp':4}]}}}
        updated=merge_patch(board,strip_hidden(event)['patch'])
        self.assertEqual([r['id'] for r in updated['sides']['b']['characters']],['zaowu'])
        for hidden in ({'removed':[{'card_id':'SECRET'}]},
                       {'id':'summon_1','removed':True},
                       {'patch':{'sides':{'b':{'characters':[{'id':'summon_1','removed':['SECRET']}]}}}},
                       {'patch':{'sides':{'b':{'characters':[{'id':'summon_1','removed':True,'seed':4}]}}}}):
            self.assertTrue(leak_markers(hidden))
            self.assertFalse(leak_markers(strip_hidden(hidden)))
