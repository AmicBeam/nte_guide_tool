from copy import deepcopy
import unittest
from app.modules.card_game.engine.duel_v2 import new_game,observe,choose_action
from app.modules.card_game.engine.duel_v2.projection import preview
from app.modules.card_game.engine.duel_v2.state import hero,card_instance

class PreviewPrivacyTest(unittest.TestCase):
    def state(self,card_id,revealed=False):
        s=new_game(seed=987,first_side='a',skip_mulligan=True)
        s['sides']['b']['ap']=1;s['sides']['b']['front']='zero'
        hero(s,'b','nanali')['base_attack']=9
        card=card_instance(s,card_id,'b');s['sides']['b']['hand']=[card]
        if revealed:s['sides']['b']['revealed_ids']=[card['instance_id']]
        return s
    def test_hidden_responses_do_not_change_preview_or_rule_choice(self):
        one=self.state('N02');two=self.state('Z05');before=deepcopy(one)
        self.assertEqual(observe(one,'a',include_previews=False)['sides']['b']['hand'],[{'hidden':True}])
        action={'type':'attack','character_id':'zero'}
        self.assertEqual(preview(one,'a',action),preview(two,'a',action))
        self.assertEqual(choose_action(one,'a'),choose_action(two,'a'))
        self.assertEqual(one,before)
    def test_revealed_response_remains_predictable(self):
        action={'type':'attack','character_id':'zero'}
        hidden=preview(self.state('N02'),'a',action)
        known=preview(self.state('N02',True),'a',action)
        self.assertEqual(hidden['攻击目标'],'零');self.assertEqual(hidden['counter'],2)
        self.assertEqual(known['攻击目标'],'娜娜莉');self.assertEqual(known['counter'],9)
