from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
from collections import Counter
from app.modules.card_game.rl.five_full_cycle import shortlist, tactical_gate
from app.modules.card_game.rl.build_acceptance import build_identity, paired_result, delivery_decision
from app.modules.card_game.content.duel_v2 import STARTER_DECKS, CARDS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action
from app.modules.card_game.rl.fixed_lineup_training import track_immune_window


class FiveCycleTest(unittest.TestCase):
    def build(self): return deepcopy(next(d for d in STARTER_DECKS if d['id']=='zhenhong'))

    def test_ten_character_candidates_preserve_baseline_and_lineup(self):
        baseline=self.build();parent=deepcopy(baseline);parent['card_ids'].remove('I03');parent['card_ids'].append('I02')
        rows=shortlist(parent,baseline,311,16)
        self.assertEqual(len(rows),16)
        self.assertEqual(len({build_identity(b) for b in rows}),len(rows))
        self.assertIn(build_identity(baseline),{build_identity(b) for b in rows})
        self.assertIn(build_identity(parent),{build_identity(b) for b in rows})
        for b in rows:
            self.assertEqual(b['character_ids'],baseline['character_ids'])
            self.assertEqual(Counter(CARDS[c]['character_id'] for c in b['card_ids']),Counter({cid:8 for cid in b['character_ids']}))
            self.assertLessEqual(max(Counter(b['card_ids']).values()),2)

    def test_empty_front_ignored_but_actual_ally_loss_recorded(self):
        b=self.build();s=new_game(seed=4,skip_mulligan=True,decks={'a':b,'b':b})
        s['sides']['a']['characters']['zhenhong']['flags']['damage_immunity_until']=s['sides']['a']['turn_count']+1
        metrics=Counter();risks=[];after=deepcopy(s)
        track_immune_window(s,after,{'type':'end_turn'},'a',metrics,risks)
        self.assertEqual(risks,[])
        self.assertFalse(any('empty' in k for k in metrics))
        after['sides']['a']['characters']['zero']['hp']=0
        track_immune_window(s,after,{'type':'attack','character_id':'zero'},'a',metrics,risks)
        self.assertEqual(metrics['allies_down_while_immune'],1)
        self.assertEqual(risks[0]['down'],['zero'])

    def test_immune_tactic_rejects_dying_ally_not_ending_turn(self):
        class Safe:
            def __init__(self,*args): pass
            def select_public_action(self,view,actions):
                if view['sides']['b']['hp']==1:
                    return next(i for i,a in enumerate(actions) if a['type']=='attack')
                return next(i for i,a in enumerate(actions) if a['type']=='end_turn')
        class Unsafe(Safe):
            def select_public_action(self,view,actions):
                if view['sides']['b']['hp']==1:return super().select_public_action(view,actions)
                return next(i for i,a in enumerate(actions) if a.get('character_id')=='zero' and a['type']=='attack')
        with patch('app.modules.card_game.rl.five_full_cycle.FixedModel',Safe):
            result=tactical_gate('unused','zhenhong',self.build())
            self.assertTrue(result['passed'],result)
        with patch('app.modules.card_game.rl.five_full_cycle.FixedModel',Unsafe):
            result=tactical_gate('unused','zhenhong',self.build())
            self.assertFalse(result['passed'])
            self.assertEqual(result['cases'][-1]['case'],'immune_no_free_sacrifice')
