"""Whole-view and numerical equality for projection-scoped modifier indexing."""
from copy import deepcopy
from unittest.mock import patch
import unittest
import numpy as np

from app.modules.card_game.rl.batched_duel.gpu_cached_policy_projection import cached_policy_public_legal_decision
from app.modules.card_game.rl.batched_duel.gpu_policy_projection import policy_public_legal_decision
from app.modules.card_game.rl.seven_lineup import encode
from tests.test_duel_v2_policy_projection import trace_states


class CachedProjectionTests(unittest.TestCase):
    def test_all_public_fields_legal_order_and_encoding_both_viewers(self):
        checked=0;phases=set()
        for state in trace_states():
            before=deepcopy(state);phases.add(state['phase'])
            for side in ('a','b'):
                expected=policy_public_legal_decision(state,side)
                actual=cached_policy_public_legal_decision(state,side)
                self.assertEqual(actual,expected)
                legal=[a for a in actual[0] if a['type']!='concede']
                x,c=encode(actual[1],legal);xx,cc=encode(expected[1],legal)
                np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(c,cc)
                checked+=1
            self.assertEqual(state,before)
        self.assertGreater(checked,400);self.assertTrue({'playing','choice','mulligan'}<=phases)

    def test_effect_inventory_is_reused_only_within_one_projection(self):
        from app.modules.card_game.engine.duel_v2 import new_game,modifiers
        state=new_game(seed=910,skip_mulligan=True)
        original=modifiers.all_effects
        with patch.object(modifiers,'all_effects',wraps=original) as counter:
            expected=policy_public_legal_decision(state,'a');baseline=counter.call_count
        with patch.object(modifiers,'all_effects',wraps=original) as counter:
            actual=cached_policy_public_legal_decision(state,'a');cached=counter.call_count
        self.assertEqual(actual,expected);self.assertLess(cached,baseline)
        cid=next(iter(state['sides']['a']['characters']));hero=state['sides']['a']['characters'][cid]
        modifiers.add_effect(hero,'test.panel','attack.panel',3,source_entity_id=hero['entity_id'])
        changed=cached_policy_public_legal_decision(state,'a')
        self.assertEqual(changed,policy_public_legal_decision(state,'a'))
        self.assertNotEqual(changed[1]['sides']['a']['characters'],actual[1]['sides']['a']['characters'])


if __name__=='__main__':unittest.main()
