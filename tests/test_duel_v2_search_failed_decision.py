"""A rule timeout cannot become a spurious information-set mismatch."""
from dataclasses import replace
import unittest
from tests import test_duel_v2_device_search as fixture
from app.modules.card_game.rl.batched_search.device_search import RULE_ERROR,KEY_ERROR


class FailedDecisionTest(unittest.TestCase):
    def _run(self,device):
        import torch
        helper=fixture.DeviceSearchTest()
        search,rules=helper.engine(device=device)
        original=rules.decision_device
        def timeout(states,observers):
            decision=original(states,observers)
            failed=states.active & (rules.cursor>0) & (rules.data[states.slots,0]==0)
            return replace(decision,errors=torch.where(failed,108,decision.errors),
                counts=torch.where(failed,0,decision.counts),
                valid=decision.valid & ~failed[:,None])
        rules.decision_device=timeout
        class CheckedKeys(fixture.InjectiveFixtureKeys):
            def __init__(self):self.hashed_empty=torch.zeros(rules.rows,dtype=torch.bool,device=rules.device)
            def hash(self,x,c,counts,active):
                self.hashed_empty.logical_or_(active & (counts==0))
                return super().hash(x,c,counts,active)
        keys=CheckedKeys();search.information_keys=keys
        helper.initialize(search,rules)
        search.tick();search.tick()
        self.assertTrue(bool((search.errors & RULE_ERROR).all()))
        self.assertFalse(bool((search.errors & KEY_ERROR).any()))
        self.assertFalse(bool(keys.hashed_empty.any()))

    def test_failed_traversal_keeps_rule_error_without_hashing_empty_candidates(self):
        self._run('cpu')

    def test_actual_cuda_failed_traversal_does_not_add_key_error(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        self._run('cuda')


if __name__=='__main__':unittest.main()
