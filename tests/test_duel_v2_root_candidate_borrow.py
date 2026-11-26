"""Static root-site binding and independent mutable-current snapshots."""
from types import SimpleNamespace
import unittest
from tests import test_duel_v2_device_search as fixture
from app.modules.card_game.rl.batched_search.device_search import DeviceGumbelSearch
from app.modules.card_game.rl.batched_search.device_frozen_rules import DeviceFrozenRules

class RootCandidateBorrowTest(unittest.TestCase):
    def test_borrow_only_root_tokens_and_clone_current(self):
        import torch
        rules=fixture.TensorTreeFixture(2);decision=rules.decision_device(rules.roots(),torch.zeros(2,dtype=torch.int64))
        facade=object.__new__(DeviceFrozenRules);facade.closed=False
        facade.t=SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda:True))
        first=SimpleNamespace(candidates=decision.candidates);child=SimpleNamespace(candidates=decision.candidates.clone())
        queue=SimpleNamespace(index=1,plans=[first,child]);facade._queues={'decision':queue}
        search=object.__new__(DeviceGumbelSearch);search.rules=facade;search.borrow_root_candidates=True
        root=search._clone_decision(decision,root=True);current=search._clone_decision(decision)
        self.assertIs(root.candidates,decision.candidates);self.assertIsNot(current.candidates,decision.candidates)
        self.assertIsNot(root.states.slots,decision.states.slots)
        self.assertIsNot(root.action_slots,decision.action_slots)
        original=root.candidates.clone();current.candidates.add_(7);child.candidates.zero_()
        self.assertTrue(torch.equal(root.candidates,original))
        for invalid in (0,2):
            queue.index=invalid
            with self.assertRaises(ValueError):facade.borrow_root_candidates_device(decision)
        queue.index=1;child.candidates=first.candidates
        with self.assertRaises(ValueError):facade.borrow_root_candidates_device(decision)
        facade.closed=True
        with self.assertRaises(RuntimeError):facade.borrow_root_candidates_device(decision)

if __name__=='__main__':unittest.main()
