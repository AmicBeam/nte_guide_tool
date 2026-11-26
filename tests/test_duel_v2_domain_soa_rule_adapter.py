"""Actual typed protocol storage, features, seeds and physical transition oracle."""
import unittest
import torch
from app.modules.card_game.rl.batched_search.domain_soa_rule_adapter import DomainSoARuleAdapter
from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch
from app.modules.card_game.rl.batched_duel.domain_soa_codec import _decode_row
from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
from app.modules.card_game.engine.duel_v2.flow import legal_actions,apply_action,acting_side
from app.modules.card_game.engine.duel_v2.projection import observe
from app.modules.card_game.rl import seven_lineup as seven
from tests.test_duel_v2_domain_soa_character_queries import state,pack

class DomainRuleAdapterTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.world=state(721);cls.L,cls.fields=pack([cls.world]);cls.core=DomainSoARuleAdapter(cls.L,rows=1,max_actions=64,device='cpu',allow_cpu_oracle=True,allow_unqualified=True)
 @classmethod
 def tearDownClass(cls):cls.core.close()
 def put(self):return self.core.pool.put_device(self.fields,torch.tensor([71]),torch.ones(1,dtype=torch.bool))
 def test_metadata_decision_and_original_features(self):
  root=self.put();side=acting_side(self.world);viewer=torch.tensor([0 if side=='a'else 1]);meta=self.core.metadata_device(root)
  self.assertEqual(int(meta.actors[0]),int(viewer[0]));self.assertFalse(meta.finished.any());self.assertFalse(meta.errors.any())
  decision=self.core.decision_device(root,viewer);legal=[a for a in legal_actions(self.world,side)if a['type']!='concede'];x,c=seven.encode(observe(self.world,side,include_previews=False),legal)
  self.assertFalse(decision.errors.any());self.assertEqual(int(decision.counts[0]),len(legal))
  self.assertTrue(torch.equal(decision.observations[0],torch.tensor(x)));encoded,errors=decision.candidates.encode_slice_device(0,1,root.active);self.assertFalse(errors.any());self.assertTrue(torch.equal(encoded.candidates[0,:len(legal)],torch.tensor(c)))
  self.core.action_pool.release_device(DeviceStateBatch(self.core.action_pool,decision.action_slots,decision.action_generations,root.owners,root.active,torch.zeros_like(root.errors)))
  self.core.release_device(root)
 def test_physical_end_turn_and_foreign_action_rejection(self):
  root=self.put();side=acting_side(self.world);viewer=torch.tensor([0 if side=='a'else 1]);decision=self.core.decision_device(root,viewer)
  advanced=self.core.step_device(root,viewer,decision,torch.zeros(1,dtype=torch.int64),physical=True)
  self.assertFalse(advanced.errors.any(),advanced.errors.tolist())
  fields=self.core.pool.gather_device(advanced);actual=_decode_row(DomainSoABatch(layout=self.L,**{k:v.contiguous().numpy()for k,v in fields.items()}),0)
  expected=apply_action(self.world,side,{'type':'end_turn'});expected.pop('events',None);expected.pop('_public_board',None);self.assertEqual(actual,expected)
  foreign=DeviceStateBatch(root.pool,root.slots,root.generations,root.owners+1,root.active,torch.zeros_like(root.errors));rejected=self.core.step_device(foreign,viewer,decision,torch.zeros(1,dtype=torch.int64))
  self.assertTrue(rejected.errors.any());self.assertFalse(rejected.active.any())
  self.core.action_pool.release_device(DeviceStateBatch(self.core.action_pool,decision.action_slots,decision.action_generations,root.owners,root.active,torch.zeros_like(root.errors)))
  self.core.release_device(advanced);self.core.release_device(root)

class AuxiliaryBoundaryTest(unittest.TestCase):
 def test_shared_auxiliary_storage_is_rejected_before_search_setup(self):
  from types import SimpleNamespace
  from app.modules.card_game.rl.batched_search.domain_soa_rule_adapter import make_gumbel_search
  from app.modules.card_game.rl.batched_search.device_value_only_inference import DeviceValueOnlyInference
  rules=object.__new__(DomainSoARuleAdapter);weights={'value':torch.zeros(1)};main=SimpleNamespace(weights=weights);auxiliary=object.__new__(DeviceValueOnlyInference);auxiliary.weights=weights
  with self.assertRaisesRegex(ValueError,'alias primary'):make_gumbel_search(rules,main,auxiliary,None,None,allow_cpu_oracle=True)
  auxiliary.weights={'value':weights['value'].clone()}
  with self.assertRaisesRegex(ValueError,'Bounded shared decision budget'):make_gumbel_search(rules,main,auxiliary,None,None,allow_cpu_oracle=True,budget_seconds=0)

if __name__=='__main__':unittest.main()
