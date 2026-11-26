"""Full fixed-budget FSM against physical rules; numerical weights are synthetic."""
import unittest
import torch
from app.modules.card_game.rl.batched_search.domain_soa_rule_adapter import DomainSoARuleAdapter,make_gumbel_search
from app.modules.card_game.rl.batched_search.device_inference import DeviceSevenInference
from app.modules.card_game.rl.batched_search.device_value_only_inference import DeviceValueOnlyInference
from app.modules.card_game.rl.batched_search.device_statistics import DeviceStatisticsPool
from app.modules.card_game.rl.batched_search.device_info_key import DeviceInformationKey
from tests.test_duel_v2_device_inference import SyntheticSevenModel
from tests.test_duel_v2_domain_soa_character_queries import pack
from tests.test_duel_v2_domain_soa_sampler import _state

class PhysicalSearch32Test(unittest.TestCase):
 def test_full_32_simulations_with_three_teacher_steps_and_independent_auxiliary(self):
  torch.set_num_threads(1);world=_state(420,a='starter',b='starter');layout,fields=pack([world]);a=64
  rules=DomainSoARuleAdapter(layout,rows=1,max_actions=a,device='cpu',allow_cpu_oracle=True,allow_unqualified=True)
  main=DeviceSevenInference({'starter':SyntheticSevenModel('starter',17)},device='cpu',allow_cpu_reference=True,max_rows=1)
  auxiliary=DeviceValueOnlyInference({'starter':SyntheticSevenModel('starter',99)},device='cpu',allow_cpu_reference=True,max_rows=1)
  statistics=DeviceStatisticsPool(33,a,device='cpu',q_scale='mctx_default',max_batch=1,max_path_len=10,allow_cpu_reference=True)
  keys=DeviceInformationKey(device='cpu',max_rows=1,allow_cpu_reference=True)
  search=make_gumbel_search(rules,main,auxiliary,statistics,keys,budget_seconds=600,allow_cpu_oracle=True)
  try:
   roots=rules.pool.put_device(fields,torch.zeros(1,dtype=torch.int64),torch.ones(1,dtype=torch.bool));ids=torch.zeros((1,2),dtype=torch.int64)
   search.initialize(roots,torch.zeros(1,dtype=torch.int64),torch.tensor([[37,0,0,0,1]],dtype=torch.int64),ids,ids)
   ticks=0
   while bool(search.unfinished_device.any())and ticks<2500:search.tick();ticks+=1
   result=search.result_device()
   self.assertFalse(result.errors.any(),result.errors.tolist());self.assertTrue(result.complete.all())
   self.assertEqual(result.completed.tolist(),[32]);self.assertEqual(result.requested.tolist(),[32]);self.assertEqual(int(result.visits.sum()),32);self.assertEqual(result.node_count.tolist(),[33])
   self.assertEqual(int(rules.pool.live.sum()),1);self.assertEqual(int(rules.action_pool.live.sum()),1)
   self.assertFalse(rules.gpu_qualified)
  finally:search.close();keys.close();auxiliary.close();main.close();rules.close()

if __name__=='__main__':unittest.main()
