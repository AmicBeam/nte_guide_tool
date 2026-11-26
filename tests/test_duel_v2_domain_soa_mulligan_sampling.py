"""Opening sample-world wrapper matches the formal search, including both seats."""
import unittest
import torch
from app.modules.card_game.rl.batched_search.domain_soa_rule_adapter import DomainSoARuleAdapter
from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch
from app.modules.card_game.rl.batched_duel.domain_soa_codec import _decode_row
from app.modules.card_game.rl.information_search import sample_world
from app.modules.card_game.rl import seven_runtime
from app.modules.card_game.engine.duel_v2.flow import apply_action,acting_side
from tests.test_duel_v2_domain_soa_sampler import _state
from tests.test_duel_v2_domain_soa_character_queries import pack
from tests.test_duel_v2_domain_soa_execution_semantics import differences

class MulliganSearchSamplingTest(unittest.TestCase):
 device="cpu"
 def test_opening_proxy_and_restoration_keep_full_visible_information(self):
  torch.set_num_threads(1);worlds=[]
  for first in ('a','b'):
   initial=_state(191,first=first,a='collapse',b='lingke',skip=False);worlds.append(initial)
   worlds.append(apply_action(initial,acting_side(initial),{'type':'mulligan','card_ids':[]}))
  seeds=[37,(1<<256)+11,(1<<319)+37,0];limbs=[]
  for seed in seeds:
   row=[(seed>>(64*i))&((1<<64)-1)for i in range(5)];limbs.append([v if v<(1<<63)else v-(1<<64)for v in row])
  layout,fields=pack(worlds,device=self.device);viewers=torch.tensor([0 if acting_side(w)=='a'else 1 for w in worlds],device=self.device);active=torch.ones(4,dtype=torch.bool,device=self.device)
  with DomainSoARuleAdapter(layout,rows=4,max_actions=64,device=self.device,allow_cpu_oracle=self.device=='cpu',allow_unqualified=True)as rules:
   roots=rules.pool.put_device(fields,torch.arange(4,device=self.device),active);sampled=rules.sample_world_device(roots,viewers,torch.tensor(limbs,dtype=torch.int64,device=self.device))
   self.assertFalse(sampled.errors.any(),sampled.errors.tolist());out=rules.pool.gather_device(sampled);batch=DomainSoABatch(layout=layout,**{k:v.contiguous().cpu().numpy()for k,v in out.items()})
   for row,world in enumerate(worlds):
    expected=sample_world(world,('a','b')[int(viewers[row])],seeds[row],runtime=seven_runtime);expected.pop('events',None);expected.pop('_public_board',None)
    with self.subTest(row=row):self.assertEqual(differences(expected,_decode_row(batch,row)),[])
   rules.release_device(sampled);rules.release_device(roots)

if __name__=='__main__':unittest.main()
