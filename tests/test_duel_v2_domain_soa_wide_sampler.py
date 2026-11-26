"""Full unsigned 320-bit per-simulation sampler seed, including bit 256."""
from copy import deepcopy
import unittest
import torch
from app.modules.card_game.rl.batched_duel.domain_soa_sampler import DomainSoASampler
from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoALayout,DomainSoABatch
from app.modules.card_game.rl.batched_duel.domain_soa_codec import _decode_row
from tests.test_duel_v2_domain_soa_sampler import _state,_torch_fields,oracle_determinize


def limbs(values):
 return torch.tensor([[((value>>(i*64))&((1<<64)-1))-(1<<64)if((value>>(i*64))&((1<<64)-1))>=(1<<63)else((value>>(i*64))&((1<<64)-1))for i in range(5)]for value in values],dtype=torch.int64)

class WideSamplerTest(unittest.TestCase):
 def test_full_bigint_matches_formal_sampler(self):
  seeds=(0,23,1<<63,(1<<128)+31,(1<<256)+127,(1<<319)+(1<<255)+73)
  states=[_state(70+i,a='collapse',b='quick-rush')for i in range(len(seeds))];L=DomainSoALayout(game_capacity=len(seeds))
  packed,fields=_torch_fields(states,L);viewers=torch.tensor([i%2 for i in range(len(seeds))],dtype=torch.int64)
  with DomainSoASampler(L,device='cpu',allow_cpu_oracle=True,seed_words=5)as sampler:
   out=sampler.determinize(fields,viewers,limbs(seeds),torch.ones(len(seeds),dtype=torch.int32))
  self.assertFalse(out['row_error'].any(),out['row_error'].tolist())
  b=DomainSoABatch(layout=L,**{k:v.numpy()for k,v in out.items()})
  for row,(st,seed)in enumerate(zip(states,seeds)):
   gold=oracle_determinize(deepcopy(st),'a'if row%2==0 else'b',seed);gold.pop('events',None);gold.pop('_public_board',None)
   self.assertEqual(_decode_row(b,row),gold,str(seed))
 def test_sampling_preserves_full_marked_operation_snapshots(self):
  from tests.test_duel_v2_domain_soa_mark_queue import MarkQueueContractTest
  st=MarkQueueContractTest().formal();seed=2**256+99;L=DomainSoALayout(game_capacity=1);packed,fields=_torch_fields([st],L)
  with DomainSoASampler(L,device='cpu',allow_cpu_oracle=True,seed_words=5)as sampler:
   out=sampler.determinize(fields,torch.zeros(1,dtype=torch.int64),limbs([seed]),torch.ones(1,dtype=torch.int32))
  self.assertFalse(out['row_error'].any(),out['row_error'].tolist())
  actual=_decode_row(DomainSoABatch(layout=L,**{k:v.numpy()for k,v in out.items()}),0)
  expected=oracle_determinize(deepcopy(st),'a',seed);expected.pop('events',None);expected.pop('_public_board',None)
  self.assertEqual(actual,expected)
 def test_wide_mode_never_silently_accepts_truncated_seed(self):
  L=DomainSoALayout(game_capacity=1);packed,fields=_torch_fields([_state(7)],L)
  with DomainSoASampler(L,device='cpu',allow_cpu_oracle=True,seed_words=5)as sampler:
   with self.assertRaises(ValueError):sampler.prepare(fields,torch.zeros(1,dtype=torch.int64),torch.zeros(1,dtype=torch.int64),torch.ones(1,dtype=torch.int32))

if __name__=='__main__':unittest.main()
