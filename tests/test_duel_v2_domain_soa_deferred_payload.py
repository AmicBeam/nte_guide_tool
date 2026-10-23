"""Serialized deferred support identity and attribute reach the physical rules."""
import unittest
import torch
from app.modules.card_game.engine.duel_v2.flow import apply_action,legal_actions
from app.modules.card_game.rl.batched_duel.domain_soa_execution import DomainSoAExecution
from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch
from app.modules.card_game.rl.batched_duel.domain_soa_codec import _decode_row
from tests.test_duel_v2_domain_soa_character_queries import state,pack
from tests.test_duel_v2_domain_soa_execution_semantics import differences

class DeferredPayloadTest(unittest.TestCase):
 def test_deferred_sync_uses_attribute_and_borrowed_entity_identity(self):
  torch.set_num_threads(1);worlds=[]
  for stale in (False,True):
   world=state(772,a='lingke',b='starter');world['active_side']='a';own=world['sides']['a'];actor='lingke';borrowed=next(c for c in own['order']if c!=actor)
   world['deferred_actions']=[dict(side='a',actor=actor,entity_id=own['characters'][actor]['entity_id'],kind='lingke_sync',attribute='咒',support_actor=dict(character_id=borrowed,entity_id='e999999'if stale else own['characters'][borrowed]['entity_id']))]
   worlds.append(world)
  layout,fields=pack(worlds);sides=torch.zeros(2,dtype=torch.int64);picks=torch.zeros(2,dtype=torch.int64);active=torch.ones(2,dtype=torch.bool)
  with DomainSoAExecution(layout,device='cpu',allow_cpu_oracle=True,allow_unqualified=True)as core:
   out=core.launch(core.prepare(fields,sides,picks,active));self.assertFalse(out['errors'].any(),out['errors'].tolist())
   batch=DomainSoABatch(layout=layout,**{k:v.numpy()for k,v in out['fields'].items()})
   for row,world in enumerate(worlds):
    gold=apply_action(world,'a',{'type':'end_turn'});gold.pop('events',None);gold.pop('_public_board',None)
    with self.subTest(stale=row):self.assertEqual(differences(gold,_decode_row(batch,row)),[])

if __name__=='__main__':unittest.main()
