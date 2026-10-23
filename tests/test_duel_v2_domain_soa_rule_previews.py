"""Anonymous native combat previews against the formal public preview oracle."""
import unittest
import torch
from app.modules.card_game.rl.batched_duel.domain_soa_actions import DomainSoAActions
from app.modules.card_game.rl.batched_duel.domain_soa_rule_previews import DomainRulePreviews
from app.modules.card_game.engine.duel_v2.flow import legal_actions
from app.modules.card_game.engine.duel_v2.projection import preview,prepared_preview_sandbox
from tests.test_duel_v2_domain_soa_execution_semantics import fixtures
from tests.test_duel_v2_domain_soa_character_queries import pack

class PhysicalPreviewTest(unittest.TestCase):
 def test_all_card_cases_and_normal_attacks_use_actual_anonymous_rules(self):
  torch.set_num_threads(1);cases,skipped=fixtures();self.assertEqual(skipped,[])
  layout,fields=pack([c[1]for c in cases]);n=len(cases);viewer=torch.zeros(n,dtype=torch.int64);active=torch.ones(n,dtype=torch.bool)
  with DomainSoAActions(layout,device='cpu',allow_cpu_oracle=True)as actions,DomainRulePreviews(layout,device='cpu',allow_cpu_oracle=True,allow_unqualified=True)as native:
   p=actions.prepare(fields,viewer,active);out=actions.launch(p)
   table={k:out[k]for k in ('counts','errors','payloads','selected','top','bottom','bottom_plural','public_cards')}
   result=native.launch(native.prepare(fields,viewer,active,table))
   self.assertFalse(result['errors'].any(),result['errors'].tolist())
   checked=0
   for row,(card,world,pick,selected)in enumerate(cases):
    legal=[a for a in legal_actions(world,'a')if a['type']!='concede']
    with prepared_preview_sandbox(world,'a'):
     for index,action in enumerate(legal):
      if index!=pick and action['type']!='attack':continue
      gold=preview(world,'a',action);checked+=1
      with self.subTest(card=card,action=action):
       for field,key in [('attack','attack'),('counter','counter'),('attack_count','攻击次数'),('genesis','创生花伤害')]:
        self.assertEqual(float(result[field][row,index]),gold.get(key,0))
   self.assertGreater(checked,400)

if __name__=='__main__':unittest.main()
