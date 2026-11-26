"""Physical covered teacher and independent value tower, not GPU qualification."""
from copy import deepcopy
import unittest
import numpy as np
import torch
from app.modules.card_game.rl.batched_search.domain_soa_h3 import DomainSoAH3
from app.modules.card_game.rl.batched_search.device_inference import DeviceSevenInference
from app.modules.card_game.rl.batched_search.device_value_only_inference import DeviceValueOnlyInference
from app.modules.card_game.rl.batched_duel.domain_soa_deadline import DomainDeviceDeadline
from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch,DomainSoALayout
from app.modules.card_game.rl.batched_duel.domain_soa_codec import _decode_row
from app.modules.card_game.engine.duel_v2.flow import acting_side
from app.modules.card_game.engine.duel_v2.simulation import simulate_action
from app.modules.card_game.engine.duel_v2.projection import observe
from app.modules.card_game.rl.covered_rollout import rule_decision
from app.modules.card_game.rl import seven_lineup as seven
from app.modules.card_game.rl.batched_search.inference import BatchedInference,InferenceRequest
from tests.test_duel_v2_device_inference import SyntheticSevenModel
from tests.test_duel_v2_domain_soa_execution_semantics import differences
from tests.test_duel_v2_domain_soa_sampler import _state
from tests.test_duel_v2_domain_soa_character_queries import pack

class PhysicalH3Test(unittest.TestCase):
 device="cpu"
 def test_teacher_states_auxiliary_values_terminal_and_shared_deadline(self):
  torch.set_num_threads(1);keys=('starter','weave-rush','quick-rush','zhenhong','murk','collapse','lingke')
  worlds=[_state(880+i,first='a',a=k,b=keys[(i+1)%7])for i,k in enumerate(keys)]
  terminal=_state(888);terminal.update(phase='finished',winner='draw',reason='player_hp');worlds.append(terminal)
  quick=_state(889,a='starter');quick['sides']['b']['hp']=1;quick['sides']['b']['discard'].extend(quick['sides']['b']['hand']);quick['sides']['b']['hand']=[];quick['sides']['b']['front']=None;quick['sides']['a']['discard'].extend(quick['sides']['a']['hand']);quick['sides']['a']['hand']=[];quick['sides']['a']['ap']=0;quick['sides']['a']['normal_attack_available']=False;quick['sides']['a']['ultimate_available']=False;quick['sides']['b']['discard'].extend(quick['sides']['b']['deck']);quick['sides']['b']['deck']=[];worlds.append(quick)
  n=len(worlds);layout,fields=pack(worlds,device=self.device,layout=DomainSoALayout(game_capacity=n,max_characters=16))
  main={'fixture':SyntheticSevenModel('fixture',17)};aux={'fixture':SyntheticSevenModel('fixture',99)}
  primary=DeviceSevenInference(main,device=self.device,allow_cpu_reference=self.device=='cpu',max_rows=n)
  auxiliary=DeviceValueOnlyInference(aux,device=self.device,allow_cpu_reference=self.device=='cpu',max_rows=n)
  deadline=DomainDeviceDeadline(n,device=self.device,allow_cpu_oracle=self.device=='cpu')
  try:
   with DomainSoAH3(layout,auxiliary,primary,device=self.device,allow_cpu_oracle=self.device=='cpu',allow_unqualified=True)as h3:
    active=torch.ones(n,dtype=torch.bool,device=self.device);viewer=torch.zeros(n,dtype=torch.int64,device=self.device);model=torch.zeros(n,dtype=torch.int64,device=self.device)
    p=h3.prepare(fields,viewer,model,active,deadline);deadline.start(torch.full((n,),600000.,dtype=torch.float64,device=self.device));out=h3.launch(p)
    self.assertFalse(out['errors'].any(),out['errors'].tolist());self.assertTrue(out['complete'].all());self.assertEqual(int(out['steps'][7]),0);self.assertLess(int(out['steps'][8]),3)
    batch=DomainSoABatch(layout=layout,**{k:v.cpu().numpy()for k,v in out['fields'].items()});runner=BatchedInference(aux,device='cpu')
    for row,world in enumerate(worlds):
     gold=deepcopy(world);steps=0
     for unused in range(3):
      if gold['phase']=='finished':break
      actor=acting_side(gold);legal,index=rule_decision(gold,actor);gold=simulate_action(gold,actor,legal[index]);steps+=1
     gold.pop('events',None);gold.pop('_public_board',None)
     with self.subTest(row=row):
      self.assertEqual(differences(gold,_decode_row(batch,row)),[]);self.assertEqual(int(out['steps'][row]),steps)
      if gold['phase']=='finished':value=0. if gold['winner']=='draw'else 1. if gold['winner']=='a'else -1.
      else:
       x,_=seven.encode(observe(gold,'a',include_previews=False),[])
       expected=runner.predict([InferenceRequest(row,'fixture',aux['fixture'].version,x,np.empty((0,1576),dtype=np.float32),acting_side(gold)=='a',True)])[0]
       value=float(expected.wdl[2]-expected.wdl[0])
      self.assertAlmostEqual(float(out['values'][row]),value,places=5)
    deadline.start(torch.zeros(n,dtype=torch.float64,device=self.device));expired=h3.launch(p)
    self.assertTrue(torch.all(expired['errors'][:7]==128));self.assertFalse(expired['complete'][:7].any());self.assertTrue(bool(expired['complete'][7]))
  finally:deadline.close();auxiliary.close();primary.close()

if __name__=='__main__':unittest.main()
