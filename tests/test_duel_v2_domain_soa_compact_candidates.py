"""All feature bytes and full information hashes survive compact snapshots."""
import unittest
import torch
from app.modules.card_game.rl import seven_lineup as seven
from app.modules.card_game.engine.duel_v2.projection import observe
from app.modules.card_game.engine.duel_v2.flow import legal_actions
from app.modules.card_game.rl.batched_duel.domain_soa_actions import DomainSoAActions
from app.modules.card_game.rl.batched_duel.domain_soa_observation import PI
from app.modules.card_game.rl.batched_search.domain_soa_compact_candidates import DomainCandidateWorkspace,DomainCompactCandidates
from app.modules.card_game.rl.batched_search.device_info_key import DeviceInformationKey
from tests.test_duel_v2_domain_soa_actions import ActionsTest
from tests.test_duel_v2_domain_soa_character_queries import pack

class CompactDomainCandidatesTest(unittest.TestCase):
 def test_all_columns_order_and_hash_with_reused_short_final_slice(self):
  states=ActionsTest().fixtures()[:10];B=len(states);L,fields=pack(states);viewers=torch.zeros(B,dtype=torch.int64);active=torch.ones(B,dtype=torch.bool);A=800
  with DomainSoAActions(L,device='cpu',allow_cpu_oracle=True)as actions:
   out=actions.project(fields,viewers,active);self.assertFalse(out['errors'].any())
   workspace=DomainCandidateWorkspace(rows=3,actions=A,cards=2*L.max_cards,selection_width=max(4,L.max_choice_cards),device='cpu',allow_cpu_oracle=True)
   try:
    data=dict(cards=out['public_cards'],actions=out['actions'],selected=out['selected'],top=out['top'],bottom=out['bottom'],counts=out['counts'],viewers=viewers,
     turn_counts=fields['player_int_value'][:,0,PI['turn_count']].long(),active=active,prior_errors=out['errors'],mask=torch.arange(A)[None,:]<out['counts'][:,None])
    compact=DomainCompactCandidates(data,workspace);saved=compact.clone()
    data['cards'].fill_(-9.)
    dense=[]
    for start,stop in compact._slices():
     encoded,err=compact.encode_slice_device(start,stop,active[start:stop]);self.assertFalse(err.any());dense.append(encoded.candidates.clone())
    dense=torch.cat(dense)
    for row,st in enumerate(states):
     legal=[a for a in legal_actions(st,'a')if a['type']!='concede'];_,gold=seven.encode(observe(st,'a',include_previews=False),legal)
     self.assertTrue(torch.equal(dense[row,:len(legal)],torch.tensor(gold)),str(row))
    key=DeviceInformationKey(device='cpu',max_rows=B,allow_cpu_reference=True)
    try:
     actual=saved.hash_device(key,out['observation'],out['counts'],active);expected=key.hash(out['observation'],dense,out['counts'],active)
     self.assertFalse(actual.errors.any());self.assertTrue(torch.equal(actual.keys,expected.keys))
    finally:key.close()
    self.assertLess(compact.allocated_bytes,dense.numel()*dense.element_size()//2)
   finally:workspace.close()

if __name__=='__main__':unittest.main()
