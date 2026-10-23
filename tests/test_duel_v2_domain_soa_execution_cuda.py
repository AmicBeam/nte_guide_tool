"""Actual CUDA graph replay of physical actions, separate from CPU oracle checks."""
from copy import deepcopy
from unittest.mock import patch
from contextlib import ExitStack
import unittest
import torch
from app.modules.card_game.rl.batched_duel.domain_soa_execution import DomainSoAExecution
from app.modules.card_game.rl.batched_duel.domain_soa_codec import _decode_row
from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch,DomainSoALayout
from app.modules.card_game.engine.duel_v2.flow import apply_action,legal_actions,acting_side
from tests.test_duel_v2_domain_soa_execution_semantics import fixtures,differences
from tests.test_duel_v2_domain_soa_character_queries import pack

class PhysicalCudaGraphTest(unittest.TestCase):
 def test_graph_replays_play_then_choice_without_host_rule_callbacks(self):
  if not torch.cuda.is_available():self.skipTest('Actual CUDA required; CPU oracle is not acceptance')
  torch.backends.cuda.matmul.allow_tf32=False
  cases=fixtures()[0];chosen=[next(c for c in cases if c[0]==card)for card in ('P03','N03')]
  worlds=[deepcopy(c[1])for c in chosen];layout,host=pack(worlds,layout=DomainSoALayout(game_capacity=2,max_characters=16))
  fields={k:v.cuda()for k,v in host.items()};side=torch.zeros(2,dtype=torch.int64,device='cuda');picks=torch.tensor([c[2]for c in chosen],device='cuda');active=torch.ones(2,dtype=torch.bool,device='cuda')
  expected=[apply_action(w,'a',c[3])for w,c in zip(worlds,chosen)]
  with DomainSoAExecution(layout,device='cuda',allow_unqualified=True)as core:
   prepared=core.prepare(fields,side,picks,active);warm=torch.cuda.Stream();warm.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(warm):
    for unused in range(2):core.launch(prepared)
   torch.cuda.current_stream().wait_stream(warm);graph=torch.cuda.CUDAGraph()
   with ExitStack()as stack:
    for name in ('cpu','numpy','item','tolist','__bool__'):stack.enter_context(patch.object(torch.Tensor,name,side_effect=AssertionError('Hot-path host conversion: '+name)))
    stack.enter_context(patch('app.modules.card_game.engine.duel_v2.flow.apply_action',side_effect=AssertionError('Host rule fallback')))
    with torch.cuda.graph(graph):captured=core.launch(prepared)
    graph.replay()
   torch.cuda.synchronize()
   for replay in range(2):
    self.assertFalse(captured['errors'].any(),captured['errors'].tolist())
    batch=DomainSoABatch(layout=layout,**{k:v.cpu().numpy()for k,v in captured['fields'].items()})
    for row,gold in enumerate(expected):
     clean={k:v for k,v in gold.items()if k not in ('events','_public_board')};self.assertEqual(differences(clean,_decode_row(batch,row)),[])
    if replay:break
    self.assertEqual(expected[0]['phase'],'choice')
    next_worlds=[];next_picks=[];next_sides=[];next_expected=[]
    for gold in expected:
     gold.pop('_public_board',None);gold['events']=[];actor=acting_side(gold);legal=[a for a in legal_actions(gold,actor)if a['type']!='concede']
     next_worlds.append(gold);next_picks.append(0);next_sides.append(0 if actor=='a'else 1);next_expected.append(apply_action(gold,actor,legal[0]))
    unused,new_fields=pack(next_worlds,layout=layout)
    for key,value in fields.items():value.copy_(new_fields[key].to('cuda'))
    picks.copy_(torch.tensor(next_picks,device='cuda'));side.copy_(torch.tensor(next_sides,device='cuda'));expected=next_expected
    with ExitStack()as stack:
     for name in ('cpu','numpy','item','tolist','__bool__'):stack.enter_context(patch.object(torch.Tensor,name,side_effect=AssertionError('Replay host conversion')))
     stack.enter_context(patch('app.modules.card_game.engine.duel_v2.flow.apply_action',side_effect=AssertionError('Replay host rules')))
     graph.replay()
    torch.cuda.synchronize()



from tests import test_duel_v2_domain_soa_execution_semantics as semantics

class PhysicalCudaSemanticsTest(semantics.NativeExecutionSemanticsTest):
 device='cuda'
 @classmethod
 def setUpClass(cls):
  if not torch.cuda.is_available():raise unittest.SkipTest('Actual CUDA required')
  torch.backends.cuda.matmul.allow_tf32=False

if __name__=='__main__':unittest.main()
