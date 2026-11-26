import unittest
from unittest.mock import patch
import torch
from app.modules.card_game.rl.batched_duel.domain_soa_zones import (
    move_card,INVALID_POSITION,INVALID_REFERENCE,CAPACITY,
)


def fixture(b,device='cpu'):
    cards=torch.full((b,2,4,16),-1,dtype=torch.int32,device=device)
    counts=torch.zeros((b,2,4),dtype=torch.int32,device=device)
    cards[:,0,0,:3]=torch.tensor([0,1,2],device=device)
    cards[:,0,1,:4]=torch.tensor([3,4,5,6],device=device)
    counts[:,0,0]=3;counts[:,0,1]=4
    side=torch.zeros(b,dtype=torch.int64,device=device)
    source=torch.ones_like(side);position=torch.zeros_like(side)
    dest=torch.zeros_like(side);insert=torch.full_like(side,3)
    active=torch.ones(b,dtype=torch.bool,device=device)
    return cards,counts,side,source,position,dest,insert,active


class ZoneMoveTest(unittest.TestCase):
    def test_ordered_draw_and_same_zone_reorder_preserve_instances(self):
        args=list(fixture(3));args[3][1]=0;args[4][1]=1;args[6][1]=2;args[7][2]=False
        before=[v.clone()for v in args]
        out=move_card(*args,max_cards=32)
        self.assertEqual(out.cards[0,0,0,:4].tolist(),[0,1,2,3])
        self.assertEqual(out.cards[0,0,1,:3].tolist(),[4,5,6])
        self.assertEqual(out.cards[1,0,0,:3].tolist(),[0,2,1])
        self.assertEqual(out.counts[1,0,0].item(),3)
        self.assertTrue(torch.equal(out.cards[2],args[0][2]))
        self.assertTrue(all(torch.equal(a,b)for a,b in zip(args,before)))
        self.assertEqual(out.errors.tolist(),[0,0,0])

    def test_capacity_duplicate_and_bad_position_roll_back(self):
        args=list(fixture(3));args[0][0,0,0,:10]=torch.arange(10);args[1][0,0,0]=10
        args[0][0,0,1,:4]=torch.arange(10,14)
        args[0][1,1,3,0]=3;args[1][1,1,3]=1
        args[4][2]=4
        out=move_card(*args,max_cards=32)
        self.assertEqual(out.errors.tolist(),[CAPACITY,INVALID_REFERENCE,INVALID_POSITION])
        self.assertTrue(torch.equal(out.cards,args[0]));self.assertTrue(torch.equal(out.counts,args[1]))

    def test_hot_path_does_not_extract_device_values(self):
        args=fixture(4)
        with patch.object(torch.Tensor,'item',side_effect=AssertionError('host read')),\
             patch.object(torch.Tensor,'cpu',side_effect=AssertionError('host read')),\
             patch.object(torch.Tensor,'numpy',side_effect=AssertionError('host read')),\
             patch.object(torch.Tensor,'tolist',side_effect=AssertionError('host read')):
            move_card(*args,max_cards=32)

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_rows_and_graph_match_cpu(self):
        cpu=fixture(400);gpu=tuple(v.cuda()for v in cpu)
        expected=move_card(*cpu,max_cards=32)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):move_card(*gpu,max_cards=32)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=move_card(*gpu,max_cards=32)
        for _ in range(3):graph.replay()
        torch.cuda.synchronize()
        for name in('cards','counts','moved','errors'):
            self.assertTrue(torch.equal(getattr(out,name).cpu(),getattr(expected,name)))
