"""Actual graph tensor residency/clone semantics, no game-rule claims."""
import unittest
import numpy as np
from app.modules.card_game.rl.batched_duel.codec import pack_states,unpack_states
from app.modules.card_game.rl.batched_duel.schema import GraphLayout,CodecCapacityError
from app.modules.card_game.rl.batched_duel.device_state import DeviceStateBatch


class DeviceStateTest(unittest.TestCase):
    def make(self,device):
        shared={'hp':7}
        states=[{'a':shared,'alias':shared,'rng':2**120+5}, {'name':'真红','values':(1,-.5,True,None)}]
        packet=pack_states(states,GraphLayout(4,64,64,512))
        return states,packet,DeviceStateBatch.from_packed(packet,device=device)

    def check_fork(self,device):
        import torch
        states,packet,batch=self.make(device)
        fork=batch.clone_rows(torch.tensor([1,0,0],device=batch.device,dtype=torch.int64))
        event=fork.ready_event()
        if device=='cuda':self.assertIsNotNone(event);event.synchronize();self.assertTrue(event.query())
        else:self.assertIsNone(event)
        restored=unpack_states(fork.to_packed())
        self.assertEqual(restored,[states[1],states[0],states[0]])
        self.assertIs(restored[1]['a'],restored[1]['alias'])
        self.assertIsNot(restored[1]['a'],restored[2]['a'])
        for name in ('nodes','edges','payload','counts','roots'):
            self.assertNotEqual(batch._arrays[name].data_ptr(),fork._arrays[name].data_ptr())
        snapshot=fork.to_packed();snapshot.nodes.fill(0)
        self.assertEqual(unpack_states(fork.to_packed()),restored)
        np.testing.assert_array_equal(batch.to_packed().nodes,packet.nodes)
        self.assertEqual(batch.allocated_bytes,packet.allocated_bytes)
        self.assertEqual(fork.allocated_bytes,packet.allocated_bytes//2*3)
        self.assertEqual(batch.clone_rows([]).n_rows,0)

    def test_cpu_oracle_roundtrip_alias_and_fork(self):self.check_fork('cpu')

    def test_index_capacity_byte_limits_and_corrupt_ingress_rejected(self):
        import torch
        _,packet,batch=self.make('cpu')
        for indices,error in (([True],TypeError),([1.5],TypeError),([-1],IndexError),([2],IndexError),([0]*5,CodecCapacityError)):
            with self.subTest(indices=indices),self.assertRaises(error):batch.clone_rows(indices)
        with self.assertRaises(ValueError):batch.clone_rows(torch.tensor([0.],dtype=torch.float32))
        with self.assertRaises(MemoryError):DeviceStateBatch.from_packed(packet,device='cpu',max_device_bytes=1)
        with self.assertRaises(ValueError):DeviceStateBatch.from_packed(packet,device='cpu',max_device_bytes=True)
        packet.roots[0]=999
        with self.assertRaises(ValueError):DeviceStateBatch.from_packed(packet,device='cpu')

    def test_real_cuda_fork_when_available(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA device required')
        self.check_fork('cuda')
        _,_,batch=self.make('cuda')
        with torch.cuda.stream(torch.cuda.Stream()),self.assertRaises(RuntimeError):batch.clone_rows([0])


if __name__=='__main__':unittest.main()
