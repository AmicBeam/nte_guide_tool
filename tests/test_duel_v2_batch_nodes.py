"""Formula and ownership equivalence for batched Gumbel statistics."""
import unittest
from dataclasses import replace
import numpy as np
from app.modules.card_game.rl.batched_search.nodes import BatchedNodePool
from app.modules.card_game.rl.information_search import Node
from app.modules.card_game.rl import recovery_search,search_policy


class BatchNodeTest(unittest.TestCase):
    def test_both_scales_match_numpy_over_full_sequential_budget(self):
        import torch
        torch.set_num_threads(1)
        for mode in ('legacy','natural_wdl'):
            with self.subTest(mode=mode):
                rng=np.random.default_rng(17)
                priors=[search_policy.normalized_exp(rng.normal(size=n)) for n in (1,3,7)]
                values=rng.uniform(-1,1,3)
                pool=BatchedNodePool(5,7,q_scale=mode)
                refs=pool.allocate(priors,values,['a','b','c'])
                nodes=[Node.create(p,v) for p,v in zip(priors,values)]
                schedules=[search_policy.visit_schedule(min(len(p),16,32),32) for p in priors]
                noise=np.zeros((3,7),dtype=np.float64)
                for step in range(32):
                    visits=[schedule[step] for schedule in schedules]
                    selected=pool.select_root(refs,noise,visits).numpy()
                    interior=pool.select_interior(refs).numpy()
                    q=pool.completed_q(refs).numpy();pi=pool.policy(refs).numpy()
                    for i,node in enumerate(nodes):
                        self.assertEqual(selected[i],recovery_search.choose_root(node,np.zeros(len(node.prior)),visits[i],mode))
                        self.assertEqual(interior[i],np.argmax(recovery_search.improved_policy(node,mode)-node.visits/(1+node.visits.sum())))
                        np.testing.assert_allclose(q[i,:len(node.prior)],recovery_search.completed_q(node,mode),atol=1e-12,rtol=1e-12)
                        np.testing.assert_allclose(pi[i,:len(node.prior)],recovery_search.improved_policy(node,mode),atol=1e-12,rtol=1e-12)
                        self.assertTrue(np.all(pi[i,len(node.prior):]==0))
                    rewards=rng.uniform(-1,1,3)
                    pool.backup([[(ref,int(a))] for ref,a in zip(refs,selected)],rewards)
                    for node,action,value in zip(nodes,selected,rewards):
                        node.visits[action]+=1;node.total[action]+=value
                snapshot=pool.snapshot(refs)
                for i,node in enumerate(nodes):
                    np.testing.assert_array_equal(snapshot['visits'][i,:len(node.visits)],node.visits)
                    np.testing.assert_allclose(snapshot['total'][i,:len(node.total)],node.total)

    def test_capacity_and_release_preserve_generation_and_copy_isolation(self):
        pool=BatchedNodePool(2,4)
        prior=np.array([.2,.8]);refs=pool.allocate([prior,prior],[0.,0.],['a','b'])
        prior[0]=1.
        np.testing.assert_array_equal(pool.snapshot([refs[0]])['prior'].numpy()[0,:2],[.2,.8])
        with self.assertRaises(OverflowError):pool.allocate([[1.]],[0.],['c'])
        with self.assertRaises(ValueError):pool.release([refs[0],refs[0]])
        pool.release([refs[0]])
        new=pool.allocate([[1.]],[0.],['c'])[0]
        self.assertEqual(new.slot,refs[0].slot)
        with self.assertRaises(ValueError):pool.policy([refs[0]])
        with self.assertRaises(ValueError):pool.policy([replace(new,slot=False)])
        another=BatchedNodePool(2,4)
        with self.assertRaises(ValueError):another.policy([new])
        self.assertEqual(pool.active_nodes,2)
        self.assertEqual(pool.allocated_bytes,2*4*20+2*16)

    def test_backup_validates_batch_atomically_and_rejects_root_overlap(self):
        pool=BatchedNodePool(3,3)
        a,b,c=pool.allocate([[.5,.5]]*3,[0.]*3,['root-a','root-a','root-b'])
        for paths,values in (([[(a,0)],[(b,1)]],[.1,.2]),([[(a,0),(c,0)]],[.1]),
                             ([[(a,0)],[(c,4)]],[.1,.2]),([[(a,True)]],[.1]),
                             ([[(a,0)]],[float('nan')]),([[(a,0)]],[1.1])):
            with self.subTest(paths=paths),self.assertRaises(ValueError):pool.backup(paths,values)
            self.assertEqual(pool.snapshot([a,b,c])['visits'].sum().item(),0)
        pool.backup([[(a,1),(a,1),(b,0)],[(c,0)]],[.25,-.5])
        np.testing.assert_array_equal(pool.snapshot([a,b,c])['visits'].numpy(),[[0,2,0],[1,0,0],[1,0,0]])
        np.testing.assert_array_equal(pool.snapshot([a,b,c])['total'].numpy(),[[0,.5,0],[.25,0,0],[-.5,0,0]])

    def test_invalid_initialization_and_selection_do_not_allocate_or_truncate(self):
        pool=BatchedNodePool(2,3)
        for priors,values,owners in (([[.1,.2]],[0.],['a']),([[float('nan')]],[0.],['a']),
            ([[1.,0.,0.,0.]],[0.],['a']),([[1.]],[float('inf')],['a']),
            ([[1.]],[1.1],['a']),([[1.]],[0.],['']),([[1.]],[],['a'])):
            with self.subTest(priors=priors),self.assertRaises(ValueError):pool.allocate(priors,values,owners)
            self.assertEqual(pool.active_nodes,0)
        ref=pool.allocate([[1.]],[0.],['a'])[0]
        with self.assertRaises(ValueError):pool.select_root([ref],np.zeros((1,3)),[True])
        with self.assertRaises(ValueError):pool.select_root([ref],np.zeros((1,3)),[4])
        with self.assertRaises(ValueError):pool.select_root([ref],np.zeros((1,2)),[0])
        snapshot=pool.snapshot([ref]);snapshot['prior'][0,0]=0
        self.assertEqual(pool.snapshot([ref])['prior'][0,0].item(),1.)

    def test_cuda_has_same_formulas_when_available(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA device required')
        priors=[[.1,.2,.7],[1.]];values=[-.2,.4];owners=['a','b']
        cpu=BatchedNodePool(3,3);gpu=BatchedNodePool(3,3,device='cuda')
        a=cpu.allocate(priors,values,owners);b=gpu.allocate(priors,values,owners)
        cpu.backup([[(a[0],1)],[(a[1],0)]],[.8,-.6])
        gpu.backup([[(b[0],1)],[(b[1],0)]],[.8,-.6])
        np.testing.assert_allclose(gpu.policy(b).cpu().numpy(),cpu.policy(a).numpy(),atol=1e-10,rtol=1e-10)
        other=torch.cuda.Stream()
        with torch.cuda.stream(other),self.assertRaises(RuntimeError):gpu.policy(b)


if __name__=='__main__':unittest.main()
