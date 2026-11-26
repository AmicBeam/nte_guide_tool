"""Exact normalization arithmetic and immutable input/parameter tensors."""
import unittest
from app.modules.card_game.rl.batched_search.device_inference import DeviceSevenInference
from app.modules.card_game.rl.batched_search.device_memory_bounded_inference import DeviceMemoryBoundInference

class BoundedNormTest(unittest.TestCase):
    def check(self,device):
        import torch as t
        t.manual_seed(92)
        weights={'n.weight':t.randn(3,41,device=device),'n.bias':t.randn(3,41,device=device),'n.active':(t.rand(3,41,device=device)>.4).float()}
        weights['n.active'][:,0]=1
        old=object.__new__(DeviceSevenInference);old.torch=t;old.weights=weights
        new=object.__new__(DeviceMemoryBoundInference);new.torch=t;new.weights=weights
        before_weights={k:v.clone() for k,v in weights.items()};ids=t.arange(8,device=device)%3
        for shape in ((8,41),(8,16,41)):
            for active in (False,True):
                for scale in (0.,1.,1e10):
                    x=t.randn(shape,device=device)*scale;x_before=x.clone()
                    reference=old._norm(x,ids,'n',active);actual=new._norm(x,ids,'n',active)
                    self.assertTrue(t.equal(reference,actual));self.assertTrue(t.equal(x,x_before))
                    for k,v in weights.items():self.assertTrue(t.equal(v,before_weights[k]))
                x=t.zeros(shape,device=device);x.reshape(-1)[0]=float('nan')
                reference=old._norm(x,ids,'n',active);actual=new._norm(x,ids,'n',active)
                self.assertTrue(t.equal(t.isnan(reference),t.isnan(actual)))
                finite=t.isfinite(reference)&t.isfinite(actual);self.assertTrue(t.equal(reference[finite],actual[finite]))
        with self.assertRaises(RuntimeError):new._norm(t.zeros((8,41),device=device,requires_grad=True),ids,'n',True)
    def test_cpu_norm_exact_and_immutable(self):self.check('cpu')
    def test_actual_cuda_norm_exact_and_immutable(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        self.check('cuda')

if __name__=='__main__':unittest.main()
