"""Real monotonic deadline calls, with no refresh extending the parent budget."""
import unittest
import torch
from app.modules.card_game.rl.batched_duel.domain_soa_deadline import DomainDeviceDeadline

class DeadlineTest(unittest.TestCase):
    def check(self,device):
        with DomainDeviceDeadline(4,device=device,allow_cpu_oracle=device=='cpu')as budget:
            active=torch.tensor([True,True,True,False],device=device)
            self.assertFalse(budget.refresh(active).any())
            ms=torch.tensor([60000.,0.,float('nan'),60000.],dtype=torch.float64,device=device)
            budget.start(ms);saved=budget.deadline.clone()
            self.assertEqual(budget.refresh(active).cpu().tolist(),[True,False,False,False])
            self.assertTrue(torch.equal(saved,budget.deadline))
            budget.deadline.fill_(1);self.assertFalse(budget.refresh(active).any())
    def test_actual_cpp_oracle(self):self.check('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_actual_cuda(self):self.check('cuda')
    def test_no_implicit_cpu_fallback(self):
        with self.assertRaises(RuntimeError):DomainDeviceDeadline(1,device='cpu')

if __name__=='__main__':unittest.main()
