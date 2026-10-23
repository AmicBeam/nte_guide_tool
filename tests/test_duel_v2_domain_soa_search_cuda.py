"""Actual CUDA teacher and opening sample-world acceptance; never CPU fallback."""
import unittest
import torch
from tests import test_duel_v2_domain_soa_h3 as h3
from tests import test_duel_v2_domain_soa_mulligan_sampling as opening

class PhysicalCudaH3Test(h3.PhysicalH3Test):
 device='cuda'
 @classmethod
 def setUpClass(cls):
  if not torch.cuda.is_available():raise unittest.SkipTest('Actual CUDA required')
  torch.backends.cuda.matmul.allow_tf32=False

class PhysicalCudaMulliganSamplingTest(opening.MulliganSearchSamplingTest):
 device='cuda'
 @classmethod
 def setUpClass(cls):
  if not torch.cuda.is_available():raise unittest.SkipTest('Actual CUDA required')
  torch.backends.cuda.matmul.allow_tf32=False

if __name__=='__main__':unittest.main()
