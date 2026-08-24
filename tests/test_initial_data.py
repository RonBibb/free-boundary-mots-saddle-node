import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from bhps.initial_data import solve

class InitialDataTests(unittest.TestCase):
    def test_ads(self):
        s=solve(0,nz=9,nr=13); self.assertTrue(s["converged"]); self.assertLess(s["max_abs_residual"],1e-10)
    def test_donor(self):
        s=solve(.1,nz=9,nr=13); self.assertTrue(s["converged"]); self.assertGreater(np.min(s["psi"]),0)

if __name__=="__main__":unittest.main()
