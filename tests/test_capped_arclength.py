import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.capped_arclength import MetricFamily,pseudo_arclength_step
from bhps.capped_continuation import solve_capped_profile


class CappedArclengthTests(unittest.TestCase):
    def setUp(self):
        self.z=np.linspace(.2,2,49);self.r=np.linspace(0,3,81)
        distance2=self.r[None,:]**2+(self.z[:,None]-self.z[-1])**2
        self.amplitudes=np.linspace(.25,.81,8)
        self.psi=np.array([1+a/np.maximum(distance2,1e-8) for a in self.amplitudes])

    def test_metric_family_is_exact_for_linear_manufactured_family(self):
        family=MetricFamily(self.amplitudes,self.z,self.r,self.psi)
        z=np.array([1.1,1.4]);r=np.array([.4,.7]);a=.43
        value,_,_=family.evaluate(a,z,r)
        expected=1+a/(r**2+(z-self.z[-1])**2)
        np.testing.assert_allclose(value,expected,rtol=2e-4)

    def test_step_advances_manufactured_tangherlini_family(self):
        family=MetricFamily(self.amplitudes,self.z,self.r,self.psi)
        profiles=[]
        for amplitude in (.36,.49):
            solved=solve_capped_profile(self.z,self.r,1+amplitude/np.maximum(
                self.r[None,:]**2+(self.z[:,None]-self.z[-1])**2,1e-8
            ),np.sqrt(amplitude),tolerance=2e-6)
            solved["amplitude"]=amplitude;profiles.append(solved)
        advanced=pseudo_arclength_step(family,*profiles,step=.08,tolerance=2e-5,nodes=100)
        self.assertTrue(advanced["converged"],advanced.get("message"))
        self.assertGreater(advanced["amplitude"],.49)
        self.assertAlmostEqual(advanced["rho_brane"],np.sqrt(advanced["amplitude"]),delta=.025)
        self.assertLess(advanced["arclength_residual"],1e-5)


if __name__=="__main__":unittest.main()
