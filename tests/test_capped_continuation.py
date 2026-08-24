import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from bhps.capped_continuation import fit_fold_normal_form,solve_capped_profile


class CappedContinuationTests(unittest.TestCase):
    def test_manufactured_constant_cap_profile(self):
        z=np.linspace(.2,2,49);r=np.linspace(0,3,81);z_b=z[-1];radius=.6
        distance2=r[None,:]**2+(z[:,None]-z_b)**2
        psi=1+radius**2/np.maximum(distance2,1e-8)
        solved=solve_capped_profile(z,r,psi,radius,tolerance=1e-6)
        self.assertTrue(solved["converged"])
        self.assertAlmostEqual(solved["rho_axis"],radius,delta=.03)

    def test_normal_form_fit_recovers_fold(self):
        fold=2.3;slope=1.7;records=[]
        for amplitude in np.linspace(2.31,2.6,15):
            separation=(slope*(amplitude-fold))**.5
            records.append({"amplitude":float(amplitude),"pair_converged":True,"radius_separation":float(separation)})
        fitted=fit_fold_normal_form(records)
        self.assertAlmostEqual(fitted["fold_amplitude"],fold,places=10)
        self.assertAlmostEqual(fitted["fit_r_squared"],1,places=10)
        self.assertLessEqual(fitted["fit_amplitude_max"],records[11]["amplitude"])


if __name__=="__main__":unittest.main()
