import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from bhps.radion_effective import epsilon_from_mass,finite_interval_weak_radion_mass_squared,gw_control_point,leading_radion_mass


class RadionEffectiveTests(unittest.TestCase):
    def test_epsilon_definition(self):
        self.assertAlmostEqual(epsilon_from_mass(0.2),(4.04)**.5-2)

    def test_mass_formula(self):
        expected=2/3**.5*.1*.2*2.718281828459045**-1
        self.assertAlmostEqual(leading_radion_mass(.1,.04,1),expected)

    def test_finite_interval_limit_approaches_large_warp_formula(self):
        epsilon=.01;b0=.02;d=8.;b1=b0*2.718281828459045**(-2*epsilon*d)
        finite=finite_interval_weak_radion_mass_squared(epsilon,b0,d)
        asymptotic=leading_radion_mass(epsilon,b1,d)**2
        self.assertLess(abs(finite/asymptotic-1),.006)

    def test_current_weak_control_is_not_heavy(self):
        result=gw_control_point(.1,.1,1,1)
        self.assertTrue(result["probe_backreaction_nominal"])
        self.assertTrue(result["small_epsilon_nominal"])
        self.assertFalse(result["necessary_heavy_condition_mu_sigma_gt_1"])


if __name__=="__main__":unittest.main()
