import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.gw_background import solve_gw_background
from bhps.tensor_ibvp import frozen_neumann_boundary_symbol,frozen_positive_robin_boundary_symbol,tensor_energy_density,weighted_neumann_tensor_spectrum,weighted_robin_scalar_spectrum


class TensorIBVPTests(unittest.TestCase):
    def test_frozen_neumann_symbol_has_no_unstable_root(self):
        for real in (.01,.1,1.,10.):
            for imag in (-10.,-1.,0.,1.,10.):
                for wave in (0.,.2,2.,20.):
                    result=frozen_neumann_boundary_symbol(real,imag,wave)
                    self.assertFalse(result["unstable_root"])
                    self.assertGreater(result["decay_rate"].real,0.)

    def test_actual_finite_wall_background_has_nonnegative_tt_spectrum(self):
        z=np.linspace(1,np.e,101);background=solve_gw_background(z,epsilon=.1,backreaction=.01,wall_stiffness=20.)
        spectrum=weighted_neumann_tensor_spectrum(z,background["psi"],count=6)
        self.assertTrue(spectrum["all_within_roundoff_nonnegative"])
        self.assertGreater(spectrum["first_positive_omega_squared"],0.)
        self.assertGreater(spectrum["zero_mode_constant_overlap"],1-1e-10)

    def test_neumann_wall_has_zero_tensor_energy_flux(self):
        result=tensor_energy_density(np.array((1.,.5)),np.array((2.,3.)),np.zeros(2))
        self.assertTrue(np.all(result["density"]>=0))
        np.testing.assert_array_equal(result["normal_flux"],0.)

    def test_positive_robin_symbol_has_no_unstable_root(self):
        for real in (.01,.1,1.,10.):
            for imag in (-10.,0.,10.):
                result=frozen_positive_robin_boundary_symbol(real,imag,2.,1.,20.)
                self.assertFalse(result["unstable_root"])

    def test_finite_wall_probe_scalar_spectrum_is_positive(self):
        z=np.linspace(1,np.e,101);background=solve_gw_background(z,epsilon=.1,backreaction=.01,wall_stiffness=20.)
        spectrum=weighted_robin_scalar_spectrum(z,background["psi"],background["mass_squared"],20.,count=6)
        self.assertTrue(spectrum["all_positive"])


if __name__=="__main__":unittest.main()
