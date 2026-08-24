import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.gw_background import solve_gw_background
from bhps.scalar_radion import coupled_scalar_radion_energy,coupled_scalar_radion_spectrum,frozen_wentzell_boundary_symbol,shoot_lowest_scalar_radion_mode


class CoupledScalarRadionTests(unittest.TestCase):
    def test_finite_wall_background_has_positive_coupled_spectrum(self):
        z=np.linspace(1,np.e,161)
        background=solve_gw_background(z,epsilon=.1,backreaction=.01,wall_stiffness=20.)
        spectrum=coupled_scalar_radion_spectrum(
            z,background["psi"],background["psi_z"],background["phi"],
            background["phi_z"],background["mass_squared"],20.,count=5,
        )
        self.assertTrue(spectrum["all_positive"])
        self.assertTrue(spectrum["positive_wall_weights"])
        self.assertTrue(np.all(spectrum["wall_alphas"]>0))

    def test_stiff_wall_limit_removes_boundary_kinetic_weights(self):
        z=np.linspace(1,np.e,121)
        background=solve_gw_background(z,epsilon=.1,backreaction=.01)
        spectrum=coupled_scalar_radion_spectrum(
            z,background["psi"],background["psi_z"],background["phi"],
            background["phi_z"],background["mass_squared"],None,count=3,
        )
        np.testing.assert_array_equal(spectrum["boundary_mass_weights"],0.)
        self.assertTrue(spectrum["all_positive"])

    def test_lowest_mode_converges_under_grid_refinement(self):
        values=[]
        for size in (65,129,257):
            z=np.linspace(1,np.e,size)
            background=solve_gw_background(z,epsilon=.1,backreaction=.01,wall_stiffness=20.)
            spectrum=coupled_scalar_radion_spectrum(
                z,background["psi"],background["psi_z"],background["phi"],
                background["phi_z"],background["mass_squared"],20.,count=2,
            )
            values.append(spectrum["minimum_mu_squared"])
        self.assertLess(abs(values[-1]-values[-2]),abs(values[-2]-values[-3]))
        self.assertLess(abs(values[-1]-values[-2])/values[-1],2e-3)

    def test_nonpositive_wall_alpha_rejects_kinetic_form(self):
        z=np.linspace(1,np.e,65)
        background=solve_gw_background(z,epsilon=.1,backreaction=.01,wall_stiffness=20.)
        with self.assertRaises(ValueError):
            coupled_scalar_radion_spectrum(
                z,background["psi"],background["psi_z"],background["phi"],
                background["phi_z"],background["mass_squared"],(.01,.01),count=2,
            )

    def test_shooting_reproduces_finite_element_lowest_mode(self):
        z=np.linspace(1,np.e,257)
        background=solve_gw_background(z,epsilon=.1,backreaction=.01,wall_stiffness=20.)
        arguments=(z,background["psi"],background["psi_z"],background["phi"],
                   background["phi_z"],background["mass_squared"],20.)
        finite_element=coupled_scalar_radion_spectrum(*arguments,count=1)
        shooting=shoot_lowest_scalar_radion_mode(
            *arguments,eigenvalue_hint=finite_element["minimum_mu_squared"],
        )
        relative=abs(shooting["mu_squared"]/finite_element["minimum_mu_squared"]-1)
        self.assertTrue(shooting["converged"])
        self.assertLess(relative,2e-4)

    def test_turning_scalar_profile_is_rejected_by_divided_master_variable(self):
        z=np.linspace(1,np.e,257)
        background=solve_gw_background(z,epsilon=.1,backreaction=.01,wall_stiffness=5.)
        self.assertTrue(np.any(background["phi_z"][:-1]*background["phi_z"][1:]<0))
        with self.assertRaisesRegex(ValueError,"changes sign"):
            coupled_scalar_radion_spectrum(
                z,background["psi"],background["psi_z"],background["phi"],
                background["phi_z"],background["mass_squared"],5.,count=2,
            )

    def test_positive_wentzell_symbol_has_no_unstable_root(self):
        for real in (.001,.1,1.,10.):
            for imag in (-20.,-1.,0.,1.,20.):
                for wave in (0.,.2,2.,20.):
                    result=frozen_wentzell_boundary_symbol(real,imag,wave,3.,7.,.04)
                    self.assertFalse(result["unstable_root"])
                    self.assertGreater(result["decay_rate"].real,0.)

    def test_coupled_energy_includes_positive_wall_kinetic_term(self):
        z=np.linspace(1,np.e,65)
        background=solve_gw_background(z,epsilon=.1,backreaction=.01,wall_stiffness=20.)
        spectrum=coupled_scalar_radion_spectrum(
            z,background["psi"],background["psi_z"],background["phi"],
            background["phi_z"],background["mass_squared"],20.,count=1,
        )
        g=np.cos(z);g_t=np.sin(z);g_z=-np.sin(z)
        energy=coupled_scalar_radion_energy(
            z,background["psi"],background["phi_z"],g,g_t,g_z,.25,
            spectrum["boundary_mass_weights"],
        )
        self.assertGreater(energy["bulk_energy"],0.)
        self.assertGreater(energy["boundary_energy"],0.)
        self.assertAlmostEqual(energy["total_energy"],energy["bulk_energy"]+energy["boundary_energy"])


if __name__=="__main__":unittest.main()
