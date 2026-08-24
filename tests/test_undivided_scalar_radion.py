import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.gw_background import solve_gw_background
from bhps.scalar_radion import coupled_scalar_radion_spectrum
from bhps.undivided_scalar_radion import (
    chebyshev_lobatto_grid_and_derivative,
    second_order_derivative_matrix,
    undivided_scalar_radion_matrices,
    undivided_scalar_radion_kinetic_norm,
    undivided_scalar_radion_spectral_control,
    undivided_scalar_radion_spectrum,
    shoot_turning_scalar_radion_mode,
)


class UndividedScalarRadionTests(unittest.TestCase):
    def test_three_point_derivative_is_exact_for_quadratic(self):
        z=np.array((.2,.5,1.1,1.7,2.8))
        derivative=second_order_derivative_matrix(z)
        np.testing.assert_allclose(derivative@(3*z**2-2*z+4),6*z-2,atol=2e-14)

    def test_lobatto_derivative_is_exact_for_polynomial(self):
        z,derivative=chebyshev_lobatto_grid_and_derivative(1,np.e,17)
        np.testing.assert_allclose(derivative@(z**4-2*z),4*z**3-2,atol=3e-12)

    def test_turning_profile_has_finite_undivided_matrices(self):
        z=np.linspace(1,np.e,129)
        background=solve_gw_background(
            z,epsilon=.1,backreaction=.01,wall_stiffness=5.,tolerance=1e-11,
        )
        matrices=undivided_scalar_radion_matrices(
            z,background["psi"],background["psi_z"],background["phi"],
            background["phi_z"],background["mass_squared"],5.,
        )
        self.assertTrue(matrices["phi_z_sign_change"])
        self.assertTrue(matrices["finite_coefficients"])
        self.assertEqual(matrices["matrix"].shape,(2*len(z)-1,2*len(z)-1))

    def test_monotone_control_reproduces_divided_low_spectrum(self):
        z=np.linspace(1,np.e,129)
        background=solve_gw_background(
            z,epsilon=.1,backreaction=.01,wall_stiffness=20.,tolerance=1e-11,
        )
        arguments=(
            z,background["psi"],background["psi_z"],background["phi"],
            background["phi_z"],background["mass_squared"],20.,
        )
        divided=coupled_scalar_radion_spectrum(*arguments,count=3)
        undivided=undivided_scalar_radion_spectrum(*arguments,count=3)
        np.testing.assert_allclose(
            undivided["mu_squared"],divided["mu_squared"],rtol=8e-4,atol=2e-9,
        )
        self.assertLess(np.max(undivided["generalized_residuals"]),2e-12)

    def test_soft_wall_negative_mode_is_grid_convergent(self):
        values=[]
        for size in (65,129,257):
            z=np.linspace(1,np.e,size)
            background=solve_gw_background(
                z,epsilon=.1,backreaction=.01,wall_stiffness=5.,tolerance=1e-11,
            )
            spectrum=undivided_scalar_radion_spectrum(
                z,background["psi"],background["psi_z"],background["phi"],
                background["phi_z"],background["mass_squared"],5.,count=3,
            )
            values.append(spectrum["closest_to_zero_mu_squared"])
            self.assertTrue(spectrum["phi_z_sign_change"])
            self.assertLess(spectrum["closest_to_zero_mu_squared"],0.)
        self.assertLess(abs(values[-1]-values[-2]),abs(values[-2]-values[-3]))
        self.assertLess(abs(values[-1]/values[-2]-1),2e-3)

    def test_lobatto_control_confirms_soft_wall_negative_mode(self):
        grid=np.linspace(1,np.e,257)
        background=solve_gw_background(
            grid,epsilon=.1,backreaction=.01,wall_stiffness=5.,tolerance=1e-11,
        )
        staggered=undivided_scalar_radion_spectrum(
            grid,background["psi"],background["psi_z"],background["phi"],
            background["phi_z"],background["mass_squared"],5.,count=1,
        )
        z,_=chebyshev_lobatto_grid_and_derivative(1,np.e,33)
        background=solve_gw_background(
            z,epsilon=.1,backreaction=.01,wall_stiffness=5.,tolerance=1e-11,
        )
        spectral=undivided_scalar_radion_spectral_control(
            z,background["psi"],background["psi_z"],background["phi"],
            background["phi_z"],background["mass_squared"],5.,count=1,
        )
        self.assertLess(spectral["closest_to_zero_mu_squared"],0.)
        self.assertLess(
            abs(staggered["closest_to_zero_mu_squared"]/
                spectral["closest_to_zero_mu_squared"]-1),5e-4,
        )

    def test_turning_mode_has_positive_regular_kinetic_norm(self):
        z,_=chebyshev_lobatto_grid_and_derivative(1,np.e,33)
        background=solve_gw_background(
            z,epsilon=.1,backreaction=.01,wall_stiffness=5.,tolerance=1e-11,
        )
        spectrum=undivided_scalar_radion_spectral_control(
            z,background["psi"],background["psi_z"],background["phi"],
            background["phi_z"],background["mass_squared"],5.,count=1,
        )
        mode=spectrum["eigenvectors"][:,0]
        norm=undivided_scalar_radion_kinetic_norm(
            z,background["psi"],mode[:len(z)],mode[len(z):],
        )
        self.assertTrue(norm["positive"])
        self.assertGreater(norm["metric_contribution"],0.)
        self.assertGreater(norm["stabilizer_contribution"],0.)

    def test_two_sided_turning_shooting_confirms_negative_mode(self):
        z=np.linspace(1,np.e,1025)
        background=solve_gw_background(
            z,epsilon=.1,backreaction=.01,wall_stiffness=5.,tolerance=1e-11,
        )
        shooting=shoot_turning_scalar_radion_mode(
            z,background["psi"],background["psi_z"],background["phi"],
            background["phi_z"],background["mass_squared"],5.,-1.08138e-5,
            turn_offset_fraction=3e-6,tolerance=1e-10,
        )
        self.assertTrue(shooting["converged"])
        self.assertLess(shooting["mu_squared"],0.)
        self.assertLess(abs(shooting["mu_squared"]/-1.08137948e-5-1),2e-5)
        self.assertLess(abs(shooting["upper_wall_residual"]),2e-11)


if __name__=="__main__":unittest.main()
