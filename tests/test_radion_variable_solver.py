import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from bhps.initial_data import make_grid
from bhps.radion_variable_solver import q_jacobian,q_residual,solve_q
from bhps.scalar_pulse import scalar_pulse


class RadionVariableSolverTests(unittest.TestCase):
    def test_zero_source_is_exact(self):
        solved=solve_q(0,nz=17,nr=25)
        self.assertTrue(solved["converged"])
        self.assertEqual(solved["max_abs_residual"],0)
        self.assertEqual(solved["interpolated_axis_extremum"],0)

    def test_constant_q_preserves_bulk_and_brane_radion_family(self):
        z,r=make_grid(nz=17,nr=25);q=np.full((len(z),len(r)),.03);zeros=np.zeros_like(q)
        residual=q_residual(q,z,r,zeros,zeros,"dirichlet").reshape(q.shape)
        self.assertLess(np.max(np.abs(residual[:,:-1])),1e-13)

    def test_exact_jacobian_matches_directional_difference(self):
        solved=solve_q(.2,nz=9,nr=13);z,r,q=solved["z"],solved["r"],solved["q"]
        _,chi_r,chi_z=scalar_pulse(z,r,.2)
        direction=np.sin(np.arange(q.size)).reshape(q.shape);step=1e-7
        finite=(q_residual(q+step*direction,z,r,chi_r,chi_z)-q_residual(q-step*direction,z,r,chi_r,chi_z))/(2*step)
        exact=q_jacobian(q,z,r,chi_r,chi_z)@direction.ravel()
        self.assertLess(np.max(np.abs(finite-exact)),2e-6)


if __name__=="__main__":unittest.main()
