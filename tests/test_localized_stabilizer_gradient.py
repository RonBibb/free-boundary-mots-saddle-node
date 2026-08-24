import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.localized_stabilizer_gradient import localized_stabilizer_gradient_diagnostics


class LocalizedStabilizerGradientTests(unittest.TestCase):
    def test_axisymmetric_quadratic_has_axis_minimum_and_turn_curve(self):
        z=np.linspace(1,2,17);r=np.linspace(0,1,19)
        zz,rr=np.meshgrid(z,r,indexing="ij")
        phi=(zz-1.6)**2+.3*rr**2
        ones=np.ones_like(phi);zeros=np.zeros_like(phi)
        result=localized_stabilizer_gradient_diagnostics(z,r,phi,ones,zeros,zeros)
        self.assertEqual(result["rays_with_compact_turn"],len(r))
        self.assertEqual(len(result["axis_stationary_points"]),1)
        point=result["axis_stationary_points"][0]
        self.assertAlmostEqual(point["z"],1.6,places=11)
        self.assertEqual(point["classification"],"minimum")
        self.assertFalse(result["divided_compact_gradient_variable_admissible"])

    def test_monotone_profile_has_no_compact_turn(self):
        z=np.linspace(1,2,17);r=np.linspace(0,1,19)
        zz,rr=np.meshgrid(z,r,indexing="ij")
        phi=-zz+.02*rr**2
        ones=np.ones_like(phi);zeros=np.zeros_like(phi)
        result=localized_stabilizer_gradient_diagnostics(z,r,phi,ones,zeros,zeros)
        self.assertEqual(result["rays_with_compact_turn"],0)
        self.assertEqual(result["axis_stationary_points"],[])
        self.assertTrue(result["divided_compact_gradient_variable_admissible"])


if __name__=="__main__":unittest.main()
