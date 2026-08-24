import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.vector_gauge import gauge_away_mixed_component


class VectorGaugeTests(unittest.TestCase):
    def test_mixed_orbifold_component_can_be_gauged_away(self):
        y=np.linspace(0,1,129);warp=.7*y+.03*np.sin(np.pi*y)**2
        mixed=np.sin(np.pi*y)*(1+.2*np.cos(2*np.pi*y))
        normal_gradient=.1*np.sin(2*np.pi*y)
        result=gauge_away_mixed_component(y,warp,mixed,normal_gradient,.4)
        self.assertTrue(result["wall_preserving_normal_gauge"])
        self.assertLess(result["maximum_transformed_residual"],2e-15)

    def test_residual_tangent_gauge_preserves_zero_mixed_component(self):
        y=np.linspace(0,1,65);warp=.8*y
        result=gauge_away_mixed_component(y,warp,np.zeros_like(y),np.zeros_like(y),1.7)
        expected=1.7*np.exp(-2*warp)
        np.testing.assert_allclose(result["xi_mu"],expected,rtol=0,atol=1e-14)
        self.assertLess(result["maximum_transformed_residual"],2e-15)


if __name__=="__main__":unittest.main()
