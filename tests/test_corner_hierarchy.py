import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.corner_hierarchy import frozen_principal_corner_hierarchy,stabilizer_acceleration_corner_residual
from bhps.israel_wave_matrix import coupled_robin_matrix


class CornerHierarchyTests(unittest.TestCase):
    def setUp(self):
        uprime=.0075
        self.robin=coupled_robin_matrix(-1.0088,uprime/6,uprime/2,20.)["matrix"]

    def test_manufactured_hierarchy_passes_through_order_ten(self):
        seed=np.linspace(-.02,.03,13)
        audit=frozen_principal_corner_hierarchy(self.robin,seed,1.3,10)
        self.assertLess(audit["propagator_commutator_norm"],1e-10)
        self.assertLess(audit["maximum_normalized_boundary_residual"],1e-12)

    def test_wall_flat_acceleration_clears_full_frozen_second_corner(self):
        audit=stabilizer_acceleration_corner_residual(self.robin,0.,0.)
        self.assertTrue(audit["passes"])

    def test_nonflat_acceleration_is_rejected(self):
        audit=stabilizer_acceleration_corner_residual(self.robin,0.,.2)
        self.assertFalse(audit["passes"])
        self.assertAlmostEqual(audit["maximum_absolute_residual"],.2)


if __name__=="__main__":unittest.main()
