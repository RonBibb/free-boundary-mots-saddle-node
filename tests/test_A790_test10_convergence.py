import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.corrected_A790_test10_convergence import (
    domain_initial_dominance,
    proper_endpoint_distances,
    temporal_tensor_sequence,
    three_grid_sequence,
    uniform_time_order,
)


class Test10ConvergenceTests(unittest.TestCase):
    def flat_state(self, z, r):
        q = np.zeros((len(z), len(r), 9))
        q[..., 2] = -1.0
        q[..., 3] = 1.0
        q[..., 6] = 1.0
        return q

    def test_flat_proper_endpoints_equal_coordinate_distances(self):
        z = np.linspace(1.0, 2.0, 81)
        r = np.linspace(0.0, 3.0, 121)
        found = proper_endpoint_distances(
            self.flat_state(z, r), z, r, rho_axis=0.4, rho_brane=1.2,
        )
        self.assertAlmostEqual(found["compact_axis_endpoint_to_brane"], 0.4, 12)
        self.assertAlmostEqual(found["radial_axis_to_brane_endpoint"], 1.2, 12)

    def test_proper_endpoint_rejects_adverse_inputs(self):
        z = np.linspace(1.0, 2.0, 9)
        r = np.linspace(0.0, 2.0, 13)
        q = self.flat_state(z, r)
        q[-1, :, 3] = -1.0
        with self.assertRaisesRegex(ValueError, "positive"):
            proper_endpoint_distances(q, z, r, 0.2, 0.8)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            proper_endpoint_distances(self.flat_state(z, r), z[::-1], r, 0.2, 0.8)

    def test_uniform_time_order_recovers_second_order(self):
        self.assertAlmostEqual(uniform_time_order(4.0, 1.0), 2.0, 12)

    def test_manufactured_temporal_tensor_order(self):
        z = np.linspace(0.0, 1.0, 17)
        r = np.linspace(0.0, 1.0, 25)
        metric = np.broadcast_to(np.eye(4), (len(z), len(r), 4, 4)).copy()
        shape = np.zeros_like(metric)
        shape[..., 0, 0] = 1.0
        fields = {"coarse": shape / 4, "medium": shape / 16, "fine": shape / 64}
        result = temporal_tensor_sequence(
            fields["coarse"], fields["medium"], fields["fine"],
            {name: metric for name in fields}, z, r,
        )
        self.assertAlmostEqual(result["order"], 2.0, 7)

    def test_manufactured_nonuniform_spatial_order(self):
        z = np.linspace(0.0, 1.0, 17)
        r = np.linspace(0.0, 1.0, 25)
        metric = np.broadcast_to(np.eye(4), (len(z), len(r), 4, 4)).copy()
        counts = (112, 128, 144)
        fields = {
            name: metric * count**-2
            for name, count in zip(("G9", "G10", "G11"), counts)
        }
        result = three_grid_sequence(
            fields, {name: metric for name in fields}, z, r, counts,
        )
        self.assertAlmostEqual(result["order"], 2.0, 7)

    def test_initial_dominance_adverse_control(self):
        dominated = domain_initial_dominance(10.0, 1.0, 10.5)
        self.assertTrue(dominated["initial_data_dominated"])
        evolved = domain_initial_dominance(10.0, 4.0, 14.0)
        self.assertFalse(evolved["initial_data_dominated"])


if __name__ == "__main__":
    unittest.main()

