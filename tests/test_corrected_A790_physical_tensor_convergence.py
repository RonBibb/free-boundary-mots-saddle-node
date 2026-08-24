import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.corrected_A790_physical_tensor_convergence import (
    adm_extrinsic_curvature_tensor,
    four_grid_orders,
    physical_tensor_difference,
    physical_tensor_l2,
    spatial_metric_tensor,
)


def flat_state(z, r):
    q = np.zeros((len(z), len(r), 9))
    q[..., 2] = -1.0
    q[..., 3] = 1.0
    q[..., 6] = 1.0
    return q


class CorrectedA790PhysicalTensorConvergenceTests(unittest.TestCase):
    def test_d_coefficient_enters_only_as_r_squared_d(self):
        z = np.linspace(0.0, 1.0, 9)
        r = np.linspace(0.0, 1.0, 13)
        q = flat_state(z, r)
        q[..., 4] = 2.5
        metric = spatial_metric_tensor(q, r)
        expected_radial = np.broadcast_to(1.0 + 2.5 * r[None, :] ** 2, q.shape[:2])
        np.testing.assert_allclose(metric[..., 1, 1], expected_radial)
        np.testing.assert_allclose(metric[:, 0, 1, 1], 1.0)
        np.testing.assert_allclose(metric[..., 2, 2], 1.0)

    def test_flat_constant_curvature_adm_tensor(self):
        z = np.linspace(0.0, 1.0, 17)
        r = np.linspace(0.0, 1.0, 21)
        q = flat_state(z, r)
        rate = 0.7
        velocity = np.zeros_like(q)
        velocity[..., 3] = -2.0 * rate
        velocity[..., 6] = -2.0 * rate
        extrinsic = adm_extrinsic_curvature_tensor(q, velocity, z, r)
        expected = np.broadcast_to(np.eye(4) * rate, extrinsic.shape)
        np.testing.assert_allclose(extrinsic, expected, rtol=0.0, atol=2e-13)

    def test_proper_volume_norm_matches_constant_flat_control(self):
        z = np.linspace(0.0, 2.0, 65)
        r = np.linspace(0.0, 1.5, 81)
        metric = spatial_metric_tensor(flat_state(z, r), r)
        tensor = np.broadcast_to(np.eye(4), metric.shape).copy()
        found = physical_tensor_l2(tensor, metric, z, r)
        volume = 2.0 * 4.0 * np.pi * 1.5**3 / 3.0
        expected = np.sqrt(4.0 * volume)
        self.assertAlmostEqual(found / expected, 1.0, places=12)

    def test_manufactured_metric_and_K_sequences_recover_order_three(self):
        z = np.linspace(0.0, 1.0, 33)
        r = np.linspace(0.0, 1.0, 41)
        base = spatial_metric_tensor(flat_state(z, r), r)
        shape = np.zeros_like(base)
        shape[..., 0, 0] = 0.3
        shape[..., 1, 1] = r[None, :] ** 2
        shape[..., 2, 2] = 0.2
        shape[..., 3, 3] = 0.2
        counts = (80, 96, 112, 128)
        fields = [value ** -3 * shape for value in counts]
        differences = []
        for left, right in zip(fields[:-1], fields[1:]):
            differences.append(physical_tensor_difference(
                left, right, base, base, z, r,
            )["absolute_difference"])
        orders = four_grid_orders(differences, counts)
        self.assertAlmostEqual(orders["coarse_triplet_order"], 3.0, places=8)
        self.assertAlmostEqual(orders["fine_triplet_order"], 3.0, places=8)


if __name__ == "__main__":
    unittest.main()
