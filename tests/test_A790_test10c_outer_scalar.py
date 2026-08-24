import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.corrected_A790_test10c_outer_scalar import (
    classify_test10c,
    correction_metrics,
    endpoint_derivative,
    independent_characteristic_terms,
    normalized_radial_difference,
    polynomial_endpoint_weights,
    proper_face_weight,
)
from bhps.nonlinear_regular_so3_evolution import (
    apply_outer_sommerfeld_acceleration,
)


class Test10COuterScalarTests(unittest.TestCase):
    @staticmethod
    def flat_state(nz=33, nr=49):
        z = np.linspace(1.0, 2.0, nz)
        r = np.linspace(0.0, 3.0, nr)
        q = np.zeros((nz, nr, 9))
        q[..., 2] = -1.0
        q[..., 3] = q[..., 6] = 1.0
        return z, r, q

    def test_polynomial_endpoint_weights_are_exact(self):
        r = np.linspace(0.0, 2.0, 15)
        field = np.zeros((3, len(r), 2))
        field[..., 0] = r[None, :] ** 5
        field[..., 1] = r[None, :] ** 3
        found = endpoint_derivative(field, r, 7)
        np.testing.assert_allclose(found[:, 0], 5.0 * r[-1] ** 4, rtol=1e-11)
        np.testing.assert_allclose(found[:, 1], 3.0 * r[-1] ** 2, rtol=1e-11)
        self.assertEqual(polynomial_endpoint_weights(r[-7:]).shape, (7,))

    def test_independent_target_matches_production_row(self):
        z, r, q0 = self.flat_state()
        q = q0.copy()
        v = np.zeros_like(q)
        rr = r[None, :] / r[-1]
        zz = (z[:, None] - z[0]) / (z[-1] - z[0])
        q[..., 7] = 0.02 * rr**3 * (1.0 + 0.1 * zz)
        q[..., 8] = -0.01 * rr**2 * (1.0 - 0.2 * zz)
        v[..., 7] = 0.03 * rr**4
        v[..., 8] = 0.015 * rr**3
        before = np.zeros_like(q)
        reference_acceleration = np.zeros_like(q)
        corrected, _ = apply_outer_sommerfeld_acceleration(
            q, v, before, q0, reference_acceleration, 0.2, r, 7,
        )
        terms = independent_characteristic_terms(
            q, v, before, q0, reference_acceleration, 0.2, r, 7,
        )
        np.testing.assert_allclose(
            corrected[1:-1, -1, 7:9], terms["target"][1:-1, 7:9],
            rtol=1e-10, atol=1e-11,
        )

    def test_uniform_metric_ratios_and_scale_invariance(self):
        z, r, q = self.flat_state()
        before = np.zeros_like(q)
        after = np.zeros_like(q)
        before[..., 7:9] = 2.0
        after[..., 7:9] = 1.0
        reference = before.copy()
        terms = {
            "target": after[:, -1].copy(),
            "term_A": before[:, -1].copy(),
            "term_V": np.zeros_like(before[:, -1]),
            "term_C": -after[:, -1].copy(),
        }
        base = correction_metrics(q, before, after, reference, terms, z, r)
        self.assertAlmostEqual(base["legacy_ratio"], 0.5)
        self.assertAlmostEqual(base["proper"]["simpson"]["ratio"], 0.5)
        self.assertAlmostEqual(base["pointwise"]["maximum"], 0.5)
        for scale in (1e-6, 1e6):
            scaled_terms = {
                key: value * scale for key, value in terms.items()
            }
            found = correction_metrics(
                q, before * scale, after * scale, reference * scale,
                scaled_terms, z, r,
            )
            for key in ("legacy_ratio", "collar_ratio"):
                self.assertAlmostEqual(found[key], base[key], places=10)
            self.assertAlmostEqual(
                found["proper"]["simpson"]["ratio"],
                base["proper"]["simpson"]["ratio"], places=10,
            )

    def test_pointwise_guard_exposes_global_hiding(self):
        z, r, q = self.flat_state(nz=65)
        before = np.zeros_like(q)
        after = np.zeros_like(q)
        before[..., 7:9] = 100.0
        after[..., 7:9] = 100.0
        middle = len(z) // 2
        before[middle, -1, 7] = 0.1
        after[middle, -1, 7] = 1.1
        terms = {
            "target": after[:, -1].copy(),
            "term_A": before[:, -1] - after[:, -1],
            "term_V": np.zeros_like(before[:, -1]),
            "term_C": np.zeros_like(before[:, -1]),
        }
        found = correction_metrics(q, before, after, before, terms, z, r)
        self.assertLess(found["legacy_ratio"], 0.05)
        self.assertGreater(found["pointwise"]["maximum"], 0.8)

    def test_nonpositive_face_metric_rejected(self):
        z, r, q = self.flat_state()
        q[5, -1, 3] = -1.0
        with self.assertRaisesRegex(ValueError, "not positive"):
            proper_face_weight(q, z, r)

    def test_radial_difference_and_classification(self):
        left = np.zeros((2, 4, 3))
        right = left.copy()
        right[:, 2, :] = 0.25
        profile = normalized_radial_difference(left, right)
        self.assertEqual(profile[0], 0.0)
        self.assertEqual(profile[2], 0.25)
        self.assertEqual(
            classify_test10c(True, False, False, True, False),
            ("pass", "legacy_normalization_inconsistency"),
        )
        self.assertEqual(
            classify_test10c(True, False, False, False, True),
            ("pass", "genuine_boundary_local_response"),
        )
        self.assertEqual(
            classify_test10c(True, True, False, False, True)[0], "fail",
        )


if __name__ == "__main__":
    unittest.main()
