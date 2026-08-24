import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.corrected_A790_test10d_boundary_resolution import (
    LEVELS,
    METRIC_KEYS,
    SCHEMES,
    classify_test10d,
    closure_gate,
    ensemble_metrics,
    evaluate_all_levels,
    refinement_flags,
    source_grid_close,
)


def synthetic_face(samples=65, spike=False):
    z = np.linspace(1.0, 2.0, samples)
    q_perp = np.ones_like(z)
    q_zz = np.ones_like(z)
    before = np.column_stack((1.0 + 2.0 * z, 0.5 + z))
    delta = np.column_stack((0.1 * (1.0 + 2.0 * z), -0.05 * (0.5 + z)))
    if spike:
        delta[:] = 0.0
        delta[0] = (1.0, -0.5)
    after = before + delta
    zero = np.zeros_like(before)
    fields = {
        "delta": delta,
        "before": before,
        "after": after,
        "reference": before.copy(),
        "term_a": -delta,
        "term_v": zero.copy(),
        "term_c": zero.copy(),
        "term_sum": -delta,
    }
    return z, q_perp, q_zz, fields


class TestRefinedBoundaryResolution(unittest.TestCase):
    def test_two_evaluators_and_levels_are_fixed(self):
        self.assertEqual(SCHEMES, ("pchip_gauss", "natural_cubic_romberg"))
        self.assertEqual(LEVELS, (4, 8, 16))

    def test_linear_primitive_has_exact_known_norm(self):
        z, q_perp, q_zz, fields = synthetic_face()
        records = evaluate_all_levels(z, 3.0, q_perp, q_zz, fields, 1.0)
        primitive = lambda x: x + 2.0 * x**2 + (4.0 / 3.0) * x**3
        exact = math.sqrt(4.0 * math.pi * 9.0 * (primitive(2.0) - primitive(1.0)))
        for scheme in SCHEMES:
            self.assertLess(abs(records[scheme][16]["norm_before"] - math.sqrt(
                exact**2 + 4.0 * math.pi * 9.0 * (
                    (0.25 * 2.0 + 0.5 * 2.0**2 + 2.0**3 / 3.0)
                    - (0.25 * 1.0 + 0.5 * 1.0**2 + 1.0**3 / 3.0)
                )
            )) / records[scheme][16]["norm_before"], 1e-10)
        flags = refinement_flags(records)
        self.assertTrue(all(flags[s][k] for s in SCHEMES for k in METRIC_KEYS))
        self.assertTrue(all(flags["cross_scheme"].values()))

    def test_scale_invariance(self):
        z, q_perp, q_zz, fields = synthetic_face()
        values = []
        for scale in (1e-6, 1.0, 1e6):
            scaled = {key: value * scale for key, value in fields.items()}
            result = ensemble_metrics(evaluate_all_levels(
                z, 2.0, q_perp, q_zz, scaled, scale,
            ))
            values.append([result[key] for key in (
                "proper_ratio", "term_balance_ratio",
                "component_phi_ratio", "component_chi_ratio",
            )])
        self.assertTrue(np.allclose(values, values[1], rtol=1e-10, atol=1e-12))

    def test_dual_near_zero_gate(self):
        self.assertTrue(closure_gate(6.2e-22, 5.0e-17))
        self.assertFalse(closure_gate(2.0e-18, 0.0))

    def test_adverse_single_node_spike_is_exposed(self):
        z, q_perp, q_zz, fields = synthetic_face(spike=True)
        records = evaluate_all_levels(z, 2.0, q_perp, q_zz, fields, 1.0)
        flags = refinement_flags(records)
        self.assertFalse(flags["cross_scheme"]["norm_delta"])

    def test_nonpositive_metric_is_rejected(self):
        z, q_perp, q_zz, fields = synthetic_face()
        q_perp[10] = 0.0
        with self.assertRaises(ValueError):
            evaluate_all_levels(z, 2.0, q_perp, q_zz, fields, 1.0)

    def test_source_grid_and_classification(self):
        self.assertTrue(source_grid_close(0.05, 0.06))
        self.assertFalse(source_grid_close(0.05, 0.10))
        self.assertEqual(
            classify_test10d(True, False, False, False, True, False),
            ("review", "boundary_local_response_not_source_grid_converged"),
        )
        self.assertEqual(
            classify_test10d(True, False, False, False, True, True),
            ("pass", "converged_boundary_local_no_resolved_interior_effect"),
        )
        self.assertEqual(
            classify_test10d(False, False, False, False, True, True),
            ("review", "invalid_refined_boundary_audit"),
        )


if __name__ == "__main__":
    unittest.main()
