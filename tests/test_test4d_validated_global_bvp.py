import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.validated_capped_surface_shooting import VInterval
from bhps.validated_global_bvp import (
    CHEBYSHEV_WEIGHT,
    CONFIGURATIONS,
    base_nonaxis_mesh,
    chebyshev_coefficients_from_lobatto,
    chebyshev_l1_nu,
    chebyshev_lobatto_nodes,
    configuration_mesh,
    finite_radii_bounds,
    first_certified_radius,
    floating_offnode_residual_diagnostics,
    grade_test4d,
    predictor_diagnostics,
    radius_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ProtocolSealTests(unittest.TestCase):
    def test_protocol_and_immutable_input_hashes_match_seal(self):
        seal = json.loads((
            ROOT / "results/test4d_validated_global_bvp_protocol_seal.json"
        ).read_text())
        protocol = seal["protocol"]
        self.assertEqual(sha256(ROOT / protocol["path"]), protocol["sha256"])
        for relative, expected in seal["immutable_inputs"].items():
            self.assertEqual(sha256(ROOT / relative), expected)
        self.assertFalse(seal["physical_outcomes_inspected_before_seal"])


class MeshAndNormTests(unittest.TestCase):
    def test_base_mesh_has_exact_geometric_and_bulk_domain_counts(self):
        mesh = base_nonaxis_mesh()
        self.assertEqual(len(mesh) - 1, 70)
        self.assertTrue(np.array_equal(
            mesh[:7], np.asarray([.001, .002, .004, .008, .016, .032, .064]),
        ))
        self.assertEqual(mesh[-1], math.pi / 2)
        self.assertTrue(np.all(np.diff(mesh) > 0.0))

    def test_configuration_ladder_is_fixed_and_bisection_exact(self):
        self.assertEqual(
            [item["name"] for item in CONFIGURATIONS],
            ["D12-M70-P160", "D16-M70-P160",
             "D16-M140-P192", "D20-M140-P256"],
        )
        coarse = configuration_mesh(CONFIGURATIONS[0])
        fine = configuration_mesh(CONFIGURATIONS[2])
        self.assertEqual(len(coarse) - 1, 70)
        self.assertEqual(len(fine) - 1, 140)
        self.assertEqual(fine[0], coarse[0])
        self.assertEqual(fine[-1], coarse[-1])
        self.assertTrue(all(value in fine for value in coarse))

    def test_weighted_chebyshev_norm_and_radius_list(self):
        coefficients = np.asarray([1.0, -2.0, 3.0])
        expected = 1.0 + 4.0*CHEBYSHEV_WEIGHT + 6.0*CHEBYSHEV_WEIGHT**2
        self.assertGreaterEqual(chebyshev_l1_nu(coefficients), expected)
        radii = radius_candidates()
        self.assertEqual(radii[0], 1e-14)
        self.assertEqual(radii[-1], 1e-4)
        self.assertTrue(all(left < right for left, right in zip(radii, radii[1:])))

    def test_lobatto_transform_reconstructs_polynomial(self):
        nodes = chebyshev_lobatto_nodes(6)
        expected = np.asarray([1.2, -0.7, 0.3, 0.1, -0.04, 0.02, -0.01])
        values = np.polynomial.chebyshev.chebval(nodes, expected)
        actual = chebyshev_coefficients_from_lobatto(values)
        self.assertTrue(np.allclose(actual, expected, rtol=0.0, atol=2e-14))

    def test_predictor_continuity_uses_adjacent_physical_endpoints(self):
        # Lobatto nodes run from +1 to -1.  The common interface is therefore
        # T(+1) in the left block and T(-1) in the right block.
        left = np.polynomial.chebyshev.chebfit(
            [-1.0, 1.0], [1.0, 2.0], 1,
        )
        right = np.polynomial.chebyshev.chebfit(
            [-1.0, 1.0], [2.0, 4.0], 1,
        )
        predictor = {
            "mesh": np.asarray([0.001, 0.002, 0.004]),
            "axis_rho_center": np.asarray([1.0, 0.0]),
            "axis_u_center": np.asarray([0.0, 0.0]),
            "rho_blocks_center": np.asarray([left, right]),
            "w_blocks_center": np.asarray([left, right]),
        }
        for component in ("axis_rho", "axis_u", "rho_blocks", "w_blocks"):
            predictor[component + "_affine_endpoint_defect"] = np.zeros_like(
                predictor[component + "_center"]
            )
        jumps = predictor_diagnostics(predictor)[
            "maximum_internal_continuity_jump"
        ]
        self.assertLess(jumps["rho"], 2e-15)
        self.assertLess(jumps["w"], 2e-15)

    def test_offnode_diagnostic_rejects_undersampling(self):
        predictor = {
            "mesh": np.asarray([0.001, 0.002]),
            "rho_blocks_center": np.zeros((1, 5)),
            "w_blocks_center": np.zeros((1, 5)),
            "w_blocks_parameter": np.zeros((1, 5)),
            "w_blocks_affine_endpoint_defect": np.zeros((1, 5)),
        }
        with self.assertRaises(ValueError):
            floating_offnode_residual_diagnostics(
                predictor, 1.0, {}, samples_per_domain=5,
            )


class ManufacturedRadiiPolynomialTests(unittest.TestCase):
    def test_linear_stable_unstable_multiple_shooting_system_certifies(self):
        flow = np.diag([math.exp(-0.4), math.exp(0.4)])
        jacobian = np.block([
            [np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))],
            [-flow, np.eye(2), np.zeros((2, 2))],
            [np.zeros((2, 2)), -flow, np.eye(2)],
        ])
        inverse = np.linalg.inv(jacobian)
        interval_jacobian = np.asarray([
            [VInterval.point(jacobian[i, j]) for j in range(6)]
            for i in range(6)
        ], dtype=object)
        residual = [VInterval(-1e-12, 1e-12) for _ in range(6)]
        bounds = finite_radii_bounds(
            inverse, jacobian, interval_jacobian, residual,
            np.ones(6), z20=1e-3, z21=1e-3,
        )
        certificate = first_certified_radius(bounds)
        self.assertIsNotNone(certificate)
        self.assertLess(certificate["radius_polynomial_upper"], 0.0)
        self.assertLess(certificate["contraction_upper"], 1.0)

    def test_uncertain_singular_jacobian_cannot_certify(self):
        jacobian = np.eye(2)
        interval_jacobian = np.asarray([
            [VInterval(0.0, 2.0), VInterval.point(0.0)],
            [VInterval.point(0.0), VInterval(0.0, 2.0)],
        ], dtype=object)
        bounds = finite_radii_bounds(
            np.eye(2), jacobian, interval_jacobian,
            [VInterval.point(0.0), VInterval.point(0.0)],
            np.ones(2),
        )
        self.assertIsNone(first_certified_radius(bounds))

    def test_unresolved_leaf_forces_review_and_fail_needs_confirmation(self):
        unresolved = [{"classification": "unresolved_radius_polynomial"}]
        positive = [{"classification": "root_free_positive"}]
        self.assertEqual(
            grade_test4d(True, unresolved, positive, 2, 0, False), "REVIEW",
        )
        root = [{"classification": "validated_root"}]
        self.assertEqual(grade_test4d(True, root, [], 0, 1, False), "REVIEW")
        self.assertEqual(grade_test4d(True, root, [], 0, 1, True), "FAIL")


if __name__ == "__main__":
    unittest.main()
