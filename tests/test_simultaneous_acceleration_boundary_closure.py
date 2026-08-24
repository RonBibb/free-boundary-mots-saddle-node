import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.simultaneous_acceleration_boundary_closure import (
    DegenerateBoundaryOwnership,
    InconsistentBoundaryConstraints,
    close_outer_face_wall_profile,
    close_regular_axis_wall_profile,
    compact_wall_rows,
    weighted_minimum_correction,
    weighted_minimum_correction_on_indices,
)


class TestWeightedMinimumCorrection(unittest.TestCase):
    def test_known_uniform_minimum_correction(self):
        result = weighted_minimum_correction(
            np.zeros(3), np.asarray([[1.0, 1.0, 0.0]]), np.asarray([1.0]),
        )
        self.assertTrue(np.allclose(result.profile, (0.5, 0.5, 0.0)))
        self.assertEqual(result.diagnostics["numerical_rank"], 1)
        self.assertLess(result.diagnostics["constraint_maximum_absolute_residual"], 1e-14)
        self.assertLess(result.diagnostics["transformed_kkt_stationarity_l2"], 1e-14)

    def test_weights_penalize_selected_correction(self):
        result = weighted_minimum_correction(
            np.zeros(2), np.asarray([[1.0, 1.0]]), np.asarray([1.0]),
            correction_weights=np.asarray([10.0, 1.0]),
        )
        self.assertTrue(np.allclose(result.profile, (1.0 / 101.0, 100.0 / 101.0)))
        self.assertAlmostEqual(result.diagnostics["weight_dynamic_range"], 10.0)

    def test_consistent_redundant_rows_are_audited(self):
        result = weighted_minimum_correction(
            np.zeros(3),
            np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.asarray([1.0, 2.0]),
        )
        self.assertTrue(np.allclose(result.profile, (1.0, 0.0, 0.0)))
        self.assertEqual(result.diagnostics["numerical_rank"], 1)
        self.assertEqual(result.diagnostics["redundant_constraint_count"], 1)
        self.assertFalse(result.diagnostics["full_row_rank"])
        with self.assertRaises(InconsistentBoundaryConstraints):
            weighted_minimum_correction(
                np.zeros(3),
                np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                np.asarray([1.0, 2.0]),
                require_full_row_rank=True,
            )

    def test_inconsistent_rows_are_rejected_with_diagnostics(self):
        with self.assertRaises(InconsistentBoundaryConstraints) as captured:
            weighted_minimum_correction(
                np.zeros(2),
                np.asarray([[1.0, 0.0], [2.0, 0.0]]),
                np.asarray([1.0, 3.0]),
            )
        diagnostic = captured.exception.diagnostics
        self.assertFalse(diagnostic["constraints_consistent"])
        self.assertGreater(
            diagnostic["equilibrated_consistency_residual_l2"],
            diagnostic["consistency_tolerance"],
        )

    def test_condition_limit_is_reported_and_optionally_enforced(self):
        matrix = np.asarray([[1.0, 0.0, 0.0], [1.0, 1e-9, 0.0]])
        result = weighted_minimum_correction(
            np.zeros(3), matrix, np.zeros(2), condition_limit=1e8,
        )
        self.assertFalse(result.diagnostics["well_conditioned"])
        self.assertGreater(
            result.diagnostics["equilibrated_effective_condition_number"], 1e8,
        )
        with self.assertRaises(np.linalg.LinAlgError):
            weighted_minimum_correction(
                np.zeros(3), matrix, np.zeros(2), condition_limit=1e8,
                require_well_conditioned=True,
            )

    def test_solution_is_minimal_against_nullspace_perturbation(self):
        preferred = np.asarray((0.2, -0.4, 0.1, 0.3))
        weights = np.asarray((1.0, 2.0, 3.0, 4.0))
        matrix = np.asarray(((1.0, 1.0, 0.0, 0.0), (0.0, 1.0, 1.0, 1.0)))
        rhs = np.asarray((0.8, -0.2))
        result = weighted_minimum_correction(
            preferred, matrix, rhs, correction_weights=weights,
        )
        _, _, vh = np.linalg.svd(matrix, full_matrices=True)
        null = vh[2]
        base = np.linalg.norm(weights * (result.profile - preferred))
        for scale in (-2.0, -0.3, 0.3, 2.0):
            candidate = result.profile + scale * null
            self.assertTrue(np.allclose(matrix @ candidate, rhs))
            self.assertGreater(np.linalg.norm(weights * (candidate - preferred)), base)

    def test_flattened_coupled_field_rows_are_supported(self):
        # Two three-node profiles flattened as (Phi[0:3], chi[0:3]).
        preferred = np.zeros(6)
        matrix = np.zeros((2, 6))
        matrix[0, 0] = 1.0
        matrix[0, 3] = 2.0
        matrix[1, 2] = -1.0
        matrix[1, 5] = 1.0
        result = weighted_minimum_correction(
            preferred, matrix, np.asarray((1.0, 0.5)),
        )
        self.assertTrue(np.allclose(matrix @ result.profile, (1.0, 0.5)))
        self.assertEqual(result.diagnostics["unknown_count"], 6)

    def test_flattened_coupled_fields_support_owner_endpoint_closure(self):
        # A manufactured h_zz/Phi-like block: each compact wall has two rows
        # whose endpoint coefficients couple both acceleration fields.
        node_count = 5
        derivative = np.asarray((
            (-2.0, 2.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, -2.0, 2.0),
        ))
        known_hzz = np.asarray((0.4, 0.2, -0.1, 0.3, -0.5))
        known_phi = np.asarray((-0.3, 0.1, 0.6, -0.2, 0.7))
        known = np.r_[known_hzz, known_phi]
        preferred = known.copy()
        adjustable = np.asarray((0, node_count - 1, node_count, 2 * node_count - 1))
        preferred[adjustable] += np.asarray((3.0, -4.0, 5.0, -6.0))

        matrix = np.zeros((4, 2 * node_count))
        matrix[0, :node_count] = derivative[0]
        matrix[0, 0] += 0.4
        matrix[0, node_count] = 0.7
        matrix[1, :node_count] = derivative[-1]
        matrix[1, node_count - 1] -= 0.2
        matrix[1, -1] = -0.6
        matrix[2, node_count:] = derivative[0]
        matrix[2, node_count] += 0.3
        matrix[2, 0] = 0.5
        matrix[3, node_count:] = derivative[-1]
        matrix[3, -1] += 0.1
        matrix[3, node_count - 1] = -0.4

        result = weighted_minimum_correction_on_indices(
            preferred, matrix, matrix @ known, adjustable,
        )
        self.assertTrue(np.allclose(result.profile, known, atol=1e-14))
        self.assertEqual(result.diagnostics["numerical_rank"], 4)
        self.assertTrue(result.diagnostics["full_row_rank"])
        self.assertEqual(result.diagnostics["maximum_fixed_preference_deviation"], 0.0)
        self.assertLess(result.diagnostics["constraint_maximum_absolute_residual"], 1e-14)

    def test_exact_full_preference_plus_conflicting_wall_rows_is_inconsistent(self):
        preferred = np.asarray((0.0, 1.0, 0.0))
        wall = np.asarray(((1.0, -1.0, 0.0), (0.0, 1.0, -1.0)))
        constraints = np.vstack((np.eye(3), wall))
        rhs = np.r_[preferred, (0.0, 0.0)]
        with self.assertRaises(InconsistentBoundaryConstraints):
            weighted_minimum_correction(preferred, constraints, rhs)
        owner = weighted_minimum_correction_on_indices(
            preferred, wall, np.zeros(2), (0, 2),
        )
        self.assertTrue(np.allclose(owner.profile, (1.0, 1.0, 1.0)))
        self.assertEqual(owner.diagnostics["fixed_preference_indices"], [1])
        self.assertEqual(owner.diagnostics["maximum_fixed_preference_deviation"], 0.0)

    def test_indexed_owner_rejects_consistent_rank_deficiency(self):
        with self.assertRaises(DegenerateBoundaryOwnership) as captured:
            weighted_minimum_correction_on_indices(
                np.zeros(2),
                np.asarray(((1.0, 0.0), (2.0, 0.0))),
                np.asarray((1.0, 2.0)),
                (0, 1),
            )
        diagnostic = captured.exception.diagnostics
        self.assertFalse(diagnostic["owner_raw_full_row_rank"])
        self.assertFalse(diagnostic["owner_resolution_passes"])

    def test_indexed_owner_rejects_ill_conditioned_block(self):
        with self.assertRaises(DegenerateBoundaryOwnership) as captured:
            weighted_minimum_correction_on_indices(
                np.zeros(2),
                np.asarray(((1.0, 0.0), (1.0, 1.0e-9))),
                np.zeros(2),
                (0, 1),
                condition_limit=1.0e8,
            )
        diagnostic = captured.exception.diagnostics
        self.assertFalse(diagnostic["owner_condition_passes"])
        self.assertGreater(
            diagnostic["owner_full_row_normalized_condition_number"], 1.0e8,
        )


class TestSimultaneousProfileClosures(unittest.TestCase):
    def setUp(self):
        self.z = np.linspace(1.0, 2.0, 33)
        self.dz = derivative_matrix(self.z, 1, 7)

    def test_frozen_nonlinear_wall_rows_recover_known_profile(self):
        x = self.z - self.z[0]
        known = 0.3 + 0.2 * x - 0.1 * x**2 + 0.05 * x**3
        derivative = self.dz.toarray()
        lower_robin, upper_robin = 0.7, -0.25
        lower_forcing = -(
            derivative[0] @ known + lower_robin * known[0]
        )
        upper_forcing = -(
            derivative[-1] @ known + upper_robin * known[-1]
        )
        rows, rhs = compact_wall_rows(
            self.dz,
            lower_robin=lower_robin,
            upper_robin=upper_robin,
            lower_forcing=lower_forcing,
            upper_forcing=upper_forcing,
        )
        self.assertTrue(np.allclose(rows @ known, rhs, atol=1e-13))
        result = close_regular_axis_wall_profile(
            known,
            self.dz,
            lower_robin=lower_robin,
            upper_robin=upper_robin,
            lower_forcing=lower_forcing,
            upper_forcing=upper_forcing,
        )
        self.assertLess(np.max(np.abs(result.profile - known)), 1e-13)
        self.assertEqual(
            result.diagnostics["closure_kind"],
            "regular_axis_plus_compact_walls",
        )

    def test_axis_profile_and_wall_rows_are_closed_simultaneously(self):
        x = self.z - self.z[0]
        preferred = 0.2 + np.sin(1.3 * x) + 0.1 * x**2
        rows, rhs = compact_wall_rows(self.dz)
        sequential_residual = rows @ preferred - rhs
        self.assertGreater(np.max(np.abs(sequential_residual)), 0.1)
        result = close_regular_axis_wall_profile(preferred, self.dz)
        self.assertLess(result.diagnostics["closed_wall_maximum_absolute_residual"], 1e-12)
        self.assertGreater(
            result.diagnostics["regular_axis_preferred_maximum_absolute_deviation"], 0.0,
        )
        self.assertTrue(result.diagnostics["constraint_residual_passes"])
        self.assertEqual(
            result.diagnostics["open_regular_axis_preferred_maximum_absolute_deviation"],
            0.0,
        )
        self.assertFalse(result.diagnostics["exact_full_preference_plus_walls_consistent"])
        self.assertEqual(result.diagnostics["adjustment_scope"], "selected_indices")

    def test_owner_solve_handles_rows_coupled_across_both_endpoints(self):
        # A dense manufactured endpoint stencil makes each wall row depend on
        # both wall-owned values.  Solving either endpoint in isolation would
        # use a stale value for the other endpoint.
        derivative = np.zeros((5, 5))
        derivative[0] = (2.0, -0.5, 0.25, 0.0, 1.0)
        derivative[-1] = (1.5, 0.0, -0.25, 0.5, -3.0)
        known = np.asarray((0.4, -0.2, 0.3, 0.7, -0.6))
        preferred = known.copy()
        preferred[[0, -1]] = (8.0, -9.0)
        lower_robin, upper_robin = 0.3, -0.2
        lower_forcing = -(
            derivative[0] @ known + lower_robin * known[0]
        )
        upper_forcing = -(
            derivative[-1] @ known + upper_robin * known[-1]
        )
        result = close_regular_axis_wall_profile(
            preferred,
            derivative,
            lower_robin=lower_robin,
            upper_robin=upper_robin,
            lower_forcing=lower_forcing,
            upper_forcing=upper_forcing,
        )
        self.assertTrue(np.allclose(result.profile, known, atol=1e-13))
        self.assertEqual(result.diagnostics["numerical_rank"], 2)
        self.assertTrue(result.diagnostics["full_row_rank"])
        self.assertLess(result.diagnostics["closed_wall_maximum_absolute_residual"], 1e-13)

    def test_outer_outgoing_and_wall_rows_are_closed_simultaneously(self):
        current = np.zeros(len(self.z))
        targets = 1.0 + 0.2 * np.sin(2.0 * np.pi * self.z[1:-1])
        rows, rhs = compact_wall_rows(self.dz)
        sequential = current.copy()
        sequential[1:-1] = targets
        self.assertGreater(np.max(np.abs(rows @ sequential - rhs)), 1.0)
        result = close_outer_face_wall_profile(current, targets, self.dz)
        self.assertLess(result.diagnostics["closed_wall_maximum_absolute_residual"], 1e-12)
        self.assertEqual(result.diagnostics["outgoing_target_l2_residual"], 0.0)
        self.assertEqual(
            result.diagnostics["closure_kind"],
            "outer_outgoing_plus_compact_walls",
        )
        self.assertEqual(result.diagnostics["outgoing_indices"], list(range(1, 32)))
        self.assertFalse(result.diagnostics["exact_full_preference_plus_walls_consistent"])

    def test_outgoing_weight_protects_selected_open_node(self):
        current = np.zeros(len(self.z))
        targets = np.linspace(1.0, 2.0, len(self.z) - 2)
        ordinary = close_outer_face_wall_profile(
            current, targets, self.dz, adjustment_scope="all_nodes",
        )
        weights = np.ones(len(self.z))
        weights[1] = 100.0
        protected = close_outer_face_wall_profile(
            current, targets, self.dz, correction_weights=weights,
            adjustment_scope="all_nodes",
        )
        ordinary_error = abs(ordinary.profile[1] - targets[0])
        protected_error = abs(protected.profile[1] - targets[0])
        self.assertLess(protected_error, ordinary_error)
        self.assertLess(protected.diagnostics["closed_wall_maximum_absolute_residual"], 1e-12)

    def test_owner_last_and_all_node_projector_are_distinct(self):
        current = np.zeros(len(self.z))
        targets = np.linspace(0.5, 1.5, len(self.z) - 2)
        owner = close_outer_face_wall_profile(current, targets, self.dz)
        projector = close_outer_face_wall_profile(
            current, targets, self.dz, adjustment_scope="all_nodes",
        )
        self.assertEqual(owner.diagnostics["outgoing_target_l2_residual"], 0.0)
        self.assertGreater(projector.diagnostics["outgoing_target_l2_residual"], 0.0)
        self.assertGreater(
            owner.diagnostics["weighted_correction_l2"],
            projector.diagnostics["weighted_correction_l2"],
        )
        self.assertLess(owner.diagnostics["closed_wall_maximum_absolute_residual"], 1e-12)
        self.assertLess(projector.diagnostics["closed_wall_maximum_absolute_residual"], 1e-12)

    def test_invalid_outer_target_map_is_rejected(self):
        current = np.zeros(len(self.z))
        with self.assertRaisesRegex(ValueError, "targets and open-face indices"):
            close_outer_face_wall_profile(
                current, np.ones(2), self.dz, outgoing_indices=np.asarray((1, 1)),
            )

    def test_wall_owned_corners_cannot_be_outgoing_target_indices(self):
        current = np.zeros(len(self.z))
        for corner in (0, len(current) - 1):
            with self.subTest(corner=corner):
                with self.assertRaisesRegex(ValueError, "open-face indices"):
                    close_outer_face_wall_profile(
                        current,
                        np.asarray((1.0,)),
                        self.dz,
                        outgoing_indices=np.asarray((corner,)),
                    )

    def test_owner_wrapper_fails_closed_on_tiny_pivot_and_huge_correction(self):
        derivative = np.zeros((3, 3))
        derivative[0, 0] = 1.0e-16
        derivative[-1, -1] = 1.0
        with self.assertRaises(DegenerateBoundaryOwnership) as captured:
            close_regular_axis_wall_profile(
                np.zeros(3),
                derivative,
                lower_forcing=-1.0,
                upper_forcing=-1.0,
            )
        diagnostic = captured.exception.diagnostics
        self.assertEqual(diagnostic["raw_numerical_rank"], 1)
        self.assertEqual(diagnostic["owner_raw_numerical_rank"], 1)
        self.assertLess(diagnostic["owner_raw_smallest_singular_value"], 1.0e-12)
        self.assertFalse(diagnostic["owner_absolute_pivot_passes"])
        self.assertFalse(diagnostic["owner_condition_passes"])
        self.assertFalse(diagnostic["owner_resolution_passes"])
        self.assertGreater(diagnostic["maximum_absolute_correction"], 1.0e15)

        probe = close_regular_axis_wall_profile(
            np.zeros(3),
            derivative,
            lower_forcing=-1.0,
            upper_forcing=-1.0,
            require_resolved_owner=False,
        )
        self.assertFalse(probe.diagnostics["owner_resolution_passes"])
        self.assertGreater(probe.diagnostics["maximum_absolute_correction"], 1.0e15)


if __name__ == "__main__":
    unittest.main()
