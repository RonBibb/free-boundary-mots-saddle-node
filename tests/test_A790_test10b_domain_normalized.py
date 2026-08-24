import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.axisymmetric_reduced_wave_evolution import axisymmetric_principal_coefficients
from bhps.corrected_A790_test10b_domain_normalized import (
    brackets_overlap,
    classify_test10b,
    common_radius_invariants,
    exact_radial_index,
    first_detection_bracket,
    invariant_transfer,
    restrict_geometry,
    restriction_identity,
    tensor_domain_transfer,
    valid_persistent_pair_history,
)


class Test10BDomainNormalizedTests(unittest.TestCase):
    @staticmethod
    def parent():
        z = np.linspace(1.0, 2.0, 9)
        r = np.linspace(0.0, 12.0, 25)
        psi = np.ones((len(z), len(r)))
        zero = np.zeros_like(psi)
        fields = np.zeros((len(z), len(r), 9))
        fields[..., 2] = -1.0
        fields[..., 3] = fields[..., 6] = 1.0
        fields[..., 7] = 0.2
        fields[..., 8] = 0.1
        first = np.arange(3 * fields.size, dtype=float).reshape((3, *fields.shape))
        second = np.arange(9 * fields.size, dtype=float).reshape((3, 3, *fields.shape))
        jet = SimpleNamespace(
            z=z, r=r, reduced_fields=fields,
            reduced_first=first, reduced_second=second,
        )
        return {
            "name": "parent", "z": z, "r": r, "psi": psi,
            "a": zero, "b": zero, "c": zero, "phi": fields[..., 7],
            "background": {"test": True}, "mass_squared": 0.0,
            "fold_amplitude": 7.9, "selector_maximum": 1e-12,
            "reference_maximum_residual": 1e-12,
            "principal": axisymmetric_principal_coefficients(psi, zero, zero, zero),
            "jet_field": jet,
        }

    def test_exact_prefix_restriction_including_cut_jets(self):
        parent = self.parent()
        restricted = restrict_geometry(parent, 8.0, "R8")
        self.assertEqual(restricted["source_grid"], [9, 17])
        self.assertTrue(restriction_identity(parent, restricted)["passed"])
        self.assertEqual(restricted["restriction_endpoint_index"], 16)

    def test_identity_audit_detects_common_node_perturbation(self):
        parent = self.parent()
        restricted = restrict_geometry(parent, 8.0, "R8")
        restricted["jet_field"].reduced_fields[2, 3, 7] = np.nextafter(
            restricted["jet_field"].reduced_fields[2, 3, 7], np.inf,
        )
        self.assertFalse(restriction_identity(parent, restricted)["passed"])

    def test_endpoint_must_be_exact_parent_node(self):
        with self.assertRaisesRegex(ValueError, "exact parent node"):
            exact_radial_index(np.linspace(0.0, 12.0, 25), 8.1)

    def test_flat_common_invariants_transfer_exactly(self):
        parent = self.parent()
        r8 = restrict_geometry(parent, 8.0, "R8")
        r10 = restrict_geometry(parent, 10.0, "R10")
        left = common_radius_invariants(
            r8["jet_field"].reduced_fields, r8["z"], r8["r"],
        )
        right = common_radius_invariants(
            r10["jet_field"].reduced_fields, r10["z"], r10["r"],
        )
        self.assertEqual(invariant_transfer(left, right)["maximum"], 0.0)

    def test_common_invariants_reject_nonpositive_spatial_metric(self):
        parent = self.parent()
        restricted = restrict_geometry(parent, 8.0, "R8")
        state = restricted["jet_field"].reduced_fields.copy()
        state[..., 3] = -1.0
        with self.assertRaisesRegex(ValueError, "not positive"):
            common_radius_invariants(state, restricted["z"], restricted["r"])

    def test_identical_states_have_zero_tensor_transfer(self):
        parent = self.parent()
        restricted = restrict_geometry(parent, 8.0, "R8")
        initial = restricted["jet_field"].reduced_fields
        state = {"position": initial.copy(), "velocity": np.zeros_like(initial)}
        result = tensor_domain_transfer(
            state, state, restricted["z"], restricted["r"], initial,
        )
        for record in result.values():
            self.assertEqual(record["absolute_difference"], 0.0)

    def test_interior_metric_perturbation_has_nonzero_tensor_transfer(self):
        parent = self.parent()
        restricted = restrict_geometry(parent, 8.0, "R8")
        initial = restricted["jet_field"].reduced_fields
        left = {"position": initial.copy(), "velocity": np.zeros_like(initial)}
        right = {"position": initial.copy(), "velocity": np.zeros_like(initial)}
        right["position"][4, 4, 3] += 0.01
        result = tensor_domain_transfer(
            left, right, restricted["z"], restricted["r"], initial,
        )
        self.assertGreater(result["full_metric"]["absolute_difference"], 0.0)

    def test_history_bracket_and_classification(self):
        counts = [0, 0, 0, 2, 2]
        self.assertTrue(valid_persistent_pair_history(counts))
        self.assertFalse(valid_persistent_pair_history([0, 2, 0]))
        bracket = first_detection_bracket(counts, 0.1)
        self.assertEqual(bracket, [0.30000000000000004, 0.4])
        self.assertTrue(brackets_overlap(bracket, [0.35, 0.45]))
        self.assertEqual(
            classify_test10b(True, True, True, True, True, True, False)[0],
            "pass",
        )
        self.assertEqual(
            classify_test10b(True, True, True, True, True, True, True)[0],
            "fail",
        )
        self.assertEqual(
            classify_test10b(False, True, True, True, True, True, False),
            ("review", "invalid_common_parent_audit"),
        )


if __name__ == "__main__":
    unittest.main()
