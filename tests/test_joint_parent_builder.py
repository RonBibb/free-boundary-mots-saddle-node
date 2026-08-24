import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.anisotropic_initial_data import anisotropic_initial_data_residual
from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_builder import joint_parent_jacobian, joint_parent_residual
from bhps.junction_preservation_diagnostic import (
    wall_junction_rows,
    wall_source_coefficients,
)


class JointParentBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.z = np.linspace(1.0, 2.0, 9)
        cls.r = np.linspace(0.0, 3.0, 9)
        zz, rr = np.meshgrid(cls.z, cls.r, indexing="ij")
        x = (zz - cls.z[0]) / (cls.z[-1] - cls.z[0])
        cls.q = 0.08 + 0.015 * np.sin(1.3 * zz) * np.exp(-0.12 * rr**2)
        cls.phi = 0.11 * np.cos(0.8 * zz) * np.exp(-0.08 * rr**2) + 0.025 * x
        envelope = x * (1.0 - x) * np.exp(-0.15 * rr**2)
        cls.a = 0.025 * envelope + 0.004 * x
        cls.b = -0.013 * envelope + 0.003 * x
        cls.c = -0.009 * envelope - 0.002 * x
        cls.chi_r = -0.018 * rr * np.exp(-0.2 * rr**2) * (1.0 + 0.1 * x)
        cls.chi_z = 0.006 * np.exp(-0.2 * rr**2) * (1.0 - 2.0 * x)
        cls.reference_q = 0.075 + 0.004 * x * np.exp(-0.1 * rr**2)
        cls.reference_phi = 0.035 * (1.0 - x) + 0.012 * x
        cls.background = {
            "mass_squared": 0.65,
            "wall_stiffness": 4.2,
            "v0": 0.18,
            "v1": -0.07,
            "beta_a": 0.72,
            "beta_b": -0.58,
            "wall_potential_a": 0.031,
            "wall_potential_b": 0.019,
        }

    def _arguments(self, reference_q=None, reference_phi=None):
        return (
            self.q,
            self.phi,
            self.z,
            self.r,
            self.a,
            self.b,
            self.c,
            self.background,
            self.chi_r,
            self.chi_z,
            self.reference_q if reference_q is None else reference_q,
            self.reference_phi if reference_phi is None else reference_phi,
            7,
        )

    def _native_position(self):
        psi = 1.0 / (self.z[:, None] + self.q)
        sphere = psi**2 * np.exp(2.0 * self.c)
        radial = psi**2 * np.exp(2.0 * self.b)
        position = np.zeros(self.q.shape + (9,))
        position[:, :, 2] = -psi**2
        position[:, :, 3] = sphere
        position[:, 1:, 4] = (
            radial[:, 1:] - sphere[:, 1:]
        ) / self.r[None, 1:] ** 2
        position[:, :, 6] = psi**2 * np.exp(2.0 * self.a)
        position[:, :, 7] = self.phi
        return position

    def test_hybrid_ownership_and_absolute_native_wall_rows(self):
        hybrid = joint_parent_residual(*self._arguments())
        established = anisotropic_initial_data_residual(*self._arguments())
        n = self.q.size
        hybrid_metric = hybrid[:n].reshape(self.q.shape)
        hybrid_phi = hybrid[n:].reshape(self.q.shape)
        established_metric = established[:n].reshape(self.q.shape)
        established_phi = established[n:].reshape(self.q.shape)

        # Every open-z node, including the radial outer face, retains the
        # established raw-reference-defect functional exactly.
        np.testing.assert_array_equal(hybrid_metric[1:-1], established_metric[1:-1])
        np.testing.assert_array_equal(hybrid_phi[1:-1], established_phi[1:-1])

        position = self._native_position()
        velocity = np.zeros_like(position)
        for name, index in (("lower", 0), ("upper", -1)):
            native = wall_junction_rows(
                position,
                velocity,
                self.z,
                self.r,
                self.background,
                name,
                7,
            )
            np.testing.assert_allclose(
                hybrid_metric[index],
                native["components"]["sphere"]["J"],
                rtol=2e-14,
                atol=2e-14,
            )
            np.testing.assert_allclose(
                hybrid_phi[index],
                native["separate_rows"]["Phi_robin"],
                rtol=2e-14,
                atol=2e-14,
            )

        # The compact walls own the radial outer corners; the legacy radial
        # row is therefore not retained at either corner.
        self.assertGreater(
            abs(hybrid_metric[0, -1] - established_metric[0, -1]), 1e-5,
        )
        self.assertGreater(
            abs(hybrid_metric[-1, -1] - established_metric[-1, -1]), 1e-5,
        )

    def test_reference_defect_is_never_subtracted_from_wall_rows(self):
        baseline = joint_parent_residual(*self._arguments())
        zz, rr = np.meshgrid(self.z, self.r, indexing="ij")
        alternate_q = self.reference_q + 0.012 * np.sin(1.1 * zz) * np.exp(-0.1 * rr)
        alternate_phi = self.reference_phi - 0.009 * np.cos(0.7 * zz) * np.exp(-0.2 * rr)
        alternate = joint_parent_residual(
            *self._arguments(alternate_q, alternate_phi),
        )
        n = self.q.size
        baseline_blocks = baseline.reshape(2, *self.q.shape)
        alternate_blocks = alternate.reshape(2, *self.q.shape)

        np.testing.assert_array_equal(
            baseline_blocks[:, (0, -1), :],
            alternate_blocks[:, (0, -1), :],
        )
        self.assertGreater(
            np.max(np.abs(baseline_blocks[:, 1:-1] - alternate_blocks[:, 1:-1])),
            1e-6,
        )

    def test_analytic_jacobian_matches_off_manifold_directional_difference(self):
        jacobian = joint_parent_jacobian(*self._arguments())
        rng = np.random.default_rng(125)
        direction = rng.normal(size=2 * self.q.size)
        direction /= np.linalg.norm(direction)
        dq = direction[: self.q.size].reshape(self.q.shape)
        dphi = direction[self.q.size :].reshape(self.q.shape)
        step = 3e-7
        plus = joint_parent_residual(
            self.q + step * dq,
            self.phi + step * dphi,
            *self._arguments()[2:],
        )
        minus = joint_parent_residual(
            self.q - step * dq,
            self.phi - step * dphi,
            *self._arguments()[2:],
        )
        finite_difference = (plus - minus) / (2.0 * step)
        analytic = jacobian @ direction
        relative_error = np.linalg.norm(finite_difference - analytic) / np.linalg.norm(
            finite_difference
        )
        self.assertLess(relative_error, 4e-6)

        wall_metric = joint_parent_residual(*self._arguments())[: self.q.size].reshape(
            self.q.shape
        )
        self.assertGreater(np.max(np.abs(wall_metric[(0, -1), :])), 1e-2)

    def test_wall_jacobian_contains_off_manifold_normalization_term(self):
        jacobian = joint_parent_jacobian(*self._arguments())
        dz = derivative_matrix(self.z, 1, 7).toarray()
        psi = 1.0 / (self.z[:, None] + self.q)
        sphere = psi**2 * np.exp(2.0 * self.c)
        scale = psi * np.exp(self.a)
        radial_index = 4
        wall_index = 0
        source = wall_source_coefficients(
            self.phi[wall_index], self.background, "lower",
        )
        orientation = source["orientation"]
        beta = source["beta"][radial_index]
        local_scale = scale[wall_index, radial_index]
        local_sphere = sphere[wall_index, radial_index]
        local_psi = psi[wall_index, radial_index]
        dscale_dq = -local_psi * local_scale
        dsphere_dq = -2.0 * local_psi * local_sphere
        robin = (
            (dz @ sphere)[wall_index, radial_index]
            + 2.0 * beta * local_scale * local_sphere
        )
        drobin_dq = (
            dz[wall_index, wall_index] * dsphere_dq
            + 2.0
            * beta
            * (dscale_dq * local_sphere + local_scale * dsphere_dq)
        )
        numerator_only = orientation * drobin_dq / (2.0 * local_scale)
        normalization_term = (
            -orientation * robin * dscale_dq / (2.0 * local_scale**2)
        )
        row = wall_index * len(self.r) + radial_index
        column = row
        self.assertGreater(abs(normalization_term), 1e-3)
        self.assertAlmostEqual(
            jacobian[row, column],
            numerator_only + normalization_term,
            places=12,
        )
        self.assertGreater(
            abs(jacobian[row, column] - numerator_only), 1e-3,
        )


if __name__ == "__main__":
    unittest.main()
