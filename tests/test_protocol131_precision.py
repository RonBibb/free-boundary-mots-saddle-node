import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.joint_parent_builder import joint_parent_residual
from bhps.protocol131_precision import (
    compensated_dot,
    extended_precision_residual,
    joint_parent_residual_longdouble,
    longdouble_capability,
    reevaluate_wall_row_mpmath,
)


class Protocol131PrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.z = np.linspace(1.0, 2.0, 9)
        cls.r = np.linspace(0.0, 3.0, 10)
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

    def arguments(self, q=None, phi=None, a=None, b=None, c=None):
        return (
            self.q if q is None else q,
            self.phi if phi is None else phi,
            self.z,
            self.r,
            self.a if a is None else a,
            self.b if b is None else b,
            self.c if c is None else c,
            self.background,
            self.chi_r,
            self.chi_z,
            self.reference_q,
            self.reference_phi,
            7,
        )

    def test_full_longdouble_residual_matches_float64_manufactured_case(self):
        extended = joint_parent_residual_longdouble(*self.arguments())
        production = joint_parent_residual(*self.arguments())
        self.assertEqual(extended.dtype, np.dtype(np.longdouble))
        self.assertEqual(extended.shape, production.shape)
        self.assertTrue(np.all(np.isfinite(extended)))
        np.testing.assert_allclose(
            np.asarray(extended, dtype=np.float64), production,
            rtol=2e-11, atol=2e-11,
        )

    def test_reference_defect_cancels_exactly_at_open_compact_nodes(self):
        zeros = np.zeros_like(self.a)
        residual = joint_parent_residual_longdouble(
            *self.arguments(
                q=self.reference_q,
                phi=self.reference_phi,
                a=zeros,
                b=zeros,
                c=zeros,
            )
        ).reshape(2, *self.q.shape)
        np.testing.assert_array_equal(
            residual[:, 1:-1, :],
            np.zeros_like(residual[:, 1:-1, :]),
        )
        # Absolute wall rows are deliberately not reference-defect balanced.
        self.assertGreater(float(np.max(np.abs(residual[:, (0, -1), :]))), 1e-3)

    def test_compensated_dot_recovers_a_lost_small_term(self):
        scale = np.longdouble(1e20)
        result = compensated_dot(
            np.asarray((scale, np.longdouble(1.0), -scale)),
            np.ones(3, dtype=np.longdouble),
        )
        self.assertEqual(result, np.longdouble(1.0))
        with self.assertRaises(ValueError):
            compensated_dot(np.ones(2), np.ones(3))

    def test_mpmath_wall_replay_matches_longdouble_rows(self):
        try:
            import mpmath as mp
        except ImportError:
            self.skipTest("mpmath is unavailable")
        extended = joint_parent_residual_longdouble(*self.arguments()).reshape(
            2, *self.q.shape,
        )
        for wall, index, radial in (("lower", 0, 3), ("upper", -1, 6)):
            replay = reevaluate_wall_row_mpmath(
                self.q,
                self.phi,
                self.z,
                self.a,
                self.c,
                self.background,
                wall,
                radial,
            )
            self.assertEqual(replay["dps"], 80)
            metric_delta = abs(
                replay["metric_row"] - mp.mpf(str(extended[0, index, radial]))
            )
            phi_delta = abs(
                replay["phi_row"] - mp.mpf(str(extended[1, index, radial]))
            )
            # Some NumPy builds alias longdouble to binary64.  The mpmath
            # control remains genuinely extended on those systems, while the
            # long-double lane can agree only to binary64 arithmetic scale.
            self.assertLess(metric_delta, mp.mpf("2e-14"))
            self.assertLess(phi_delta, mp.mpf("2e-14"))

    def test_capability_record_is_explicit_and_width_is_frozen(self):
        capability = longdouble_capability()
        self.assertIn("wider_than_float64", capability)
        self.assertGreaterEqual(capability["itemsize"], 8)
        with self.assertRaises(ValueError):
            joint_parent_residual_longdouble(*self.arguments()[:-1], 5)
        with self.assertRaises(ValueError):
            reevaluate_wall_row_mpmath(
                self.q,
                self.phi,
                self.z,
                self.a,
                self.c,
                self.background,
                "lower",
                0,
                dps=79,
            )

    def test_extended_wrapper_certifies_wall_only_load_bearing_rows(self):
        zeros = np.zeros_like(self.a)
        parent = {
            "q": self.reference_q,
            "phi": self.reference_phi,
            "z": self.z,
            "r": self.r,
            "a": zeros,
            "b": zeros,
            "c": zeros,
            "background": self.background,
            "chi_r": self.chi_r,
            "chi_z": self.chi_z,
            "reference_q": self.reference_q,
            "reference_phi": self.reference_phi,
        }
        binary = joint_parent_residual(*(
            self.reference_q, self.reference_phi, self.z, self.r,
            zeros, zeros, zeros, self.background, self.chi_r, self.chi_z,
            self.reference_q, self.reference_phi, 7,
        ))
        summary, arrays = extended_precision_residual(
            parent, binary, {}, {},
            {"modes": [], "mode_left_vectors": np.empty((len(binary), 0))},
        )
        self.assertTrue(summary["mp_certified"])
        self.assertTrue(summary["complete"])
        self.assertEqual(arrays["longdouble_residual"].shape, binary.shape)
        if not summary["capability"]["wider_than_float64"]:
            self.assertEqual(
                summary["eta_F_scope"],
                "80-digit-certified-load-bearing-wall-rows",
            )
            self.assertLessEqual(
                summary["eta_F"],
                max(
                    item["absolute_difference_from_binary64"]
                    for item in summary["mp_wall_rows"]
                ),
            )

        poisoned = binary.copy()
        poisoned[len(self.r) + 2] = 2e-10
        incomplete, _ = extended_precision_residual(
            parent, poisoned, {}, {},
            {"modes": [], "mode_left_vectors": np.empty((len(binary), 0))},
        )
        self.assertFalse(incomplete["mp_certified"])
        if not incomplete["capability"]["wider_than_float64"]:
            self.assertFalse(incomplete["complete"])


if __name__ == "__main__":
    unittest.main()
