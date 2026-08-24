import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.anisotropic_capped_surface import _splines
from bhps.capped_surface_shooting_audit import (
    adjacent_status_cells,
    regular_axis_initial_state,
    summarize_shooting_scan,
)


class CappedSurfaceShootingAuditTests(unittest.TestCase):
    def test_flat_regular_axis_series(self):
        z = np.linspace(0.8, 3.0, 9)
        r = np.linspace(0.0, 2.0, 11)
        ones = np.ones((len(z), len(r)))
        zeros = np.zeros_like(ones)
        splines = _splines(z, r, ones, zeros, zeros, zeros)
        state, diagnostic = regular_axis_initial_state(0.7, 1e-3, z[-1], splines)
        self.assertAlmostEqual(diagnostic["axis_barrier"], 3.0, places=12)
        self.assertAlmostEqual(diagnostic["axis_second_derivative"], 2.1, places=12)
        self.assertAlmostEqual(state[0], 0.7 + 1.05e-6, places=14)
        self.assertAlmostEqual(state[1], 0.0021, places=14)

    def test_summary_detects_sign_change_and_tangent_sample(self):
        records = [
            {"axis_radius": 1.0, "status": "reached_brane", "brane_residual": 0.2},
            {"axis_radius": 1.1, "status": "reached_brane", "brane_residual": -0.01},
            {"axis_radius": 1.2, "status": "reached_brane", "brane_residual": 0.3},
        ]
        summary = summarize_shooting_scan(records)
        self.assertEqual(summary["sign_change_count"], 2)
        self.assertAlmostEqual(summary["minimum_absolute_residual"], 0.01)
        self.assertFalse(summary["all_reached_residuals_positive"])

    def test_adjacent_status_cells_preserve_mixed_as_unresolved(self):
        records = [
            {"axis_radius": 1.0, "status": "upper_band_exit"},
            {"axis_radius": 1.1, "status": "upper_band_exit"},
            {"axis_radius": 1.2, "status": "reached_brane", "brane_residual": 0.1},
            {"axis_radius": 1.3, "status": "reached_brane", "brane_residual": 0.2},
        ]
        cells = adjacent_status_cells(records)
        self.assertEqual(cells["classes"]["same_exit_endpoints"], 1)
        self.assertEqual(cells["classes"]["mixed_or_unresolved"], 1)
        self.assertEqual(cells["classes"]["positive_residual_endpoints"], 1)


if __name__ == "__main__":
    unittest.main()
