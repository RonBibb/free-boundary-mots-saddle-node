import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.validated_capped_surface_shooting import VInterval
from run_test4b_validated_interval_no_horizon_certificate import (
    BASE_CELL_COUNT,
    RHO_BOUNDS,
    base_cell,
    cell_containing,
    summarize_interval_probes,
)


class Test4BValidatedPipelineTests(unittest.TestCase):
    def test_base_cells_cover_exact_launch_band_without_gaps(self):
        cells = [base_cell(index) for index in range(BASE_CELL_COUNT)]
        self.assertEqual(cells[0].lower, RHO_BOUNDS[0])
        self.assertEqual(cells[-1].upper, RHO_BOUNDS[1])
        for left, right in zip(cells[:-1], cells[1:]):
            self.assertEqual(left.upper, right.lower)

    def test_cell_lookup_is_deterministic_at_minimum_navigation_points(self):
        for value in (1.2046626301041539, 1.2046482837562702):
            index = cell_containing(value)
            self.assertTrue(base_cell(index).contains(value))

    def test_unresolved_probe_can_never_aggregate_to_pass(self):
        probes = {
            "G9": [{"classification": "brane_root_free_positive"}],
            "G10": [{"classification": "unresolved_step"}],
            "A794_G7": [{"classification": "unresolved_zero_residual"}],
        }
        summary = summarize_interval_probes(probes)
        self.assertEqual(summary["closed_probe_count"], 1)
        self.assertEqual(summary["unresolved_probe_count"], 2)

    def test_interval_point_and_hull_serial_boundaries(self):
        point = VInterval.point(0.5)
        hull = point.hull(VInterval(0.4, 0.6))
        self.assertEqual((hull.lower, hull.upper), (0.4, 0.6))


if __name__ == "__main__":
    unittest.main()
