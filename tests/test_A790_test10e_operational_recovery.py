import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_corrected_A790_test10e_operational_recovery as recovery


class Test10EOperationalRecovery(unittest.TestCase):
    def test_adapter_only_supplies_missing_runtime_metadata(self):
        surface = {
            "converged": True,
            "solver_success": True,
            "message": "ok",
            "in_domain": True,
            "iterations": 1,
            "mesh_nodes_used": 2,
            "rho_axis": 1.0,
            "rho_brane": 2.0,
            "rho_min": 1.0,
            "rho_max": 2.0,
            "boundary_slope_error": 0.0,
            "local_expansion_interior_maximum": 0.0,
            "local_expansion_full_maximum": 0.0,
            "ode_defect_maximum": 0.0,
            "primary_evaluator_crosscheck": {},
            "interior_point_count": 3,
        }
        presented = recovery.public_surface_with_unavailable_runtime(surface)
        self.assertIsNone(presented["runtime_seconds"])
        self.assertNotIn("runtime_seconds", surface)
        for key, value in surface.items():
            self.assertEqual(presented[key], value)

    def test_existing_runtime_is_preserved(self):
        surface = {"error": "diagnostic", "runtime_seconds": 3.5}
        self.assertEqual(
            recovery.public_surface_with_unavailable_runtime(surface), surface,
        )


if __name__ == "__main__":
    unittest.main()

