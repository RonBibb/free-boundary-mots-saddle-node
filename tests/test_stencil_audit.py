import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from bhps.stencil_audit import manufactured_stencil_errors

class StencilAuditTests(unittest.TestCase):
    def test_manufactured_errors_decrease_under_refinement(self):
        coarse=manufactured_stencil_errors(17,25)
        fine=manufactured_stencil_errors(33,49)
        for key in ("interior_max_error","axis_max_error"):
            self.assertLess(fine[key],coarse[key])
        # The manufactured z-dependence is quadratic, so the second-order
        # one-sided Robin stencils are exact apart from floating-point noise.
        self.assertLess(fine["brane_A_max_error"],1e-13)
        self.assertLess(fine["brane_B_max_error"],1e-13)
        self.assertLess(fine["outer_boundary_max_error"],1e-13)

if __name__=="__main__":unittest.main()
