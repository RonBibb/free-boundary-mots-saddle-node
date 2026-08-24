import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.global_bvp_collocation import (
    global_collocation_sparsity,
    increasing_lobatto_nodes,
    lobatto_differentiation_matrix,
    shared_nodal_layout,
    unpack_shared_vector,
)
from bhps.validated_global_bvp import CONFIGURATIONS


class SharedNodeCollocationTests(unittest.TestCase):
    def test_lobatto_derivative_is_exact_for_polynomial(self):
        degree = 12
        nodes = increasing_lobatto_nodes(degree)
        values = nodes**5 - 2.0 * nodes**3 + 0.3 * nodes
        expected = 5.0 * nodes**4 - 6.0 * nodes**2 + 0.3
        actual = lobatto_differentiation_matrix(degree) @ values
        self.assertTrue(np.allclose(actual, expected, rtol=0.0, atol=2e-11))

    def test_shared_layout_has_protocol_size_and_no_duplicate_endpoints(self):
        configuration = CONFIGURATIONS[0]
        layout = shared_nodal_layout(configuration)
        self.assertEqual(layout["size"], 1714)
        vector = np.arange(layout["size"], dtype=float)
        state = unpack_shared_vector(vector, configuration)
        self.assertEqual(state["rho_blocks"].shape, (70, 13))
        self.assertEqual(state["w_blocks"].shape, (70, 13))
        self.assertTrue(np.array_equal(
            state["rho_blocks"][:-1, -1], state["rho_blocks"][1:, 0],
        ))
        self.assertTrue(np.array_equal(
            state["w_blocks"][:-1, -1], state["w_blocks"][1:, 0],
        ))

    def test_sparsity_is_square_and_covers_axis_and_local_blocks(self):
        configuration = CONFIGURATIONS[0]
        layout = shared_nodal_layout(configuration)
        pattern = global_collocation_sparsity(configuration)
        self.assertEqual(pattern.shape, (layout["size"], layout["size"]))
        self.assertEqual(pattern[:34, :34].nnz, 34 * 34)
        self.assertTrue(np.all(np.diff(pattern.indptr) > 0))


if __name__ == "__main__":
    unittest.main()
