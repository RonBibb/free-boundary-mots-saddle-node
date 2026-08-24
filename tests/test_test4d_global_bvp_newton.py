import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.global_bvp_newton import (
    greedy_disjoint_row_column_groups,
    sparse_colored_central_jacobian,
)


class ColoredNewtonTests(unittest.TestCase):
    def test_columns_in_each_group_have_disjoint_supported_rows(self):
        pattern = csr_matrix(np.asarray([
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ], dtype=bool))
        groups = greedy_disjoint_row_column_groups(pattern)
        csc = pattern.tocsc()
        for group in groups:
            occupied = set()
            for column in group:
                rows = set(csc.indices[csc.indptr[column]:csc.indptr[column + 1]])
                self.assertTrue(occupied.isdisjoint(rows))
                occupied.update(rows)

    def test_colored_difference_recovers_sparse_linear_jacobian(self):
        matrix = np.asarray([
            [3.0, 0.0, -2.0, 0.0],
            [1.0, 4.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 5.0],
            [0.0, 0.0, 2.0, 1.0],
        ])
        pattern = csr_matrix(matrix != 0.0)
        jacobian, diagnostics = sparse_colored_central_jacobian(
            lambda value: matrix @ value,
            np.asarray([0.2, -0.4, 0.7, 1.1]),
            pattern,
        )
        self.assertTrue(np.allclose(jacobian.toarray(), matrix, atol=2e-11))
        self.assertEqual(
            diagnostics["residual_evaluation_count"],
            2 * diagnostics["color_count"],
        )


if __name__ == "__main__":
    unittest.main()
