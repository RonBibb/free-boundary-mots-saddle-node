import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.test4c_aggregation import grade_test4c
from run_test4c_correlated_validated_bounded_certificate import (
    LABEL_POLICY,
    launch_cell,
)


class Test4CCoverTests(unittest.TestCase):
    def test_each_sealed_base_cover_is_exact_and_adjacent(self):
        for label, policy in LABEL_POLICY.items():
            cells = [launch_cell(label, index) for index in range(256)]
            self.assertEqual(cells[0].lower, policy["lower"])
            self.assertEqual(cells[-1].upper, policy["upper"])
            for left, right in zip(cells[:-1], cells[1:]):
                self.assertEqual(left.upper, right.lower)

    def test_binary_children_cover_parent_exactly(self):
        parent = launch_cell("G9", 217, "101")
        left = launch_cell("G9", 217, "1010")
        right = launch_cell("G9", 217, "1011")
        self.assertEqual(left.lower, parent.lower)
        self.assertEqual(left.upper, right.lower)
        self.assertEqual(right.upper, parent.upper)

    def test_one_unresolved_required_leaf_forces_review(self):
        status = grade_test4c(
            True,
            [{"classification": "unresolved_correlated_step"}],
            [{"classification": "root_free_positive"}],
            2, 0,
        )
        self.assertEqual(status, "REVIEW")

    def test_fail_requires_validated_and_independently_confirmed_root(self):
        leaf = [{"classification": "validated_root"}]
        self.assertEqual(grade_test4c(True, leaf, [], 2, 0, False), "REVIEW")
        self.assertEqual(grade_test4c(True, leaf, [], 2, 0, True), "FAIL")


if __name__ == "__main__":
    unittest.main()
