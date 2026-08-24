import unittest

from run_corrected_A790_t008_long_evolution import (
    classify_long_run,
    motion_summary,
)


class A790LongEvolutionProtocolTests(unittest.TestCase):
    def test_classification_preserves_persistence_review(self):
        acceptance = {"evolution": True, "transfer": False, "archive": True}
        self.assertEqual(
            classify_long_run(acceptance, surfaces_persist=True),
            ("review", "two_grid_R8_persistence_candidate_with_long_evolution_review"),
        )
        self.assertEqual(
            classify_long_run(acceptance, surfaces_persist=False),
            ("review", "unresolved_or_lost_branch"),
        )

    def test_motion_summary_uses_ordered_endpoint_and_area_histories(self):
        history = []
        for index in range(3):
            history.append({
                "search": {"clusters": [
                    {"signature": [1.0, 1.4 - 0.1 * index]},
                    {"signature": [1.1, 1.5 + 0.1 * index]},
                ]},
                "representatives": [
                    {"geometry": {"one_sided_cap_area": 40.0 - index}},
                    {"geometry": {"one_sided_cap_area": 41.0 + index}},
                ],
            })
        result = motion_summary({"G7": history})["G7"]
        self.assertLess(result["inner"]["area_relative_change"], 0.0)
        self.assertGreater(result["outer"]["area_relative_change"], 0.0)
        self.assertTrue(result["inner"]["area_monotone"])
        self.assertTrue(result["outer"]["brane_monotone"])


if __name__ == "__main__":
    unittest.main()
