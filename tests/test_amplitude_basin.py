import unittest

from bhps.amplitude_basin import (
    aggregate_basin_status,
    amplitude_tag,
    matched_radial_count,
    monotone_onset_diagnostic,
    persistent_late_pair,
    sampled_onset,
)


class AmplitudeBasinTests(unittest.TestCase):
    def test_tag_and_matched_spacing(self):
        self.assertEqual(amplitude_tag(7.84), "A784")
        self.assertEqual(matched_radial_count(121), 181)
        self.assertEqual(matched_radial_count(145), 217)
        with self.assertRaises(ValueError):
            matched_radial_count(122)

    def test_sampled_onset_requires_persistence(self):
        times = [0.0005, 0.001, 0.0015, 0.002]
        result = sampled_onset(times, [0, 0, 2, 2])
        self.assertEqual(result["classification"], "sampled_persistent_zero_to_two")
        self.assertEqual(result["bracket"], {"lower": 0.001, "upper": 0.0015})
        self.assertEqual(
            sampled_onset(times, [0, 2, 0, 2])["classification"],
            "nonpersistent_or_extra_branch",
        )

    def test_late_pair(self):
        counts = {
            f"{grid}_t{time}": 2
            for grid in ("G7", "G8") for time in ("0.003", "0.004")
        }
        self.assertTrue(persistent_late_pair(counts))
        counts["G8_t0.003"] = 0
        self.assertFalse(persistent_late_pair(counts))

    def test_aggregate_pass_and_review(self):
        fixed = {f"A{value}": {"primary_pass": True} for value in (784, 786, 788, 790, 792)}
        anchors = {f"A{value}": {"primary_pass": True} for value in (784, 790, 792)}
        passed = aggregate_basin_status(fixed, anchors, True)
        self.assertEqual(passed["status"], "pass")
        anchors["A784"]["primary_pass"] = False
        reviewed = aggregate_basin_status(fixed, anchors, True)
        self.assertEqual(reviewed["status"], "review")

    def test_hard_failure_and_onset_diagnostic(self):
        fixed = {f"A{value}": {"primary_pass": True} for value in (784, 786, 788, 790, 792)}
        anchors = {f"A{value}": {"primary_pass": True} for value in (784, 790, 792)}
        fixed["A786"]["hard_failure"] = True
        self.assertEqual(aggregate_basin_status(fixed, anchors, True)["status"], "fail")
        records = {
            "A784": {"sampled_onset": {"bracket": {"upper": 0.002}}},
            "A786": {"sampled_onset": {"bracket": {"upper": 0.0015}}},
            "A788": {"sampled_onset": {"bracket": {"upper": 0.001}}},
            "A790": {"sampled_onset": {"bracket": {"upper": 0.000625}}},
            "A792": {"sampled_onset": {"bracket": {"upper": 0.00025}}},
        }
        self.assertTrue(monotone_onset_diagnostic(records)["nonincreasing_where_defined"])


if __name__ == "__main__":
    unittest.main()
