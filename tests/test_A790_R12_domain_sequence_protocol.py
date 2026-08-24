import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_corrected_A790_R12_domain_sequence import (
    classify_domain_sequence,
    valid_single_transition,
)


class R12DomainSequenceProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rules = {
            "clean_R12_construction_and_initial_transfer": True,
            "both_initial_searches_admit_zero_caps": True,
            "R12_evolutions_and_cross_grid_fields_pass": True,
            "fine_transition_and_long_pair_persistence_pass": True,
            "R10_R12_and_R12_cross_grid_surface_and_field_transfers_pass": True,
            "causal_boundary_timing_pass": True,
            "state_archive_complete_and_finite": True,
        }
        self.counts = {
            "G7": [0, 0, 0, 0, 0, 0, 0, 2],
            "G8": [0, 0, 0, 0, 0, 0, 0, 2],
        }

    def test_valid_single_transition(self):
        self.assertTrue(valid_single_transition([0, 0, 0, 2, 2]))
        self.assertTrue(valid_single_transition([0, 0, 0]))
        self.assertFalse(valid_single_transition([0, 2, 0]))
        self.assertFalse(valid_single_transition([0, 3]))

    def test_agreement_category_is_prospective(self):
        stats = {
            "median_absolute_change_ratio": 0.5,
            "fraction_absolute_change_ratios_below_one": 0.8,
            "fraction_signed_changes_continuing": 0.5,
        }
        self.assertEqual(
            classify_domain_sequence(self.rules, self.counts, stats, True),
            "R10_R12_agreement_R8_boundary_sensitive_outlier_supported",
        )

    def test_continuing_drift_category(self):
        stats = {
            "median_absolute_change_ratio": 0.9,
            "fraction_absolute_change_ratios_below_one": 0.6,
            "fraction_signed_changes_continuing": 0.8,
        }
        self.assertEqual(
            classify_domain_sequence(self.rules, self.counts, stats, True),
            "continuing_domain_drift",
        )

    def test_invalid_audit_has_priority(self):
        self.rules["R12_evolutions_and_cross_grid_fields_pass"] = False
        stats = {
            "median_absolute_change_ratio": 0.5,
            "fraction_absolute_change_ratios_below_one": 1.0,
            "fraction_signed_changes_continuing": 1.0,
        }
        self.assertEqual(
            classify_domain_sequence(self.rules, self.counts, stats, True),
            "invalid_R12_domain_audit",
        )


if __name__ == "__main__":
    unittest.main()

