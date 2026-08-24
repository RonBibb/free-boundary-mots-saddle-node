import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.corrected_A790_gauge_variations import (
    BASELINE,
    VARIATIONS,
    GaugeVariation,
    brackets_baseline,
)
from bhps.gh_source_driver import source_driver_rhs


class CorrectedA790GaugeVariationTests(unittest.TestCase):
    def test_variations_are_distinct_and_bracket_baseline(self):
        self.assertEqual([item.name for item in VARIATIONS], [
            "slow_soft", "fast_strong",
        ])
        self.assertTrue(brackets_baseline())
        self.assertTrue(all(item != BASELINE for item in VARIATIONS))
        self.assertTrue(all(item.target_power == BASELINE.target_power for item in VARIATIONS))

    def test_invalid_driver_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            GaugeVariation("invalid", 0.0, 1.0, 0.2, 0.2)

    def test_memory_initialization_preserves_fixed_source_time_jet(self):
        rng = np.random.default_rng(817)
        source = rng.normal(size=(3, 5, 3))
        target = rng.normal(size=source.shape)
        source_time = rng.normal(size=source.shape)
        advection = rng.normal(size=source.shape)
        for variation in (BASELINE, *VARIATIONS):
            memory = (
                source_time - advection
                + variation.driver_mu * (source - target)
            )
            source_dot, _ = source_driver_rhs(
                source, memory, target, variation.driver_mu,
                variation.driver_eta, advection,
            )
            np.testing.assert_allclose(source_dot, source_time, rtol=0, atol=2e-15)


if __name__ == "__main__":
    unittest.main()
