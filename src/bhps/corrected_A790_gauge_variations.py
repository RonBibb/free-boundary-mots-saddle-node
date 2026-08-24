"""Prospectively sealed live-gauge variations for the A=7.90 audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GaugeVariation:
    """Numerical generalized-harmonic source-driver parameters."""

    name: str
    driver_mu: float
    driver_eta: float
    target_mu_lapse: float
    target_mu_shift: float
    target_power: float = 0.5

    def __post_init__(self):
        if self.driver_mu <= 0 or self.driver_eta <= 0:
            raise ValueError("source-driver rates must be positive")
        if (
            self.target_mu_lapse < 0 or self.target_mu_shift < 0
            or self.target_power < 0
        ):
            raise ValueError("target parameters must be nonnegative")

    def public(self):
        return asdict(self)


BASELINE = GaugeVariation("baseline", 2.0, 1.25, 0.40, 0.60)
VARIATIONS = (
    GaugeVariation("slow_soft", 1.25, 0.75, 0.25, 0.35),
    GaugeVariation("fast_strong", 3.0, 2.00, 0.65, 0.90),
)


def configure_live_module(module, variation):
    """Apply one sealed parameter tuple to the existing live evolution."""
    if not isinstance(variation, GaugeVariation):
        raise TypeError("variation must be a GaugeVariation")
    module.DRIVER_MU = variation.driver_mu
    module.DRIVER_ETA = variation.driver_eta
    module.TARGET_MU_LAPSE = variation.target_mu_lapse
    module.TARGET_MU_SHIFT = variation.target_mu_shift
    module.TARGET_POWER = variation.target_power


def brackets_baseline():
    """Return whether the two variations bracket all four baseline rates."""
    slow, fast = VARIATIONS
    names = (
        "driver_mu", "driver_eta", "target_mu_lapse", "target_mu_shift",
    )
    return all(
        getattr(slow, name) < getattr(BASELINE, name) < getattr(fast, name)
        for name in names
    )
