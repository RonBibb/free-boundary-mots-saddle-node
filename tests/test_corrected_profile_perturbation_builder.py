import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.corrected_profile_perturbation_builder import profile_perturbation_norms


def test_width_perturbations_are_small_nonzero_and_nearly_symmetric():
    z = np.exp(np.linspace(0., 1., 33))
    r = np.linspace(0., 8., 65)
    narrow = profile_perturbation_norms(z, r, .99)
    broad = profile_perturbation_norms(z, r, 1.01)
    for name in ("pulse_relative_L2_difference", "gradient_relative_L2_difference"):
        assert 0 < narrow[name] < .03
        assert 0 < broad[name] < .03
        assert abs(narrow[name] - broad[name]) / max(narrow[name], broad[name]) < .10


def test_width_must_be_positive():
    z = np.exp(np.linspace(0., 1., 9))
    r = np.linspace(0., 2., 13)
    with pytest.raises(ValueError):
        profile_perturbation_norms(z, r, 0.)
