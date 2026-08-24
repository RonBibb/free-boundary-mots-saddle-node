import numpy as np

import run_corrected_A790_test2c_proper_arclength_convergence as run


def synthetic_spatial_sequence():
    D = np.linspace(0.0, 1.0, 25)
    S = np.linspace(0.1, 2.0, 33)
    DD, SS = np.meshgrid(D, S, indexing="ij")
    pattern = 1.0 + 0.2 * DD + 0.1 * SS
    sequence = {"distance": D, "arclength": S}
    for label, intervals in zip(run.PRIMARY_GRIDS, (112.0, 128.0, 144.0)):
        sequence[label] = {
            "metric_increment": 2.0 + intervals**-2 * pattern,
            "weight": np.ones_like(pattern),
        }
    return sequence


def test_second_order_spatial_sequence_passes_both_norms():
    sequence = synthetic_spatial_sequence()
    result = run.sequence_score(
        sequence, (sequence, sequence, sequence), "metric_increment",
    )
    assert result["passed"]
    assert result["L2_passed"]
    assert result["weighted_q95_passed"]
    assert result["order_interval"][0] > 1.98
    assert result["weighted_q95_order_interval"][0] > 1.98
    assert result["sign_coherence"] == 1.0


def test_reused_test2b_charts_have_valid_hashes():
    result = run.validate_old_recovery()
    assert result["validated_chart_stages"] == 54


def test_test2c_manufactured_controls_pass():
    result = run.manufactured_controls()
    assert result["passed"]
    assert result["nonmonotone_areal_radius_reproduced"]
    assert result["proper_arclength_error"] < 1e-10
