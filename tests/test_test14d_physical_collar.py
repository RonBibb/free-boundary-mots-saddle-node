import json
from pathlib import Path

from bhps.test14d_physical_collar import (
    physical_collar_record,
)


def test_archived_pilot_record_preserves_thin_limit_and_wall_magnitude():
    test14b_path = Path("results/corrected_A790_test14b_balance_closure.json")
    test14c_path = Path("results/corrected_A790_test14c_coupled_seam.json")
    if not test14b_path.is_file() or not test14c_path.is_file():
        return
    test14b = json.loads(test14b_path.read_text())
    test14c = json.loads(test14c_path.read_text())
    index = test14c["times"].index(0.001)
    thin = test14c["physical_records"]["G7"]["inner"]["1"][index]
    base = test14b["balance_records"]["G7"]["inner"]["1"][index]
    record = physical_collar_record(
        thin, base, "compact_c2", 1.0 / 256.0, 128, "geometry",
    )
    assert record["finite"]
    assert record["stored_thin_formula_error"] < 2e-10
    assert record["integrated_israel_magnitude_error"] < 0.01
    assert (
        record["finite_seam_relative_scale_error"] < 0.02
        or record["finite_seam_balance_normalized_difference"] < 0.01
    )
