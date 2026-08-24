from __future__ import annotations

import numpy as np
import pytest

from bhps.joint_parent_selective_algebra import (
    SELECTIVE_MAXIMUM_CONDITION,
    SELECTIVE_MAXIMUM_NORMALIZED_RESIDUAL,
    SELECTIVE_MINIMUM_PIVOT,
    SelectiveWallAlgebraicGateError,
    solve_selective_wall_endpoint_block,
    summarize_selective_field_evidence,
)


def _well_conditioned_block():
    matrix = np.asarray(((1.7, 0.2), (0.1, 1.4)))
    full = np.asarray((
        (1.7, -0.3, 0.2, 0.1, 0.2),
        (0.1, 0.2, -0.4, 0.3, 1.4),
    ))
    right = np.asarray(((0.8, -0.4), (0.3, 0.9)))
    return matrix, full, right


def test_selective_helper_uses_full_rows_and_retains_each_field_evidence():
    matrix, full, right = _well_conditioned_block()
    solved, evidence = solve_selective_wall_endpoint_block(
        matrix, full, right, ("h_00", "chi"), radial_index=0,
    )
    np.testing.assert_allclose(matrix@solved, right, rtol=0.0, atol=2e-16)
    assert set(evidence) == {"h_00", "chi"}
    expected_norms = np.linalg.norm(full, axis=1)
    expected_singular = np.linalg.svd(
        matrix/expected_norms[:, None], compute_uv=False,
    )
    for field, record in evidence.items():
        assert record["field"] == field
        assert record["rank"] == 2
        np.testing.assert_array_equal(record["full_row_norms"], expected_norms)
        assert record["equilibrated_condition"] == pytest.approx(
            expected_singular[0]/expected_singular[-1], rel=0.0, abs=0.0,
        )
        assert record["normalized_pivot"] == pytest.approx(
            expected_singular[-1], rel=0.0, abs=0.0,
        )
        assert record["equilibration"] == (
            "endpoint_matrix/full_compact_stencil_row_L2_norm"
        )
        assert record["normalized_linear_residual"] < (
            SELECTIVE_MAXIMUM_NORMALIZED_RESIDUAL
        )
        assert record["passed"]

    records = {
        "h_00": [evidence["h_00"]],
        "chi": [evidence["chi"]],
    }
    summary = summarize_selective_field_evidence(records, ("h_00", "chi"))
    assert summary["each_field_gated_separately"]
    assert summary["chi_credited_only_with_chi_block"]
    assert summary["fields"]["chi"]["minimum_rank"] == 2
    assert summary["fields"]["chi"]["maximum_equilibrated_condition"] <= (
        SELECTIVE_MAXIMUM_CONDITION
    )
    assert summary["fields"]["chi"]["minimum_normalized_pivot"] >= (
        SELECTIVE_MINIMUM_PIVOT
    )


@pytest.mark.parametrize(
    ("matrix", "gate"),
    (
        (np.asarray(((1.0, 1.0), (2.0, 2.0))), "rank_condition_pivot"),
        (np.asarray(((1.0, 0.0), (0.0, 1e-12))), "rank_condition_pivot"),
    ),
)
def test_selective_helper_fails_closed_on_rank_or_normalized_pivot(matrix, gate):
    full = np.zeros((2, 5))
    full[:, [0, -1]] = matrix
    if matrix[1, 1] == pytest.approx(1e-12):
        # Keep the full compact row at physical scale so equilibration exposes
        # the endpoint block's genuinely weak second pivot.
        full[1, 2] = 1.0
    right = np.ones((2, 1))
    with pytest.raises(SelectiveWallAlgebraicGateError) as caught:
        solve_selective_wall_endpoint_block(
            matrix, full, right, ("chi",), radial_index=4,
        )
    error = caught.value
    assert error.radial_index == 4
    assert error.field == "chi"
    assert error.gate == gate
    assert error.diagnostics["rank"] <= 2
    assert (
        error.diagnostics["rank"] != 2
        or error.diagnostics["normalized_pivot"] < SELECTIVE_MINIMUM_PIVOT
        or error.diagnostics["equilibrated_condition"] > SELECTIVE_MAXIMUM_CONDITION
    )


def test_selective_summary_rejects_missing_chi_radius_evidence():
    matrix, full, right = _well_conditioned_block()
    _, evidence = solve_selective_wall_endpoint_block(
        matrix, full, right[:, :1], ("chi",), radial_index=1,
    )
    with pytest.raises(ValueError, match="radial evidence is incomplete"):
        summarize_selective_field_evidence({"chi": [evidence["chi"]]}, ("chi",))
