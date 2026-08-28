import pytest

from transfer_core import adjacent_leaf_transfer, classify, symmetric_relative


def leaf(area=40.0, radius=1.6, length=2.8, axis=1.2, brane=1.5, eigenvalue=0.25, classification="outward-stable"):
    return {
        "geometry": {
            "one_sided_cap_area": area,
            "equivalent_area_radius": radius,
            "proper_meridional_length": length,
            "rho_axis": axis,
            "rho_brane": brane,
        },
        "stability": {"fine_principal_eigenvalue": eigenvalue, "classification": classification},
    }


def test_symmetric_relative():
    assert symmetric_relative(10.0, 9.0) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        symmetric_relative(float("nan"), 1.0)


def test_adjacent_transfer_passes_inherited_limits():
    result = adjacent_leaf_transfer(leaf(), leaf(area=40.1, eigenvalue=0.26))
    assert result["passed"]
    assert result["geometry_pass_below_1_percent"]
    assert result["stability_pass_below_10_percent_or_0p02_absolute"]


def test_adjacent_transfer_fails_geometry_or_stability():
    assert not adjacent_leaf_transfer(leaf(), leaf(area=41.0))["passed"]
    assert not adjacent_leaf_transfer(leaf(), leaf(eigenvalue=0.40))["passed"]
    assert not adjacent_leaf_transfer(leaf(), leaf(classification="outward-unstable"))["passed"]


def test_ordered_classification():
    cases = (
        ((False, True, True, True), "BOUNDED-SPATIAL-TRANSFER-LOCAL-GRID-FAIL"),
        ((True, True, False, True), "BOUNDED-SPATIAL-TRANSFER-CONTROL-FAIL"),
        ((True, False, True, True), "BOUNDED-SPATIAL-TRANSFER-GEOMETRY-OR-STABILITY-FAIL"),
        ((True, True, True, False), "BOUNDED-SPATIAL-TRANSFER-CAUSAL-SIGNATURE-FAIL"),
        ((True, True, True, True), "G9-G10-G11-BOUNDED-OUTER-TUBE-SPATIAL-TRANSFER-PASS"),
    )
    for values, expected in cases:
        assert classify(*values)[0] == expected
