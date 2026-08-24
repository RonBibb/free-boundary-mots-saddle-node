from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bhps.joint_parent_bulk_audit import EQUATION_ORDER
from bhps.joint_parent_bulk_reference import (
    SOURCE_CELL_MIDPOINT_SPECS,
    FiniteWallReferenceHermitePair,
)
from bhps.joint_parent_bulk_validation import (
    BULK_THRESHOLDS,
    CANDIDATE_JET_FIELDS,
    COMMON_V2_FLOOR,
    MASK_WIDTH,
    PROTOCOL_IDENTIFIER,
    bind_protocol125_bulk_identity,
    compare_protocol125_common_v2,
    constrained_position_candidate_jets,
    constrained_position_coordinate_jets,
    evaluate_protocol125_bulk_lane,
    frozen_bulk_region_masks,
    score_jet_representation_sensitivity,
    score_open_bulk_record,
    score_parent_strip_layer_growth,
)
from bhps.joint_parent_representation import NATIVE_CHANNEL_ORDER
from bhps.joint_parent_refinement_diagnostics import frozen_validation_meshes
from bhps.matched_staged_continuum import hash_arrays


def _immutable(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class _PolynomialPositionState:
    state_name = "position"
    z_degree = 5

    def evaluate_physical_channels(self, z, r, *, z_order=0, r_order=0):
        z = np.asarray(z, dtype=float)
        r = np.asarray(r, dtype=float)
        zz, rr = np.meshgrid(z, r, indexing="ij")
        base = {
            "h_zz": 2.0+zz**2+3.0*rr**2,
            "h_rr": 3.0+2.0*zz**2+4.0*rr**2,
            "h_perp": 3.0+2.0*zz**2+4.0*rr**2,
            "Phi": 0.2+3.0*zz**2+5.0*rr**2,
            "chi": -0.1+4.0*zz**2+6.0*rr**2,
        }
        z_coefficients = {
            "h_zz": 1.0,
            "h_rr": 2.0,
            "h_perp": 2.0,
            "Phi": 3.0,
            "chi": 4.0,
        }
        r_coefficients = {
            "h_zz": 3.0,
            "h_rr": 4.0,
            "h_perp": 4.0,
            "Phi": 5.0,
            "chi": 6.0,
        }
        output = np.zeros((len(z), len(r), len(NATIVE_CHANNEL_ORDER)))
        for name in CANDIDATE_JET_FIELDS:
            if (z_order, r_order) == (0, 0):
                value = base[name]
            elif (z_order, r_order) == (1, 0):
                value = 2.0*z_coefficients[name]*zz
            elif (z_order, r_order) == (0, 1):
                value = 2.0*r_coefficients[name]*rr
            elif (z_order, r_order) == (2, 0):
                value = np.full_like(zz, 2.0*z_coefficients[name])
            elif (z_order, r_order) == (0, 2):
                value = np.full_like(zz, 2.0*r_coefficients[name])
            elif (z_order, r_order) == (1, 1):
                value = np.zeros_like(zz)
            else:
                raise AssertionError("unexpected derivative order")
            output[:, :, NATIVE_CHANNEL_ORDER.index(name)] = value
        return output

    def evaluate_reduced(self, z, r):
        channels = self.evaluate_physical_channels(z, r)
        output = np.zeros((*channels.shape[:2], 9))
        output[:, :, 2] = channels[:, :, NATIVE_CHANNEL_ORDER.index("h00")]
        output[:, :, 3] = channels[:, :, NATIVE_CHANNEL_ORDER.index("h_perp")]
        output[:, :, 4] = 0.0
        output[:, :, 6] = channels[:, :, NATIVE_CHANNEL_ORDER.index("h_zz")]
        output[:, :, 7] = channels[:, :, NATIVE_CHANNEL_ORDER.index("Phi")]
        output[:, :, 8] = channels[:, :, NATIVE_CHANNEL_ORDER.index("chi")]
        return output

    def evaluate_coordinate_components(self, z, r, *, z_order=0, r_order=0):
        channels = self.evaluate_physical_channels(
            z, r, z_order=z_order, r_order=r_order,
        )
        output = np.zeros((*channels.shape[:2], 9))
        for output_name, channel_name in (
            ("h00", "h00"),
            ("h_perp", "h_perp"),
            ("h_rr", "h_rr"),
            ("h_zz", "h_zz"),
            ("Phi", "Phi"),
            ("chi", "chi"),
        ):
            output[:, :, (
                "h_z0", "h_zr", "h00", "h_perp", "h_rr", "h_0r",
                "h_zz", "Phi", "chi",
            ).index(output_name)] = channels[
                :, :, NATIVE_CHANNEL_ORDER.index(channel_name)
            ]
        return output


def test_constrained_state_adapter_emits_exact_five_bulk_jet_mappings():
    z = np.linspace(1.0, 2.0, 15)
    r = np.linspace(0.0, 3.0, 17)
    jets = constrained_position_candidate_jets(_PolynomialPositionState(), z, r)
    assert tuple(jets) == CANDIDATE_JET_FIELDS
    assert all(tuple(jets[name]) == ("value", "z", "r", "zz", "rr") for name in jets)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    np.testing.assert_array_equal(jets["h_zz"]["z"], 2.0*zz)
    np.testing.assert_array_equal(jets["h_zz"]["r"], 6.0*rr)
    np.testing.assert_array_equal(jets["Phi"]["zz"], 6.0)
    np.testing.assert_array_equal(jets["chi"]["rr"], 12.0)
    assert all(
        not jets[name][lane].flags.writeable
        for name in jets for lane in jets[name]
    )
    coordinate = constrained_position_coordinate_jets(
        _PolynomialPositionState(), z, r,
    )
    assert len(coordinate) == 9
    assert all(tuple(field) == (
        "value", "z", "r", "zz", "zr", "rr",
    ) for field in coordinate.values())


def test_lane_orchestrator_uses_source_fd_and_analytic_backends_without_rows():
    z = np.linspace(1.0, np.e, 17)
    r = np.linspace(0.0, 12.0, 19)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    reference = FiniteWallReferenceHermitePair.build(
        z,
        r,
        0.2+0.01*zz+0.002*(rr/12.0)**2,
        0.03*np.cos(zz)*(1.0+0.01*(rr/12.0)**2),
    ).primary
    state = _PolynomialPositionState()
    source = evaluate_protocol125_bulk_lane(
        state,
        reference,
        z,
        r,
        {"mass_squared": 0.5},
        backend="source_fd7",
        physical_faces=True,
    )
    assert source["backend"] == "source_fd7"
    assert source["terms"]["method"] == "source_node_polynomial_fd7_open_bulk"
    assert source["scores"]["provenance"]["no_boundary_rows_inserted"]
    assert source["candidate_jets"] is None

    midpoint_z = 0.5*(z[:-1]+z[1:])
    midpoint_r = 0.5*(r[:-1]+r[1:])
    analytic = evaluate_protocol125_bulk_lane(
        state,
        reference,
        midpoint_z,
        midpoint_r,
        {"mass_squared": 0.5},
        backend="analytic",
        physical_faces=False,
    )
    assert analytic["backend"] == "analytic"
    assert analytic["terms"]["method"] == (
        "analytic_candidate_and_reference_jets_open_bulk"
    )
    assert analytic["terms"]["axis_treatment"] == "off_axis_only"
    assert analytic["candidate_jets"] is not None
    assert analytic["reference_jets"] is not None
    assert not analytic["masks"]["faces"]


def test_seven_index_masks_have_explicit_face_overlap_and_midpoint_semantics():
    z = np.linspace(1.0, 2.0, 21)
    r = np.linspace(0.0, 3.0, 23)
    physical = frozen_bulk_region_masks(z, r, physical_faces=True)
    assert MASK_WIDTH == 7
    assert np.count_nonzero(physical["retained"]) == (21-14)*(23-14)
    assert np.all(physical["retained"][7:-7, 7:-7])
    assert not np.any(physical["retained"][:7])
    assert set(physical["faces"]) == {
        "lower_compact", "upper_compact", "axis", "outer",
    }
    assert np.count_nonzero(
        physical["seven_index_strips"]["lower_compact"]
    ) == 7*23
    assert np.all(
        physical["seven_index_strips"]["lower_compact"]
        [physical["faces"]["lower_compact"]]
    )
    assert physical["provenance"]["strip_includes_physical_face"]
    assert not physical["retained"].flags.writeable

    midpoint = frozen_bulk_region_masks(z, r, physical_faces=False)
    assert not midpoint["faces"]
    assert not midpoint["provenance"]["strip_includes_physical_face"]
    assert midpoint["provenance"]["retained_slice"] == "[7:-7,7:-7]"


def _bulk_record(shape, raw_h, defect_h, raw_phi, defect_phi):
    raw = {
        "hamiltonian": np.full(shape, raw_h, dtype=float),
        "Phi": np.full(shape, raw_phi, dtype=float),
    }
    defect = {
        "hamiltonian": np.full(shape, defect_h, dtype=float),
        "Phi": np.full(shape, defect_phi, dtype=float),
    }
    balanced = {name: raw[name]-defect[name] for name in EQUATION_ORDER}
    denominator = {
        name: np.maximum(1.0, np.abs(raw[name])+np.abs(defect[name]))
        for name in EQUATION_ORDER
    }
    return {
        "equation_order": EQUATION_ORDER,
        "raw": raw,
        "defect": defect,
        "balanced": balanced,
        "common_denominator": denominator,
        "raw_normalized": {
            name: np.abs(raw[name])/denominator[name] for name in EQUATION_ORDER
        },
        "balanced_normalized": {
            name: np.abs(balanced[name])/denominator[name]
            for name in EQUATION_ORDER
        },
        "reassembly_defect": {
            name: np.zeros(shape) for name in EQUATION_ORDER
        },
        "reassembly_Linf": 0.0,
        "method": "manufactured",
        "axis_treatment": "test",
        "source_stencil_width": 7,
        "normalization": "max(1,abs(raw)+abs(reference_defect))",
    }


def test_bulk_scoring_concatenates_equations_and_keeps_regions_separate():
    z = np.linspace(1.0, 2.0, 19)
    r = np.linspace(0.0, 3.0, 20)
    masks = frozen_bulk_region_masks(z, r, physical_faces=True)
    record = _bulk_record((len(z), len(r)), 3.0, 1.0, 4.0, 1.0)
    scored = score_open_bulk_record(record, masks)
    balanced = scored["retained"]["balanced_normalized"]
    expected = np.sqrt(0.5*((2.0/4.0)**2+(3.0/5.0)**2))
    assert balanced["combined_RMS"] == pytest.approx(expected)
    assert balanced["combined_Linf"] == pytest.approx(3.0/5.0)
    assert balanced["combined_sample_count"] == 2*(19-14)*(20-14)
    assert set(scored["faces"]) == set(masks["faces"])
    assert set(scored["seven_index_strips"]) == set(
        masks["seven_index_strips"]
    )
    assert not scored["numerical_gate_pass"]
    assert scored["provenance"]["thresholds"] == dict(BULK_THRESHOLDS)


def test_bulk_scoring_rejects_nonfinite_region_before_any_gate():
    z = np.linspace(1.0, 2.0, 17)
    r = np.linspace(0.0, 3.0, 17)
    masks = frozen_bulk_region_masks(z, r, physical_faces=True)
    record = _bulk_record((17, 17), 0.0, 0.0, 0.0, 0.0)
    record["raw"]["hamiltonian"][0, 0] = np.nan
    with pytest.raises(ValueError, match="hamiltonian is invalid"):
        score_open_bulk_record(record, masks)


def test_bulk_scoring_recomputes_reassembly_and_rejects_false_pass_metadata():
    z = np.linspace(1.0, 2.0, 17)
    r = np.linspace(0.0, 3.0, 17)
    masks = frozen_bulk_region_masks(z, r, physical_faces=True)
    record = _bulk_record((17, 17), 0.0, 0.0, 0.0, 0.0)
    record["balanced"]["Phi"][8, 8] = 2e-8
    record["reassembly_Linf"] = 0.0
    scored = score_open_bulk_record(record, masks)
    assert scored["reassembly"]["recomputed_reassembly_Linf"] == 2e-8
    assert not scored["gates"]["reassembly_Linf"]
    assert not scored["numerical_gate_pass"]


def _nested_jets(shape, offset=0.0):
    return {
        name: {
            lane: np.full(shape, index+offset, dtype=float)
            for index, lane in enumerate(
                ("value", "z", "r", "zz", "zr", "rr"), 1,
            )
        }
        for name in ("q", "Phi")
    }


def test_q53_q33_sensitivity_uses_pointwise_scaling_and_group_ceilings():
    primary = _nested_jets((3, 4))
    close = _nested_jets((3, 4), offset=5e-9)
    passed = score_jet_representation_sensitivity(
        primary, close, ("q", "Phi"),
    )
    assert passed["pass"]
    assert passed["groups"]["first"]["combined_sample_count"] == 2*2*3*4
    far = _nested_jets((3, 4), offset=2e-4)
    failed = score_jet_representation_sensitivity(
        primary, far, ("q", "Phi"),
    )
    assert not failed["pass"]
    assert not failed["gates"]["value"]


def _strip_score(value):
    return {
        strip: {
            family: {
                "equations": {
                    equation: {"Linf": value}
                    for equation in EQUATION_ORDER
                },
            }
            for family in (
                "balanced_normalized", "absolute_raw_normalized",
            )
        }
        for strip in ("lower_compact", "upper_compact", "axis", "outer")
    }


def test_parent_strip_layer_gate_applies_both_families_per_equation_without_rms():
    lanes = {
        "V1": {"authoritative": {"scores": {
            "seven_index_strips": _strip_score(2e-7),
        }}},
        "V2": {"authoritative": {"scores": {
            "seven_index_strips": _strip_score(1e-7),
        }}},
    }
    passed = score_parent_strip_layer_growth(lanes)
    assert passed["pass"]
    assert set(passed["comparisons"]) == {
        "balanced_normalized", "absolute_raw_normalized",
    }
    assert not passed["RMS_can_rescue"]
    lanes["V2"]["authoritative"]["scores"]["seven_index_strips"][
        "outer"
    ]["absolute_raw_normalized"]["equations"]["Phi"]["Linf"] = 3e-7
    failed = score_parent_strip_layer_growth(lanes)
    assert not failed["pass"]
    assert not failed["comparisons"]["absolute_raw_normalized"]["Phi"][
        "outer"
    ]["passed"]


class _FakeReferenceRepresentation:
    def __init__(self, z, r, values, endpoints, degree, fingerprint):
        self.source_z = z
        self.source_r = r
        self.source_values = values
        self.endpoint_z_first = endpoints
        self.surface = SimpleNamespace(
            z_degree=degree, z_boundary="clamped_source_width7_z_first",
        )
        self.stencil_width = 7
        self.recipe = "primary-Q53" if degree == 5 else "comparator-Q33"
        self.channel_order = ("q", "Phi")
        self._fingerprint = fingerprint

    def fingerprint(self):
        return self._fingerprint


class _FakeRepresentation:
    def __init__(self, state, source, endpoints, fingerprint):
        self.position = state
        self.source_fingerprint = source
        self.endpoint_fingerprint = endpoints
        self._fingerprint = fingerprint

    def fingerprint(self):
        return self._fingerprint


class _FakeReferencePair:
    def __init__(self, primary, comparator):
        self.primary = primary
        self.comparator = comparator

    def fingerprint(self):
        return "reference-pair"


class _FakeCandidatePair:
    def __init__(self, primary, comparator):
        self.primary = primary
        self.comparator = comparator

    def fingerprint(self):
        return "candidate-pair"


def _fake_binding_inputs(writable=False):
    nz, nr = SOURCE_CELL_MIDPOINT_SPECS["N0"]["source_shape"]
    z = np.linspace(1.0, np.e, nz)
    r = np.linspace(0.0, 12.0, nr)
    values = np.zeros((nz, nr, 2))
    endpoints = np.zeros((2, nr, 2))
    if not writable:
        z, r, values, endpoints = map(
            _immutable, (z, r, values, endpoints),
        )
    state_common = {
        "source_z": z,
        "source_r": r,
        "stored_z_first_endpoints": endpoints,
        "outer_ownership_mask": _immutable(np.ones(8, dtype=bool)),
        "compact_wall_contract_id": "compact",
        "outer_open_face_contract_id": "outer",
        "compact_wall_contract_fingerprint": "compact-fingerprint",
        "outer_open_face_contract_fingerprint": "outer-fingerprint",
    }
    primary_state = SimpleNamespace(
        state_name="position", z_degree=5, **state_common,
    )
    comparator_state = SimpleNamespace(
        state_name="position", z_degree=3, **state_common,
    )
    candidate = _FakeCandidatePair(
        _FakeRepresentation(primary_state, "source", "endpoint", "Q53"),
        _FakeRepresentation(comparator_state, "source", "endpoint", "Q33"),
    )
    reference = _FakeReferencePair(
        _FakeReferenceRepresentation(z, r, values, endpoints, 5, "R53"),
        _FakeReferenceRepresentation(z, r, values, endpoints, 3, "R33"),
    )
    return candidate, reference


def test_binding_enforces_canonical_shared_immutable_source_and_reference():
    candidate, reference = _fake_binding_inputs()
    identity = bind_protocol125_bulk_identity("N0", candidate, reference)
    assert identity["source_shape"] == (145, 325)
    assert identity["source_coordinate_sha256"] == (
        SOURCE_CELL_MIDPOINT_SPECS["N0"]["source_coordinate_sha256"]
    )
    assert identity["shared_source_coordinates_bitwise"]
    assert identity["shared_reference_inputs_bitwise"]
    assert identity["inputs_immutable"]
    assert identity["binding_sha256"]

    candidate, reference = _fake_binding_inputs(writable=True)
    with pytest.raises(ValueError, match="must be immutable"):
        bind_protocol125_bulk_identity("N0", candidate, reference)

    candidate, reference = _fake_binding_inputs()
    changed = np.asarray(reference.comparator.source_values).copy()
    changed[0, 0, 0] = -0.0
    reference.comparator.source_values = _immutable(changed)
    with pytest.raises(ValueError, match="exact source inputs"):
        bind_protocol125_bulk_identity("N0", candidate, reference)


def _common_v2_audit(label, metrics, *, mask_sha="mask"):
    v2_hash = frozen_validation_meshes()["V2"]["sha256"]
    return {
        "protocol": PROTOCOL_IDENTIFIER,
        "parent_label": label,
        "identity": {
            "parent_label": label,
            "binding_sha256": f"binding-{label}",
        },
        "adjudication": {"parent_bulk_pass": True},
        "lanes": {
            "V2": {
                "authoritative": {
                    "coordinates": {"sha256": v2_hash},
                    "masks": {"provenance": {"mask_sha256": mask_sha}},
                    "scores": {
                        "provenance": {"protocol": PROTOCOL_IDENTIFIER},
                        "retained": {
                            "balanced_normalized": {
                                "combined_RMS": metrics[0],
                                "combined_Linf": metrics[1],
                            },
                            "absolute_raw_normalized": {
                                "combined_RMS": metrics[2],
                                "combined_Linf": metrics[3],
                            },
                        },
                        "seven_index_strips": {
                            strip: {
                                family: {
                                    "equations": {
                                        equation: {"Linf": metrics[0]}
                                        for equation in EQUATION_ORDER
                                    },
                                }
                                for family in (
                                    "balanced_normalized",
                                    "absolute_raw_normalized",
                                )
                            }
                            for strip in (
                                "lower_compact", "upper_compact", "axis", "outer",
                            )
                        },
                    },
                },
            },
        },
    }


def test_common_v2_predicate_and_strip_nonworsening_are_explicit():
    n0 = _common_v2_audit("N0", (2e-7, 3e-6, 4e-7, 5e-6))
    n1 = _common_v2_audit("N1", (1e-7, 3e-6, 2e-7, 4e-6))
    comparison = compare_protocol125_common_v2(n0, n1)
    assert comparison["core_nonworsening_pass"]
    assert comparison["predicate"] == "N1<=N0, or both values<=1e-12"
    assert COMMON_V2_FLOOR == 1e-12
    assert comparison["strip_nonworsening_pass"]
    assert comparison["protocol_common_V2_pass"]
    assert not comparison["fail_closed"]

    worsened = _common_v2_audit("N1", (3e-7, 3e-6, 2e-7, 4e-6))
    comparison = compare_protocol125_common_v2(n0, worsened)
    assert not comparison["core_nonworsening_pass"]
    assert not comparison["comparisons"]["combined_balanced_RMS"]["passed"]

    strip_only = _common_v2_audit("N1", (1e-7, 3e-6, 2e-7, 4e-6))
    strip_only["lanes"]["V2"]["authoritative"]["scores"][
        "seven_index_strips"
    ]["outer"]["absolute_raw_normalized"]["equations"]["Phi"]["Linf"] = 4e-7
    comparison = compare_protocol125_common_v2(n0, strip_only)
    assert comparison["core_nonworsening_pass"]
    assert not comparison["strip_nonworsening_pass"]
    assert not comparison["protocol_common_V2_pass"]


def test_common_v2_floor_allows_only_both_below_floor_when_refined_worsens():
    n0 = _common_v2_audit("N0", (0.5e-12, 2e-6, 2e-7, 3e-6))
    n1 = _common_v2_audit("N1", (0.9e-12, 2e-6, 2e-7, 3e-6))
    comparison = compare_protocol125_common_v2(n0, n1)
    assert comparison["comparisons"]["combined_balanced_RMS"]["passed"]
    above = _common_v2_audit("N1", (1.1e-12, 2e-6, 2e-7, 3e-6))
    comparison = compare_protocol125_common_v2(n0, above)
    assert not comparison["comparisons"]["combined_balanced_RMS"]["passed"]
