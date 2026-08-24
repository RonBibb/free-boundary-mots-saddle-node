from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from bhps.joint_parent_bulk_audit import EQUATION_ORDER
from bhps.joint_parent_bulk_validation import PROTOCOL_IDENTIFIER as BULK_PROTOCOL
from bhps.joint_parent_construction import (
    build_protocol125_successful_parent_provenance_record,
)
from bhps.joint_parent_gate_ledger import Protocol125GateLedger
from bhps.joint_parent_native_evidence import _digest_tree as _native_evidence_digest_tree
from bhps.joint_parent_preacceleration import (
    NATIVE_EVIDENCE_PROTOCOL_IDENTIFIER,
    NATIVE_POSITION_TANGENT_LANES,
)
from bhps.joint_parent_refinement_diagnostics import (
    axis_acceleration_derivative_image_profile,
    correction_profile,
    frozen_validation_meshes,
)
from bhps.joint_parent_two_parent import (
    CORRECTION_REFINEMENT_LANE_ORDER,
    INPUT_HASH_KEYS,
    NUMERICAL_FLOOR,
    REPRESENTATION_LANE_ORDER,
    TWO_PARENT_RECORD_ORDER,
    Protocol125TwoParentInputs,
    compose_protocol125_two_parent_records,
    nonworsen_f,
    protocol125_two_parent_input_hashes,
    strict_decrease_f,
)


PARENT_IDENTITIES = {"N0": "0"*64, "N1": "1"*64}


def _construction_provenance(label):
    return build_protocol125_successful_parent_provenance_record(
        label,
        PARENT_IDENTITIES[label],
        ("2" if label == "N0" else "3")*64,
        finite_wall_maximum_residual=1e-12,
        joint_hybrid_maximum_residual=2e-12,
    )


def _native_completion_evidence(
    label,
    *,
    anisotropy,
    chi,
    q4_axis_image,
    lapse,
):
    values = {
        "lapse_h00_normalized_Linf": float(lapse),
        "anisotropy_hrr_normalized_Linf": float(anisotropy),
        "chi_normalized_Linf": float(chi),
        "axis_q4_second_derivative_image_normalized_Linf": float(q4_axis_image),
    }
    gates = {
        "lapse_owned_correction": bool(lapse <= 0.05),
        "anisotropy_physical_correction": bool(anisotropy <= 1e-10),
        "chi_physical_correction": bool(chi <= 1e-10),
        "q4_axis_image_correction": bool(q4_axis_image <= 1e-10),
    }
    details = {
        "gates": gates,
        "recorded_corrections": dict(values),
        "recomputed_corrections": dict(values),
        "recorded_correction_reproduction": {
            name: True for name in values
        },
        "passed": bool(all(gates.values())),
    }
    lane = {
        "complete": True,
        "provenance_valid": True,
        "passed": details["passed"],
        "fingerprint": "",
        "details": details,
    }
    payload = {
        "lane": "native_position_completion",
        "complete": lane["complete"],
        "provenance_valid": lane["provenance_valid"],
        "passed": lane["passed"],
        "details": lane["details"],
    }
    lane["fingerprint"] = _native_evidence_digest_tree(
        payload, root="native/native_position_completion",
    )
    lanes = {"native_position_completion": lane}
    for lane_name in NATIVE_POSITION_TANGENT_LANES[1:]:
        other_details = {"passed": True, "manufactured": True}
        other = {
            "complete": True,
            "provenance_valid": True,
            "passed": True,
            "fingerprint": "",
            "details": other_details,
        }
        other_payload = {
            "lane": lane_name,
            "complete": True,
            "provenance_valid": True,
            "passed": True,
            "details": other_details,
        }
        other["fingerprint"] = _native_evidence_digest_tree(
            other_payload, root=f"native/{lane_name}",
        )
        lanes[lane_name] = other
    return {
        "protocol_identifier": NATIVE_EVIDENCE_PROTOCOL_IDENTIFIER,
        "parent_label": label,
        "parent_identity": PARENT_IDENTITIES[label],
        "source_coordinate_sha256": "4"*64,
        "position_sha256": "5"*64,
        "input_fingerprint_before": "6"*64,
        "input_fingerprint_after": "6"*64,
        "lanes": lanes,
        "complete": True,
        "provenance_valid": True,
        "passed": lane["passed"],
    }


class _ContinuousState:
    def __init__(
        self,
        state_name,
        *,
        q4,
        q5,
        axis_q4_offset=0.0,
        mutate_on_evaluate=False,
    ):
        self.state_name = str(state_name)
        self.q4 = float(q4)
        self.q5 = float(q5)
        self.axis_q4_offset = float(axis_q4_offset)
        self.mutate_on_evaluate = bool(mutate_on_evaluate)
        self.epoch = 0

    def fingerprint(self):
        payload = (
            f"{self.state_name}|{self.q4:.17g}|{self.q5:.17g}|"
            f"{self.axis_q4_offset:.17g}|{self.epoch}"
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    def _touch(self):
        if self.mutate_on_evaluate and self.epoch == 0:
            self.epoch = 1

    def evaluate_coordinate_components(self, z, r, *, z_order=0, r_order=0):
        self._touch()
        z = np.asarray(z, dtype=float)
        r = np.asarray(r, dtype=float)
        result = np.zeros((len(z), len(r), 9))
        if (z_order, r_order) == (0, 0):
            if self.state_name == "position":
                result[:, :, 2] = -1.0
                result[:, :, 3] = 1.0
                result[:, :, 6] = 1.0
                result[:, :, 4] = 1.0+r[None, :]**2*self.q4
            else:
                result[:, :, 4] = r[None, :]**2*self.q4
            result[:, :, 5] = r[None, :]*self.q5
        elif (z_order, r_order) == (0, 1):
            result[:, :, 4] = 2.0*r[None, :]*self.q4
            result[:, :, 5] = self.q5
        elif (z_order, r_order) == (0, 2):
            result[:, :, 4] = 2.0*self.q4
        elif (z_order, r_order) not in ((1, 0), (2, 0), (1, 1)):
            raise ValueError("unsupported manufactured derivative")
        return result

    def evaluate_reduced(self, z, r, *, z_order=0, r_order=0):
        z = np.asarray(z, dtype=float)
        r = np.asarray(r, dtype=float)
        result = np.zeros((len(z), len(r), 9))
        if (z_order, r_order) == (0, 0):
            if self.state_name == "position":
                result[:, :, 2] = -1.0
                result[:, :, 3] = 1.0
                result[:, :, 6] = 1.0
            result[:, :, 4] = self.q4
            result[:, :, 5] = self.q5
            if self.axis_q4_offset:
                result[:, 0, 4] += self.axis_q4_offset
        return result


def _flat_position(nz, r):
    result = np.zeros((nz, len(r), 9))
    result[:, :, 2] = -1.0
    result[:, :, 3] = 1.0
    result[:, :, 6] = 1.0
    return result


def _axis_profile(q4, q5):
    z = frozen_validation_meshes()["V2"]["z"]
    bulk = np.zeros((len(z), 9))
    compatible = np.zeros_like(bulk)
    compatible[:, 4] = q4
    compatible[:, 5] = q5
    return axis_acceleration_derivative_image_profile(bulk, compatible, z)


def _common_v2_audit(label, metrics):
    v2_hash = frozen_validation_meshes()["V2"]["sha256"]
    return {
        "protocol": BULK_PROTOCOL,
        "parent_label": label,
        "identity": {
            "parent_label": label,
            "binding_sha256": ("a" if label == "N0" else "b")*64,
        },
        "adjudication": {"parent_bulk_pass": True},
        "lanes": {
            "V2": {
                "authoritative": {
                    "coordinates": {"sha256": v2_hash},
                    "masks": {"provenance": {"mask_sha256": "c"*64}},
                    "scores": {
                        "provenance": {"protocol": BULK_PROTOCOL},
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


def _inputs():
    frozen = frozen_validation_meshes()
    v_meshes = {name: frozen[name] for name in ("V0", "V1", "V2")}
    dense_r = frozen["dense_wall"]["r"]
    dense_position = _flat_position(3, dense_r)
    bulk = np.zeros_like(dense_position)
    compatible_n0 = bulk.copy()
    compatible_n1 = bulk.copy()
    compatible_n0[[0, -1], :, 6] = 0.004
    compatible_n1[[0, -1], :, 6] = 0.003
    correction_n0 = correction_profile(
        dense_position, bulk, compatible_n0, dense_r,
    )
    correction_n1 = correction_profile(
        dense_position, bulk, compatible_n1, dense_r,
    )
    v2_position = _flat_position(len(frozen["V2"]["z"]), frozen["V2"]["r"])
    v2_scalar = np.zeros(v2_position.shape[:2])
    return Protocol125TwoParentInputs(
        n0_position_state=_ContinuousState("position", q4=0.01, q5=-0.02),
        n1_position_state=_ContinuousState(
            "position", q4=0.0100001, q5=-0.0199999,
        ),
        n0_acceleration_state=_ContinuousState(
            "acceleration", q4=0.001, q5=0.002,
        ),
        n1_acceleration_state=_ContinuousState(
            "acceleration", q4=0.001001, q5=0.002001,
        ),
        n0_native_completion_evidence=_native_completion_evidence(
            "N0",
            anisotropy=8e-11,
            chi=7e-11,
            q4_axis_image=6e-11,
            lapse=0.02,
        ),
        n1_native_completion_evidence=_native_completion_evidence(
            "N1",
            anisotropy=4e-11,
            chi=3e-11,
            q4_axis_image=2e-11,
            lapse=0.015,
        ),
        n0_construction_provenance=_construction_provenance("N0"),
        n1_construction_provenance=_construction_provenance("N1"),
        v_meshes=v_meshes,
        dense_wall_r=dense_r,
        n0_bulk_audit=_common_v2_audit("N0", (2e-7, 3e-6, 4e-7, 5e-6)),
        n1_bulk_audit=_common_v2_audit("N1", (1e-7, 3e-6, 2e-7, 4e-6)),
        n0_correction_profile=correction_n0,
        n1_correction_profile=correction_n1,
        n0_position_v2=v2_position,
        n1_position_v2=v2_position.copy(),
        n0_hzz_zz_v2=v2_scalar,
        n1_hzz_zz_v2=v2_scalar.copy(),
        n0_a_hzz_v2=v2_scalar.copy(),
        n1_a_hzz_v2=v2_scalar.copy(),
        n0_axis_image_profile=_axis_profile(0.002, 0.004),
        n1_axis_image_profile=_axis_profile(0.0015, 0.003),
    )


@pytest.fixture(scope="module")
def valid_inputs():
    return _inputs()


def test_passing_composition_has_exact_lanes_stable_hashes_and_ledger_schema(
    valid_inputs,
):
    hashes = protocol125_two_parent_input_hashes(valid_inputs)
    assert tuple(hashes) == INPUT_HASH_KEYS
    records = compose_protocol125_two_parent_records(
        valid_inputs, parent_identities=PARENT_IDENTITIES,
    )
    assert tuple(records) == TWO_PARENT_RECORD_ORDER
    representation = records["N0_N1_representation"]
    correction = records["correction_refinement"]
    assert representation["required_lane_order"] == REPRESENTATION_LANE_ORDER
    assert correction["required_lane_order"] == CORRECTION_REFINEMENT_LANE_ORDER
    for record in records.values():
        assert record["complete"] is True
        assert record["provenance_valid"] is True
        assert record["passed"] is True
        assert record["parent_identities"] == PARENT_IDENTITIES
        assert len(record["fingerprint"]) == 64
        assert record["input_hashes_before"] == record["input_hashes_after"]
        assert not record["scientific_execution_authorized"]
        assert not record["artifact_written"]

    ledger = Protocol125GateLedger(
        {
            "status": "DRAFT — INVALID-specification",
            "protocol_sha256": "a"*64,
            "adjudicator_sha256": "b"*64,
            "frozen_before_parent_data": True,
            "independent_review_passed": True,
            "scientific_candidates_absent_at_freeze": True,
        },
        PARENT_IDENTITIES,
    )
    for name in TWO_PARENT_RECORD_ORDER:
        ledger = ledger.append_two_parent_gate(name, records[name])
    assert tuple(ledger.two_parent_records) == TWO_PARENT_RECORD_ORDER


def test_q4_q5_axis_image_failure_cannot_hide_in_acceleration_coordinate_lane(
    valid_inputs,
):
    adverse = replace(
        valid_inputs,
        n1_acceleration_state=_ContinuousState(
            "acceleration",
            q4=0.001001,
            q5=0.002001,
            axis_q4_offset=0.002,
        ),
    )
    record = compose_protocol125_two_parent_records(
        adverse, parent_identities=PARENT_IDENTITIES,
    )["N0_N1_representation"]
    assert record["lanes"]["acceleration"]["passed"]
    assert not record["lanes"]["acceleration_q4_q5_derivative_images"]["passed"]
    assert not record["passed"]


def test_common_v2_bulk_nonworsening_is_a_mandatory_representation_lane(
    valid_inputs,
):
    worsened = replace(
        valid_inputs,
        n1_bulk_audit=_common_v2_audit("N1", (3e-7, 3e-6, 2e-7, 4e-6)),
    )
    record = compose_protocol125_two_parent_records(
        worsened, parent_identities=PARENT_IDENTITIES,
    )["N0_N1_representation"]
    assert record["lanes"]["state_spatial"]["passed"]
    assert not record["lanes"]["bulk_common_V2_nonworsening"]["passed"]
    assert not record["passed"]


def test_correction_refinement_record_preserves_dense_and_axis_constituents(
    valid_inputs,
):
    no_decrease = replace(
        valid_inputs,
        n1_correction_profile=valid_inputs.n0_correction_profile,
    )
    record = compose_protocol125_two_parent_records(
        no_decrease, parent_identities=PARENT_IDENTITIES,
    )["correction_refinement"]
    assert not record["lanes"]["dense_wall_correction_refinement"]["passed"]
    assert record["lanes"]["V2_hzz_zz_difference"]["passed"]
    assert record["lanes"]["V2_a_hzz_difference"]["passed"]
    assert record["lanes"]["V2_axis_acceleration_derivative_images"]["passed"]
    assert not record["passed"]


def test_identities_meshes_and_in_score_input_stability_fail_closed(valid_inputs):
    with pytest.raises(ValueError, match="must be distinct"):
        compose_protocol125_two_parent_records(
            valid_inputs,
            parent_identities={"N0": "0"*64, "N1": "0"*64},
        )

    bad_meshes = {
        name: dict(record) for name, record in valid_inputs.v_meshes.items()
    }
    bad_meshes["V1"]["sha256"] = "f"*64
    with pytest.raises(ValueError, match="V1 mesh differs"):
        protocol125_two_parent_input_hashes(
            replace(valid_inputs, v_meshes=bad_meshes),
        )

    mutating = replace(
        valid_inputs,
        n0_position_state=_ContinuousState(
            "position", q4=0.01, q5=-0.02, mutate_on_evaluate=True,
        ),
    )
    with pytest.raises(ValueError, match="changed while scoring"):
        compose_protocol125_two_parent_records(
            mutating, parent_identities=PARENT_IDENTITIES,
        )


@pytest.mark.parametrize(
    ("field", "gate_name", "refined_value"),
    (
        ("anisotropy", "anisotropy_physical_correction", 8.5e-11),
        ("chi", "chi_physical_correction", 7.5e-11),
        ("q4_axis_image", "q4_axis_image_correction", 6.5e-11),
    ),
)
def test_each_physical_completion_refinement_is_a_mandatory_representation_lane(
    valid_inputs, field, gate_name, refined_value,
):
    values = {
        "anisotropy": 4e-11,
        "chi": 3e-11,
        "q4_axis_image": 2e-11,
        "lapse": 0.015,
    }
    values[field] = refined_value
    adverse = replace(
        valid_inputs,
        n1_native_completion_evidence=_native_completion_evidence(
            "N1", **values,
        ),
    )
    records = compose_protocol125_two_parent_records(
        adverse, parent_identities=PARENT_IDENTITIES,
    )
    lane = records["N0_N1_representation"]["lanes"][
        "native_completion_correction_refinement"
    ]
    assert lane["parents"]["N0"]["passed"]
    assert lane["parents"]["N1"]["passed"]
    assert not lane["refinement"][gate_name]["passed"]
    assert not lane["passed"]
    assert not records["N0_N1_representation"]["passed"]
    assert records["correction_refinement"]["passed"]


@pytest.mark.parametrize(
    ("field", "gate_name", "refined_value"),
    (
        ("anisotropy", "anisotropy_physical_correction", 1.1e-10),
        ("chi", "chi_physical_correction", 1.1e-10),
        ("q4_axis_image", "q4_axis_image_correction", 1.1e-10),
        ("lapse", "lapse_owned_correction", 0.051),
    ),
)
def test_each_native_completion_per_parent_ceiling_cannot_be_hidden(
    valid_inputs, field, gate_name, refined_value,
):
    values = {
        "anisotropy": 4e-11,
        "chi": 3e-11,
        "q4_axis_image": 2e-11,
        "lapse": 0.015,
    }
    values[field] = refined_value
    adverse = replace(
        valid_inputs,
        n1_native_completion_evidence=_native_completion_evidence(
            "N1", **values,
        ),
    )
    lane = compose_protocol125_two_parent_records(
        adverse, parent_identities=PARENT_IDENTITIES,
    )["N0_N1_representation"]["lanes"][
        "native_completion_correction_refinement"
    ]
    assert not lane["parents"]["N1"]["ceiling_gates"][gate_name]
    assert not lane["parents"]["N1"]["native_completion_lane_passed"]
    assert not lane["passed"]


def test_native_completion_refinement_uses_the_exact_universal_floor(valid_inputs):
    assert NUMERICAL_FLOOR == 1e-12
    assert strict_decrease_f(5e-13, 5e-13)
    assert not strict_decrease_f(5e-13, 1.1e-12)
    assert nonworsen_f(5e-13, 1e-12)
    assert not nonworsen_f(5e-13, 1.1e-12)

    floor_resolved = replace(
        valid_inputs,
        n0_native_completion_evidence=_native_completion_evidence(
            "N0",
            anisotropy=5e-13,
            chi=5e-13,
            q4_axis_image=5e-13,
            lapse=5e-13,
        ),
        n1_native_completion_evidence=_native_completion_evidence(
            "N1",
            anisotropy=5e-13,
            chi=5e-13,
            q4_axis_image=5e-13,
            lapse=1e-12,
        ),
    )
    floor_lane = compose_protocol125_two_parent_records(
        floor_resolved, parent_identities=PARENT_IDENTITIES,
    )["N0_N1_representation"]["lanes"][
        "native_completion_correction_refinement"
    ]
    assert floor_lane["passed"]

    above_floor = replace(
        floor_resolved,
        n1_native_completion_evidence=_native_completion_evidence(
            "N1",
            anisotropy=1.1e-12,
            chi=5e-13,
            q4_axis_image=5e-13,
            lapse=1.1e-12,
        ),
    )
    adverse_lane = compose_protocol125_two_parent_records(
        above_floor, parent_identities=PARENT_IDENTITIES,
    )["N0_N1_representation"]["lanes"][
        "native_completion_correction_refinement"
    ]
    assert not adverse_lane["refinement"][
        "anisotropy_physical_correction"
    ]["passed"]
    assert not adverse_lane["refinement"]["lapse_owned_correction"]["passed"]
    assert not adverse_lane["passed"]


def test_native_evidence_and_successful_provenance_are_cryptographically_bound(
    valid_inputs,
):
    wrong_branch = dict(valid_inputs.n1_construction_provenance)
    wrong_branch["branch_identifier"] = "f"*64
    with pytest.raises(ValueError, match="branch identifier"):
        protocol125_two_parent_input_hashes(
            replace(valid_inputs, n1_construction_provenance=wrong_branch),
        )

    tampered_evidence = dict(valid_inputs.n1_native_completion_evidence)
    lanes = dict(tampered_evidence["lanes"])
    completion = dict(lanes["native_position_completion"])
    details = dict(completion["details"])
    recomputed = dict(details["recomputed_corrections"])
    recomputed["chi_normalized_Linf"] = 2.5e-11
    details["recomputed_corrections"] = recomputed
    completion["details"] = details
    lanes["native_position_completion"] = completion
    tampered_evidence["lanes"] = lanes
    with pytest.raises(ValueError, match="fingerprint differs"):
        protocol125_two_parent_input_hashes(
            replace(
                valid_inputs,
                n1_native_completion_evidence=tampered_evidence,
            ),
        )

    with pytest.raises(ValueError, match="identity or input binding"):
        compose_protocol125_two_parent_records(
            valid_inputs,
            parent_identities={"N0": "a"*64, "N1": "1"*64},
        )
