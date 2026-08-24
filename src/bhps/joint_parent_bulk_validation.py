"""Result-independent Protocol-125 bulk-validation orchestration.

This module joins three already-defined numerical objects without changing any
of them: the constrained native-metric parent representation, the finite-wall
Q53/Q33 reference pair, and the open-bulk equation backends.  It deliberately
does not construct N0/N1, repair a field, write an artifact, or authorize an
acceleration/evolution calculation.

The retained bulk score is fully explicit.  Exactly seven coordinate indices
are removed from every side, Hamiltonian and stationary-Phi samples are
concatenated, and RMS/Linf are computed on that concatenated vector.  Physical
faces and seven-index strips (the latter include the face) are reported
separately.  Midpoint grids have only near-face strips.

The frozen layer-growth rule is applied independently to both normalized
families, both equations, and all four face strips: V2 Linf may not exceed
``max(V1 Linf, 1e-12)``.  The two-parent common-V2 comparison additionally
applies the frozen floor-aware nonworsening predicate to every such strip.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from bhps.joint_parent_bulk_audit import (
    EQUATION_ORDER,
    SOURCE_STENCIL_WIDTH,
    open_anisotropic_bulk_terms_fd,
    open_anisotropic_bulk_terms_from_jets,
)
from bhps.joint_parent_bulk_reference import (
    REFERENCE_CHANNEL_ORDER,
    SOURCE_CELL_MIDPOINT_SPECS,
    source_cell_midpoint_coordinates,
)
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_refinement_diagnostics import frozen_validation_meshes
from bhps.joint_parent_representation import (
    COORDINATE_COMPONENT_ORDER,
    NATIVE_CHANNEL_ORDER,
)
from bhps.matched_staged_continuum import hash_arrays


PROTOCOL_IDENTIFIER = "Protocol-125-draft-result-independent-bulk-audit"
MASK_WIDTH = 7
CANDIDATE_JET_FIELDS = ("h_zz", "h_rr", "h_perp", "Phi", "chi")
JET_LANES = ("value", "z", "r", "zz", "rr")
REPRESENTATION_JET_LANES = ("value", "z", "r", "zz", "zr", "rr")
OFF_SOURCE_LANES = ("midpoint", "V0", "V1", "V2")
ALL_LANES = ("source", *OFF_SOURCE_LANES)
SCORE_FAMILIES = (
    "balanced_normalized",
    "absolute_raw_normalized",
    "raw_unscaled",
    "reference_defect_unscaled",
    "balanced_unscaled",
)
BULK_THRESHOLDS = MappingProxyType({
    "combined_balanced_RMS": 1e-6,
    "combined_balanced_Linf": 1e-5,
    "combined_absolute_raw_RMS": 1e-6,
    "combined_absolute_raw_Linf": 1e-5,
    "reassembly_Linf": 1e-12,
})
REPRESENTATION_THRESHOLDS = MappingProxyType({
    "value": 1e-8,
    "first": 1e-7,
    "second": 1e-5,
})
COMMON_V2_FLOOR = 1e-12


def _immutable_array(value, dtype=float):
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze_mapping(value):
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(name): _freeze_mapping(item) for name, item in value.items()
        })
    if isinstance(value, np.ndarray):
        if value.flags.c_contiguous and not value.flags.writeable:
            return value
        return _immutable_array(value, None)
    if isinstance(value, tuple):
        return tuple(_freeze_mapping(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_mapping(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _update_digest(digest, name, value):
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(name).encode())
    digest.update(b"\0")
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())


def _named_array_fingerprint(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        _update_digest(digest, name, value)
    return digest.hexdigest()


def _object_fingerprint(value, label):
    method = getattr(value, "fingerprint", None)
    if method is not None and callable(method):
        found = str(method())
    else:
        coefficients = getattr(value, "coefficient_arrays", None)
        if coefficients is None or not callable(coefficients):
            raise ValueError(f"{label} must expose fingerprint() or coefficient_arrays()")
        found = _named_array_fingerprint(coefficients())
    if not found:
        raise ValueError(f"{label} fingerprint is empty")
    return found


def _require_immutable(array, label):
    array = np.asarray(array)
    if array.flags.writeable:
        raise ValueError(f"{label} must be immutable before bulk validation")
    return array


def _arrays_bitwise_equal(left, right):
    left = np.ascontiguousarray(np.asarray(left))
    right = np.ascontiguousarray(np.asarray(right))
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def _validate_coordinates(z, r):
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    if z.ndim != 1 or r.ndim != 1:
        raise ValueError("bulk-validation coordinates must be one-dimensional")
    if len(z) <= 2*MASK_WIDTH or len(r) <= 2*MASK_WIDTH:
        raise ValueError("bulk-validation grid has no retained seven-strip interior")
    if (
        np.any(~np.isfinite(z))
        or np.any(~np.isfinite(r))
        or np.any(np.diff(z) <= 0.0)
        or np.any(np.diff(r) <= 0.0)
    ):
        raise ValueError("bulk-validation coordinates must be finite and increasing")
    return z, r


def constrained_position_candidate_jets(state, z, r):
    """Adapt one constrained position state to the five analytic bulk jets."""
    z, r = _validate_coordinates(z, r)
    if str(getattr(state, "state_name", "")) != "position":
        raise ValueError("bulk candidate jets require a constrained position state")
    evaluator = getattr(state, "evaluate_physical_channels", None)
    if evaluator is None or not callable(evaluator):
        raise TypeError("constrained position state lacks physical-channel evaluation")
    derivative_orders = {
        "value": (0, 0),
        "z": (1, 0),
        "r": (0, 1),
        "zz": (2, 0),
        "rr": (0, 2),
    }
    evaluated = {}
    expected_shape = (len(z), len(r), len(NATIVE_CHANNEL_ORDER))
    for lane, (z_order, r_order) in derivative_orders.items():
        values = np.asarray(
            evaluator(z, r, z_order=z_order, r_order=r_order),
            dtype=float,
        )
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise ValueError(f"constrained candidate {lane} channels are invalid")
        evaluated[lane] = values
    channel_indices = {
        name: NATIVE_CHANNEL_ORDER.index(name) for name in CANDIDATE_JET_FIELDS
    }
    return MappingProxyType({
        name: MappingProxyType({
            lane: _immutable_array(values[:, :, channel_indices[name]])
            for lane, values in evaluated.items()
        })
        for name in CANDIDATE_JET_FIELDS
    })


def constrained_position_coordinate_jets(state, z, r):
    """Return all nine physical-coordinate position/spatial jet mappings."""
    z, r = _validate_coordinates(z, r)
    if str(getattr(state, "state_name", "")) != "position":
        raise ValueError("coordinate jets require a constrained position state")
    evaluator = getattr(state, "evaluate_coordinate_components", None)
    if evaluator is None or not callable(evaluator):
        raise TypeError("constrained position state lacks coordinate evaluation")
    derivative_orders = {
        "value": (0, 0),
        "z": (1, 0),
        "r": (0, 1),
        "zz": (2, 0),
        "zr": (1, 1),
        "rr": (0, 2),
    }
    evaluated = {}
    expected_shape = (len(z), len(r), len(COORDINATE_COMPONENT_ORDER))
    for lane, (z_order, r_order) in derivative_orders.items():
        values = np.asarray(
            evaluator(z, r, z_order=z_order, r_order=r_order), dtype=float,
        )
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise ValueError(f"constrained coordinate {lane} channels are invalid")
        evaluated[lane] = values
    return MappingProxyType({
        name: MappingProxyType({
            lane: _immutable_array(values[:, :, index])
            for lane, values in evaluated.items()
        })
        for index, name in enumerate(COORDINATE_COMPONENT_ORDER)
    })


def frozen_bulk_region_masks(z, r, *, physical_faces):
    """Return the exact retained, face, and seven-index strip masks.

    A strip includes its physical face when ``physical_faces=True``.  Thus the
    face and strip reports intentionally overlap, while the retained mask is
    disjoint from every strip.  On direct source-cell midpoints there are no
    physical face samples and the same masks are named near-face strips.
    """
    z, r = _validate_coordinates(z, r)
    physical_faces = bool(physical_faces)
    shape = (len(z), len(r))
    retained = np.zeros(shape, dtype=bool)
    retained[MASK_WIDTH:-MASK_WIDTH, MASK_WIDTH:-MASK_WIDTH] = True

    lower_z = np.zeros(shape, dtype=bool)
    upper_z = np.zeros(shape, dtype=bool)
    lower_r = np.zeros(shape, dtype=bool)
    upper_r = np.zeros(shape, dtype=bool)
    lower_z[:MASK_WIDTH, :] = True
    upper_z[-MASK_WIDTH:, :] = True
    lower_r[:, :MASK_WIDTH] = True
    upper_r[:, -MASK_WIDTH:] = True
    physical_strip_names = {
        "lower_compact": lower_z,
        "upper_compact": upper_z,
        "axis": lower_r,
        "outer": upper_r,
    }
    if physical_faces:
        strips = physical_strip_names
    else:
        strips = {
            f"near_{name}": value for name, value in physical_strip_names.items()
        }
    faces = {}
    if physical_faces:
        for name, index in (
            ("lower_compact", (0, slice(None))),
            ("upper_compact", (-1, slice(None))),
            ("axis", (slice(None), 0)),
            ("outer", (slice(None), -1)),
        ):
            mask = np.zeros(shape, dtype=bool)
            mask[index] = True
            faces[name] = mask
    mask_arrays = {
        "retained": retained,
        **{f"strip_{name}": value for name, value in strips.items()},
        **{f"face_{name}": value for name, value in faces.items()},
    }
    mask_names = tuple(sorted(mask_arrays))
    overlap_counts = {
        f"{left}__AND__{right}": int(np.count_nonzero(
            mask_arrays[left]&mask_arrays[right]
        ))
        for left_index, left in enumerate(mask_names)
        for right in mask_names[left_index+1:]
    }
    region_names = tuple(name for name in mask_names if name != "retained")
    overlap_multiplicity = np.sum(
        np.stack([mask_arrays[name] for name in region_names]), axis=0,
    )
    overlap_multiplicity_counts = {
        str(multiplicity): int(np.count_nonzero(
            overlap_multiplicity == multiplicity
        ))
        for multiplicity in range(len(region_names)+1)
        if np.any(overlap_multiplicity == multiplicity)
    }
    result = {
        "retained": _immutable_array(retained, bool),
        "faces": {
            name: _immutable_array(value, bool) for name, value in faces.items()
        },
        "seven_index_strips": {
            name: _immutable_array(value, bool) for name, value in strips.items()
        },
        "provenance": {
            "mask_width": MASK_WIDTH,
            "retained_slice": "[7:-7,7:-7]",
            "physical_faces": physical_faces,
            "strip_includes_physical_face": physical_faces,
            "face_strip_overlap_is_intentional": physical_faces,
            "coordinate_sha256": hash_arrays(z, r),
            "mask_sha256": _named_array_fingerprint(mask_arrays),
            "retained_point_count": int(np.count_nonzero(retained)),
            "mask_point_counts": {
                name: int(np.count_nonzero(value))
                for name, value in mask_arrays.items()
            },
            "pairwise_overlap_point_counts": overlap_counts,
            "region_overlap_multiplicity_point_counts": (
                overlap_multiplicity_counts
            ),
        },
    }
    return _freeze_mapping(result)


def _field_scores(fields, mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("bulk score mask must select at least one point")
    arrays = {}
    equations = {}
    for equation in EQUATION_ORDER:
        if equation not in fields:
            raise ValueError(f"bulk score is missing equation {equation}")
        values = np.asarray(fields[equation], dtype=float)
        if values.shape != mask.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"bulk score equation {equation} is invalid")
        selected = values[mask]
        arrays[equation] = selected
        equations[equation] = {
            "RMS": float(np.sqrt(np.mean(selected**2))),
            "Linf": float(np.max(np.abs(selected))),
            "sample_count": int(selected.size),
        }
    combined = np.concatenate([arrays[name] for name in EQUATION_ORDER])
    return {
        "equations": equations,
        "combined_RMS": float(np.sqrt(np.mean(combined**2))),
        "combined_Linf": float(np.max(np.abs(combined))),
        "combined_sample_count": int(combined.size),
        "aggregation": (
            "concatenate Hamiltonian then Phi over the identical mask; "
            "RMS=sqrt(mean(x^2)); Linf=max(abs(x))"
        ),
    }


def _score_bundle(record, mask):
    families = {
        "balanced_normalized": record["balanced_normalized"],
        "absolute_raw_normalized": record["raw_normalized"],
        "raw_unscaled": record["raw"],
        "reference_defect_unscaled": record["defect"],
        "balanced_unscaled": record["balanced"],
    }
    return {name: _field_scores(fields, mask) for name, fields in families.items()}


def _verify_open_bulk_reassembly(record, shape):
    """Recompute every algebraic lane rather than trusting reported metadata."""
    maxima = {
        "balanced_equals_raw_minus_defect": 0.0,
        "common_denominator": 0.0,
        "absolute_raw_normalized": 0.0,
        "balanced_normalized": 0.0,
        "reported_reassembly_defect": 0.0,
    }
    for equation in EQUATION_ORDER:
        arrays = {}
        for family in (
            "raw", "defect", "balanced", "common_denominator",
            "raw_normalized", "balanced_normalized", "reassembly_defect",
        ):
            try:
                value = np.asarray(record[family][equation], dtype=float)
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"bulk record is missing {family}/{equation}"
                ) from error
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"bulk record {family}/{equation} is invalid")
            arrays[family] = value
        expected_balanced = arrays["raw"]-arrays["defect"]
        expected_denominator = np.maximum(
            1.0, np.abs(arrays["raw"])+np.abs(arrays["defect"]),
        )
        expected_raw_normalized = np.abs(arrays["raw"])/expected_denominator
        expected_balanced_normalized = (
            np.abs(expected_balanced)/expected_denominator
        )
        maxima["balanced_equals_raw_minus_defect"] = max(
            maxima["balanced_equals_raw_minus_defect"],
            float(np.max(np.abs(arrays["balanced"]-expected_balanced))),
        )
        maxima["common_denominator"] = max(
            maxima["common_denominator"],
            float(np.max(np.abs(arrays["common_denominator"]-expected_denominator))),
        )
        maxima["absolute_raw_normalized"] = max(
            maxima["absolute_raw_normalized"],
            float(np.max(np.abs(
                arrays["raw_normalized"]-expected_raw_normalized
            ))),
        )
        maxima["balanced_normalized"] = max(
            maxima["balanced_normalized"],
            float(np.max(np.abs(
                arrays["balanced_normalized"]-expected_balanced_normalized
            ))),
        )
        maxima["reported_reassembly_defect"] = max(
            maxima["reported_reassembly_defect"],
            float(np.max(np.abs(arrays["reassembly_defect"]))),
        )
    reported = float(record.get("reassembly_Linf", np.nan))
    if not np.isfinite(reported):
        raise ValueError("bulk reassembly metric is nonfinite")
    recomputed = float(max(maxima.values()))
    maxima["reported_reassembly_Linf"] = reported
    maxima["recomputed_reassembly_Linf"] = recomputed
    return maxima


def score_open_bulk_record(record, masks):
    """Score one open-bulk backend record on frozen retained/region masks."""
    if not isinstance(record, Mapping) or not isinstance(masks, Mapping):
        raise TypeError("bulk record and masks must be mappings")
    if tuple(record.get("equation_order", ())) != tuple(EQUATION_ORDER):
        raise ValueError("bulk equation order changed")
    retained = np.asarray(masks["retained"], dtype=bool)
    reassembly_record = _verify_open_bulk_reassembly(record, retained.shape)
    reassembly = max(
        reassembly_record["reported_reassembly_Linf"],
        reassembly_record["recomputed_reassembly_Linf"],
    )
    if record.get("lapse_used") not in (None, False):
        raise ValueError("bulk record improperly depends on lapse")
    if record.get("candidate_chi_jets_reused_in_defect") not in (None, True):
        raise ValueError("bulk reference defect does not reuse candidate chi jets")
    retained_scores = _score_bundle(record, retained)
    face_scores = {
        name: _score_bundle(record, mask)
        for name, mask in masks["faces"].items()
    }
    strip_scores = {
        name: _score_bundle(record, mask)
        for name, mask in masks["seven_index_strips"].items()
    }
    balanced = retained_scores["balanced_normalized"]
    raw = retained_scores["absolute_raw_normalized"]
    gates = {
        "combined_balanced_RMS": bool(
            balanced["combined_RMS"] <= BULK_THRESHOLDS["combined_balanced_RMS"]
        ),
        "combined_balanced_Linf": bool(
            balanced["combined_Linf"] <= BULK_THRESHOLDS["combined_balanced_Linf"]
        ),
        "combined_absolute_raw_RMS": bool(
            raw["combined_RMS"] <= BULK_THRESHOLDS["combined_absolute_raw_RMS"]
        ),
        "combined_absolute_raw_Linf": bool(
            raw["combined_Linf"] <= BULK_THRESHOLDS["combined_absolute_raw_Linf"]
        ),
        "reassembly_Linf": bool(
            reassembly <= BULK_THRESHOLDS["reassembly_Linf"]
        ),
    }
    return _freeze_mapping({
        "retained": retained_scores,
        "faces": face_scores,
        "seven_index_strips": strip_scores,
        "gates": gates,
        "numerical_gate_pass": bool(all(gates.values())),
        "reassembly_Linf": reassembly,
        "reassembly": reassembly_record,
        "provenance": {
            "protocol": PROTOCOL_IDENTIFIER,
            "equation_order": tuple(EQUATION_ORDER),
            "score_families": SCORE_FAMILIES,
            "thresholds": dict(BULK_THRESHOLDS),
            "normalization": record.get("normalization"),
            "method": record.get("method"),
            "axis_treatment": record.get("axis_treatment"),
            "source_stencil_width": record.get("source_stencil_width"),
            "mask": dict(masks["provenance"]),
            "no_boundary_rows_inserted": True,
        },
    })


def _nested_reference_jets(mapping):
    required = {
        "q": "q",
        "Phi": "phi" if "phi" in mapping else "Phi",
    }
    result = {}
    for output_name, source_name in required.items():
        lanes = {"value": np.asarray(mapping[source_name], dtype=float)}
        for lane in REPRESENTATION_JET_LANES[1:]:
            key = f"{source_name}_{lane}"
            if key not in mapping:
                raise ValueError(f"reference jet mapping is missing {key}")
            lanes[lane] = np.asarray(mapping[key], dtype=float)
        result[output_name] = lanes
    return result


def score_jet_representation_sensitivity(primary, comparator, field_order):
    """Return pointwise-scaled Q53/Q33 value/first/second jet sensitivity."""
    field_order = tuple(field_order)
    if not field_order:
        raise ValueError("jet sensitivity requires at least one field")
    groups = {
        "value": ("value",),
        "first": ("z", "r"),
        "second": ("zz", "zr", "rr"),
    }
    output = {}
    gates = {}
    for group, lanes in groups.items():
        scaled = []
        per_field = {}
        for field in field_order:
            if field not in primary or field not in comparator:
                raise ValueError(f"jet sensitivity is missing field {field}")
            local = []
            per_lane = {}
            for lane in lanes:
                left = np.asarray(primary[field][lane], dtype=float)
                right = np.asarray(comparator[field][lane], dtype=float)
                if left.shape != right.shape or not all(
                    np.all(np.isfinite(value)) for value in (left, right)
                ):
                    raise ValueError(f"jet sensitivity {field}/{lane} is invalid")
                scaled_lane = (
                    np.abs(left-right)
                    /np.maximum.reduce((np.ones_like(left), np.abs(left), np.abs(right)))
                )
                local.append(scaled_lane)
                per_lane[lane] = {
                    "scaled_RMS": float(np.sqrt(np.mean(scaled_lane**2))),
                    "scaled_Linf": float(np.max(scaled_lane)),
                    "sample_count": int(scaled_lane.size),
                }
            local = np.concatenate([value.ravel() for value in local])
            per_field[field] = {
                "scaled_RMS": float(np.sqrt(np.mean(local**2))),
                "scaled_Linf": float(np.max(local)),
                "sample_count": int(local.size),
                "derivative_directions": per_lane,
            }
            scaled.append(local)
        combined = np.concatenate(scaled)
        output[group] = {
            "fields": per_field,
            "combined_scaled_RMS": float(np.sqrt(np.mean(combined**2))),
            "combined_scaled_Linf": float(np.max(combined)),
            "combined_sample_count": int(combined.size),
            "lanes": lanes,
        }
        gates[group] = bool(
            output[group]["combined_scaled_Linf"]
            <= REPRESENTATION_THRESHOLDS[group]
        )
    return _freeze_mapping({
        "groups": output,
        "thresholds": dict(REPRESENTATION_THRESHOLDS),
        "gates": gates,
        "pass": bool(all(gates.values())),
        "normalization": "abs(Q53-Q33)/max(1,abs(Q53),abs(Q33)) pointwise",
        "aggregation": "fields and listed derivative lanes concatenated per group",
    })


def _term_surface_sensitivity(primary, comparator, mask):
    output = {}
    for family in ("raw_normalized", "balanced_normalized"):
        fields = {}
        for equation in EQUATION_ORDER:
            left = np.asarray(primary[family][equation], dtype=float)
            right = np.asarray(comparator[family][equation], dtype=float)
            if left.shape != right.shape:
                raise ValueError("reference-sensitivity bulk surfaces differ in shape")
            fields[equation] = (
                np.abs(left-right)
                /np.maximum.reduce((np.ones_like(left), np.abs(left), np.abs(right)))
            )
        output[family] = _field_scores(fields, mask)
    return _freeze_mapping(output)


def _score_metric_changes(primary, comparator):
    """Record Q53/Q33 change in every retained, face, and strip score."""
    def bundle_change(primary_bundle, comparator_bundle):
        output = {}
        for family in SCORE_FAMILIES:
            primary_family = primary_bundle[family]
            comparator_family = comparator_bundle[family]
            equations = {}
            for equation in EQUATION_ORDER:
                equations[equation] = {}
                for metric in ("RMS", "Linf"):
                    q53 = float(primary_family["equations"][equation][metric])
                    q33 = float(comparator_family["equations"][equation][metric])
                    equations[equation][metric] = {
                        "Q53": q53,
                        "Q33": q33,
                        "signed_change_Q33_minus_Q53": q33-q53,
                        "absolute_change": abs(q33-q53),
                    }
            combined = {}
            for metric in ("combined_RMS", "combined_Linf"):
                q53 = float(primary_family[metric])
                q33 = float(comparator_family[metric])
                combined[metric] = {
                    "Q53": q53,
                    "Q33": q33,
                    "signed_change_Q33_minus_Q53": q33-q53,
                    "absolute_change": abs(q33-q53),
                }
            output[family] = {"equations": equations, "combined": combined}
        return output

    if set(primary["faces"]) != set(comparator["faces"]):
        raise ValueError("Q53/Q33 bulk face-score domains differ")
    if set(primary["seven_index_strips"]) != set(
        comparator["seven_index_strips"]
    ):
        raise ValueError("Q53/Q33 bulk strip-score domains differ")
    return _freeze_mapping({
        "retained": bundle_change(primary["retained"], comparator["retained"]),
        "faces": {
            name: bundle_change(primary["faces"][name], comparator["faces"][name])
            for name in primary["faces"]
        },
        "seven_index_strips": {
            name: bundle_change(
                primary["seven_index_strips"][name],
                comparator["seven_index_strips"][name],
            )
            for name in primary["seven_index_strips"]
        },
    })


def score_parent_strip_layer_growth(lanes):
    """Apply the frozen per-parent V1-to-V2 strip Linf growth gate."""
    try:
        v1 = lanes["V1"]["authoritative"]["scores"]["seven_index_strips"]
        v2 = lanes["V2"]["authoritative"]["scores"]["seven_index_strips"]
    except (KeyError, TypeError) as error:
        raise ValueError("parent strip-layer audit lacks V1/V2 scores") from error
    if set(v1) != set(v2):
        raise ValueError("V1/V2 face-strip domains differ")
    comparisons = {}
    for family in ("balanced_normalized", "absolute_raw_normalized"):
        comparisons[family] = {}
        for equation in EQUATION_ORDER:
            comparisons[family][equation] = {}
            for strip in v1:
                coarse = float(v1[strip][family]["equations"][equation]["Linf"])
                refined = float(v2[strip][family]["equations"][equation]["Linf"])
                ceiling = max(coarse, COMMON_V2_FLOOR)
                comparisons[family][equation][strip] = {
                    "V1_Linf": coarse,
                    "V2_Linf": refined,
                    "ceiling_max_V1_floor": ceiling,
                    "passed": bool(refined <= ceiling),
                }
    passed = bool(all(
        local["passed"]
        for family in comparisons.values()
        for equation in family.values()
        for local in equation.values()
    ))
    return _freeze_mapping({
        "comparisons": comparisons,
        "pass": passed,
        "predicate": "V2 strip Linf <= max(V1 strip Linf,1e-12)",
        "families": ("balanced_normalized", "absolute_raw_normalized"),
        "RMS_can_rescue": False,
    })


def _freeze_bulk_record(record):
    output = {}
    for key, value in record.items():
        if key in (
            "raw", "defect", "balanced", "common_denominator",
            "raw_normalized", "balanced_normalized", "reassembly_defect",
        ):
            output[key] = {
                equation: _immutable_array(array)
                for equation, array in value.items()
            }
        else:
            output[key] = value
    return _freeze_mapping(output)


def evaluate_protocol125_bulk_lane(
    state,
    reference_primary,
    z,
    r,
    background,
    *,
    backend,
    physical_faces,
):
    """Evaluate and score one source-FD or analytic Protocol-125 lane."""
    z, r = _validate_coordinates(z, r)
    backend = str(backend)
    masks = frozen_bulk_region_masks(z, r, physical_faces=physical_faces)
    if backend == "source_fd7":
        if not bool(physical_faces):
            raise ValueError("source FD lane must contain physical face nodes")
        if not (
            np.array_equal(z, np.asarray(reference_primary.source_z))
            and np.array_equal(r, np.asarray(reference_primary.source_r))
        ):
            raise ValueError("source FD lane differs from the bound reference source grid")
        position = np.asarray(state.evaluate_reduced(z, r), dtype=float)
        source = np.asarray(reference_primary.source_values, dtype=float)
        record = open_anisotropic_bulk_terms_fd(
            position,
            z,
            r,
            source[:, :, REFERENCE_CHANNEL_ORDER.index("q")],
            source[:, :, REFERENCE_CHANNEL_ORDER.index("Phi")],
            background,
        )
        candidate_jets = None
        reference_jets = None
    elif backend == "analytic":
        candidate_jets = constrained_position_candidate_jets(state, z, r)
        reference_jet = reference_primary.evaluate(z, r)
        reference_jets = reference_jet.as_derivative_mapping()
        record = open_anisotropic_bulk_terms_from_jets(
            candidate_jets,
            reference_jets,
            z,
            r,
            background,
            regular_axis=bool(np.any(r == 0.0)),
        )
    else:
        raise ValueError("bulk lane backend must be source_fd7 or analytic")
    frozen_record = _freeze_bulk_record(record)
    return _freeze_mapping({
        "coordinates": {
            "z": _immutable_array(z),
            "r": _immutable_array(r),
            "sha256": hash_arrays(z, r),
        },
        "backend": backend,
        "masks": masks,
        "terms": frozen_record,
        "scores": score_open_bulk_record(frozen_record, masks),
        "candidate_jets": candidate_jets,
        "reference_jets": reference_jets,
    })


def _position_pair_members(candidate_pair):
    try:
        primary_member = candidate_pair.primary
        comparator_member = candidate_pair.comparator
    except AttributeError as error:
        raise TypeError("bulk validation requires a position Q53/Q33 pair") from error
    primary_state = getattr(primary_member, "position", primary_member)
    comparator_state = getattr(comparator_member, "position", comparator_member)
    source_fingerprint = getattr(candidate_pair, "source_fingerprint", None)
    endpoint_fingerprint = getattr(candidate_pair, "endpoint_fingerprint", None)
    if source_fingerprint is None:
        source_fingerprint = getattr(primary_member, "source_fingerprint", None)
        comparator_source = getattr(comparator_member, "source_fingerprint", None)
        if source_fingerprint is None or source_fingerprint != comparator_source:
            raise ValueError("candidate Q53/Q33 source fingerprints differ or are missing")
    if endpoint_fingerprint is None:
        endpoint_fingerprint = getattr(primary_member, "endpoint_fingerprint", None)
        comparator_endpoint = getattr(comparator_member, "endpoint_fingerprint", None)
        if endpoint_fingerprint is None or endpoint_fingerprint != comparator_endpoint:
            raise ValueError("candidate Q53/Q33 endpoint fingerprints differ or are missing")
    return (
        primary_state,
        comparator_state,
        str(source_fingerprint),
        str(endpoint_fingerprint),
        primary_member,
        comparator_member,
    )


def bind_protocol125_bulk_identity(parent_label, candidate_pair, reference_pair):
    """Bind and fingerprint one immutable candidate/reference source identity."""
    label = str(parent_label)
    if label not in SOURCE_CELL_MIDPOINT_SPECS:
        raise ValueError("Protocol-125 parent label must be N0 or N1")
    try:
        (
            primary_state,
            comparator_state,
            candidate_source_fingerprint,
            candidate_endpoint_fingerprint,
            primary_member,
            comparator_member,
        ) = _position_pair_members(candidate_pair)
        reference_primary = reference_pair.primary
        reference_comparator = reference_pair.comparator
    except AttributeError as error:
        raise TypeError("bulk validation requires constrained and reference Q53/Q33 pairs") from error
    if (
        str(getattr(primary_state, "state_name", "")) != "position"
        or str(getattr(comparator_state, "state_name", "")) != "position"
        or int(getattr(primary_state, "z_degree", -1)) != 5
        or int(getattr(comparator_state, "z_degree", -1)) != 3
    ):
        raise ValueError("candidate pair must contain position Q53/Q33 states")
    source_z = _require_immutable(primary_state.source_z, "candidate source z")
    source_r = _require_immutable(primary_state.source_r, "candidate source r")
    for name, found, expected in (
        ("candidate comparator z", comparator_state.source_z, source_z),
        ("candidate comparator r", comparator_state.source_r, source_r),
        ("reference primary z", reference_primary.source_z, source_z),
        ("reference primary r", reference_primary.source_r, source_r),
        ("reference comparator z", reference_comparator.source_z, source_z),
        ("reference comparator r", reference_comparator.source_r, source_r),
    ):
        found = _require_immutable(found, name)
        if not _arrays_bitwise_equal(found, expected):
            raise ValueError(f"{name} differs from the shared source grid")
    for name, value in (
        ("reference primary source values", reference_primary.source_values),
        ("reference primary endpoint traces", reference_primary.endpoint_z_first),
        ("reference comparator source values", reference_comparator.source_values),
        ("reference comparator endpoint traces", reference_comparator.endpoint_z_first),
    ):
        _require_immutable(value, name)
    if not (
        _arrays_bitwise_equal(
            reference_primary.source_values, reference_comparator.source_values,
        )
        and _arrays_bitwise_equal(
            reference_primary.endpoint_z_first,
            reference_comparator.endpoint_z_first,
        )
    ):
        raise ValueError("reference Q53/Q33 pair does not share exact source inputs")
    if not (
        int(reference_primary.stencil_width) == SOURCE_STENCIL_WIDTH
        and int(reference_comparator.stencil_width) == SOURCE_STENCIL_WIDTH
        and int(reference_primary.surface.z_degree) == 5
        and int(reference_comparator.surface.z_degree) == 3
        and str(reference_primary.recipe) == "primary-Q53"
        and str(reference_comparator.recipe) == "comparator-Q33"
        and str(reference_primary.surface.z_boundary)
        == "clamped_source_width7_z_first"
        and str(reference_comparator.surface.z_boundary)
        == "clamped_source_width7_z_first"
        and tuple(reference_primary.channel_order) == tuple(REFERENCE_CHANNEL_ORDER)
        and tuple(reference_comparator.channel_order) == tuple(REFERENCE_CHANNEL_ORDER)
    ):
        raise ValueError("reference pair degree/channel/stencil contract changed")
    for attribute in (
        "compact_wall_contract_id",
        "outer_open_face_contract_id",
        "compact_wall_contract_fingerprint",
        "outer_open_face_contract_fingerprint",
    ):
        if str(getattr(primary_state, attribute)) != str(
            getattr(comparator_state, attribute)
        ):
            raise ValueError(f"candidate Q53/Q33 {attribute} differs")
    if not _arrays_bitwise_equal(
        primary_state.stored_z_first_endpoints,
        comparator_state.stored_z_first_endpoints,
    ):
        raise ValueError("candidate Q53/Q33 endpoint arrays differ bitwise")
    if not _arrays_bitwise_equal(
        primary_state.outer_ownership_mask,
        comparator_state.outer_ownership_mask,
    ):
        raise ValueError("candidate Q53/Q33 outer ownership differs bitwise")
    midpoint = source_cell_midpoint_coordinates(label, source_z, source_r)
    reference_values_sha256 = hash_arrays(
        reference_primary.source_values,
        reference_primary.endpoint_z_first,
    )
    identity = {
        "protocol": PROTOCOL_IDENTIFIER,
        "parent_label": label,
        "source_shape": (len(source_z), len(source_r)),
        "source_coordinate_sha256": hash_arrays(source_z, source_r),
        "midpoint_coordinate_sha256": midpoint.coordinate_sha256,
        "candidate_source_fingerprint": candidate_source_fingerprint,
        "candidate_endpoint_fingerprint": candidate_endpoint_fingerprint,
        "candidate_primary_fingerprint": _object_fingerprint(
            primary_member, "candidate primary",
        ),
        "candidate_comparator_fingerprint": _object_fingerprint(
            comparator_member, "candidate comparator",
        ),
        "candidate_pair_fingerprint": _object_fingerprint(
            candidate_pair, "candidate position pair",
        ),
        "reference_pair_fingerprint": _object_fingerprint(
            reference_pair, "finite-wall reference pair",
        ),
        "reference_primary_fingerprint": _object_fingerprint(
            reference_primary, "finite-wall reference primary",
        ),
        "reference_comparator_fingerprint": _object_fingerprint(
            reference_comparator, "finite-wall reference comparator",
        ),
        "reference_source_and_trace_sha256": reference_values_sha256,
        "source_stencil_width": SOURCE_STENCIL_WIDTH,
        "shared_source_coordinates_bitwise": True,
        "shared_reference_inputs_bitwise": True,
        "inputs_immutable": True,
    }
    digest = hashlib.sha256()
    for name, value in sorted(identity.items()):
        digest.update(str(name).encode())
        digest.update(b"\0")
        digest.update(str(value).encode())
        digest.update(b"\0")
    identity["binding_sha256"] = digest.hexdigest()
    return _freeze_mapping(identity)


def _reference_sensitivity(reference_pair, z, r):
    primary_mapping = reference_pair.primary.evaluate(z, r).as_derivative_mapping()
    comparator_mapping = reference_pair.comparator.evaluate(z, r).as_derivative_mapping()
    return score_jet_representation_sensitivity(
        _nested_reference_jets(primary_mapping),
        _nested_reference_jets(comparator_mapping),
        ("q", "Phi"),
    )


def _candidate_sensitivity(candidate_pair, z, r):
    primary_state, comparator_state, *_ = _position_pair_members(candidate_pair)
    return score_jet_representation_sensitivity(
        constrained_position_coordinate_jets(primary_state, z, r),
        constrained_position_coordinate_jets(comparator_state, z, r),
        COORDINATE_COMPONENT_ORDER,
    )


def run_protocol125_bulk_validation(
    parent_label,
    candidate_pair,
    reference_pair,
    background,
):
    """Run the frozen source/midpoint/V0/V1/V2 bulk measurement matrix.

    The return value is an in-memory diagnostic record only.  A single-parent
    result cannot authorize acceleration until the independent N0/N1
    common-V2 comparison also passes.
    """
    if not isinstance(candidate_pair, PositionOnlyConstrainedHermitePair):
        raise TypeError(
            "bulk prerequisite requires the position-only constrained Q53/Q33 pair"
        )
    identity_before = bind_protocol125_bulk_identity(
        parent_label, candidate_pair, reference_pair,
    )
    mass_squared = float(background["mass_squared"])
    if not np.isfinite(mass_squared):
        raise ValueError("Protocol-125 mass_squared must be finite")
    frozen_background = MappingProxyType({"mass_squared": mass_squared})
    state, _, _, _, _, _ = _position_pair_members(candidate_pair)
    source_z = np.asarray(state.source_z)
    source_r = np.asarray(state.source_r)
    midpoint = source_cell_midpoint_coordinates(parent_label, source_z, source_r)
    validation = frozen_validation_meshes()
    coordinates = {
        "source": (source_z, source_r, True, "source_fd7"),
        "midpoint": (midpoint.z, midpoint.r, False, "analytic"),
        **{
            name: (
                np.asarray(validation[name]["z"]),
                np.asarray(validation[name]["r"]),
                True,
                "analytic",
            )
            for name in ("V0", "V1", "V2")
        },
    }
    lanes = {}
    representation_gates = []
    for lane_name in ALL_LANES:
        z, r, physical_faces, backend = coordinates[lane_name]
        primary = evaluate_protocol125_bulk_lane(
            state,
            reference_pair.primary,
            z,
            r,
            frozen_background,
            backend=backend,
            physical_faces=physical_faces,
        )
        lane = {"authoritative": primary}
        if backend == "source_fd7":
            lane["reference_Q33_bulk_sensitivity"] = {
                "identically_zero": True,
                "reason": "Q53 and Q33 bind the exact same source q/Phi arrays",
            }
        else:
            reference_sensitivity = _reference_sensitivity(reference_pair, z, r)
            lane["reference_Q53_Q33_jet_sensitivity"] = reference_sensitivity
            candidate_jets = primary["candidate_jets"]
            comparator_reference = reference_pair.comparator.evaluate(
                z, r,
            ).as_derivative_mapping()
            comparator_terms = _freeze_bulk_record(
                open_anisotropic_bulk_terms_from_jets(
                    candidate_jets,
                    comparator_reference,
                    z,
                    r,
                    frozen_background,
                    regular_axis=bool(np.any(r == 0.0)),
                )
            )
            comparator_scores = score_open_bulk_record(
                comparator_terms, primary["masks"],
            )
            lane["reference_Q33_bulk_sensitivity"] = {
                "terms": comparator_terms,
                "scores": comparator_scores,
                "score_changes": _score_metric_changes(
                    primary["scores"], comparator_scores,
                ),
                "surface_difference": _term_surface_sensitivity(
                    primary["terms"], comparator_terms, primary["masks"]["retained"],
                ),
            }
            representation_gates.append(bool(reference_sensitivity["pass"]))
        if lane_name in ("V0", "V1", "V2"):
            candidate_sensitivity = _candidate_sensitivity(candidate_pair, z, r)
            lane["candidate_Q53_Q33_position_spatial_sensitivity"] = (
                candidate_sensitivity
            )
            representation_gates.append(bool(candidate_sensitivity["pass"]))
        lanes[lane_name] = lane

    identity_after = bind_protocol125_bulk_identity(
        parent_label, candidate_pair, reference_pair,
    )
    if dict(identity_before) != dict(identity_after):
        raise RuntimeError("bulk source/reference identity changed during validation")
    lane_gates = {
        name: bool(lanes[name]["authoritative"]["scores"]["numerical_gate_pass"])
        for name in ALL_LANES
    }
    strip_layer = score_parent_strip_layer_growth(lanes)
    parent_bulk_pass = bool(
        all(lane_gates.values())
        and all(representation_gates)
        and strip_layer["pass"]
    )
    return _freeze_mapping({
        "protocol": PROTOCOL_IDENTIFIER,
        "parent_label": str(parent_label),
        "identity": identity_before,
        "background": dict(frozen_background),
        "lanes": lanes,
        "adjudication": {
            "lane_numerical_gates": lane_gates,
            "all_lane_numerical_gates_pass": bool(all(lane_gates.values())),
            "all_Q53_Q33_bulk_jet_gates_pass": bool(all(representation_gates)),
            "strip_layer_growth_gate": strip_layer,
            "parent_bulk_pass": parent_bulk_pass,
            "common_V2_two_parent_gate_required": True,
            "protocol_two_parent_bulk_pass": False,
            "fail_closed_pending_second_parent": True,
        },
        "scientific_artifact_written": False,
        "acceleration_authorized": False,
    })


def _common_v2_metric_record(audit):
    try:
        protocol = audit["protocol"]
        identity = audit["identity"]
        parent_bulk_pass = audit["adjudication"]["parent_bulk_pass"]
        lane = audit["lanes"]["V2"]["authoritative"]
        score = lane["scores"]["retained"]
        strip_scores = lane["scores"]["seven_index_strips"]
        coordinate_sha256 = lane["coordinates"]["sha256"]
        mask_sha256 = lane["masks"]["provenance"]["mask_sha256"]
        score_protocol = lane["scores"]["provenance"]["protocol"]
    except (KeyError, TypeError) as error:
        raise ValueError("common-V2 audit record is incomplete") from error
    if protocol != PROTOCOL_IDENTIFIER or score_protocol != PROTOCOL_IDENTIFIER:
        raise ValueError("common-V2 protocol provenance differs")
    if identity["parent_label"] != audit["parent_label"]:
        raise ValueError("common-V2 source identity label differs")
    if not identity.get("binding_sha256"):
        raise ValueError("common-V2 source/reference binding is missing")
    if not bool(parent_bulk_pass):
        raise ValueError("common-V2 comparison requires each parent bulk gate to pass")
    expected = frozen_validation_meshes()["V2"]["sha256"]
    if coordinate_sha256 != expected:
        raise ValueError("common-V2 coordinates differ from the frozen mesh")
    strip_linf = {
        family: {
            equation: {
                strip: float(
                    strip_scores[strip][family]["equations"][equation]["Linf"]
                )
                for strip in strip_scores
            }
            for equation in EQUATION_ORDER
        }
        for family in ("balanced_normalized", "absolute_raw_normalized")
    }
    return {
        "combined_balanced_RMS": float(
            score["balanced_normalized"]["combined_RMS"]
        ),
        "combined_balanced_Linf": float(
            score["balanced_normalized"]["combined_Linf"]
        ),
        "combined_absolute_raw_RMS": float(
            score["absolute_raw_normalized"]["combined_RMS"]
        ),
        "combined_absolute_raw_Linf": float(
            score["absolute_raw_normalized"]["combined_Linf"]
        ),
        "coordinate_sha256": coordinate_sha256,
        "mask_sha256": mask_sha256,
        "binding_sha256": str(identity["binding_sha256"]),
        "strip_Linf": strip_linf,
    }


def _nonworsening_with_floor(coarse, refined, floor=COMMON_V2_FLOOR):
    return bool(refined <= coarse or (coarse <= floor and refined <= floor))


def compare_protocol125_common_v2(n0_audit, n1_audit):
    """Apply retained and per-strip common-V2 N1 nonworsening predicates."""
    if str(n0_audit.get("parent_label")) != "N0":
        raise ValueError("coarse common-V2 audit must be N0")
    if str(n1_audit.get("parent_label")) != "N1":
        raise ValueError("refined common-V2 audit must be N1")
    n0 = _common_v2_metric_record(n0_audit)
    n1 = _common_v2_metric_record(n1_audit)
    if n0["coordinate_sha256"] != n1["coordinate_sha256"]:
        raise ValueError("N0/N1 common-V2 coordinate identities differ")
    if n0["mask_sha256"] != n1["mask_sha256"]:
        raise ValueError("N0/N1 common-V2 retained masks differ")
    if n0["binding_sha256"] == n1["binding_sha256"]:
        raise ValueError("N0/N1 must not reuse one source/reference binding")
    metric_names = (
        "combined_balanced_RMS",
        "combined_balanced_Linf",
        "combined_absolute_raw_RMS",
        "combined_absolute_raw_Linf",
    )
    comparisons = {
        name: {
            "N0": n0[name],
            "N1": n1[name],
            "floor": COMMON_V2_FLOOR,
            "passed": _nonworsening_with_floor(n0[name], n1[name]),
        }
        for name in metric_names
    }
    core_pass = bool(all(value["passed"] for value in comparisons.values()))
    if set(n0["strip_Linf"]) != set(n1["strip_Linf"]):
        raise ValueError("N0/N1 common-V2 strip families differ")
    strip_comparisons = {}
    for family in n0["strip_Linf"]:
        strip_comparisons[family] = {}
        if set(n0["strip_Linf"][family]) != set(n1["strip_Linf"][family]):
            raise ValueError("N0/N1 common-V2 strip equations differ")
        for equation in n0["strip_Linf"][family]:
            strip_comparisons[family][equation] = {}
            coarse_strips = n0["strip_Linf"][family][equation]
            refined_strips = n1["strip_Linf"][family][equation]
            if set(coarse_strips) != set(refined_strips):
                raise ValueError("N0/N1 common-V2 strip domains differ")
            for strip in coarse_strips:
                coarse = coarse_strips[strip]
                refined = refined_strips[strip]
                strip_comparisons[family][equation][strip] = {
                    "N0": coarse,
                    "N1": refined,
                    "floor": COMMON_V2_FLOOR,
                    "passed": _nonworsening_with_floor(coarse, refined),
                }
    strip_pass = bool(all(
        item["passed"]
        for family in strip_comparisons.values()
        for equation in family.values()
        for item in equation.values()
    ))
    return _freeze_mapping({
        "comparisons": comparisons,
        "core_nonworsening_pass": core_pass,
        "strip_comparisons": strip_comparisons,
        "strip_nonworsening_pass": strip_pass,
        "predicate": "N1<=N0, or both values<=1e-12",
        "common_coordinate_sha256": n0["coordinate_sha256"],
        "common_mask_sha256": n0["mask_sha256"],
        "protocol_common_V2_pass": bool(core_pass and strip_pass),
        "fail_closed": False,
    })
