#!/usr/bin/env python3
"""Archive-only decomposition of the failed A790 matched Phase A.

This executable never rebuilds the parent and never calls an evolution RHS.
It reads the immutable Phase-A parent/result artifacts and separates:

* regular q4/q5 coefficient error from reconstructed coordinate-metric error;
* compact-wall collars from a fixed compact interior;
* spline-degree sensitivity from compact-endpoint prescription sensitivity;
* the native ``DJ[q0].a0`` wall defect by wall, row, and radius.

The original Phase-A recovery namespace is input-only.  The sole output is a
new JSON diagnostic artifact under ``results/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bhps.junction_preservation_diagnostic import _orthonormal_frames  # noqa: E402
from bhps.junction_second_preservation_diagnostic import (  # noqa: E402
    wall_junction_second_tangent,
)
from bhps.matched_staged_continuum import (  # noqa: E402
    ContinuousReducedParent,
    ProjectedJetField,
    TensorSplineSurface,
    balanced_constraint_audit,
    build_mode_neutral_case,
    normalized_error,
    projected_second_wall_audit,
    raw_hamiltonian_audit,
)
from bhps.phase_a2_diagnostics import (  # noqa: E402
    endpoint_data_matched_q53_parent,
    endpoint_trace_comparison,
)
from bhps.recovery_indexer import atomic_write_json, sha256_file  # noqa: E402
from bhps.regular_so3_gh_reduction import FIELD_ORDER  # noqa: E402


PARENT = Path(
    "results/corrected_A790_matched_staged_continuum_recovery/"
    "phase_a_parent_projection.npz"
)
PHASE_A = Path("results/corrected_A790_matched_staged_continuum_phase_a.json")
RECOVERY_INDEX = Path(
    "results/corrected_A790_matched_staged_continuum_recovery/index.json"
)
PROTOCOL = Path("notes/120_A790_matched_staged_continuum_protocol.md")
OUTPUT = Path("results/corrected_A790_phase_a2_failure_decomposition.json")
TARGETS = ("G8", "G9", "G10")
EXPECTED_PARENT_SHA256 = (
    "30c578ce142159a8e0842a22afab8b436bd85af55f1516c1bfe11fce96968dc7"
)
EXPECTED_PHASE_A_SHA256 = (
    "af602f00236550027d6e934e652917cb8e63a56a40f32ae6b80fb718265ee47a"
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _jet(archive, prefix):
    return ProjectedJetField(
        np.asarray(archive[f"{prefix}_z"]),
        np.asarray(archive[f"{prefix}_r"]),
        np.asarray(archive[f"{prefix}_q"]),
        np.asarray(archive[f"{prefix}_first"]),
        np.asarray(archive[f"{prefix}_second"]),
    )


def _as_lane(array):
    """Flatten every component after the leading (z,r) coordinates."""
    value = np.asarray(array, dtype=float)
    if value.ndim < 2:
        raise ValueError("comparison lane must contain z and r axes")
    return value.reshape(value.shape[0], value.shape[1], -1)


def _lane_statistics(reference, comparison, z, r):
    x = _as_lane(reference)
    y = _as_lane(comparison)
    if x.shape != y.shape or x.shape[:2] != (len(z), len(r)):
        raise ValueError("comparison lane shape mismatch")
    difference = y - x
    scale = max(1.0, float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    absolute = np.abs(difference)
    maximum_index = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
    z_index, r_index, component = (int(value) for value in maximum_index)
    xi = (np.asarray(z) - float(z[0])) / float(z[-1] - z[0])
    distance = np.minimum(xi, 1.0-xi)
    fixed_five = distance <= 0.05 + 1e-15
    fixed_interior = distance > 0.05 + 1e-15
    seven = np.zeros(len(z), dtype=bool)
    seven[:7] = True
    seven[-7:] = True
    lower_seven = np.zeros(len(z), dtype=bool)
    lower_seven[:7] = True
    upper_seven = np.zeros(len(z), dtype=bool)
    upper_seven[-7:] = True

    total_energy = float(np.sum(difference**2))

    def region(mask):
        local = difference[np.asarray(mask, dtype=bool)]
        energy = float(np.sum(local**2))
        return {
            "z_node_count": int(np.count_nonzero(mask)),
            "scaled_RMS": float(np.sqrt(np.mean(local**2))/scale),
            "scaled_Linf": float(np.max(np.abs(local))/scale),
            "squared_error_fraction": (
                float(energy/total_energy) if total_energy > 0.0 else 0.0
            ),
        }

    return {
        "scale": scale,
        "scaled_RMS": float(np.sqrt(np.mean(difference**2))/scale),
        "scaled_Linf": float(np.max(absolute)/scale),
        "absolute_Linf": float(np.max(absolute)),
        "maximum": {
            "z_index": z_index,
            "r_index": r_index,
            "component_index": component,
            "z": float(z[z_index]),
            "r": float(r[r_index]),
            "compact_wall_fraction": float(distance[z_index]),
        },
        "compact_regions": {
            "fixed_five_percent_wall_collars": region(fixed_five),
            "fixed_five_percent_interior": region(fixed_interior),
            "seven_node_wall_collars": region(seven),
            "lower_seven_node_collar": region(lower_seven),
            "upper_seven_node_collar": region(upper_seven),
            "seven_node_interior": region(~seven),
        },
    }


def _reduced_lanes(jet, fields=slice(None)):
    q = np.asarray(jet.reduced_fields)[:, :, fields]
    first = np.asarray(jet.reduced_first)[1:, :, :, fields]
    acceleration = np.asarray(jet.reduced_second)[0, 0, :, :, fields]
    spatial_second = np.asarray(jet.reduced_second)[1:, 1:, :, :, fields]
    return {
        "position": _as_lane(q),
        "first_spatial": np.moveaxis(first, (0, 1, 2), (2, 0, 1)).reshape(
            len(jet.z), len(jet.r), -1
        ),
        "acceleration": _as_lane(acceleration),
        "spatial_second": np.moveaxis(
            spatial_second, (0, 1, 2, 3), (2, 3, 0, 1),
        ).reshape(len(jet.z), len(jet.r), -1),
    }


def _coordinate_transform(r, q4q5_only=False):
    """Return M, M_r, M_rr for physical coordinate components M(r)q."""
    r = np.asarray(r, dtype=float)
    if q4q5_only:
        matrix = np.zeros((len(r), 2, 9))
        first = np.zeros_like(matrix)
        second = np.zeros_like(matrix)
        matrix[:, 0, 4] = r**2
        first[:, 0, 4] = 2.0*r
        second[:, 0, 4] = 2.0
        matrix[:, 1, 5] = r
        first[:, 1, 5] = 1.0
        names = ("r^2 q4 contribution to h_rr", "r q5=h_0r")
        return matrix, first, second, names

    # (h_z0,h_zr,h_00,h_perp,h_rr,h_0r,h_zz)
    matrix = np.zeros((len(r), 7, 9))
    first = np.zeros_like(matrix)
    second = np.zeros_like(matrix)
    matrix[:, 0, 0] = 1.0
    matrix[:, 1, 1] = r
    first[:, 1, 1] = 1.0
    matrix[:, 2, 2] = 1.0
    matrix[:, 3, 3] = 1.0
    matrix[:, 4, 3] = 1.0
    matrix[:, 4, 4] = r**2
    first[:, 4, 4] = 2.0*r
    second[:, 4, 4] = 2.0
    matrix[:, 5, 5] = r
    first[:, 5, 5] = 1.0
    matrix[:, 6, 6] = 1.0
    names = ("h_z0", "h_zr", "h_00", "h_perp", "h_rr", "h_0r", "h_zz")
    return matrix, first, second, names


def _physical_lanes(jet, q4q5_only=False):
    q = np.asarray(jet.reduced_fields)
    first = np.asarray(jet.reduced_first)
    second = np.asarray(jet.reduced_second)
    matrix, matrix_r, matrix_rr, names = _coordinate_transform(
        jet.r, q4q5_only=q4q5_only,
    )

    def apply(operator, values):
        return np.einsum("rof,zrf->zro", operator, values)

    position = apply(matrix, q)
    first_z = apply(matrix, first[1])
    first_r = apply(matrix, first[2]) + apply(matrix_r, q)
    second_zz = apply(matrix, second[1, 1])
    second_zr = apply(matrix, second[1, 2]) + apply(matrix_r, first[1])
    second_rr = (
        apply(matrix, second[2, 2])
        + 2.0*apply(matrix_r, first[2])
        + apply(matrix_rr, q)
    )
    acceleration = apply(matrix, second[0, 0])
    return {
        "component_order": names,
        "position": position,
        "first_spatial": np.concatenate((first_z, first_r), axis=2),
        "acceleration": acceleration,
        "spatial_second": np.concatenate(
            (second_zz, second_zr, second_zr, second_rr), axis=2,
        ),
    }


def _comparison(reference, comparison):
    if not (
        np.array_equal(reference.z, comparison.z)
        and np.array_equal(reference.r, comparison.r)
    ):
        raise ValueError("comparison grids differ")
    z = np.asarray(reference.z)
    r = np.asarray(reference.r)
    reduced = _reduced_lanes(reference)
    reduced_other = _reduced_lanes(comparison)
    raw = _reduced_lanes(reference, slice(4, 6))
    raw_other = _reduced_lanes(comparison, slice(4, 6))
    induced = _physical_lanes(reference, q4q5_only=True)
    induced_other = _physical_lanes(comparison, q4q5_only=True)
    physical = _physical_lanes(reference)
    physical_other = _physical_lanes(comparison)
    result = {
        "raw_reduced_all_fields": {},
        "raw_q4_q5": {},
        "q4_q5_induced_coordinate_components": {
            "component_order": list(induced["component_order"]),
        },
        "reconstructed_coordinate_metric": {
            "component_order": list(physical["component_order"]),
        },
        "q4_q5_squared_error_fraction_of_reduced_error": {},
    }
    for lane in ("position", "first_spatial", "acceleration", "spatial_second"):
        result["raw_reduced_all_fields"][lane] = _lane_statistics(
            reduced[lane], reduced_other[lane], z, r,
        )
        result["raw_q4_q5"][lane] = _lane_statistics(
            raw[lane], raw_other[lane], z, r,
        )
        result["q4_q5_induced_coordinate_components"][lane] = _lane_statistics(
            induced[lane], induced_other[lane], z, r,
        )
        result["reconstructed_coordinate_metric"][lane] = _lane_statistics(
            physical[lane], physical_other[lane], z, r,
        )
        numerator = float(np.sum((raw_other[lane]-raw[lane])**2))
        denominator = float(np.sum((reduced_other[lane]-reduced[lane])**2))
        result["q4_q5_squared_error_fraction_of_reduced_error"][lane] = (
            float(numerator/denominator) if denominator > 0.0 else 0.0
        )
    return result


def _not_a_knot_parent(parent_jet, degree, identity):
    z = np.asarray(parent_jet.z)
    r = np.asarray(parent_jet.r)
    return ContinuousReducedParent(
        TensorSplineSurface.build(
            z, r, parent_jet.reduced_fields, degree=degree,
        ),
        TensorSplineSurface.build(
            z, r, parent_jet.reduced_first[0], degree=degree,
        ),
        TensorSplineSurface.build(
            z, r, parent_jet.reduced_second[0, 0], degree=degree,
        ),
        parent_identity=identity,
    )


def _source_lanes(jet, background, mass_squared, label):
    geometry = {
        "name": label,
        "z": np.asarray(jet.z),
        "r": np.asarray(jet.r),
        "background": background,
        "mass_squared": float(mass_squared),
        "fold_amplitude": 7.90,
        "jet_field": jet,
    }
    bundle = build_mode_neutral_case(geometry, label)
    return {
        "source": np.asarray(bundle.source0),
        "source_time": np.asarray(bundle.source_time0),
        "memory": np.asarray(bundle.memory0),
    }


def _source_comparison(reference, comparison, z, r):
    return {
        lane: _lane_statistics(reference[lane], comparison[lane], z, r)
        for lane in ("source", "source_time", "memory")
    }


def _proper_radius(q, r, wall_index):
    radial_metric = q[wall_index, :, 3] + r**2*q[wall_index, :, 4]
    if np.any(radial_metric <= 0.0):
        raise RuntimeError("wall radial metric is not positive")
    speed = np.sqrt(radial_metric)
    increments = 0.5*(speed[1:]+speed[:-1])*np.diff(r)
    return np.concatenate(([0.0], np.cumsum(increments)))


def _profile_summary(values, r, proper, outer_points=7):
    values = np.asarray(values, dtype=float)
    absolute = np.abs(values)
    maximum = int(np.argmax(absolute))
    squared = absolute**2
    total = float(np.sum(squared))
    cumulative = np.cumsum(squared)
    r90_index = int(np.searchsorted(cumulative, 0.9*total)) if total else 0
    outer_start = max(0, len(r)-int(outer_points))
    samples = {}
    for requested in (0.0, 1.0, 2.0):
        index = int(np.argmin(np.abs(r-requested)))
        samples[f"r_{requested:g}"] = {
            "index": index, "r": float(r[index]), "value": float(values[index]),
        }
    return {
        "RMS": float(np.sqrt(np.mean(values**2))),
        "Linf": float(absolute[maximum]),
        "maximum_index": maximum,
        "maximum_r": float(r[maximum]),
        "maximum_proper_radius": float(proper[maximum]),
        "axis_value": float(values[0]),
        "r_containing_90_percent_squared_profile": float(r[r90_index]),
        "proper_radius_containing_90_percent_squared_profile": float(
            proper[r90_index]
        ),
        "squared_profile_fraction_r_at_most_2": (
            float(np.sum(squared[r <= 2.0+1e-15])/total) if total else 0.0
        ),
        "outer_seven_point_squared_profile_fraction": (
            float(np.sum(squared[outer_start:])/total) if total else 0.0
        ),
        "samples": samples,
    }


def _wall_record(jet, background):
    q = np.asarray(jet.reduced_fields)
    v = np.asarray(jet.reduced_first[0])
    a = np.asarray(jet.reduced_second[0, 0])
    r = np.asarray(jet.r)
    result = {
        "maximum_absolute_velocity": float(np.max(np.abs(v))),
        "walls": {},
    }
    for wall, wall_index in (("lower", 0), ("upper", -1)):
        record = wall_junction_second_tangent(
            q, v, a, jet.z, r, background, wall, 7,
        )
        frames, frame_defect = _orthonormal_frames(record["metric_tensor"])

        def orthonormal(key):
            return np.einsum(
                "nai,nab,nbj->nij", frames, record[key], frames,
            )

        linear_hat = orthonormal("DJ_acceleration_tensor")
        hessian_hat = orthonormal("D2J_velocity_velocity_tensor")
        second_hat = orthonormal("DX2J_tensor")
        metric_linear = np.linalg.norm(linear_hat, axis=(1, 2))
        metric_hessian = np.linalg.norm(hessian_hat, axis=(1, 2))
        metric_second = np.linalg.norm(second_hat, axis=(1, 2))
        metric_normalized = metric_second/(1.0+metric_linear+metric_hessian)
        separate = record["separate_rows"]
        phi_linear = np.asarray(separate["DJ_Phi_robin_acceleration"])
        phi_hessian = np.asarray(separate["D2_Phi_robin_velocity_velocity"])
        phi_second = np.asarray(separate["DX2_Phi_robin"])
        phi_normalized = np.abs(phi_second)/(
            1.0+np.abs(phi_linear)+np.abs(phi_hessian)
        )
        chi_linear = np.asarray(separate["DJ_chi_neumann_acceleration"])
        chi_second = np.asarray(separate["DX2_chi_neumann"])
        chi_normalized = np.abs(chi_second)/(1.0+np.abs(chi_linear))
        proper = _proper_radius(q, r, wall_index)
        component_profiles = {
            name: np.asarray(record["components"][name]["DJ_acceleration"])
            for name in ("tt", "rr", "sphere", "tr")
        }
        profiles = {
            "metric_orthonormal_frobenius": metric_linear,
            "metric_normalized": metric_normalized,
            "Phi_DJ_acceleration": phi_linear,
            "Phi_normalized": phi_normalized,
            "chi_DJ_acceleration": chi_linear,
            "chi_normalized": chi_normalized,
            **{f"metric_component_{name}": value
               for name, value in component_profiles.items()},
        }
        result["walls"][wall] = {
            "proper_radius": proper.tolist(),
            "frame_defect_Linf": float(np.max(np.abs(frame_defect))),
            "DX2_equals_DJ_plus_D2_Linf": float(np.max(np.abs(
                np.asarray(record["DX2J_tensor"])
                - np.asarray(record["DJ_acceleration_tensor"])
                - np.asarray(record["D2J_velocity_velocity_tensor"])
            ))),
            "D2_velocity_velocity_orthonormal_Linf": float(
                np.max(metric_hessian)
            ),
            "profiles": {
                name: {
                    "values": np.asarray(profile).tolist(),
                    "summary": _profile_summary(profile, r, proper),
                } for name, profile in profiles.items()
            },
        }
    return result


def _wall_archive_crosscheck(phase_a, label, recomputed):
    archived = phase_a["targets"][label]["projected_second_wall"]
    maximum = 0.0
    mapping = {
        "metric_normalized": "metric_normalized",
        "Phi_DJ_acceleration": "Phi_DJ_acceleration",
        "Phi_normalized": "Phi_normalized",
        "chi_DJ_acceleration": "chi_DJ_acceleration",
        "chi_normalized": "chi_normalized",
    }
    for wall in ("lower", "upper"):
        profiles = archived["walls"][wall]["profiles"]
        for local_name, archive_name in mapping.items():
            local = np.asarray(
                recomputed["walls"][wall]["profiles"][local_name]["values"]
            )
            saved = np.asarray(profiles[archive_name])
            maximum = max(maximum, float(np.max(np.abs(local-saved))))
    return {
        "maximum_absolute_profile_difference": maximum,
        "bitwise_reproduced": bool(maximum == 0.0),
    }


def _constraint_record(jet, background, mass_squared, reference):
    return {
        "balanced": balanced_constraint_audit(jet, background, reference),
        "raw_hamiltonian": raw_hamiltonian_audit(jet, mass_squared),
    }


def analyze(include_sources=True):
    if sha256_file(PARENT) != EXPECTED_PARENT_SHA256:
        raise RuntimeError("immutable Phase-A parent SHA-256 mismatch")
    if sha256_file(PHASE_A) != EXPECTED_PHASE_A_SHA256:
        raise RuntimeError("immutable Phase-A result SHA-256 mismatch")
    phase_a = json.loads(PHASE_A.read_text())
    with np.load(PARENT, allow_pickle=False) as archive:
        parent = _jet(archive, "p11")
        primary_parent = ContinuousReducedParent.from_arrays(archive)
        cubic_nak_parent = _not_a_knot_parent(parent, 3, "P11-cubic-not-a-knot")
        quintic_nak_parent = _not_a_knot_parent(parent, 5, "P11-quintic-not-a-knot")
        q53_parent = endpoint_data_matched_q53_parent(
            parent, parent.z, parent.r,
        )
        background = json.loads(str(archive["background_json"]))
        mass_squared = float(archive["mass_squared"])
        q5_keys = [
            key for key in archive.files
            if key.endswith(("_q", "_first", "_second"))
            and key.startswith(("g8", "g9", "g10"))
        ]
        q5_exact_zero = bool(all(
            np.count_nonzero(np.asarray(archive[key])[..., 5]) == 0
            for key in q5_keys
        ))

        targets = {}
        for label in TARGETS:
            prefix = label.lower()
            primary = _jet(archive, prefix)
            adverse = _jet(archive, f"{prefix}_quintic")
            p10 = _jet(archive, f"{prefix}_p10")
            cubic_nak = cubic_nak_parent.project(primary.z, primary.r)
            quintic_nak = quintic_nak_parent.project(primary.z, primary.r)
            q53 = q53_parent.project(primary.z, primary.r)
            reference_path = Path(
                "results/corrected_A790_matched_staged_continuum_recovery/"
                f"phase_a_reference_{label}.npz"
            )
            with np.load(reference_path, allow_pickle=False) as reference_archive:
                reference = {
                    "q": np.asarray(reference_archive["q"]),
                    "phi": np.asarray(reference_archive["phi"]),
                }
            endpoint_comparisons = {
                "primary_clamped_cubic_vs_cubic_not_a_knot": _comparison(
                    primary, cubic_nak,
                ),
                "primary_clamped_cubic_vs_endpoint_data_matched_Q53": (
                    _comparison(primary, q53)
                ),
                "degree_only_cubic_not_a_knot_vs_quintic_not_a_knot": (
                    _comparison(cubic_nak, quintic_nak)
                ),
                "primary_clamped_cubic_vs_quintic_not_a_knot": _comparison(
                    primary, quintic_nak,
                ),
            }
            source_records = {}
            if include_sources:
                source_primary = _source_lanes(
                    primary, background, mass_squared, f"{label}-primary",
                )
                source_cubic_nak = _source_lanes(
                    cubic_nak, background, mass_squared, f"{label}-cubic-NAK",
                )
                source_quintic_nak = _source_lanes(
                    quintic_nak, background, mass_squared, f"{label}-quintic-NAK",
                )
                source_q53 = _source_lanes(
                    q53, background, mass_squared, f"{label}-Q53-endpoint-matched",
                )
                source_records = {
                    "primary_clamped_cubic_vs_cubic_not_a_knot": (
                        _source_comparison(
                            source_primary, source_cubic_nak,
                            primary.z, primary.r,
                        )
                    ),
                    "degree_only_cubic_not_a_knot_vs_quintic_not_a_knot": (
                        _source_comparison(
                            source_cubic_nak, source_quintic_nak,
                            primary.z, primary.r,
                        )
                    ),
                    "primary_clamped_cubic_vs_endpoint_data_matched_Q53": (
                        _source_comparison(
                            source_primary, source_q53,
                            primary.z, primary.r,
                        )
                    ),
                    "primary_clamped_cubic_vs_quintic_not_a_knot": (
                        _source_comparison(
                            source_primary, source_quintic_nak,
                            primary.z, primary.r,
                        )
                    ),
                }
            wall = _wall_record(primary, background)
            targets[label] = {
                "coordinates": {
                    "nz": len(primary.z), "nr": len(primary.r),
                    "z_min": float(primary.z[0]), "z_max": float(primary.z[-1]),
                    "r_min": float(primary.r[0]), "r_max": float(primary.r[-1]),
                },
                "archived_comparisons": {
                    "primary_vs_adverse_quintic": _comparison(primary, adverse),
                    "P11_primary_vs_P10_primary": _comparison(primary, p10),
                },
                "endpoint_identical_comparisons": endpoint_comparisons,
                "Q53_shared_endpoint_trace_check": endpoint_trace_comparison(
                    primary_parent, q53_parent, primary.z, primary.r,
                ),
                "quintic_projection_crosscheck": {
                    "scaled_Linf": normalized_error(
                        adverse.reduced_fields, quintic_nak.reduced_fields,
                    )["scaled_Linf"],
                    "explanation": (
                        "RectBivariateSpline adverse lane versus coefficient-"
                        "persistable TensorSplineSurface quintic, both not-a-knot"
                    ),
                },
                "source_comparisons": source_records,
                "constraints_by_representation": {
                    "primary_clamped_cubic": _constraint_record(
                        primary, background, mass_squared, reference,
                    ),
                    "cubic_not_a_knot": _constraint_record(
                        cubic_nak, background, mass_squared, reference,
                    ),
                    "quintic_not_a_knot": _constraint_record(
                        quintic_nak, background, mass_squared, reference,
                    ),
                    "endpoint_data_matched_Q53": _constraint_record(
                        q53, background, mass_squared, reference,
                    ),
                },
                "DJ_acceleration": wall,
                "wall_archive_crosscheck": _wall_archive_crosscheck(
                    phase_a, label, wall,
                ),
            }

        native_parent_wall = _wall_record(parent, background)

    return {
        "schema": "A790-phase-a2-failure-decomposition-v1",
        "scope": "archive-only; no parent solve, target repair, RHS, or RK call",
        "inputs": {
            "parent": str(PARENT),
            "parent_sha256": sha256_file(PARENT),
            "phase_a_result": str(PHASE_A),
            "phase_a_result_sha256": sha256_file(PHASE_A),
            "recovery_index": str(RECOVERY_INDEX),
            "recovery_index_sha256": sha256_file(RECOVERY_INDEX),
            "protocol": str(PROTOCOL),
            "protocol_sha256": sha256_file(PROTOCOL),
            "analysis_executable": str(Path(__file__).relative_to(ROOT)),
            "analysis_executable_sha256_before_output": sha256_file(Path(__file__)),
        },
        "field_order": list(FIELD_ORDER),
        "q5_identically_zero_in_all_saved_target_comparison_jets": q5_exact_zero,
        "comparison_conventions": {
            "error": "comparison-reference",
            "scaled_norm": "max(1, Linf(reference), Linf(comparison))",
            "fixed_compact_collars": (
                "lower and upper five percent of normalized compact interval"
            ),
            "degree_only_lane": (
                "same P11 nodal data and same not-a-knot endpoint prescription; "
                "only tensor-product spline degree changes from 3 to 5"
            ),
            "physical_components": (
                "product rules are propagated through first and second radial "
                "derivatives; q4 is not discarded at the axis"
            ),
        },
        "sources_included": bool(include_sources),
        "targets": targets,
        "native_P11_DJ_acceleration": native_parent_wall,
        "conclusions": {
            "representation": (
                "The sealed aggregate comparison conflates a q4 near-axis "
                "regular-coefficient defect with a distinct compact-endpoint "
                "physical-derivative defect."
            ),
            "wall_acceleration": (
                "Because v0 is zero, the second wall tangent equals DJ[q0].a0. "
                "The dominant scalar and spatial-metric residuals already exist "
                "on native P11 and are not target boundary-order artifacts."
            ),
            "physics": (
                "This artifact diagnoses initial-data representation and "
                "acceleration compatibility only; it does not test or authorize "
                "new interface physics."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-sources", action="store_true",
        help="omit the slower source/source-time/memory reconstruction lanes",
    )
    args = parser.parse_args()
    result = analyze(include_sources=not args.skip_sources)
    atomic_write_json(OUTPUT, _json_safe(result))
    print(json.dumps({
        "classification": "DIAGNOSTIC-only",
        "output": str(OUTPUT),
        "output_sha256": sha256_file(OUTPUT),
        "parent_sha256": result["inputs"]["parent_sha256"],
        "phase_a_result_sha256": result["inputs"]["phase_a_result_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
