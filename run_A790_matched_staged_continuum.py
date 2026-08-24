#!/usr/bin/env python3
"""Prospective Phase-A and bounded-G8 staged-continuum runner.

The full 18-lane matrix is intentionally absent.  This executable can only:

* build/cache and audit the P10/P11 parent plus G8/G9/G10 projections; or
* after a passing Phase A, capture t=0 and run one four-step G8 D1 segment in
  each boundary mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bhps.corrected_A790_R12_builder import (  # noqa: E402
    build_A790_R12_pair,
    build_A790_R12_refined,
)
from bhps.corrected_A790_physical_tensor_convergence import (  # noqa: E402
    generalized_order_nonuniform,
)
from bhps.finite_wall_high_order_solver import (  # noqa: E402
    solve_finite_wall_high_order_slice,
)
from bhps.gw_slice_high_order_solver import derivative_matrix  # noqa: E402
from bhps.matched_staged_continuum import (  # noqa: E402
    BOUNDARY_MODES,
    MANDATORY_LANDMARKS,
    PARENT_R_MAX,
    TARGET_GRIDS,
    TARGET_R_MAX,
    ContinuousPrimitiveParent,
    ContinuousReducedParent,
    ProjectedJetField,
    axis_even_crossfit_audit,
    balanced_constraint_audit,
    build_mode_neutral_case,
    extended_case_fingerprint,
    hash_arrays,
    initial_native_audit,
    lorentzian_position_signature,
    normalized_error,
    projected_geometry,
    projected_second_wall_audit,
    projection_fingerprint,
    quintic_adverse_projection,
    raw_hamiltonian_audit,
    reconstruct_driver_stage,
    representation_reconstruction,
    round_trip_to_parent,
    run_rk2_segment,
    second_wall_closure_audit,
    target_coordinates,
    trace_observational_check,
    validate_bundle_integrity,
)
from bhps.nonlinear_regular_so3_evolution import (  # noqa: E402
    _native_regular_axis_quotient_images,
    reduced_state_jets,
)
from bhps.recovery_indexer import (  # noqa: E402
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)


PROTOCOL = Path("notes/120_A790_matched_staged_continuum_protocol.md")
PROTOCOL_SHA256 = "baf9fa6e3612a79d21f2d17d575f9e16d49b8badc2f6400a0554eb93062401e1"
VALIDATION = Path(
    "results/corrected_A790_matched_staged_continuum_protocol_validation.json"
)
RECOVERY_ROOT = Path("results/corrected_A790_matched_staged_continuum_recovery")
MANIFEST = RECOVERY_ROOT / "index.json"
PARENT_PROJECTION = RECOVERY_ROOT / "phase_a_parent_projection.npz"
PHASE_A_RESULT = Path("results/corrected_A790_matched_staged_continuum_phase_a.json")
PHASE_B_RESULT = Path("results/corrected_A790_matched_staged_continuum_pilot.json")
DT = 0.000125
PILOT_STEPS = 4


STATIC_INPUTS = (
    Path(__file__),
    PROTOCOL,
    VALIDATION,
    Path("tests/test_matched_staged_continuum.py"),
    Path("tests/test_matched_staged_phase_b_audits.py"),
    Path("results/corrected_family_knot_A8_state.npz"),
)


def transitive_inputs():
    """Seal the local scientific source tree, not a hand-maintained subset."""
    local_modules = tuple(sorted(Path("src/bhps").rglob("*.py")))
    return tuple(dict.fromkeys((*STATIC_INPUTS, *local_modules)))


def public_environment():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "numpy_build_configuration": _json_safe(
            getattr(np.__config__, "CONFIG", {})
        ),
    }


def environment_fingerprint():
    payload = json.dumps(
        public_environment(), sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def stage_provenance_metadata():
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "environment_sha256": environment_fingerprint(),
    }


def validate_scientific_npz(path, required_shapes=None, embedded=None):
    """Validate mixed numeric/Unicode NPZ archives without skipping numerics."""
    record = validate_npz(
        path, required_shapes=required_shapes, require_finite=False,
    )
    with np.load(path, allow_pickle=False) as archive:
        unsupported = []
        nonfinite = []
        for key in archive.files:
            value = archive[key]
            if value.dtype.kind in "biufc":
                if not np.all(np.isfinite(value)):
                    nonfinite.append(key)
            elif value.dtype.kind not in "SU":
                unsupported.append((key, value.dtype.str))
        if unsupported:
            raise ValueError(f"unsupported NPZ dtypes: {unsupported}")
        if nonfinite:
            raise ValueError(f"nonfinite numeric NPZ arrays: {nonfinite}")
        for key, expected in (embedded or {}).items():
            if key not in archive.files or str(archive[key]) != str(expected):
                raise ValueError(f"embedded NPZ identity mismatch for {key}")
    return record


def expected_inputs():
    inputs = transitive_inputs()
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing prospective inputs: {missing}")
    return {str(path): sha256_file(path) for path in inputs}


def case_provenance():
    """Return the immutable protocol, code, and environment identity."""
    return {
        "protocol": {
            "path": str(PROTOCOL),
            "sha256": PROTOCOL_SHA256,
        },
        "code": expected_inputs(),
        "environment": public_environment(),
    }


def validate_authorization():
    protocol_hash = sha256_file(PROTOCOL)
    if protocol_hash != PROTOCOL_SHA256:
        raise RuntimeError(
            f"sealed protocol mismatch: {protocol_hash} != {PROTOCOL_SHA256}"
        )
    record = json.loads(VALIDATION.read_text())
    if record.get("protocol_sha256") != PROTOCOL_SHA256:
        raise RuntimeError("validation record does not name the sealed protocol")
    decisions = record.get("validations", {})
    for name in ("PARENT-REPRESENTATION", "OUTER-REFERENCE"):
        if decisions.get(name, {}).get("decision") != "accepted":
            raise RuntimeError(f"missing accepted validation: {name}")
    authorization = record.get("authorization", {})
    if authorization.get("full_18_run_matrix") != "not authorized":
        raise RuntimeError("validation record does not forbid the full matrix")
    return record


def recovery_index():
    return RecoveryIndex(
        MANIFEST, PROTOCOL, expected_inputs(), maximum_stage_seconds=21600.0,
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
    return repr(value)


def _background_json(background):
    return json.dumps(_json_safe(background), sort_keys=True, separators=(",", ":"))


def _build_p10_p11():
    print("matched continuum: building fresh P10/P11 R12 parent chain", flush=True)
    _, g8 = build_A790_R12_pair()
    g9 = build_A790_R12_refined(
        g8, 113, 253, "G9A790R12-matched-seed",
        selector_iterations=55, slice_iterations=360,
    )
    p10 = build_A790_R12_refined(
        g9, 129, 289, "P10A790R12-matched-comparator",
        selector_iterations=60, slice_iterations=380,
    )
    p11 = build_A790_R12_refined(
        p10, 145, 325, "P11A790R12-matched-parent",
        selector_iterations=65, slice_iterations=400,
    )
    return p10, p11


def _primitive_arrays(prefix, geometry):
    return {
        f"{prefix}_psi": np.asarray(geometry["psi"]),
        f"{prefix}_a": np.asarray(geometry["a"]),
        f"{prefix}_b": np.asarray(geometry["b"]),
        f"{prefix}_c": np.asarray(geometry["c"]),
        f"{prefix}_phi": np.asarray(geometry["phi"]),
    }


def _prefixed_arrays(prefix, arrays):
    return {f"{prefix}{name}": value for name, value in arrays.items()}


def _jet_arrays(prefix, jet):
    arrays = {
        f"{prefix}_z": np.asarray(jet.z),
        f"{prefix}_r": np.asarray(jet.r),
        f"{prefix}_q": np.asarray(jet.reduced_fields),
        f"{prefix}_first": np.asarray(jet.reduced_first),
        f"{prefix}_second": np.asarray(jet.reduced_second),
    }
    primitive_fields = getattr(jet, "primitive_fields", None)
    if primitive_fields is not None:
        for name, value in sorted(primitive_fields.items()):
            arrays[f"{prefix}_primitive_{name}"] = np.asarray(value)
    return arrays


def _jet_from_archive(archive, prefix):
    primitive = {
        key.removeprefix(f"{prefix}_primitive_"): np.asarray(archive[key])
        for key in archive.files if key.startswith(f"{prefix}_primitive_")
    }
    return ProjectedJetField(
        np.asarray(archive[f"{prefix}_z"]),
        np.asarray(archive[f"{prefix}_r"]),
        np.asarray(archive[f"{prefix}_q"]),
        np.asarray(archive[f"{prefix}_first"]),
        np.asarray(archive[f"{prefix}_second"]),
        primitive or None,
    )


def _parent_geometry_from_archive(archive):
    return {
        "name": str(archive["p11_name"]),
        "z": np.asarray(archive["p11_z"]),
        "r": np.asarray(archive["p11_r"]),
        "psi": np.asarray(archive["p11_psi"]),
        "a": np.asarray(archive["p11_a"]),
        "b": np.asarray(archive["p11_b"]),
        "c": np.asarray(archive["p11_c"]),
        "phi": np.asarray(archive["p11_phi"]),
        "background": json.loads(str(archive["background_json"])),
        "mass_squared": float(archive["mass_squared"]),
        "fold_amplitude": float(archive["fold_amplitude"]),
        "reference_maximum_residual": float(archive["p11_reference_residual"]),
        "selector_maximum": float(archive["p11_selector_residual"]),
        "continuous_parent_fingerprint": str(archive["primary_fingerprint"]),
        "primitive_parent_fingerprint": str(archive["primitive_fingerprint"]),
        "jet_field": _jet_from_archive(archive, "p11"),
    }


def _phase_a_parent_projection(index):
    stage_id = "phase_a/parent_projection"
    metadata = {
        **stage_provenance_metadata(),
        "parent": "P11", "parent_shape": [145, 325],
        "p10_shape": [129, 289], "target_grids": {
            label: [spec.nz, spec.nr_r10] for label, spec in TARGET_GRIDS.items()
        },
        "primary": "clamped tensor cubic in z,s=(r/12)^2",
        "adverse": "zero-smoothing quintic RectBivariateSpline",
    }
    index.register(stage_id, "parent-projection", 21600.0, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        validate_scientific_npz(cached, embedded={
            "protocol_sha256": PROTOCOL_SHA256,
            "schema": "matched-staged-parent-projection-v1",
        })
        return cached
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        p10, p11 = _build_p10_p11()
        primary = ContinuousReducedParent.from_jet_field(
            p11["jet_field"], p11["z"], p11["r"], degree=3,
            parent_identity=p11["name"], expected_shape=(145, 325),
            require_full_radial_domain=True,
        )
        primitive = ContinuousPrimitiveParent.from_geometry(
            p11, degree=3, expected_shape=(145, 325),
            require_full_radial_domain=True,
        )
        p10_representation = ContinuousReducedParent.from_jet_field(
            p10["jet_field"], p10["z"], p10["r"], degree=3,
            parent_identity=p10["name"], expected_shape=(129, 289),
            require_full_radial_domain=True,
        )
        arrays = {
            "schema": np.asarray("matched-staged-parent-projection-v1"),
            "protocol_sha256": np.asarray(PROTOCOL_SHA256),
            "p10_name": np.asarray(p10["name"]),
            "p11_name": np.asarray(p11["name"]),
            "p10_reference_residual": np.asarray(p10["reference_maximum_residual"]),
            "p10_selector_residual": np.asarray(p10["selector_maximum"]),
            "p11_reference_residual": np.asarray(p11["reference_maximum_residual"]),
            "p11_selector_residual": np.asarray(p11["selector_maximum"]),
            "mass_squared": np.asarray(p11["mass_squared"]),
            "fold_amplitude": np.asarray(p11["fold_amplitude"]),
            "background_json": np.asarray(_background_json(p11["background"])),
            "primary_fingerprint": np.asarray(primary.fingerprint()),
            "primitive_fingerprint": np.asarray(primitive.fingerprint()),
            "p10_representation_fingerprint": np.asarray(
                p10_representation.fingerprint()
            ),
            **_jet_arrays("p10", p10["jet_field"]),
            **_jet_arrays("p11", p11["jet_field"]),
            **_primitive_arrays("p10", p10),
            **_primitive_arrays("p11", p11),
            **primary.coefficient_arrays(),
            **primitive.coefficient_arrays(),
            **_prefixed_arrays(
                "p10_representation_", p10_representation.coefficient_arrays(),
            ),
        }
        for label, specification in TARGET_GRIDS.items():
            geometry = projected_geometry(
                p11, primary, specification, primitive,
            )
            projected = geometry["jet_field"]
            adverse = quintic_adverse_projection(
                p11["jet_field"], p11["z"], p11["r"],
                projected.z, projected.r,
            )
            p10_projection = p10_representation.project(
                projected.z, projected.r,
            )
            arrays.update(_jet_arrays(label.lower(), projected))
            arrays.update(_jet_arrays(f"{label.lower()}_quintic", adverse))
            arrays.update(_jet_arrays(f"{label.lower()}_p10", p10_projection))
            arrays[f"{label.lower()}_projection_fingerprint"] = np.asarray(
                projection_fingerprint(projected)
            )
        atomic_write_npz(PARENT_PROJECTION, **arrays)
        validate_scientific_npz(
            PARENT_PROJECTION,
            embedded={
                "protocol_sha256": PROTOCOL_SHA256,
                "schema": "matched-staged-parent-projection-v1",
            },
        )
        index.mark_complete(
            stage_id, PARENT_PROJECTION, time.perf_counter()-started,
            {"parent_fingerprint": hash_arrays(
                p11["z"], p11["r"],
                p11["jet_field"].reduced_fields,
                p11["jet_field"].reduced_first,
                p11["jet_field"].reduced_second,
            ), "continuous_parent_fingerprint": primary.fingerprint(),
             "primitive_parent_fingerprint": primitive.fingerprint()},
        )
        return PARENT_PROJECTION
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def _reference_path(label):
    return RECOVERY_ROOT / f"phase_a_reference_{label}.npz"


def _phase_a_reference(index, label, projected):
    path = _reference_path(label)
    stage_id = f"phase_a/reference/{label}"
    metadata = {
        **stage_provenance_metadata(),
        "grid": label, "shape": [len(projected.z), len(projected.r)],
        "projection_fingerprint": projection_fingerprint(projected),
        "purpose": "balanced reference q/Phi only; projected fields are never solved",
    }
    index.register(stage_id, "balanced-reference", 7200.0, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        validate_scientific_npz(cached, embedded={
            "protocol_sha256": PROTOCOL_SHA256,
            "grid_label": label,
            "projection_fingerprint": projection_fingerprint(projected),
        })
        return cached
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        reference = solve_finite_wall_high_order_slice(
            7.90, nz=len(projected.z), nr=len(projected.r),
            r_max=TARGET_R_MAX, wall_stiffness=20.0, epsilon=0.1,
            backreaction=0.01, tolerance=1e-10, iterations=400,
        )
        if not (
            np.array_equal(reference["z"], projected.z)
            and np.array_equal(reference["r"], projected.r)
        ):
            raise RuntimeError("balanced reference grid mismatch")
        atomic_write_npz(
            path, z=reference["z"], r=reference["r"], q=reference["q"],
            phi=reference["phi"], converged=np.asarray(reference["converged"]),
            max_abs_residual=np.asarray(reference["max_abs_residual"]),
            background_json=np.asarray(_background_json(reference["background"])),
            projected_fields_solved=np.asarray(False),
            protocol_sha256=np.asarray(PROTOCOL_SHA256),
            grid_label=np.asarray(label),
            projection_fingerprint=np.asarray(projection_fingerprint(projected)),
        )
        validate_scientific_npz(path, {
            "z": (len(projected.z),), "r": (len(projected.r),),
            "q": projected.reduced_fields.shape[:2],
            "phi": projected.reduced_fields.shape[:2],
        }, embedded={
            "protocol_sha256": PROTOCOL_SHA256,
            "grid_label": label,
            "projection_fingerprint": projection_fingerprint(projected),
        })
        index.mark_complete(stage_id, path, time.perf_counter()-started)
        return path
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def _geometry_from_projected(parent, label, projected):
    specification = TARGET_GRIDS[label]
    expected_z, expected_r = target_coordinates(parent["z"], specification)
    if not (
        np.array_equal(projected.z, expected_z)
        and np.array_equal(projected.r, expected_r)
    ):
        raise RuntimeError(f"{label} projected coordinates are not canonical")
    return {
        "name": f"{label}-A790-P11-direct-R10-projection",
        "z": projected.z, "r": projected.r,
        "source_grid": [len(parent["z"]), len(parent["r"])],
        "target_grid": [len(projected.z), len(projected.r)],
        "background": parent["background"],
        "mass_squared": parent["mass_squared"],
        "fold_amplitude": parent["fold_amplitude"],
        "jet_field": projected,
        "continuous_parent": parent["name"],
        "continuous_parent_fingerprint": parent[
            "continuous_parent_fingerprint"
        ],
        "primitive_parent_fingerprint": parent[
            "primitive_parent_fingerprint"
        ],
        "projection_fingerprint": projection_fingerprint(projected),
        "projection": {
            "parent": parent["name"],
            "grid_label": str(label),
            "coordinates": "z,s=(r/12)^2",
            "primary": "tensor_product_cubic_clamped_by_parent_endpoint_q_z",
            "target_to_target_interpolation": False,
            "constraint_resolve": False, "endpoint_repair": False,
        },
    }


def _case_path(label):
    return RECOVERY_ROOT / f"phase_a_case_{label}.npz"


def _parent_case_coefficient_arrays():
    prefixes = (
        "position_", "velocity_", "acceleration_", "parent_",
        "reduced_field_order", "primitive_",
    )
    with np.load(PARENT_PROJECTION, allow_pickle=False) as archive:
        return {
            f"P11_{key}": np.asarray(archive[key])
            for key in archive.files if key.startswith(prefixes)
        }


def _case_arrays(bundle):
    jet = bundle.geometry["jet_field"]
    z = np.asarray(jet.z)
    r = np.asarray(jet.r)
    return {
        **_jet_arrays("projected", jet),
        "source0": bundle.source0,
        "memory0": bundle.memory0,
        "source_time0": bundle.source_time0,
        "source_second_time0": bundle.source_second_time0,
        "taylor_source_value": bundle.taylor.source_value,
        "taylor_source_first": bundle.taylor.source_first,
        "taylor_fingerprint": np.asarray(bundle.taylor_fingerprint),
        "outer_reference_position": bundle.outer_reference_position,
        "outer_reference_acceleration": bundle.outer_reference_acceleration,
        "common_input_fingerprint": np.asarray(bundle.common_input_fingerprint),
        "background_json": np.asarray(_background_json(
            bundle.geometry["background"]
        )),
        "mass_squared": np.asarray(bundle.geometry["mass_squared"]),
        "fold_amplitude": np.asarray(bundle.geometry["fold_amplitude"]),
        "configuration_json": np.asarray(json.dumps(
            _json_safe(bundle.configuration), sort_keys=True,
        )),
        "protocol_sha256": np.asarray(PROTOCOL_SHA256),
        "environment_sha256": np.asarray(environment_fingerprint()),
        "stencil_z_first": np.asarray(derivative_matrix(z, 1).toarray()),
        "stencil_z_second": np.asarray(derivative_matrix(z, 2).toarray()),
        "stencil_r_first": np.asarray(derivative_matrix(r, 1).toarray()),
        "stencil_r_second": np.asarray(derivative_matrix(r, 2).toarray()),
    }


def _phase_a_case(index, label, geometry):
    path = _case_path(label)
    stage_id = f"phase_a/case/{label}"
    projected = geometry["jet_field"]
    metadata = {
        **stage_provenance_metadata(),
        "grid": label, "projection_fingerprint": projection_fingerprint(projected),
        "outer_reference": "direct P11 projection, mode neutral",
        "parent_artifact_sha256": sha256_file(PARENT_PROJECTION),
    }
    index.register(stage_id, "mode-neutral-case", 3600.0, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        validate_scientific_npz(cached, embedded={
            "protocol_sha256": PROTOCOL_SHA256,
            "environment_sha256": environment_fingerprint(),
        })
        return cached
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        bundle = build_mode_neutral_case(
            geometry, f"{label}-matched", provenance=case_provenance(),
        )
        arrays = {**_case_arrays(bundle), **_parent_case_coefficient_arrays()}
        atomic_write_npz(path, **arrays)
        validate_scientific_npz(path, {
            "source0": projected.reduced_fields.shape[:2]+(3,),
            "memory0": projected.reduced_fields.shape[:2]+(3,),
            "outer_reference_position": projected.reduced_fields.shape,
            "outer_reference_acceleration": projected.reduced_fields.shape,
        }, embedded={
            "protocol_sha256": PROTOCOL_SHA256,
            "environment_sha256": environment_fingerprint(),
        })
        index.mark_complete(
            stage_id, path, time.perf_counter()-started,
            {
                "common_input_fingerprint": bundle.common_input_fingerprint,
                "outer_position_sha256": hash_arrays(bundle.outer_reference_position),
                "outer_acceleration_sha256": hash_arrays(
                    bundle.outer_reference_acceleration
                ),
            },
        )
        return path
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def _case_matches_archive(bundle, path):
    with np.load(path, allow_pickle=False) as archive:
        comparisons = {
            key: np.array_equal(np.asarray(archive[key]), np.asarray(value))
            for key, value in (
                ("source0", bundle.source0), ("memory0", bundle.memory0),
                ("source_time0", bundle.source_time0),
                ("source_second_time0", bundle.source_second_time0),
                ("outer_reference_position", bundle.outer_reference_position),
                ("outer_reference_acceleration", bundle.outer_reference_acceleration),
                ("projected_z", bundle.geometry["jet_field"].z),
                ("projected_r", bundle.geometry["jet_field"].r),
                ("projected_q", bundle.geometry["jet_field"].reduced_fields),
                ("projected_first", bundle.geometry["jet_field"].reduced_first),
                ("projected_second", bundle.geometry["jet_field"].reduced_second),
                ("taylor_source_value", bundle.taylor.source_value),
                ("taylor_source_first", bundle.taylor.source_first),
            )
        }
        for name, value in (bundle.geometry["jet_field"].primitive_fields or {}).items():
            key = f"projected_primitive_{name}"
            comparisons[key] = bool(
                key in archive.files and np.array_equal(archive[key], value)
            )
        z = bundle.geometry["jet_field"].z
        r = bundle.geometry["jet_field"].r
        for key, value in (
            ("stencil_z_first", derivative_matrix(z, 1).toarray()),
            ("stencil_z_second", derivative_matrix(z, 2).toarray()),
            ("stencil_r_first", derivative_matrix(r, 1).toarray()),
            ("stencil_r_second", derivative_matrix(r, 2).toarray()),
        ):
            comparisons[key] = bool(np.array_equal(archive[key], value))
        comparisons["common_input_fingerprint"] = bool(
            str(archive["common_input_fingerprint"])
            == bundle.common_input_fingerprint
        )
        comparisons["taylor_fingerprint"] = bool(
            str(archive["taylor_fingerprint"]) == bundle.taylor_fingerprint
        )
        comparisons["protocol_sha256"] = bool(
            str(archive["protocol_sha256"]) == PROTOCOL_SHA256
        )
        comparisons["environment_sha256"] = bool(
            str(archive["environment_sha256"]) == environment_fingerprint()
        )
        comparisons["configuration"] = bool(
            json.loads(str(archive["configuration_json"]))
            == _json_safe(bundle.configuration)
        )
        comparisons["background"] = bool(
            json.loads(str(archive["background_json"]))
            == _json_safe(bundle.geometry["background"])
        )
        comparisons["mass_squared"] = bool(
            float(archive["mass_squared"]) == bundle.geometry["mass_squared"]
        )
        comparisons["fold_amplitude"] = bool(
            float(archive["fold_amplitude"]) == bundle.geometry["fold_amplitude"]
        )
    return {**comparisons, "all": bool(all(comparisons.values()))}


def _fixed_interior_derivative_error(projected, buffer_points=7):
    first, second = reduced_state_jets(
        projected.reduced_fields, projected.reduced_first[0],
        projected.z, projected.r, 7,
    )
    sl = (slice(None), slice(buffer_points, -buffer_points),
          slice(buffer_points, -buffer_points), slice(None))
    first_difference = first[1:][sl]-projected.reduced_first[1:][sl]
    second_difference = (
        second[1:, 1:][(slice(None), slice(None), *sl[1:])]
        - projected.reduced_second[1:, 1:][
            (slice(None), slice(None), *sl[1:])
        ]
    )
    first_scale = max(1.0, float(np.max(np.abs(projected.reduced_first[1:][sl]))))
    second_scale = max(1.0, float(np.max(np.abs(
        projected.reduced_second[1:, 1:][
            (slice(None), slice(None), *sl[1:])
        ]
    ))))
    return {
        "first_scaled_RMS": float(np.sqrt(np.mean(first_difference**2))/first_scale),
        "first_scaled_Linf": float(np.max(np.abs(first_difference))/first_scale),
        "second_scaled_RMS": float(np.sqrt(np.mean(second_difference**2))/second_scale),
        "second_scaled_Linf": float(np.max(np.abs(second_difference))/second_scale),
    }


def _sequence_convergence(values, intervals, minimum_order):
    """Apply the frozen three-grid monotonicity/order/floor decision."""
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        return {
            "values": values.tolist(), "differences": [None, None],
            "monotone_nonincreasing": False, "floor_resolved": False,
            "generalized_order": None, "passes": False,
        }
    differences = np.abs(np.diff(values))
    floor_resolved = bool(np.all(differences <= 1e-12))
    monotone = bool(values[0] >= values[1] >= values[2])
    order = generalized_order_nonuniform(
        differences[0], differences[1], intervals,
    )
    passes = bool(
        floor_resolved
        or (monotone and order is not None and order >= float(minimum_order))
    )
    return {
        "values": values.tolist(), "differences": differences.tolist(),
        "monotone_nonincreasing": monotone,
        "floor_resolved": floor_resolved,
        "generalized_order": order,
        "minimum_order": float(minimum_order),
        "passes": passes,
    }


def _finite_field_summary(jet, extra=None):
    arrays = {
        "z": jet.z, "r": jet.r, "q": jet.reduced_fields,
        "first": jet.reduced_first, "second": jet.reduced_second,
        **(jet.primitive_fields or {}), **(extra or {}),
    }
    per_field = {
        str(name): bool(np.all(np.isfinite(np.asarray(value))))
        for name, value in arrays.items()
    }
    return {"per_field": per_field, "all_finite": bool(all(per_field.values()))}


def _archive_numeric_finite(path):
    with np.load(path, allow_pickle=False) as archive:
        numeric = {
            key: bool(np.all(np.isfinite(archive[key])))
            for key in archive.files if archive[key].dtype.kind in "biufc"
        }
    return {
        "numeric_array_count": len(numeric), "per_array": numeric,
        "all_finite": bool(numeric and all(numeric.values())),
    }


def _normalized_rows_pass(record, row_names, rms_limit, linf_limit):
    return bool(
        record.get("finite", False)
        and all(
            record["walls"][wall][name]["normalized_RMS"] <= rms_limit
            and record["walls"][wall][name]["normalized_Linf"] <= linf_limit
            for wall in ("lower", "upper") for name in row_names
        )
    )


def _reference_record(path):
    with np.load(path, allow_pickle=False) as archive:
        return {
            "q": np.asarray(archive["q"]),
            "phi": np.asarray(archive["phi"]),
            "background": json.loads(str(archive["background_json"])),
            "converged": bool(archive["converged"]),
            "max_abs_residual": float(archive["max_abs_residual"]),
            "protocol_sha256": str(archive["protocol_sha256"]),
            "grid_label": str(archive["grid_label"]),
            "projection_fingerprint": str(archive["projection_fingerprint"]),
        }


def _projection_comparisons(projected, adverse, p10_projected):
    return {
        "primary_vs_quintic": {
            "position": normalized_error(
                projected.reduced_fields, adverse.reduced_fields,
            ),
            "first_spatial": normalized_error(
                projected.reduced_first[1:], adverse.reduced_first[1:],
            ),
            "second_spatial_and_acceleration": normalized_error(
                np.concatenate((
                    projected.reduced_second[1:, 1:].reshape(-1),
                    projected.reduced_second[0, 0].reshape(-1),
                )),
                np.concatenate((
                    adverse.reduced_second[1:, 1:].reshape(-1),
                    adverse.reduced_second[0, 0].reshape(-1),
                )),
            ),
        },
        "p10_vs_p11_primary": {
            "position": normalized_error(
                projected.reduced_fields, p10_projected.reduced_fields,
            ),
            "first_spatial": normalized_error(
                projected.reduced_first[1:], p10_projected.reduced_first[1:],
            ),
            "second_spatial_and_acceleration": normalized_error(
                np.concatenate((
                    projected.reduced_second[1:, 1:].reshape(-1),
                    projected.reduced_second[0, 0].reshape(-1),
                )),
                np.concatenate((
                    p10_projected.reduced_second[1:, 1:].reshape(-1),
                    p10_projected.reduced_second[0, 0].reshape(-1),
                )),
            ),
        },
    }


def _source_projection_comparisons(primary, adverse):
    return {
        "source": normalized_error(primary.source0, adverse.source0),
        "source_time": normalized_error(
            primary.source_time0, adverse.source_time0,
        ),
        "memory": normalized_error(primary.memory0, adverse.memory0),
    }


def _axis_parity_audit(projected):
    profiles = {
        "q_r": projected.reduced_first[2, :, 0],
        "q_zr": projected.reduced_second[1, 2, :, 0],
        "v_r": projected.reduced_second[0, 2, :, 0],
    }
    scale = max(
        1.0, float(np.max(np.abs(projected.reduced_fields))),
        float(np.max(np.abs(projected.reduced_first))),
        float(np.max(np.abs(projected.reduced_second))),
    )
    return {
        "profiles": {name: value.tolist() for name, value in profiles.items()},
        "scaled_Linf": float(max(
            np.max(np.abs(value)) for value in profiles.values()
        )/scale),
        "finite": bool(all(np.all(np.isfinite(value)) for value in profiles.values())),
    }


def _phase_a_audit(index, parent_path):
    stage_id = "phase_a/audit"
    metadata = {
        **stage_provenance_metadata(), "targets": list(TARGET_GRIDS),
        "parent_artifact_sha256": sha256_file(parent_path),
    }
    index.register(stage_id, "phase-a-audit", 7200.0, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        result = json.loads(cached.read_text())
        if (
            result.get("protocol_sha256") != PROTOCOL_SHA256
            or result.get("environment_sha256") != environment_fingerprint()
            or result.get("parent_artifact_sha256") != sha256_file(parent_path)
        ):
            raise RuntimeError("cached Phase-A audit identity mismatch")
        return result
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        projections = {}
        references = {}
        cases = {}
        records = {}
        with np.load(parent_path, allow_pickle=False) as archive:
            parent_array_finiteness = {
                key: bool(np.all(np.isfinite(archive[key])))
                for key in archive.files if archive[key].dtype.kind in "biufc"
            }
            parent_solver_values = {
                name: float(archive[name]) for name in (
                    "p10_reference_residual", "p10_selector_residual",
                    "p11_reference_residual", "p11_selector_residual",
                )
            }
            parent = _parent_geometry_from_archive(archive)
            primary = ContinuousReducedParent.from_arrays(archive)
            primitive = ContinuousPrimitiveParent.from_arrays(archive)
            if (
                primary.fingerprint() != str(archive["primary_fingerprint"])
                or primitive.fingerprint() != str(archive["primitive_fingerprint"])
            ):
                raise RuntimeError("parent representation fingerprint mismatch")
            parent_reprojected = primary.project(parent["z"], parent["r"])
            p11_r10 = primary.project(
                parent["z"], parent["r"][parent["r"] <= TARGET_R_MAX+1e-12],
            )
            reconstruction = representation_reconstruction(
                parent["jet_field"], primary,
                parent["z"], parent["r"],
            )
            midpoint_z = 0.5*(parent["z"][:-1]+parent["z"][1:])
            midpoint_r = 0.5*(parent["r"][:-1]+parent["r"][1:])

            def validation_record(z_values, r_values):
                base = primary.project(z_values, r_values)
                jet = ProjectedJetField(
                    base.z, base.r, base.reduced_fields,
                    base.reduced_first, base.reduced_second,
                    primitive.project(z_values, r_values),
                )
                return (
                    _finite_field_summary(jet),
                    lorentzian_position_signature(
                        jet.z, jet.r, jet.reduced_fields,
                    ),
                )

            validation_mesh = {}
            parent_signatures = {}
            validation_coordinates = {
                "P11_nodes": (parent["z"], parent["r"]),
                "P11_z_edge_midpoints": (midpoint_z, parent["r"]),
                "P11_r_edge_midpoints": (parent["z"], midpoint_r),
                "P11_cell_midpoints": (midpoint_z, midpoint_r),
            }
            for mesh_name, coordinates in validation_coordinates.items():
                finite_record, signature_record = validation_record(*coordinates)
                validation_mesh[mesh_name] = finite_record
                parent_signatures[mesh_name] = signature_record
            for label in TARGET_GRIDS:
                prefix = label.lower()
                projected = _jet_from_archive(archive, prefix)
                if projection_fingerprint(projected) != str(
                    archive[f"{prefix}_projection_fingerprint"]
                ):
                    raise RuntimeError(f"{label} projection fingerprint mismatch")
                adverse = _jet_from_archive(archive, f"{prefix}_quintic")
                p10_projected = _jet_from_archive(archive, f"{prefix}_p10")
                projections[label] = projected
                reference_path = _phase_a_reference(index, label, projected)
                references[label] = _reference_record(reference_path)
                geometry = _geometry_from_projected(parent, label, projected)
                case_path = _phase_a_case(index, label, geometry)
                bundle = build_mode_neutral_case(
                    geometry, f"{label}-matched",
                    provenance=case_provenance(),
                )
                adverse_geometry = _geometry_from_projected(
                    parent, label, adverse,
                )
                adverse_geometry["projection"]["primary"] = (
                    "adverse_zero_smoothing_quintic_not_evolved"
                )
                adverse_bundle = build_mode_neutral_case(
                    adverse_geometry, f"{label}-adverse-quintic",
                    provenance=case_provenance(),
                )
                native = initial_native_audit(bundle)
                cases[label] = bundle
                records[label] = {
                    "projection_fingerprint": projection_fingerprint(projected),
                    "validation_fields": _finite_field_summary(
                        projected, {
                            "source": bundle.source0,
                            "source_time": bundle.source_time0,
                            "source_second_time": bundle.source_second_time0,
                            "memory": bundle.memory0,
                            "outer_position": bundle.outer_reference_position,
                            "outer_acceleration": bundle.outer_reference_acceleration,
                        },
                    ),
                    "comparisons": _projection_comparisons(
                        projected, adverse, p10_projected,
                    ),
                    "source_comparisons": _source_projection_comparisons(
                        bundle, adverse_bundle,
                    ),
                    "axis_parity": _axis_parity_audit(projected),
                    "round_trip": round_trip_to_parent(projected, p11_r10),
                    "finite_difference_comparator": _fixed_interior_derivative_error(
                        projected
                    ),
                    "raw_hamiltonian_diagnostic": raw_hamiltonian_audit(
                        projected, parent["mass_squared"],
                    ),
                    "balanced_constraint": balanced_constraint_audit(
                        projected, parent["background"], references[label],
                    ),
                    "native": native,
                    "projected_second_wall": native["projected_second_wall"],
                    "case_archive_bitwise": _case_matches_archive(
                        bundle, case_path,
                    ),
                    "common_input_fingerprint": bundle.common_input_fingerprint,
                    "outer_reference_hashes": {
                        "position": hash_arrays(bundle.outer_reference_position),
                        "acceleration": hash_arrays(bundle.outer_reference_acceleration),
                    },
                    "reference": {
                        "converged": references[label]["converged"],
                        "maximum_residual": references[label]["max_abs_residual"],
                        "projected_fields_solved": False,
                    },
                }
            order_independence = {}
            for label in reversed(tuple(TARGET_GRIDS)):
                direct_base = primary.project(
                    projections[label].z, projections[label].r,
                )
                direct = ProjectedJetField(
                    direct_base.z, direct_base.r, direct_base.reduced_fields,
                    direct_base.reduced_first, direct_base.reduced_second,
                    primitive.project(direct_base.z, direct_base.r),
                )
                order_independence[label] = bool(
                    projection_fingerprint(direct)
                    == projection_fingerprint(projections[label])
                )
        labels = tuple(TARGET_GRIDS)
        intervals = [TARGET_GRIDS[label].nz-1 for label in labels]
        fd_first = [
            records[label]["finite_difference_comparator"]["first_scaled_RMS"]
            for label in labels
        ]
        fd_second = [
            records[label]["finite_difference_comparator"]["second_scaled_RMS"]
            for label in labels
        ]
        balanced = [
            records[label]["balanced_constraint"][
                "combined_retained_normalized_RMS"
            ] for label in labels
        ]
        wall = [
            records[label]["native"]["junction"]["combined_normalized_RMS"]
            for label in labels
        ]
        compact_wall_row_values = {}
        for wall_name in ("lower", "upper"):
            for row in ("metric", "Phi", "chi"):
                compact_wall_row_values[
                    f"junction/{wall_name}/{row}"
                ] = [
                    records[label]["native"]["junction"]["walls"][wall_name][
                        f"{row}_normalized_RMS"
                    ] for label in labels
                ]
            compact_wall_row_values[
                f"normal_gauge/{wall_name}"
            ] = [
                records[label]["native"]["normal_gauge_wall_profiles"][
                    "walls"
                ][wall_name]["normalized_RMS"] for label in labels
            ]
            for row in ("metric", "Phi", "chi"):
                compact_wall_row_values[
                    f"projected_second/{wall_name}/{row}"
                ] = [
                    records[label]["projected_second_wall"]["walls"][wall_name][
                        row
                    ]["normalized_RMS"] for label in labels
                ]
        sequence = {
            "finite_difference_first": _sequence_convergence(
                fd_first, intervals, 3.0,
            ),
            "finite_difference_second": _sequence_convergence(
                fd_second, intervals, 3.0,
            ),
            "balanced_constraint": _sequence_convergence(
                balanced, intervals, 1.5,
            ),
            "compact_wall": _sequence_convergence(
                wall, intervals, 1.5,
            ),
            "compact_wall_rows": {
                name: _sequence_convergence(values, intervals, 1.5)
                for name, values in compact_wall_row_values.items()
            },
        }

        def comparison_gate(section, quantity, limit):
            return bool(all(
                records[label]["comparisons"][section][quantity]["finite"]
                and records[label]["comparisons"][section][quantity][
                    "scaled_Linf"
                ] <= limit for label in labels
            ))

        def source_comparison_gate(limit):
            return bool(all(
                records[label]["source_comparisons"][name]["finite"]
                and records[label]["source_comparisons"][name][
                    "scaled_Linf"
                ] <= limit
                for label in labels
                for name in ("source", "source_time", "memory")
            ))

        def initial_wall_rows_pass(label):
            native = records[label]["native"]
            junction = native["junction"]
            normal = native["normal_gauge_wall_profiles"]
            return bool(
                junction["finite"] and normal["finite"]
                and all(
                    junction["walls"][wall_name][f"{row}_normalized_RMS"]
                    <= 1e-5
                    and junction["walls"][wall_name][f"{row}_normalized_Linf"]
                    <= 1e-4
                    for wall_name in ("lower", "upper")
                    for row in ("metric", "Phi", "chi")
                )
                and all(
                    normal["walls"][wall_name]["normalized_RMS"] <= 1e-5
                    and normal["walls"][wall_name]["normalized_Linf"] <= 1e-4
                    for wall_name in ("lower", "upper")
                )
            )

        def analytic_wall_rows_pass(label):
            native = records[label]["native"]
            endpoint = native["analytic_endpoint_wall"]
            normal = native["analytic_normal_gauge_wall_profiles"]
            return bool(
                endpoint["finite"] and normal["finite"]
                and endpoint["position_normalized_RMS"] <= 1e-5
                and endpoint["position_normalized_Linf"] <= 1e-4
                and endpoint["tangent_normalized_RMS"] <= 1e-5
                and endpoint["tangent_normalized_Linf"] <= 1e-4
                and all(
                    normal["walls"][wall_name]["normalized_RMS"] <= 1e-5
                    and normal["walls"][wall_name]["normalized_Linf"] <= 1e-4
                    for wall_name in ("lower", "upper")
                )
            )

        parent_gates = {
            "P10_reference_below_1e_9": bool(
                parent_solver_values["p10_reference_residual"] < 1e-9
            ),
            "P10_selector_below_1e_9": bool(
                parent_solver_values["p10_selector_residual"] < 1e-9
            ),
            "P11_reference_below_1e_9": bool(
                parent_solver_values["p11_reference_residual"] < 1e-9
            ),
            "P11_selector_below_1e_9": bool(
                parent_solver_values["p11_selector_residual"] < 1e-9
            ),
            "every_parent_numeric_array_finite": bool(
                parent_array_finiteness and all(parent_array_finiteness.values())
            ),
            "parent_node_and_midpoint_fields_finite": bool(all(
                record["all_finite"] for record in validation_mesh.values()
            )),
            "parent_node_and_midpoint_signatures_lorentzian": bool(all(
                record["finite"]
                and record["all_points_one_negative_direction"]
                and record["minimum_eigenvalue_margin"] >= 1e-8
                for record in parent_signatures.values()
            )),
            "parent_representation_reconstructs_nodes_below_1e_12": bool(all(
                reconstruction[name]["finite"]
                and reconstruction[name]["scaled_Linf"] <= 1e-12
                for name in (
                    "position", "velocity", "acceleration",
                    "endpoint_z_first_stored_comparator",
                    "endpoint_velocity_z_first_stored_comparator",
                )
            )),
            "stored_spatial_jet_comparators_finite": bool(all(
                reconstruction[name]["finite"] for name in (
                    "first_spatial_stored_comparator",
                    "second_spatial_stored_comparator",
                )
            )),
        }
        representation_gates = {
            "cubic_quintic_position_Einf_at_most_1e_8": comparison_gate(
                "primary_vs_quintic", "position", 1e-8,
            ),
            "cubic_quintic_first_spatial_Einf_at_most_1e_7": comparison_gate(
                "primary_vs_quintic", "first_spatial", 1e-7,
            ),
            "cubic_quintic_second_and_a_Einf_at_most_1e_5": comparison_gate(
                "primary_vs_quintic", "second_spatial_and_acceleration", 1e-5,
            ),
            "cubic_quintic_source_lanes_Einf_at_most_1e_7": (
                source_comparison_gate(1e-7)
            ),
            "P10_P11_position_Einf_at_most_1e_4": comparison_gate(
                "p10_vs_p11_primary", "position", 1e-4,
            ),
            "P10_P11_first_spatial_Einf_at_most_5e_4": comparison_gate(
                "p10_vs_p11_primary", "first_spatial", 5e-4,
            ),
            "P10_P11_second_and_a_Einf_at_most_2e_3": comparison_gate(
                "p10_vs_p11_primary", "second_spatial_and_acceleration", 2e-3,
            ),
            "analytic_axis_parity_at_most_1e_10": bool(all(
                records[label]["axis_parity"]["finite"]
                and records[label]["axis_parity"]["scaled_Linf"] <= 1e-10
                for label in labels
            )),
            "target_build_order_bitwise_independent": bool(all(
                order_independence.values()
            )),
        }
        projection_gates = {
            "all_target_reconstructed_fields_finite": bool(all(
                records[label]["validation_fields"]["all_finite"]
                for label in labels
            )),
            "balanced_RMS_all_at_most_1e_6": bool(all(
                value <= 1e-6 for value in balanced
            )),
            "balanced_Linf_all_at_most_1e_5": bool(all(
                records[label]["balanced_constraint"][
                    "combined_retained_normalized_Linf"
                ] <= 1e-5 for label in labels
            )),
            "initial_metric_Phi_chi_normal_rows_within_limits": bool(all(
                initial_wall_rows_pass(label) for label in labels
            )),
            "analytic_endpoint_and_normal_rows_within_limits": bool(all(
                analytic_wall_rows_pass(label) for label in labels
            )),
            "projected_second_metric_Phi_chi_rows_within_limits": bool(all(
                _normalized_rows_pass(
                    records[label]["projected_second_wall"],
                    ("metric", "Phi", "chi"), 1e-5, 1e-4,
                ) for label in labels
            )),
            "all_target_signatures_lorentzian_with_margin": bool(all(
                records[label]["native"]["signature"]["finite"]
                and records[label]["native"]["signature"][
                    "all_points_one_negative_direction"
                ]
                and records[label]["native"]["signature"][
                    "minimum_eigenvalue_margin"
                ] >= 1e-8 for label in labels
            )),
            "all_initial_residual_records_finite": bool(all(
                records[label]["balanced_constraint"]["finite"]
                and records[label]["raw_hamiltonian_diagnostic"]["finite"]
                and records[label]["native"]["junction"]["finite"]
                and records[label]["native"]["projected_second_wall"]["finite"]
                and records[label]["native"]["normal_gauge_wall_profiles"]["finite"]
                for label in labels
            )),
            "balanced_references_converged_below_1e_9": bool(all(
                records[label]["reference"]["converged"]
                and records[label]["reference"]["maximum_residual"] < 1e-9
                and not records[label]["reference"]["projected_fields_solved"]
                for label in labels
            )),
            "all_case_archives_bitwise": bool(all(
                records[label]["case_archive_bitwise"]["all"] for label in labels
            )),
        }
        sequence_gates = {
            "finite_difference_first_monotone_order_at_least_3": bool(
                sequence["finite_difference_first"]["monotone_nonincreasing"]
                and sequence["finite_difference_first"]["generalized_order"]
                is not None
                and sequence["finite_difference_first"]["generalized_order"]
                >= 3.0
                and fd_first[-1] <= 1e-5
            ),
            "finite_difference_second_monotone_order_at_least_3": bool(
                sequence["finite_difference_second"]["monotone_nonincreasing"]
                and sequence["finite_difference_second"]["generalized_order"]
                is not None
                and sequence["finite_difference_second"]["generalized_order"]
                >= 3.0
                and fd_second[-1] <= 1e-5
            ),
            "balanced_monotone_order1p5_or_floor": bool(
                sequence["balanced_constraint"]["passes"]
            ),
            "compact_wall_monotone_order1p5_or_floor": bool(
                sequence["compact_wall"]["passes"]
            ),
            "every_compact_wall_row_monotone_order1p5_or_floor": bool(all(
                record["passes"]
                for record in sequence["compact_wall_rows"].values()
            )),
        }
        gates = {
            **parent_gates, **representation_gates,
            **projection_gates, **sequence_gates,
        }
        result = {
            "schema": "A790-matched-staged-continuum-phase-a-v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "environment_sha256": environment_fingerprint(),
            "parent_artifact_sha256": sha256_file(parent_path),
            "classification": (
                "PASS-phase-a" if all(gates.values()) else "FAIL-audit-phase-a"
            ),
            "full_matrix_authorized": False,
            "parent_solver_values": parent_solver_values,
            "parent_array_finiteness": {
                "numeric_array_count": len(parent_array_finiteness),
                "all_finite": bool(all(parent_array_finiteness.values())),
                "per_array": parent_array_finiteness,
            },
            "parent_reconstruction": reconstruction,
            "parent_signatures": parent_signatures,
            "validation_mesh": validation_mesh,
            "target_build_order_independence": order_independence,
            "targets": records,
            "sequences": sequence,
            "gates": {**gates, "all_phase_a_gates_pass": bool(all(gates.values()))},
            "artifacts": {
                "parent_projection": str(parent_path),
                "references": {label: str(_reference_path(label)) for label in TARGET_GRIDS},
                "cases": {label: str(_case_path(label)) for label in TARGET_GRIDS},
            },
            "environment": public_environment(),
        }
        atomic_write_json(PHASE_A_RESULT, result)
        index.mark_complete(stage_id, PHASE_A_RESULT, time.perf_counter()-started)
        return result
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def _nested_finite(value):
    if isinstance(value, dict):
        return all(_nested_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_nested_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(
            value.dtype.kind not in "biufc" or np.all(np.isfinite(value))
        )
    if isinstance(value, (float, int, np.floating, np.integer)):
        return bool(np.isfinite(value))
    return True


def _expected_stage_signature(mode):
    if mode == "wall_owner_last_experimental":
        return [(name, None) for name in MANDATORY_LANDMARKS[mode]]
    signature = [
        ("bulk_positive_radius", None), ("initial_axis_fill", None),
    ]
    for iteration in range(4):
        signature.extend((
            ("normal_iteration_wall_endpoint_solve", iteration),
            ("normal_iteration_post_wall_axis_fill", iteration),
            ("normal_iteration_gzz_solve", iteration),
        ))
    signature.extend((
        ("final_compact_wall_endpoint_solve", None),
        ("final_compact_post_wall_axis_fill", None),
        ("pre_outer", None), ("post_outer", None),
        ("post_axis_operator_repair", None),
    ))
    return signature


def _physical_acceleration_components(acceleration, r):
    value = np.asarray(acceleration, dtype=float)
    radius = np.asarray(r, dtype=float)[None, :]
    return np.stack((
        value[:, :, 0], value[:, :, 2], value[:, :, 6],
        value[:, :, 3]+radius**2*value[:, :, 4], value[:, :, 3],
        radius*value[:, :, 1], radius*value[:, :, 5],
        value[:, :, 7], value[:, :, 8],
    ), axis=-1)


def _technical_stage_audit(bundle, mode, record):
    """Evaluate every applicable frozen per-stage pilot gate."""
    validate_bundle_integrity(bundle, mode)
    q_shape = bundle.geometry["jet_field"].reduced_fields.shape
    source_shape = q_shape[:2]+(3,)
    stages = record["boundary_stages"]
    actual_signature = [
        (str(stage["name"]), stage.get("iteration")) for stage in stages
    ]
    expected_signature = _expected_stage_signature(mode)
    shape_checks = {
        name: tuple(np.asarray(record[name]).shape) == expected
        for name, expected in (
            ("q", q_shape), ("v", q_shape), ("acceleration", q_shape),
            ("source", source_shape), ("memory", source_shape),
            ("source_time", source_shape), ("memory_time", source_shape),
            ("target", source_shape), ("advection", source_shape),
            ("source_second_time", source_shape),
        )
    }
    shape_checks["all_stage_accelerations"] = bool(all(
        np.asarray(stage["acceleration"]).shape == q_shape for stage in stages
    ))
    signature = lorentzian_position_signature(
        bundle.geometry["z"], bundle.geometry["r"], record["q"],
    )
    preservation = record["staged_preservation"]
    causal_defects = [
        jump["walls"][wall]["causal_identity_maximum_absolute_defect"]
        for jump in preservation["jumps"] for wall in ("lower", "upper")
    ]
    hessian_changes = [
        jump["walls"][wall]["velocity_hessian_change_maximum_absolute"]
        for jump in preservation["jumps"] for wall in ("lower", "upper")
    ]
    causal_maximum = max(causal_defects, default=0.0)
    hessian_maximum = max(hessian_changes, default=0.0)
    diagnostic = record["diagnostic"]
    outer = diagnostic.get("outer_sommerfeld") or {}
    outer_source = record.get("outer_source_diagnostic") or {}
    correction_values = [
        float(item["relative_norm"])
        for item in diagnostic.get("wall_corrections", [])
    ]
    correction_values.extend(float(outer[name]) for name in (
        "metric_relative_correction", "scalar_relative_correction",
    ) if name in outer)
    if "relative_correction" in outer_source:
        correction_values.append(float(outer_source["relative_correction"]))
    normal = diagnostic.get("normal_wall_gauge") or {}
    if mode == "wall_owner_last_experimental":
        coupled = normal.get("coupled_block") or {}
        if "relative_correction" in coupled:
            correction_values.append(float(coupled["relative_correction"]))
    else:
        coupled = {}
        correction_values.extend(
            float(item["relative_correction"])
            for item in normal.get("iterations", [])
            if "relative_correction" in item
        )
    maximum_correction = max(correction_values, default=float("inf"))

    closure_names = ["final_compact_wall_endpoint_solve"]
    if mode == "wall_owner_last_experimental":
        closure_names.append("post_wall_owner_reconciliation")
    closures = {
        name: second_wall_closure_audit(
            record["q"], record["v"], record["landmarks"][name],
            bundle.geometry["z"], bundle.geometry["r"],
            bundle.geometry["background"], 0,
        ) for name in closure_names
    }
    closure_gate = bool(all(
        item["finite"] and item["combined_normalized_Linf"] < 1e-10
        for item in closures.values()
    ))

    owner = mode == "wall_owner_last_experimental"
    final_axis_stage = record["landmarks"]["post_axis_operator_repair"]
    pre_axis_stage = record["landmarks"][
        "post_wall_owner_reconciliation" if owner else "post_outer"
    ]
    final_stage_metadata = next(
        stage for stage in stages
        if stage["name"] == "post_axis_operator_repair"
    )
    pointwise_fields = (0, 2, 3, 6, 7, 8)
    quotient_fields = (1, 4, 5)
    native_quotient_target = _native_regular_axis_quotient_images(
        final_axis_stage, bundle.geometry["r"],
    )
    repair_physical_before = _physical_acceleration_components(
        pre_axis_stage, bundle.geometry["r"],
    )
    repair_physical_after = _physical_acceleration_components(
        final_axis_stage, bundle.geometry["r"],
    )
    axis_operator_repair = {
        "positive_radius_bitwise_unchanged": bool(np.array_equal(
            pre_axis_stage[:, 1:], final_axis_stage[:, 1:],
        )),
        "compact_wall_pointwise_bitwise_unchanged": bool(np.array_equal(
            pre_axis_stage[[0, -1], 0][:, pointwise_fields],
            final_axis_stage[[0, -1], 0][:, pointwise_fields],
        )),
        "physical_wall_bitwise_unchanged": bool(np.array_equal(
            repair_physical_before[[0, -1]],
            repair_physical_after[[0, -1]],
        )),
        "native_quotient_images_bitwise_exact": bool(np.array_equal(
            final_axis_stage[:, 0][:, quotient_fields],
            native_quotient_target,
        )),
        "returned_acceleration_bitwise_exact": bool(np.array_equal(
            final_axis_stage, record["acceleration"],
        )),
        "declared_direct_open_z_channels_exact": bool(
            tuple(final_stage_metadata.get("direct_open_z_channels", ()))
            == pointwise_fields
        ),
        "declared_native_quotient_channels_exact": bool(
            tuple(final_stage_metadata.get("native_quotient_channels", ()))
            == quotient_fields
        ),
    }
    axis_operator_repair["passes"] = bool(all(
        axis_operator_repair.values()
    ))
    open_face = {"applicable": owner, "bitwise_unchanged": True}
    reconciliation = {"applicable": owner, "passes": True}
    if owner:
        before_open = record["landmarks"]["outer_open_face_before_wall"]
        returned = record["landmarks"]["post_wall_owner_reconciliation"]
        open_face["bitwise_unchanged"] = bool(
            np.array_equal(
                before_open[1:-1, -1],
                record["acceleration"][1:-1, -1],
            )
        )
        before = record["landmarks"]["final_compact_wall_endpoint_solve"]
        after = returned
        positive_radius_unchanged = bool(np.array_equal(
            before[:, 1:], after[:, 1:],
        ))
        other_fields = tuple(
            index for index in range(q_shape[-1])
            if index not in quotient_fields
        )
        other_fields_unchanged = bool(np.array_equal(
            before[:, :, other_fields], after[:, :, other_fields],
        ))
        physical_before = _physical_acceleration_components(
            before, bundle.geometry["r"],
        )
        physical_after = _physical_acceleration_components(
            after, bundle.geometry["r"],
        )
        physical_change = normalized_error(
            physical_before, physical_after,
        )["scaled_Linf"]
        axis_fit = diagnostic["axis_fit_preference_defect"]
        reconciliation_target = _native_regular_axis_quotient_images(
            after, bundle.geometry["r"],
        )
        native_reconciliation_exact = bool(np.array_equal(
            after[:, 0][:, quotient_fields], reconciliation_target,
        ))
        reconciliation = {
            "applicable": True,
            "positive_radius_bitwise_unchanged": positive_radius_unchanged,
            "non_q4_q5_fields_bitwise_unchanged": other_fields_unchanged,
            "physical_wall_tensor_scaled_change": physical_change,
            "native_quotient_images_bitwise_exact": native_reconciliation_exact,
            "all_field_even_fit_defect_observational": float(
                axis_fit["relative"]
            ),
            "all_field_even_fit_by_field_observational": axis_fit["by_field"],
            "passes": bool(
                positive_radius_unchanged and other_fields_unchanged
                and physical_change <= 1e-12
                and native_reconciliation_exact
            ),
        }

    crossfit = axis_even_crossfit_audit(
        record["q"], record["acceleration"], bundle.geometry["r"],
    )
    coupled_gate = bool(
        not owner or (
            coupled.get("passed", False)
            and coupled.get("minimum_rank") == 4
            and coupled.get("maximum_condition", float("inf")) <= 1e12
            and coupled.get("minimum_pivot_strength", 0.0) >= 1e-10
            and coupled.get(
                "maximum_normalized_linear_residual", float("inf")
            ) < 1e-12
        )
    )
    normal_gate = bool(
        not owner or (
            normal.get("passed", False)
            and normal.get("radial_buffer") == 0
            and normal.get("final_residual", {}).get(
                "maximum", float("inf")
            ) < 1e-10
        )
    )
    gates = {
        "all_values_and_diagnostics_finite": bool(
            record["finite"] and _nested_finite(record)
        ),
        "lorentzian_signature_with_1e_8_margin": bool(
            signature["finite"]
            and signature["all_points_one_negative_direction"]
            and signature["minimum_eigenvalue_margin"] >= 1e-8
        ),
        "mandatory_stage_order_and_full_shapes": bool(
            actual_signature == expected_signature and all(shape_checks.values())
        ),
        "staged_causal_identity_at_most_1e_12": bool(
            preservation["finite"] and causal_maximum <= 1e-12
        ),
        "velocity_hessian_stage_change_bitwise_zero": bool(
            hessian_maximum == 0.0
        ),
        "outer_acceleration_residual_below_1e_10": bool(
            outer.get(
                "maximum_normalized_acceleration_residual", float("inf")
            ) < 1e-10
        ),
        "outer_source_residual_below_1e_10": bool(
            outer_source.get("maximum_normalized", float("inf")) < 1e-10
        ),
        "native_boundary_closure_valid": bool(
            outer.get(
                "maximum_normalized_acceleration_residual", float("inf")
            ) < 1e-10
            and outer_source.get("maximum_normalized", float("inf")) < 1e-10
            and closure_gate and coupled_gate and normal_gate
        ),
        "owner_open_face_bitwise_unchanged": bool(
            open_face["bitwise_unchanged"]
        ),
        "owner_coupled_block_valid": coupled_gate,
        "owner_full_normal_GH_below_1e_10": normal_gate,
        "required_raw_second_wall_rows_below_1e_10": closure_gate,
        "owner_reconciliation_valid": bool(reconciliation["passes"]),
        "native_axis_operator_valid": bool(axis_operator_repair["passes"]),
    }
    return {
        "mode": mode, "time": float(record["time"]),
        "actual_stage_signature": actual_signature,
        "expected_stage_signature": expected_signature,
        "shape_checks": shape_checks, "signature": signature,
        "causal_identity_maximum_absolute_defect": causal_maximum,
        "velocity_hessian_change_maximum_absolute": hessian_maximum,
        "maximum_relative_boundary_correction": maximum_correction,
        "open_face": open_face, "coupled_block": coupled,
        "normal_wall_gauge": normal, "second_wall_closures": closures,
        "reconciliation": reconciliation,
        "axis_operator_repair": axis_operator_repair,
        # Compatibility alias for frozen zero-step reporters.  It is retained
        # as observational metadata only; the native operator gate above is
        # the qualification authority.
        "axis_crossfit": crossfit,
        "axis_crossfit_observational_only": crossfit,
        "gates": {**gates, "all_technical_gates_pass": bool(all(gates.values()))},
    }


def _stage_metadata(record):
    diagnostic = {
        key: value for key, value in record["diagnostic"].items()
        if key != "boundary_stages"
    }
    stages = []
    for stage in record["boundary_stages"]:
        stages.append({
            key: _json_safe(value) for key, value in stage.items()
            if key != "acceleration"
        })
    return {
        "step": record.get("step"), "rk_stage": record.get("rk_stage"),
        "time": record["time"], "mode": record["mode"],
        "diagnostic": _json_safe(diagnostic),
        "outer_source_diagnostic": _json_safe(record["outer_source_diagnostic"]),
        "staged_preservation": _json_safe(record["staged_preservation"]),
        "technical_audit": _json_safe(record.get("technical_audit")),
        "finite": bool(record["finite"]),
        "ordered_stages": stages,
    }


def _record_arrays(prefix, record):
    arrays = {
        f"{prefix}_q": record["q"], f"{prefix}_v": record["v"],
        f"{prefix}_source": record["source"],
        f"{prefix}_memory": record["memory"],
        f"{prefix}_source_time": record["source_time"],
        f"{prefix}_memory_time": record["memory_time"],
        f"{prefix}_target": record["target"],
        f"{prefix}_advection": record["advection"],
        f"{prefix}_source_second_time": record["source_second_time"],
        f"{prefix}_acceleration": record["acceleration"],
        f"{prefix}_all_stage_accelerations": np.stack([
            stage["acceleration"] for stage in record["boundary_stages"]
        ]),
        f"{prefix}_metadata_json": np.asarray(json.dumps(
            _stage_metadata(record), sort_keys=True,
        )),
    }
    for name, acceleration in record["landmarks"].items():
        arrays[f"{prefix}_landmark_{name}"] = acceleration
    return arrays


def _archive_stage_audit(path):
    with np.load(path, allow_pickle=False) as archive:
        keys = sorted(
            key for key in archive.files if key.endswith("_metadata_json")
        )
        metadata = [json.loads(str(archive[key])) for key in keys]
    technical = [record.get("technical_audit") for record in metadata]
    all_present = bool(
        metadata and all(isinstance(record, dict) for record in technical)
    )
    all_pass = bool(
        all_present and all(
            record.get("gates", {}).get("all_technical_gates_pass", False)
            for record in technical
        )
    )
    return {
        "record_count": len(metadata),
        "metadata_keys": keys,
        "all_staged_preservation_present": bool(
            metadata and all(record.get("staged_preservation") for record in metadata)
        ),
        "all_technical_audits_present": all_present,
        "all_technical_gates_pass": all_pass,
        "metadata": metadata,
    }


def _t0_archive_audit(path, bundle, mode):
    validate_scientific_npz(path, {
        "t0_q": bundle.geometry["jet_field"].reduced_fields.shape,
        "t0_v": bundle.geometry["jet_field"].reduced_fields.shape,
        "t0_acceleration": bundle.geometry["jet_field"].reduced_fields.shape,
    }, embedded={
        "mode": mode, "protocol_sha256": PROTOCOL_SHA256,
        "environment_sha256": environment_fingerprint(),
        "common_input_fingerprint": bundle.common_input_fingerprint,
        "extended_fingerprint": extended_case_fingerprint(bundle, mode, DT),
    })
    stages = _archive_stage_audit(path)
    with np.load(path, allow_pickle=False) as archive:
        trace = json.loads(str(archive["trace_observational_json"]))
        outer_reference_equal = bool(
            np.array_equal(
                archive["outer_reference_position"],
                bundle.outer_reference_position,
            )
            and np.array_equal(
                archive["outer_reference_acceleration"],
                bundle.outer_reference_acceleration,
            )
        )
    trace_pass = bool(
        trace.get("acceleration_bitwise_equal")
        and trace.get("source_time_bitwise_equal")
        and trace.get("captured_arrays_are_independent_copies")
        and trace.get("finite")
    )
    return {
        **stages, "trace": trace, "trace_pass": trace_pass,
        "outer_reference_bitwise": outer_reference_equal,
        "passes": bool(
            stages["all_technical_gates_pass"]
            and trace_pass and outer_reference_equal
        ),
    }


def _segment_archive_audit(path, bundle, mode):
    shape = bundle.geometry["jet_field"].reduced_fields.shape
    validate_scientific_npz(path, {
        "end_q": shape, "end_v": shape,
        "end_source": shape[:2]+(3,), "end_memory": shape[:2]+(3,),
    }, embedded={
        "mode": mode, "protocol_sha256": PROTOCOL_SHA256,
        "environment_sha256": environment_fingerprint(),
        "common_input_fingerprint": bundle.common_input_fingerprint,
        "extended_fingerprint": extended_case_fingerprint(bundle, mode, DT),
    })
    stages = _archive_stage_audit(path)
    with np.load(path, allow_pickle=False) as archive:
        completed = bool(archive["completed"])
        technical_pass = bool(archive["technical_pass"])
        endpoint_fingerprint = str(archive["endpoint_fingerprint"])
        recomputed = hash_arrays(
            archive["end_q"], archive["end_v"],
            archive["end_source"], archive["end_memory"],
        )
        outer_reference_equal = bool(
            np.array_equal(
                archive["outer_reference_position"],
                bundle.outer_reference_position,
            )
            and np.array_equal(
                archive["outer_reference_acceleration"],
                bundle.outer_reference_acceleration,
            )
        )
    record_count_valid = bool((not completed) or stages["record_count"] == 8)
    passes = bool(
        completed and technical_pass and record_count_valid
        and stages["all_technical_gates_pass"]
        and endpoint_fingerprint == recomputed and outer_reference_equal
    )
    return {
        **stages, "completed": completed,
        "technical_pass": technical_pass,
        "record_count_valid": record_count_valid,
        "endpoint_fingerprint_valid": endpoint_fingerprint == recomputed,
        "outer_reference_bitwise": outer_reference_equal,
        "passes": passes,
    }


def _t0_path(mode):
    short = "legacy" if mode.startswith("legacy") else "owner"
    return RECOVERY_ROOT / f"phase_b_G8_t0_{short}.npz"


def _phase_b_t0(index, bundle, mode):
    path = _t0_path(mode)
    stage_id = f"phase_b/G8/t0/{mode}"
    metadata = {
        **stage_provenance_metadata(),
        "mode": mode, "common_input_fingerprint": bundle.common_input_fingerprint,
        "extended_fingerprint": extended_case_fingerprint(bundle, mode, DT),
    }
    index.register(stage_id, "t0-stage-capture", 3600.0, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        _t0_archive_audit(cached, bundle, mode)
        return cached
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        observational, record = trace_observational_check(bundle, mode)
        record["technical_audit"] = _technical_stage_audit(
            bundle, mode, record,
        )
        arrays = {
            **_record_arrays("t0", record),
            "trace_observational_json": np.asarray(json.dumps(
                observational, sort_keys=True,
            )),
            "outer_reference_position": bundle.outer_reference_position,
            "outer_reference_acceleration": bundle.outer_reference_acceleration,
            "common_input_fingerprint": np.asarray(bundle.common_input_fingerprint),
            "extended_fingerprint": np.asarray(
                extended_case_fingerprint(bundle, mode, DT)
            ),
            "mode": np.asarray(mode),
            "protocol_sha256": np.asarray(PROTOCOL_SHA256),
            "environment_sha256": np.asarray(environment_fingerprint()),
        }
        atomic_write_npz(path, **arrays)
        audit = _t0_archive_audit(path, bundle, mode)
        index.mark_complete(
            stage_id, path, time.perf_counter()-started,
            {
                "trace_acceleration_bitwise": observational[
                    "acceleration_bitwise_equal"
                ],
                "all_technical_gates_pass": audit[
                    "all_technical_gates_pass"
                ],
                "passes": audit["passes"],
            },
        )
        return path
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def _pilot_path(mode):
    short = "legacy" if mode.startswith("legacy") else "owner"
    return RECOVERY_ROOT / f"phase_b_G8_D1_{short}_steps_001_004.npz"


def _phase_b_segment(index, bundle, mode):
    path = _pilot_path(mode)
    stage_id = f"phase_b/G8/D1/{mode}/steps_001_004"
    metadata = {
        **stage_provenance_metadata(),
        "mode": mode, "dt": DT, "steps": PILOT_STEPS,
        "common_input_fingerprint": bundle.common_input_fingerprint,
        "extended_fingerprint": extended_case_fingerprint(bundle, mode, DT),
        "outer_position_sha256": hash_arrays(bundle.outer_reference_position),
        "outer_acceleration_sha256": hash_arrays(bundle.outer_reference_acceleration),
    }
    index.register(stage_id, "bounded-g8-rk2-segment", 7200.0, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        _segment_archive_audit(cached, bundle, mode)
        return cached
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        segment = run_rk2_segment(
            bundle, mode, DT, PILOT_STEPS,
            stage_validator=lambda record: _technical_stage_audit(
                bundle, mode, record,
            ),
        )
        endpoint_fingerprint = hash_arrays(*segment["end_state"])
        arrays = {
            "mode": np.asarray(mode), "dt": np.asarray(DT),
            "steps": np.asarray(PILOT_STEPS),
            "completed": np.asarray(segment["completed"]),
            "technical_pass": np.asarray(segment["technical_pass"]),
            "protocol_sha256": np.asarray(PROTOCOL_SHA256),
            "environment_sha256": np.asarray(environment_fingerprint()),
            "common_input_fingerprint": np.asarray(bundle.common_input_fingerprint),
            "extended_fingerprint": np.asarray(
                extended_case_fingerprint(bundle, mode, DT)
            ),
            "outer_reference_position": bundle.outer_reference_position,
            "outer_reference_acceleration": bundle.outer_reference_acceleration,
            "end_q": segment["end_state"][0],
            "end_v": segment["end_state"][1],
            "end_source": segment["end_state"][2],
            "end_memory": segment["end_state"][3],
            "endpoint_fingerprint": np.asarray(endpoint_fingerprint),
        }
        for record in segment["records"]:
            prefix = f"step_{record['step']:03d}_rk{record['rk_stage']}"
            arrays.update(_record_arrays(prefix, record))
        atomic_write_npz(path, **arrays)
        audit = _segment_archive_audit(path, bundle, mode)
        with np.load(path, allow_pickle=False) as archive:
            reload_equal = bool(all(np.array_equal(
                archive[key], expected,
            ) for key, expected in (
                ("end_q", segment["end_state"][0]),
                ("end_v", segment["end_state"][1]),
                ("end_source", segment["end_state"][2]),
                ("end_memory", segment["end_state"][3]),
            )))
        if not reload_equal:
            raise RuntimeError("pilot recovery endpoint reload is not bitwise")
        index.mark_complete(
            stage_id, path, time.perf_counter()-started,
            {
                "endpoint_reload_bitwise": True,
                "finite": segment["finite"], "passes": audit["passes"],
                "record_count": audit["record_count"],
            },
        )
        return path
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def _load_phase_a_bundle(index, label="G8"):
    parent_path = index.validated_path("phase_a/parent_projection")
    case_path = index.validated_path(f"phase_a/case/{label}")
    if parent_path is None or case_path is None:
        raise RuntimeError("Phase-B parent/case recovery artifacts are not validated")
    validate_scientific_npz(parent_path, embedded={
        "protocol_sha256": PROTOCOL_SHA256,
        "schema": "matched-staged-parent-projection-v1",
    })
    with np.load(parent_path, allow_pickle=False) as archive:
        parent = _parent_geometry_from_archive(archive)
        projected = _jet_from_archive(archive, label.lower())
    geometry = _geometry_from_projected(parent, label, projected)
    bundle = build_mode_neutral_case(
        geometry, f"{label}-matched", provenance=case_provenance(),
    )
    match = _case_matches_archive(bundle, case_path)
    if not match["all"]:
        raise RuntimeError("reconstructed Phase-B case differs from Phase A")
    return bundle


def run_phase_a(index):
    parent_path = _phase_a_parent_projection(index)
    result = _phase_a_audit(index, parent_path)
    print(json.dumps({
        "phase": "A", "classification": result["classification"],
        "result": str(PHASE_A_RESULT),
        "sha256": sha256_file(PHASE_A_RESULT),
    }, indent=2, sort_keys=True))
    return result


def run_phase_b(index):
    phase_a_path = index.validated_path("phase_a/audit")
    if phase_a_path is None:
        raise RuntimeError("Phase B requires a hash-validated Phase-A result")
    phase_a = json.loads(phase_a_path.read_text())
    if (
        phase_a.get("protocol_sha256") != PROTOCOL_SHA256
        or phase_a.get("environment_sha256") != environment_fingerprint()
        or not phase_a.get("gates", {}).get("all_phase_a_gates_pass")
        or phase_a.get("classification") != "PASS-phase-a"
    ):
        raise RuntimeError("Phase A did not pass; bounded G8 pilot is blocked")
    bundle = _load_phase_a_bundle(index, "G8")
    result_stage_id = "phase_b/result"
    result_metadata = {
        **stage_provenance_metadata(),
        "phase_a_sha256": sha256_file(phase_a_path),
        "scope": "bounded G8 t0 plus four-step D1 segment only",
    }
    index.register(result_stage_id, "bounded-g8-pilot-result", 21600.0,
                   result_metadata)
    cached_result = index.validated_path(result_stage_id)
    if cached_result is not None:
        result = json.loads(cached_result.read_text())
        if (
            result.get("protocol_sha256") != PROTOCOL_SHA256
            or result.get("environment_sha256") != environment_fingerprint()
            or result.get("phase_a_sha256") != sha256_file(phase_a_path)
        ):
            raise RuntimeError("cached bounded-pilot result identity mismatch")
        print(json.dumps({
            "phase": "B", "classification": result["classification"],
            "result": str(cached_result),
            "sha256": sha256_file(cached_result),
        }, indent=2, sort_keys=True))
        return result
    index.mark_running(result_stage_id)
    started = time.perf_counter()
    disk = shutil.disk_usage(RECOVERY_ROOT.parent)
    if disk.free < 100*1024**3:
        index.mark_failed(result_stage_id, "free disk below frozen 100 GiB gate")
        raise RuntimeError("bounded pilot requires at least 100 GiB free disk")
    try:
        authorization = validate_authorization()
        rhs0 = bundle.rhs_by_mode[BOUNDARY_MODES[0]]
        rhs1 = bundle.rhs_by_mode[BOUNDARY_MODES[1]]
        common_outer = bool(
            np.array_equal(
                rhs0.outer_reference_position, rhs1.outer_reference_position,
            )
            and np.array_equal(
                rhs0.outer_reference_acceleration,
                rhs1.outer_reference_acceleration,
            )
            and np.array_equal(
                rhs0.outer_reference_position, bundle.outer_reference_position,
            )
            and np.array_equal(
                rhs0.outer_reference_acceleration,
                bundle.outer_reference_acceleration,
            )
        )
        planned_common = {
            f"{mode}/dt={dt:.9f}": bundle.common_input_fingerprint
            for mode in BOUNDARY_MODES for dt in (DT, DT/2.0, DT/4.0)
        }

        t0_paths = {}
        t0_audits = {}
        for mode in BOUNDARY_MODES:
            path = _phase_b_t0(index, bundle, mode)
            t0_paths[mode] = path
            t0_audits[mode] = _t0_archive_audit(path, bundle, mode)
            if not t0_audits[mode]["passes"]:
                break

        segments = {}
        segment_audits = {}
        if len(t0_audits) == len(BOUNDARY_MODES) and all(
            audit["passes"] for audit in t0_audits.values()
        ):
            for mode in BOUNDARY_MODES:
                path = _phase_b_segment(index, bundle, mode)
                segments[mode] = path
                segment_audits[mode] = _segment_archive_audit(
                    path, bundle, mode,
                )
                if not segment_audits[mode]["passes"]:
                    break

        def compact_audit(audit):
            failures = []
            for position, record in enumerate(audit.get("metadata", [])):
                technical = record.get("technical_audit") or {}
                for name, passed in technical.get("gates", {}).items():
                    if name != "all_technical_gates_pass" and not passed:
                        failures.append({"record": position, "gate": name})
            return {
                key: value for key, value in audit.items()
                if key != "metadata"
            } | {"technical_failures": failures}

        artifact_records = {
            "t0": {
                mode: {
                    "path": str(path), "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "audit": compact_audit(t0_audits[mode]),
                } for mode, path in t0_paths.items()
            },
            "segments": {
                mode: {
                    "path": str(path), "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "audit": compact_audit(segment_audits[mode]),
                } for mode, path in segments.items()
            },
        }
        all_artifacts = [
            record for group in artifact_records.values()
            for record in group.values()
        ]
        all_manifest_valid = bool(all(
            index.validated_path(
                f"phase_b/G8/t0/{mode}"
                if kind == "t0" else
                f"phase_b/G8/D1/{mode}/steps_001_004"
            ) is not None
            for kind, paths in (("t0", t0_paths), ("segments", segments))
            for mode in paths
        ))
        storage = {
            "free_bytes_before_phase_b": int(disk.free),
            "free_GiB_before_phase_b": float(disk.free/1024**3),
            "projected_total_matrix_bytes": None,
            "projected_total_matrix_GiB": None,
            "projection_safety_factor": 1.25,
        }
        if len(segments) == len(BOUNDARY_MODES):
            segment_bytes = sum(path.stat().st_size for path in segments.values())
            g8_points = TARGET_GRIDS["G8"].nz*TARGET_GRIDS["G8"].nr_r10
            point_ratios = sum(
                spec.nz*spec.nr_r10/g8_points
                for spec in TARGET_GRIDS.values()
            )
            phase_a_bytes = sum(
                Path(path).stat().st_size for path in (
                    phase_a["artifacts"]["parent_projection"],
                    *phase_a["artifacts"]["references"].values(),
                    *phase_a["artifacts"]["cases"].values(),
                )
            )
            projected = int(np.ceil(
                1.25*(segment_bytes*28*point_ratios + phase_a_bytes)
            ))
            storage["projected_total_matrix_bytes"] = projected
            storage["projected_total_matrix_GiB"] = projected/1024**3

        validation_gate = bool(
            authorization.get("protocol_sha256") == PROTOCOL_SHA256
            and all(
                authorization["validations"].get(name, {}).get("decision")
                == "accepted"
                for name in ("PARENT-REPRESENTATION", "OUTER-REFERENCE")
            )
        )
        all_t0 = bool(
            len(t0_audits) == len(BOUNDARY_MODES)
            and all(audit["passes"] for audit in t0_audits.values())
        )
        all_segments = bool(
            len(segment_audits) == len(BOUNDARY_MODES)
            and all(audit["passes"] for audit in segment_audits.values())
        )
        gates = {
            "validation_records_match_protocol": validation_gate,
            "phase_a_passed_and_hash_validated": True,
            "planned_lane_common_input_fingerprints_match": bool(
                len(set(planned_common.values())) == 1
            ),
            "common_outer_position_and_acceleration_bitwise": common_outer,
            "both_t0_trace_and_technical_audits_pass": all_t0,
            "both_four_step_segments_complete_and_technical": all_segments,
            "all_mandatory_stages_shapes_and_owner_open_face_pass": bool(
                all_t0 and all_segments
            ),
            "all_artifacts_hash_validated_and_reloadable": bool(
                all_manifest_valid and all(
                    record["audit"].get("passes", False)
                    for record in all_artifacts
                )
            ),
            "all_chunks_below_1_GiB": bool(
                all_artifacts
                and all(record["bytes"] < 1024**3 for record in all_artifacts)
            ),
            "projected_total_matrix_at_most_50_GiB": bool(
                storage["projected_total_matrix_bytes"] is not None
                and storage["projected_total_matrix_bytes"] <= 50*1024**3
            ),
            "free_disk_before_authorization_at_least_100_GiB": bool(
                disk.free >= 100*1024**3
            ),
            "prospective_protocol_and_inputs_unchanged": bool(
                sha256_file(PROTOCOL) == PROTOCOL_SHA256
                and index.data.get("expected_inputs") == expected_inputs()
            ),
            "full_matrix_remains_unauthorized": True,
        }
        passed = bool(all(gates.values()))
        result = {
            "schema": "A790-matched-staged-continuum-bounded-g8-pilot-v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "environment_sha256": environment_fingerprint(),
            "phase_a_sha256": sha256_file(phase_a_path),
            "classification": "PASS-pilot" if passed else "FAIL-audit-pilot",
            "gates": {**gates, "all_pilot_gates_pass": passed},
            "common_input_fingerprint": bundle.common_input_fingerprint,
            "planned_lane_common_input_fingerprints": planned_common,
            "outer_reference_hashes": {
                "position": hash_arrays(bundle.outer_reference_position),
                "acceleration": hash_arrays(bundle.outer_reference_acceleration),
            },
            "storage": storage,
            "artifacts": artifact_records,
            "full_matrix_authorized": False,
            "full_matrix_authorization_requires_separate_result_note": True,
            "environment": public_environment(),
        }
        if not _nested_finite(result):
            raise RuntimeError("nonfinite bounded-pilot result metadata")
        atomic_write_json(PHASE_B_RESULT, result)
        index.mark_complete(
            result_stage_id, PHASE_B_RESULT, time.perf_counter()-started,
            {"classification": result["classification"], "passed": passed},
        )
        print(json.dumps({
            "phase": "B", "classification": result["classification"],
            "result": str(PHASE_B_RESULT),
            "sha256": sha256_file(PHASE_B_RESULT),
        }, indent=2, sort_keys=True))
        return result
    except Exception as error:
        index.mark_failed(result_stage_id, f"{type(error).__name__}: {error}")
        raise


def plan():
    return {
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA256,
        "validation": str(VALIDATION),
        "available_actions": ["--phase-a", "--phase-b-g8"],
        "phase_a": "P10/P11 plus three target projections and qualification",
        "phase_b_g8": "t=0 plus one four-step D1 segment in two modes",
        "full_matrix_action_present": False,
        "no_output_generated": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--phase-a", action="store_true")
    group.add_argument("--phase-b-g8", action="store_true")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    validate_authorization()
    if not args.phase_a and not args.phase_b_g8:
        print(json.dumps(plan(), indent=2, sort_keys=True))
        return 0
    index = recovery_index()
    if args.phase_a:
        run_phase_a(index)
    else:
        run_phase_b(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
