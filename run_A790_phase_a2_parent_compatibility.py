#!/usr/bin/env python3
"""Bounded native-P11 compatible-acceleration diagnostic.

The executable implements Protocol 123 only.  It can reconstruct and cache
the native P11 source data, complete the archived bulk acceleration at the
compact walls, and audit that correction.  It contains no target projection,
evolution RHS call, RK step, or continuum-matrix scheduler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bhps.gw_slice_high_order_solver import derivative_matrix  # noqa: E402
from bhps.junction_second_preservation_diagnostic import (  # noqa: E402
    wall_junction_second_tangent,
)
from bhps.matched_staged_continuum import (  # noqa: E402
    DriverConfiguration,
    ProjectedJetField,
    axis_even_crossfit_audit,
    hash_arrays,
    projected_second_wall_audit,
)
from bhps.gh_source_driver import (  # noqa: E402
    regular_so3_live_source_shift_advection,
    regular_so3_nonlinear_anchored_damped_wave_target,
    source_driver_rhs,
)
from bhps.nonlinear_regular_so3_evolution import (  # noqa: E402
    CompactWallCoupledAlgebraicGateError,
    apply_compact_wall_acceleration,
    compact_wall_normal_gauge_acceleration_residuals,
    impose_compact_wall_normal_tangential_acceleration,
    gauge_taylor_source_from_initial_jets,
    live_regular_source_second_time,
    regular_source_spatial_derivatives,
    reconcile_wall_owner_axis_null_channels,
    solve_compact_wall_coupled_phi_normal_acceleration,
)
from bhps.recovery_indexer import (  # noqa: E402
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
)
from bhps.regular_so3_gh_reduction import FIELD_ORDER  # noqa: E402


PROTOCOL = Path("notes/123_A790_phase_a2_parent_compatibility_protocol.md")
PROTOCOL_SHA256 = (
    "c7219f24b659620057ed380273b3932613c566fe3bfe503fd04cc4fa47b35e08"
)
RECOVERY_ROOT = Path(
    "results/corrected_A790_phase_a2_parent_compatibility_recovery"
)
VALIDATION = RECOVERY_ROOT / "validation.json"
PARENT = Path(
    "results/corrected_A790_matched_staged_continuum_recovery/"
    "phase_a_parent_projection.npz"
)
PARENT_SHA256 = (
    "30c578ce142159a8e0842a22afab8b436bd85af55f1516c1bfe11fce96968dc7"
)
PHASE_A = Path("results/corrected_A790_matched_staged_continuum_phase_a.json")
PHASE_A_SHA256 = (
    "af602f00236550027d6e934e652917cb8e63a56a40f32ae6b80fb718265ee47a"
)
DECOMPOSITION = Path("results/corrected_A790_phase_a2_failure_decomposition.json")
DECOMPOSITION_SHA256 = (
    "29a669ed1eecda45b1556e9198def443ac695b8ae22032f835cd190a75eaabd3"
)
MANIFEST = RECOVERY_ROOT / "index.json"
SOURCE_ARTIFACT = RECOVERY_ROOT / "native_P11_source.npz"
CORRECTION_ARTIFACT = RECOVERY_ROOT / "native_P11_compatible_acceleration.npz"
RESULT = RECOVERY_ROOT / "result.json"


class PhaseA2ScientificGateFailure(RuntimeError):
    """A classified scientific stop, distinct from an implementation crash."""

    def __init__(self, result):
        super().__init__(str(result.get("failure", "Phase-A2 scientific gate failed")))
        self.result = result


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


def _nested_finite(value):
    """Return False for any nonfinite numeric leaf in a diagnostic record."""
    if isinstance(value, dict):
        return all(_nested_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_nested_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(
            value.dtype.kind not in "iufc"
            or np.all(np.isfinite(value))
        )
    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        return bool(np.isfinite(value))
    return True


def public_environment():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
    }


def environment_fingerprint():
    payload = json.dumps(
        public_environment(), sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def transitive_inputs():
    local_modules = tuple(sorted(Path("src/bhps").rglob("*.py")))
    static = (
        Path(__file__), PROTOCOL, VALIDATION, PARENT, PHASE_A, DECOMPOSITION,
        Path("tests/test_phase_a2_parent_compatibility.py"),
        Path("tests/test_phase_a2_diagnostics.py"),
    )
    return tuple(dict.fromkeys((*static, *local_modules)))


def expected_inputs():
    missing = [str(path) for path in transitive_inputs() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Phase-A2 inputs: {missing}")
    return {str(path): sha256_file(path) for path in transitive_inputs()}


def validate_authorization():
    identities = (
        (PROTOCOL, PROTOCOL_SHA256),
        (PARENT, PARENT_SHA256),
        (PHASE_A, PHASE_A_SHA256),
        (DECOMPOSITION, DECOMPOSITION_SHA256),
    )
    for path, expected in identities:
        found = sha256_file(path)
        if found != expected:
            raise RuntimeError(f"sealed identity mismatch for {path}: {found}")
    record = json.loads(VALIDATION.read_text())
    if record.get("protocol_sha256") != PROTOCOL_SHA256:
        raise RuntimeError("validation does not name Protocol 123")
    authorization = record.get("authorization", {})
    expected = {
        "native_P11_compatible_acceleration": "authorized",
        "target_projection": "not authorized",
        "RHS_or_RK": "not authorized",
        "full_matrix": "not authorized",
        "new_interface_physics": "not authorized",
    }
    if authorization != expected:
        raise RuntimeError("validation authorization scope mismatch")
    bound_files = record.get("candidate_file_sha256", {})
    required_bound_files = (
        Path(__file__),
        Path("src/bhps/nonlinear_regular_so3_evolution.py"),
        Path("tests/test_phase_a2_parent_compatibility.py"),
    )
    for path in required_bound_files:
        expected_hash = bound_files.get(str(path))
        if expected_hash is None or sha256_file(path) != expected_hash:
            raise RuntimeError(f"validation candidate hash mismatch for {path}")
    frozen_mtimes = record.get("immutable_input_mtime_ns", {})
    for path in (PARENT, PHASE_A, DECOMPOSITION):
        expected_mtime = frozen_mtimes.get(str(path))
        if expected_mtime is None or path.stat().st_mtime_ns != int(expected_mtime):
            raise RuntimeError(f"immutable input mtime mismatch for {path}")
    return record


def recovery_index():
    return RecoveryIndex(
        MANIFEST, PROTOCOL, expected_inputs(), maximum_stage_seconds=3600.0,
    )


def _validate_npz(path, required, embedded):
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(required)-set(archive.files))
        if missing:
            raise ValueError(f"missing arrays in {path}: {missing}")
        for key, shape in required.items():
            if tuple(archive[key].shape) != tuple(shape):
                raise ValueError(f"{key} has shape {archive[key].shape}, expected {shape}")
        for key in archive.files:
            value = archive[key]
            if value.dtype.kind in "biufc" and not np.all(np.isfinite(value)):
                raise ValueError(f"nonfinite array {key} in {path}")
            if value.dtype.kind not in "biufcSU":
                raise ValueError(f"unsupported dtype for {key}: {value.dtype}")
        for key, expected in embedded.items():
            if key not in archive.files or str(archive[key]) != str(expected):
                raise ValueError(f"embedded identity mismatch for {key}")


def _require_provenance_revalidation(record):
    failed = [name for name, passed in record.items() if not bool(passed)]
    if failed:
        raise RuntimeError(
            "Phase-A2 recovery/provenance revalidation failed: "
            + ", ".join(failed)
        )


def _load_parent():
    with np.load(PARENT, allow_pickle=False) as archive:
        z = np.asarray(archive["p11_z"])
        r = np.asarray(archive["p11_r"])
        q = np.asarray(archive["p11_q"])
        first = np.asarray(archive["p11_first"])
        second = np.asarray(archive["p11_second"])
        background = json.loads(str(archive["background_json"]))
        for value in (z, r, q, first, second):
            value.setflags(write=False)
        return {
            "name": str(archive["p11_name"]),
            "z": z,
            "r": r,
            "q": q,
            "first": first,
            "second": second,
            "background": background,
            "mass_squared": float(archive["mass_squared"]),
            "fold_amplitude": float(archive["fold_amplitude"]),
            "jet": ProjectedJetField(z, r, q, first, second),
        }


def _source_stage(index, parent):
    stage_id = "phase_a2/native_P11_source"
    metadata = {
        "protocol_sha256": PROTOCOL_SHA256,
        "environment_sha256": environment_fingerprint(),
        "parent_sha256": PARENT_SHA256,
        "shape": list(parent["q"].shape),
        "scope": "source reconstruction only; no RHS acceleration call",
        "driver": DriverConfiguration().public(),
    }
    index.register(stage_id, "native-P11-source", 1200.0, metadata)
    cached = index.validated_path(stage_id)
    required = {
        "z": parent["z"].shape,
        "r": parent["r"].shape,
        "source": parent["q"].shape[:2]+(3,),
        "source_time": parent["q"].shape[:2]+(3,),
        "source_second_time": parent["q"].shape[:2]+(3,),
        "memory": parent["q"].shape[:2]+(3,),
    }
    embedded = {
        "schema": "A790-phase-a2-native-P11-source-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "parent_sha256": PARENT_SHA256,
        "environment_sha256": environment_fingerprint(),
    }
    if cached is not None:
        _validate_npz(cached, required, embedded)
        return cached
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        driver = DriverConfiguration()
        q = np.asarray(parent["q"])
        v = np.asarray(parent["first"][0])
        taylor = gauge_taylor_source_from_initial_jets(
            parent["jet"], parent["z"], parent["r"],
        )
        source = np.asarray(taylor.source_reduced).copy()
        source_time = np.asarray(taylor.source_time_reduced).copy()
        target = regular_so3_nonlinear_anchored_damped_wave_target(
            q, q, source, parent["r"], driver.target_mu_lapse,
            driver.target_mu_shift, driver.target_power,
        )
        source_z, source_r = regular_source_spatial_derivatives(
            source, parent["z"], parent["r"], driver.stencil_width,
        )
        advection = regular_so3_live_source_shift_advection(
            q, parent["r"], source, source_z, source_r,
        )
        memory = (
            source_time-advection+driver.driver_mu*(source-target)
        )
        source_dot, memory_dot = source_driver_rhs(
            source, memory, target, driver.driver_mu, driver.driver_eta,
            advection,
        )
        source_second = live_regular_source_second_time(
            q, v, q, source, source, source_dot, memory_dot,
            parent["z"], parent["r"], driver.driver_mu,
            driver.target_mu_lapse, driver.target_mu_shift,
            driver.target_power, driver.stencil_width,
        )
        atomic_write_npz(
            SOURCE_ARTIFACT,
            schema=np.asarray(embedded["schema"]),
            protocol_sha256=np.asarray(PROTOCOL_SHA256),
            parent_sha256=np.asarray(PARENT_SHA256),
            environment_sha256=np.asarray(environment_fingerprint()),
            z=parent["z"], r=parent["r"],
            source=source,
            source_time=source_time,
            source_second_time=source_second,
            memory=memory,
            driver_configuration_json=np.asarray(json.dumps(
                driver.public(), sort_keys=True,
            )),
            q_fingerprint=np.asarray(hash_arrays(parent["q"])),
            v_fingerprint=np.asarray(hash_arrays(parent["first"][0])),
        )
        _validate_npz(SOURCE_ARTIFACT, required, embedded)
        index.mark_complete(
            stage_id, SOURCE_ARTIFACT, time.perf_counter()-started,
            {"source_fingerprint": hash_arrays(
                source, source_time, source_second, memory,
            )},
        )
        return SOURCE_ARTIFACT
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def _endpoint_derivative(values, z):
    matrix = derivative_matrix(np.asarray(z), 1, 7)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    derivative = np.einsum("ij,jrf->irf", matrix, np.asarray(values))
    return np.stack((derivative[0], derivative[-1]))


def _support_audit(bulk, coupled, physical, gauge, final):
    shape = bulk.shape
    endpoint_mask = np.zeros(shape, dtype=bool)
    endpoint_mask[[0, -1], :, :] = True
    null_axis_mask = np.zeros(shape, dtype=bool)
    null_axis_mask[:, 0, 4] = True
    null_axis_mask[:, 0, 5] = True
    coupled_allowed = np.zeros(shape, dtype=bool)
    coupled_allowed[[0, -1], :, 6] = True
    coupled_allowed[[0, -1], :, 7] = True
    physical_allowed = np.zeros(shape, dtype=bool)
    physical_allowed[[0, -1], :, 2] = True
    physical_allowed[[0, -1], :, 3] = True
    physical_allowed[[0, -1], :, 4] = True
    physical_allowed[[0, -1], :, 5] = True
    physical_allowed[[0, -1], 0, 4] = False
    physical_allowed[[0, -1], 0, 5] = False
    physical_allowed[[0, -1], :, 7] = True
    physical_allowed[[0, -1], :, 8] = True
    gauge_allowed = np.zeros(shape, dtype=bool)
    gauge_allowed[[0, -1], :, 0] = True
    gauge_allowed[[0, -1], :, 1] = True
    def byte_change(left, right):
        left_bits = np.ascontiguousarray(left).view(np.uint64).reshape(shape)
        right_bits = np.ascontiguousarray(right).view(np.uint64).reshape(shape)
        return left_bits != right_bits

    coupled_change = byte_change(coupled, bulk)
    physical_change = byte_change(physical, bulk)
    physical_step_change = byte_change(physical, coupled)
    gauge_change = byte_change(gauge, physical)
    reconciliation_change = byte_change(final, gauge)
    final_change = byte_change(final, bulk)
    numerical_final_change = final != bulk
    return {
        "coupled_changes_only_wall_Phi_gzz": bool(
            not np.any(coupled_change & ~coupled_allowed)
        ),
        "physical_stage_changes_only_owned_wall_fields": bool(
            not np.any(physical_step_change & ~physical_allowed)
        ),
        "gauge_stage_changes_only_normal_tangential_wall_fields": bool(
            not np.any(gauge_change & ~gauge_allowed)
        ),
        "physical_stage_preserves_coupled_gzz_bitwise": bool(
            not np.any(physical_step_change[[0, -1], :, 6])
        ),
        "reconciliation_changes_only_q4_q5_axis": bool(
            not np.any(reconciliation_change & ~null_axis_mask)
        ),
        "final_changes_only_wall_or_q4_q5_axis": bool(
            not np.any(final_change & ~(endpoint_mask | null_axis_mask))
        ),
        "coupled_changed_value_count": int(np.count_nonzero(coupled_change)),
        "physical_changed_value_count": int(np.count_nonzero(physical_change)),
        "physical_step_changed_value_count": int(
            np.count_nonzero(physical_step_change)
        ),
        "gauge_changed_value_count": int(np.count_nonzero(gauge_change)),
        "reconciliation_changed_value_count": int(
            np.count_nonzero(reconciliation_change)
        ),
        "final_changed_value_count": int(np.count_nonzero(final_change)),
        "final_numerically_changed_value_count": int(
            np.count_nonzero(numerical_final_change)
        ),
        "signed_zero_byte_change_count": int(np.count_nonzero(
            final_change & ~numerical_final_change
        )),
        "stage_arrays_do_not_share_memory": bool(not any(
            np.shares_memory(left, right)
            for i, left in enumerate((bulk, coupled, physical, gauge, final))
            for right in (bulk, coupled, physical, gauge, final)[i+1:]
        )),
    }


def _support_masks(bulk, coupled, physical, gauge, final):
    shape = bulk.shape

    def changed(left, right):
        left_bits = np.ascontiguousarray(left).view(np.uint64).reshape(shape)
        right_bits = np.ascontiguousarray(right).view(np.uint64).reshape(shape)
        return (left_bits != right_bits).astype(np.uint8)

    allowed_coupled = np.zeros(shape, dtype=np.uint8)
    allowed_coupled[[0, -1], :, 6:8] = 1
    allowed_physical = np.zeros(shape, dtype=np.uint8)
    allowed_physical[[0, -1], :, 2:6] = 1
    allowed_physical[[0, -1], :, 7:9] = 1
    allowed_physical[[0, -1], 0, 4:6] = 0
    allowed_gauge = np.zeros(shape, dtype=np.uint8)
    allowed_gauge[[0, -1], :, 0:2] = 1
    allowed_reconciliation = np.zeros(shape, dtype=np.uint8)
    allowed_reconciliation[:, 0, 4:6] = 1
    allowed_final = np.maximum.reduce((
        allowed_coupled, allowed_physical, allowed_gauge,
        allowed_reconciliation,
    ))
    return {
        "mask_actual_coupled": changed(coupled, bulk),
        "mask_actual_physical": changed(physical, coupled),
        "mask_actual_gauge": changed(gauge, physical),
        "mask_actual_reconciliation": changed(final, gauge),
        "mask_actual_final": changed(final, bulk),
        "mask_numerical_coupled": (coupled != bulk).astype(np.uint8),
        "mask_numerical_physical": (physical != coupled).astype(np.uint8),
        "mask_numerical_gauge": (gauge != physical).astype(np.uint8),
        "mask_numerical_reconciliation": (final != gauge).astype(np.uint8),
        "mask_numerical_final": (final != bulk).astype(np.uint8),
        "mask_allowed_coupled": allowed_coupled,
        "mask_allowed_physical": allowed_physical,
        "mask_allowed_gauge": allowed_gauge,
        "mask_allowed_reconciliation": allowed_reconciliation,
        "mask_allowed_final": allowed_final,
    }


def _physical_acceleration(acceleration, r):
    a = np.asarray(acceleration, dtype=float)
    radius = np.asarray(r, dtype=float)[None, :]
    return np.stack((
        a[:, :, 0], radius*a[:, :, 1], a[:, :, 2], a[:, :, 3],
        a[:, :, 3]+radius**2*a[:, :, 4], radius*a[:, :, 5],
        a[:, :, 6], a[:, :, 7], a[:, :, 8],
    ), axis=-1)


def _positive_zero_wall_gauge(acceleration):
    gauge = np.ascontiguousarray(
        np.asarray(acceleration)[[0, -1], :, 0:2], dtype=np.float64,
    )
    return bool(np.all(gauge.view(np.uint64) == 0))


def classify_correction(structural_pass, normalized_linf, weighted_rms):
    """Apply the frozen Protocol-123 classification boundaries."""
    normalized_linf = float(normalized_linf)
    weighted_rms = float(weighted_rms)
    if not bool(structural_pass) or not np.isfinite(
        [normalized_linf, weighted_rms]
    ).all() or normalized_linf > 5e-1:
        return "FAIL-parent-acceleration"
    if normalized_linf <= 5e-2 and weighted_rms <= 1e-2:
        return "PASS-small-parent-acceleration"
    return "REVIEW-large-correction"


def _proper_radius(position, r, wall_index):
    radial = position[wall_index, :, 3] + r**2*position[wall_index, :, 4]
    if np.any(radial <= 0.0):
        raise RuntimeError("nonpositive radial wall metric")
    increments = 0.5*(np.sqrt(radial[1:])+np.sqrt(radial[:-1]))*np.diff(r)
    return np.concatenate(([0.0], np.cumsum(increments)))


def _line_weighted_wall_rms(values, proper):
    values = np.asarray(values, dtype=float)
    proper = np.asarray(proper, dtype=float)
    denominator = float((proper[-1]-proper[0])*values.shape[-1])
    if denominator <= 0.0:
        raise RuntimeError("degenerate proper wall measure")
    numerator = float(np.sum([
        np.trapezoid(values[:, component]**2, proper)
        for component in range(values.shape[-1])
    ]))
    return float(np.sqrt(numerator/denominator))


def _nodal_widths(r):
    r = np.asarray(r, dtype=float)
    widths = np.empty_like(r)
    widths[0] = 0.5*(r[1]-r[0])
    widths[-1] = 0.5*(r[-1]-r[-2])
    widths[1:-1] = 0.5*(r[2:]-r[:-2])
    return widths


def _r90_index(density):
    density = np.asarray(density, dtype=float)
    cumulative = np.cumsum(density)
    if not len(cumulative) or cumulative[-1] <= 0.0:
        return 0
    return int(np.searchsorted(cumulative, 0.9*float(cumulative[-1])))


def _proper_wall_weights(position, r, wall_index):
    radial = position[wall_index, :, 3]+r**2*position[wall_index, :, 4]
    sphere = position[wall_index, :, 3]
    if np.any(radial <= 0.0) or np.any(sphere <= 0.0):
        raise RuntimeError("nonpositive proper-wall metric")
    return (
        4.0*np.pi*r**2*sphere*np.sqrt(radial)*_nodal_widths(r)
    )


def correction_norms(position, bulk, compatible, r):
    names = (
        "h_z0", "h_zr", "h_00", "h_perp", "h_rr", "h_0r", "h_zz",
        "Phi", "chi",
    )
    physical_bulk = _physical_acceleration(bulk, r)
    physical_compatible = _physical_acceleration(compatible, r)
    physical_delta = physical_compatible-physical_bulk
    normalized = np.abs(physical_delta)/np.maximum.reduce((
        np.ones_like(physical_delta), np.abs(physical_bulk),
        np.abs(physical_compatible),
    ))
    walls = {}
    line_weighted = []
    proper_numerator = 0.0
    proper_denominator = 0.0
    component_accumulator = {name: [] for name in names}
    combined_radial_density = np.zeros(len(r))
    global_unweighted_radial_density = np.sum(normalized**2, axis=(0, 2))
    global_maximum = np.unravel_index(
        int(np.argmax(normalized)), normalized.shape,
    )
    for wall, index in (("lower", 0), ("upper", -1)):
        proper = _proper_radius(position, r, index)
        local = normalized[index]
        line_weighted.append(_line_weighted_wall_rms(local, proper))
        weights = _proper_wall_weights(position, r, index)
        radial_metric = position[index, :, 3]+r**2*position[index, :, 4]
        line_weights = np.sqrt(radial_metric)*_nodal_widths(r)
        proper_numerator += float(np.sum(weights[:, None]*local**2))
        proper_denominator += float(local.shape[-1]*np.sum(weights))
        local_density = np.sum(weights[:, None]*local**2, axis=1)
        combined_radial_density += local_density
        local_r90 = _r90_index(local_density)
        local_unweighted_r90 = _r90_index(np.sum(local**2, axis=1))
        components = {}
        for component, name in enumerate(names):
            profile = local[:, component]
            maximum = int(np.argmax(profile))
            area_density = weights*profile**2
            line_density = line_weights*profile**2
            absolute_density = line_weights*physical_delta[
                index, :, component
            ]**2

            unweighted_r90 = _r90_index(profile**2)
            area_r90 = _r90_index(area_density)
            line_r90 = _r90_index(line_density)
            absolute_r90 = _r90_index(absolute_density)
            component_accumulator[name].append((
                profile, weights, line_weights,
            ))
            components[name] = {
                "normalized_Linf": float(profile[maximum]),
                "normalized_RMS": float(np.sqrt(np.mean(profile**2))),
                "proper_wall_measure_weighted_RMS": float(np.sqrt(
                    np.sum(area_density)/max(np.sum(weights), 1e-300)
                )),
                "proper_radial_line_weighted_RMS": float(np.sqrt(
                    np.sum(line_density)/max(np.sum(line_weights), 1e-300)
                )),
                "maximum_r": float(r[maximum]),
                "maximum_proper_radius": float(proper[maximum]),
                "unweighted_squared_correction_r90": float(
                    r[unweighted_r90]
                ),
                "unweighted_e_squared_r90": float(r[unweighted_r90]),
                "area_weighted_r90": float(r[area_r90]),
                "line_weighted_r90": float(r[line_r90]),
                "absolute_delta_line_weighted_r90": float(r[absolute_r90]),
                "raw_delta_Linf": float(np.max(np.abs(
                    physical_delta[index, :, component]
                ))),
            }
        walls[wall] = {
            "proper_radius_maximum": float(proper[-1]),
            "proper_radial_line_weighted_RMS": line_weighted[-1],
            "proper_wall_measure_weighted_RMS": float(np.sqrt(
                np.sum(weights[:, None]*local**2)
                /max(local.shape[-1]*np.sum(weights), 1e-300)
            )),
            "combined_normalized_RMS": float(np.sqrt(np.mean(local**2))),
            "unweighted_squared_correction_r90": float(
                r[local_unweighted_r90]
            ),
            "unweighted_e_squared_r90": float(r[local_unweighted_r90]),
            "area_weighted_combined_r90": float(r[local_r90]),
            "combined_normalized_Linf": float(np.max(local)),
            "components": components,
        }
    global_components = {}
    for component, (name, records) in enumerate(component_accumulator.items()):
        full_profile = normalized[:, :, component]
        maximum = np.unravel_index(
            int(np.argmax(full_profile)), full_profile.shape,
        )
        unweighted_r90 = _r90_index(np.sum(full_profile**2, axis=0))
        area_numerator = sum(float(np.sum(
            record[1]*record[0]**2
        )) for record in records)
        area_denominator = sum(float(np.sum(record[1])) for record in records)
        line_numerator = sum(float(np.sum(
            record[2]*record[0]**2
        )) for record in records)
        line_denominator = sum(float(np.sum(record[2])) for record in records)
        global_components[name] = {
            "normalized_Linf": float(full_profile[maximum]),
            "normalized_RMS": float(np.sqrt(np.mean(full_profile**2))),
            "maximum_z_index": int(maximum[0]),
            "maximum_r_index": int(maximum[1]),
            "maximum_r": float(r[maximum[1]]),
            "unweighted_squared_correction_r90": float(r[unweighted_r90]),
            "unweighted_e_squared_r90": float(r[unweighted_r90]),
            "proper_wall_measure_weighted_RMS": float(np.sqrt(
                area_numerator/max(area_denominator, 1e-300)
            )),
            "proper_radial_line_weighted_RMS": float(np.sqrt(
                line_numerator/max(line_denominator, 1e-300)
            )),
        }
    combined_r90 = _r90_index(combined_radial_density)
    global_unweighted_r90 = _r90_index(global_unweighted_radial_density)
    raw_delta = compatible-bulk
    return {
        "physical_component_order": list(names),
        "global_normalized_Linf": float(np.max(normalized)),
        "global_normalized_RMS": float(np.sqrt(np.mean(normalized**2))),
        "global_maximum_z_index": int(global_maximum[0]),
        "global_maximum_r_index": int(global_maximum[1]),
        "global_maximum_component_index": int(global_maximum[2]),
        "global_maximum_r": float(r[global_maximum[1]]),
        "global_unweighted_squared_correction_r90": float(
            r[global_unweighted_r90]
        ),
        "global_unweighted_e_squared_r90": float(r[global_unweighted_r90]),
        "combined_proper_wall_weighted_RMS": float(np.sqrt(
            proper_numerator/max(proper_denominator, 1e-300)
        )),
        "combined_proper_radial_line_weighted_RMS": float(np.sqrt(np.mean(
            np.asarray(line_weighted)**2
        ))),
        "proper_wall_measure": "4*pi*r^2*g_sphere*sqrt(g_rr)*nodal_dr",
        "proper_wall_gate_equation": (
            "sqrt(sum_wall,r,component(w*e^2)/(9*sum_wall,r(w)))"
        ),
        "combined_area_weighted_r90": float(r[combined_r90]),
        "global_components": global_components,
        "walls": walls,
        "raw_reduced_q4": {
            "absolute_Linf": float(np.max(np.abs(raw_delta[:, :, 4]))),
            "scaled_Linf": float(np.max(np.abs(raw_delta[:, :, 4]))/max(
                1.0, float(np.max(np.abs(bulk[:, :, 4]))),
                float(np.max(np.abs(compatible[:, :, 4]))),
            )),
        },
        "raw_reduced_q5": {
            "absolute_Linf": float(np.max(np.abs(raw_delta[:, :, 5]))),
            "scaled_Linf": float(np.max(np.abs(raw_delta[:, :, 5]))/max(
                1.0, float(np.max(np.abs(bulk[:, :, 5]))),
                float(np.max(np.abs(compatible[:, :, 5]))),
            )),
        },
        "axis_derivative_images": {
            "2_delta_q4_for_drr_hrr_minus_hperp": {
                "absolute_Linf": float(np.max(np.abs(
                    2.0*raw_delta[:, 0, 4]
                ))),
                "values": (2.0*raw_delta[:, 0, 4]).tolist(),
            },
            "delta_q5_for_dr_h0r": {
                "absolute_Linf": float(np.max(np.abs(
                    raw_delta[:, 0, 5]
                ))),
                "values": raw_delta[:, 0, 5].tolist(),
            },
        },
        "small_Linf_gate": bool(np.max(normalized) <= 5e-2),
        "small_weighted_RMS_gate": bool(
            np.sqrt(proper_numerator/max(proper_denominator, 1e-300)) <= 1e-2
        ),
        "order_one_failure": bool(np.max(normalized) > 5e-1),
    }


def _wall_reconciliation_invariance(position, velocity, before, after, z, r, background):
    maximum_metric = 0.0
    maximum_phi = 0.0
    maximum_chi = 0.0
    for wall_name in ("lower", "upper"):
        left = wall_junction_second_tangent(
            position, velocity, before, z, r, background, wall_name, 7,
        )
        right = wall_junction_second_tangent(
            position, velocity, after, z, r, background, wall_name, 7,
        )
        maximum_metric = max(maximum_metric, float(np.max(np.abs(
            left["DX2J_tensor"]-right["DX2J_tensor"]
        ))))
        maximum_phi = max(maximum_phi, float(np.max(np.abs(
            left["separate_rows"]["DX2_Phi_robin"]
            - right["separate_rows"]["DX2_Phi_robin"]
        ))))
        maximum_chi = max(maximum_chi, float(np.max(np.abs(
            left["separate_rows"]["DX2_chi_neumann"]
            - right["separate_rows"]["DX2_chi_neumann"]
        ))))
    return {
        "metric_DX2J_Linf_change": maximum_metric,
        "Phi_DX2_Linf_change": maximum_phi,
        "chi_DX2_Linf_change": maximum_chi,
        "unchanged_to_1e_12": bool(max(
            maximum_metric, maximum_phi, maximum_chi,
        ) <= 1e-12),
    }


def _stage_second_wall_audits(parent, stages):
    """Persist unbuffered wall profiles for every frozen acceleration stage."""
    records = {}
    for name, acceleration in stages.items():
        second = np.asarray(parent["second"]).copy()
        second[0, 0] = np.asarray(acceleration)
        jet = ProjectedJetField(
            parent["z"], parent["r"], parent["q"], parent["first"], second,
        )
        records[name] = projected_second_wall_audit(
            jet, parent["background"], radial_buffer=0,
        )
    return records


def _row_implied_physical_endpoint_derivative(
    position, velocity, acceleration, source, source_time,
    source_second_time, z, r, background,
):
    """Recover physical endpoint ``a_z`` from the native wall rows.

    The two normal-tangential fields are Dirichlet gauge data, so their
    derivatives remain explicitly stencil-derived and are excluded from the
    row-defined mask.  Every other physical component has unit or known
    nonzero coefficient in its native differentiated wall row.
    """
    direct_reduced = _endpoint_derivative(acceleration, z)
    direct_physical = _physical_acceleration(direct_reduced, r)
    row_implied = direct_physical.copy()
    normal = compact_wall_normal_gauge_acceleration_residuals(
        position, velocity, acceleration, source, source_time,
        source_second_time, z, r, background, 7, radial_buffer=0,
        capture_profiles=True,
    )
    normal_by_wall = {item["wall"]: item for item in normal["walls"]}
    component_slot = {"tt": 2, "sphere": 3, "rr": 4, "tr": 5}
    for wall_number, wall_name in enumerate(("lower", "upper")):
        record = wall_junction_second_tangent(
            position, velocity, acceleration, z, r, background, wall_name, 7,
        )
        scale = (
            2.0*np.asarray(record["source"]["sqrt_gzz"])
            / float(record["orientation"])
        )
        for name, slot in component_slot.items():
            row_implied[wall_number, :, slot] = (
                np.asarray(record["components"][name]["normal_derivative_tt"])
                - scale*np.asarray(
                    record["components"][name]["DJ_acceleration"]
                )
            )
        row_implied[wall_number, :, 6] = (
            direct_physical[wall_number, :, 6]
            - np.asarray(
                normal_by_wall[wall_name]["profiles"]["residual"]
            )
        )
        separate = record["separate_rows"]
        row_implied[wall_number, :, 7] = (
            direct_physical[wall_number, :, 7]
            - np.asarray(separate["DJ_Phi_robin_acceleration"])
        )
        row_implied[wall_number, :, 8] = (
            direct_physical[wall_number, :, 8]
            - np.asarray(separate["DJ_chi_neumann_acceleration"])
        )
    row_defined = np.asarray(
        [False, False, True, True, True, True, True, True, True],
        dtype=np.uint8,
    )
    scored = row_defined.astype(bool)[None, None, :]
    denominator = np.maximum.reduce((
        np.ones_like(direct_physical), np.abs(direct_physical),
        np.abs(row_implied),
    ))
    scaled_difference = np.where(
        scored, np.abs(direct_physical-row_implied)/denominator, 0.0,
    )
    return {
        "direct_reduced": direct_reduced,
        "direct_physical": direct_physical,
        "row_implied_physical": row_implied,
        "row_defined_mask": row_defined,
        "maximum_scaled_difference": float(np.max(scaled_difference)),
        "normal_profiles": normal,
    }


def _axis_null_fit_record(before, after, r, window=0.5, degree=3):
    r = np.asarray(r, dtype=float)
    keep = np.flatnonzero((r > 0.0) & (r <= float(window)+1e-12))
    if len(keep) < int(degree)+1:
        raise ValueError("axis null-channel fit has too few points")
    coordinate = r[keep]**2
    coefficients = np.empty((before.shape[0], 2, int(degree)+1))
    for z_index in range(before.shape[0]):
        for local, field in enumerate((4, 5)):
            coefficients[z_index, local] = np.polynomial.polynomial.polyfit(
                coordinate, before[z_index, keep, field], int(degree),
            )
    preferred = coefficients[:, :, 0]
    pre_defect = before[:, 0, 4:6]-preferred
    post_defect = after[:, 0, 4:6]-preferred
    return {
        "indices": keep,
        "coefficients": coefficients,
        "pre_defect": pre_defect,
        "post_defect": post_defect,
        "window": float(window),
        "degree": int(degree),
    }


def _substep_boundary_copies(bulk, coupled, physical, gauge, compatible):
    """Return distinct pre/post arrays for every frozen construction substep."""
    sources = {
        "a_pre_coupled": bulk,
        "a_post_coupled": coupled,
        "a_pre_physical": coupled,
        "a_post_physical": physical,
        "a_pre_gauge": physical,
        "a_post_gauge": gauge,
        "a_pre_reconciliation": gauge,
        "a_post_reconciliation": compatible,
    }
    return {name: np.asarray(value).copy() for name, value in sources.items()}


def _write_coupled_gate_failure(parent, source_path, error):
    """Persist the required FAIL classification when the 4x4 gate rejects."""
    result = {
        "schema": "A790-phase-a2-parent-compatibility-result-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "environment_sha256": environment_fingerprint(),
        "parent_sha256": PARENT_SHA256,
        "phase_a_sha256": PHASE_A_SHA256,
        "decomposition_sha256": DECOMPOSITION_SHA256,
        "source_artifact": str(source_path),
        "source_artifact_sha256": sha256_file(source_path),
        "correction_artifact": None,
        "correction_artifact_sha256": None,
        "classification": "FAIL-parent-acceleration",
        "failure_stage": "coupled_Phi_gzz_algebraic_gate",
        "failure": f"{type(error).__name__}: {error}",
        "coupled_gate_failure": {
            "gate": error.gate,
            "radial_index": error.radial_index,
            "radius": float(parent["r"][error.radial_index]),
            "diagnostics": _json_safe(error.diagnostics),
        },
        "projection_authorized": False,
        "target_projection_authorized": False,
        "phase_b_authorized": False,
        "evolution_RHS_or_RK_called": False,
        "source_driver_rhs_used_for_frozen_source_reconstruction": True,
        "full_matrix_authorized": False,
        "new_interface_physics_authorized": False,
        "known_independent_stops": {
            "representation": "failed in immutable Phase-A2 decomposition",
            "balanced_bulk_constraint": (
                "failed in immutable Phase-A2 decomposition"
            ),
            "acceleration_correction_cannot_change_these": True,
        },
        "gates": {
            "coupled_block_passes": False,
            "all_structural_gates_pass": False,
            "all_smallness_gates_pass": False,
        },
        "scope": (
            "classified algebraic stop before a compatible-acceleration "
            "artifact; no downstream wall solve or audit was attempted"
        ),
        "parent_grid": [len(parent["z"]), len(parent["r"])],
        "environment": public_environment(),
    }
    atomic_write_json(RESULT, _json_safe(result))
    return result


def _correction_stage(index, parent, source_path):
    stage_id = "phase_a2/native_P11_correction"
    metadata = {
        "protocol_sha256": PROTOCOL_SHA256,
        "environment_sha256": environment_fingerprint(),
        "parent_sha256": PARENT_SHA256,
        "source_sha256": sha256_file(source_path),
        "ordering": (
            "coupled_Phi_gzz -> physical_wall_rows -> normal_tangential_gauge "
            "-> selective_q4_q5_axis"
        ),
        "outer_overwrite": False,
    }
    index.register(stage_id, "native-P11-correction", 600.0, metadata)
    cached = index.validated_path(stage_id)
    shape = parent["q"].shape
    required = {
        "z": parent["z"].shape, "r": parent["r"].shape,
        "a_bulk": shape, "a_coupled": shape, "a_physical": shape,
        "a_gauge": shape,
        "a_compatible": shape, "delta_a": shape,
        **{name: shape for name in (
            "a_pre_coupled", "a_post_coupled", "a_pre_physical",
            "a_post_physical", "a_pre_gauge", "a_post_gauge",
            "a_pre_reconciliation", "a_post_reconciliation",
        )},
        "a_z_endpoints": (2, len(parent["r"]), shape[-1]),
        "a_z_physical_endpoints": (2, len(parent["r"]), shape[-1]),
        "a_z_row_implied_physical_endpoints": (
            2, len(parent["r"]), shape[-1],
        ),
        "a_z_row_defined_mask": (shape[-1],),
        "source": shape[:2]+(3,), "source_time": shape[:2]+(3,),
        "source_second_time": shape[:2]+(3,),
        "memory": shape[:2]+(3,),
        "axis_fit_coefficients": (shape[0], 2, 4),
        "axis_fit_pre_defect": (shape[0], 2),
        "axis_fit_post_defect": (shape[0], 2),
        **{name: shape for name in (
            "mask_actual_coupled", "mask_actual_physical",
            "mask_actual_gauge", "mask_actual_reconciliation",
            "mask_actual_final", "mask_allowed_coupled",
            "mask_numerical_coupled", "mask_numerical_physical",
            "mask_numerical_gauge", "mask_numerical_reconciliation",
            "mask_numerical_final",
            "mask_allowed_physical", "mask_allowed_gauge",
            "mask_allowed_reconciliation", "mask_allowed_final",
        )},
    }
    embedded = {
        "schema": "A790-phase-a2-native-P11-compatible-acceleration-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "parent_sha256": PARENT_SHA256,
        "source_sha256": sha256_file(source_path),
        "environment_sha256": environment_fingerprint(),
    }
    if cached is not None:
        if Path(cached).suffix == ".json":
            result = json.loads(Path(cached).read_text())
            if not (
                result.get("protocol_sha256") == PROTOCOL_SHA256
                and result.get("classification") == "FAIL-parent-acceleration"
                and result.get("failure_stage")
                == "coupled_Phi_gzz_algebraic_gate"
                and result.get("source_artifact_sha256")
                == sha256_file(source_path)
            ):
                raise RuntimeError(
                    "cached terminal Phase-A2 scientific result is invalid"
                )
            raise PhaseA2ScientificGateFailure(result)
        _validate_npz(cached, required, embedded)
        return cached
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        with np.load(source_path, allow_pickle=False) as source_archive:
            source = np.asarray(source_archive["source"])
            source_time = np.asarray(source_archive["source_time"])
            source_second = np.asarray(source_archive["source_second_time"])
            memory = np.asarray(source_archive["memory"])
        q = np.asarray(parent["q"])
        v = np.asarray(parent["first"][0])
        a_bulk = np.asarray(parent["second"][0, 0]).copy()
        try:
            a_coupled, coupled_record = (
                solve_compact_wall_coupled_phi_normal_acceleration(
                    q, v, a_bulk, source, source_time, source_second,
                    parent["z"], parent["r"], parent["background"], 7,
                    capture_profiles=True,
                )
            )
        except CompactWallCoupledAlgebraicGateError as error:
            result = _write_coupled_gate_failure(parent, source_path, error)
            raise PhaseA2ScientificGateFailure(result) from error
        normal = np.stack((a_coupled[0, :, 6], a_coupled[-1, :, 6]))
        a_physical, wall_corrections = apply_compact_wall_acceleration(
            q, v, a_coupled, parent["z"], parent["r"],
            parent["background"], normal, 7, fill_axis_after=False,
            impose_normal_tangential=False,
        )
        a_gauge = impose_compact_wall_normal_tangential_acceleration(
            a_physical,
        )
        a_compatible = reconcile_wall_owner_axis_null_channels(
            a_gauge, parent["r"], window=0.5, degree=3,
        )
        endpoint_derivative = _row_implied_physical_endpoint_derivative(
            q, v, a_compatible, source, source_time, source_second,
            parent["z"], parent["r"], parent["background"],
        )
        a_z_endpoints = endpoint_derivative["direct_reduced"]
        support = _support_audit(
            a_bulk, a_coupled, a_physical, a_gauge, a_compatible,
        )
        support_masks = _support_masks(
            a_bulk, a_coupled, a_physical, a_gauge, a_compatible,
        )
        substep_copies = _substep_boundary_copies(
            a_bulk, a_coupled, a_physical, a_gauge, a_compatible,
        )
        axis_fit = _axis_null_fit_record(
            a_gauge, a_compatible, parent["r"], 0.5, 3,
        )
        invariance = _wall_reconciliation_invariance(
            q, v, a_gauge, a_compatible, parent["z"], parent["r"],
            parent["background"],
        )
        stage_second_wall = _stage_second_wall_audits(parent, {
            "bulk": a_bulk,
            "coupled": a_coupled,
            "physical": a_physical,
            "gauge": a_gauge,
            "compatible": a_compatible,
        })
        coupled_json = json.dumps(_json_safe(coupled_record), sort_keys=True)
        wall_json = json.dumps(_json_safe(wall_corrections), sort_keys=True)
        support_json = json.dumps(_json_safe(support), sort_keys=True)
        invariance_json = json.dumps(_json_safe(invariance), sort_keys=True)
        normal_json = json.dumps(
            _json_safe(endpoint_derivative["normal_profiles"]), sort_keys=True,
        )
        stage_second_wall_json = json.dumps(
            _json_safe(stage_second_wall), sort_keys=True,
        )
        atomic_write_npz(
            CORRECTION_ARTIFACT,
            schema=np.asarray(embedded["schema"]),
            protocol_sha256=np.asarray(PROTOCOL_SHA256),
            parent_sha256=np.asarray(PARENT_SHA256),
            source_sha256=np.asarray(sha256_file(source_path)),
            environment_sha256=np.asarray(environment_fingerprint()),
            z=parent["z"], r=parent["r"],
            a_bulk=a_bulk, a_coupled=np.asarray(a_coupled).copy(),
            a_physical=np.asarray(a_physical).copy(),
            a_gauge=np.asarray(a_gauge).copy(),
            a_compatible=np.asarray(a_compatible).copy(),
            delta_a=np.asarray(a_compatible-a_bulk),
            **substep_copies,
            a_z_endpoints=a_z_endpoints,
            a_z_physical_endpoints=_physical_acceleration(
                a_z_endpoints, parent["r"],
            ),
            a_z_row_implied_physical_endpoints=endpoint_derivative[
                "row_implied_physical"
            ],
            a_z_row_defined_mask=endpoint_derivative["row_defined_mask"],
            a_z_row_maximum_scaled_difference=np.asarray(
                endpoint_derivative["maximum_scaled_difference"]
            ),
            source=source, source_time=source_time,
            source_second_time=source_second,
            memory=memory,
            axis_fit_indices=axis_fit["indices"],
            axis_fit_coefficients=axis_fit["coefficients"],
            axis_fit_pre_defect=axis_fit["pre_defect"],
            axis_fit_post_defect=axis_fit["post_defect"],
            axis_fit_window=np.asarray(axis_fit["window"]),
            axis_fit_degree=np.asarray(axis_fit["degree"]),
            **support_masks,
            coupled_record_json=np.asarray(coupled_json),
            wall_corrections_json=np.asarray(wall_json),
            support_json=np.asarray(support_json),
            reconciliation_invariance_json=np.asarray(invariance_json),
            normal_profiles_json=np.asarray(normal_json),
            stage_second_wall_json=np.asarray(stage_second_wall_json),
            q_fingerprint=np.asarray(hash_arrays(q)),
            v_fingerprint=np.asarray(hash_arrays(v)),
            a_bulk_fingerprint=np.asarray(hash_arrays(a_bulk)),
            a_coupled_fingerprint=np.asarray(hash_arrays(a_coupled)),
            a_physical_fingerprint=np.asarray(hash_arrays(a_physical)),
            a_gauge_fingerprint=np.asarray(hash_arrays(a_gauge)),
            a_compatible_fingerprint=np.asarray(hash_arrays(a_compatible)),
            **{
                f"{name}_fingerprint": np.asarray(hash_arrays(value))
                for name, value in substep_copies.items()
            },
            source_fingerprint=np.asarray(hash_arrays(
                source, source_time, source_second, memory,
            )),
            compatible_boundary_fingerprint=np.asarray(hash_arrays(
                parent["z"], parent["r"], a_compatible, a_z_endpoints,
                endpoint_derivative["row_implied_physical"], source,
                source_time, source_second, memory,
            )),
            compact_derivative_fingerprint=np.asarray(hash_arrays(
                derivative_matrix(parent["z"], 1, 7).toarray(),
            )),
        )
        _validate_npz(CORRECTION_ARTIFACT, required, embedded)
        index.mark_complete(
            stage_id, CORRECTION_ARTIFACT, time.perf_counter()-started,
            {
                "compatible_acceleration_fingerprint": hash_arrays(a_compatible),
                "coupled_passed": bool(coupled_record["passed"]),
            },
        )
        return CORRECTION_ARTIFACT
    except PhaseA2ScientificGateFailure as error:
        index.mark_complete(
            stage_id, RESULT, time.perf_counter()-started,
            {
                "classification": error.result["classification"],
                "terminal_scientific_result": True,
                "failure_stage": error.result["failure_stage"],
            },
        )
        raise
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def _wall_rows_pass(record):
    return bool(
        record["finite"]
        and all(
            record["walls"][wall][row]["normalized_Linf"] < 1e-10
            for wall in ("lower", "upper")
            for row in ("metric", "Phi", "chi")
        )
    )


def _audit_stage(index, parent, source_path, correction_path):
    stage_id = "phase_a2/audit"
    metadata = {
        "protocol_sha256": PROTOCOL_SHA256,
        "environment_sha256": environment_fingerprint(),
        "parent_sha256": PARENT_SHA256,
        "source_sha256": sha256_file(source_path),
        "correction_sha256": sha256_file(correction_path),
    }
    index.register(stage_id, "phase-A2-audit", 600.0, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        result = json.loads(cached.read_text())
        if (
            result.get("protocol_sha256") != PROTOCOL_SHA256
            or result.get("correction_artifact_sha256")
            != sha256_file(correction_path)
        ):
            raise RuntimeError("cached Phase-A2 result identity mismatch")
        return result
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        with np.load(correction_path, allow_pickle=False) as archive:
            a_bulk = np.asarray(archive["a_bulk"])
            a_coupled = np.asarray(archive["a_coupled"])
            a_physical = np.asarray(archive["a_physical"])
            a_gauge = np.asarray(archive["a_gauge"])
            a_compatible = np.asarray(archive["a_compatible"])
            delta_a = np.asarray(archive["delta_a"])
            substep_names = (
                "a_pre_coupled", "a_post_coupled", "a_pre_physical",
                "a_post_physical", "a_pre_gauge", "a_post_gauge",
                "a_pre_reconciliation", "a_post_reconciliation",
            )
            substep_copies = {
                name: np.asarray(archive[name]) for name in substep_names
            }
            a_z_endpoints = np.asarray(archive["a_z_endpoints"])
            a_z_physical = np.asarray(archive["a_z_physical_endpoints"])
            a_z_row_implied = np.asarray(
                archive["a_z_row_implied_physical_endpoints"]
            )
            a_z_row_defined_mask = np.asarray(
                archive["a_z_row_defined_mask"]
            )
            source = np.asarray(archive["source"])
            source_time = np.asarray(archive["source_time"])
            source_second = np.asarray(archive["source_second_time"])
            memory = np.asarray(archive["memory"])
            saved_axis_fit = {
                "indices": np.asarray(archive["axis_fit_indices"]),
                "coefficients": np.asarray(archive["axis_fit_coefficients"]),
                "pre_defect": np.asarray(archive["axis_fit_pre_defect"]),
                "post_defect": np.asarray(archive["axis_fit_post_defect"]),
                "window": float(archive["axis_fit_window"]),
                "degree": int(archive["axis_fit_degree"]),
            }
            coupled = json.loads(str(archive["coupled_record_json"]))
            wall_corrections = json.loads(str(archive["wall_corrections_json"]))
            stored_support = json.loads(str(archive["support_json"]))
            invariance = json.loads(str(
                archive["reconciliation_invariance_json"]
            ))
            stored_normal = json.loads(str(archive["normal_profiles_json"]))
            stage_second_wall = json.loads(str(
                archive["stage_second_wall_json"]
            ))
            stored_fingerprints = {
                name: str(archive[f"{name}_fingerprint"])
                for name in (
                    "q", "v", "a_bulk", "a_coupled", "a_physical",
                    "a_gauge", "a_compatible", "source", *substep_names,
                )
            }
            stored_boundary_fingerprint = str(
                archive["compatible_boundary_fingerprint"]
            )
            stored_derivative_fingerprint = str(
                archive["compact_derivative_fingerprint"]
            )
            stored_masks = {
                name: np.asarray(archive[name])
                for name in archive.files
                if name.startswith("mask_")
            }
        with np.load(source_path, allow_pickle=False) as source_archive:
            frozen_source = np.asarray(source_archive["source"])
            frozen_source_time = np.asarray(source_archive["source_time"])
            frozen_source_second = np.asarray(
                source_archive["source_second_time"]
            )
            frozen_memory = np.asarray(source_archive["memory"])
            frozen_q_fingerprint = str(source_archive["q_fingerprint"])
            frozen_v_fingerprint = str(source_archive["v_fingerprint"])
        q = np.asarray(parent["q"])
        v = np.asarray(parent["first"][0])
        parent_q_fingerprint = hash_arrays(q)
        parent_v_fingerprint = hash_arrays(v)
        support = _support_audit(
            a_bulk, a_coupled, a_physical, a_gauge, a_compatible,
        )
        recomputed_masks = _support_masks(
            a_bulk, a_coupled, a_physical, a_gauge, a_compatible,
        )
        expected_substep_copies = _substep_boundary_copies(
            a_bulk, a_coupled, a_physical, a_gauge, a_compatible,
        )
        recomputed_axis_fit = _axis_null_fit_record(
            a_gauge, a_compatible, parent["r"], 0.5, 3,
        )
        recomputed_stage_second_wall = _stage_second_wall_audits(parent, {
            "bulk": a_bulk,
            "coupled": a_coupled,
            "physical": a_physical,
            "gauge": a_gauge,
            "compatible": a_compatible,
        })
        second_wall = recomputed_stage_second_wall["compatible"]
        normal = compact_wall_normal_gauge_acceleration_residuals(
            q, v, a_compatible, source, source_time, source_second,
            parent["z"], parent["r"], parent["background"], 7,
            radial_buffer=0, capture_profiles=True,
        )
        direct_endpoint = _endpoint_derivative(a_compatible, parent["z"])
        direct_endpoint_physical = _physical_acceleration(
            direct_endpoint, parent["r"],
        )
        row_mask = a_z_row_defined_mask.astype(bool)[None, None, :]
        row_denominator = np.maximum.reduce((
            np.ones_like(a_z_physical), np.abs(a_z_physical),
            np.abs(a_z_row_implied),
        ))
        row_scaled_difference = np.where(
            row_mask,
            np.abs(a_z_physical-a_z_row_implied)/row_denominator,
            0.0,
        )
        endpoint_maximum_scaled_difference = float(np.max(
            row_scaled_difference
        ))
        stage_fingerprint_pass = bool(all((
            stored_fingerprints["a_bulk"] == hash_arrays(a_bulk),
            stored_fingerprints["a_coupled"] == hash_arrays(a_coupled),
            stored_fingerprints["a_physical"] == hash_arrays(a_physical),
            stored_fingerprints["a_gauge"] == hash_arrays(a_gauge),
            stored_fingerprints["a_compatible"] == hash_arrays(a_compatible),
        )))
        substep_copies_pass = bool(all(
            np.array_equal(substep_copies[name], expected)
            and stored_fingerprints[name] == hash_arrays(substep_copies[name])
            for name, expected in expected_substep_copies.items()
        ))
        delta_a_pass = bool(np.array_equal(
            delta_a, a_compatible-a_bulk,
        ))
        axis_fit_reopens_pass = bool(
            np.array_equal(
                saved_axis_fit["indices"], recomputed_axis_fit["indices"]
            )
            and np.array_equal(
                saved_axis_fit["coefficients"],
                recomputed_axis_fit["coefficients"],
            )
            and np.array_equal(
                saved_axis_fit["pre_defect"],
                recomputed_axis_fit["pre_defect"],
            )
            and np.array_equal(
                saved_axis_fit["post_defect"],
                recomputed_axis_fit["post_defect"],
            )
            and saved_axis_fit["window"] == recomputed_axis_fit["window"]
            and saved_axis_fit["degree"] == recomputed_axis_fit["degree"]
        )
        source_identity_pass = bool(all((
            hash_arrays(source) == hash_arrays(frozen_source),
            hash_arrays(source_time) == hash_arrays(frozen_source_time),
            hash_arrays(source_second) == hash_arrays(frozen_source_second),
            hash_arrays(memory) == hash_arrays(frozen_memory),
            stored_fingerprints["source"] == hash_arrays(
                source, source_time, source_second, memory,
            ),
        )))
        parent_identity_pass = bool(all((
            stored_fingerprints["q"] == parent_q_fingerprint,
            stored_fingerprints["v"] == parent_v_fingerprint,
            frozen_q_fingerprint == parent_q_fingerprint,
            frozen_v_fingerprint == parent_v_fingerprint,
            stored_fingerprints["a_bulk"] == hash_arrays(
                parent["second"][0, 0]
            ),
        )))
        support_summary_pass = bool(stored_support == support)
        support_masks_pass = bool(
            set(stored_masks) == set(recomputed_masks)
            and all(np.array_equal(stored_masks[name], value)
                    for name, value in recomputed_masks.items())
        )
        normal_profile_pass = bool(
            json.dumps(_json_safe(normal), sort_keys=True)
            == json.dumps(stored_normal, sort_keys=True)
        )
        stage_profile_pass = bool(
            _nested_finite(stage_second_wall)
            and json.dumps(stage_second_wall, sort_keys=True)
            == json.dumps(
                _json_safe(recomputed_stage_second_wall), sort_keys=True,
            )
        )
        boundary_fingerprint_pass = bool(
            stored_boundary_fingerprint == hash_arrays(
                parent["z"], parent["r"], a_compatible, a_z_endpoints,
                a_z_row_implied, source, source_time, source_second, memory,
            )
        )
        derivative_fingerprint_pass = bool(
            stored_derivative_fingerprint == hash_arrays(
                derivative_matrix(parent["z"], 1, 7).toarray(),
            )
        )
        endpoint_reopens_pass = bool(
            np.array_equal(a_z_endpoints, direct_endpoint)
            and np.array_equal(a_z_physical, direct_endpoint_physical)
        )
        row_mask_reopens_pass = bool(np.array_equal(
            a_z_row_defined_mask,
            np.asarray(
                [False, False, True, True, True, True, True, True, True],
                dtype=np.uint8,
            ),
        ))
        provenance_revalidation = {
            "parent_and_q_v_fingerprints": parent_identity_pass,
            "frozen_source_arrays": source_identity_pass,
            "stage_fingerprints": stage_fingerprint_pass,
            "separate_pre_post_substep_copies": substep_copies_pass,
            "delta_a_identity": delta_a_pass,
            "axis_fit_arrays": axis_fit_reopens_pass,
            "support_summary": support_summary_pass,
            "support_masks": support_masks_pass,
            "normal_profiles": normal_profile_pass,
            "all_stage_wall_profiles": stage_profile_pass,
            "endpoint_derivative_arrays": endpoint_reopens_pass,
            "row_defined_mask": row_mask_reopens_pass,
            "compatible_boundary_fingerprint": boundary_fingerprint_pass,
            "compact_derivative_fingerprint": derivative_fingerprint_pass,
        }
        _require_provenance_revalidation(provenance_revalidation)
        norms = correction_norms(
            q, a_bulk, a_compatible, parent["r"],
        )
        preferred = reconcile_wall_owner_axis_null_channels(
            a_compatible, parent["r"], window=0.5, degree=3,
        )
        even_fit = float(np.max(np.abs(
            preferred[:, 0, 4:6]-a_compatible[:, 0, 4:6]
        )))
        crossfit = axis_even_crossfit_audit(q, a_compatible, parent["r"])
        hessian_maximum = 0.0
        dx2_minus_dj_maximum = 0.0
        decomposition_maximum = 0.0
        for wall_name in ("lower", "upper"):
            record = wall_junction_second_tangent(
                q, v, a_compatible, parent["z"], parent["r"],
                parent["background"], wall_name, 7,
            )
            hessian_maximum = max(hessian_maximum, float(np.max(np.abs(
                record["D2J_velocity_velocity_tensor"]
            ))))
            hessian_maximum = max(
                hessian_maximum,
                float(np.max(np.abs(record["separate_rows"][
                    "D2_Phi_robin_velocity_velocity"
                ]))),
                float(np.max(np.abs(record["separate_rows"][
                    "D2_chi_neumann_velocity_velocity"
                ]))),
            )
            dx2_minus_dj_maximum = max(
                dx2_minus_dj_maximum,
                float(np.max(np.abs(
                    record["DX2J_tensor"]
                    - record["DJ_acceleration_tensor"]
                ))),
                float(np.max(np.abs(
                    record["separate_rows"]["DX2_Phi_robin"]
                    - record["separate_rows"][
                        "DJ_Phi_robin_acceleration"
                    ]
                ))),
                float(np.max(np.abs(
                    record["separate_rows"]["DX2_chi_neumann"]
                    - record["separate_rows"][
                        "DJ_chi_neumann_acceleration"
                    ]
                ))),
            )
            decomposition_maximum = max(
                decomposition_maximum,
                float(record["decomposition_maximum_absolute_defect"]),
            )
        finite = bool(all(np.all(np.isfinite(value)) for value in (
            a_bulk, a_coupled, a_physical, a_gauge, a_compatible,
            a_z_endpoints, a_z_physical, a_z_row_implied,
            source, source_time, source_second, memory,
        )) and all(_nested_finite(value) for value in (
            coupled, wall_corrections, support, invariance, stored_normal,
            stage_second_wall, normal, second_wall, norms, crossfit,
        )))
        ownership_pass = bool(all(support[name] for name in (
            "coupled_changes_only_wall_Phi_gzz",
            "physical_stage_changes_only_owned_wall_fields",
            "physical_stage_preserves_coupled_gzz_bitwise",
            "gauge_stage_changes_only_normal_tangential_wall_fields",
            "reconciliation_changes_only_q4_q5_axis",
            "final_changes_only_wall_or_q4_q5_axis",
            "stage_arrays_do_not_share_memory",
        )))
        algebraic_pass = bool(
            coupled.get("passed", False)
            and coupled.get("minimum_rank") == 4
            and coupled.get("maximum_condition", float("inf")) <= 1e12
            and coupled.get("minimum_pivot_strength", 0.0) >= 1e-10
            and coupled.get(
                "maximum_normalized_linear_residual", float("inf")
            ) < 1e-12
        )
        gates = {
            "q0_v0_bitwise_unchanged": True,
            "frozen_source_arrays_bitwise_unchanged": True,
            "gauge_q0_q1_are_exact_positive_zero": (
                _positive_zero_wall_gauge(a_gauge)
            ),
            "all_arrays_finite": finite,
            "coupled_block_passes": algebraic_pass,
            "metric_Phi_chi_second_wall_Linf_below_1e_10": (
                _wall_rows_pass(second_wall)
            ),
            "normal_GH_acceleration_Linf_below_1e_10": bool(
                normal["maximum"] < 1e-10
            ),
            "velocity_hessian_bitwise_zero": bool(hessian_maximum == 0.0),
            "DX2_minus_DJ_a_below_1e_12": bool(
                dx2_minus_dj_maximum <= 1e-12
            ),
            "DX2_decomposition_below_1e_12": bool(
                decomposition_maximum <= 1e-12
            ),
            "ownership_support_exact": ownership_pass,
            "q4_q5_even_fit_below_1e_12": bool(even_fit <= 1e-12),
            "reconciliation_preserves_physical_rows_to_1e_12": bool(
                invariance["unchanged_to_1e_12"]
            ),
            "physical_correction_Linf_at_most_5e_2": bool(
                norms["small_Linf_gate"]
            ),
            "proper_wall_weighted_RMS_at_most_1e_2": bool(
                norms["small_weighted_RMS_gate"]
            ),
        }
        structural_names = tuple(gates)[:-2]
        structural_pass = bool(all(gates[name] for name in structural_names))
        small_pass = bool(all(gates[name] for name in tuple(gates)[-2:]))
        classification = classify_correction(
            structural_pass, norms["global_normalized_Linf"],
            norms["combined_proper_wall_weighted_RMS"],
        )
        result = {
            "schema": "A790-phase-a2-parent-compatibility-result-v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "environment_sha256": environment_fingerprint(),
            "parent_sha256": PARENT_SHA256,
            "phase_a_sha256": PHASE_A_SHA256,
            "decomposition_sha256": DECOMPOSITION_SHA256,
            "source_artifact": str(source_path),
            "source_artifact_sha256": sha256_file(source_path),
            "correction_artifact": str(correction_path),
            "correction_artifact_sha256": sha256_file(correction_path),
            "classification": classification,
            "provenance_revalidation": provenance_revalidation,
            "projection_authorized": False,
            "target_projection_authorized": False,
            "phase_b_authorized": False,
            "evolution_RHS_or_RK_called": False,
            "source_driver_rhs_used_for_frozen_source_reconstruction": True,
            "full_matrix_authorized": False,
            "new_interface_physics_authorized": False,
            "known_independent_stops": {
                "representation": "failed in immutable Phase-A2 decomposition",
                "balanced_bulk_constraint": (
                    "failed in immutable Phase-A2 decomposition"
                ),
                "acceleration_correction_cannot_change_these": True,
            },
            "coupled_block": coupled,
            "wall_corrections": wall_corrections,
            "support": support,
            "reconciliation_invariance": invariance,
            "compatible_second_wall": second_wall,
            "all_stage_second_wall_profiles": stage_second_wall,
            "normal_GH_acceleration": normal,
            "correction_norms": norms,
            "axis_even_fit": {
                "preferred_q4_q5_defect": even_fit,
                "crossfit_observational": crossfit,
            },
            "compatible_endpoint_a_z": {
                "shape": list(a_z_endpoints.shape),
                "fingerprint": hash_arrays(a_z_endpoints),
                "finite": bool(np.all(np.isfinite(a_z_endpoints))),
                "physical_row_implied_fingerprint": hash_arrays(
                    a_z_row_implied
                ),
                "row_defined_component_mask": (
                    a_z_row_defined_mask.astype(bool).tolist()
                ),
                "maximum_scaled_direct_vs_row_implied": (
                    endpoint_maximum_scaled_difference
                ),
                "direct_vs_row_implied_status": "observational_not_a_gate",
                "normal_tangential_q0_q1_derivatives": (
                    "stencil-derived only; Dirichlet values are row-owned"
                ),
                "must_bind_future_acceleration_representation": True,
            },
            "decomposition_checks": {
                "velocity_hessian_Linf": hessian_maximum,
                "DX2_minus_DJ_a_Linf": dx2_minus_dj_maximum,
                "DX2_minus_DJ_a_minus_D2_vv_Linf": decomposition_maximum,
            },
            "gates": {
                **gates,
                "all_structural_gates_pass": structural_pass,
                "all_smallness_gates_pass": small_pass,
            },
            "environment": public_environment(),
        }
        atomic_write_json(RESULT, _json_safe(result))
        index.mark_complete(
            stage_id, RESULT, time.perf_counter()-started,
            {"classification": classification},
        )
        return result
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def run():
    validate_authorization()
    index = recovery_index()
    parent = _load_parent()
    source = _source_stage(index, parent)
    try:
        correction = _correction_stage(index, parent, source)
    except PhaseA2ScientificGateFailure as failure:
        result = failure.result
        print(json.dumps({
            "classification": result["classification"],
            "failure_stage": result["failure_stage"],
            "result": str(RESULT),
            "result_sha256": sha256_file(RESULT),
            "correction_artifact": None,
            "target_projection_authorized": False,
            "evolution_RHS_or_RK_called": False,
        }, indent=2, sort_keys=True))
        return
    result = _audit_stage(index, parent, source, correction)
    print(json.dumps({
        "classification": result["classification"],
        "result": str(RESULT),
        "result_sha256": sha256_file(RESULT),
        "correction_artifact": str(correction),
        "correction_artifact_sha256": sha256_file(correction),
        "target_projection_authorized": False,
        "evolution_RHS_or_RK_called": False,
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-p11-correction", action="store_true",
        help="run only the Protocol-123 native P11 correction diagnostic",
    )
    args = parser.parse_args()
    if not args.native_p11_correction:
        parser.error("no action selected; use --native-p11-correction")
    run()


if __name__ == "__main__":
    main()
