"""Archive-only residual/Jacobian diagnostics for Protocol 131.

Nothing in this module constructs, updates, or projects a parent.  Its public
entry points operate on the immutable Protocol-128 failure checkpoints and
return diagnostics for their already-recorded terminal states.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.linalg import LinAlgError
from scipy.sparse import csr_matrix
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import (
    ArpackError,
    ArpackNoConvergence,
    LinearOperator,
    eigsh,
    lsmr,
    lsqr,
    onenormest,
    splu,
    svds,
)

from bhps.anisotropic_initial_data import _raw_residual_and_jacobian
from bhps.gw_background import solve_gw_background
from bhps.joint_parent_builder import (
    _wall_primitives,
    joint_parent_residual,
    joint_parent_residual_and_jacobian,
)
from bhps.joint_parent_construction import (
    AMPLITUDE,
    COEFFICIENT_SHA256,
    KNOT_STATE,
    KNOT_STATE_SHA256,
    PARENT_SPECS,
    SHAPE_NORMALIZATION_SHA256,
    STENCIL_WIDTH,
    _construction_input_fingerprint,
    load_frozen_common_seed,
    validate_protocol125_construction_failure_record,
)
from bhps.joint_parent_environment_contract import (
    validate_protocol125_environment_contract,
)
from bhps.joint_parent_production_adapter import _unpack_roots
from bhps.joint_parent_scientific_runner import _reload_stage_payload
from bhps.joint_parent_shape import frozen_shape_fields_with_radial_derivative
from bhps.matched_staged_continuum import hash_arrays
from bhps.recovery_indexer import sha256_file
from bhps.scalar_pulse import scalar_pulse


PROTOCOL_IDENTIFIER = "Protocol-131-residual-jacobian-postmortem-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIRECTORY = (
    PROJECT_ROOT / "results/corrected_A790_joint_parent_rebuild_recovery_v2"
)
ARCHIVE_HASHES = {
    "parent_N0.npz": "47d0d29427606a06f676eba15a15dd6a1bef4a40b069a9a1aacd09c2b4481905",
    "parent_N1.npz": "d51f33b93d8b86976e87a4951f2afc43d38d8ddda6302feb14018953acad5ec6",
    "adjudication_final.json": "0c87d938573cc9013aba0d8865d48762730d1b051f457d6cbce7c1af4599bdf3",
    "recovery_index.json": "c488f6ea91166024d9bd13358c8209877610375ca019087243578aaee9af6269",
}
ARCHIVE_BYTE_COUNTS = {
    "parent_N0.npz": 1_696_871,
    "parent_N1.npz": 2_087_422,
    "adjudication_final.json": 1_897,
    "recovery_index.json": 20_755,
}
PARENT_LABELS = ("N0", "N1")
TARGET_CEILING = 1.0e-10
TARGET_FLOOR = 1.0e-11
FD_STEPS = tuple(2.0**-power for power in (8, 12, 16, 20))
FD_TOLERANCE = 1.0e-5
ITERATIVE_TOLERANCE = 1.0e-13
ITERATIVE_CONLIM = 1.0e15
ITERATIVE_MAXITER = 4096
SINGULAR_K = 8
SINGULAR_K_EXTENSION = 16
TOP_K = 8
_TINY = np.finfo(float).tiny


class Protocol131AuditError(RuntimeError):
    """A trust-gate failure for which no scientific classification is valid."""


def _float_bits(value):
    return np.asarray(float(value), dtype=np.float64).tobytes().hex()


def _csr_sha256(matrix):
    matrix = csr_matrix(matrix, dtype=float)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return hash_arrays(
        np.asarray(matrix.shape, dtype=np.int64),
        np.asarray(matrix.indptr, dtype=np.int64),
        np.asarray(matrix.indices, dtype=np.int64),
        np.asarray(matrix.data, dtype=np.float64),
    )


def _require_archive_bytes():
    for name, expected_hash in ARCHIVE_HASHES.items():
        path = ARCHIVE_DIRECTORY / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != ARCHIVE_BYTE_COUNTS[name]
            or sha256_file(path) != expected_hash
        ):
            raise Protocol131AuditError(f"immutable Protocol-128 artifact differs: {name}")
    recovery = json.loads(
        (ARCHIVE_DIRECTORY / "recovery_index.json").read_text(encoding="utf-8")
    )
    for label in PARENT_LABELS:
        stage = recovery.get("stages", {}).get(f"parent/{label}")
        name = f"parent_{label}.npz"
        if not isinstance(stage, Mapping) or not (
            stage.get("status") == "complete"
            and int(stage.get("byte_count", -1)) == ARCHIVE_BYTE_COUNTS[name]
            and str(stage.get("sha256", "")) == ARCHIVE_HASHES[name]
        ):
            raise Protocol131AuditError(f"recovery index does not bind parent/{label}")
    return recovery


def load_terminal_parent(label):
    """Load and exactly reconstruct one immutable Protocol-128 terminal state."""
    label = str(label)
    if label not in PARENT_LABELS:
        raise ValueError("parent label must be N0 or N1")
    _require_archive_bytes()
    validate_protocol125_environment_contract()
    path = ARCHIVE_DIRECTORY / f"parent_{label}.npz"
    checkpoint = _reload_stage_payload(
        path,
        expected_stage_id=f"parent/{label}",
        expected_kind="parent",
    )
    roots = _unpack_roots(checkpoint["arrays"], checkpoint["metadata"])
    if tuple(roots) != ("construction_failure",):
        raise Protocol131AuditError(f"{label} checkpoint roots differ")
    record = validate_protocol125_construction_failure_record(
        roots["construction_failure"]
    )
    if not (
        record["parent_label"] == label
        and record["failure_gate"] == "joint_hybrid_residual"
        and record["classification"] == "FAIL-parent-bulk"
        and record["complete"] is True
        and record["provenance_valid"] is True
        and record["retry_authorized"] is False
    ):
        raise Protocol131AuditError(f"{label} failure record semantics differ")

    payload = record["scientific_payload"]
    required = (
        "z", "r", "reference_q", "reference_phi", "selected_q",
        "selected_phi", "selected_psi", "selected_history",
        "selected_damping_history",
    )
    if any(name not in payload for name in required):
        raise Protocol131AuditError(f"{label} terminal payload is incomplete")
    z = np.asarray(payload["z"], dtype=float)
    r = np.asarray(payload["r"], dtype=float)
    q = np.asarray(payload["selected_q"], dtype=float)
    phi = np.asarray(payload["selected_phi"], dtype=float)
    reference_q = np.asarray(payload["reference_q"], dtype=float)
    reference_phi = np.asarray(payload["reference_phi"], dtype=float)
    shape = (len(z), len(r))
    if any(item.shape != shape for item in (q, phi, reference_q, reference_phi)):
        raise Protocol131AuditError(f"{label} terminal field shapes differ")
    if hash_arrays(z, r) != PARENT_SPECS[label]["coordinate_sha256"]:
        raise Protocol131AuditError(f"{label} coordinate fingerprint differs")

    seed = load_frozen_common_seed()
    expected_input = _construction_input_fingerprint(seed, label, PARENT_SPECS[label])
    if str(record["construction_input_fingerprint"]) != expected_input:
        raise Protocol131AuditError(f"{label} construction input fingerprint differs")
    if sha256_file(KNOT_STATE) != KNOT_STATE_SHA256:
        raise Protocol131AuditError("frozen family-knot artifact differs")
    with np.load(KNOT_STATE, allow_pickle=False) as archive:
        coefficients = np.asarray(archive["coefficients"], dtype=float)
    if hash_arrays(coefficients) != COEFFICIENT_SHA256:
        raise Protocol131AuditError("frozen shape coefficients differ")
    a, b, c, _, _, _, shape_record = frozen_shape_fields_with_radial_derivative(
        z, r, coefficients,
    )
    if shape_record["sha256"] != SHAPE_NORMALIZATION_SHA256:
        raise Protocol131AuditError("shape normalization differs")
    _, chi_r, chi_z = scalar_pulse(z, r, AMPLITUDE)
    background = solve_gw_background(
        z, 0.1, 0.01, wall_stiffness=20.0,
    )
    if not bool(background["converged"]):
        raise Protocol131AuditError(f"{label} deterministic background rebuild failed")
    rebuilt = {
        "label": label,
        "record": record,
        "z": z,
        "r": r,
        "q": q,
        "phi": phi,
        "reference_q": reference_q,
        "reference_phi": reference_phi,
        "a": np.asarray(a),
        "b": np.asarray(b),
        "c": np.asarray(c),
        "chi_r": np.asarray(chi_r),
        "chi_z": np.asarray(chi_z),
        "background": background,
        "generated_input_sha256": hash_arrays(
            a, b, c, chi_r, chi_z,
            np.asarray(background["psi"]),
            np.asarray(background["phi"]),
            np.asarray(background["phi_z"]),
        ),
    }
    return rebuilt


def _residual_arguments(parent, q=None, phi=None):
    return (
        parent["q"] if q is None else q,
        parent["phi"] if phi is None else phi,
        parent["z"], parent["r"], parent["a"], parent["b"], parent["c"],
        parent["background"], parent["chi_r"], parent["chi_z"],
        parent["reference_q"], parent["reference_phi"], STENCIL_WIDTH,
    )


def replay_residual_and_jacobian(parent):
    """Recompute F and J and enforce bitwise equality of archived F norms."""
    residual, jacobian = joint_parent_residual_and_jacobian(
        *_residual_arguments(parent)
    )
    residual = np.asarray(residual, dtype=float)
    jacobian = csr_matrix(jacobian, dtype=float)
    jacobian.sum_duplicates()
    jacobian.sort_indices()
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian.data)):
        raise Protocol131AuditError(f"{parent['label']} replay is nonfinite")
    maximum = float(np.max(np.abs(residual)))
    rms = float(np.sqrt(np.mean(residual**2)))
    record = parent["record"]
    archived_max = float(record["measured_value"])
    archived_rms = float(record["solver_diagnostics"]["selected_residual_l2"])
    if _float_bits(maximum) != _float_bits(archived_max):
        raise Protocol131AuditError(
            f"{parent['label']} maximum residual does not replay bit-for-bit"
        )
    if _float_bits(rms) != _float_bits(archived_rms):
        raise Protocol131AuditError(
            f"{parent['label']} RMS residual does not replay bit-for-bit"
        )
    return residual, jacobian, {
        "maximum": maximum,
        "maximum_ieee754_hex": _float_bits(maximum),
        "rms": rms,
        "rms_ieee754_hex": _float_bits(rms),
        "residual_sha256": hash_arrays(residual),
        "jacobian_csr_sha256": _csr_sha256(jacobian),
        "jacobian_shape": list(jacobian.shape),
        "jacobian_nnz": int(jacobian.nnz),
    }


def _zone_slices(size):
    if size < 15:
        raise ValueError("Protocol-131 localization requires at least 15 nodes")
    return (
        ("wall_or_axis", np.arange(0, 1)),
        ("lower_or_axis_collar", np.arange(1, 7)),
        ("open_or_face_interior", np.arange(7, size - 7)),
        ("upper_or_outer_collar", np.arange(size - 7, size - 1)),
        ("wall_or_outer_face", np.arange(size - 1, size)),
    )


def _top_entries(values, mask, z, r, count=TOP_K):
    flat = np.asarray(values).ravel(order="C")
    indices = np.flatnonzero(np.asarray(mask).ravel(order="C"))
    if len(indices) == 0:
        return []
    order = np.lexsort((indices, -np.abs(flat[indices])))[: int(count)]
    nr = len(r)
    result = []
    for flat_index in indices[order]:
        i, j = divmod(int(flat_index), nr)
        result.append({
            "flat_index": int(flat_index), "i": i, "j": j,
            "z": float(z[i]), "r": float(r[j]),
            "value": float(flat[flat_index]),
            "absolute_value": float(abs(flat[flat_index])),
        })
    return result


def _stats(values, mask, total_energy, z, r):
    array = np.asarray(values, dtype=float)
    selected = array[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        return {
            "count": 0, "signed_sum": 0.0, "L1": 0.0, "RMS": 0.0,
            "L2": 0.0, "Linf": 0.0, "L2_energy_fraction": 0.0,
            "argmax": None, "top": [],
        }
    squared = float(np.dot(selected, selected))
    top = _top_entries(array, mask, z, r)
    return {
        "count": int(selected.size),
        "signed_sum": float(np.sum(selected)),
        "L1": float(np.sum(np.abs(selected))),
        "RMS": float(np.sqrt(squared / selected.size)),
        "L2": float(np.sqrt(squared)),
        "Linf": float(np.max(np.abs(selected))),
        "L2_energy_fraction": float(squared / max(total_energy, _TINY)),
        "argmax": top[0] if top else None,
        "top": top,
    }


def residual_localization(parent, residual):
    """Return the prospectively fixed 2 x 5 x 5 residual decomposition."""
    nz, nr = parent["q"].shape
    blocks = np.asarray(residual).reshape(2, nz, nr)
    global_energy = float(np.sum(blocks**2))
    z_zones = _zone_slices(nz)
    r_zones = _zone_slices(nr)
    atoms = {}
    block_summaries = {}
    binned = np.zeros((2, 16, 16), dtype=float)
    u = (parent["z"] - 1.0) / (math.e - 1.0)
    v = parent["r"] / 12.0
    ubin = np.minimum(15, np.maximum(0, np.floor(16 * u).astype(int)))
    vbin = np.minimum(15, np.maximum(0, np.floor(16 * v).astype(int)))
    for block_index, block_name in enumerate(("metric", "Phi")):
        values = blocks[block_index]
        total_energy = global_energy
        block_summaries[block_name] = _stats(
            values, np.ones_like(values, dtype=bool), total_energy,
            parent["z"], parent["r"],
        )
        for z_name, z_indices in z_zones:
            for r_name, r_indices in r_zones:
                mask = np.zeros((nz, nr), dtype=bool)
                mask[np.ix_(z_indices, r_indices)] = True
                name = f"{block_name}/{z_name}/{r_name}"
                atoms[name] = _stats(
                    values, mask, total_energy, parent["z"], parent["r"],
                )
        for i in range(nz):
            for j in range(nr):
                binned[block_index, ubin[i], vbin[j]] += values[i, j] ** 2
    total = float(np.sum(binned))
    if total > 0.0:
        binned /= total
    dominant = max(atoms, key=lambda name: (atoms[name]["Linf"], name))
    return {
        "atoms": atoms,
        "blocks": block_summaries,
        "dominant_atom_by_Linf": dominant,
        "binned_energy_16x16": binned,
        "wall_even": (blocks[:, 0, :] + blocks[:, -1, :]) / math.sqrt(2.0),
        "wall_odd": (blocks[:, 0, :] - blocks[:, -1, :]) / math.sqrt(2.0),
    }


def residual_cancellation_terms(parent, residual):
    """Expose final raw/defect subtraction and wall numerator cancellation."""
    args = _residual_arguments(parent)
    q, phi, z, r, a, b, c, background, chi_r, chi_z, rq, rp, width = args
    raw = _raw_residual_and_jacobian(
        q, phi, z, r, a, b, c, background, chi_r, chi_z, rq, rp, width, False,
    )
    zero = np.zeros_like(a)
    defect = _raw_residual_and_jacobian(
        rq, rp, z, r, zero, zero, zero, background, chi_r, chi_z,
        rq, rp, width, False,
    )
    raw = np.asarray(raw).reshape(2, len(z), len(r))
    defect = np.asarray(defect).reshape(2, len(z), len(r))
    balanced = raw - defect
    final = np.asarray(residual).reshape(2, len(z), len(r))
    wall = _wall_primitives(q, phi, z, a, c, background, width)
    wall_terms = {}
    eps = np.finfo(float).eps
    for item in wall["walls"]:
        index = int(item["index"])
        source = item["source"]
        orientation = float(source["orientation"])
        scale = wall["normal_scale"][index]
        sphere = wall["sphere_metric"][index]
        derivative_metric = (wall["Dz"] @ wall["sphere_metric"])[index]
        source_metric = 2.0 * np.asarray(source["beta"]) * scale * sphere
        derivative_phi = (wall["Dz"] @ phi)[index]
        source_phi = (
            orientation * 0.5 * float(background["wall_stiffness"])
            * (phi[index] - float(source["target"])) * scale
        )
        metric_normalized = np.asarray(
            orientation * (derivative_metric + source_metric) / (2.0 * scale)
        )
        phi_numerator = np.asarray(derivative_phi + source_phi)
        wall_terms[item["name"]] = {
            "normal_scale": np.asarray(scale),
            "metric_denominator": np.asarray(2.0*scale),
            "metric_derivative": np.asarray(derivative_metric),
            "metric_source": np.asarray(source_metric),
            "metric_numerator": np.asarray(derivative_metric + source_metric),
            "metric_normalized": metric_normalized,
            "metric_roundoff_bound": np.asarray(
                eps * (
                    (np.abs(derivative_metric) + np.abs(source_metric))
                    / (2.0*np.abs(scale))
                    + np.abs(metric_normalized)
                )
            ),
            "Phi_derivative": np.asarray(derivative_phi),
            "Phi_source": np.asarray(source_phi),
            "Phi_numerator": phi_numerator,
            "Phi_roundoff_bound": np.asarray(
                eps * (
                    np.abs(derivative_phi) + np.abs(source_phi)
                    + np.abs(phi_numerator)
                )
            ),
        }
    subtraction_bound = eps * (np.abs(raw) + np.abs(defect))
    final_bound = subtraction_bound.copy()
    for item in wall["walls"]:
        index = int(item["index"])
        terms = wall_terms[item["name"]]
        final_bound[0, index] = terms["metric_roundoff_bound"]
        final_bound[1, index] = terms["Phi_roundoff_bound"]
    return {
        "raw": raw,
        "reference_defect": defect,
        "balanced_before_wall_override": balanced,
        "final": final,
        "final_subtraction_roundoff_bound": subtraction_bound,
        "final_row_roundoff_bound": final_bound,
        "wall_terms": wall_terms,
    }


def _direction_masks(parent):
    nz, nr = parent["q"].shape
    n = nz * nr
    full = np.ones((2, nz, nr), dtype=bool)
    q_only = np.zeros_like(full); q_only[0] = True
    phi_only = np.zeros_like(full); phi_only[1] = True
    lower = np.zeros_like(full); lower[:, 0, :] = True
    upper = np.zeros_like(full); upper[:, -1, :] = True
    axis = np.zeros_like(full); axis[:, :, 0] = True
    outer = np.zeros_like(full); outer[:, :, -1] = True
    return {
        "mixed_global_0": full, "mixed_global_1": full,
        "mixed_global_2": full, "mixed_global_3": full,
        "q_only_0": q_only, "q_only_1": q_only,
        "Phi_only_0": phi_only, "Phi_only_1": phi_only,
        "lower_wall": lower, "upper_wall": upper,
        "axis": axis, "outer_face": outer,
    }


def _fixed_direction(parent, name, mask):
    x = np.concatenate((parent["q"].ravel(), parent["phi"].ravel()))
    seed = int.from_bytes(
        hashlib.sha256(f"{PROTOCOL_IDENTIFIER}/{parent['label']}/{name}".encode()).digest()[:8],
        "little",
    )
    rng = np.random.default_rng(seed)
    sign = rng.choice(np.asarray((-1.0, 1.0)), size=x.size)
    direction = sign * np.maximum(1.0, np.abs(x)) * mask.ravel(order="C")
    maximum = float(np.max(np.abs(direction)))
    if maximum == 0.0:
        raise Protocol131AuditError(f"empty directional support: {name}")
    return direction / maximum


def audit_analytic_jacobian(parent, residual, jacobian):
    """Run the frozen twelve-direction centered-difference audit."""
    n = parent["q"].size
    directions = {}
    all_passed = True
    for name, mask in _direction_masks(parent).items():
        direction = _fixed_direction(parent, name, mask)
        dq = direction[:n].reshape(parent["q"].shape)
        dphi = direction[n:].reshape(parent["phi"].shape)
        analytic = np.asarray(jacobian @ direction)
        if not np.all(np.isfinite(analytic)):
            directions[name] = {
                "passed": False, "samples": [],
                "direction_sha256": hash_arrays(direction),
                "failure": "analytic product is nonfinite",
            }
            all_passed = False
            continue
        records = []
        for step in FD_STEPS:
            plus_q = parent["q"] + step * dq
            minus_q = parent["q"] - step * dq
            plus_phi = parent["phi"] + step*dphi
            minus_phi = parent["phi"] - step*dphi
            finite_states = bool(
                np.all(np.isfinite(plus_q))
                and np.all(np.isfinite(minus_q))
                and np.all(np.isfinite(plus_phi))
                and np.all(np.isfinite(minus_phi))
            )
            valid = bool(
                finite_states
                and np.min(parent["z"][:, None] + plus_q) > 0.0
                and np.min(parent["z"][:, None] + minus_q) > 0.0
            )
            if not valid:
                records.append({
                    "step": step,
                    "valid": False,
                    "failure": (
                        "nonfinite perturbed state"
                        if not finite_states else "nonpositive z+q"
                    ),
                })
                continue
            plus = joint_parent_residual(
                *_residual_arguments(parent, plus_q, plus_phi)
            )
            minus = joint_parent_residual(
                *_residual_arguments(parent, minus_q, minus_phi)
            )
            if not (
                np.all(np.isfinite(plus)) and np.all(np.isfinite(minus))
            ):
                records.append({"step": step, "valid": False, "failure": "nonfinite residual"})
                continue
            finite_difference = (np.asarray(plus) - np.asarray(minus)) / (2.0*step)
            delta = finite_difference - analytic
            l2 = float(np.linalg.norm(delta) / max(np.linalg.norm(finite_difference), _TINY))
            linf = float(
                np.max(np.abs(delta)) / max(np.max(np.abs(finite_difference)), _TINY)
            )
            records.append({
                "step": step, "valid": True, "relative_L2": l2,
                "relative_Linf": linf,
            })
        passed = bool(
            all(item.get("valid") for item in records)
            and any(
                item["relative_L2"] <= FD_TOLERANCE
                and item["relative_Linf"] <= FD_TOLERANCE
                for item in records
            )
        )
        all_passed = all_passed and passed
        directions[name] = {
            "passed": bool(passed), "samples": records,
            "direction_sha256": hash_arrays(direction),
        }
    return {"passed": bool(all_passed), "directions": directions}


def right_power_two_scaling(jacobian):
    squared = np.asarray(jacobian.power(2).sum(axis=0)).ravel()
    norms = np.sqrt(squared)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0.0):
        raise Protocol131AuditError("Jacobian has a zero or nonfinite column")
    exponents = -np.rint(np.log2(norms)).astype(np.int64)
    scaling = np.ldexp(np.ones_like(norms), exponents)
    scaled = csr_matrix(jacobian.multiply(scaling[None, :]))
    scaled.sum_duplicates(); scaled.sort_indices()
    return scaled, scaling, exponents


def power_two_ruiz(matrix, sweeps=4):
    """Fixed power-of-two Ruiz control; returns transformed matrix and scales."""
    transformed = csr_matrix(matrix, dtype=float).copy()
    row_scale = np.ones(transformed.shape[0])
    column_scale = np.ones(transformed.shape[1])
    records = []
    for _ in range(int(sweeps)):
        row_norm = np.sqrt(np.asarray(transformed.power(2).sum(axis=1)).ravel())
        if np.any(row_norm == 0.0) or np.any(~np.isfinite(row_norm)):
            raise Protocol131AuditError("Ruiz control has a zero/nonfinite row")
        row_factor = np.ldexp(np.ones_like(row_norm), -np.rint(np.log2(row_norm)).astype(int))
        transformed = csr_matrix(transformed.multiply(row_factor[:, None]))
        row_scale *= row_factor
        column_norm = np.sqrt(np.asarray(transformed.power(2).sum(axis=0)).ravel())
        if np.any(column_norm == 0.0) or np.any(~np.isfinite(column_norm)):
            raise Protocol131AuditError("Ruiz control has a zero/nonfinite column")
        column_factor = np.ldexp(
            np.ones_like(column_norm), -np.rint(np.log2(column_norm)).astype(int)
        )
        transformed = csr_matrix(transformed.multiply(column_factor[None, :]))
        column_scale *= column_factor
        records.append({
            "row_norm_min": float(np.min(row_norm)),
            "row_norm_max": float(np.max(row_norm)),
            "column_norm_min": float(np.min(column_norm)),
            "column_norm_max": float(np.max(column_norm)),
        })
    transformed.sum_duplicates(); transformed.sort_indices()
    return transformed, row_scale, column_scale, records


def _projection_record(matrix, rhs, method):
    if method == "lsmr":
        result = lsmr(
            matrix, -rhs, atol=ITERATIVE_TOLERANCE, btol=ITERATIVE_TOLERANCE,
            conlim=ITERATIVE_CONLIM, maxiter=ITERATIVE_MAXITER,
        )
        solution, istop, iterations = result[0], int(result[1]), int(result[2])
        solver_record = {
            "istop": istop, "iterations": iterations,
            "reported_normr": float(result[3]),
            "reported_normar": float(result[4]),
            "reported_norma": float(result[5]),
            "reported_conda": float(result[6]),
            "reported_normx": float(result[7]),
        }
    elif method == "lsqr":
        result = lsqr(
            matrix, -rhs, atol=ITERATIVE_TOLERANCE, btol=ITERATIVE_TOLERANCE,
            conlim=ITERATIVE_CONLIM, iter_lim=ITERATIVE_MAXITER,
            show=False,
        )
        solution, istop, iterations = result[0], int(result[1]), int(result[2])
        solver_record = {
            "istop": istop, "iterations": iterations,
            "reported_r1norm": float(result[3]),
            "reported_r2norm": float(result[4]),
            "reported_anorm": float(result[5]),
            "reported_acond": float(result[6]),
            "reported_arnorm": float(result[7]),
            "reported_xnorm": float(result[8]),
        }
    else:
        raise ValueError("unknown projection method")
    projected = np.asarray(rhs + matrix @ solution)
    return solution, projected, {
        **solver_record,
        "projected_L2": float(np.linalg.norm(projected)),
        "projected_Linf": float(np.max(np.abs(projected))),
        "relative_projected_L2": float(
            np.linalg.norm(projected) / max(np.linalg.norm(rhs), _TINY)
        ),
        "relative_projected_Linf": float(
            np.max(np.abs(projected)) / max(np.max(np.abs(rhs)), _TINY)
        ),
    }


def _largest_singular_value(matrix, seed):
    value = svds(
        matrix, k=1, which="LM", return_singular_vectors=False,
        tol=1.0e-8, maxiter=500, rng=np.random.default_rng(seed),
    )
    maximum = float(np.max(np.abs(value)))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise Protocol131AuditError(
            "largest-singular-value lane returned a nonpositive or nonfinite value"
        )
    return maximum


def _inverse_singular_modes(matrix, lu, sigma_max, seed, k=SINGULAR_K):
    size = matrix.shape[0]
    inverse_normal = LinearOperator(
        (size, size), dtype=np.dtype(float),
        matvec=lambda vector: lu.solve(lu.solve(np.asarray(vector), trans="N"), trans="T"),
        rmatvec=lambda vector: lu.solve(lu.solve(np.asarray(vector), trans="N"), trans="T"),
    )
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=size)
    v0 /= np.linalg.norm(v0)
    eigenvalues, left_vectors = eigsh(
        inverse_normal, k=int(k), ncv=max(32, 2*int(k)+1), which="LA",
        tol=1.0e-8, maxiter=300, v0=v0,
    )
    order = np.argsort(eigenvalues)[::-1]
    records = []
    vectors = []
    for index in order:
        mu = float(eigenvalues[index])
        if not np.isfinite(mu) or mu <= 0.0:
            raise Protocol131AuditError("inverse Lanczos returned an invalid eigenvalue")
        sigma = float(1.0 / math.sqrt(mu))
        u = np.asarray(left_vectors[:, index], dtype=float)
        u /= np.linalg.norm(u)
        right_raw = np.asarray(matrix.T @ u)
        right_norm = np.linalg.norm(right_raw)
        if right_norm == 0.0:
            v = np.zeros_like(u)
        else:
            v = right_raw / right_norm
        left_defect = float(np.linalg.norm(matrix @ v - sigma*u) / max(sigma_max, _TINY))
        right_defect = float(np.linalg.norm(matrix.T @ u - sigma*v) / max(sigma_max, _TINY))
        defect = max(left_defect, right_defect)
        relative = sigma / sigma_max
        records.append({
            "sigma": sigma, "relative_sigma": relative,
            "left_triplet_defect": left_defect,
            "right_triplet_defect": right_defect,
            "relative_interval_lower": max(0.0, relative-defect),
            "relative_interval_upper": relative+defect,
        })
        vectors.append((u, v))
    sorting = np.argsort([record["sigma"] for record in records])
    return [records[i] for i in sorting], [vectors[i] for i in sorting]


def _propack_singular_modes(matrix, sigma_max, seed, k=SINGULAR_K):
    u, singular, vt = svds(
        matrix, k=int(k), which="SM", solver="propack", tol=1.0e-10,
        maxiter=2000, rng=np.random.default_rng(seed),
    )
    order = np.argsort(singular)
    records, vectors = [], []
    for index in order:
        sigma = float(singular[index])
        left = np.asarray(u[:, index]); left /= np.linalg.norm(left)
        right = np.asarray(vt[index]); right /= np.linalg.norm(right)
        ld = float(np.linalg.norm(matrix @ right - sigma*left) / max(sigma_max, _TINY))
        rd = float(np.linalg.norm(matrix.T @ left - sigma*right) / max(sigma_max, _TINY))
        defect = max(ld, rd); relative = sigma / sigma_max
        records.append({
            "sigma": sigma, "relative_sigma": relative,
            "left_triplet_defect": ld, "right_triplet_defect": rd,
            "relative_interval_lower": max(0.0, relative-defect),
            "relative_interval_upper": relative+defect,
        })
        vectors.append((left, right))
    return records, vectors


def _annotate_singular_modes(
    mode_records, mode_vectors, *, residual, tau_rank, q_size,
):
    for record, (left, right) in zip(mode_records, mode_vectors):
        record["definitely_null"] = bool(
            record["relative_interval_upper"] <= tau_rank
        )
        record["definitely_nonnull"] = bool(
            record["relative_interval_lower"] > tau_rank
        )
        pairing = float(np.dot(left, residual))
        record["left_pairing"] = pairing
        record["required_modal_correction"] = float(
            abs(pairing) / max(record["sigma"], _TINY)
        )
        record["left_L1"] = float(np.sum(np.abs(left)))
        record["q_right_fraction"] = float(
            np.dot(right[:q_size], right[:q_size])
        )
        record["Phi_right_fraction"] = float(
            np.dot(right[q_size:], right[q_size:])
        )
    return mode_records


def linear_range_analysis(parent, residual, jacobian):
    """Run the fixed primary range, condition, spectrum, and Ruiz lanes."""
    matrix, column_scale, exponents = right_power_two_scaling(jacobian)
    seed = int.from_bytes(
        hashlib.sha256(f"{PROTOCOL_IDENTIFIER}/{parent['label']}/linear".encode()).digest()[:8],
        "little",
    )
    tau_rank = matrix.shape[0] * np.finfo(float).eps
    try:
        sigma_max = _largest_singular_value(matrix, seed)
    except (
        ArpackError, ArpackNoConvergence, LinAlgError, RuntimeError,
        Protocol131AuditError,
    ) as error:
        return {
            "analysis_complete": False,
            "failure_stage": "largest-singular-value",
            "matrix": matrix,
            "column_scale": column_scale,
            "column_exponents": exponents,
            "sigma_max": None,
            "tau_rank": tau_rank,
            "lu": {"succeeded": False, "not_reached": True},
            "lsmr": {"not_reached": True},
            "lsqr": {"not_reached": True},
            "projection_accepted": False,
            "projection_numerically_zero_floor": False,
            "mode_method": "not-reached",
            "spectrum_certified": False,
            "spectrum_attempts": [],
            "spectrum_errors": [
                f"sigma-max: {type(error).__name__}: {error}"
            ],
            "modes": [],
            "mode_left_vectors": np.empty((matrix.shape[0], 0)),
            "mode_right_vectors": np.empty((matrix.shape[1], 0)),
            "mode_extension_used": False,
            "high_nullity_unresolved": False,
            "ruiz": {
                "not_reached": True,
                "obstruction_certificate_complete": False,
            },
        }
    lu = None
    direct_solution = None
    direct_projected = None
    lu_record = {"succeeded": False}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lu = splu(
                matrix.tocsc(), permc_spec="COLAMD", diag_pivot_thresh=1.0,
                options={"Equil": False, "IterRefine": "DOUBLE"},
            )
        direct_solution = np.asarray(lu.solve(-np.asarray(residual)))
        direct_projected = np.asarray(residual + matrix @ direct_solution)
        norm_a_1 = float(np.max(np.asarray(np.abs(matrix).sum(axis=0)).ravel()))
        inverse_operator = LinearOperator(
            matrix.shape, dtype=np.dtype(float),
            matvec=lambda vector: lu.solve(np.asarray(vector)),
            rmatvec=lambda vector: lu.solve(np.asarray(vector), trans="T"),
        )
        inverse_norm_1 = float(onenormest(inverse_operator, t=2, itmax=5))
        pivots = np.abs(lu.U.diagonal())
        lu_record = {
            "succeeded": True,
            "direct_projected_L2": float(np.linalg.norm(direct_projected)),
            "direct_projected_Linf": float(np.max(np.abs(direct_projected))),
            "direct_relative_backward_L2": float(
                np.linalg.norm(direct_projected)
                / max(
                    sigma_max*np.linalg.norm(direct_solution)
                    + np.linalg.norm(residual),
                    _TINY,
                )
            ),
            "direct_scaled_step_Linf": float(np.max(np.abs(direct_solution))),
            "matrix_one_norm": norm_a_1,
            "inverse_one_norm_estimate": inverse_norm_1,
            "condition_one_estimate": norm_a_1*inverse_norm_1,
            "minimum_absolute_U_pivot": float(np.min(pivots)),
            "maximum_absolute_U_pivot": float(np.max(pivots)),
            "L_nnz": int(lu.L.nnz), "U_nnz": int(lu.U.nnz),
        }
    except RuntimeError as error:
        lu_record = {"succeeded": False, "failure": f"{type(error).__name__}: {error}"}

    lsmr_solution, lsmr_projected, lsmr_record = _projection_record(
        matrix, np.asarray(residual), "lsmr"
    )
    lsqr_solution, lsqr_projected, lsqr_record = _projection_record(
        matrix, np.asarray(residual), "lsqr"
    )
    physical_scale = physical_variable_scale(parent)
    if direct_solution is not None:
        direct_correction = column_scale*np.asarray(direct_solution)
        lu_record["direct_physical_correction_Linf"] = float(
            np.max(np.abs(direct_correction))
        )
        lu_record["direct_dimensionless_correction_Linf"] = float(
            np.max(np.abs(direct_correction)/physical_scale)
        )
    for solution, record in (
        (lsmr_solution, lsmr_record), (lsqr_solution, lsqr_record),
    ):
        correction = column_scale*np.asarray(solution)
        record["physical_correction_Linf"] = float(np.max(np.abs(correction)))
        record["dimensionless_correction_Linf"] = float(
            np.max(np.abs(correction)/physical_scale)
        )
    projection_denominator = max(
        np.linalg.norm(lsmr_projected)*np.linalg.norm(lsqr_projected), _TINY,
    )
    projection_correlation = float(
        np.dot(lsmr_projected, lsqr_projected) / projection_denominator
    )
    for projected, record in (
        (lsmr_projected, lsmr_record), (lsqr_projected, lsqr_record),
    ):
        record["normality"] = float(
            np.linalg.norm(matrix.T @ projected)
            / max(sigma_max*np.linalg.norm(projected), _TINY)
        )
        record["normality_passed"] = bool(record["normality"] <= 10.0*tau_rank)
    agreement_l2 = abs(
        np.linalg.norm(lsmr_projected)-np.linalg.norm(lsqr_projected)
    ) / max(np.linalg.norm(lsmr_projected), np.linalg.norm(lsqr_projected), _TINY)
    agreement_linf = abs(
        np.max(np.abs(lsmr_projected))-np.max(np.abs(lsqr_projected))
    ) / max(np.max(np.abs(lsmr_projected)), np.max(np.abs(lsqr_projected)), _TINY)
    numerical_zero_floor = bool(
        lsmr_record["projected_Linf"] <= TARGET_FLOOR
        and lsqr_record["projected_Linf"] <= TARGET_FLOOR
    )
    projection_accepted = bool(
        numerical_zero_floor
        or (
            lsmr_record["normality_passed"] and lsqr_record["normality_passed"]
            and agreement_l2 <= 0.05 and agreement_linf <= 0.05
            and projection_correlation >= 0.99
        )
    )

    mode_method = "inverse-lanczos"
    spectrum_attempts = []
    spectrum_errors = []
    mode_records, mode_vectors = [], []
    try:
        if lu is None:
            raise Protocol131AuditError("SuperLU unavailable")
        mode_records, mode_vectors = _inverse_singular_modes(
            matrix, lu, sigma_max, seed+1, SINGULAR_K,
        )
        if any(
            max(item["left_triplet_defect"], item["right_triplet_defect"]) > 1.0e-7
            for item in mode_records
        ):
            raise Protocol131AuditError("inverse triplet verification failed")
    except (
        ArpackError, ArpackNoConvergence, LinAlgError, RuntimeError,
        Protocol131AuditError,
    ) as error:
        spectrum_attempts.append(
            f"inverse-lanczos: {type(error).__name__}: {error}"
        )
        mode_method = "propack"
        try:
            mode_records, mode_vectors = _propack_singular_modes(
                matrix, sigma_max, seed+2, SINGULAR_K,
            )
            if any(
                max(item["left_triplet_defect"], item["right_triplet_defect"])
                > 1.0e-7 for item in mode_records
            ):
                raise Protocol131AuditError("PROPACK triplet verification failed")
        except (LinAlgError, RuntimeError, Protocol131AuditError) as error:
            spectrum_errors.append(f"propack: {type(error).__name__}: {error}")
            mode_records, mode_vectors = [], []

    _annotate_singular_modes(
        mode_records, mode_vectors, residual=residual, tau_rank=tau_rank,
        q_size=parent["q"].size,
    )
    physical_scale_for_modes = physical_variable_scale(parent)
    for record, (_, right) in zip(mode_records, mode_vectors):
        physical_mode_norm = float(
            np.max(np.abs(column_scale*right)/physical_scale_for_modes)
        )
        record["physical_mode_dimensionless_Linf"] = physical_mode_norm
        record["required_physical_dimensionless_correction"] = float(
            record["required_modal_correction"]*physical_mode_norm
        )

    extension_used = False
    high_nullity = False
    if mode_records and all(item["definitely_null"] for item in mode_records):
        extension_used = True
        try:
            if mode_method == "inverse-lanczos" and lu is not None:
                mode_records, mode_vectors = _inverse_singular_modes(
                    matrix, lu, sigma_max, seed+3, SINGULAR_K_EXTENSION,
                )
            else:
                mode_records, mode_vectors = _propack_singular_modes(
                    matrix, sigma_max, seed+3, SINGULAR_K_EXTENSION,
                )
            _annotate_singular_modes(
                mode_records, mode_vectors, residual=residual,
                tau_rank=tau_rank, q_size=parent["q"].size,
            )
            for record, (_, right) in zip(mode_records, mode_vectors):
                physical_mode_norm = float(
                    np.max(np.abs(column_scale*right)/physical_scale_for_modes)
                )
                record["physical_mode_dimensionless_Linf"] = physical_mode_norm
                record["required_physical_dimensionless_correction"] = float(
                    record["required_modal_correction"]*physical_mode_norm
                )
            high_nullity = bool(
                all(item["definitely_null"] for item in mode_records)
            )
            if any(
                max(item["left_triplet_defect"], item["right_triplet_defect"])
                > 1.0e-7 for item in mode_records
            ):
                raise Protocol131AuditError("extended triplet verification failed")
        except (
            ArpackError, ArpackNoConvergence, LinAlgError, RuntimeError,
            Protocol131AuditError,
        ) as error:
            spectrum_errors.append(
                f"k16-extension: {type(error).__name__}: {error}"
            )
            mode_records, mode_vectors = [], []

    spectrum_certified = bool(
        mode_records
        and not spectrum_errors
        and not high_nullity
        and all(
            item["definitely_null"] or item["definitely_nonnull"]
            for item in mode_records
        )
        and all(
            max(item["left_triplet_defect"], item["right_triplet_defect"])
            <= 1.0e-7 for item in mode_records
        )
    )

    ruiz_matrix, ruiz_row, ruiz_column, ruiz_sweeps = power_two_ruiz(matrix)
    ruiz_rhs = ruiz_row * np.asarray(residual)
    _, ruiz_lsmr_projected_scaled, ruiz_lsmr_record = _projection_record(
        ruiz_matrix, ruiz_rhs, "lsmr"
    )
    _, ruiz_lsqr_projected_scaled, ruiz_lsqr_record = _projection_record(
        ruiz_matrix, ruiz_rhs, "lsqr"
    )
    ruiz_lsmr_physical = ruiz_lsmr_projected_scaled / ruiz_row
    ruiz_lsqr_physical = ruiz_lsqr_projected_scaled / ruiz_row
    ruiz_record = {
        "sweeps": ruiz_sweeps,
        "row_scale_sha256": hash_arrays(ruiz_row),
        "column_scale_sha256": hash_arrays(ruiz_column),
        "lsmr": ruiz_lsmr_record, "lsqr": ruiz_lsqr_record,
        "lsmr_physical_Linf": float(np.max(np.abs(ruiz_lsmr_physical))),
        "lsqr_physical_Linf": float(np.max(np.abs(ruiz_lsqr_physical))),
        # No singular/dual certificate is claimed for this control.  It can
        # reveal scaling sensitivity but cannot by itself certify obstruction.
        "obstruction_certificate_complete": False,
    }
    return {
        "analysis_complete": True,
        "matrix": matrix,
        "column_scale": column_scale,
        "column_exponents": exponents,
        "sigma_max": sigma_max,
        "tau_rank": tau_rank,
        "lu": lu_record,
        "direct_solution_scaled": direct_solution,
        "direct_projected": direct_projected,
        "lsmr_solution_scaled": lsmr_solution,
        "lsmr_projected": lsmr_projected,
        "lsqr_solution_scaled": lsqr_solution,
        "lsqr_projected": lsqr_projected,
        "lsmr": lsmr_record, "lsqr": lsqr_record,
        "projection_agreement_L2": float(agreement_l2),
        "projection_agreement_Linf": float(agreement_linf),
        "projection_correlation": projection_correlation,
        "projection_numerically_zero_floor": numerical_zero_floor,
        "projection_accepted": projection_accepted,
        "mode_method": mode_method,
        "spectrum_certified": spectrum_certified,
        "spectrum_attempts": spectrum_attempts,
        "spectrum_errors": spectrum_errors,
        "modes": mode_records,
        "mode_left_vectors": (
            np.column_stack([item[0] for item in mode_vectors])
            if mode_vectors else np.empty((matrix.shape[0], 0))
        ),
        "mode_right_vectors": (
            np.column_stack([item[1] for item in mode_vectors])
            if mode_vectors else np.empty((matrix.shape[1], 0))
        ),
        "mode_extension_used": extension_used,
        "high_nullity_unresolved": high_nullity,
        "ruiz": ruiz_record,
        "ruiz_lsmr_physical": ruiz_lsmr_physical,
        "ruiz_lsqr_physical": ruiz_lsqr_physical,
    }


def frozen_newton_merit_curve(parent, residual, jacobian, linear):
    """Evaluate the direct correction on the fixed alpha ladder without updating."""
    scaled_solution = linear.get("direct_solution_scaled")
    if scaled_solution is None:
        return {"available": False, "samples": []}
    dx = linear["column_scale"] * np.asarray(scaled_solution)
    variable_scale = physical_variable_scale(parent)
    n = parent["q"].size
    dq = dx[:n].reshape(parent["q"].shape)
    dphi = dx[n:].reshape(parent["phi"].shape)
    jdx = np.asarray(jacobian @ dx)
    samples = []
    for power in range(31):
        alpha = 2.0**-power
        candidate_q = parent["q"] + alpha*dq
        positive = bool(np.min(parent["z"][:, None] + candidate_q) > 0.0)
        if not positive:
            samples.append({"power": power, "alpha": alpha, "positive": False})
            continue
        trial = np.asarray(joint_parent_residual(
            *_residual_arguments(
                parent, candidate_q, parent["phi"] + alpha*dphi,
            )
        ))
        linear_prediction = residual + alpha*jdx
        if not (
            np.all(np.isfinite(trial))
            and np.all(np.isfinite(linear_prediction))
        ):
            samples.append({
                "power": power, "alpha": alpha, "positive": True,
                "finite": False,
            })
            continue
        remainder = trial-linear_prediction
        samples.append({
            "power": power, "alpha": alpha, "positive": True,
            "finite": True,
            "maximum": float(np.max(np.abs(trial))),
            "rms": float(np.sqrt(np.mean(trial**2))),
            "linear_maximum": float(np.max(np.abs(linear_prediction))),
            "taylor_remainder_ratio": float(
                np.max(np.abs(remainder))
                / max(np.max(np.abs(residual)), np.max(np.abs(alpha*jdx)), _TINY)
            ),
            "physical_dimensionless_step_Linf": float(
                np.max(np.abs(alpha*dx)/variable_scale)
            ),
        })
    eligible_maxima = [
        item["maximum"] for item in samples
        if item.get("positive") and "maximum" in item
    ]
    return {
        "available": True,
        "step_sha256": hash_arrays(dx),
        "samples": samples,
        "best_maximum": min(eligible_maxima) if eligible_maxima else None,
    }


def physical_variable_scale(parent):
    """Return the Protocol-131 dimensionless physical correction scale."""
    q_scale = parent["z"][:, None] + parent["q"]
    background = parent.get("background", {})
    phi_scale = max(
        1.0,
        abs(float(background.get("v0", 0.0))),
        abs(float(background.get("v1", 0.0))),
        float(np.max(np.abs(parent["phi"]))),
    )
    return np.concatenate((q_scale.ravel(), np.full(parent["phi"].size, phi_scale)))


def certify_linear_trust_radius(parent, residual, jacobian, linear):
    """Probe the predeclared directions on the fixed physical-radius ladder."""
    scale = physical_variable_scale(parent)
    n = parent["q"].size
    directions = []
    for name, mask in _direction_masks(parent).items():
        direction = _fixed_direction(parent, name, mask)
        normalization = np.max(np.abs(direction) / scale)
        directions.append((name, direction / normalization))
    for index, record in enumerate(linear["modes"]):
        if not (
            record.get("definitely_null")
            or not record.get("definitely_nonnull")
        ):
            continue
        right = linear["mode_right_vectors"][:, index]
        direction = linear["column_scale"] * right
        normalization = np.max(np.abs(direction) / scale)
        if normalization > 0.0:
            directions.append((f"candidate_mode_{index}", direction/normalization))
    radii = tuple(2.0**-power for power in range(20, 1, -2))
    radius_records = []
    for radius in radii:
        maximum_remainder = 0.0
        minimum_margin = float("inf")
        passed = True
        failed_direction = None
        for name, direction in directions:
            dq = direction[:n].reshape(parent["q"].shape)
            dphi = direction[n:].reshape(parent["phi"].shape)
            jdirection = np.asarray(jacobian @ direction)
            for sign in (-1.0, 1.0):
                step = sign*radius
                candidate_q = parent["q"] + step*dq
                margin = float(np.min(parent["z"][:, None] + candidate_q))
                minimum_margin = min(minimum_margin, margin)
                if margin <= 0.0:
                    passed = False; failed_direction = f"{name}/{sign:+.0f}"
                    continue
                trial = np.asarray(joint_parent_residual(
                    *_residual_arguments(
                        parent, candidate_q, parent["phi"] + step*dphi,
                    )
                ))
                prediction = residual + step*jdirection
                if not (
                    np.all(np.isfinite(trial))
                    and np.all(np.isfinite(prediction))
                ):
                    passed = False
                    failed_direction = f"{name}/{sign:+.0f}/nonfinite"
                    continue
                ratio = float(
                    np.max(np.abs(trial-prediction))
                    / max(np.max(np.abs(residual)), np.max(np.abs(step*jdirection)), _TINY)
                )
                if not np.isfinite(ratio):
                    passed = False
                    failed_direction = f"{name}/{sign:+.0f}/nonfinite-ratio"
                    continue
                maximum_remainder = max(maximum_remainder, ratio)
                if ratio > 0.1:
                    passed = False; failed_direction = f"{name}/{sign:+.0f}"
        radius_records.append({
            "rho": radius, "passed": bool(passed),
            "maximum_taylor_remainder_ratio": maximum_remainder,
            "minimum_z_plus_q": minimum_margin,
            "first_or_last_failed_direction": failed_direction,
        })
    passed_radii = [item["rho"] for item in radius_records if item["passed"]]
    rho_linear = max(passed_radii, default=0.0)
    return {
        "rho_linear": float(rho_linear),
        "direction_count": len(directions),
        "samples": radius_records,
    }


def annotate_dual_certificates(parent, jacobian, linear, trust_radius):
    """Attach the fixed local Linf dual certificate to every returned mode."""
    scale = physical_variable_scale(parent)
    rho = float(trust_radius["rho_linear"])
    for index, record in enumerate(linear["modes"]):
        left = linear["mode_left_vectors"][:, index]
        dual_l1 = float(np.sum(np.abs(scale * np.asarray(jacobian.T @ left))))
        numerator = max(0.0, abs(record["left_pairing"])-rho*dual_l1)
        record["physical_dual_L1"] = dual_l1
        record["G_at_rho_linear"] = float(numerator/max(record["left_L1"], _TINY))
        record["rho_linear"] = rho
    return linear["modes"]


def compact_parent_summary(parent, replay, localization, jacobian_audit, linear, merit):
    """Drop large arrays and solver handles from one JSON summary."""
    linear_keys = (
        "analysis_complete", "failure_stage",
        "sigma_max", "tau_rank", "lu", "lsmr", "lsqr",
        "projection_agreement_L2", "projection_agreement_Linf",
        "projection_correlation", "projection_numerically_zero_floor",
        "projection_accepted", "mode_method",
        "spectrum_certified", "spectrum_attempts", "spectrum_errors", "modes",
        "mode_extension_used", "high_nullity_unresolved", "ruiz",
    )
    return {
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "parent_label": parent["label"],
        "parent_identity": str(parent["record"]["parent_identity"]),
        "generated_input_sha256": parent["generated_input_sha256"],
        "replay": replay,
        "localization": {
            "atoms": localization["atoms"],
            "blocks": localization["blocks"],
            "dominant_atom_by_Linf": localization["dominant_atom_by_Linf"],
        },
        "jacobian_audit": jacobian_audit,
        "linear": {
            key: linear[key] for key in linear_keys if key in linear
        },
        "merit_curve": merit,
    }


def parent_array_payload(residual, localization, cancellation, linear, parent=None):
    """Return the fixed machine-readable arrays for one atomic stage."""
    arrays = {
        "residual": np.asarray(residual),
        "binned_energy_16x16": np.asarray(localization["binned_energy_16x16"]),
        "wall_even": np.asarray(localization["wall_even"]),
        "wall_odd": np.asarray(localization["wall_odd"]),
        "raw": np.asarray(cancellation["raw"]),
        "reference_defect": np.asarray(cancellation["reference_defect"]),
        "balanced_before_wall_override": np.asarray(
            cancellation["balanced_before_wall_override"]
        ),
        "final_subtraction_roundoff_bound": np.asarray(
            cancellation["final_subtraction_roundoff_bound"]
        ),
        "final_row_roundoff_bound": np.asarray(
            cancellation["final_row_roundoff_bound"]
        ),
        "column_scale": np.asarray(linear["column_scale"]),
        "column_exponents": np.asarray(linear["column_exponents"]),
        "mode_left_vectors": np.asarray(linear["mode_left_vectors"]),
        "mode_right_vectors": np.asarray(linear["mode_right_vectors"]),
    }
    for name in (
        "lsmr_projected", "lsqr_projected", "lsmr_solution_scaled",
        "lsqr_solution_scaled", "ruiz_lsmr_physical",
        "ruiz_lsqr_physical",
    ):
        if name in linear:
            arrays[name] = np.asarray(linear[name])
    if parent is not None:
        arrays["z"] = np.asarray(parent["z"])
        arrays["r"] = np.asarray(parent["r"])
        arrays["physical_variable_scale"] = physical_variable_scale(parent)
    if linear.get("direct_solution_scaled") is not None:
        arrays["direct_solution_scaled"] = np.asarray(
            linear["direct_solution_scaled"]
        )
        arrays["direct_projected"] = np.asarray(linear["direct_projected"])
    for wall_name, wall in cancellation["wall_terms"].items():
        for name, value in wall.items():
            arrays[f"wall_{wall_name}_{name}"] = np.asarray(value)
    return arrays


def _relative_difference(first, second):
    return float(abs(float(first)-float(second))/max(abs(float(first)), abs(float(second)), _TINY))


def _profile_correlation(first, second):
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    return float(np.dot(first, second)/max(np.linalg.norm(first)*np.linalg.norm(second), _TINY))


def _interpolated_left_null_subspace(summary, arrays):
    modes = summary["linear"]["modes"]
    null_indices = [
        index for index, item in enumerate(modes) if item.get("definitely_null")
    ]
    if not null_indices:
        return np.zeros((2*129*257, 0))
    z = np.asarray(arrays["z"], dtype=float)
    r = np.asarray(arrays["r"], dtype=float)
    vectors = np.asarray(arrays["mode_left_vectors"], dtype=float)[:, null_indices]
    nz, nr = len(z), len(r)
    target_z = np.linspace(1.0, math.e, 129)
    target_r = np.linspace(0.0, 12.0, 257)
    zz, rr = np.meshgrid(target_z, target_r, indexing="ij")
    target = np.column_stack((zz.ravel(), rr.ravel()))
    columns = []
    for column in range(vectors.shape[1]):
        field = vectors[:, column].reshape(2, nz, nr)
        sampled = []
        for block in range(2):
            interpolator = RegularGridInterpolator(
                (z, r), field[block], method="linear", bounds_error=True,
            )
            sampled.append(interpolator(target))
        joined = np.concatenate(sampled)
        norm = np.linalg.norm(joined)
        if norm == 0.0:
            raise Protocol131AuditError("interpolated null mode vanished")
        columns.append(joined/norm)
    matrix = np.column_stack(columns)
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape)*np.finfo(float).eps*singular[0]
    if int(np.count_nonzero(singular > tolerance)) != matrix.shape[1]:
        raise Protocol131AuditError("interpolated null subspace lost rank")
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    return basis


def _canonical_subspace_correlation(summary0, arrays0, summary1, arrays1):
    basis0 = _interpolated_left_null_subspace(summary0, arrays0)
    basis1 = _interpolated_left_null_subspace(summary1, arrays1)
    if basis0.shape[1] != basis1.shape[1] or basis0.shape[1] == 0:
        return 0.0
    correlations = np.linalg.svd(basis0.T @ basis1, compute_uv=False)
    return float(np.min(correlations))


def _precision_record(summary):
    record = summary.get("precision", {})
    return {
        "complete": bool(record.get("complete", False)),
        "mp_certified": bool(record.get("mp_certified", False)),
        "dual_certified": bool(record.get("dual_certified", False)),
        "eta_F": float(record.get("eta_F", float("inf"))),
        "arithmetic_max_below_target": bool(
            record.get("arithmetic_max_below_target", False)
        ),
        "longdouble_maximum": float(record.get("longdouble_maximum", float("nan"))),
    }


def _parent_solver_stagnation(summary):
    linear = summary["linear"]
    if any(item.get("definitely_null") for item in linear["modes"]):
        return False
    if any(not item.get("definitely_nonnull") for item in linear["modes"]):
        return False
    if not linear["projection_accepted"]:
        return False
    if max(linear["lsmr"]["projected_Linf"], linear["lsqr"]["projected_Linf"]) > TARGET_FLOOR:
        return False
    rho = float(summary.get("trust_radius", {}).get("rho_linear", 0.0))
    correction = linear["lu"].get(
        "direct_dimensionless_correction_Linf", float("inf")
    )
    if correction > rho:
        return False
    probes = {
        int(item["power"]): item for item in summary["merit_curve"]["samples"]
        if item.get("positive") and "maximum" in item
    }
    return any(
        power in probes and probes[power]["maximum"] < TARGET_CEILING
        for power in (0, 1, 2, 3)
    )


def _parent_ill_conditioned(summary, arrays):
    linear = summary["linear"]
    if not linear["modes"] or not linear["modes"][0].get("definitely_nonnull"):
        return False
    sigma_min = float(linear["modes"][0]["sigma"])
    condition_two = float(linear["sigma_max"] / max(sigma_min, _TINY))
    residual_maximum = float(summary["replay"]["maximum"])
    threshold = 0.1*TARGET_CEILING/residual_maximum
    if condition_two*np.finfo(float).eps < threshold:
        return False
    column_scale = np.asarray(arrays["column_scale"], dtype=float)
    physical_scale = np.asarray(arrays["physical_variable_scale"], dtype=float)
    lsmr_correction = (
        column_scale*np.asarray(arrays["lsmr_solution_scaled"], dtype=float)
        / physical_scale
    )
    lsqr_correction = (
        column_scale*np.asarray(arrays["lsqr_solution_scaled"], dtype=float)
        / physical_scale
    )
    correction_disagreement = float(
        np.max(np.abs(lsmr_correction-lsqr_correction))
        / max(
            np.max(np.abs(lsmr_correction)),
            np.max(np.abs(lsqr_correction)),
            _TINY,
        )
    )
    rho = float(summary.get("trust_radius", {}).get("rho_linear", 0.0))
    modal = max(
        (
            item.get("required_physical_dimensionless_correction", 0.0)
            for item in linear["modes"]
        ),
        default=0.0,
    )
    direct_bad = bool(
        not linear["lu"].get("succeeded")
        or linear["lu"].get("direct_relative_backward_L2", float("inf"))
        > 0.1*TARGET_CEILING
    )
    precision_bad = bool(
        float(summary["precision"].get("eta_F", float("inf")))
        > 0.1*TARGET_CEILING
    )
    return bool(
        correction_disagreement > 0.20 or modal > rho
        or direct_bad or precision_bad
    )


def classify_protocol131(parent_summaries, parent_arrays):
    """Apply the frozen ordered two-parent Protocol-131 classification."""
    if set(parent_summaries) != set(PARENT_LABELS) or set(parent_arrays) != set(PARENT_LABELS):
        raise Protocol131AuditError("two-parent classifier requires exact N0/N1 records")
    summaries = [parent_summaries[label] for label in PARENT_LABELS]
    if any(not item["jacobian_audit"]["passed"] for item in summaries):
        classification = "INVALID-JACOBIAN"
        reason = "at least one fixed directional analytic-Jacobian gate failed"
    else:
        precision = [_precision_record(item) for item in summaries]
        arithmetic = [
            item["complete"]
            and (
                item["arithmetic_max_below_target"]
                or item["eta_F"] >= 0.25*(summary["replay"]["maximum"]-TARGET_CEILING)
            )
            for item, summary in zip(precision, summaries)
        ]
        if all(arithmetic):
            classification = "ARITHMETIC-LIMITED"
            reason = "both failing maxima meet the frozen extended-precision arithmetic gate"
        elif any(arithmetic):
            classification = "INCONCLUSIVE-MIXED"
            reason = "the arithmetic classification is not persistent across N0/N1"
        elif not all(item["complete"] for item in precision):
            classification = "INCONCLUSIVE-MIXED"
            reason = "the frozen precision certification is incomplete on a load-bearing row"
        elif not all(item["linear"].get("spectrum_certified", False) for item in summaries):
            classification = "INCONCLUSIVE-MIXED"
            reason = "the fixed singular-spectrum certification is unresolved"
        elif all(_parent_solver_stagnation(item) for item in summaries):
            classification = "SOLVER-STAGNATION"
            reason = "both full-rank linear corrections cross the frozen nonlinear gate"
        else:
            null_counts = [
                sum(mode.get("definitely_null", False) for mode in item["linear"]["modes"])
                for item in summaries
            ]
            try:
                correlation = _canonical_subspace_correlation(
                    summaries[0], parent_arrays["N0"],
                    summaries[1], parent_arrays["N1"],
                ) if null_counts[0] == null_counts[1] and null_counts[0] > 0 else 0.0
            except Protocol131AuditError:
                correlation = 0.0
            best_g = [
                max((
                    mode.get("G_at_rho_linear", 0.0)
                    for mode in item["linear"]["modes"]
                    if mode.get("definitely_null")
                ), default=0.0)
                for item in summaries
            ]
            projected = [
                max(item["linear"]["lsmr"]["projected_Linf"], item["linear"]["lsqr"]["projected_Linf"])
                for item in summaries
            ]
            same_atom = (
                summaries[0]["localization"]["dominant_atom_by_Linf"]
                == summaries[1]["localization"]["dominant_atom_by_Linf"]
            )
            blocks_persist = all(
                abs(
                    summaries[0]["localization"]["blocks"][block]["L2_energy_fraction"]
                    - summaries[1]["localization"]["blocks"][block]["L2_energy_fraction"]
                ) <= 0.20
                for block in ("metric", "Phi")
            )
            binned_correlation = _profile_correlation(
                parent_arrays["N0"]["binned_energy_16x16"],
                parent_arrays["N1"]["binned_energy_16x16"],
            )
            ruiz_persists = all(
                min(
                    item["linear"]["ruiz"]["lsmr_physical_Linf"],
                    item["linear"]["ruiz"]["lsqr_physical_Linf"],
                ) >= TARGET_CEILING
                for item in summaries
            )
            obstruction = bool(
                null_counts[0] == null_counts[1] and null_counts[0] > 0
                and all(item["linear"]["projection_accepted"] for item in summaries)
                and all(p["complete"] and p["dual_certified"] for p in precision)
                and all(
                    g >= TARGET_CEILING+4.0*p["eta_F"]
                    for g, p in zip(best_g, precision)
                )
                and _relative_difference(best_g[0], best_g[1]) <= 0.20
                and _relative_difference(projected[0], projected[1]) <= 0.20
                and correlation >= 0.90 and same_atom and blocks_persist
                and binned_correlation >= 0.90 and ruiz_persists
                and all(
                    item["linear"]["ruiz"].get(
                        "obstruction_certificate_complete", False
                    ) for item in summaries
                )
                and not any(item["linear"]["high_nullity_unresolved"] for item in summaries)
            )
            ill_conditioned = [
                _parent_ill_conditioned(summary, parent_arrays[label])
                for label, summary in zip(PARENT_LABELS, summaries)
            ]
            if obstruction:
                classification = "DISCRETE-COMPATIBILITY-OBSTRUCTION"
                reason = (
                    "both grids certify a persistent semi-discrete linearized "
                    "compatibility obstruction within the fixed "
                    "probe-qualified neighborhood"
                )
            elif all(ill_conditioned):
                classification = "ILL-CONDITIONED"
                reason = "both grids meet the frozen conditioning-sensitivity gate"
            elif any(ill_conditioned):
                classification = "INCONCLUSIVE-MIXED"
                reason = (
                    "the frozen conditioning diagnosis is not persistent "
                    "across N0/N1"
                )
            elif all(
                item["linear"]["projection_accepted"]
                and max(
                    item["linear"]["lsmr"]["projected_Linf"],
                    item["linear"]["lsqr"]["projected_Linf"],
                ) < TARGET_CEILING
                and item["merit_curve"].get("available") is True
                and len(item["merit_curve"].get("samples", ())) == 31
                and {
                    int(sample.get("power", -1))
                    for sample in item["merit_curve"]["samples"]
                } == set(range(31))
                and not any(
                    sample.get("positive")
                    and sample.get("maximum", float("inf")) < TARGET_CEILING
                    for sample in item["merit_curve"]["samples"]
                )
                for item in summaries
            ):
                classification = "NONLINEAR/GLOBALIZATION-UNRESOLVED"
                reason = "linear floors cross the gate but fixed nonlinear probes do not qualify"
            else:
                classification = "INCONCLUSIVE-MIXED"
                reason = "the frozen persistence or numerical-certification gates do not select a stronger class"
    return {
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "classification": classification,
        "complete": True,
        "provenance_valid": True,
        "reason": reason,
        "parent_labels": list(PARENT_LABELS),
        "paper1_unblocked": False,
        "parent_construction_authorized": False,
        "phase_a_authorized": False,
        "evolution_authorized": False,
        "new_interface_physics_authorized": False,
        "lin_manufactured_controls_next": True,
    }
