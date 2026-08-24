#!/usr/bin/env python3
"""Audit whether sealed Test-10E chunks can support staged D_X^2 J.

This is deliberately archive-only.  It does not import or call the native
RHS, rebuild an initial-data family, solve a constraint, or approximate a
missing acceleration by differencing accepted states.  Its output is a
machine-readable go/no-go gate for exact operation-order attribution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RECOVERY = ROOT / (
    "results/corrected_A790_test10e_genuine_high_z_boundary_resolution_recovery"
)
MANIFEST = RECOVERY / "index.json"
OUTPUT = ROOT / "results/corrected_A790_staged_junction_archive_sufficiency.json"
STENCIL_WIDTH = 7
FIELD_COUNT = 9

# The two physically compared accepted times.  Each is a chunk endpoint, so
# q, v, source, and memory are all present in the selected chunk.
TARGETS = (
    ("G9_R10", 8, 0.001, 0.000125),
    ("G9_R10", 16, 0.002, 0.000125),
    ("G10_R10", 8, 0.001, 0.000125),
    ("G10_R10", 16, 0.002, 0.000125),
    ("G10H_R10", 16, 0.001, 0.0000625),
    ("G10H_R10", 32, 0.002, 0.0000625),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_bounds(step: int) -> tuple[int, int]:
    start = ((int(step) - 1) // 4) * 4
    return start, start + 4


def chunk_path(label: str, step: int) -> Path:
    start, end = chunk_bounds(step)
    return RECOVERY / f"physical_{label}_steps_{start + 1:03d}_{end:03d}.npz"


def manifest_stage(label: str, step: int) -> str:
    start, end = chunk_bounds(step)
    return f"physical/{label}/steps_{start + 1:03d}_{end:03d}"


def coincident_rk_record(
    manifest: dict, label: str, step: int, time_value: float, total: int,
):
    """Locate a stored RK evaluation at an accepted endpoint, if one exists."""
    if step >= total:
        return None
    following = chunk_path(label, step + 1)
    if not following.is_file():
        return None
    stage_id = manifest_stage(label, step + 1)
    stage = manifest["stages"].get(stage_id)
    actual_hash = sha256(following)
    expected_hash = None if stage is None else stage.get("sha256")
    with np.load(following) as archive:
        matches = np.flatnonzero(np.isclose(
            np.asarray(archive["time"], dtype=float),
            float(time_value), rtol=0.0, atol=1e-15,
        ))
        if len(matches) != 1:
            return None
        index = int(matches[0])
        return {
            "path": str(following.relative_to(ROOT)),
            "sha256": actual_hash,
            "manifest_stage": stage_id,
            "manifest_sha256": expected_hash,
            "manifest_hash_matches": bool(expected_hash == actual_hash),
            "record_index": index,
            "rk_stage": int(archive["rk_stage"][index]),
            "stored_before_shape": list(archive["before"][index].shape),
            "stored_after_shape": list(archive["after"][index].shape),
            "stored_fields": ["Phi", "chi"],
            "stored_support": "outer radial face, compact-open z nodes only",
            "is_full_acceleration": False,
            "is_operation_stage_history": False,
        }


def target_record(manifest: dict, label: str, step: int, time_value: float, dt: float):
    path = chunk_path(label, step)
    if not path.is_file():
        raise FileNotFoundError(path)
    stage_id = manifest_stage(label, step)
    stage = manifest["stages"].get(stage_id)
    actual_hash = sha256(path)
    expected_hash = None if stage is None else stage.get("sha256")
    total = 32 if label == "G10H_R10" else 16
    with np.load(path) as archive:
        end = int(archive["end_step"])
        if end != step:
            raise RuntimeError(f"selected endpoint {label}/{step} is not a chunk end")
        position = np.asarray(archive["end_position"])
        velocity = np.asarray(archive["end_velocity"])
        source = np.asarray(archive["end_source"])
        memory = np.asarray(archive["end_memory"])
        increment_key = f"step_{step:03d}_increment"
        source_increment_key = f"step_{step:03d}_source_increment"
        accepted_keys = (
            "end_position", "end_velocity", "end_source", "end_memory",
            increment_key, f"step_{step:03d}_velocity", source_increment_key,
        )
        accepted_present = {key: key in archive.files for key in accepted_keys}
        stored_acceleration_keys = [
            key for key in archive.files
            if key in ("before", "after", "target", "term_A", "term_V", "term_C")
        ]
        staged_full_keys = [
            key for key in archive.files
            if key.startswith("boundary_stage_") or key.startswith("acceleration_stage_")
        ]
        initial_position_derivable = bool(increment_key in archive.files)
        initial_source_derivable = bool(source_increment_key in archive.files)

    return {
        "label": label,
        "step": step,
        "time": time_value,
        "dt": dt,
        "path": str(path.relative_to(ROOT)),
        "sha256": actual_hash,
        "manifest_stage": stage_id,
        "manifest_status": None if stage is None else stage.get("status"),
        "manifest_sha256": expected_hash,
        "manifest_hash_matches": bool(expected_hash == actual_hash),
        "accepted_endpoint": {
            "keys_present": accepted_present,
            "position_shape": list(position.shape),
            "velocity_shape": list(velocity.shape),
            "source_shape": list(source.shape),
            "memory_shape": list(memory.shape),
            "all_finite": bool(all(np.all(np.isfinite(item)) for item in (
                position, velocity, source, memory,
            ))),
            "initial_position_derivable": initial_position_derivable,
            "initial_source_derivable": initial_source_derivable,
        },
        "stored_acceleration_lane": {
            "keys": stored_acceleration_keys,
            "array_semantics": (
                "two scalar fields (Phi, chi) on the compact-open outer face "
                "at recorded RK evaluations"
            ),
            "full_metric_fields_present": False,
            "compact_wall_endpoints_present": False,
            "axis_column_present": False,
            "pre_wall_stage_present": False,
            "post_wall_pre_axis_stage_present": False,
            "post_axis_pre_outer_stage_present": False,
            "post_outer_full_stage_present": False,
            "candidate_full_stage_keys": staged_full_keys,
        },
        "coincident_rk_record": coincident_rk_record(
            manifest, label, step, time_value, total,
        ),
        "exact_DX2J_from_archive": False,
        "exact_endpoint_RHS_replay_without_family_rebuild": False,
        "missing_for_exact_endpoint_replay": [
            "full initial outer-reference acceleration for all nine fields",
            "initial Taylor source-time array",
            "initial live source second-time array",
            "serialized background/mass/case configuration (or an immutable case artifact)",
        ],
        "missing_for_direct_stage_evaluation": [
            "all-field acceleration on the first and last seven compact rows",
            "wall endpoint accelerations",
            "r=0 acceleration values before and after the post-wall axis fill",
            "r=R acceleration values before and after the post-wall outer overwrite",
        ],
    }


def capture_contract() -> dict:
    records = []
    raw_bytes = 0
    # Shapes are known from the accepted endpoint archives.  The support union
    # is enough for both wall D_z rows and the complete outgoing face check.
    for label, step, time_value, dt in TARGETS:
        path = chunk_path(label, step)
        with np.load(path) as archive:
            nz, nr, fields = archive["end_position"].shape
        if fields != FIELD_COUNT:
            raise RuntimeError("unexpected reduced-field count")
        wall_slab_values = 2 * STENCIL_WIDTH * nr * fields
        outer_face_values = nz * fields
        per_stage_bytes = 8 * (wall_slab_values + outer_face_values)
        raw_bytes += per_stage_bytes
        records.append({
            "label": label,
            "step": step,
            "time": time_value,
            "dt": dt,
            "grid_shape": [nz, nr, fields],
            "per_acceleration_stage_raw_bytes": per_stage_bytes,
        })
    return {
        "evaluation_times": [0.001, 0.002],
        "runs": ["G9_R10", "G10_R10", "G10H_R10"],
        "stencil_width": STENCIL_WIDTH,
        "at_each_accepted_state_save": [
            "position and velocity (or immutable hashes plus exact source archive references)",
            "z and r coordinate arrays",
            "background scalar parameters, mass_squared, stencil/axis-fit settings, and closure mode",
            "source, source_time, and source_second_time at both compact walls for the normal-GH audit",
            "for every captured acceleration stage: first seven and last seven z rows for all r and fields",
            "for every captured acceleration stage: the full r=R face for all z and fields",
            "ordered stage name, iteration number, and code/configuration hashes",
        ],
        "required_stage_landmarks": [
            "initial_axis_fill",
            "final_compact_wall_endpoint_solve",
            "final_compact_post_wall_axis_fill (legacy lane)",
            "outer_open_face_before_wall (owner-last lane)",
            "post_outer (legacy lane) or post_wall_owner_reconciliation (owner-last lane)",
        ],
        "why_boundary_support_is_sufficient": (
            "the production seven-point D_z wall row only samples the first or "
            "last seven compact rows; the full outer face separately proves that "
            "all outgoing-owned open nodes remain unchanged"
        ),
        "raw_bytes_per_one_stage_across_all_six_endpoints": raw_bytes,
        "estimated_raw_bytes_for_eight_stages_across_all_six_endpoints": 8 * raw_bytes,
        "compression_not_included": True,
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    records = [target_record(manifest, *target) for target in TARGETS]
    hashes_valid = bool(all(
        item["manifest_hash_matches"]
        and (
            item["coincident_rk_record"] is None
            or item["coincident_rk_record"]["manifest_hash_matches"]
        )
        for item in records
    ))
    endpoints_complete = bool(all(
        all(item["accepted_endpoint"]["keys_present"].values())
        and item["accepted_endpoint"]["all_finite"]
        for item in records
    ))
    payload = {
        "schema": "bhps-staged-junction-archive-sufficiency-v1",
        "status": "complete_no_go_for_exact_archive_replay",
        "scope": (
            "read-only Test-10E G9/G10/G10H accepted-endpoint archive inventory; "
            "no RHS, constraint, evolution, or finite-difference acceleration"
        ),
        "inputs": {
            str(MANIFEST.relative_to(ROOT)): sha256(MANIFEST),
            str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__).resolve()),
            **{item["path"]: item["sha256"] for item in records},
            **{
                item["coincident_rk_record"]["path"]:
                    item["coincident_rk_record"]["sha256"]
                for item in records if item["coincident_rk_record"] is not None
            },
        },
        "records": records,
        "gates": {
            "sealed_chunk_hashes_match_manifest": hashes_valid,
            "accepted_endpoint_state_source_memory_complete": endpoints_complete,
            "full_metric_acceleration_present": False,
            "wall_endpoint_acceleration_present": False,
            "operation_stage_snapshots_present": False,
            "initial_reference_bundle_complete": False,
            "exact_archive_only_DX2J_stage_audit_authorized": False,
        },
        "decision": {
            "archive_only_exact_stage_test": "STOP",
            "accepted_state_DXJ_replay": "available and already completed separately",
            "time_differenced_acceleration": (
                "not a substitute: it cannot identify which boundary operation "
                "created a wall-row defect and is not the native semi-discrete RHS"
            ),
            "next_action": (
                "capture the minimal boundary-support contract during a fresh, "
                "fully hashed RHS evaluation or matched physical run"
            ),
        },
        "minimal_new_capture_contract": capture_contract(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({
        "status": payload["status"],
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT),
        "gates": payload["gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
