"""Causal stage audit for the compact-wall acceleration closure.

The state ``(q, v)`` is held fixed while successive acceleration snapshots
are compared.  Consequently every jump in ``D_X^2 J`` between two stages is
exactly a jump in the linear ``D J[q].a`` lane; the velocity-Hessian lane is
unchanged.  This isolates which boundary operation creates or removes a
semi-discrete wall-preservation defect without changing the field equations.
"""

from __future__ import annotations

import numpy as np

from bhps.junction_preservation_diagnostic import (
    WALLS,
    _maximum_record,
    _orthonormal_frames,
    _proper_profile_statistics,
    radial_zones,
)
from bhps.junction_second_preservation_diagnostic import (
    summarize_wall_second_tangent,
    wall_junction_second_tangent,
)


def _tensor_jump_summary(reference, comparison, r, buffer_points):
    metric = np.asarray(reference["metric_tensor"], dtype=float)
    frames, frame_defect = _orthonormal_frames(metric)
    delta_second = (
        np.asarray(comparison["DX2J_tensor"])
        - np.asarray(reference["DX2J_tensor"])
    )
    delta_linear = (
        np.asarray(comparison["DJ_acceleration_tensor"])
        - np.asarray(reference["DJ_acceleration_tensor"])
    )
    delta_hessian = (
        np.asarray(comparison["D2J_velocity_velocity_tensor"])
        - np.asarray(reference["D2J_velocity_velocity_tensor"])
    )
    causal_defect = delta_second - delta_linear
    second_hat = np.einsum(
        "nai,nab,nbj->nij", frames, delta_second, frames,
    )
    second_norm = np.linalg.norm(second_hat, axis=(1, 2))
    zones = {}
    for zone, indices in radial_zones(r, buffer_points).items():
        zones[zone] = {
            "DX2J_jump": _maximum_record(second_norm, indices, r),
            "proper_statistics": _proper_profile_statistics(
                second_norm, indices, r, metric,
            ),
        }
    return {
        "zones": zones,
        "causal_identity_maximum_absolute_defect": float(
            np.max(np.abs(causal_defect))
        ),
        "velocity_hessian_change_maximum_absolute": float(
            np.max(np.abs(delta_hessian))
        ),
        "frame_defect_maximum": float(np.max(frame_defect)),
        "finite": bool(
            np.all(np.isfinite(second_norm))
            and np.all(np.isfinite(causal_defect))
            and np.all(np.isfinite(delta_hessian))
        ),
    }


def _public_wall_record(record, r, buffer_points):
    summary = summarize_wall_second_tangent(record, r, buffer_points)
    separate = record["separate_rows"]
    summary.update({
        "Phi_DX2_maximum_absolute": float(
            np.max(np.abs(separate["DX2_Phi_robin"]))
        ),
        "chi_DX2_maximum_absolute": float(
            np.max(np.abs(separate["DX2_chi_neumann"]))
        ),
        "decomposition_maximum_absolute_defect": record[
            "decomposition_maximum_absolute_defect"
        ],
        "raw_vs_cancellation_exposed_maximum_absolute_defect": record[
            "raw_vs_cancellation_exposed_maximum_absolute_defect"
        ],
    })
    return summary


def evaluate_boundary_stage_sequence(
    position, velocity, boundary_stages, z, r, background, stencil_width=7,
    buffer_points=7,
):
    """Evaluate ``D_X^2J`` at every captured acceleration stage.

    ``boundary_stages`` is the list returned in the diagnostic dictionary by
    ``NativeRegularSO3RHS.acceleration(..., capture_boundary_stages=True)``.
    The output contains scalar summaries only and is therefore suitable for a
    JSON audit artifact.  It does not claim a covariant or moving-cap second
    derivative.
    """
    if boundary_stages is None or not isinstance(boundary_stages, (list, tuple)):
        raise ValueError("boundary_stages must be a captured stage sequence")
    if not boundary_stages:
        raise ValueError("boundary stage sequence is empty")
    r = np.asarray(r, dtype=float)
    raw = []
    public = []
    for index, stage in enumerate(boundary_stages):
        if not isinstance(stage, dict) or "name" not in stage or "acceleration" not in stage:
            raise ValueError("each stage must contain name and acceleration")
        acceleration = np.asarray(stage["acceleration"], dtype=float)
        walls = {
            wall: wall_junction_second_tangent(
                position, velocity, acceleration, z, r, background, wall,
                stencil_width,
            ) for wall in WALLS
        }
        metadata = {
            key: value for key, value in stage.items()
            if key != "acceleration"
        }
        raw.append(walls)
        public.append({
            "index": int(index),
            **metadata,
            "walls": {
                wall: _public_wall_record(walls[wall], r, buffer_points)
                for wall in WALLS
            },
        })

    jumps = []
    for index in range(1, len(raw)):
        jumps.append({
            "from_index": int(index - 1),
            "to_index": int(index),
            "from_name": str(boundary_stages[index - 1]["name"]),
            "to_name": str(boundary_stages[index]["name"]),
            "walls": {
                wall: _tensor_jump_summary(
                    raw[index - 1][wall], raw[index][wall], r, buffer_points,
                ) for wall in WALLS
            },
            "Phi_DX2_jump_maximum_absolute": {
                wall: float(np.max(np.abs(
                    raw[index][wall]["separate_rows"]["DX2_Phi_robin"]
                    - raw[index - 1][wall]["separate_rows"]["DX2_Phi_robin"]
                ))) for wall in WALLS
            },
            "chi_DX2_jump_maximum_absolute": {
                wall: float(np.max(np.abs(
                    raw[index][wall]["separate_rows"]["DX2_chi_neumann"]
                    - raw[index - 1][wall]["separate_rows"]["DX2_chi_neumann"]
                ))) for wall in WALLS
            },
        })
    return {
        "scope": (
            "fixed-grid coordinate-time second derivative of the semi-discrete "
            "wall rows; q and v fixed across stages; noncovariant and without "
            "moving-surface terms"
        ),
        "stage_count": len(public),
        "stages": public,
        "jumps": jumps,
        "finite": bool(
            all(stage["walls"][wall]["finite"] for stage in public for wall in WALLS)
            and all(jump["walls"][wall]["finite"] for jump in jumps for wall in WALLS)
        ),
    }
