"""Deterministic gauge-source map for the Protocol-125 parent fixed point.

This module does not evolve a state and does not apply an outer boundary
condition.  It reconstructs ``(H,H_t,H_tt)`` and the algebraic driver memory
from one represented parent acceleration, while reapplying the compact-wall
normal-source trace owned by the parent construction.
"""

from __future__ import annotations

import numpy as np

from bhps.joint_parent_native_completion import (
    complete_normal_gauge_source_wall,
)
from bhps.matched_staged_continuum import DriverConfiguration
from bhps.nonlinear_regular_so3_evolution import (
    gauge_taylor_source_from_initial_jets,
    live_regular_source_second_time,
    regular_source_spatial_derivatives,
)
from bhps.gh_source_driver import (
    regular_so3_live_source_shift_advection,
    regular_so3_nonlinear_anchored_damped_wave_target,
    source_driver_rhs,
)


SOURCE_SECOND_TIME_DIFFERENCE_STEP = 1e-6


def _scaled_linf(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("scaled-Linf arrays must have equal shapes")
    denominator = np.maximum.reduce((
        np.ones_like(left), np.abs(left), np.abs(right),
    ))
    return float(np.max(np.abs(left-right)/denominator))


def initial_driver_source_triplet_from_acceleration(
    jet_field,
    z,
    r,
    background,
    *,
    driver=DriverConfiguration(),
    hdot_tolerance=1e-12,
):
    """Return the frozen Protocol-125 source map for one acceleration iterate.

    ``jet_field`` must contain the represented position, exact zero velocity,
    candidate acceleration, and their analytic spatial jets.  Memory is
    reconstructed algebraically on every call; it is never carried from a
    previous fixed-point iterate.
    """
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(jet_field.reduced_fields, dtype=float)
    first = np.asarray(jet_field.reduced_first, dtype=float)
    expected = (len(z), len(r), 9)
    if q.shape != expected or first.shape != (3, *expected):
        raise ValueError("invalid represented parent jet")
    if not (
        np.array_equal(z, np.asarray(jet_field.z, dtype=float))
        and np.array_equal(r, np.asarray(jet_field.r, dtype=float))
    ):
        raise ValueError("source-map coordinates differ from the represented jet")
    velocity = first[0]
    if np.any(velocity != 0.0) or np.any(np.signbit(velocity)):
        raise ValueError("joint parent source map requires IEEE positive-zero velocity")
    if not isinstance(driver, DriverConfiguration):
        raise TypeError("driver must be the frozen DriverConfiguration")
    tolerance = float(hdot_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Hdot tolerance must be positive")

    taylor = gauge_taylor_source_from_initial_jets(jet_field, z, r)
    raw_source = np.asarray(taylor.source_reduced, dtype=float)
    source_time = np.asarray(taylor.source_time_reduced, dtype=float).copy()
    source, wall_record = complete_normal_gauge_source_wall(
        q,
        raw_source,
        z,
        r,
        background,
        stencil_width=driver.stencil_width,
    )
    target = regular_so3_nonlinear_anchored_damped_wave_target(
        q,
        q,
        source,
        r,
        driver.target_mu_lapse,
        driver.target_mu_shift,
        driver.target_power,
    )
    source_z, source_r = regular_source_spatial_derivatives(
        source, z, r, driver.stencil_width,
    )
    advection = regular_so3_live_source_shift_advection(
        q, r, source, source_z, source_r,
    )
    memory = (
        source_time
        - advection
        + driver.driver_mu*(source-target)
    )
    source_dot, memory_dot = source_driver_rhs(
        source,
        memory,
        target,
        driver.driver_mu,
        driver.driver_eta,
        advection,
    )
    hdot_defect = _scaled_linf(source_dot, source_time)
    if hdot_defect > tolerance:
        raise RuntimeError(
            "initial source-driver reconstruction does not reproduce H_t: "
            f"defect={hdot_defect}, tolerance={tolerance}"
        )
    source_second_time = live_regular_source_second_time(
        q,
        velocity,
        q,
        source,
        source,
        source_dot,
        memory_dot,
        z,
        r,
        driver.driver_mu,
        driver.target_mu_lapse,
        driver.target_mu_shift,
        driver.target_power,
        driver.stencil_width,
        difference_step=SOURCE_SECOND_TIME_DIFFERENCE_STEP,
    )
    arrays = (
        source,
        source_time,
        source_second_time,
        memory,
        memory_dot,
        target,
        advection,
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise RuntimeError("joint parent source triplet is nonfinite")
    return {
        "source": source,
        "source_time": source_time,
        "source_second_time": source_second_time,
        "memory": memory,
        "memory_time": memory_dot,
        "target": target,
        "advection": advection,
        "raw_geometric_source": raw_source.copy(),
        "normal_wall_completion": wall_record,
        "Hdot_reassembly_scaled_Linf": hdot_defect,
        "difference_step": SOURCE_SECOND_TIME_DIFFERENCE_STEP,
        "driver": driver.public(),
        "outer_source_overwrite_applied": False,
        "memory_carried_from_previous_iterate": False,
    }

