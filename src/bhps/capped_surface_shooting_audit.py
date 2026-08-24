"""Regular-axis shooting audit for donor-capped minimal polar graphs."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from bhps.anisotropic_capped_surface import anisotropic_rho_second
from bhps.capped_surface_barrier_certificate import point_barrier_from_splines


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def regular_axis_initial_state(axis_radius, theta_cut, z_brane, splines):
    """Second-order even-axis start implied by the regular EL equation."""
    axis_radius = float(axis_radius)
    theta_cut = float(theta_cut)
    barrier = _scalar(point_barrier_from_splines(
        0.0, axis_radius, z_brane, splines,
    ))
    second = axis_radius * barrier
    return np.asarray([
        axis_radius + 0.5 * second * theta_cut**2,
        second * theta_cut,
    ]), {
        "axis_barrier": barrier,
        "axis_second_derivative": second,
        "theta_cut": theta_cut,
    }


def shoot_axis_radius(
    axis_radius, z_brane, splines, rho_bounds=(0.10, 1.67),
    theta_cut=1e-3, relative_tolerance=2e-9, absolute_tolerance=2e-11,
    maximum_step=0.01, graph_slope_guard=100.0,
):
    """Shoot one regular-axis solution and classify its first terminal event."""
    initial, axis = regular_axis_initial_state(
        axis_radius, theta_cut, z_brane, splines,
    )
    lower, upper = map(float, rho_bounds)
    if initial[0] > upper:
        return {
            "axis_radius": float(axis_radius), "status": "upper_band_exit",
            "solver_success": True, "message": "axis series exits upper band",
            "end_theta": float(theta_cut), "end_rho": float(initial[0]),
            "end_slope": float(initial[1]), "rho_minimum": float(initial[0]),
            "rho_maximum": float(initial[0]),
            "maximum_absolute_slope": float(abs(initial[1])),
            "function_evaluations": 0, **axis,
        }
    if initial[0] < lower:
        return {
            "axis_radius": float(axis_radius), "status": "lower_band_exit",
            "solver_success": True, "message": "axis series exits lower band",
            "end_theta": float(theta_cut), "end_rho": float(initial[0]),
            "end_slope": float(initial[1]), "rho_minimum": float(initial[0]),
            "rho_maximum": float(initial[0]),
            "maximum_absolute_slope": float(abs(initial[1])),
            "function_evaluations": 0, **axis,
        }

    def equation(theta, state):
        second = anisotropic_rho_second(
            np.asarray([theta]), np.asarray([state[0]]),
            np.asarray([state[1]]), z_brane, splines,
        )
        return np.asarray([state[1], _scalar(second)])

    def upper_exit(theta, state):
        return upper - state[0]

    def lower_exit(theta, state):
        return state[0] - lower

    def graph_singularity(theta, state):
        return float(graph_slope_guard) - abs(state[1])

    upper_exit.terminal = True
    upper_exit.direction = -1
    lower_exit.terminal = True
    lower_exit.direction = -1
    graph_singularity.terminal = True
    graph_singularity.direction = -1
    try:
        solved = solve_ivp(
            equation, (float(theta_cut), math.pi / 2), initial,
            method="DOP853", rtol=float(relative_tolerance),
            atol=float(absolute_tolerance), max_step=float(maximum_step),
            events=(upper_exit, lower_exit, graph_singularity),
        )
    except (FloatingPointError, ValueError, OverflowError) as error:
        return {
            "axis_radius": float(axis_radius), "status": "integration_failure",
            "message": repr(error), **axis,
        }

    event_counts = [len(values) for values in solved.t_events]
    reached_brane = bool(solved.success and solved.t[-1] >= math.pi / 2 - 2e-12)
    if reached_brane:
        status = "reached_brane"
    elif event_counts[0]:
        status = "upper_band_exit"
    elif event_counts[1]:
        status = "lower_band_exit"
    elif event_counts[2]:
        status = "graph_slope_guard"
    else:
        status = "integration_failure"
    rho = np.asarray(solved.y[0])
    slope = np.asarray(solved.y[1])
    result = {
        "axis_radius": float(axis_radius), "status": status,
        "solver_success": bool(solved.success), "message": str(solved.message),
        "end_theta": float(solved.t[-1]), "end_rho": float(rho[-1]),
        "end_slope": float(slope[-1]), "rho_minimum": float(np.min(rho)),
        "rho_maximum": float(np.max(rho)),
        "maximum_absolute_slope": float(np.max(np.abs(slope))),
        "function_evaluations": int(solved.nfev), **axis,
    }
    if reached_brane:
        result["brane_residual"] = float(slope[-1])
        result["brane_radius"] = float(rho[-1])
    return result


def shooting_scan(axis_radii, z_brane, splines, **kwargs):
    return [
        shoot_axis_radius(value, z_brane, splines, **kwargs)
        for value in np.asarray(axis_radii, dtype=float)
    ]


def summarize_shooting_scan(records):
    status_counts = {}
    for record in records:
        status = record["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    reached = [record for record in records if record["status"] == "reached_brane"]
    reached.sort(key=lambda item: item["axis_radius"])
    sign_changes = []
    for left, right in zip(reached[:-1], reached[1:]):
        if left["brane_residual"] * right["brane_residual"] < 0.0:
            sign_changes.append([
                left["axis_radius"], right["axis_radius"],
                left["brane_residual"], right["brane_residual"],
            ])
    if reached:
        minimum = min(reached, key=lambda item: item["brane_residual"])
        maximum = max(reached, key=lambda item: item["brane_residual"])
        minimum_absolute = min(reached, key=lambda item: abs(item["brane_residual"]))
    else:
        minimum = maximum = minimum_absolute = None
    return {
        "sample_count": len(records),
        "status_counts": status_counts,
        "reached_brane_count": len(reached),
        "sign_change_count": len(sign_changes),
        "sign_change_brackets": sign_changes,
        "minimum_residual": None if minimum is None else minimum["brane_residual"],
        "minimum_residual_axis_radius": None if minimum is None else minimum["axis_radius"],
        "maximum_residual": None if maximum is None else maximum["brane_residual"],
        "minimum_absolute_residual": (
            None if minimum_absolute is None else abs(minimum_absolute["brane_residual"])
        ),
        "minimum_absolute_residual_axis_radius": (
            None if minimum_absolute is None else minimum_absolute["axis_radius"]
        ),
        "all_reached_residuals_positive": bool(
            reached and all(record["brane_residual"] > 0.0 for record in reached)
        ),
    }


def adjacent_status_cells(records):
    """Classify sampled cells without pretending this is interval validation."""
    ordered = sorted(records, key=lambda item: item["axis_radius"])
    classes = {
        "positive_residual_endpoints": 0,
        "same_exit_endpoints": 0,
        "mixed_or_unresolved": 0,
        "root_sign_change": 0,
    }
    mixed = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        ls = left["status"]
        rs = right["status"]
        if ls == rs == "reached_brane":
            product = left["brane_residual"] * right["brane_residual"]
            key = "root_sign_change" if product < 0.0 else "positive_residual_endpoints"
            classes[key] += 1
        elif ls == rs and ls in {
            "upper_band_exit", "lower_band_exit", "graph_slope_guard",
        }:
            classes["same_exit_endpoints"] += 1
        else:
            classes["mixed_or_unresolved"] += 1
            mixed.append([left["axis_radius"], right["axis_radius"], ls, rs])
    total = max(len(ordered) - 1, 1)
    return {
        "cell_count": len(ordered) - 1,
        "classes": classes,
        "mixed_cells": mixed,
        "non_root_sampled_fraction": (
            classes["positive_residual_endpoints"] + classes["same_exit_endpoints"]
        ) / total,
        "warning": (
            "Endpoint/status cell classification is exhaustive sampling, not "
            "an interval image of the continuum shooting residual."
        ),
    }
