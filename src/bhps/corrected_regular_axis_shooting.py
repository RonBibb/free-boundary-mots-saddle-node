"""Floating navigation shoot with the corrected regular-axis factor."""

from __future__ import annotations

import math
import numpy as np
from scipy.integrate import solve_ivp

from bhps.anisotropic_capped_surface import anisotropic_rho_second
from bhps.capped_surface_barrier_certificate import point_barrier_from_splines


def corrected_regular_axis_initial_state(axis_radius, theta_cut, z_brane, splines):
    axis_radius = float(axis_radius)
    theta_cut = float(theta_cut)
    barrier = float(np.asarray(point_barrier_from_splines(
        0.0, axis_radius, z_brane, splines,
    )).reshape(-1)[0])
    second = axis_radius * barrier / 3.0
    return np.asarray([
        axis_radius + 0.5 * second * theta_cut**2,
        second * theta_cut,
    ]), {
        "axis_barrier": barrier,
        "regular_axis_second_derivative": second,
        "theta_cut": theta_cut,
    }


def corrected_shoot_axis_radius(
    axis_radius, z_brane, splines, rho_bounds=(0.10, 1.67),
    theta_cut=1e-4, relative_tolerance=2e-10, absolute_tolerance=2e-12,
    maximum_step=0.005, graph_slope_guard=100.0,
):
    initial, axis = corrected_regular_axis_initial_state(
        axis_radius, theta_cut, z_brane, splines,
    )
    lower, upper = map(float, rho_bounds)

    def equation(theta, state):
        second = anisotropic_rho_second(
            np.asarray([theta]), np.asarray([state[0]]),
            np.asarray([state[1]]), z_brane, splines,
        )
        return np.asarray([state[1], float(second[0])])

    def upper_exit(theta, state):
        return upper - state[0]

    def lower_exit(theta, state):
        return state[0] - lower

    def slope_guard(theta, state):
        return float(graph_slope_guard) - abs(state[1])

    upper_exit.terminal = True
    upper_exit.direction = -1
    lower_exit.terminal = True
    lower_exit.direction = -1
    slope_guard.terminal = True
    slope_guard.direction = -1
    solved = solve_ivp(
        equation, (theta_cut, math.pi / 2), initial,
        method="DOP853", rtol=relative_tolerance, atol=absolute_tolerance,
        max_step=maximum_step, events=(upper_exit, lower_exit, slope_guard),
    )
    reached = bool(solved.success and solved.t[-1] >= math.pi / 2 - 2e-12)
    if reached:
        status = "reached_brane"
    elif len(solved.t_events[0]):
        status = "upper_band_exit"
    elif len(solved.t_events[1]):
        status = "lower_band_exit"
    elif len(solved.t_events[2]):
        status = "graph_slope_guard"
    else:
        status = "integration_failure"
    result = {
        "axis_radius": float(axis_radius), "status": status,
        "solver_success": bool(solved.success), "message": str(solved.message),
        "end_theta": float(solved.t[-1]), "end_rho": float(solved.y[0, -1]),
        "end_slope": float(solved.y[1, -1]), "function_evaluations": int(solved.nfev),
        **axis,
    }
    if reached:
        result["brane_residual"] = result["end_slope"]
        result["brane_radius"] = result["end_rho"]
    return result
