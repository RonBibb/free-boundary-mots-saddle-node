"""Numerical primitives for Protocol 229 pseudo-arclength continuation."""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicSpline
from scipy.linalg import eig


def physical_mode_to_radial(extension, normal_factor, physical_mode):
    """Map an interior physical-normal mode to a full radial-graph mode.

    The stability-matrix unknown is physical displacement ``f`` at interior
    nodes.  With ``w=f/delta_rho`` and Neumann extension ``E``, the full radial
    perturbation is exactly ``E @ (f / w_interior)``.
    """
    extension = np.asarray(extension, dtype=float)
    normal_factor = np.asarray(normal_factor, dtype=float)
    physical_mode = np.asarray(physical_mode, dtype=float)
    node_count = normal_factor.size
    if extension.shape != (node_count, node_count - 2):
        raise ValueError("physical-mode extension shape differs")
    if physical_mode.shape != (node_count - 2,):
        raise ValueError("physical-mode vector shape differs")
    if (
        np.any(~np.isfinite(extension))
        or np.any(~np.isfinite(normal_factor))
        or np.any(~np.isfinite(physical_mode))
        or np.any(normal_factor <= 0)
    ):
        raise ValueError("physical-mode mapping values differ")
    return extension @ (physical_mode / normal_factor[1:-1])


def dyadic_backoff(attempt, initial_step, minimum_step):
    """Run a prospectively bounded dyadic retry sequence.

    ``attempt(step)`` must return an exact mapping with ``success``, ``reason``,
    ``metrics``, and ``payload``.  Only the payload of the first successful
    attempt is returned; archived attempt records never carry numerical state.
    """
    initial_step = float(initial_step)
    minimum_step = float(minimum_step)
    if not (
        np.isfinite(initial_step) and np.isfinite(minimum_step)
        and initial_step > 0 and minimum_step > 0
        and initial_step >= minimum_step
    ):
        raise ValueError("invalid dyadic backoff bounds")
    step = initial_step
    attempts = []
    while step >= minimum_step:
        result = attempt(step)
        if set(result) != {"success", "reason", "metrics", "payload"}:
            raise ValueError("dyadic attempt schema differs")
        if type(result["success"]) is not bool or type(result["reason"]) is not str:
            raise ValueError("dyadic attempt types differ")
        if not isinstance(result["metrics"], dict):
            raise ValueError("dyadic attempt metrics differ")
        attempts.append({
            "step_size": step,
            "success": result["success"],
            "reason": result["reason"],
            "metrics": result["metrics"],
        })
        if result["success"]:
            return {
                "success": True,
                "accepted_step_size": step,
                "attempts": attempts,
                "payload": result["payload"],
            }
        step *= 0.5
    return {
        "success": False,
        "accepted_step_size": None,
        "attempts": attempts,
        "payload": None,
    }


def trapezoid(values, coordinates):
    values = np.asarray(values, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(coordinates)))


def profile_on(point, theta):
    curve = CubicSpline(
        np.asarray(point["theta"], dtype=float),
        np.asarray(point["rho"], dtype=float),
        bc_type=((1, 0.0), (1, 0.0)),
    )
    return curve(theta), curve(theta, 1)


def product_norm(theta, rho_component, tau_component):
    theta = np.asarray(theta, dtype=float)
    rho_component = np.asarray(rho_component, dtype=float)
    length = float(theta[-1] - theta[0])
    spatial = trapezoid(rho_component * rho_component, theta) / length
    return math.sqrt(max(spatial + float(tau_component) ** 2, 0.0))


def normalized_secant(previous, current, theta):
    previous_rho, _ = profile_on(previous, theta)
    current_rho, _ = profile_on(current, theta)
    delta_rho = current_rho - previous_rho
    delta_tau = float(current["tau"] - previous["tau"])
    norm = product_norm(theta, delta_rho, delta_tau)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("degenerate continuation secant")
    return delta_rho / norm, delta_tau / norm, norm


def pseudo_arclength_step(
    second_derivative, previous, current, step_size, *, nodes=121,
    tolerance=2e-6, maximum_nodes=6000, dense_nodes=501,
):
    """Take one augmented BVP pseudo-arclength step.

    ``second_derivative(tau, theta, rho, slope)`` supplies the physical local
    surface ODE.  Time is dimensionless here; the caller owns its scaling.
    """
    theta = np.linspace(1e-4, np.pi / 2.0, int(nodes))
    current_rho, current_slope = profile_on(current, theta)
    tangent_rho, tangent_tau, secant_norm = normalized_secant(previous, current, theta)
    step_size = float(step_size)
    predicted_rho = current_rho + step_size * tangent_rho
    predicted_tau = float(current["tau"] + step_size * tangent_tau)
    predicted_curve = CubicSpline(
        theta, predicted_rho, bc_type=((1, 0.0), (1, 0.0)),
    )
    tangent_curve = CubicSpline(theta, tangent_rho, bc_type="natural")
    length = float(theta[-1] - theta[0])
    state = np.vstack((predicted_rho, predicted_curve(theta, 1), np.zeros_like(theta)))

    def equation(angle, values, parameter):
        rho, slope = values[0], values[1]
        predictor = predicted_curve(angle)
        tangent = tangent_curve(angle)
        integral = (rho - predictor) * tangent / length
        return np.vstack((slope, second_derivative(float(parameter[0]), angle, rho, slope), integral))

    def boundary(left, right, parameter):
        return np.array((
            left[1], right[1], left[2],
            right[2] + (float(parameter[0]) - predicted_tau) * tangent_tau,
        ))

    solved = solve_bvp(
        equation, boundary, theta, state, p=np.array((predicted_tau,)),
        tol=float(tolerance), max_nodes=int(maximum_nodes), verbose=0,
    )
    dense_theta = np.linspace(theta[0], theta[-1], int(dense_nodes))
    values = solved.sol(dense_theta)
    derivative = solved.sol(dense_theta, 1)
    tau = float(solved.p[0])
    rho, slope = values[0], values[1]
    second = derivative[1]
    ode_second = second_derivative(tau, dense_theta, rho, slope)
    predicted_dense = predicted_curve(dense_theta)
    tangent_dense = tangent_curve(dense_theta)
    arclength = float(
        trapezoid((rho - predicted_dense) * tangent_dense, dense_theta)
        / (dense_theta[-1] - dense_theta[0])
        + (tau - predicted_tau) * tangent_tau
    )
    point = {"theta": dense_theta, "rho": rho, "slope": slope, "tau": tau}
    return {
        "success": bool(solved.success and np.all(np.isfinite(values)) and np.isfinite(tau)),
        "message": str(solved.message),
        "iterations": int(solved.niter),
        "mesh_nodes_used": int(len(solved.x)),
        "point": point,
        "predicted_tau": predicted_tau,
        "corrected_tau": tau,
        "secant_norm": float(secant_norm),
        "arclength_residual": abs(arclength),
        "boundary_slope_error": float(max(abs(slope[0]), abs(slope[-1]))),
        "ode_second_defect": float(np.max(np.abs(second - ode_second))),
    }


def principal_modes(matrix):
    matrix = np.asarray(matrix, dtype=float)
    values, left_vectors, right_vectors = eig(matrix, left=True, right=True)
    order = np.argsort(values.real)
    values = values[order]
    left_vectors = left_vectors[:, order]
    right_vectors = right_vectors[:, order]
    value = values[0]
    right = right_vectors[:, 0].astype(complex, copy=True)
    left = left_vectors[:, 0].astype(complex, copy=True)
    pivot = int(np.argmax(np.abs(right)))
    phase = np.exp(-1j * np.angle(right[pivot]))
    right *= phase
    left /= np.conj(phase)
    right /= max(float(np.max(np.abs(right))), 1e-300)
    overlap = np.vdot(left, right)
    if abs(overlap) <= 1e-14:
        raise ValueError("left/right principal modes are orthogonal")
    left /= np.conj(overlap)
    return {
        "eigenvalue": value,
        "next_eigenvalue": values[1],
        "right": right,
        "left": left,
        "overlap": np.vdot(left, right),
    }


def linear_square_root_fit(times, separations):
    times = np.asarray(times, dtype=float)
    separations = np.asarray(separations, dtype=float)
    if times.ndim != 1 or times.shape != separations.shape or len(times) < 3:
        raise ValueError("square-root fit needs at least three paired points")
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(separations)) or np.any(separations <= 0):
        raise ValueError("invalid square-root fit data")
    design = np.column_stack((times, np.ones_like(times)))
    slope, intercept = np.linalg.lstsq(design, separations**2, rcond=None)[0]
    prediction = slope * times + intercept
    observed = separations**2
    ss_residual = float(np.sum((observed - prediction) ** 2))
    ss_total = float(np.sum((observed - np.mean(observed)) ** 2))
    r_squared = 1.0 - ss_residual / max(ss_total, 1e-300)
    critical_time = float(-intercept / slope) if slope != 0 else math.nan
    positive = times > critical_time
    exponent = math.nan
    if np.count_nonzero(positive) >= 3:
        exponent = float(np.polyfit(
            np.log(times[positive] - critical_time),
            np.log(separations[positive]), 1,
        )[0])
    return {
        "slope": float(slope), "intercept": float(intercept),
        "critical_time": critical_time, "R_squared": float(r_squared),
        "log_exponent": exponent,
    }
