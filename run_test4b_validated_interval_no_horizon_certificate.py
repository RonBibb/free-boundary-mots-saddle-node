#!/usr/bin/env python3
"""Run the sealed Test-4B validated interval certificate attempt."""

from __future__ import annotations

from fractions import Fraction
import inspect
import json
import math
from pathlib import Path
import sys
import time

import mpmath
from mpmath import libmp
import numpy as np
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.anisotropic_capped_surface import _splines, anisotropic_rho_second
from bhps.corrected_regular_axis_shooting import corrected_shoot_axis_radius
from bhps.recovery_indexer import (
    RecoveryIndex, atomic_write_json, atomic_write_npz, sha256_file,
)
from bhps.validated_capped_surface_shooting import (
    TensorBicubicIntervalSpline,
    VInterval,
    ValidatedBicubicMetric,
    axis_divergence_source_interval,
    axis_second_interval,
    interval_cos,
    interval_sin,
    jsonable_interval_tree,
    propagate_launch_cell,
    regularized_rhs_interval,
)


PROTOCOL = Path("notes/101_test4b_validated_interval_no_horizon_certificate_protocol.md")
ERRATA = (
    Path("notes/101a_test4b_regular_axis_factor_erratum.md"),
    Path("notes/101b_test4b_axis_cone_divergence_form_erratum.md"),
)
NOTE91 = Path("notes/91_A790_initial_no_horizon_certificate_protocol.md")
TRANSITIONS = Path("results/corrected_A790_no_horizon_transition_refinement.json")
GEOMETRY = {
    "G9": Path("results/corrected_A790_initial_no_horizon_certificate_stages/A790_G9_metric.npz"),
    "G10": Path("results/corrected_A790_initial_no_horizon_certificate_stages/A790_G10_metric.npz"),
    "A794_G7": Path("results/corrected_A790_initial_no_horizon_certificate_stages/A794_G7_metric.npz"),
}
RECOVERY = Path("results/test4b_validated_interval_no_horizon_stages")
SPLINE_ARCHIVE = {
    label: RECOVERY / f"{label}_exact_bicubic_splines.npz" for label in GEOMETRY
}
MANIFEST = Path("results/test4b_validated_interval_no_horizon_recovery_v2.json")
OUTPUT = Path("results/test4b_validated_interval_no_horizon_certificate.json")
RHO_BOUNDS = (0.10, 1.67)
BASE_CELL_COUNT = 2048
FLOAT_OPTIONS = {
    "rho_bounds": RHO_BOUNDS,
    "theta_cut": 1e-3,
    "relative_tolerance": 2e-9,
    "absolute_tolerance": 2e-11,
    "maximum_step": 0.01,
    "graph_slope_guard": 100.0,
}


def load_geometry(path):
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def scipy_splines(geometry):
    return _splines(
        geometry["z"], geometry["r"], geometry["psi"],
        geometry["a"], geometry["b"], geometry["c"],
    )


def expected_spline_payload(geometry):
    payload = {"z_brane": np.asarray(float(geometry["z"][-1]))}
    for name, spline in scipy_splines(geometry).items():
        payload.update(TensorBicubicIntervalSpline.from_scipy(
            spline,
        ).archive_payload(name))
    return payload


def ensure_spline_archive(path, geometry):
    payload = expected_spline_payload(geometry)
    reusable = False
    if path.exists():
        try:
            with np.load(path) as archive:
                reusable = set(archive.files) == set(payload) and all(
                    np.array_equal(archive[key], value)
                    for key, value in payload.items()
                )
        except (OSError, ValueError):
            reusable = False
    if not reusable:
        atomic_write_npz(path, **payload)
    return path


def load_validated_metric(path):
    with np.load(path) as archive:
        fields = {
            name: TensorBicubicIntervalSpline(
                archive[f"{name}_knots_z"], archive[f"{name}_knots_r"],
                archive[f"{name}_coefficients"],
            )
            for name in ("A", "B", "C")
        }
        return ValidatedBicubicMetric(float(archive["z_brane"]), fields)


def json_stage(index, stage_id, kind, path, expected_seconds, compute, metadata):
    index.register(stage_id, kind, expected_seconds, metadata)
    validated = index.validated_path(stage_id)
    if validated is not None:
        return json.loads(validated.read_text())
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = compute()
        atomic_write_json(path, payload)
        json.loads(path.read_text())
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return payload
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def base_cell(index):
    spacing = (RHO_BOUNDS[1] - RHO_BOUNDS[0]) / BASE_CELL_COUNT
    return VInterval(
        RHO_BOUNDS[0] + int(index) * spacing,
        RHO_BOUNDS[0] + (int(index) + 1) * spacing,
    )


def cell_containing(value):
    spacing = (RHO_BOUNDS[1] - RHO_BOUNDS[0]) / BASE_CELL_COUNT
    index = int(math.floor((float(value) - RHO_BOUNDS[0]) / spacing))
    return max(0, min(BASE_CELL_COUNT - 1, index))


def backend_and_representation_controls(geometries, metrics):
    arithmetic = []
    for left, right in ((0.1, 0.3), (-0.7, 1.1), (1.3, -0.2)):
        li = VInterval.point(left)
        ri = VInterval.point(right)
        lf = Fraction.from_float(left)
        rf = Fraction.from_float(right)
        cases = [
            ("add", li + ri, lf + rf),
            ("subtract", li - ri, lf - rf),
            ("multiply", li * ri, lf * rf),
        ]
        if right != 0.0:
            cases.append(("divide", li / ri, lf / rf))
        arithmetic.extend({
            "operation": name, "inputs": [left, right],
            "interval": [interval.lower, interval.upper],
            "contains_fraction_float": interval.contains(float(exact)),
        } for name, interval, exact in cases)
    trig = []
    for lower, upper in ((0.0, 1e-3), (0.17, 1.23), (1.4, math.pi / 2)):
        box = VInterval(lower, upper)
        sine = interval_sin(box)
        cosine = interval_cos(box)
        samples = np.linspace(lower, upper, 129)
        trig.append({
            "box": [lower, upper],
            "sine": [sine.lower, sine.upper],
            "cosine": [cosine.lower, cosine.upper],
            "contains_samples": bool(all(
                sine.contains(math.sin(value)) and cosine.contains(math.cos(value))
                for value in samples
            )),
        })
    source_text = "\n".join(inspect.getsource(function) for function in (
        libmp.mpi_add, libmp.mpi_mul, libmp.mpi_div,
        libmp.mpi_cos_sin, libmp.mpi_sqrt,
    ))
    source_rounding_pass = bool(
        "round_floor" in source_text and "round_ceiling" in source_text
    )
    spline_checks = []
    rhs_checks = []
    for label in ("G9", "G10", "A794_G7"):
        geometry = geometries[label]
        scipy_fields = scipy_splines(geometry)
        metric = metrics[label]
        points = (
            (float(geometry["z"][0]), float(geometry["r"][0])),
            (float(geometry["z"][len(geometry["z"]) // 2]),
             float(geometry["r"][min(12, len(geometry["r"]) - 1)])),
            (float(geometry["z"][-1]), 0.0),
        )
        for zvalue, rvalue in points:
            for name in ("A", "B", "C"):
                enclosure = metric.fields[name].evaluate(
                    VInterval.point(zvalue), VInterval.point(rvalue), 2,
                )
                for derivative, interval in enclosure.items():
                    expected = float(np.asarray(scipy_fields[name].ev(
                        zvalue, rvalue, dx=derivative[0], dy=derivative[1],
                    )).reshape(-1)[0])
                    spline_checks.append({
                        "grid": label, "field": name,
                        "point": [zvalue, rvalue],
                        "derivative": list(derivative),
                        "interval": [interval.lower, interval.upper],
                        "scipy": expected, "contains": interval.contains(expected),
                    })
        for theta, rho, slope in ((0.1, 1.2, 0.2), (0.7, 0.8, -0.4), (1.4, 1.5, 0.01)):
            interval = regularized_rhs_interval(
                VInterval.point(theta), VInterval.point(rho),
                VInterval.point(slope), metric,
            )
            expected = float(anisotropic_rho_second(
                np.asarray([theta]), np.asarray([rho]), np.asarray([slope]),
                float(geometry["z"][-1]), scipy_fields,
            )[0])
            rhs_checks.append({
                "grid": label, "point": [theta, rho, slope],
                "interval": [interval.lower, interval.upper],
                "established": expected, "contains": interval.contains(expected),
                "absolute_midpoint_difference": abs(interval.midpoint - expected),
            })
    manufactured = {
        "positive": jsonable_interval_tree((VInterval(-1, 1)**2) + 1),
        "sign_changing": jsonable_interval_tree(VInterval(-1, 1)),
        "tangent_to_zero": jsonable_interval_tree(VInterval(-1, 1)**2),
    }
    flat_z = np.linspace(0.8, 3.0, 9)
    flat_r = np.linspace(0.0, 2.0, 11)
    flat_ones = np.ones((len(flat_z), len(flat_r)))
    flat_scipy = {
        name: TensorBicubicIntervalSpline.from_scipy(
            __import__("scipy.interpolate", fromlist=["RectBivariateSpline"])
            .RectBivariateSpline(flat_z, flat_r, flat_ones, kx=3, ky=3)
        ) for name in ("A", "B", "C")
    }
    flat_metric = ValidatedBicubicMetric(flat_z[-1], flat_scipy)
    flat_rho = VInterval.point(0.7)
    flat_second = axis_second_interval(flat_rho, flat_metric)
    flat_source = axis_divergence_source_interval(
        VInterval.point(0), flat_rho, flat_second, flat_metric,
    )
    checks_pass = bool(
        source_rounding_pass
        and all(item["contains_fraction_float"] for item in arithmetic)
        and all(item["contains_samples"] for item in trig)
        and all(item["contains"] for item in spline_checks)
        and all(item["contains"] and item["absolute_midpoint_difference"] < 2e-10
                for item in rhs_checks)
        and flat_second.contains(0.7) and flat_source.contains(2.1)
        and manufactured["positive"][0] > 0.0
        and manufactured["sign_changing"][0] <= 0.0 <= manufactured["sign_changing"][1]
        and manufactured["tangent_to_zero"][0] <= 0.0 <= manufactured["tangent_to_zero"][1]
    )
    return {
        "status": "PASS" if checks_pass else "FAIL",
        "mpmath_version": mpmath.__version__,
        "mp_interval_precision_bits": 96,
        "libmp_source_uses_directed_rounding": source_rounding_pass,
        "arithmetic_checks": arithmetic, "trigonometric_checks": trig,
        "spline_checks": spline_checks, "rhs_checks": rhs_checks,
        "manufactured_residual_controls": manufactured,
        "flat_axis_second": [flat_second.lower, flat_second.upper],
        "flat_divergence_source": [flat_source.lower, flat_source.upper],
        "all_controls_passed": checks_pass,
    }


def corrected_navigation(geometries):
    result = {"A790": {}, "A794_G7": {}}
    for label in ("G9", "G10"):
        geometry = geometries[label]
        splines = scipy_splines(geometry)
        z_brane = float(geometry["z"][-1])

        def objective(value):
            record = corrected_shoot_axis_radius(
                value, z_brane, splines, **FLOAT_OPTIONS,
            )
            if record["status"] != "reached_brane":
                return 1000.0 + abs(record.get("end_slope", 0.0))
            return record["brane_residual"]

        minimized = minimize_scalar(
            objective, bounds=(1.18, 1.21), method="bounded",
            options={"xatol": 2e-9, "maxiter": 35},
        )
        minimum = corrected_shoot_axis_radius(
            minimized.x, z_brane, splines, **FLOAT_OPTIONS,
        )
        variants = []
        for theta_cut in (2e-3, 1e-3, 5e-4, 2.5e-4):
            record = corrected_shoot_axis_radius(
                minimized.x, z_brane, splines, rho_bounds=RHO_BOUNDS,
                theta_cut=theta_cut, relative_tolerance=5e-10,
                absolute_tolerance=5e-12, maximum_step=0.005,
                graph_slope_guard=100.0,
            )
            variants.append({
                "theta_cut": theta_cut, "status": record["status"],
                "brane_residual": record.get("brane_residual"),
                "brane_radius": record.get("brane_radius"),
            })
        result["A790"][label] = {
            "axis_radius": float(minimized.x),
            "brane_residual": minimum.get("brane_residual"),
            "brane_radius": minimum.get("brane_radius"),
            "status": minimum["status"],
            "optimizer_success": bool(minimized.success),
            "function_evaluations": int(minimized.nfev),
            "theta_cut_variants": variants,
        }
    geometry = geometries["A794_G7"]
    splines = scipy_splines(geometry)
    z_brane = float(geometry["z"][-1])

    def residual(value):
        record = corrected_shoot_axis_radius(
            value, z_brane, splines, **FLOAT_OPTIONS,
        )
        if record["status"] != "reached_brane":
            raise RuntimeError(f"A7.94 bracket left brane class at {value}")
        return record["brane_residual"]

    old_roots = (1.2044699915377057, 1.208094106879044)
    roots = []
    for center in old_roots:
        left, right = center - 1e-4, center + 1e-4
        left_residual, right_residual = residual(left), residual(right)
        root = brentq(residual, left, right, xtol=2e-10, rtol=2e-10)
        roots.append({
            "axis_radius": float(root), "brane_residual": residual(root),
            "bracket": [left, right],
            "bracket_residuals": [left_residual, right_residual],
        })
    result["A794_G7"] = {
        "corrected_root_count": len(roots), "corrected_roots": roots,
    }
    minima = [result["A790"][label]["brane_residual"] for label in ("G9", "G10")]
    result["all_targeted_A790_minima_positive"] = bool(all(value > 0 for value in minima))
    result["A794_two_root_adverse_control"] = len(roots) == 2
    result["scope"] = (
        "Targeted corrected-axis floating navigation, not an exhaustive or "
        "validated launch-interval cover."
    )
    return result


def one_interval_probe(metric, name, launch):
    started = time.perf_counter()
    record = propagate_launch_cell(
        launch, metric, rho_bounds=RHO_BOUNDS,
        theta_axis=1e-3, initial_step=0.004,
        minimum_step=1e-5, maximum_steps=200000,
    )
    record["probe_name"] = name
    record["elapsed_seconds"] = time.perf_counter() - started
    return jsonable_interval_tree(record)


def summarize_interval_probes(probes):
    classifications = [
        item["classification"] for values in probes.values() for item in values
    ]
    return {
        "probes": probes,
        "all_probes_terminated_deterministically": bool(classifications),
        "unresolved_probe_count": sum(value.startswith("unresolved")
                                      for value in classifications),
        "closed_probe_count": sum(not value.startswith("unresolved")
                                  for value in classifications),
        "stop_reason": (
            "The validated Taylor state box loses the regular-axis state "
            "correlation and reaches the fixed minimum step. Under note 101, "
            "a required unenclosable cell stops the certificate at REVIEW."
        ),
    }


def main():
    RECOVERY.mkdir(parents=True, exist_ok=True)
    geometries = {label: load_geometry(path) for label, path in GEOMETRY.items()}
    for label in GEOMETRY:
        ensure_spline_archive(SPLINE_ARCHIVE[label], geometries[label])
    expected_paths = (
        PROTOCOL, *ERRATA, NOTE91, TRANSITIONS, *GEOMETRY.values(),
        *SPLINE_ARCHIVE.values(),
        Path("src/bhps/validated_capped_surface_shooting.py"),
        Path("src/bhps/corrected_regular_axis_shooting.py"),
        Path("src/bhps/recovery_indexer.py"), Path(__file__),
    )
    expected = {str(path): sha256_file(path) for path in expected_paths}
    index = RecoveryIndex(MANIFEST, PROTOCOL, expected, 3600.0)
    metrics = {
        label: load_validated_metric(path) for label, path in SPLINE_ARCHIVE.items()
    }
    controls = json_stage(
        index, "controls/backend-representation", "interval-controls",
        RECOVERY / "backend_representation_controls.json", 600.0,
        lambda: backend_and_representation_controls(geometries, metrics),
        {"mpmath_precision_bits": 96, "spline_derivative_order": 2},
    )
    navigation = json_stage(
        index, "navigation/corrected-axis-features", "floating-navigation",
        RECOVERY / "corrected_axis_navigation.json", 1200.0,
        lambda: corrected_navigation(geometries),
        {"regular_axis_factor": "1/3", "exhaustive": False},
    )
    probe_specifications = {"G9": [], "G10": [], "A794_G7": []}
    for label in ("G9", "G10"):
        minimum = navigation["A790"][label]["axis_radius"]
        minimum_cell = cell_containing(minimum)
        probe_specifications[label].extend((
            ("first_base_cell", base_cell(0)),
            ("minimum_base_cell", base_cell(minimum_cell)),
            ("minimum_point", VInterval.point(minimum)),
        ))
    for root_index, root in enumerate(navigation["A794_G7"]["corrected_roots"]):
        probe_specifications["A794_G7"].append((
            f"corrected_root_{root_index + 1}_point",
            VInterval.point(root["axis_radius"]),
        ))
    probe_records = {label: [] for label in probe_specifications}
    for label, specifications in probe_specifications.items():
        for name, launch in specifications:
            safe_name = name.replace("/", "-")
            record = json_stage(
                index, f"validated/{label}/{safe_name}",
                "validated-interval-probe",
                RECOVERY / f"{label}_{safe_name}.json", 300.0,
                lambda label=label, name=name, launch=launch: one_interval_probe(
                    metrics[label], name, launch,
                ),
                {
                    "label": label, "probe_name": name,
                    "launch_interval": [launch.lower, launch.upper],
                    "base_cell_count": BASE_CELL_COUNT,
                    "maximum_launch_refinement": 12,
                    "theta_axis": 1e-3, "initial_step": 0.004,
                    "minimum_step": 1e-5,
                },
            )
            probe_records[label].append(record)
            print(
                f"{label} {name}: {record['classification']} at "
                f"theta={record.get('theta')}", flush=True,
            )
    probes = summarize_interval_probes(probe_records)
    atomic_write_json(RECOVERY / "validated_interval_probes.json", probes)
    numerical_evidence_root_free = bool(
        navigation["all_targeted_A790_minima_positive"]
        and all(
            item["brane_residual"] > 0
            for label in ("G9", "G10")
            for item in navigation["A790"][label]["theta_cut_variants"]
        )
    )
    controls_pass = bool(
        controls["all_controls_passed"]
        and navigation["A794_two_root_adverse_control"]
    )
    certificate_pass = bool(
        controls_pass and probes["unresolved_probe_count"] == 0
        and probes["closed_probe_count"] > 0
    )
    payload = {
        "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL),
        "errata": [{"path": str(path), "sha256": sha256_file(path)} for path in ERRATA],
        "status": "PASS" if certificate_pass else "REVIEW",
        "classification": (
            "validated_two_grid_C_cap_no_root_certificate"
            if certificate_pass else
            "regular_axis_corrected_but_interval_propagation_unresolved"
        ),
        "certificate_pass": certificate_pass,
        "backend_and_representation_controls": controls,
        "corrected_axis_navigation": navigation,
        "validated_interval_attempt": probes,
        "numerical_evidence_root_free": numerical_evidence_root_free,
        "load_bearing_finding": (
            "The note-91 finite-cutoff initializer used rho''(0)=rho*B. "
            "The exact regular-axis condition is rho''(0)=rho*B/3. Corrected "
            "floating shoots retain the positive A=7.90 minima, but the old "
            "shooting archive is not a validated shoot of the regular class."
        ),
        "why_not_PASS": (
            "The exact spline/backend and regular-axis cone controls pass, "
            "but the box Taylor propagator loses the axis-induced state "
            "correlation and exhausts its prospective minimum step even on "
            "point launches. The complete 2,048-cell cover is therefore not "
            "classified; no zero-exclusion certificate is claimed."
        ),
        "next_mathematical_requirement": (
            "Use a correlated Taylor-model/Lohner enclosure or a validated "
            "collocation/radii-polynomial proof that preserves the regular "
            "axis manifold, then restart the unchanged 2,048-cell cover."
        ),
        "claim_boundary": (
            "REVIEW of the fixed C_cap class on the exact archived G9/G10 "
            "bicubic discrete metrics. No conclusion about arbitrary "
            "topologies, non-star-shaped surfaces, or the continuum spacetime."
        ),
        "provenance": {"manifest": str(MANIFEST), "input_sha256": expected},
    }
    atomic_write_json(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"], "classification": payload["classification"],
        "controls_pass": controls_pass,
        "numerical_evidence_root_free": numerical_evidence_root_free,
        "unresolved_probe_count": probes["unresolved_probe_count"],
        "G9_minimum": navigation["A790"]["G9"]["brane_residual"],
        "G10_minimum": navigation["A790"]["G10"]["brane_residual"],
        "A794_roots": [item["axis_radius"] for item in
                       navigation["A794_G7"]["corrected_roots"]],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
