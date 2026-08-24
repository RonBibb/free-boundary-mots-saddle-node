#!/usr/bin/env python3
"""Mechanical R8 reference-residual adapter for sealed Test-6 runner.

The pre-sealed R8 builder computes the finite-wall reference residual but does
not expose it in its returned geometry dictionary.  This wrapper recomputes
that same deterministic reference solve, attaches only the missing diagnostic
key, and then invokes the unchanged sealed runner.
"""

from __future__ import annotations

import numpy as np

import run_corrected_amplitude_basin_map as basin
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice


_sealed_construction_pass = basin.construction_pass


def construction_pass_with_exposed_reference_residual(geometries):
    for label, geometry in geometries.items():
        if "reference_maximum_residual" in geometry:
            continue
        nz, nr = map(int, geometry["source_grid"])
        amplitude = float(geometry["fold_amplitude"])
        r_max = float(np.asarray(geometry["r"])[-1])
        iterations = 270 if label == "G7" else 280
        reference = solve_finite_wall_high_order_slice(
            amplitude, nz=nz, nr=nr, r_max=r_max, wall_stiffness=20.0,
            epsilon=0.1, backreaction=0.01, tolerance=1e-10,
            iterations=iterations,
        )
        geometry["reference_maximum_residual"] = float(
            reference["max_abs_residual"]
        )
    return _sealed_construction_pass(geometries)


basin.construction_pass = construction_pass_with_exposed_reference_residual


if __name__ == "__main__":
    basin.main()
