#!/usr/bin/env python3
"""Numerical-runtime bootstrap for the prospective t/ell=0.016 study."""

from __future__ import annotations

import os
import sys
from pathlib import Path


EXPECTED_CONTROLS = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def main():
    if not sys.flags.dont_write_bytecode:
        raise SystemExit("authorized launch requires python -B")
    if "numpy" in sys.modules or "scipy" in sys.modules:
        raise SystemExit("NumPy/SciPy was imported before the runtime bootstrap")
    differences = {
        key: os.environ.get(key)
        for key, expected in EXPECTED_CONTROLS.items()
        if os.environ.get(key) != expected
    }
    if differences:
        raise SystemExit(f"thread controls differ: {differences}")
    root = Path(__file__).resolve().parent
    if Path.cwd().resolve() != root:
        raise SystemExit(f"authorized launch directory is {root}")
    from run_corrected_A790_t016_long_time_unfolding import main as scientific_main

    scientific_main()


if __name__ == "__main__":
    main()
