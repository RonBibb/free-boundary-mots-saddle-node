#!/usr/bin/env python3
"""Authorized Protocol-228 entry point; validate controls before numerical imports."""
from __future__ import annotations

import os
import sys


THREAD_VARS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)

if not sys.dont_write_bytecode:
    raise SystemExit("Protocol228 requires Python -B")
if any(os.environ.get(name) != "1" for name in THREAD_VARS):
    raise SystemExit("Protocol228 requires all five thread controls to equal 1")
if "numpy" in sys.modules or "scipy" in sys.modules:
    raise SystemExit("numerical modules were imported before Protocol228 bootstrap")

import protocol228

protocol228.main()
