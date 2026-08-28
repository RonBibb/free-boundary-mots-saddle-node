#!/usr/bin/env python3
"""Authorized Protocol 249 bootstrap."""

import os
import sys


CONTROLS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
if not sys.dont_write_bytecode: raise SystemExit("Protocol 249 requires Python -B")
if any(os.environ.get(name) != "1" for name in CONTROLS): raise SystemExit("Protocol 249 requires all five thread controls to equal 1")
if "numpy" in sys.modules or "scipy" in sys.modules: raise SystemExit("numerical modules were imported before Protocol 249 bootstrap")

import runner

runner.main()
