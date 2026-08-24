#!/usr/bin/env python3
"""Resume sealed Test 10E after a presentation-only surface-key mismatch."""

from pathlib import Path

import run_corrected_A790_test10e_genuine_high_z_boundary_resolution as runner
from bhps.recovery_indexer import sha256_file


ADDENDUM = Path("notes/110_A790_test10E_operational_recovery_addendum.md")
ADDENDUM_SHA256 = "2e63fc8d68d0c3e5a9bc1e8a191344b24c2703ac8ddcdc23c2ee8c945bd526b0"
_ORIGINAL_PUBLIC_SURFACE = runner.test10b.public_bvp_surface


def public_surface_with_unavailable_runtime(surface):
    """Supply only the presentation metadata omitted by recover_surface."""
    if "error" in surface or "runtime_seconds" in surface:
        return _ORIGINAL_PUBLIC_SURFACE(surface)
    presented = dict(surface)
    presented["runtime_seconds"] = None
    return _ORIGINAL_PUBLIC_SURFACE(presented)


def main():
    if sha256_file(ADDENDUM) != ADDENDUM_SHA256:
        raise RuntimeError("Test10E operational recovery addendum identity changed")
    runner.test10b.public_bvp_surface = public_surface_with_unavailable_runtime
    runner.main()


if __name__ == "__main__":
    main()

