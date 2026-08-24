#!/usr/bin/env python3
"""Controls and recovery qualification for sealed Test 4D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, sha256_file
from bhps.validated_capped_surface_shooting import VInterval
from bhps.validated_global_bvp import (
    CONFIGURATIONS,
    CHEBYSHEV_WEIGHT,
    configuration_mesh,
    finite_radii_bounds,
    first_certified_radius,
    radius_candidates,
)


PROTOCOL = Path("notes/104_test4d_validated_global_bvp_certificate_protocol.md")
SEAL = Path("results/test4d_validated_global_bvp_protocol_seal.json")
RECOVERY = Path("results/test4d_validated_global_bvp_stages")
QUALIFICATION = RECOVERY / "restart_and_control_qualification.json"
TEST_MODULE = "tests.test_test4d_validated_global_bvp"


def validate_seal():
    seal = json.loads(SEAL.read_text())
    checks = {
        "protocol_hash": sha256_file(PROTOCOL) == seal["protocol"]["sha256"],
        "no_physical_outcome_before_seal": (
            seal["physical_outcomes_inspected_before_seal"] is False
        ),
        "immutable_input_hashes": all(
            Path(path).is_file() and sha256_file(path) == expected
            for path, expected in seal["immutable_inputs"].items()
        ),
        "mesh_ladder": [
            len(configuration_mesh(configuration)) - 1
            for configuration in CONFIGURATIONS
        ] == [70, 70, 140, 140],
        "weight": CHEBYSHEV_WEIGHT == 1.05,
        "radius_endpoints": (
            radius_candidates()[0] == 1e-14
            and radius_candidates()[-1] == 1e-4
        ),
    }
    return checks


def manufactured_global_control():
    flow = np.diag([np.exp(-0.4), np.exp(0.4)])
    jacobian = np.block([
        [np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))],
        [-flow, np.eye(2), np.zeros((2, 2))],
        [np.zeros((2, 2)), -flow, np.eye(2)],
    ])
    inverse = np.linalg.inv(jacobian)
    interval_jacobian = np.asarray([
        [VInterval.point(jacobian[i, j]) for j in range(6)]
        for i in range(6)
    ], dtype=object)
    residual = [VInterval(-1e-12, 1e-12) for _ in range(6)]
    bounds = finite_radii_bounds(
        inverse, jacobian, interval_jacobian, residual,
        np.ones(6), z20=1e-3, z21=1e-3,
    )
    certificate = first_certified_radius(bounds)
    return {
        "system": "stable_and_unstable_three_node_multiple_shooting",
        "bounds": bounds,
        "certificate": certificate,
        "status": "PASS" if certificate is not None else "FAIL",
    }


def restart_qualification():
    with tempfile.TemporaryDirectory(prefix="test4d-restart-") as directory:
        root = Path(directory)
        policy = root / "policy.json"
        sealed_policy = {
            "configuration_names": [item["name"] for item in CONFIGURATIONS],
            "mesh_counts": [
                len(configuration_mesh(item)) - 1 for item in CONFIGURATIONS
            ],
            "precision_bits": [item["precision_bits"] for item in CONFIGURATIONS],
            "nu": CHEBYSHEV_WEIGHT,
            "radii": radius_candidates(),
            "launch_bounds": {"A790": [1.18, 1.209], "A794": [1.2038, 1.2087]},
        }
        atomic_write_json(policy, sealed_policy)
        expected = {str(policy): sha256_file(policy), str(SEAL): sha256_file(SEAL)}
        manifest_path = root / "manifest.json"
        index = RecoveryIndex(manifest_path, PROTOCOL, expected, 60.0)
        metadata = {"ordinal": 0, "configuration": CONFIGURATIONS[0]["name"]}
        index.register("attempt/0", "qualification-attempt", 10.0, metadata)
        index.mark_running("attempt/0")
        restarted = RecoveryIndex(manifest_path, PROTOCOL, expected, 60.0)
        interrupted_reset = (
            restarted.data["stages"]["attempt/0"]["status"] == "pending"
        )
        output = root / "attempt.json"
        atomic_write_json(output, {"status": "solution_certified"})
        restarted.mark_running("attempt/0")
        restarted.mark_complete("attempt/0", output, 0.0)
        valid_before_corruption = restarted.validated_path("attempt/0") == output
        atomic_write_json(output, {"status": "corrupted"})
        corruption_rejected = restarted.validated_path("attempt/0") is None

        changed = dict(sealed_policy)
        changed["nu"] = 1.051
        changed["precision_bits"] = [160, 192, 192, 256]
        changed["mesh_counts"] = [71, 71, 142, 142]
        atomic_write_json(policy, changed)
        policy_change_rejected = False
        try:
            RecoveryIndex(manifest_path, PROTOCOL, expected, 60.0)
        except RuntimeError:
            policy_change_rejected = True
    return {
        "running_reset_to_pending": interrupted_reset,
        "completed_output_valid_before_corruption": valid_before_corruption,
        "corrupted_completed_output_rejected": corruption_rejected,
        "mesh_precision_norm_policy_change_rejected": policy_change_rejected,
    }


def qualify():
    seal_checks = validate_seal()
    manufactured = manufactured_global_control()
    restart = restart_qualification()
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", TEST_MODULE],
        capture_output=True, text=True, timeout=120,
    )
    passed = bool(
        all(seal_checks.values())
        and manufactured["status"] == "PASS"
        and all(restart.values())
        and tests.returncode == 0
    )
    payload = {
        "status": "PASS" if passed else "FAIL",
        "physical_outcomes_evaluated": False,
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "seal": str(SEAL),
        "seal_sha256": sha256_file(SEAL),
        "seal_checks": seal_checks,
        "manufactured_global_control": manufactured,
        "restart_qualification": restart,
        "focused_tests": {
            "module": TEST_MODULE,
            "return_code": tests.returncode,
            "stdout": tests.stdout,
            "stderr": tests.stderr,
        },
        "source_sha256": {
            "runner": sha256_file(Path(__file__)),
            "global_bvp_backend": sha256_file("src/bhps/validated_global_bvp.py"),
            "recovery_backend": sha256_file("src/bhps/recovery_indexer.py"),
            "focused_tests": sha256_file(
                "tests/test_test4d_validated_global_bvp.py"
            ),
        },
    }
    RECOVERY.mkdir(parents=True, exist_ok=True)
    atomic_write_json(QUALIFICATION, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualify", action="store_true")
    arguments = parser.parse_args()
    if not arguments.qualify:
        raise SystemExit("No physical runner is enabled before qualification")
    result = qualify()
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
