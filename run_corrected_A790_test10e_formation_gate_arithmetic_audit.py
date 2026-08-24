#!/usr/bin/env python3
"""Post-run arithmetic audit of the sealed Test 10E formation gate."""

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bhps.recovery_indexer import atomic_write_json, sha256_file


RESULT = Path("results/corrected_A790_test10e_genuine_high_z_boundary_resolution.json")
FORMATION = Path(
    "results/corrected_A790_test10e_genuine_high_z_boundary_resolution_recovery/"
    "analysis_formation.json"
)
OUTPUT = Path("results/corrected_A790_test10e_formation_gate_arithmetic_audit.json")
RESULT_SHA256 = "69edd095a38dff64333813711e67797493446e044fea789a3636f31f4be26cf0"
FORMATION_SHA256 = "9881cf48c4dfdc6bb9dcb59de8742c9f5f99a6babce2b3a60b81d9cad9c831b4"
ABSOLUTE_TOLERANCE = 1e-15
EXPECTED_BRACKET = (0.001125, 0.00125)


def audit(formation):
    bracket_records = {}
    bracket_gate = True
    for label, measured in formation["brackets"].items():
        differences = [abs(float(left) - right) for left, right in zip(measured, EXPECTED_BRACKET)]
        close = all(
            math.isclose(float(left), right, rel_tol=0.0, abs_tol=ABSOLUTE_TOLERANCE)
            for left, right in zip(measured, EXPECTED_BRACKET)
        )
        bracket_records[label] = {
            "measured": measured,
            "expected": list(EXPECTED_BRACKET),
            "absolute_differences": differences,
            "within_arithmetic_tolerance": close,
        }
        bracket_gate &= close
    corrected_gate = bool(
        formation["initial_zero"]
        and formation["primary_history_gate"]
        and formation["temporal_formation_gate"]
        and bracket_gate
    )
    return {
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": 0.0,
        "brackets": bracket_records,
        "tolerance_bracket_gate": bool(bracket_gate),
        "original_sealed_gate": bool(formation["gate"]),
        "arithmetic_corrected_formation_gate": corrected_gate,
    }


def main():
    if sha256_file(RESULT) != RESULT_SHA256:
        raise RuntimeError("Test10E final result identity changed")
    if sha256_file(FORMATION) != FORMATION_SHA256:
        raise RuntimeError("Test10E formation analysis identity changed")
    result = json.loads(RESULT.read_text())
    formation = json.loads(FORMATION.read_text())["analysis"]
    arithmetic = audit(formation)
    failed_independent_gates = sorted(
        key for key, value in result["acceptance"].items()
        if not value and key != "formation"
    )
    payload = {
        "scope": "post-run arithmetic audit; not a protocol regrade",
        "sealed_protocol_sha256": result["protocol_sha256"],
        "sealed_result_sha256": RESULT_SHA256,
        "sealed_formation_analysis_sha256": FORMATION_SHA256,
        "arithmetic": arithmetic,
        "sealed_classification": result["classification"],
        "failed_independent_gates": failed_independent_gates,
        "classification_unchanged": bool(failed_independent_gates),
        "interpretation": (
            "The formation histories and bracket agree at floating-point precision; "
            "correcting this arithmetic comparison does not cure the boundary-audit failures."
        ),
    }
    atomic_write_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

