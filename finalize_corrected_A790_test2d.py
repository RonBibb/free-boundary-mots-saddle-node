#!/usr/bin/env python3
"""Assemble the independently verified, prospectively sealed Test-2D result."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, atomic_write_npz, sha256_file


PROTOCOL = Path("notes/110_A790_test2D_high_order_ragged_chart_protocol.md")
PROTOCOL_SHA256 = "f11989b23cff2d5b87bf6e730ff91c77b1e095eb2a7a49d1a277a9d3bf2666e5"
QUALIFICATION = Path("results/corrected_A790_test2d_high_order_ragged_chart_qualification_v3.json")
CHART_MANIFEST = Path("results/corrected_A790_test2d_high_order_ragged_chart_recovery/chart_index_v2.json")
FIELDS = Path("results/corrected_A790_test2d_high_order_ragged_chart_fields.json")
SURFACES = Path("results/corrected_A790_test2d_high_order_ragged_chart_surfaces.json")
COMMON_PARENT = Path("results/corrected_A790_test2d_high_order_ragged_chart_common_parent.json")
AUDIT = Path("results/corrected_A790_test2d_high_order_ragged_chart_independent_audit.json")
OUTPUT = Path("results/corrected_A790_test2d_high_order_ragged_chart.json")
STATE_OUTPUT = Path("results/corrected_A790_test2d_high_order_ragged_chart_state.npz")
ROOT = Path("results/corrected_A790_test2d_high_order_ragged_chart_final_recovery")
MANIFEST = ROOT / "index.json"


def classify(audit):
    if not audit.get("verifier_passed"):
        return "REVIEW", "invalid_test2d_numerical_audit"
    status = audit["status"]
    expected = {
        "PASS": "high_order_ragged_chart_above_first_order_continuum_evidence",
        "REVIEW": "high_order_ragged_chart_convergence_mixed",
        "FAIL": "high_order_ragged_chart_nonconvergence_or_branch_failure",
    }
    if status not in expected or audit.get("classification") != expected[status]:
        return "REVIEW", "invalid_test2d_numerical_audit"
    return status, expected[status]


def main():
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("sealed Test-2D protocol hash mismatch")
    sources = (QUALIFICATION, CHART_MANIFEST, FIELDS, SURFACES, COMMON_PARENT, AUDIT)
    payloads = {str(path): json.loads(path.read_text()) for path in sources}
    audit = payloads[str(AUDIT)]
    status, classification = classify(audit)
    ROOT.mkdir(parents=True, exist_ok=True)
    inputs = {
        str(path): sha256_file(path) for path in (*sources, Path(__file__),
                                                   Path("verify_corrected_A790_test2d.py"))
    }
    index = RecoveryIndex(MANIFEST, PROTOCOL, inputs, maximum_stage_seconds=300.0)
    result = {
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA256,
        "status": status, "classification": classification,
        "qualification": payloads[str(QUALIFICATION)],
        "chart_manifest": str(CHART_MANIFEST),
        "chart_manifest_sha256": sha256_file(CHART_MANIFEST),
        "fields": payloads[str(FIELDS)], "surfaces": payloads[str(SURFACES)],
        "common_parent": payloads[str(COMMON_PARENT)],
        "independent_audit": audit,
        "source_hashes": inputs,
        "claim_boundary": (
            "SO(3)-symmetric foliation-dependent continuum evidence only; "
            "no event-horizon, topology, throat, mass-transfer, halo, or cosmological claim"
        ),
    }
    atomic_write_json(OUTPUT, result)
    atomic_write_npz(
        STATE_OUTPUT, protocol_sha256=np.asarray(PROTOCOL_SHA256),
        status=np.asarray(status), classification=np.asarray(classification),
    )
    index.register("final/result", "test2d-final-result", 300.0, {})
    if index.validated_path("final/result") is None:
        index.mark_running("final/result")
        final = ROOT / "final.json"
        atomic_write_json(final, {
            "protocol_sha256": PROTOCOL_SHA256,
            "status": status, "classification": classification,
            "output": str(OUTPUT), "output_sha256": sha256_file(OUTPUT),
            "state": str(STATE_OUTPUT), "state_sha256": sha256_file(STATE_OUTPUT),
            "audit": str(AUDIT), "audit_sha256": sha256_file(AUDIT),
        })
        index.mark_complete("final/result", final, 0.0)
    print(json.dumps({
        "status": status, "classification": classification,
        "output": str(OUTPUT), "output_sha256": sha256_file(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
