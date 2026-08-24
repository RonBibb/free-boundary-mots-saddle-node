#!/usr/bin/env python3
"""Recovery-first manufactured-control runner for Test 14D."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, sha256_file
from bhps.test14d_thick_collar import (
    FAMILIES,
    RESOLUTIONS,
    WIDTH_RATIOS,
    collar_record,
    manufactured_controls,
)


PROTOCOL = Path("notes/104_A790_test14D_manufactured_collar_erratum.md")
NOTE103 = Path("notes/103_A790_test14D_thick_collar_israel_rate_protocol.md")
NOTE99 = Path("notes/99_A790_test14C_coupled_thin_seam_protocol.md")
NOTE100 = Path("notes/100_A790_test14C_intrinsic_anisotropy_addendum.md")
SOURCE = Path("src/bhps/test14d_thick_collar.py")
RECOVERY_SOURCE = Path("src/bhps/recovery_indexer.py")
OUTPUT = Path("results/corrected_A790_test14d_manufactured_controls_v3.json")
RECOVERY_OUTPUT = Path("results/corrected_A790_test14d_recovery_qualification_v3.json")
MANIFEST = Path("results/corrected_A790_test14d_recovery_v3.json")
STAGES = Path("results/corrected_A790_test14d_stages_v3")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def stage_json(index, stage_id, path, compute, metadata, maximum_seconds=60.0):
    index.register(
        stage_id, "test14d-thick-collar", maximum_seconds, metadata,
    )
    validated = index.validated_path(stage_id)
    if validated is not None:
        return json.loads(validated.read_text())
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = _jsonable(compute())
        atomic_write_json(path, payload)
        index.mark_complete(
            stage_id, path, time.perf_counter() - started,
            {"finite": bool(payload.get("finite", True))},
        )
        return payload
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def recovery_qualification():
    """Qualify resume, interrupted-stage reset, and corruption rejection."""
    with tempfile.TemporaryDirectory(prefix="bhps-test14d-recovery-") as directory:
        root = Path(directory)
        source = root / "fixed_input.txt"
        source.write_text("fixed Test-14D recovery input\n")
        expected = {str(source): sha256_file(source)}
        manifest = root / "manifest.json"
        completed = root / "completed.json"

        first = RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        first.register("complete", "smoke", 10.0, {"case": "resume"})
        first.mark_running("complete")
        atomic_write_json(completed, {"finite": True, "value": 1})
        first.mark_complete("complete", completed, 0.01)
        first.register("interrupted", "smoke", 10.0, {"case": "interrupt"})
        first.mark_running("interrupted")

        second = RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        resumed = second.validated_path("complete") == completed
        interrupted_reset = (
            second.data["stages"]["interrupted"]["status"] == "pending"
        )
        atomic_write_json(completed, {"finite": True, "value": 2})
        corruption_rejected = second.validated_path("complete") is None
        return {
            "schema": "bhps-test14d-recovery-qualification-v1",
            "valid_partial_restart": bool(resumed),
            "interrupted_stage_reset": bool(interrupted_reset),
            "corruption_rejected": bool(corruption_rejected),
            "passed": bool(resumed and interrupted_reset and corruption_rejected),
        }


def main():
    STAGES.mkdir(parents=True, exist_ok=True)
    required = (
        PROTOCOL, NOTE103, NOTE99, NOTE100, SOURCE, RECOVERY_SOURCE,
        Path(__file__),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Test-14D inputs: {missing}")
    expected = {str(path): sha256_file(path) for path in required}

    recovery = recovery_qualification()
    atomic_write_json(RECOVERY_OUTPUT, {
        **recovery,
        "protocol_sha256": sha256_file(PROTOCOL),
        "inputs": expected,
    })
    if not recovery["passed"]:
        raise RuntimeError("Test-14D recovery qualification failed")

    index = RecoveryIndex(MANIFEST, PROTOCOL, expected, 60.0)
    records = []
    area_radius = 1.7
    for family in FAMILIES:
        for ratio in WIDTH_RATIOS:
            epsilon = ratio * area_radius
            ratio_label = f"{ratio:.8f}".replace(".", "p")
            for resolution in RESOLUTIONS:
                stage_id = (
                    f"controls/{family}/width{ratio_label}/npe{resolution}"
                )
                path = STAGES / (
                    f"control_{family}_width{ratio_label}_npe{resolution}.json"
                )
                records.append(stage_json(
                    index, stage_id, path,
                    lambda fam=family, eps=epsilon, res=resolution: collar_record(
                        fam, eps, res, area_radius=area_radius,
                    ),
                    {
                        "phase": "manufactured_control",
                        "grid": "manufactured",
                        "branch": "symmetric",
                        "time": 0.0,
                        "stride": 0,
                        "profile": family,
                        "epsilon_over_RA": ratio,
                        "resolution": resolution,
                        "rate_path": "manufactured",
                    },
                ))

    controls = stage_json(
        index, "controls/assessment", STAGES / "control_assessment.json",
        lambda: manufactured_controls(records),
        {
            "phase": "manufactured_assessment",
            "record_count": len(records),
            "protocol_sha256": sha256_file(PROTOCOL),
        },
    )
    result = {
        "schema": "bhps-test14d-manufactured-result-v1",
        "status": "complete" if controls["passed"] else "failed_controls",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "inputs": expected,
        "recovery": recovery,
        "controls": controls,
        "passed": bool(recovery["passed"] and controls["passed"]),
    }
    atomic_write_json(OUTPUT, result)
    if not result["passed"]:
        failed = [name for name, passed in controls["gates"].items() if not passed]
        raise RuntimeError(f"Test-14D manufactured controls failed: {failed}")
    print(json.dumps({
        "status": result["status"],
        "protocol_sha256": result["protocol_sha256"],
        "output": str(OUTPUT),
        "output_sha256": sha256_file(OUTPUT),
        "control_summary": controls["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
