#!/usr/bin/env python3
"""Restartable four-tube physical collar pilot for Test 14D."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, sha256_file
from bhps.test14d_physical_collar import (
    FAMILIES,
    RATE_PATHS,
    RESOLUTIONS,
    WIDTH_RATIOS,
    physical_collar_record,
    physical_pilot_assessment,
)


PROTOCOL = Path("notes/104_A790_test14D_manufactured_collar_erratum.md")
NOTE103 = Path("notes/103_A790_test14D_thick_collar_israel_rate_protocol.md")
TEST14B = Path("results/corrected_A790_test14b_balance_closure.json")
TEST14C = Path("results/corrected_A790_test14c_coupled_seam.json")
CONTROLS = Path("results/corrected_A790_test14d_manufactured_controls_v3.json")
COLLAR_SOURCE = Path("src/bhps/test14d_thick_collar.py")
SOURCE = Path("src/bhps/test14d_physical_collar.py")
RECOVERY_SOURCE = Path("src/bhps/recovery_indexer.py")
OUTPUT = Path("results/corrected_A790_test14d_physical_pilot.json")
MANIFEST = Path("results/corrected_A790_test14d_pilot_recovery_v1.json")
STAGES = Path("results/corrected_A790_test14d_pilot_stages")
PILOT_TIME = 0.001


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
    index.register(stage_id, "test14d-physical-pilot", maximum_seconds, metadata)
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
    with tempfile.TemporaryDirectory(prefix="bhps-test14d-pilot-") as directory:
        root = Path(directory)
        source = root / "input.txt"
        source.write_text("fixed Test-14D pilot input\n")
        expected = {str(source): sha256_file(source)}
        manifest = root / "manifest.json"
        output = root / "stage.json"
        first = RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        first.register("done", "smoke", 10.0, {"stage": "done"})
        first.mark_running("done")
        atomic_write_json(output, {"finite": True, "value": 1})
        first.mark_complete("done", output, 0.01)
        first.register("interrupted", "smoke", 10.0, {"stage": "interrupted"})
        first.mark_running("interrupted")
        second = RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        resumed = second.validated_path("done") == output
        reset = second.data["stages"]["interrupted"]["status"] == "pending"
        atomic_write_json(output, {"finite": True, "value": 2})
        rejected = second.validated_path("done") is None
        return {
            "valid_partial_restart": bool(resumed),
            "interrupted_stage_reset": bool(reset),
            "corruption_rejected": bool(rejected),
            "passed": bool(resumed and reset and rejected),
        }


def main():
    STAGES.mkdir(parents=True, exist_ok=True)
    required = (
        PROTOCOL, NOTE103, TEST14B, TEST14C, CONTROLS, COLLAR_SOURCE,
        SOURCE, RECOVERY_SOURCE, Path(__file__),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Test-14D pilot inputs: {missing}")
    expected = {str(path): sha256_file(path) for path in required}
    controls = json.loads(CONTROLS.read_text())
    if not controls.get("passed"):
        raise RuntimeError("Test-14D manufactured controls are not passed")
    recovery = recovery_qualification()
    if not recovery["passed"]:
        raise RuntimeError("Test-14D pilot recovery qualification failed")
    index = RecoveryIndex(MANIFEST, PROTOCOL, expected, 60.0)

    test14b = json.loads(TEST14B.read_text())
    test14c = json.loads(TEST14C.read_text())
    times = np.asarray(test14c["times"], dtype=float)
    matches = np.flatnonzero(np.isclose(times, PILOT_TIME, rtol=0.0, atol=1e-12))
    if len(matches) != 1:
        raise RuntimeError("Test-14D pilot time is not unique")
    time_index = int(matches[0])
    records = []
    for grid in ("G7", "G8"):
        for branch in ("inner", "outer"):
            thin = test14c["physical_records"][grid][branch]["1"][time_index]
            base = test14b["balance_records"][grid][branch]["1"][time_index]
            for family in FAMILIES:
                for ratio in WIDTH_RATIOS:
                    ratio_label = f"{ratio:.8f}".replace(".", "p")
                    for resolution in RESOLUTIONS:
                        for rate_path in RATE_PATHS:
                            stage_id = (
                                f"pilot/{grid}/{branch}/{family}/"
                                f"width{ratio_label}/npe{resolution}/{rate_path}"
                            )
                            path = STAGES / (
                                f"pilot_{grid}_{branch}_{family}_width{ratio_label}_"
                                f"npe{resolution}_{rate_path}.json"
                            )
                            records.append(stage_json(
                                index, stage_id, path,
                                lambda tr=thin, br=base, fam=family, wr=ratio,
                                res=resolution, rp=rate_path: physical_collar_record(
                                    tr, br, fam, wr, res, rp,
                                ),
                                {
                                    "phase": "physical_pilot",
                                    "grid": grid,
                                    "branch": branch,
                                    "time": PILOT_TIME,
                                    "stride": 1,
                                    "profile": family,
                                    "epsilon_over_RA": ratio,
                                    "resolution": resolution,
                                    "rate_path": rate_path,
                                },
                            ))

    assessment = stage_json(
        index, "pilot/assessment", STAGES / "pilot_assessment.json",
        lambda: physical_pilot_assessment(records),
        {
            "phase": "physical_pilot_assessment",
            "time": PILOT_TIME,
            "record_count": len(records),
        },
    )
    result = {
        "schema": "bhps-test14d-physical-pilot-result-v1",
        "status": "complete",
        "pilot_time": PILOT_TIME,
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "base_protocol_sha256": sha256_file(NOTE103),
        "inputs": expected,
        "recovery": recovery,
        "records": records,
        "assessment": assessment,
        "claim_boundary": (
            "Local collar and archived compatibility pilot only; not a "
            "dynamically evolved thick brane."
        ),
    }
    atomic_write_json(OUTPUT, result)
    print(json.dumps({
        "status": result["status"],
        "pilot_grade": assessment["pilot_grade"],
        "numerical_collar_subgrade": assessment["numerical_collar_subgrade"],
        "physical_israel_rate_subgrade": assessment[
            "physical_israel_rate_subgrade"
        ],
        "gates": assessment["gates"],
        "summary": assessment["summary"],
        "output": str(OUTPUT),
        "output_sha256": sha256_file(OUTPUT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
