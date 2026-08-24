#!/usr/bin/env python3
"""Restartable dense primary-window Test-14D collar evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, sha256_file
from bhps.test14d_dense_collar import (
    BRANCHES,
    FAMILIES,
    GRIDS,
    PRIMARY_LEFT,
    PRIMARY_RIGHT,
    RATE_PATHS,
    STRIDES,
    WIDTH_RATIOS,
    dense_assessment,
    dense_physical_collar_record,
)


PROTOCOL = Path("notes/104_A790_test14D_manufactured_collar_erratum.md")
NOTE103 = Path("notes/103_A790_test14D_thick_collar_israel_rate_protocol.md")
TEST14B = Path("results/corrected_A790_test14b_balance_closure.json")
TEST14C = Path("results/corrected_A790_test14c_coupled_seam.json")
CONTROLS = Path("results/corrected_A790_test14d_manufactured_controls_v3.json")
PILOT = Path("results/corrected_A790_test14d_physical_pilot.json")
COLLAR_SOURCE = Path("src/bhps/test14d_thick_collar.py")
PHYSICAL_SOURCE = Path("src/bhps/test14d_physical_collar.py")
SOURCE = Path("src/bhps/test14d_dense_collar.py")
RECOVERY_SOURCE = Path("src/bhps/recovery_indexer.py")
OUTPUT = Path("results/corrected_A790_test14d_dense_collar.json")
MANIFEST = Path("results/corrected_A790_test14d_dense_recovery_v1.json")
STAGES = Path("results/corrected_A790_test14d_dense_stages")
RESOLUTION = 128


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
    index.register(stage_id, "test14d-dense-collar", maximum_seconds, metadata)
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
    with tempfile.TemporaryDirectory(prefix="bhps-test14d-dense-") as directory:
        root = Path(directory)
        source = root / "input.txt"
        source.write_text("fixed Test-14D dense input\n")
        expected = {str(source): sha256_file(source)}
        manifest = root / "manifest.json"
        output = root / "stage.json"
        first = RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        first.register("done", "smoke", 10.0, {"case": "done"})
        first.mark_running("done")
        atomic_write_json(output, {"finite": True})
        first.mark_complete("done", output, 0.01)
        first.register("interrupted", "smoke", 10.0, {"case": "interrupted"})
        first.mark_running("interrupted")
        second = RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        resumed = second.validated_path("done") == output
        reset = second.data["stages"]["interrupted"]["status"] == "pending"
        atomic_write_json(output, {"finite": False})
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
        PROTOCOL, NOTE103, TEST14B, TEST14C, CONTROLS, PILOT,
        COLLAR_SOURCE, PHYSICAL_SOURCE, SOURCE, RECOVERY_SOURCE, Path(__file__),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Test-14D dense inputs: {missing}")
    expected = {str(path): sha256_file(path) for path in required}
    controls = json.loads(CONTROLS.read_text())
    pilot = json.loads(PILOT.read_text())
    if not controls.get("passed"):
        raise RuntimeError("Test-14D controls are not passed")
    if pilot["assessment"]["numerical_collar_subgrade"] != "PASS":
        raise RuntimeError("Test-14D pilot numerical collar did not pass")
    recovery = recovery_qualification()
    if not recovery["passed"]:
        raise RuntimeError("Test-14D dense recovery qualification failed")
    index = RecoveryIndex(MANIFEST, PROTOCOL, expected, 60.0)

    test14b = json.loads(TEST14B.read_text())
    test14c = json.loads(TEST14C.read_text())
    all_times = np.asarray(test14c["times"], dtype=float)
    time_indices = [
        index for index, current in enumerate(all_times)
        if PRIMARY_LEFT - 1e-12 <= current <= PRIMARY_RIGHT + 1e-12
    ]
    records = []
    for grid in GRIDS:
        for branch in BRANCHES:
            for stride in STRIDES:
                thin_history = test14c["physical_records"][grid][branch][str(stride)]
                base_history = test14b["balance_records"][grid][branch][str(stride)]
                for time_index in time_indices:
                    current_time = float(all_times[time_index])
                    time_label = f"{current_time:.6f}".replace(".", "p")
                    thin = thin_history[time_index]
                    base = base_history[time_index]
                    for family in FAMILIES:
                        for ratio in WIDTH_RATIOS:
                            ratio_label = f"{ratio:.8f}".replace(".", "p")
                            for rate_path in RATE_PATHS:
                                stage_id = (
                                    f"dense/{grid}/{branch}/stride{stride}/{time_label}/"
                                    f"{family}/width{ratio_label}/{rate_path}"
                                )
                                path = STAGES / (
                                    f"dense_{grid}_{branch}_stride{stride}_{time_label}_"
                                    f"{family}_width{ratio_label}_{rate_path}.json"
                                )
                                records.append(stage_json(
                                    index, stage_id, path,
                                    lambda tr=thin, br=base, fam=family, wr=ratio,
                                    rp=rate_path: dense_physical_collar_record(
                                        tr, br, fam, wr, RESOLUTION, rp,
                                    ),
                                    {
                                        "phase": "dense_primary",
                                        "grid": grid,
                                        "branch": branch,
                                        "time": current_time,
                                        "stride": stride,
                                        "profile": family,
                                        "epsilon_over_RA": ratio,
                                        "resolution": RESOLUTION,
                                        "rate_path": rate_path,
                                    },
                                ))

    assessment = stage_json(
        index, "dense/assessment", STAGES / "dense_assessment.json",
        lambda: dense_assessment(records),
        {
            "phase": "dense_assessment",
            "record_count": len(records),
            "primary_left": PRIMARY_LEFT,
            "primary_right": PRIMARY_RIGHT,
        },
    )
    result = {
        "schema": "bhps-test14d-dense-result-v1",
        "status": "complete",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "base_protocol_sha256": sha256_file(NOTE103),
        "inputs": expected,
        "recovery": recovery,
        "times": [float(all_times[index]) for index in time_indices],
        "records": records,
        "assessment": assessment,
        "claim_boundary": (
            "Local finite-collar limit on archived thin-boundary tube data; "
            "not a thick-brane evolution or mass-transfer result."
        ),
    }
    atomic_write_json(OUTPUT, result)
    print(json.dumps({
        "status": result["status"],
        "record_count": len(records),
        "overall_grade": assessment["overall_grade"],
        "numerical_balance_subgrade": assessment["numerical_balance_subgrade"],
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
