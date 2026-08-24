from __future__ import annotations

import json

import pytest

from bhps.protocol131_environment_contract import (
    CONTRACT_PATH,
    Protocol131EnvironmentContractError,
    build_protocol131_environment_contract,
    validate_protocol131_environment_contract,
)


def test_checked_in_protocol131_runtime_matches_active_environment():
    recorded = validate_protocol131_environment_contract()
    assert recorded["protocol_identifier"].startswith("Protocol-131-")
    assert recorded["mpmath"]["version"] == "1.3.0"
    assert recorded["longdouble"]["itemsize"] == 8
    assert recorded["longdouble"]["mantissa_bits"] == 52


def test_runtime_contract_builder_reproduces_checked_in_record():
    expected = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert build_protocol131_environment_contract() == expected


def test_runtime_contract_tamper_fails_closed(tmp_path):
    payload = build_protocol131_environment_contract()
    payload["mpmath"]["version"] = "tampered"
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Protocol131EnvironmentContractError):
        validate_protocol131_environment_contract(path)
