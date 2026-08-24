from __future__ import annotations

import json

import pytest

import bhps.joint_parent_environment_contract as environment


def test_repository_environment_contract_exactly_matches_active_runtime():
    generated = environment.build_protocol125_environment_contract()
    validated = environment.validate_protocol125_environment_contract()

    assert validated["fingerprint"] == generated["fingerprint"]
    assert tuple(validated["interpreter"]["version_info"]) == tuple(
        generated["interpreter"]["version_info"]
    )
    assert validated["protocol_identifier"] == environment.PROTOCOL_IDENTIFIER
    assert validated["interpreter"]["implementation_name"] == "cpython"
    assert validated["interpreter"]["abi"]["soabi"]
    assert validated["platform"]["machine"]
    for library in ("numpy", "scipy"):
        record = validated["numeric_libraries"][library]
        assert record["version"]
        assert tuple(record["blas_lapack"]) == ("blas", "lapack")
        assert record["blas_lapack"]["blas"]["found"] is True
        assert record["blas_lapack"]["lapack"]["found"] is True


def test_environment_contract_rejects_fingerprint_valid_runtime_drift(tmp_path):
    contract = environment.build_protocol125_environment_contract()
    contract["platform"]["machine"] = "manufactured-other-machine"
    payload = {
        name: contract[name] for name in environment._TOP_LEVEL_KEYS[:-1]
    }
    contract["fingerprint"] = environment._fingerprint(payload)
    path = tmp_path/"environment.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(
        environment.Protocol125EnvironmentContractError,
        match="active runtime differs",
    ):
        environment.validate_protocol125_environment_contract(path)


def test_environment_contract_rejects_duplicate_or_nonfinite_json(tmp_path):
    duplicate = tmp_path/"duplicate.json"
    duplicate.write_text(
        '{"protocol_identifier":"x","protocol_identifier":"y"}',
        encoding="utf-8",
    )
    with pytest.raises(
        environment.Protocol125EnvironmentContractError,
        match="repeats key",
    ):
        environment.validate_protocol125_environment_contract(duplicate)

    nonfinite = tmp_path/"nonfinite.json"
    nonfinite.write_text(
        '{"protocol_identifier":NaN}', encoding="utf-8",
    )
    with pytest.raises(
        environment.Protocol125EnvironmentContractError,
        match="contains NaN",
    ):
        environment.validate_protocol125_environment_contract(nonfinite)
