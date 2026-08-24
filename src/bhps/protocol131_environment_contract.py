"""Exact additional runtime contract for the Protocol-131 diagnostics.

Protocol 131 reuses the already frozen Protocol-125 Python/NumPy/SciPy/BLAS
contract and additionally depends on mpmath, the platform long-double format,
and numerical thread-control environment variables.  Validation is read-only
and never rewrites the checked-in contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import mpmath
import numpy as np

from bhps.joint_parent_environment_contract import (
    CONTRACT_PATH as PROTOCOL125_CONTRACT_PATH,
    validate_protocol125_environment_contract,
)


PROTOCOL_IDENTIFIER = "Protocol-131-runtime-environment-contract-v1"
CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "protocol131_runtime_environment_contract.json"
)
_TOP_LEVEL_KEYS = (
    "protocol_identifier",
    "protocol125_runtime",
    "mpmath",
    "longdouble",
    "thread_controls",
    "fingerprint",
)
_THREAD_VARIABLES = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class Protocol131EnvironmentContractError(RuntimeError):
    """Raised when the active diagnostic runtime differs from the freeze."""


def _sha256_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Protocol131EnvironmentContractError(
            "Protocol-125 runtime contract is not a regular file"
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value, label="runtime"):
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise Protocol131EnvironmentContractError(
                f"{label} contains a non-string key"
            )
        return {
            key: _json_value(value[key], f"{label}/{key}")
            for key in value
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item, f"{label}[]") for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise Protocol131EnvironmentContractError(
        f"{label} contains unsupported value {type(value).__name__}"
    )


def _fingerprint(payload):
    encoded = json.dumps(
        _json_value(payload), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_protocol131_runtime_environment():
    """Return the active Protocol-131 additions to the base runtime."""
    base = validate_protocol125_environment_contract()
    info = np.finfo(np.longdouble)
    payload = {
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "protocol125_runtime": {
            "contract_sha256": _sha256_file(PROTOCOL125_CONTRACT_PATH),
            "fingerprint": str(base["fingerprint"]),
        },
        "mpmath": {"version": str(mpmath.__version__)},
        "longdouble": {
            "dtype": np.dtype(np.longdouble).name,
            "epsilon": str(info.eps),
            "itemsize": int(np.dtype(np.longdouble).itemsize),
            "mantissa_bits": int(info.nmant),
        },
        "thread_controls": {
            name: os.environ.get(name) for name in _THREAD_VARIABLES
        },
    }
    return _json_value(payload)


def build_protocol131_environment_contract():
    """Build the exact contract for the active diagnostic runtime."""
    payload = capture_protocol131_runtime_environment()
    return {**payload, "fingerprint": _fingerprint(payload)}


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Protocol131EnvironmentContractError(
                f"runtime contract repeats key {key}"
            )
        result[key] = value
    return result


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def validate_protocol131_environment_contract(path=CONTRACT_PATH):
    """Strictly compare a frozen regular JSON file with the active runtime."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Protocol131EnvironmentContractError(
            "Protocol-131 runtime contract is not a regular file"
        )
    try:
        recorded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Protocol131EnvironmentContractError(
                    f"runtime contract contains {token}"
                )
            ),
        )
    except Protocol131EnvironmentContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Protocol131EnvironmentContractError(
            "Protocol-131 runtime contract is unreadable"
        ) from error
    if not isinstance(recorded, Mapping) or tuple(recorded) != _TOP_LEVEL_KEYS:
        raise Protocol131EnvironmentContractError(
            "Protocol-131 runtime contract schema differs"
        )
    payload = {key: recorded[key] for key in _TOP_LEVEL_KEYS[:-1]}
    if str(recorded["fingerprint"]) != _fingerprint(payload):
        raise Protocol131EnvironmentContractError(
            "Protocol-131 runtime fingerprint differs"
        )
    expected = build_protocol131_environment_contract()
    if recorded != expected:
        changed = tuple(
            key for key in _TOP_LEVEL_KEYS[:-1]
            if recorded.get(key) != expected.get(key)
        )
        raise Protocol131EnvironmentContractError(
            f"active Protocol-131 runtime differs from freeze: {changed}"
        )
    return _freeze(recorded)


__all__ = [
    "CONTRACT_PATH",
    "Protocol131EnvironmentContractError",
    "build_protocol131_environment_contract",
    "capture_protocol131_runtime_environment",
    "validate_protocol131_environment_contract",
]
