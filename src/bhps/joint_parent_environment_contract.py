"""Exact runtime-environment contract for Protocol 125.

The dependency lock files constrain installable packages, but they do not
identify the interpreter ABI, host architecture, or the numerical libraries
actually linked into NumPy and SciPy.  This module builds a deterministic JSON
record for those runtime facts and verifies the checked-in record byte-for-
semantic-byte before the production adapter exposes its freeze inventory.

No scientific state is constructed here.  Updating the repository contract is
an explicit pre-freeze maintenance action; validation never rewrites it.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import struct
import sys
import sysconfig
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import scipy


PROTOCOL_IDENTIFIER = "Protocol-125-runtime-environment-contract-v1"
CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "protocol125_runtime_environment_contract.json"
)
_TOP_LEVEL_KEYS = (
    "protocol_identifier",
    "interpreter",
    "platform",
    "numeric_libraries",
    "fingerprint",
)


class Protocol125EnvironmentContractError(RuntimeError):
    """Raised when the recorded and active numerical environments differ."""


def _json_value(value, path="environment"):
    if isinstance(value, Mapping):
        if any(type(name) is not str or not name for name in value):
            raise Protocol125EnvironmentContractError(
                f"{path} contains an invalid mapping key"
            )
        return {
            name: _json_value(value[name], f"{path}/{name}")
            for name in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        return [
            _json_value(item, f"{path}/{index}")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise Protocol125EnvironmentContractError(
                f"{path} contains a nonfinite float"
            )
        return value
    raise Protocol125EnvironmentContractError(
        f"{path} contains unsupported value {type(value).__name__}"
    )


def _blas_lapack_configuration(show_config, library_name):
    try:
        configuration = show_config(mode="dicts")
    except Exception as error:
        raise Protocol125EnvironmentContractError(
            f"{library_name} cannot report its build configuration"
        ) from error
    if not isinstance(configuration, Mapping):
        raise Protocol125EnvironmentContractError(
            f"{library_name} build configuration is not a mapping"
        )
    dependencies = configuration.get("Build Dependencies")
    if not isinstance(dependencies, Mapping):
        raise Protocol125EnvironmentContractError(
            f"{library_name} omits Build Dependencies"
        )
    result = {}
    for backend in ("blas", "lapack"):
        record = dependencies.get(backend)
        if not isinstance(record, Mapping) or not record:
            raise Protocol125EnvironmentContractError(
                f"{library_name} omits its {backend.upper()} configuration"
            )
        result[backend] = _json_value(
            record, f"{library_name}/{backend}",
        )
    return result


def capture_protocol125_runtime_environment():
    """Return the canonical active interpreter and numerical-runtime record."""
    version = sys.version_info
    payload = {
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "interpreter": {
            "implementation_name": str(sys.implementation.name),
            "implementation_label": platform.python_implementation(),
            "version": platform.python_version(),
            "version_info": [
                int(version.major),
                int(version.minor),
                int(version.micro),
                str(version.releaselevel),
                int(version.serial),
            ],
            "cache_tag": str(sys.implementation.cache_tag),
            "abi": {
                "soabi": sysconfig.get_config_var("SOABI"),
                "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
                "multiarch": sysconfig.get_config_var("MULTIARCH"),
                "abi_flags": str(getattr(sys, "abiflags", "")),
                "gil_disabled": sysconfig.get_config_var("Py_GIL_DISABLED"),
                "byteorder": str(sys.byteorder),
                "pointer_bits": int(8*struct.calcsize("P")),
            },
        },
        "platform": {
            "sys_platform": str(sys.platform),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "numeric_libraries": {
            "numpy": {
                "version": str(np.__version__),
                "blas_lapack": _blas_lapack_configuration(
                    np.show_config, "NumPy",
                ),
            },
            "scipy": {
                "version": str(scipy.__version__),
                "blas_lapack": _blas_lapack_configuration(
                    scipy.show_config, "SciPy",
                ),
            },
        },
    }
    return _json_value(payload)


def _fingerprint(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_protocol125_environment_contract():
    """Generate the exact contract expected for the active runtime."""
    runtime = capture_protocol125_runtime_environment()
    payload = {
        name: runtime[name] for name in _TOP_LEVEL_KEYS[:-1]
    }
    return {
        **payload,
        "fingerprint": _fingerprint(payload),
    }


def _reject_duplicate_keys(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise Protocol125EnvironmentContractError(
                f"environment contract repeats key {name}"
            )
        result[name] = value
    return result


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(name): _freeze(item) for name, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def validate_protocol125_environment_contract(path=CONTRACT_PATH):
    """Strictly compare a regular checked-in contract with the active runtime."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Protocol125EnvironmentContractError(
            "runtime environment contract is not a regular file"
        )
    try:
        recorded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                Protocol125EnvironmentContractError(
                    f"environment contract contains {value}"
                )
            ),
        )
    except Protocol125EnvironmentContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Protocol125EnvironmentContractError(
            "runtime environment contract is unreadable"
        ) from error
    if not isinstance(recorded, Mapping) or tuple(recorded) != _TOP_LEVEL_KEYS:
        raise Protocol125EnvironmentContractError(
            "runtime environment contract schema differs"
        )
    payload = {
        name: recorded[name] for name in _TOP_LEVEL_KEYS[:-1]
    }
    if str(recorded["fingerprint"]) != _fingerprint(payload):
        raise Protocol125EnvironmentContractError(
            "runtime environment contract fingerprint differs"
        )
    expected = build_protocol125_environment_contract()
    if recorded != expected:
        changed = tuple(
            name for name in _TOP_LEVEL_KEYS[:-1]
            if recorded.get(name) != expected.get(name)
        )
        raise Protocol125EnvironmentContractError(
            f"active runtime differs from the frozen environment contract: {changed}"
        )
    return _freeze(recorded)
