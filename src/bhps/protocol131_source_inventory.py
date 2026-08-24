"""Exact transitive local-source inventory for Protocol 131.

The inventory is derived from Python import syntax starting at the production
runner and its local ``bhps`` dependencies.  It is evaluated before candidate
directory creation and on every restart, so an omitted or substituted local
module cannot be authorized by a merely nonempty manifest.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "bhps"
RUNNER_PATH = PROJECT_ROOT / "run_A790_protocol131_postmortem.py"
INPUT_PATHS = {
    "environment:protocol125-runtime-contract": (
        PROJECT_ROOT / "protocol125_runtime_environment_contract.json"
    ),
    "environment:pyproject": PROJECT_ROOT / "pyproject.toml",
    "environment:uv-lock": PROJECT_ROOT / "uv.lock",
    "input:protocol125-protocol": (
        PROJECT_ROOT / "notes/125_A790_joint_parent_builder_protocol.md"
    ),
    "input:protocol128-protocol": (
        PROJECT_ROOT
        / "notes/128_A790_protocol125_recovery_corrected_rerun_protocol.md"
    ),
    "input:protocol128-independent-review": (
        PROJECT_ROOT / "notes/129_A790_protocol128_independent_freeze_review.md"
    ),
    "input:protocol128-freeze-authority-snapshot": (
        PROJECT_ROOT / "protocol128_freeze_authority_snapshot.json"
    ),
    "input:protocol128-parent-N0": (
        PROJECT_ROOT
        / "results/corrected_A790_joint_parent_rebuild_recovery_v2/parent_N0.npz"
    ),
    "input:protocol128-parent-N1": (
        PROJECT_ROOT
        / "results/corrected_A790_joint_parent_rebuild_recovery_v2/parent_N1.npz"
    ),
    "input:protocol128-adjudication": (
        PROJECT_ROOT
        / "results/corrected_A790_joint_parent_rebuild_recovery_v2/adjudication_final.json"
    ),
    "input:protocol128-recovery-index": (
        PROJECT_ROOT
        / "results/corrected_A790_joint_parent_rebuild_recovery_v2/recovery_index.json"
    ),
    "input:sealed-protocol120-parent": (
        PROJECT_ROOT
        / "results/corrected_A790_matched_staged_continuum_recovery/phase_a_parent_projection.npz"
    ),
    "input:frozen-family-knot-state": (
        PROJECT_ROOT / "results/corrected_family_knot_A8_state.npz"
    ),
}


class Protocol131SourceInventoryError(RuntimeError):
    """Raised when the local production-source closure is not exact."""


def _module_path(module_name):
    if module_name == "bhps":
        return PACKAGE_ROOT / "__init__.py"
    if not module_name.startswith("bhps."):
        raise Protocol131SourceInventoryError(
            f"local module is outside bhps: {module_name}"
        )
    return PACKAGE_ROOT.joinpath(*module_name[5:].split(".")).with_suffix(".py")


def _imported_local_modules(module_name, path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise Protocol131SourceInventoryError(
            f"cannot parse local source {module_name}"
        ) from error
    package_parts = module_name.split(".")[:-1]
    imported = set()
    for node in ast.walk(tree):
        candidates = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level > len(package_parts):
                    raise Protocol131SourceInventoryError(
                        f"invalid relative import in {module_name}"
                    )
                base = package_parts[:len(package_parts)-node.level+1]
                if node.module:
                    base.extend(node.module.split("."))
                candidates.append(".".join(base))
            elif node.module:
                candidates.append(node.module)
            if candidates == ["bhps"]:
                candidates.extend(
                    f"bhps.{alias.name}" for alias in node.names
                    if alias.name != "*"
                )
        for candidate in candidates:
            if candidate == "bhps":
                imported.add(candidate)
            elif candidate.startswith("bhps.") and _module_path(candidate).is_file():
                imported.add(candidate)
    return imported


def _transitive_modules():
    reached = {"bhps"}
    pending = list(
        _imported_local_modules(
            "run_A790_protocol131_postmortem", RUNNER_PATH,
        )
    )
    while pending:
        module_name = pending.pop()
        if module_name in reached:
            continue
        path = _module_path(module_name)
        if path.is_symlink() or not path.is_file():
            raise Protocol131SourceInventoryError(
                f"local source is unavailable: {module_name}"
            )
        reached.add(module_name)
        pending.extend(
            imported for imported in _imported_local_modules(module_name, path)
            if imported not in reached
        )
    return tuple(sorted(reached))


def protocol131_source_inventory():
    """Return the exact logical-name to absolute-path production closure."""
    if RUNNER_PATH.is_symlink() or not RUNNER_PATH.is_file():
        raise Protocol131SourceInventoryError(
            "Protocol-131 production runner is unavailable"
        )
    result = {"runner:protocol131": str(RUNNER_PATH.resolve())}
    for module_name in _transitive_modules():
        logical_name = (
            "source:bhps-package-init"
            if module_name == "bhps" else f"source:{module_name}"
        )
        path = _module_path(module_name)
        if path.is_symlink() or not path.is_file():
            raise Protocol131SourceInventoryError(
                f"Protocol-131 source is unavailable: {module_name}"
            )
        result[logical_name] = str(path.resolve())
    if len(set(result.values())) != len(result):
        raise Protocol131SourceInventoryError(
            "Protocol-131 source inventory reuses a path"
        )
    return MappingProxyType(dict(sorted(result.items())))


def validate_protocol131_source_manifest(manifest):
    """Require a freeze-authority manifest to match the exact source closure."""
    if not isinstance(manifest, Mapping):
        raise Protocol131SourceInventoryError(
            "Protocol-131 source manifest is not a mapping"
        )
    expected = protocol131_source_inventory()
    if set(manifest) != set(expected):
        missing = tuple(sorted(set(expected)-set(manifest)))
        extra = tuple(sorted(set(manifest)-set(expected)))
        raise Protocol131SourceInventoryError(
            f"Protocol-131 source manifest differs: missing={missing}, extra={extra}"
        )
    for logical_name, expected_path in expected.items():
        entry = manifest[logical_name]
        if not isinstance(entry, Mapping) or str(entry.get("path", "")) != expected_path:
            raise Protocol131SourceInventoryError(
                f"Protocol-131 source path differs: {logical_name}"
            )
    return expected


def protocol131_input_inventory():
    """Return the exact immutable-input logical-name/path inventory."""
    result = {}
    for logical_name, path in sorted(INPUT_PATHS.items()):
        if path.is_symlink() or not path.is_file():
            raise Protocol131SourceInventoryError(
                f"Protocol-131 immutable input is unavailable: {logical_name}"
            )
        result[logical_name] = str(path.resolve())
    if len(set(result.values())) != len(result):
        raise Protocol131SourceInventoryError(
            "Protocol-131 input inventory reuses a path"
        )
    return MappingProxyType(result)


def validate_protocol131_input_manifest(manifest):
    """Require a freeze-authority manifest to match all immutable inputs."""
    if not isinstance(manifest, Mapping):
        raise Protocol131SourceInventoryError(
            "Protocol-131 input manifest is not a mapping"
        )
    expected = protocol131_input_inventory()
    if set(manifest) != set(expected):
        missing = tuple(sorted(set(expected)-set(manifest)))
        extra = tuple(sorted(set(manifest)-set(expected)))
        raise Protocol131SourceInventoryError(
            f"Protocol-131 input manifest differs: missing={missing}, extra={extra}"
        )
    for logical_name, expected_path in expected.items():
        entry = manifest[logical_name]
        if not isinstance(entry, Mapping) or str(entry.get("path", "")) != expected_path:
            raise Protocol131SourceInventoryError(
                f"Protocol-131 input path differs: {logical_name}"
            )
    return expected


__all__ = [
    "Protocol131SourceInventoryError",
    "protocol131_input_inventory",
    "protocol131_source_inventory",
    "validate_protocol131_input_manifest",
    "validate_protocol131_source_manifest",
]
