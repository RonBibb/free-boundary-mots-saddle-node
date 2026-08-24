from __future__ import annotations

from pathlib import Path

import pytest

from bhps.protocol131_source_inventory import (
    Protocol131SourceInventoryError,
    protocol131_input_inventory,
    protocol131_source_inventory,
    validate_protocol131_input_manifest,
    validate_protocol131_source_manifest,
)


def _manifest_from_inventory():
    return {
        name: {"path": path, "sha256": "0" * 64}
        for name, path in protocol131_source_inventory().items()
    }


def test_inventory_is_exact_unique_regular_source_closure():
    inventory = protocol131_source_inventory()
    assert len(inventory) == 63
    assert len(set(inventory.values())) == len(inventory)
    assert "runner:protocol131" in inventory
    for required in (
        "source:bhps.protocol131_environment_contract",
        "source:bhps.protocol131_freeze_authority",
        "source:bhps.protocol131_postmortem",
        "source:bhps.protocol131_precision",
        "source:bhps.protocol131_source_inventory",
        "source:bhps.joint_parent_adjudication",
    ):
        assert required in inventory
    assert all(Path(path).is_file() for path in inventory.values())


def test_exact_manifest_paths_validate():
    manifest = _manifest_from_inventory()
    assert validate_protocol131_source_manifest(manifest) == protocol131_source_inventory()


def test_omitted_extra_or_substituted_source_fails_closed(tmp_path):
    manifest = _manifest_from_inventory()
    manifest.pop("source:bhps.protocol131_precision")
    with pytest.raises(Protocol131SourceInventoryError, match="missing"):
        validate_protocol131_source_manifest(manifest)

    manifest = _manifest_from_inventory()
    manifest["source:extra"] = next(iter(manifest.values())).copy()
    with pytest.raises(Protocol131SourceInventoryError, match="extra"):
        validate_protocol131_source_manifest(manifest)

    manifest = _manifest_from_inventory()
    fake = tmp_path / "fake.py"
    fake.write_text("pass\n", encoding="utf-8")
    manifest["source:bhps.protocol131_precision"]["path"] = str(fake)
    with pytest.raises(Protocol131SourceInventoryError, match="path differs"):
        validate_protocol131_source_manifest(manifest)


def test_input_inventory_is_exact_unique_and_complete():
    inventory = protocol131_input_inventory()
    assert len(inventory) == 13
    assert len(set(inventory.values())) == len(inventory)
    for required in (
        "input:protocol128-parent-N0",
        "input:protocol128-parent-N1",
        "input:protocol128-adjudication",
        "input:protocol128-recovery-index",
        "input:protocol128-freeze-authority-snapshot",
        "input:sealed-protocol120-parent",
        "input:frozen-family-knot-state",
        "environment:protocol125-runtime-contract",
    ):
        assert required in inventory


def test_input_manifest_must_match_exact_paths():
    manifest = {
        name: {"path": path, "sha256": "0" * 64}
        for name, path in protocol131_input_inventory().items()
    }
    assert validate_protocol131_input_manifest(manifest) == protocol131_input_inventory()
    manifest.pop("input:protocol128-parent-N1")
    with pytest.raises(Protocol131SourceInventoryError, match="missing"):
        validate_protocol131_input_manifest(manifest)
