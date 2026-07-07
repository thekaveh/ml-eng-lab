# tests/test_manifest.py
from __future__ import annotations

import pytest

from scripts.docs.manifest import (
    ManifestError,
    Section,
    parse_manifest,
    load_manifest,
)


VALID = """
surfaces: [repo, site, wiki]
numbering: baked
sections:
  - id: overview
    number: "1"
    title: Overview
    source: docs/index.md
  - id: architecture
    number: "2"
    title: Architecture
    children:
      - id: system
        number: "2.1"
        title: System & context view
        source: docs/architecture.md
        diagrams: [system]
notebooks:
  - task: tabular_classification-iris-mlp-pytorch
    number: "8.1"
    family: tabular
    depth: full
    doc: docs/notebooks/tabular_classification-iris-mlp-pytorch.md
    spec: notebooks/tabular_classification-iris-mlp-pytorch/docs/spec.yaml
    diagrams: [mlp-architecture]
diagrams:
  - id: system
    master: docs/diagrams/ml-eng-lab-system.html
"""


def test_parse_manifest_returns_dataclasses():
    m = parse_manifest(VALID)
    assert isinstance(m, Manifest := type(m))  # noqa: F841
    assert m.surfaces == ("repo", "site", "wiki")
    assert m.numbering == "baked"
    assert len(m.sections) == 2
    arch = m.sections[1]
    assert isinstance(arch, Section)
    assert arch.id == "architecture" and arch.number == "2"
    assert len(arch.children) == 1 and arch.children[0].number == "2.1"
    assert m.notebooks[0].task == "tabular_classification-iris-mlp-pytorch"
    assert m.diagrams[0].id == "system"


def test_parse_manifest_rejects_non_baked_numbering():
    with pytest.raises(ManifestError, match="numbering"):
        parse_manifest(VALID.replace("numbering: baked", "numbering: auto"))


def test_load_manifest_validates_files_exist(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/index.md").write_text("# 1. Overview\n", encoding="utf-8")
    (tmp_path / "manifest.yaml").write_text(VALID, encoding="utf-8")
    with pytest.raises(ManifestError, match="source.*does not exist"):
        load_manifest(tmp_path / "manifest.yaml", tmp_path)


def test_load_manifest_passes_when_files_exist(tmp_path):
    for p in [
        "docs/index.md",
        "docs/architecture.md",
        "docs/notebooks/tabular_classification-iris-mlp-pytorch.md",
        "notebooks/tabular_classification-iris-mlp-pytorch/docs/spec.yaml",
        "docs/diagrams/ml-eng-lab-system.html",
    ]:
        path = tmp_path / p
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    (tmp_path / "manifest.yaml").write_text(VALID, encoding="utf-8")
    m = load_manifest(tmp_path / "manifest.yaml", tmp_path)
    assert m.sections[0].source == "docs/index.md"


def test_parse_manifest_wraps_malformed_yaml():
    with pytest.raises(ManifestError, match="not valid YAML"):
        parse_manifest("surfaces: [repo, site, wiki\n  : : bad")


def test_parse_manifest_wraps_missing_required_key():
    bad = VALID.replace("    depth: full\n", "")  # notebook entry missing `depth`
    with pytest.raises(ManifestError, match="missing required key"):
        parse_manifest(bad)
