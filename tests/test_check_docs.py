# tests/test_check_docs.py
from __future__ import annotations

from scripts.docs.check_docs import (
    check_completeness,
    check_placeholders,
    check_self_containment,
)
from scripts.docs.manifest import parse_manifest

MANIFEST_YAML = """
surfaces: [repo, site, wiki]
numbering: baked
sections:
  - id: overview
    number: "1"
    title: Overview
    source: docs/index.md
notebooks:
  - task: t-iris-mlp-pytorch
    number: "8.1"
    family: tabular
    depth: full
    doc: docs/notebooks/t.md
    spec: notebooks/t/docs/spec.yaml
diagrams: []
"""


def test_self_contamination_flags_cross_surface_link(tmp_path):
    site = tmp_path / "site/page.md"
    site.parent.mkdir(parents=True)
    site.write_text("see [repo](https://github.com/thekaveh/ml-eng-lab/blob/main/README.md)", encoding="utf-8")
    findings = check_self_containment(tmp_path)
    assert any(f.severity == "error" and "site" in f.message.lower() for f in findings)


def test_self_contamination_clean(tmp_path):
    (tmp_path / "site").mkdir()
    (tmp_path / "site/page.md").write_text("see [next](next.md)", encoding="utf-8")
    assert check_self_containment(tmp_path) == []


def test_completeness_flags_missing_spec(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/index.md").write_text("x", encoding="utf-8")
    m = parse_manifest(MANIFEST_YAML)
    findings = check_completeness(m, tmp_path)
    assert any("spec" in f.message.lower() or "doc" in f.message.lower() for f in findings)


def test_placeholders_flag_tbd(tmp_path):
    (tmp_path / "site").mkdir()
    (tmp_path / "site/p.md").write_text("## TODO fill this in\nTBD\n", encoding="utf-8")
    findings = check_placeholders(tmp_path)
    assert findings and findings[0].severity == "error"
