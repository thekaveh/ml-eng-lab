# tests/test_wiki.py
from __future__ import annotations

from pathlib import Path

from scripts.docs.wiki import render_wiki
from scripts.docs.manifest import parse_manifest

MANIFEST_YAML = """
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
        title: System view
        source: docs/architecture.md
        diagrams: [system]
notebooks:
  - task: tabular_classification-iris-mlp-pytorch
    number: "8.1"
    family: tabular
    depth: full
    doc: docs/notebooks/tabular_classification-iris-mlp-pytorch.md
    spec: notebooks/tabular_classification-iris-mlp-pytorch/docs/spec.yaml
diagrams:
  - id: system
    master: docs/diagrams/ml-eng-lab-system.html
"""


def _seed(repo: Path) -> None:
    for rel, text in {
        "docs/index.md": "# 1. Overview\n",
        "docs/architecture.md": "## 2.1 System view\n\n![d](diagrams/img/system.png)\n",
        "docs/notebooks/tabular_classification-iris-mlp-pytorch.md": "## 8.1 Iris MLP\n",
        "notebooks/tabular_classification-iris-mlp-pytorch/docs/spec.yaml": "title: x\n",
        "docs/diagrams/ml-eng-lab-system.html": "<html></html>",
        "docs/diagrams/img/system.png": "PNG",
    }.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def test_render_wiki_writes_home_sidebar_pages_and_images(tmp_path):
    _seed(tmp_path)
    m = parse_manifest(MANIFEST_YAML)
    out = tmp_path / "generated/wiki"
    render_wiki(m, tmp_path, out)
    assert (out / "Home.md").exists()
    assert (out / "_Sidebar.md").exists() and (out / "_Footer.md").exists()
    assert (out / "2-Architecture.md").exists() or (out / "2-1-System-view.md").exists()
    # image rewritten to wiki-local path and PNG copied into wiki
    arch = (out / "2-1-System-view.md").read_text()
    assert "img/system.png" in arch and "diagrams/img/system.png" not in arch
    assert (out / "img/system.png").exists()


def test_render_wiki_strips_forbidden_links(tmp_path):
    _seed(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        "see [site](https://thekaveh.github.io/ml-eng-lab/).", encoding="utf-8"
    )
    m = parse_manifest(MANIFEST_YAML)
    out = tmp_path / "generated/wiki"
    render_wiki(m, tmp_path, out)
    home = (out / "Home.md").read_text()
    assert "https://thekaveh.github.io" not in home and "see site." in home
