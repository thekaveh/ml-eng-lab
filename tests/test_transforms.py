# tests/test_transforms.py
from __future__ import annotations

from scripts.docs.transforms import build_source_map, rewrite_for_surface
from scripts.docs.manifest import parse_manifest

MANIFEST = parse_manifest(
    """
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
    source: docs/architecture.md
    children:
      - id: system
        number: "2.1"
        title: System view
notebooks:
  - task: tabular_classification-iris-mlp-pytorch
    number: "8.1"
    family: tabular
    depth: full
    doc: docs/notebooks/tabular_classification-iris-mlp-pytorch.md
    spec: notebooks/tabular_classification-iris-mlp-pytorch/docs/spec.yaml
diagrams: []
"""
)


def test_build_source_map_site():
    sm = build_source_map(MANIFEST, surface="site")
    assert sm["docs/index.md"] == "index.md"
    assert sm["docs/architecture.md"] == "architecture.md"
    assert sm["docs/notebooks/tabular_classification-iris-mlp-pytorch.md"] == "notebooks/tabular_classification-iris-mlp-pytorch.md"


def test_build_source_map_wiki():
    sm = build_source_map(MANIFEST, surface="wiki")
    assert sm["docs/index.md"] == "Home.md"
    assert sm["docs/architecture.md"] == "2-Architecture.md"
    assert sm["docs/notebooks/tabular_classification-iris-mlp-pytorch.md"] == "8-1-tabular-classification-iris-mlp-pytorch.md"


def test_rewrite_strips_forbidden_links_to_bare_text():
    md = "see [the README](https://github.com/thekaveh/ml-eng-lab/blob/main/README.md) for more."
    out = rewrite_for_surface(md, surface="site", source_map={})
    assert out == "see the README for more."


def test_rewrite_rewrites_md_links_via_source_map():
    sm = {"docs/architecture.md": "architecture.md"}
    md = "see [arch](docs/architecture.md)."
    assert rewrite_for_surface(md, "site", sm) == "see [arch](architecture.md)."


def test_rewrite_drops_ipynb_links_to_bare_text():
    md = "open [the notebook](notebook.ipynb)."
    assert rewrite_for_surface(md, "site", {}) == "open the notebook."
