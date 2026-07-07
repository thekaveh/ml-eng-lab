# tests/test_build_docs.py
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.docs.build_docs import build, render_mkdocs_yml, render_site
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
    files = {
        "docs/index.md": "# 1. Overview\n",
        "docs/architecture.md": "## 2.1 System view\n\n![d](diagrams/img/system.png)\n",
        "docs/notebooks/tabular_classification-iris-mlp-pytorch.md": "## 8.1 Iris MLP\n",
        "notebooks/tabular_classification-iris-mlp-pytorch/docs/spec.yaml": "title: x\n",
        "docs/diagrams/ml-eng-lab-system.html": "<html><svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'></svg></html>",
        "docs/stylesheets/extra.css": "/* obsidian */\n",
        "docs/manifest.yaml": MANIFEST_YAML,
    }
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def test_render_site_writes_pages_and_assets(tmp_path):
    _seed(tmp_path)
    m = parse_manifest(MANIFEST_YAML)
    out = tmp_path / "generated/site"
    written = render_site(m, tmp_path, out)
    assert (out / "index.md").read_text().startswith("# 1. Overview")
    assert (out / "architecture.md").exists()
    assert (out / "notebooks/tabular_classification-iris-mlp-pytorch.md").exists()
    # PNG image ref rewritten to site SVG asset
    arch = (out / "architecture.md").read_text()
    assert "assets/img/system.svg" in arch and "diagrams/img/system.png" not in arch
    # stylesheet copied
    assert (out / "stylesheets/extra.css").exists()
    assert any(p.name == "index.md" for p in written)


def test_render_mkdocs_yml_has_generated_nav_and_no_repo_url(tmp_path):
    _seed(tmp_path)
    m = parse_manifest(MANIFEST_YAML)
    text = render_mkdocs_yml(m, tmp_path, tmp_path / "generated/site")
    assert "docs_dir: generated/site" in text
    assert "site_dir: site" in text
    assert "repo_url" not in text and "edit_uri" not in text
    parsed = yaml.safe_load(text)
    titles = [item if isinstance(item, str) else list(item)[0] for item in parsed["nav"]]
    assert any("1. Overview" in t for t in titles)
    assert any("2. Architecture" in t for t in titles)


def test_build_check_is_deterministic(tmp_path):
    _seed(tmp_path)
    rc1 = build(tmp_path / "docs/manifest.yaml", tmp_path, site=True, check=True)
    rc2 = build(tmp_path / "docs/manifest.yaml", tmp_path, site=True, check=True)
    assert rc1 == 0 and rc2 == 0


def test_rewrite_images_site_preserves_subdir_prefix():
    from scripts.docs.build_docs import _rewrite_images_site
    # deep-dive in notebooks/ uses ../diagrams/img/... → must keep ../ for the generated site
    assert _rewrite_images_site("![MLP](../diagrams/img/mlp.png)") == "![MLP](../assets/img/mlp.svg)"
    # root doc (no prefix) still resolves at the site root
    assert _rewrite_images_site("![x](diagrams/img/system.png)") == "![x](assets/img/system.svg)"


def test_assert_dirs_equal_catches_content_drift(tmp_path):
    from scripts.docs.build_docs import _assert_dirs_equal

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "x.md").write_text("one")
    (b / "x.md").write_text("two")  # same path, different content
    with pytest.raises(AssertionError, match="content-diff"):
        _assert_dirs_equal(a, b)
    (b / "x.md").write_text("one")  # now byte-identical
    _assert_dirs_equal(a, b)  # no raise
