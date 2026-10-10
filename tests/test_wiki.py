# tests/test_wiki.py
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.docs.wiki import render_wiki
from scripts.docs.manifest import load_manifest, parse_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]

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
    poster = repo / "docs/assets/ml-eng-lab-poster.png"
    poster.parent.mkdir(parents=True, exist_ok=True)
    poster.write_bytes(b"poster")
    badge = repo / "docs/assets/badges/python.svg"
    badge.parent.mkdir(parents=True, exist_ok=True)
    badge.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")


def test_render_wiki_writes_home_sidebar_pages_and_images(tmp_path):
    _seed(tmp_path)
    m = parse_manifest(MANIFEST_YAML)
    out = tmp_path / "generated/wiki"
    render_wiki(m, tmp_path, out, trusted_output_root=tmp_path)
    assert (out / "Home.md").exists()
    assert (out / "_Sidebar.md").exists() and (out / "_Footer.md").exists()
    assert (out / "2-Architecture.md").exists() or (out / "2-1-System-view.md").exists()
    # image rewritten to wiki-local path and PNG copied into wiki
    arch = (out / "2-1-System-view.md").read_text()
    assert "img/system.png" in arch and "diagrams/img/system.png" not in arch
    assert (out / "img/system.png").exists()
    assert (out / "assets/ml-eng-lab-poster.png").read_bytes() == b"poster"
    assert (out / "assets/badges/python.svg").exists()


def test_real_manifest_renders_numbered_security_page_and_sidebar_entry(tmp_path):
    manifest = load_manifest(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT)
    out = tmp_path / "generated/wiki"

    render_wiki(manifest, REPO_ROOT, out, trusted_output_root=tmp_path)
    policy = (out / "13-Security-policy.md").read_text(encoding="utf-8")
    sidebar = (out / "_Sidebar.md").read_text(encoding="utf-8")

    assert "](6-1-Dependency-ledger)" in policy
    assert "[13. Security policy](13-Security-policy)" in sidebar


def test_real_manifest_projects_current_vulnerability_snapshot_to_wiki(tmp_path):
    manifest = load_manifest(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT)
    out = tmp_path / "generated/wiki"
    render_wiki(manifest, REPO_ROOT, out, trusted_output_root=tmp_path)
    ledger = (out / "6-1-Dependency-ledger.md").read_text(encoding="utf-8")
    assert "### 6.1.1.2 Current Issue #63 locked four-surface audit" in ledger
    assert "2026-08-12" in ledger
    assert "archived audit provenance only" in ledger
    assert "../SECURITY.md" not in ledger
    assert "](#6112-current-issue-63-locked-four-surface-audit)" in ledger


def test_real_manifest_projects_advisory_baseline_contract_to_wiki(tmp_path):
    manifest = load_manifest(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT)
    out = tmp_path / "generated/wiki"
    render_wiki(manifest, REPO_ROOT, out, trusted_output_root=tmp_path)

    ledger = " ".join(
        (out / "6-1-Dependency-ledger.md").read_text(encoding="utf-8").split()
    )
    assert "`security/accepted-advisories.json` is the policy artifact" in ledger
    assert "`make audit-advisories` runs all four audit surfaces without suppression" in ledger
    assert "reconciliation evidence, not proof of remediation" in ledger
    assert "JSON policy and current Markdown ledger rows together through review" in ledger
    assert "Refresh snapshot metadata, raw JSON hashes and counts, summary, and current tables" in ledger
    assert "Run focused comparator tests, full `make test`, `make verify`, `make lint`" in ledger
    assert "`make docs-check`, `make docs-wiki`, and live `make audit-advisories`" in ledger
    assert "feature-to-`develop` pull request, then a `develop`-to-`main` pull request" in ledger


def test_real_manifest_projects_verified_nlp_asset_contract_to_wiki(tmp_path):
    manifest = load_manifest(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT)
    out = tmp_path / "generated/wiki"
    render_wiki(manifest, REPO_ROOT, out, trusted_output_root=tmp_path)

    ledger = (out / "6-1-Dependency-ledger.md").read_text(encoding="utf-8")
    sentiment = (
        out / "8-18-sentiment-classification-vader-mlp-pytorch.md"
    ).read_text(encoding="utf-8")
    assert "8adba4294eef3964d820bf655e37e61bdc3a341994356af59b74fb3b4a36ce5c" in ledger
    assert "make verify-nlp-assets" in ledger
    assert "nltk.download" not in sentiment


def test_render_wiki_removes_stale_generated_files(tmp_path):
    _seed(tmp_path)
    out = tmp_path / "generated/wiki"
    out.mkdir(parents=True)
    (out / "obsolete.md").write_text("stale", encoding="utf-8")

    render_wiki(
        parse_manifest(MANIFEST_YAML),
        tmp_path,
        out,
        trusted_output_root=tmp_path,
    )

    assert not (out / "obsolete.md").exists()


def test_render_wiki_removes_stale_empty_directories_recursively(tmp_path):
    _seed(tmp_path)
    out = tmp_path / "generated/wiki"
    stale = out / "old/nested"
    stale.mkdir(parents=True)
    (stale / "obsolete.md").write_text("stale", encoding="utf-8")

    render_wiki(
        parse_manifest(MANIFEST_YAML),
        tmp_path,
        out,
        trusted_output_root=tmp_path,
    )

    assert not (out / "old").exists()


def test_render_wiki_rejects_symlinked_generated_parent(tmp_path):
    _seed(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    generated = tmp_path / "generated"
    generated.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="generated output.*symlink"):
        render_wiki(
            parse_manifest(MANIFEST_YAML),
            tmp_path,
            generated / "wiki",
            trusted_output_root=tmp_path,
        )

    assert list(outside.iterdir()) == []


def test_render_wiki_strips_forbidden_links(tmp_path):
    _seed(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        "see [site](https://thekaveh.github.io/ml-eng-lab/).", encoding="utf-8"
    )
    m = parse_manifest(MANIFEST_YAML)
    out = tmp_path / "generated/wiki"
    render_wiki(m, tmp_path, out, trusted_output_root=tmp_path)
    home = (out / "Home.md").read_text()
    assert "https://thekaveh.github.io" not in home and "see site." in home


def test_sidebar_lists_a_parent_page_and_children(tmp_path):
    _seed(tmp_path)
    (tmp_path / "docs/env-setup.md").write_text("# 4. Environment\n", encoding="utf-8")
    (tmp_path / "docs/jupyterhub-integration.md").write_text("# 4.1 JupyterHub\n", encoding="utf-8")
    manifest = parse_manifest(
        MANIFEST_YAML.replace(
            "  - id: architecture\n    number: \"2\"\n    title: Architecture\n    children:",
            "  - id: architecture\n    number: \"2\"\n    title: Architecture\n    source: docs/env-setup.md\n    children:",
        ).replace("source: docs/architecture.md", "source: docs/jupyterhub-integration.md"),
    )

    render_wiki(
        manifest,
        tmp_path,
        tmp_path / "generated/wiki",
        trusted_output_root=tmp_path,
    )

    sidebar = (tmp_path / "generated/wiki/_Sidebar.md").read_text(encoding="utf-8")
    assert "[2. Architecture](2-Architecture)" in sidebar
    assert "[2.1. System view](2-1-System-view)" in sidebar


def test_sidebar_places_notebooks_before_later_numbered_sections(tmp_path):
    _seed(tmp_path)
    (tmp_path / "docs/findings.md").write_text("# 9 Findings\n", encoding="utf-8")
    manifest = parse_manifest(
        MANIFEST_YAML.replace(
            "notebooks:",
            "  - id: findings\n    number: \"9\"\n    title: Findings\n"
            "    source: docs/findings.md\nnotebooks:",
        )
    )

    render_wiki(
        manifest,
        tmp_path,
        tmp_path / "generated/wiki",
        trusted_output_root=tmp_path,
    )

    sidebar = (tmp_path / "generated/wiki/_Sidebar.md").read_text(encoding="utf-8")
    assert sidebar.index("- 8. Notebooks") < sidebar.index("[9. Findings]")
