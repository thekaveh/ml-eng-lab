# tests/test_check_docs.py
from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from scripts.docs.check_docs import (
    check_notebook_infrastructure,
    check_completeness,
    check_numbering,
    check_placeholders,
    check_project_opening,
    check_repo_self_containment,
    check_self_containment,
    manifest_markdown_sources,
)
from scripts.docs.manifest import load_manifest, parse_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASED_010_CHANGELOG_SHA256 = (
    "628842f9c4bd84577f16e0fef091ef7c6b3e5251b39a8b27c038011c5dd78f13"
)


def _released_changelog_section(text: str, version: str) -> str:
    heading = f"## [{version}]"
    start = text.index(heading)
    following = text[start + len(heading) :]
    next_heading = re.search(r"^## \[", following, re.MULTILINE)
    end = len(text) if next_heading is None else start + len(heading) + next_heading.start()
    return text[start:end]


def test_released_010_changelog_history_is_immutable():
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    released = _released_changelog_section(changelog, "0.1.0")

    assert hashlib.sha256(released.encode()).hexdigest() == RELEASED_010_CHANGELOG_SHA256


def test_released_changelog_guard_rejects_history_mutation():
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    released = _released_changelog_section(changelog, "0.1.0")
    mutated = released.replace("Known issue (historical)", "Known issue", 1)
    assert mutated != released

    assert hashlib.sha256(mutated.encode()).hexdigest() != RELEASED_010_CHANGELOG_SHA256

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

ROOT_GOVERNANCE_MANIFEST_YAML = MANIFEST_YAML.replace(
    "notebooks:\n",
    "  - id: support\n"
    "    number: \"13\"\n"
    "    title: Support\n"
    "    source: SUPPORT.md\n"
    "notebooks:\n",
)


def test_documentation_convention_uses_copyable_build_module_commands():
    conventions = (REPO_ROOT / "docs/conventions.md").read_text(encoding="utf-8")

    assert "`python -m scripts.docs.build_docs --site`" in conventions
    assert "`python -m scripts.docs.build_docs --wiki`" in conventions
    assert "scripts/docs/build_docs" not in conventions


BADGE_GROUPS = (
    (
        "Core ML",
        (
            ("Python", "python.svg"),
            ("Jupyter", "jupyter.svg"),
            ("NumPy", "numpy.svg"),
            ("pandas", "pandas.svg"),
            ("PyTorch", "pytorch.svg"),
            ("PyTorch Geometric", "pytorch-geometric.svg"),
            ("scikit-learn", "scikit-learn.svg"),
        ),
    ),
    (
        "NLP and graphs",
        (
            ("spaCy", "spacy.svg"),
            ("NLTK", "nltk.svg"),
            ("NetworkX", "networkx.svg"),
        ),
    ),
    (
        "Runtime",
        (
            ("Atlas", "atlas.svg"),
            ("Docker", "docker.svg"),
            ("VS Code", "vscode.svg"),
            ("GitHub Codespaces", "github-codespaces.svg"),
        ),
    ),
    (
        "Engineering",
        (
            ("NNx", "nnx.svg"),
            ("Papermill", "papermill.svg"),
            ("pytest", "pytest.svg"),
            ("Ruff", "ruff.svg"),
            ("GitHub Actions", "github-actions.svg"),
        ),
    ),
)

PROJECT_SUMMARY = (
    "ml-eng-lab is a portfolio of self-contained machine-learning notebook experiments built for "
    "local editing in VS Code and recommended remote execution through JupyterHub on Atlas's ML "
    "Engineering track. Unlike a loose notebook collection, each task declares its runtime needs in a "
    "checked infrastructure contract, keeping notebook dependencies explicit as the lab expands beyond "
    "JupyterHub. Narrative experiments, reproducible execution tiers, exact dependency pins, validation "
    "gates, and the reusable thekaveh-nnx toolkit evolve together.\n\n"
    "Contributors can use Browser JupyterLab for mounted-workspace tasks or choose a local virtual "
    "environment, Docker, or GitHub Codespaces when Atlas is not the right fit. Host-native Ollama is "
    "the only approved Ollama source whenever a future task needs it; containerized Ollama is "
    "intentionally excluded. This makes the lab both a practical portfolio and a controlled environment "
    "for growing machine-learning systems without hiding operational assumptions inside notebooks. "
    "The collection has 21 active task folders and 29 active notebooks."
)


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


def test_self_containment_rejects_missing_local_raw_html_image(tmp_path):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        '<img src="assets/ml-eng-lab-poster.png">\n'
        '<img src="https://example.com/remote-badge.svg">\n',
        encoding="utf-8",
    )

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "local image target missing" in findings[0].message
    assert "assets/ml-eng-lab-poster.png" in findings[0].message


def test_self_containment_ignores_raw_html_image_examples_in_code(tmp_path):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "```html\n"
        '<img src="examples/fenced.png">\n'
        "```\n\n"
        'Use `<img src="examples/inline.png">` as an example.\n'
        '<img src="assets/rendered.png">\n',
        encoding="utf-8",
    )

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "assets/rendered.png" in findings[0].message


@pytest.mark.parametrize(
    "source",
    ["../outside.png", "/../outside.png", "%2e%2e/outside.png"],
)
def test_self_containment_rejects_raw_html_image_root_escapes(tmp_path, source):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    (tmp_path / "outside.png").write_bytes(b"outside")
    page.write_text(f'<img src="{source}">\n', encoding="utf-8")

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "escapes generated surface" in findings[0].message


def test_self_containment_rejects_raw_html_image_symlink_escape(tmp_path):
    page = tmp_path / "site/index.md"
    assets = tmp_path / "site/assets"
    assets.mkdir(parents=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (assets / "redirect.png").symlink_to(outside)
    page.write_text('<img src="assets/redirect.png">\n', encoding="utf-8")

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "escapes generated surface" in findings[0].message


@pytest.mark.parametrize("source", ["file:assets/existing.png", "ftp://example.com/image.png"])
def test_self_containment_rejects_non_web_raw_html_image_schemes(tmp_path, source):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    (page.parent / "assets").mkdir()
    (page.parent / "assets/existing.png").write_bytes(b"image")
    page.write_text(f'<img src="{source}">\n', encoding="utf-8")

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "unsupported raw HTML image source" in findings[0].message


def test_self_containment_allows_web_data_and_contained_root_images(tmp_path):
    page = tmp_path / "site/index.md"
    asset = tmp_path / "site/assets/existing.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"image")
    page.write_text(
        '<img src="https://example.com/image.png">\n'
        '<img src="//example.com/image.png">\n'
        '<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==">\n'
        '<img src="/assets/existing.png">\n',
        encoding="utf-8",
    )

    assert check_self_containment(tmp_path) == []


def test_self_containment_parses_unquoted_raw_html_image_src(tmp_path):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    page.write_text("<img src=assets/missing.png>\n", encoding="utf-8")

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "assets/missing.png" in findings[0].message


def test_self_containment_does_not_mistake_data_src_for_src(tmp_path):
    page = tmp_path / "site/index.md"
    asset = tmp_path / "site/assets/existing.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"image")
    page.write_text(
        '<img data-src="assets/existing.png">\n'
        '<img data-src="assets/existing.png" src=assets/missing.png>\n',
        encoding="utf-8",
    )

    messages = [finding.message for finding in check_self_containment(tmp_path)]

    assert len(messages) == 2
    assert any("missing src" in message for message in messages)
    assert any("assets/missing.png" in message for message in messages)


def test_self_containment_ignores_raw_html_images_in_comments_and_indented_code(tmp_path):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        '<!-- <img src="examples/comment.png"> -->\n'
        '    <img src="examples/spaces.png">\n'
        '\t<img src="examples/tab.png">\n'
        '<img src="assets/rendered.png">\n',
        encoding="utf-8",
    )

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "assets/rendered.png" in findings[0].message


def test_self_containment_flags_indented_raw_html_image_inside_html_block(tmp_path):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "<div>\n"
        "    <img src=assets/missing.png>\n"
        "</div>\n",
        encoding="utf-8",
    )

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "assets/missing.png" in findings[0].message


def test_self_containment_ignores_indented_raw_html_image_after_void_html_block(tmp_path):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "<hr>\n\n"
        "    <img src=assets/missing.png>\n",
        encoding="utf-8",
    )

    assert check_self_containment(tmp_path) == []


def test_self_containment_flags_indented_raw_html_image_inside_details(tmp_path):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "<details>\n"
        "    <img src=assets/missing.png>\n"
        "</details>\n",
        encoding="utf-8",
    )

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "assets/missing.png" in findings[0].message


@pytest.mark.parametrize("wrapper", ["li", "tbody", "center", "custom-element"])
def test_self_containment_flags_indented_raw_html_image_inside_paired_wrapper(
    tmp_path,
    wrapper,
):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        f"<{wrapper}>\n"
        "    <img src=assets/missing.png>\n"
        f"</{wrapper}>\n",
        encoding="utf-8",
    )

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "assets/missing.png" in findings[0].message


def test_self_containment_clears_inline_html_context_at_blank_line(tmp_path):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "Prose with <custom-element>inline HTML\n\n"
        "    <img src=assets/missing.png>\n",
        encoding="utf-8",
    )

    assert check_self_containment(tmp_path) == []


def test_repo_self_containment_rejects_site_and_wiki_links(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        "https://thekaveh.github.io/ml-eng-lab/\n", encoding="utf-8"
    )
    (tmp_path / "docs/page.md").write_text(
        "[wiki](https://github.com/thekaveh/ml-eng-lab/wiki/Page)\n",
        encoding="utf-8",
    )

    findings = check_repo_self_containment(tmp_path)

    assert len(findings) == 2
    assert all(finding.severity == "error" for finding in findings)


def test_repo_self_containment_allows_relative_links(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("[docs](docs/page.md)\n", encoding="utf-8")
    (tmp_path / "docs/page.md").write_text("[next](next.md)\n", encoding="utf-8")

    assert check_repo_self_containment(tmp_path) == []


def test_repo_self_containment_scans_manifest_declared_root_markdown(tmp_path):
    _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/manifest.yaml").write_text(
        ROOT_GOVERNANCE_MANIFEST_YAML,
        encoding="utf-8",
    )
    (tmp_path / "SUPPORT.md").write_text(
        "[wiki](https://github.com/thekaveh/ml-eng-lab/wiki/Support)\n",
        encoding="utf-8",
    )

    findings = check_repo_self_containment(tmp_path)

    assert len(findings) == 1
    assert "SUPPORT.md" in findings[0].message


def test_completeness_flags_missing_spec(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/index.md").write_text("x", encoding="utf-8")
    m = parse_manifest(MANIFEST_YAML)
    findings = check_completeness(m, tmp_path)
    assert any("spec" in f.message.lower() or "doc" in f.message.lower() for f in findings)


def test_completeness_flags_unmanifested_markdown(tmp_path):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/extra.md").write_text("# 9 Extra\n", encoding="utf-8")

    findings = check_completeness(manifest, tmp_path)

    assert any("docs/extra.md" in finding.message and "not declared" in finding.message for finding in findings)


def test_completeness_accepts_existing_manifest_declared_root_markdown(tmp_path):
    _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "SUPPORT.md").write_text("# 13 Support\n", encoding="utf-8")
    manifest = parse_manifest(ROOT_GOVERNANCE_MANIFEST_YAML)

    assert check_completeness(manifest, tmp_path) == []


def test_load_manifest_rejects_missing_declared_root_markdown(tmp_path):
    _write_valid_notebook_infrastructure_fixture(tmp_path)
    manifest_path = tmp_path / "docs/manifest.yaml"
    manifest_path.write_text(ROOT_GOVERNANCE_MANIFEST_YAML, encoding="utf-8")

    with pytest.raises(ValueError, match="section source 'SUPPORT.md' does not exist"):
        load_manifest(manifest_path, tmp_path)


def test_manifest_markdown_sources_contains_sections_and_notebooks():
    manifest = parse_manifest(MANIFEST_YAML)

    assert manifest_markdown_sources(manifest) == {
        "docs/index.md",
        "docs/notebooks/t.md",
    }


def test_real_manifest_declares_every_canonical_markdown_file():
    manifest = load_manifest(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT)
    declared = manifest_markdown_sources(manifest)
    actual_docs = {
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "docs").rglob("*.md")
    }
    declared_root_sources = {
        source
        for source in declared
        if Path(source).parent == Path(".") and source.endswith(".md")
    }

    assert declared_root_sources == {"SECURITY.md"}
    assert declared == actual_docs | declared_root_sources


def test_real_manifest_sections_are_source_leaves_or_children_groups():
    manifest = load_manifest(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT)

    assert all(bool(section.source) ^ bool(section.children) for section in manifest.sections)


def test_real_manifest_declares_issue_58_design_and_implementation_records():
    manifest = load_manifest(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT)
    design_records = next(
        section for section in manifest.sections if section.id == "design-records"
    )
    issue_58_start = next(
        index
        for index, child in enumerate(design_records.children)
        if child.id == "issue-58-nnx-wheel-contract-design"
    )

    assert [
        (child.number, child.source)
        for child in design_records.children[issue_58_start : issue_58_start + 2]
    ] == [
        (
            "12.13",
            "docs/superpowers/specs/2026-08-12-issue-58-nnx-wheel-contract-design.md",
        ),
        (
            "12.14",
            "docs/superpowers/plans/2026-08-12-issue-58-nnx-wheel-contract-implementation-plan.md",
        ),
    ]


def test_real_manifest_declares_issue_59_design_and_implementation_records():
    manifest = load_manifest(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT)
    records = next(section for section in manifest.sections if section.id == "design-records")
    start = next(
        index
        for index, child in enumerate(records.children)
        if child.id == "issue-59-vulnerability-ledger-design"
    )
    assert [
        (child.number, child.source) for child in records.children[start : start + 2]
    ] == [
        (
            "12.15",
            "docs/superpowers/specs/2026-08-12-issue-59-vulnerability-ledger-design.md",
        ),
        (
            "12.16",
            "docs/superpowers/plans/2026-08-12-issue-59-vulnerability-ledger-implementation-plan.md",
        ),
    ]


def test_real_manifest_declares_issue_60_design_and_implementation_records_consecutively():
    manifest = load_manifest(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT)
    records = next(section for section in manifest.sections if section.id == "design-records")
    start = next(
        index
        for index, child in enumerate(records.children)
        if child.id == "issue-60-advisory-baseline-design"
    )
    assert [
        (child.number, child.source) for child in records.children[start : start + 2]
    ] == [
        (
            "12.17",
            "docs/superpowers/specs/2026-08-13-issue-60-advisory-baseline-design.md",
        ),
        (
            "12.18",
            "docs/superpowers/plans/2026-08-13-issue-60-advisory-baseline-implementation-plan.md",
        ),
    ]


def test_real_user_docs_publish_advisory_baseline_contract():
    docs = {
        path: (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "CHANGELOG.md",
            "docs/dependency-contracts.md",
            "docs/conventions.md",
            "docs/architecture.md",
            "docs/maintenance/overnight-2026-07-04.md",
        )
    }
    ledger = " ".join(docs["docs/dependency-contracts.md"].split())

    assert "security/                                  (accepted-advisory policy)" in docs["docs/conventions.md"]
    assert "`make audit-advisories`" in docs["CONTRIBUTING.md"]
    assert "`dependency-audit`" in docs["CONTRIBUTING.md"]
    assert "does not claim an automated vulnerability-baseline gate" not in docs["SECURITY.md"]
    assert "`security/accepted-advisories.json` is the policy artifact" in ledger
    assert "`make audit-advisories` runs all four audit surfaces without suppression" in ledger
    assert "The audit projections are generated from the committed locks" in ledger
    assert "explicit non-PyPI records for the PyG extension wheels" in ledger
    assert "The commands below are historical capture evidence" in ledger
    assert "New primary advisory IDs and accepted-version drift fail the gate." in ledger
    assert "reconciliation evidence, not proof of remediation" in ledger
    assert "JSON policy and current Markdown ledger rows together through review" in ledger
    assert "Refresh snapshot metadata, raw JSON hashes and counts, summary, and current tables" in ledger
    assert "Run focused comparator tests, full `make test`, `make verify`, `make lint`" in ledger
    assert "`make docs-check`, `make docs-wiki`, and live `make audit-advisories`" in ledger
    assert "feature-to-`develop` pull request, then a `develop`-to-`main` pull request" in ledger
    assert "Issue #63 adds complete dependency locks" in ledger
    assert "does not initialize Atlas or start a service" in ledger
    assert "`dependency-audit`" in docs["docs/conventions.md"]
    assert "exact lock-derived, no-resolve projections" in " ".join(
        docs["docs/conventions.md"].split()
    )
    assert "selected platform lock rather than an audit projection" in ledger
    assert "pyg-extension-audit-requirements.txt" in docs["docs/conventions.md"]
    assert "isolated `dependency-audit` signal" in docs["docs/architecture.md"]
    assert "| OM-047 |" in docs["docs/maintenance/overnight-2026-07-04.md"]
    assert "| Fixed | Issue #60 added the reviewed JSON policy" in docs[
        "docs/maintenance/overnight-2026-07-04.md"
    ]
    assert "Machine-readable advisory baseline gate" in docs["CHANGELOG.md"]
    assert "Automated vulnerability-baseline enforcement remains deferred to Issue\n  #60." not in docs["CHANGELOG.md"]


def test_current_runtime_surface_comments_describe_the_selected_torch_contract():
    expected = {
        "Makefile": (
            "# Issue #62 canonical CPU stack: Torch 2.11, binary pyg-lib/scatter/sparse, "
            "NNx 0.2.0 last."
        ),
        ".github/workflows/ci.yml": (
            "# Issue #63: the root lock already includes docs; no duplicate resolver pass."
        ),
        "Dockerfile": "# Issue #63 locked CPU image: no service startup and no source-built PyG extension.",
        ".devcontainer/devcontainer.json": (
            "// Issue #63 setup delegates to the hash-locked make codespace-setup; it starts no service."
        ),
    }

    for path, comment in expected.items():
        source = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert comment in source, path


def test_real_user_docs_publish_current_vulnerability_snapshot():
    ledger = (REPO_ROOT / "docs/dependency-contracts.md").read_text(encoding="utf-8")
    security = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "### 6.1.1.1 Reproducible four-surface audit" in ledger
    assert "### 6.1.1.2 Current Issue #63 locked four-surface audit" in ledger
    assert "### 6.1.1.3 Alias-aware historical reconciliation" in ledger
    assert "2026-08-12" in ledger
    assert "--disable-pip -r docs-requirements.txt" in ledger
    assert (
        "Any exit other than 0/1, missing output, or malformed JSON invalidates the "
        "observation."
        in ledger
    )
    assert (
        "Absent from the 2026-08-12 snapshot; archived audit provenance only" in ledger
    )
    assert "`security/accepted-advisories.json` is the reviewed policy artifact" in security
    assert "Four-surface vulnerability ledger refresh" in changelog


def test_nnx_wheel_contract_is_consistent_across_canonical_user_docs():
    docs = {
        path: (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "CONTRIBUTING.md",
            "CHANGELOG.md",
            "docs/conventions.md",
            "docs/dependency-contracts.md",
            "docs/nnx-library.md",
        )
    }
    combined = "\n".join(docs.values())

    assert "[canonical dependency contract](docs/dependency-contracts.md)" in docs["README.md"]
    assert "`NNX_ALLOW_EDITABLE=1 make test-nnx-surface`" in docs["README.md"]

    assert "`make verify-nnx-install`" in docs["CONTRIBUTING.md"]
    assert "development-surface evidence, never released-wheel evidence" in docs["CONTRIBUTING.md"]

    conventions = docs["docs/conventions.md"]
    assert "NNx wheel is binary-only within that lock" in conventions
    assert "`pytest-repository`" in conventions
    assert "`pytest-nnx-surface`" in conventions
    assert conventions.count("`make verify-nnx-install`") >= 2
    assert "required merge-blocking" in conventions
    assert "focused diagnostic" in " ".join(conventions.split())

    dependency_contract = docs["docs/dependency-contracts.md"]
    for evidence in (
        "thekaveh-nnx[lm]==0.2.0",
        "rejects any `direct_url.json`",
        "`WHEEL`, `RECORD`, and `nnx/__init__.py`",
        "distribution-owned import origin",
        "redacted",
        "NNX_ALLOW_EDITABLE=1 make test-nnx-surface",
        "Issue #63",
    ):
        assert evidence in dependency_contract
    assert "The lock records the wheel hash" in dependency_contract
    assert "dist._path" not in combined
    assert '{"editable": true}' not in combined

    nnx_library = docs["docs/nnx-library.md"]
    assert "`NNX_ALLOW_EDITABLE=1 make test-nnx-surface`" in nnx_library
    assert "`make verify-nnx-install`" in nnx_library
    assert "released-wheel evidence" in nnx_library

    changelog = docs["CHANGELOG.md"]
    assert "canonical nnx wheel evidence" in changelog.lower()
    for changed_fact in (
        "`pytest-repository`",
        "`pytest-nnx-surface`",
        "`--only-binary=thekaveh-nnx`",
        "validated editable-development mode",
    ):
        assert changed_fact in changelog


def test_real_user_facing_docs_match_the_atlas_runtime_contract():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    runtime_docs = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "docs/env-setup.md",
            "docs/jupyterhub-integration.md",
            "docs/vscode-remote-access.md",
            "docs/atlas-pin-bump-runbook.md",
        )
    )
    infrastructure = (REPO_ROOT / "docs/notebook-infrastructure.md").read_text(
        encoding="utf-8"
    )

    assert "MkDocs" not in readme
    assert "mkdocs.yml" not in readme
    assert "short-lived" not in readme + runtime_docs
    assert "token-bearing" in readme + runtime_docs
    assert "track defaults are not notebook authorization" in infrastructure
    assert "Additional Atlas services stay inactive" not in infrastructure


def test_issue67_docs_define_machine_readable_environment_admission():
    infrastructure = (REPO_ROOT / "docs/notebook-infrastructure.md").read_text(
        encoding="utf-8"
    )
    runbook = (REPO_ROOT / "docs/atlas-pin-bump-runbook.md").read_text(
        encoding="utf-8"
    )
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    normalized_infrastructure = " ".join(infrastructure.split())
    normalized_runbook = " ".join(runbook.split())
    normalized_contributing = " ".join(contributing.split())
    normalized_changelog = " ".join(changelog.split())

    for phrase in (
        "`required_env`",
        "`name` and `service`",
        "does not authorize notebook use",
        "does not prove that the service is enabled or healthy",
        "must not contain environment values",
    ):
        assert phrase in normalized_infrastructure
    for phrase in (
        "`{name, service}`",
        "targeted Atlas JupyterHub smoke",
        "never its value",
    ):
        assert phrase in normalized_runbook
    for phrase in (
        "`required_env`",
        "central source",
        "targeted JupyterHub smoke",
    ):
        assert phrase in normalized_contributing
    assert "Issue #67" in changelog.split("## [0.1.0]", 1)[0]
    assert "does not admit a new Atlas service" in normalized_changelog


def _write_project_opening(
    repo_root: Path,
    *,
    readme_tagline: str = "Local notebooks. Remote Atlas execution. Explicit infrastructure contracts.",
    landing_tagline: str = "Local notebooks. Remote Atlas execution. Explicit infrastructure contracts.",
    readme_summary: str = PROJECT_SUMMARY,
    landing_summary: str = PROJECT_SUMMARY,
) -> None:
    badge_assets = repo_root / "docs/assets/badges"
    badge_assets.mkdir(parents=True)
    for _, badges in BADGE_GROUPS:
        for _, filename in badges:
            (badge_assets / filename).write_text("badge", encoding="utf-8")
    (repo_root / "docs/assets/ml-eng-lab-poster.png").write_text(
        "poster", encoding="utf-8"
    )

    def badge_html(prefix: str) -> str:
        rows = []
        for label, badges in BADGE_GROUPS:
            images = " ".join(
                f'<img alt="{alt}" src="{prefix}badges/{filename}">'
                for alt, filename in badges
            )
            rows.append(
                '<p align="center">\n'
                f"  <sub><strong>{label}</strong></sub><br>\n"
                f"  {images}\n"
                "</p>"
            )
        return "\n\n".join(rows)

    def opener(
        *, asset_prefix: str, title: str, tagline: str, summary: str
    ) -> str:
        return (
            '<p align="center">\n'
            f'  <img src="{asset_prefix}ml-eng-lab-poster.png" '
            'alt="ML Eng Lab — notebooks, systems, and reproducibility" width="100%">\n'
            "</p>\n\n"
            f'<h1 align="center">{title}</h1>\n\n'
            f'<p align="center"><strong>{tagline}</strong></p>\n\n'
            f"{badge_html(asset_prefix)}\n\n"
            "<!-- project-summary:start -->\n"
            f"{summary}\n"
            "<!-- project-summary:end -->\n"
            "\n## 1.1 Repository map\n"
        )

    (repo_root / "README.md").write_text(
        opener(
            asset_prefix="docs/assets/",
            title="ML ENG LAB",
            tagline=readme_tagline,
            summary=readme_summary,
        ),
        encoding="utf-8",
    )
    (repo_root / "docs/index.md").write_text(
        opener(
            asset_prefix="assets/",
            title="1 · ML ENG LAB",
            tagline=landing_tagline,
            summary=landing_summary,
        ),
        encoding="utf-8",
    )


def test_project_opening_rejects_tagline_and_inventory_drift(tmp_path):
    _write_project_opening(
        tmp_path,
        landing_tagline="A different tagline.",
        landing_summary=PROJECT_SUMMARY.replace("21 active", "20 active"),
    )

    messages = [finding.message for finding in check_project_opening(tmp_path)]

    assert any("tagline" in message for message in messages)
    assert any("active task inventory" in message for message in messages)


def test_project_opening_rejects_missing_poster_and_inventory(tmp_path):
    short_summary = f"ml-eng-lab {' '.join(['short'] * 20)}"
    _write_project_opening(
        tmp_path,
        readme_summary=short_summary,
        landing_summary=short_summary,
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            '  <img src="docs/assets/ml-eng-lab-poster.png" '
            'alt="ML Eng Lab — notebooks, systems, and reproducibility" width="100%">\n',
            "",
        ),
        encoding="utf-8",
    )

    messages = [finding.message for finding in check_project_opening(tmp_path)]

    assert any("poster" in message for message in messages)
    assert any("inventory" in message for message in messages)


def test_project_opening_rejects_missing_project_title(tmp_path):
    _write_project_opening(tmp_path)
    landing = tmp_path / "docs/index.md"
    landing.write_text(
        landing.read_text(encoding="utf-8").replace(
            '<h1 align="center">1 · ML ENG LAB</h1>',
            '<h1 align="center">1 · Overview</h1>',
        ),
        encoding="utf-8",
    )

    assert any("title" in finding.message for finding in check_project_opening(tmp_path))


def test_project_opening_rejects_content_inserted_before_poster(tmp_path):
    _write_project_opening(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            '<p align="center">\n  <img src="docs/assets/ml-eng-lab-poster.png"',
            'Unexpected prose.\n\n<p align="center">\n  <img src="docs/assets/ml-eng-lab-poster.png"',
        ),
        encoding="utf-8",
    )

    assert any("order" in finding.message for finding in check_project_opening(tmp_path))


def test_project_opening_rejects_runtime_diagram_as_poster(tmp_path):
    _write_project_opening(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            "docs/assets/ml-eng-lab-poster.png",
            "docs/diagrams/img/runtime-flow.png",
            1,
        ),
        encoding="utf-8",
    )

    assert any(
        "runtime-flow" in finding.message
        for finding in check_project_opening(tmp_path)
    )


def test_project_opening_rejects_left_aligned_markdown_title(tmp_path):
    _write_project_opening(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            '<h1 align="center">ML ENG LAB</h1>', "# ML ENG LAB", 1
        ),
        encoding="utf-8",
    )

    assert any(
        "centered HTML title" in finding.message
        for finding in check_project_opening(tmp_path)
    )


def test_project_opening_rejects_missing_badge_and_plain_text_stack(tmp_path):
    _write_project_opening(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            '<img alt="Python" src="docs/assets/badges/python.svg">',
            "Python",
            1,
        ),
        encoding="utf-8",
    )

    assert any("badge" in finding.message for finding in check_project_opening(tmp_path))


def test_project_opening_rejects_missing_local_asset(tmp_path):
    _write_project_opening(tmp_path)
    (tmp_path / "docs/assets/badges/python.svg").unlink()

    assert any(
        "asset missing" in finding.message
        for finding in check_project_opening(tmp_path)
    )


def test_project_opening_rejects_missing_poster_asset(tmp_path):
    _write_project_opening(tmp_path)
    (tmp_path / "docs/assets/ml-eng-lab-poster.png").unlink()

    assert any(
        "poster asset missing" in finding.message
        for finding in check_project_opening(tmp_path)
    )


def test_project_opening_allows_single_paragraph_summary(tmp_path):
    _write_project_opening(tmp_path)
    landing = tmp_path / "docs/index.md"
    landing.write_text(
        landing.read_text(encoding="utf-8").replace("together.\n\nContributors", "together. Contributors", 1),
        encoding="utf-8",
    )

    assert check_project_opening(tmp_path) == []


def test_project_opening_rejects_runtime_flow_in_summary_tail(tmp_path):
    _write_project_opening(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            "<!-- project-summary:end -->\n\n## 1.1 Repository map",
            "<!-- project-summary:end -->\n\n"
            "![Runtime flow](docs/diagrams/img/runtime-flow.png)\n\n"
            "## 1.1 Repository map",
        ),
        encoding="utf-8",
    )

    assert any(
        "runtime-flow" in finding.message
        for finding in check_project_opening(tmp_path)
    )


def test_project_opening_rejects_nonwhitespace_summary_tail(tmp_path):
    _write_project_opening(tmp_path)
    landing = tmp_path / "docs/index.md"
    landing.write_text(
        landing.read_text(encoding="utf-8").replace(
            "<!-- project-summary:end -->\n\n## 1.1 Repository map",
            "<!-- project-summary:end -->\n\nOne-surface tail drift.\n\n"
            "## 1.1 Repository map",
        ),
        encoding="utf-8",
    )

    assert any(
        "only whitespace" in finding.message
        for finding in check_project_opening(tmp_path)
    )


def test_project_opening_allows_summary_line_reflow(tmp_path):
    _write_project_opening(
        tmp_path,
        landing_summary=PROJECT_SUMMARY.replace(
            "recommended remote execution through JupyterHub",
            "recommended remote execution\nthrough JupyterHub",
        ),
    )

    assert check_project_opening(tmp_path) == []


def test_project_opening_allows_moved_summary_paragraph_boundary(tmp_path):
    _write_project_opening(
        tmp_path,
        landing_summary=PROJECT_SUMMARY.replace(
            "together.\n\nContributors can use Browser JupyterLab",
            "together. Contributors can use Browser\n\nJupyterLab",
        ),
    )

    assert check_project_opening(tmp_path) == []


def test_project_opening_ignores_summary_tail_whitespace_drift(tmp_path):
    _write_project_opening(tmp_path)
    landing = tmp_path / "docs/index.md"
    landing.write_text(
        landing.read_text(encoding="utf-8").replace(
            "<!-- project-summary:end -->\n\n## 1.1 Repository map",
            "<!-- project-summary:end -->\n \n\n   \n## 1.1 Repository map",
        ),
        encoding="utf-8",
    )

    assert check_project_opening(tmp_path) == []


def test_real_project_opening_is_canonical():
    assert check_project_opening(REPO_ROOT) == []


def test_atlas_migration_records_are_marked_complete():
    design = (
        REPO_ROOT
        / "docs/superpowers/specs/2026-07-30-atlas-infrastructure-migration-design.md"
    ).read_text(encoding="utf-8")
    plan = (
        REPO_ROOT
        / "docs/superpowers/plans/2026-07-30-atlas-infrastructure-migration-implementation-plan.md"
    ).read_text(encoding="utf-8")

    assert "**Status:** Implemented" in design
    assert "**Status:** Completed" in plan
    assert "- [ ]" not in plan


def test_numbering_requires_manifest_h1_and_hierarchical_children(tmp_path):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        "# Overview\n\n## 1. Wrong depth\n\n### Child without a number\n",
        encoding="utf-8",
    )

    messages = [finding.message for finding in check_numbering(manifest, tmp_path)]

    assert any("H1" in message for message in messages)
    assert any("H2" in message for message in messages)
    assert any("H3" in message for message in messages)


def test_numbering_accepts_hierarchical_headings_and_ignores_fences(tmp_path):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        "# 1 Overview\n\n## 1.1 First\n\n### 1.1.1 Child\n\n"
        "```bash\n# shell comment\n## not a heading\n```\n",
        encoding="utf-8",
    )

    assert check_numbering(manifest, tmp_path) == []


def test_numbering_accepts_centered_html_h1(tmp_path):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        '<h1 align="center">1 · ML ENG LAB</h1>\n\n## 1.1 Repository map\n',
        encoding="utf-8",
    )

    assert check_numbering(manifest, tmp_path) == []


def test_numbering_ignores_markdown_and_html_headings_in_html_comments(tmp_path):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        "# 1 Overview\n\n"
        "<!-- ## 1.1 Single-line Markdown comment -->\n"
        '<!-- <h1 align="center">1 Commented title</h1> -->\n'
        "<!--\n"
        "## 1.1 Multiline Markdown comment\n"
        "-->\n"
        "<!--\n"
        '<h1 align="center">1 Multiline commented title</h1>\n'
        "-->\n"
        "## 1.1 Repository map\n",
        encoding="utf-8",
    )

    assert check_numbering(manifest, tmp_path) == []


@pytest.mark.parametrize(
    ("opener", "shorter_closer", "trailing_closer", "closer"),
    [
        ("````", "```", "```` trailing", "````"),
        ("~~~~", "~~~", "~~~~ trailing", "~~~~"),
    ],
)
def test_numbering_keeps_fences_open_for_shorter_or_trailing_closers(
    tmp_path, opener, shorter_closer, trailing_closer, closer
):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        "# 1 Overview\n\n"
        f"{opener}text\n"
        "## 1.1 Still in the fence\n"
        f"{shorter_closer}\n"
        "## 1.1 Still in the fence\n"
        f"{trailing_closer}\n"
        "## 1.1 Still in the fence\n"
        f"{closer}\n\n"
        "## 1.1 Repository map\n",
        encoding="utf-8",
    )

    assert check_numbering(manifest, tmp_path) == []


@pytest.mark.parametrize("marker", ["```", "~~~"])
def test_numbering_keeps_fence_open_for_overindented_closer(tmp_path, marker):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        "# 1 Overview\n\n"
        f"{marker}\n"
        f"    {marker}\n"
        '<h1 align="center">Second title still inside fence</h1>\n'
        f"{marker}\n\n"
        "## 1.1 Repository map\n",
        encoding="utf-8",
    )

    assert check_numbering(manifest, tmp_path) == []


def test_placeholders_flag_tbd(tmp_path):
    (tmp_path / "site").mkdir()
    (tmp_path / "site/p.md").write_text("## TODO fill this in\nTBD\n", encoding="utf-8")
    findings = check_placeholders(tmp_path)
    assert findings and findings[0].severity == "error"


def _write_valid_notebook_infrastructure_fixture(tmp_path):
    (tmp_path / "docs/notebooks").mkdir(parents=True)
    (tmp_path / "docs/index.md").write_text("# 1. Overview\n", encoding="utf-8")
    (tmp_path / "docs/notebooks/t.md").write_text("# 8.1 Task\n", encoding="utf-8")
    (tmp_path / "notebooks/t/docs").mkdir(parents=True)
    (tmp_path / "notebooks/t/docs/spec.yaml").write_text(
        "title: Task\ntier: A\natlas:\n  executor: jupyterhub\n  default_mode: vscode-remote\n"
        "  required_services: [jupyterhub]\n  required_env: []\n  workspace_access: remote\n"
        "  artifact_policy: atlas-jupyter-volume\n  constraints: []\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/verify_repo_config.yaml").write_text(
        "active_task_dirs: [t-iris-mlp-pytorch]\n", encoding="utf-8"
    )
    return parse_manifest(MANIFEST_YAML)


def test_notebook_infrastructure_flags_only_a_drifted_canonical_table(tmp_path):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/notebook-infrastructure.md").write_text(
        "<!-- atlas-task-contracts:start -->\n| stale |\n<!-- atlas-task-contracts:end -->\n",
        encoding="utf-8",
    )

    findings = check_notebook_infrastructure(manifest, tmp_path)

    assert len(findings) == 1
    assert findings[0].severity == "error" and "drifted" in findings[0].message


def test_contract_failure_stops_before_generated_surface_build(tmp_path, monkeypatch):
    from scripts.docs import build_docs, check_docs

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/manifest.yaml").write_text(MANIFEST_YAML, encoding="utf-8")
    (tmp_path / "docs/index.md").write_text("# 1. Overview\n", encoding="utf-8")
    (tmp_path / "docs/notebooks").mkdir()
    (tmp_path / "docs/notebooks/t.md").write_text("# 8.1 Task\n", encoding="utf-8")
    (tmp_path / "notebooks/t/docs").mkdir(parents=True)
    (tmp_path / "notebooks/t/docs/spec.yaml").write_text("title: Task\n", encoding="utf-8")
    monkeypatch.setattr(check_docs, "check_notebook_infrastructure", lambda *_: [check_docs.Finding("error", "bad contract")])
    monkeypatch.setattr(build_docs, "build", lambda *args, **kwargs: pytest.fail("generated build ran"))

    assert check_docs.check(tmp_path, tmp_path / "generated") == 1


def _same_level_section(text: str, heading: str) -> str:
    marker = f"## {heading}\n"
    assert text.count(marker) == 1
    body = text.split(marker, 1)[1]
    return body.split("\n## ", 1)[0].strip()


def _between(text: str, start: str, end: str) -> str:
    assert text.count(start) == 1
    assert text.count(end) == 1
    return text.split(start, 1)[1].split(end, 1)[0]


_ISSUE62_REQUIRED_CURRENT_FACTS = (
    "torch==2.11.0",
    "torchvision==0.26.0",
    "torchaudio==2.11.0",
    "pytorch-lightning==2.6.1",
    "torchmetrics==1.9.0",
    "torchao==0.18.0",
    "torch-geometric==2.8.0.post1",
    "pyg-lib==0.8.0",
    "torch-scatter==2.1.2",
    "torch-sparse==0.6.18",
    "thekaveh-nnx[lm]==0.2.0",
    "make install-torch-stack",
    "python -m pip check",
    "make verify-torch-stack",
    "make verify-nnx-install",
    "complete hash-required platform lock",
    "Darwin arm64",
    "Linux x86_64",
    "Issue #65",
    "Issue #66",
    "no containerized Ollama",
    "Torch 2.11.0 with outer torch-geometric 2.8.0.post1",
    "Torch 2.11.0 with outer torch-sparse 0.6.18",
    "DeprecationWarning",
    "`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.",
    "torch/jit/_script.py",
    'Torch 2.11.0 + torchao 0.18.0 + thekaveh-nnx 0.2.0 + qat_config="8da4w"',
    "exactly one identity-UserWarning",
    "Deprecation: TorchAODType is deprecated, please use the torch.intN dtype instead "
    "(e.g. TorchAODType.INT4 -> torch.int4)",
    "torchao/quantization/quant_primitives.py",
    "fresh-interpreter",
    "no global filter is allowed",
    "fresh environment or rebuilt image",
    "Feed disappearance is reconciliation evidence, never proof of remediation.",
    "Issue #62 completed the runtime matrix",
    "verified immutable locks",
)

_ISSUE62_FORBIDDEN_CURRENT_FACTS = (
    "torch==2.4.1",
    "torchao>=",
    "torch-cluster==",
    "torch-spline-conv==",
    "pip install --upgrade pip wheel",
    "--no-build-isolation",
    "five canaries",
    "twelve components",
    "graph backends are unavailable on Darwin",
    "Issue #62 alone qualifies the complete quantization notebook",
    "Issue #62 upgrades Atlas",
    "complete final acceptance is proven",
)

_ISSUE62_PENDING_PLATFORM_SENTENCE = (
    "Linux is CPU-only; Darwin arm64, native Linux arm64 Docker, and Linux x86_64 PR-gate "
    "evidence remain pending and are required in Task 7."
)

_ISSUE62_PREMATURE_PLATFORM_VERDICTS = (
    "accepted",
    "qualified",
    "completed",
    "succeeded",
)


def _issue62_current_documents() -> dict[str, str]:
    def read(path: str) -> str:
        return (REPO_ROOT / path).read_text(encoding="utf-8")

    readme = read("README.md")
    contributing = read("CONTRIBUTING.md")
    security = read("SECURITY.md")
    changelog = read("CHANGELOG.md")
    ledger = read("docs/dependency-contracts.md")
    return {
        "README.md": _between(readme, "### 3.3. Local venv\n", "## 4. Tasks\n"),
        "CONTRIBUTING.md": _same_level_section(contributing, "6. Verification"),
        "SECURITY.md": _same_level_section(security, "13.6 Dependency advisories"),
        "CHANGELOG.md": changelog[: changelog.index("## [0.1.0]")],
        "docs/dependency-contracts.md": "\n".join(
            (
                _between(
                    ledger,
                    "### 6.1.1.2 Current Issue #63 locked four-surface audit\n",
                    "### 6.1.1.3 Alias-aware historical reconciliation\n",
                ),
                _same_level_section(ledger, "6.1.2 Torch Stack Pin"),
                _same_level_section(ledger, "6.1.3 Automated Quantization Notebook"),
                _same_level_section(ledger, "6.1.9 Atlas Versus Local/CI Dependency Boundaries"),
                _same_level_section(ledger, "6.1.11 Canonical Bootstrap Tooling"),
            )
        ),
        **{
            path: read(path)
            for path in (
                "docs/env-setup.md",
                "docs/architecture.md",
                "docs/FINDINGS-ATLAS.md",
                "docs/notebook-infrastructure.md",
                "docs/notebooks/node_classification-reddit-gnn-pyg.md",
                "docs/notebooks/pruning-mnist-ffnn-pytorch.md",
                "docs/notebooks/quantization-mnist-ffnn-pytorch.md",
                "notebooks/node_classification-reddit-gnn-pyg/README.md",
                "notebooks/node_classification-reddit-gnn-pyg/docs/spec.yaml",
                "notebooks/quantization-mnist-ffnn-pytorch/README.md",
                "notebooks/quantization-mnist-ffnn-pytorch/docs/spec.yaml",
                "Makefile",
                ".github/workflows/ci.yml",
                "Dockerfile",
                ".devcontainer/devcontainer.json",
                "docs/assets/badges/pytorch.svg",
            )
        },
    }


def _assert_issue62_current_contract(documents: Mapping[str, str]) -> None:
    current = "\n".join(documents.values())
    for required in _ISSUE62_REQUIRED_CURRENT_FACTS:
        assert required in current
    for forbidden in _ISSUE62_FORBIDDEN_CURRENT_FACTS:
        assert forbidden not in current


def test_issue62_current_surfaces_publish_only_the_implemented_contract() -> None:
    _assert_issue62_current_contract(_issue62_current_documents())


@pytest.mark.parametrize("required", _ISSUE62_REQUIRED_CURRENT_FACTS)
def test_issue62_current_contract_rejects_required_fact_omission(required: str) -> None:
    documents = _issue62_current_documents()
    mutated = False
    for path, current in documents.items():
        if required in current:
            documents[path] = current.replace(required, "removed-current-fact")
            mutated = True
    assert mutated

    with pytest.raises(AssertionError):
        _assert_issue62_current_contract(documents)


@pytest.mark.parametrize("forbidden", _ISSUE62_FORBIDDEN_CURRENT_FACTS)
def test_issue62_current_contract_rejects_obsolete_fact(forbidden: str) -> None:
    documents = _issue62_current_documents()
    documents["README.md"] += f"\n{forbidden}\n"

    with pytest.raises(AssertionError):
        _assert_issue62_current_contract(documents)


def _assert_issue62_platform_qualification_is_complete_and_locked(section: str) -> None:
    normalized = " ".join(section.split())
    assert "requirements/lock-policy.toml" in section
    assert "reproducible for that qualified platform lock" in normalized
    assert _ISSUE62_PENDING_PLATFORM_SENTENCE not in section


def test_issue62_platform_qualification_is_complete_and_lock_preserved() -> None:
    text = (REPO_ROOT / "docs/env-setup.md").read_text(encoding="utf-8")
    local_venv = _same_level_section(text, "4.1.3 Local Python venv")

    _assert_issue62_platform_qualification_is_complete_and_locked(local_venv)


def test_issue62_platform_contract_rejects_restored_pending_claim() -> None:
    text = (REPO_ROOT / "docs/env-setup.md").read_text(encoding="utf-8")
    local_venv = _same_level_section(text, "4.1.3 Local Python venv")
    mutated = f"{local_venv}\n\n{_ISSUE62_PENDING_PLATFORM_SENTENCE}"
    assert mutated != local_venv
    assert _ISSUE62_PENDING_PLATFORM_SENTENCE in mutated

    with pytest.raises(AssertionError):
        _assert_issue62_platform_qualification_is_complete_and_locked(mutated)


def test_issue66_quantization_guidance_uses_tier_b_contract() -> None:
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    nnx_overview = (REPO_ROOT / "docs/nnx-library.md").read_text(encoding="utf-8")
    design = (
        REPO_ROOT
        / "docs/superpowers/specs/2026-08-22-issue-66-quantization-ci-design.md"
    ).read_text(encoding="utf-8")
    task_spec = (
        REPO_ROOT / "notebooks/quantization-mnist-ffnn-pytorch/docs/spec.yaml"
    ).read_text(encoding="utf-8")
    generated_task_doc = (
        REPO_ROOT / "docs/notebooks/quantization-mnist-ffnn-pytorch.md"
    ).read_text(encoding="utf-8")
    task_readme = (
        REPO_ROOT / "notebooks/quantization-mnist-ffnn-pytorch/README.md"
    ).read_text(encoding="utf-8")
    surfaces = {
        "docs/env-setup.md": _between(
            (REPO_ROOT / "docs/env-setup.md").read_text(), "**How to use**:\n", "See `.devcontainer"
        ),
        "CONTRIBUTING.md": _same_level_section(contributing, "4. Modifying shared code"),
        "notebooks/quantization-mnist-ffnn-pytorch/README.md": _same_level_section(
            task_readme, "4. How to run"
        ),
    }

    for path, current in surfaces.items():
        assert (
            "Tier B" in current
            or "Tier-B" in current
            or "smoke-tier-b" in current
        ), path
        assert "manual-only" not in current.lower(), path
        assert "outside Tier A/B/C" not in current, path
    assert "make smoke-tier-b" in surfaces["docs/env-setup.md"]
    assert "make install-torch-stack" in surfaces[
        "notebooks/quantization-mnist-ffnn-pytorch/README.md"
    ]
    unsupported = _between(
        (REPO_ROOT / "docs/env-setup.md").read_text(), "**Scenarios this does NOT support**:\n", "**How to use**:\n"
    )
    assert "quantization-mnist-ffnn-pytorch" not in unsupported
    extending_nnx = _same_level_section(nnx_overview, "7.4 Extending NNx")
    assert "Manual quantization validation" not in extending_nnx
    assert "Tier B quantization validation" in extending_nnx
    assert "artifact upload" not in design
    for current in (task_spec, generated_task_doc):
        assert "torch.load(qat_checkpoint_path, weights_only=False)" in current
        assert "NNCheckpoint.load" not in current


def test_codespaces_guidance_avoids_unbenchmarked_setup_duration() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Setup time varies with image-cache and network state." in readme
    assert "~2-3 minutes" not in readme
    assert "~2–3 minutes" not in readme


def test_graphconv_guidance_matches_nnx_gcnconv_wrapper() -> None:
    task_doc = (
        REPO_ROOT / "docs/notebooks/node_classification-reddit-gnn-pyg.md"
    ).read_text(encoding="utf-8")

    assert "`GraphConvNN` wraps `pyg.nn.GCNConv`" in task_doc
    assert "symmetrically normalized sum" in task_doc
    assert "symmetrically-normalized mean" not in task_doc


def test_published_opener_record_uses_clinical_artifact_provenance() -> None:
    record = (
        REPO_ROOT
        / "docs/superpowers/plans/2026-08-01-opener-visual-remediation-implementation-plan.md"
    ).read_text(encoding="utf-8")

    assert "Artifact provenance:" in record
    assert "Use this production prompt:" not in record
    assert "Scene/backdrop:" not in record
    assert "Lighting/mood:" not in record


_ISSUE66_TIER_MATRIX_GUIDANCE = (
    "Every NNx release review must run the complete Tier A, Tier B, and Tier C matrix in a "
    "fresh canonical environment installed by `make install-torch-stack`"
)

def _assert_issue66_nnx_review_uses_tier_matrix(section: str) -> None:
    assert _ISSUE66_TIER_MATRIX_GUIDANCE in section
    assert "Tier B includes the full PTQ/QAT lifecycle" in section
    assert "focused manual quantization probe" not in section
    assert "outside Tier A/B/C" not in section


def test_issue66_nnx_review_requires_the_quantization_tier_matrix() -> None:
    text = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    shared_code = _same_level_section(text, "4. Modifying shared code")

    _assert_issue66_nnx_review_uses_tier_matrix(shared_code)


def test_issue66_nnx_review_rejects_restored_manual_probe() -> None:
    text = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    shared_code = _same_level_section(text, "4. Modifying shared code")
    replacement = (
        "Every NNx release review must run the complete Tier A, Tier B, and Tier C matrix plus "
        "the focused manual quantization probe"
    )
    mutated = shared_code.replace(_ISSUE66_TIER_MATRIX_GUIDANCE, replacement, 1)
    assert mutated != shared_code

    with pytest.raises(AssertionError):
        _assert_issue66_nnx_review_uses_tier_matrix(mutated)


def test_issue62_dependency_sections_replace_complete_old_contracts():
    text = (REPO_ROOT / "docs/dependency-contracts.md").read_text(encoding="utf-8")
    torch_section = _same_level_section(text, "6.1.2 Torch Stack Pin")
    for exact in (
        "torch==2.11.0",
        "pytorch-lightning==2.6.1",
        "torch-geometric==2.8.0.post1",
        "complete hash-required platform lock",
        "shared sanitized boundary",
        "four-surface advisory reconciliation from six commands",
        "Tier A/B/C 18/7/4",
    ):
        assert exact in torch_section
    for obsolete in (
        "2.4.1", "torch-cluster", "torch-spline-conv", "--no-build-isolation",
        "source build", "deliberately stable local/CI compatibility baseline",
    ):
        assert obsolete not in torch_section
    bootstrap = _same_level_section(text, "6.1.11 Canonical Bootstrap Tooling")
    assert " ".join(bootstrap.split()) == (
        "The canonical installer uses the hash-required bootstrap lock and then installs the "
        "complete platform lock selected by `requirements/lock-policy.toml`. Docker, Codespaces, "
        "CI, and local setup delegate to that shared boundary; none carries a second dependency "
        "algorithm. Compiler tooling is isolated in its own lock and is never part of the runtime "
        "environment."
    )


def test_issue62_notebook_specs_drive_exact_generated_rows():
    graph = yaml.safe_load((
        REPO_ROOT / "notebooks/node_classification-reddit-gnn-pyg/docs/spec.yaml"
    ).read_text(encoding="utf-8"))
    quant = yaml.safe_load((
        REPO_ROOT / "notebooks/quantization-mnist-ffnn-pytorch/docs/spec.yaml"
    ).read_text(encoding="utf-8"))
    assert graph["atlas"]["constraints"] == [
        "Issue #62 requires preferred pyg-lib sampling and forced torch-sparse fallback on the "
        "repository Torch 2.11 CPU stack; the retained Atlas Torch 2.13 runtime executes through "
        "pyg-lib and intentionally has no legacy torch-sparse wheel."
    ]
    assert quant["atlas"]["constraints"] == [
        "Tier-B smoke is deterministic and bounded to one epoch for FP32 plus one epoch for QAT.",
        "QAT acceptance reconstructs the saved FP-shadow checkpoint with exact state/metadata "
        "parity and separately proves final torchao conversion.",
    ]
    generated = (
        REPO_ROOT / "docs/notebook-infrastructure.md"
    ).read_text(encoding="utf-8").splitlines()
    graph_row = next(line for line in generated if "node_classification-reddit-gnn-pyg" in line)
    quant_row = next(line for line in generated if "quantization-mnist-ffnn-pytorch" in line)
    assert all(
        token in graph_row
        for token in ("pyg-lib", "torch-sparse", "retained Atlas Torch 2.13 runtime")
    )
    assert all(
        token in quant_row
        for token in ("| B |", "one epoch", "state/metadata parity", "torchao conversion")
    )


def test_issue66_tier_b_uses_actual_environment_heading():
    text = (REPO_ROOT / "docs/env-setup.md").read_text(encoding="utf-8")
    tier_mapping = _same_level_section(text, "4.1.6 Tier mapping")
    tier_b = next(line for line in tier_mapping.splitlines() if line.startswith("- **Tier B:**"))
    assert "one FP32 epoch plus one QAT epoch" in tier_mapping
    assert "exactly" in tier_mapping
    assert "manual-only" not in tier_b.lower()


def test_issue62_graph_canonical_page_has_current_release_guidance():
    text = (
        REPO_ROOT / "docs/notebooks/node_classification-reddit-gnn-pyg.md"
    ).read_text(encoding="utf-8")
    pitfalls = _same_level_section(text, "8.13.7 Pitfalls")
    exact = (
        "- **Run both graph tiers during release review.** Issue #62 requires mandatory "
        "zero-skip graph tests plus Tier B and Tier C execution on the supported Torch 2.11 "
        "CPU stack. Sampling must prove preferred pyg-lib selection and forced torch-sparse "
        "fallback; install with make install-torch-stack and prove with make verify-torch-stack."
    )
    assert exact in pitfalls
    assert "Issue #61 completed Tier B and Tier C" not in pitfalls
    assert "with `torch_sparse==0.6.18`" not in pitfalls


_NNX_RETAINED_TRIAL_FACTS = (
    "thekaveh-nnx[lm]==0.2.0",
    "1,350",
    "18/18",
    "6/6",
    "4/4",
    "Torch 2.11.0",
    "torchvision 0.26.0",
    "torchao 0.18.0",
)


def _assert_nnx_retain_decision_docs(documents: dict[str, str]) -> None:
    readme = documents["README.md"]
    ledger = documents["docs/dependency-contracts.md"]
    overview = documents["docs/nnx-library.md"]
    changelog = documents["CHANGELOG.md"]
    current_changelog = changelog[: changelog.index("## [0.1.0]")]
    quantization_surfaces = (
        documents["docs/notebooks/quantization-mnist-ffnn-pytorch.md"],
        documents["notebooks/quantization-mnist-ffnn-pytorch/README.md"],
        documents["notebooks/quantization-mnist-ffnn-pytorch/notebook.ipynb"],
    )
    current_documents = {
        path: current_changelog if path == "CHANGELOG.md" else text
        for path, text in documents.items()
    }
    combined = "\n".join(current_documents.values())

    for fact in _NNX_RETAINED_TRIAL_FACTS:
        assert fact in ledger or fact in overview or fact in current_changelog
    assert "thekaveh-nnx[lm]==0.2.0" in readme
    assert "Atlas JupyterHub" in readme
    assert (
        "Tier B and Tier C completed on Darwin arm64 with `torch_sparse==0.6.18` imported"
        in current_changelog
    )
    assert "Every NNx release review must run the complete Tier A, Tier B, and Tier C matrix" in ledger
    assert (
        "canonical trial passed `1,350` repository tests, Tier A `18/18`, "
        "Tier B `6/6`, and Tier C `4/4`"
    ) in ledger
    assert "Darwin arm64" in ledger
    assert "torch_sparse==0.6.18" in ledger
    assert "28 of the 29 active notebooks" in overview
    assert "target_col" in documents["docs/FINDINGS-NNX.md"]
    assert "y_col" not in documents["docs/FINDINGS-NNX.md"]
    active_guidance = combined.replace("`NNRun.load(\"best\")` (not a v0.2.0 idiom", "")
    active_guidance = active_guidance.replace("no `NNRun.load(\"best\")`", "")
    assert "NNRun.load(\"best\")" not in active_guidance
    assert "NNRun.load('best')" not in active_guidance
    assert "torch>=2.5" not in combined
    assert "torch >=2.5" not in combined
    assert "Torch >=2.5" not in combined
    assert "torch >= 2.5" not in combined
    assert "torch ≥ 2.5" not in combined
    for surface in quantization_surfaces:
        assert "Torch 2.11.0" in surface
        assert "torchvision 0.26.0" in surface or "torchvision==0.26.0" in surface
        assert "torchao 0.18.0" in surface


def _nnx_current_documents() -> dict[str, str]:
    paths = (
        "README.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "docs/FINDINGS-NNX.md",
        "docs/concepts.md",
        "docs/dependency-contracts.md",
        "docs/nnx-library.md",
        "docs/notebooks/dim_reduction-iris-autoencoder-pytorch.md",
        "docs/notebooks/node_classification-reddit-gnn-pyg.md",
        "docs/notebooks/quantization-mnist-ffnn-pytorch.md",
        "docs/notebooks/tabular_classification-iris-mlp-pytorch.md",
        "notebooks/quantization-mnist-ffnn-pytorch/README.md",
        "notebooks/quantization-mnist-ffnn-pytorch/notebook.ipynb",
    )
    return {path: (REPO_ROOT / path).read_text(encoding="utf-8") for path in paths}


def test_nnx_current_docs_record_completed_trial_and_retained_default_runtime():
    _assert_nnx_retain_decision_docs(_nnx_current_documents())


@pytest.mark.parametrize(
    ("path", "old", "new"),
    (
        ("README.md", "thekaveh-nnx[lm]==0.2.0", "thekaveh-nnx[lm]==0.2.2"),
        ("docs/dependency-contracts.md", "Tier C `4/4`", "Tier C pending"),
        ("docs/dependency-contracts.md", "Torch 2.11.0", "Torch >=2.5"),
        (
            "docs/notebooks/quantization-mnist-ffnn-pytorch.md",
            "Torch 2.11.0",
            "Torch >=2.5",
        ),
        ("docs/nnx-library.md", "28 of the 29 active notebooks", "about two dozen active notebooks"),
    ),
)
def test_nnx_retain_decision_docs_reject_mutations(path, old, new):
    documents = _nnx_current_documents()
    assert old in documents[path]
    documents[path] = documents[path].replace(old, new)
    with pytest.raises(AssertionError):
        _assert_nnx_retain_decision_docs(documents)


_NNX_FINDING_TRIAGE_URLS = (
    "https://github.com/thekaveh/NNx/issues/188",
    "https://github.com/thekaveh/NNx/issues/81",
    "https://github.com/thekaveh/NNx/issues/189",
    "https://github.com/thekaveh/NNx/issues/190",
    "https://github.com/thekaveh/ml-eng-lab/issues/146",
)


def _assert_nnx_finding_triage_docs(text: str) -> None:
    headings = (
        "### 9.1.1.1 `NNDataset` default `batch_size` packs the whole train set into one batch",
        "### 9.1.1.2 `nnx.deepen` is function-preserving only for `Activations.RELU`",
        "### 9.1.1.3 `NNTabularDataset` coerces targets to `torch.long` (classification-only)",
        '### 9.1.1.4 `EarlyStopping(monitor=...)` default is `"val_edp.error"`, which is unset (`None`) for regression EDPs',
        "### 9.1.1.5 training completion messages expose absolute run paths",
    )
    for heading in headings:
        assert text.count(heading) == 1

    assert text.count("**Triage summary (last reviewed 2026-08-22):**") == 1
    assert "surfaced here for an upstream PR follow-up" not in text

    row_matches = re.findall(r"^\| (F[1-5]) \| (.+)$", text, re.MULTILINE)
    rows = {finding_id: row for finding_id, row in row_matches}
    assert set(rows) == {"F1", "F2", "F3", "F4", "F5"}

    section_matches = list(
        re.finditer(r"^### 9\.1\.1\.([1-5]) .+$", text, re.MULTILINE)
    )
    assert len(section_matches) == 5
    sections = {}
    for index, match in enumerate(section_matches):
        end = section_matches[index + 1].start() if index + 1 < len(section_matches) else len(text)
        sections[f"F{match.group(1)}"] = text[match.start():end]

    expected_tokens = {
        "F1": (
            "https://github.com/thekaveh/NNx/issues/188",
            "diffusion-mnist-ddpm-pytorch",
            "Issue #69",
            "no follow-up",
        ),
        "F2": (
            "documented design constraint",
            "not an open upstream bug",
            "model_surgery-mnist-ffnn-pytorch",
            "no follow-up",
        ),
        "F3": (
            "https://github.com/thekaveh/NNx/issues/81",
            "v0.2.2",
            "tabular_regression-diabetes-mlp-pytorch",
            "https://github.com/thekaveh/ml-eng-lab/issues/146",
        ),
        "F4": (
            "https://github.com/thekaveh/NNx/issues/189",
            "tabular_regression-diabetes-mlp-pytorch",
            '`monitor="val_edp.loss"`',
            "no local code follow-up",
        ),
        "F5": (
            "https://github.com/thekaveh/NNx/issues/190",
            "Any notebook that captures NNx training output",
            "E13.stale_active_notebook_path",
            "no local follow-up",
        ),
    }
    detail_tokens = {
        "F1": ("https://github.com/thekaveh/NNx/issues/188", "Issue #69"),
        "F2": ("not an open upstream bug", "model_surgery-mnist-ffnn-pytorch"),
        "F3": (
            "https://github.com/thekaveh/NNx/issues/81",
            "https://github.com/thekaveh/ml-eng-lab/issues/146",
        ),
        "F4": ("https://github.com/thekaveh/NNx/issues/189", '`monitor="val_edp.loss"`'),
        "F5": (
            "https://github.com/thekaveh/NNx/issues/190",
            "E13.stale_active_notebook_path",
        ),
    }
    standard_labels = (
        "**Upstream disposition:**",
        "**Release evidence:**",
        "**Affected notebooks:**",
        "**Local workaround/status:**",
        "**Remaining work:**",
    )
    for finding_id, tokens in expected_tokens.items():
        assert all(token in rows[finding_id] for token in tokens)
        assert all(label in sections[finding_id] for label in standard_labels)
        normalized_section = re.sub(r"\s+", " ", sections[finding_id])
        assert all(token in normalized_section for token in detail_tokens[finding_id])

    for url in _NNX_FINDING_TRIAGE_URLS:
        assert url in text

    assert "v0.2.2" in text
    assert "documented design constraint" in text
    assert "not an open upstream bug" in text
    assert "`NNModel.train()` and `Trainer.train()`" in text
    assert "`NNRun.save()` prints an absolute path" not in text
    for notebook in (
        "diffusion-mnist-ddpm-pytorch",
        "moe-fmnist-mixture-of-experts-pytorch",
        "self_supervised-fmnist-jepa-pytorch",
        "model_surgery-mnist-ffnn-pytorch",
        "tabular_regression-diabetes-mlp-pytorch",
    ):
        assert notebook in text


def test_nnx_findings_have_durable_upstream_triage():
    text = (REPO_ROOT / "docs/FINDINGS-NNX.md").read_text(encoding="utf-8")
    _assert_nnx_finding_triage_docs(text)


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("| F1 |", "| missing-F1 |"),
        ("https://github.com/thekaveh/NNx/issues/81", "https://github.com/thekaveh/NNx/issues/999999"),
        ("v0.2.2", "unreleased"),
        ("documented design constraint", "unresolved bug"),
        ("`NNModel.train()` and `Trainer.train()`", "`NNRun.save()`"),
    ),
)
def test_nnx_finding_triage_docs_reject_mutations(old, new):
    text = (REPO_ROOT / "docs/FINDINGS-NNX.md").read_text(encoding="utf-8")
    assert old in text
    with pytest.raises(AssertionError):
        _assert_nnx_finding_triage_docs(text.replace(old, new))


def test_nnx_finding_triage_docs_reject_cross_associated_issue_links():
    text = (REPO_ROOT / "docs/FINDINGS-NNX.md").read_text(encoding="utf-8")
    swapped = (
        text.replace("/NNx/issues/188", "/NNx/issues/swap")
        .replace("/NNx/issues/189", "/NNx/issues/188")
        .replace("/NNx/issues/swap", "/NNx/issues/189")
    )

    with pytest.raises(AssertionError):
        _assert_nnx_finding_triage_docs(swapped)


def test_nnx_finding_triage_docs_require_standard_labels_per_finding():
    text = (REPO_ROOT / "docs/FINDINGS-NNX.md").read_text(encoding="utf-8")
    f3_start = text.index("### 9.1.1.3")
    f4_start = text.index("### 9.1.1.4")
    f3 = text[f3_start:f4_start].replace("**Release evidence:**", "**Evidence:**", 1)
    mutated = text[:f3_start] + f3 + text[f4_start:]

    with pytest.raises(AssertionError):
        _assert_nnx_finding_triage_docs(mutated)


def test_model_surgery_docs_do_not_claim_init_can_preserve_non_relu_deepening():
    paths = (
        "notebooks/model_surgery-mnist-ffnn-pytorch/README.md",
        "notebooks/model_surgery-mnist-ffnn-pytorch/notebook.ipynb",
        "docs/notebooks/model_surgery-mnist-ffnn-pytorch.md",
    )
    for path in paths:
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert "different post-insertion init" not in text
        assert "different post-insertion bias init" not in text
        assert "materially different surgery operator" in text


_EARLY_STOPPING_REGRESSION_DOCS = (
    "notebooks/tabular_regression-diabetes-mlp-pytorch/README.md",
    "docs/notebooks/tabular_regression-diabetes-mlp-pytorch.md",
    "docs/superpowers/plans/2026-08-22-issue-72-nnx-upstream-triage-implementation-plan.md",
)


def _assert_early_stopping_regression_field_wording(documents: Mapping[str, str]) -> None:
    assert set(documents) == set(_EARLY_STOPPING_REGRESSION_DOCS)
    for text in documents.values():
        assert "no `error` field" not in text
        assert "unset (`None`)" in text


def test_early_stopping_regression_field_wording_is_synced_across_surfaces():
    documents = {
        path: (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in _EARLY_STOPPING_REGRESSION_DOCS
    }
    _assert_early_stopping_regression_field_wording(documents)


@pytest.mark.parametrize("path", _EARLY_STOPPING_REGRESSION_DOCS)
def test_early_stopping_regression_field_wording_rejects_surface_drift(path):
    documents = {
        candidate: (REPO_ROOT / candidate).read_text(encoding="utf-8")
        for candidate in _EARLY_STOPPING_REGRESSION_DOCS
    }
    assert "unset (`None`)" in documents[path]
    documents[path] = documents[path].replace("unset (`None`)", "absent", 1)

    with pytest.raises(AssertionError):
        _assert_early_stopping_regression_field_wording(documents)


def test_nnx_current_pin_and_issue63_advisory_snapshot_match_restored_manifest():
    requirements = (REPO_ROOT / "requirements.txt").read_bytes()
    ledger = (REPO_ROOT / "docs/dependency-contracts.md").read_text(encoding="utf-8")
    audit = ledger.split("### 6.1.1.2 Current Issue #63 locked four-surface audit", 1)[1].split(
        "### 6.1.1.3 Alias-aware historical reconciliation", 1
    )[0]

    assert hashlib.sha256(requirements).hexdigest() == (
        "6e86caa5a287e9566e15bdffbb6628249397307783dee3b6e98e728ef06275b9"
    )
    assert "Last reviewed: 2026-08-16" in audit
    assert "`requirements.txt` | `6e86caa5a287e9566e15bdffbb6628249397307783dee3b6e98e728ef06275b9`" in audit
    assert "thekaveh-nnx==0.2.2" not in audit


def test_nnx_documented_consumer_count_matches_tracked_active_notebooks():
    tracked = subprocess.run(
        ["git", "ls-files", "--", "notebooks/**/*.ipynb"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    active = [
        path
        for path in tracked
        if not path.startswith("notebooks/archive/")
        and ".ipynb_checkpoints" not in path
    ]
    consumers = [
        path
        for path in active
        if re.search(
            r"(?:from|import)\s+nnx(?:\b|\.)",
            (REPO_ROOT / path).read_text(encoding="utf-8"),
        )
    ]

    assert (len(consumers), len(active)) == (28, 29)


def test_nnx_historical_output_notices_do_not_claim_current_022_source():
    paths = (
        "notebooks/knowledge_distillation-mnist-ffnn-pytorch/notebook.ipynb",
        "notebooks/peft-mnist-to-fmnist-dora-vs-lora-pytorch/notebook.ipynb",
        "notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook1.ipynb",
        "notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook2.ipynb",
    )
    for path in paths:
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert "temporary artifacts" in text, path
        assert "0.2.0" in text, path
        assert "current NNx 0.2.2" not in text, path


_ISSUE63_CURRENT_DOC_PATHS = (
    "README.md",
    "CONTRIBUTING.md",
    "docs/conventions.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "docs/env-setup.md",
    "docs/architecture.md",
    "docs/dependency-contracts.md",
    "docs/notebook-infrastructure.md",
    "docs/jupyterhub-integration.md",
    "docs/nnx-library.md",
    "docs/notebooks/text_classification-agnews-spacy-mlp-pytorch.md",
    "notebooks/text_classification-agnews-spacy-mlp-pytorch/docs/spec.yaml",
)


def _assert_issue63_current_docs(documents):
    for path in (
        "README.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "docs/env-setup.md",
        "docs/architecture.md",
        "docs/dependency-contracts.md",
        "docs/notebook-infrastructure.md",
    ):
        assert "requirements/lock-policy.toml" in documents[path]
    assert "reproducible for the qualified platform lock" in documents["README.md"]
    assert "make lock-check" in documents["README.md"]
    assert "make lock-write" in documents["CONTRIBUTING.md"]
    assert "make lock-check" in documents["CONTRIBUTING.md"]
    assert "make image-lock-check" in documents["CONTRIBUTING.md"]
    assert "Issue #63" in documents["CHANGELOG.md"].split("## [0.1.0]", 1)[0]
    assert "### 6.1.1.2 Current Issue #63 locked four-surface audit" in documents[
        "docs/dependency-contracts.md"
    ]
    for issue in ("Issue #64", "Issue #65", "Issue #66"):
        assert issue in documents["docs/dependency-contracts.md"]
    assert "make image-lock-check" in documents["docs/dependency-contracts.md"]
    assert "reproducible for that qualified platform lock" in " ".join(
        documents["docs/env-setup.md"].split()
    )
    assert "CONDA_AUTO_ACTIVATE_BASE=false" in documents["docs/architecture.md"]
    assert "hash-required Linux lock" in documents["docs/conventions.md"]
    assert "requirements/locks/atlas-contract.txt" in documents[
        "docs/jupyterhub-integration.md"
    ]
    assert "selected hash-required platform lock" in documents["docs/nnx-library.md"]
    assert "python -m spacy download" not in documents["CONTRIBUTING.md"]
    assert "python -m spacy download" not in documents[
        "docs/notebooks/text_classification-agnews-spacy-mlp-pytorch.md"
    ]
    assert "python -m spacy download" not in documents[
        "notebooks/text_classification-agnews-spacy-mlp-pytorch/docs/spec.yaml"
    ]
    assert "en-core-web-sm" in documents[
        "docs/notebooks/text_classification-agnews-spacy-mlp-pytorch.md"
    ]


def _issue63_current_documents():
    return {
        path: (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in _ISSUE63_CURRENT_DOC_PATHS
    }


def test_issue63_current_docs_describe_the_immutable_lock_contract():
    _assert_issue63_current_docs(_issue63_current_documents())


@pytest.mark.parametrize(
    ("path", "old", "new"),
    (
        ("README.md", "make lock-check", "pip install -r requirements.txt"),
        ("CONTRIBUTING.md", "make lock-write", "uv pip compile"),
        ("SECURITY.md", "requirements/lock-policy.toml", "requirements.txt"),
        (
            "docs/env-setup.md",
            "reproducible for that qualified",
            "perfectly reproducible across every platform",
        ),
        (
            "docs/architecture.md",
            "CONDA_AUTO_ACTIVATE_BASE=false",
            "CONDA_AUTO_ACTIVATE_BASE=true",
        ),
        (
            "docs/dependency-contracts.md",
            "Current Issue #63 locked four-surface audit",
            "docker pull latest",
        ),
        (
            "docs/conventions.md",
            "hash-required Linux lock",
            "requirements.txt with --only-binary=thekaveh-nnx",
        ),
        (
            "docs/jupyterhub-integration.md",
            "requirements/locks/atlas-contract.txt",
            "focused dependency manifest",
        ),
        (
            "docs/nnx-library.md",
            "selected hash-required platform lock",
            "pip install -r requirements.txt",
        ),
    ),
)
def test_issue63_current_docs_reject_stale_claim_mutations(path, old, new):
    documents = _issue63_current_documents()
    assert old in documents[path]
    documents[path] = documents[path].replace(old, new, 1)
    with pytest.raises(AssertionError):
        _assert_issue63_current_docs(documents)


_ISSUE64_CURRENT_DOC_PATHS = (
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "docs/conventions.md",
    "docs/env-setup.md",
    "docs/dependency-contracts.md",
    "docs/notebook-infrastructure.md",
    "docs/notebooks/sentiment_classification-vader-mlp-pytorch.md",
    "notebooks/sentiment_classification-vader-mlp-pytorch/README.md",
    "notebooks/sentiment_classification-vader-mlp-pytorch/notebook.ipynb",
    "notebooks/text_classification-agnews-spacy-mlp-pytorch/README.md",
)
_VADER_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/"
    "sentiment/vader_lexicon.zip"
)
_VADER_SHA256 = "8adba4294eef3964d820bf655e37e61bdc3a341994356af59b74fb3b4a36ce5c"


def _issue64_current_documents():
    return {
        path: (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in _ISSUE64_CURRENT_DOC_PATHS
    }


def _assert_issue64_current_docs(documents):
    current_changelog = documents["CHANGELOG.md"].split("## [0.1.0]", 1)[0]
    ledger = documents["docs/dependency-contracts.md"]
    sentiment = documents[
        "docs/notebooks/sentiment_classification-vader-mlp-pytorch.md"
    ]
    current = "\n".join(
        current_changelog if path == "CHANGELOG.md" else text
        for path, text in documents.items()
    )

    assert _VADER_URL in ledger
    assert _VADER_SHA256 in ledger
    assert "90,486" in ledger
    assert "make nlp-assets" in documents["README.md"]
    assert "make verify-nlp-assets" in documents["README.md"]
    assert "make verify-nlp-assets" in documents["docs/env-setup.md"]
    assert "offline" in documents["docs/env-setup.md"].lower()
    assert "make nlp-assets" in documents["docs/notebook-infrastructure.md"]
    assert "make verify-nlp-assets" in documents["docs/notebook-infrastructure.md"]
    assert "nltk.download" not in documents["docs/notebook-infrastructure.md"]
    assert "make verify-nlp-assets" in documents["CONTRIBUTING.md"]
    assert "nltk.download" not in documents["CONTRIBUTING.md"]
    assert "five-check oracle" in documents["CONTRIBUTING.md"]
    assert "runs five checks" in documents["docs/conventions.md"]
    assert "Assets (`D11`)" in documents["docs/conventions.md"]
    sentiment_readme = documents[
        "notebooks/sentiment_classification-vader-mlp-pytorch/README.md"
    ]
    sentiment_notebook = documents[
        "notebooks/sentiment_classification-vader-mlp-pytorch/notebook.ipynb"
    ]
    agnews_readme = documents[
        "notebooks/text_classification-agnews-spacy-mlp-pytorch/README.md"
    ]
    assert "make verify-nlp-assets" in sentiment_readme
    assert "nltk.download" not in sentiment_readme
    assert "make verify-nlp-assets" in sentiment_notebook
    assert "lazily downloaded above" not in sentiment_notebook
    assert "en-core-web-sm==3.8.0" in agnews_readme
    assert "spacy download" not in agnews_readme
    assert "make nlp-assets" in current
    assert "make verify-nlp-assets" in current
    assert "offline" in current.lower()
    assert "Issue #63" in current_changelog
    assert "Atlas" in ledger and "projection" in ledger
    assert "integrity" in sentiment.lower()
    assert "nltk.download" not in sentiment
    assert "lazy lexicon download" not in sentiment.lower()


def test_issue64_current_docs_describe_verified_nlp_assets():
    _assert_issue64_current_docs(_issue64_current_documents())


@pytest.mark.parametrize(
    ("path", "old", "new"),
    (
        ("docs/dependency-contracts.md", _VADER_URL, "https://example.invalid/vader.zip"),
        ("docs/dependency-contracts.md", _VADER_SHA256, "0" * 64),
        ("README.md", "make verify-nlp-assets", "python -m nltk.downloader vader_lexicon"),
        ("docs/env-setup.md", "offline", "downloads on every verification"),
        ("docs/notebook-infrastructure.md", "make nlp-assets", "nltk.download"),
        ("CONTRIBUTING.md", "make verify-nlp-assets", "nltk.download"),
        ("CONTRIBUTING.md", "five-check oracle", "four-check oracle"),
        ("docs/conventions.md", "Assets (`D11`)", "Assets (`D12`)"),
        (
            "notebooks/sentiment_classification-vader-mlp-pytorch/README.md",
            "make verify-nlp-assets",
            "nltk.download",
        ),
        (
            "notebooks/sentiment_classification-vader-mlp-pytorch/notebook.ipynb",
            "make verify-nlp-assets",
            "lazily downloaded above",
        ),
        (
            "notebooks/text_classification-agnews-spacy-mlp-pytorch/README.md",
            "en-core-web-sm==3.8.0",
            "spacy download",
        ),
        (
            "docs/notebooks/sentiment_classification-vader-mlp-pytorch.md",
            "integrity",
            "lazy lexicon download",
        ),
    ),
)
def test_issue64_current_docs_reject_asset_contract_mutations(path, old, new):
    documents = _issue64_current_documents()
    assert old in documents[path]
    documents[path] = documents[path].replace(old, new)
    with pytest.raises(AssertionError):
        _assert_issue64_current_docs(documents)


_ISSUE65_CURRENT_DOC_PATHS = (
    "README.md",
    "CHANGELOG.md",
    "docs/env-setup.md",
    "docs/atlas-pin-bump-runbook.md",
    "docs/dependency-contracts.md",
    "docs/notebook-infrastructure.md",
)
_ISSUE65_CURRENT_SHA = "41ba856f7cd35f0b559d6875e08443eac3e98a98"
_ISSUE65_ROLLBACK_SHA = "61c7c5103660e2226bf107c115dae42bf46f8374"


def _issue65_current_documents() -> dict[str, str]:
    return {
        path: (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in _ISSUE65_CURRENT_DOC_PATHS
    }


def _assert_issue65_current_docs(documents: Mapping[str, str]) -> None:
    marker = f"Current reviewed Atlas pin: `{_ISSUE65_CURRENT_SHA}`."
    for path in (
        "README.md",
        "docs/env-setup.md",
        "docs/atlas-pin-bump-runbook.md",
    ):
        assert documents[path].count(marker) == 1, path

    ledger = documents["docs/dependency-contracts.md"]
    for expected in (
        "Issue #65 Atlas pin review",
        "30 first-parent commits",
        "43 total commits",
        "empty current-to-main range",
        "Torch 2.13.0",
        "fastmcp==3.4.4",
        "managed-host-process",
        "ComfyUI remains disabled",
        "Last verified: 2026-08-22",
        "CPython 3.11.10",
        "datasets` 5.0.1",
        "tokenizers` 0.22.2",
        "torch` 2.13.0+cpu",
        "torchvision` 0.28.0+cpu",
        "torchao` 0.17.0",
        "torch-geometric` 2.7.0",
        "NLTK 3.10.1",
        "47 mandatory imports",
        "removed the four historical unused Phase-3 `SparseTensor` imports",
        "no executable notebook imports `torchaudio`",
        _ISSUE65_CURRENT_SHA,
        _ISSUE65_ROLLBACK_SHA,
    ):
        assert expected in ledger, expected

    unreleased = documents["CHANGELOG.md"].split("## [0.1.0]", 1)[0]
    assert "Issue #65" in unreleased
    assert _ISSUE65_CURRENT_SHA in unreleased
    assert "Issue #65" not in documents["docs/notebook-infrastructure.md"]


def test_issue65_current_docs_record_the_atlas_retain_decision() -> None:
    _assert_issue65_current_docs(_issue65_current_documents())


@pytest.mark.parametrize(
    ("path", "old", "new"),
    (
        ("README.md", _ISSUE65_CURRENT_SHA, _ISSUE65_ROLLBACK_SHA),
        ("docs/env-setup.md", _ISSUE65_CURRENT_SHA, _ISSUE65_ROLLBACK_SHA),
        (
            "docs/atlas-pin-bump-runbook.md",
            _ISSUE65_CURRENT_SHA,
            _ISSUE65_ROLLBACK_SHA,
        ),
        ("docs/dependency-contracts.md", "30 first-parent commits", "29 first-parent commits"),
        ("docs/dependency-contracts.md", "43 total commits", "42 total commits"),
        ("docs/dependency-contracts.md", "empty current-to-main range", "unreviewed range"),
        ("docs/dependency-contracts.md", "Torch 2.13.0", "Torch unknown"),
        ("docs/dependency-contracts.md", "fastmcp==3.4.4", "fastmcp unpinned"),
        ("docs/dependency-contracts.md", "managed-host-process", "manual host process"),
        ("docs/dependency-contracts.md", "ComfyUI remains disabled", "ComfyUI enabled"),
        ("docs/dependency-contracts.md", "CPython 3.11.10", "CPython unknown"),
        ("docs/dependency-contracts.md", "47 mandatory imports", "48 imports"),
        (
            "docs/dependency-contracts.md",
            "removed the four historical unused Phase-3 `SparseTensor` imports",
            "retained the four historical unused Phase-3 `SparseTensor` imports",
        ),
        (
            "docs/dependency-contracts.md",
            "no executable notebook imports `torchaudio`",
            "torchaudio is assumed",
        ),
        ("CHANGELOG.md", _ISSUE65_CURRENT_SHA, _ISSUE65_ROLLBACK_SHA),
        (
            "docs/notebook-infrastructure.md",
            "retained Atlas Torch 2.13 runtime",
            "Atlas runtime ownership remains Issue #65",
        ),
    ),
)
def test_issue65_current_docs_reject_independent_mutations(path, old, new) -> None:
    documents = _issue65_current_documents()
    assert old in documents[path], path
    documents[path] = documents[path].replace(old, new, 1)

    with pytest.raises(AssertionError):
        _assert_issue65_current_docs(documents)


def test_issue71_current_docs_publish_notebook_freshness_hash_contract() -> None:
    documents = {
        path: (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "CONTRIBUTING.md",
            "CHANGELOG.md",
            "docs/conventions.md",
            "docs/notebook-infrastructure.md",
            "docs/maintenance/overnight-2026-07-04.md",
            "docs/superpowers/specs/2026-08-22-issue-71-notebook-freshness-hashes-design.md",
            "docs/superpowers/plans/2026-08-22-issue-71-notebook-freshness-hashes-implementation-plan.md",
        )
    }

    contract = " ".join(documents["docs/conventions.md"].split())
    for expected in (
        "metadata.source_hash",
        "bare lowercase 64-character SHA-256 digest",
        "exact UTF-8 logical code-cell source",
        "concatenates its string elements with no separator",
        "Whitespace, comments, magics, line endings, and syntax are not normalized",
        "Output bytes are never hashed",
        "E8",
        "active_task_dirs",
        "missing, malformed, stale, or orphan",
        "notebooks/archive/",
        "outputless code cells",
        "markdown or raw cell is `E8.source_hash_orphan`",
        "raw nbformat-4 schema",
        "may inherit prior source hashes",
        "--clear",
        "explicit notebook paths",
        "rejects `--all-active`",
    ):
        assert expected in contract, expected

    execution = " ".join(documents["docs/notebook-infrastructure.md"].split())
    for expected in (
        "run-tier-a", "smoke-tier-a", "smoke-tier-b", "smoke-tier-c",
        "scripts/stamp_notebook_source_hashes.py", "only after success",
        "On a nonzero Papermill exit", "atomically removes every cell's `metadata.source_hash`",
        "artifact exists", "success stamper is not invoked",
        "stamper failure fails the Make target",
        "cleanup failure cannot make the target succeed",
        "uncatchable host or process kill",
    ):
        assert expected in execution, expected

    for path in (
        "docs/superpowers/specs/2026-08-22-issue-71-notebook-freshness-hashes-design.md",
        "docs/superpowers/plans/2026-08-22-issue-71-notebook-freshness-hashes-implementation-plan.md",
    ):
        text = " ".join(documents[path].split())
        for expected in (
            "Papermill inputs may carry prior hashes",
            "failure cleanup atomically removes them from every cell",
            "raw nbformat-4 schema",
            "uncatchable host or process kill",
        ):
            assert expected in text, f"{path}: {expected}"

    contributor = documents["CONTRIBUTING.md"]
    assert "python scripts/stamp_notebook_source_hashes.py --all-active" in contributor
    assert "maintenance/repair command" in contributor
    assert "not a substitute for executing changed source" in contributor

    readme = " ".join(documents["README.md"].split())
    assert "retained outputs correspond to the current source" in readme
    assert "not that nondeterministic output rendering is byte-reproducible" in readme

    ledger = documents["docs/maintenance/overnight-2026-07-04.md"].lower()
    for expected in (
        "om-010", "resolved", "issue #71", "29 active notebooks",
        "189 retained-output cells", "189 current markers", "byte-stable idempotence",
        "no expensive rerun was performed solely for metadata",
    ):
        assert expected in ledger, expected

    changelog = documents["CHANGELOG.md"].split("## [0.1.0]", 1)[0]
    assert "Issue #71" in changelog
    assert "every cell's `metadata.source_hash`" in changelog


def test_project_opening_allows_concise_semantically_equivalent_summaries(tmp_path):
    _write_project_opening(
        tmp_path,
        readme_summary="ml-eng-lab has 21 active task folders and 29 notebooks for machine learning.",
        landing_summary="Explore ml-eng-lab notebooks. The collection includes 21 active task directories and 29 active notebooks.",
    )
    assert check_project_opening(tmp_path) == []
