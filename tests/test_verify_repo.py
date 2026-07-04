"""Tests for scripts/verify_repo.py — the four-check oracle."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "verify_repo.py"
ACTIVE_FIXTURE_DIR = "notebooks/image_classification-mnist-ffnn-numpy"
TEST_SUBPROCESS_TIMEOUT = 30


def run_verify(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )


def _temp_repo(tmp_path: Path) -> Path:
    (tmp_path / ACTIVE_FIXTURE_DIR).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    return tmp_path


def test_help_lists_all_checks():
    r = run_verify("--help")
    assert r.returncode == 0
    for ch in ("structure", "execution", "docs", "comments", "all"):
        assert ch in r.stdout


def test_help_does_not_require_adjacent_config(tmp_path):
    script_copy = tmp_path / "scripts" / "verify_repo.py"
    script_copy.parent.mkdir()
    script_copy.write_text(SCRIPT.read_text())
    r = subprocess.run(
        [sys.executable, str(script_copy), "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    assert r.returncode == 0, r.stderr
    assert "--check" in r.stdout


def test_unknown_check_errors():
    r = run_verify("--check", "garbage")
    assert r.returncode != 0


def test_missing_check_errors_without_phase_b_out():
    r = run_verify()
    assert r.returncode != 0
    assert "--check is required unless --phase-b-out is used" in r.stderr


def test_emits_valid_json_schema(tmp_path):
    out = tmp_path / "findings.json"
    r = run_verify("--check", "structure", "--out", str(out), "--fast")
    assert out.exists(), f"no output file; stderr={r.stderr}"
    data = json.loads(out.read_text())
    assert isinstance(data, dict)
    assert "schema_version" in data
    assert data["schema_version"] == 1
    assert "findings" in data
    assert isinstance(data["findings"], list)
    assert "summary" in data
    assert "checks_run" in data["summary"]
    assert "structure" in data["summary"]["checks_run"]


def test_finding_shape():
    """Every finding must have id, check, severity, location, message."""
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    for f in data.get("findings", []):
        assert {"id", "check", "severity", "location", "message"} <= set(f.keys())
        assert f["severity"] in ("error", "warning")


def test_structure_s1_notebooks_parse():
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s1 = [f for f in data["findings"] if f["id"].startswith("S1")]
    assert data["summary"]["by_check"]["structure"] == len(s1) + sum(
        1 for f in data["findings"] if not f["id"].startswith("S1") and f["check"] == "structure"
    )


def test_structure_s1_flags_missing_notebook_cell_id(tmp_path):
    """nbformat currently auto-fills missing cell ids, so check raw JSON too."""
    repo = _temp_repo(tmp_path)
    name = "missing-cell-id.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    fake.write_text(json.dumps({
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "x = 1\n",
        }],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }))
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S1.cell_id" and name in f["location"]
    ]
    assert hits, f"expected S1.cell_id for {name}; got {data.get('findings')}"


def test_structure_s5_no_common_imports():
    """No `from common.` import anywhere in active task notebooks or scripts."""
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s5 = [f for f in data["findings"] if f["id"].startswith("S5")]
    assert s5 == [], f"S5 found stray common.* imports: {s5}"


def test_structure_s2_checks_every_module_in_multi_import(tmp_path):
    """A valid first import must not hide a missing second import on the same line."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "multi-import-missing.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("import json, definitely_missing_module_for_s2\n")
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_module_for_s2" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import for second import; got {data.get('findings')}"


def test_structure_s2_checks_multi_import_after_notebook_magic(tmp_path):
    """Notebook magics must not push S2 back to a first-module-only regex fallback."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "magic-multi-import-missing.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "%matplotlib inline\nimport json, definitely_missing_module_after_magic\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_module_after_magic" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import after notebook magic; got {data.get('findings')}"


def test_structure_s2_ignores_non_python_cell_magic_body(tmp_path):
    """Shell cell magics must not make S2 scan shell text as Python imports."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "bash-cell-magic-import-text.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "%%bash\n"
            "echo import definitely_missing_module_inside_shell_magic\n"
            "import definitely_missing_module_inside_shell_magic\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if name in f["location"]
        and "definitely_missing_module_inside_shell_magic" in f["message"]
    ]
    assert hits == [], f"shell cell magic body should not be scanned as Python; got {hits}"


def test_structure_s2_flags_notebook_relative_imports(tmp_path):
    """Relative imports in notebooks are runtime-broken and should be explicit findings."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "relative-import.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("from . import definitely_missing_relative_helper\n")
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.relative_import"
        and name in f["location"]
        and "definitely_missing_relative_helper" in f["message"]
    ]
    assert hits, f"expected S2.relative_import for notebook relative import; got {data.get('findings')}"


def test_structure_s7_no_pycache_tracked():
    """No __pycache__, .ipynb_checkpoints, .DS_Store should be tracked."""
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s7 = [f for f in data["findings"] if f["id"].startswith("S7")]
    assert s7 == [], f"S7 found tracked bloat: {s7}"


def test_structure_s6_allows_committed_superpowers_specs_and_plans(tmp_path):
    """Committed Superpowers spec/plan docs are intentional planning records."""
    repo = _temp_repo(tmp_path)
    (repo / ".gitignore").write_text("docs/superpowers/\n", encoding="utf-8")
    spec = repo / "docs" / "superpowers" / "specs" / "design.md"
    plan = repo / "docs" / "superpowers" / "plans" / "plan.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    plan.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text("# Design\n", encoding="utf-8")
    plan.write_text("# Plan\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", str(spec), str(plan)],
        cwd=repo,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}

    forbidden = {str(spec.relative_to(repo)), str(plan.relative_to(repo))}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S6.tracked_bloat" and f["location"] in forbidden
    ]
    assert not hits, f"intentional planning docs were flagged as bloat: {hits}"


def test_structure_s6_flags_other_tracked_superpowers_files(tmp_path):
    """Only committed spec/plan records are exempt from docs/superpowers bloat."""
    repo = _temp_repo(tmp_path)
    (repo / ".gitignore").write_text("docs/superpowers/\n", encoding="utf-8")
    scratch = repo / "docs" / "superpowers" / "scratch.md"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text("# Scratch\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", str(scratch)],
        cwd=repo,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}

    hits = [
        f for f in data["findings"]
        if f["id"] == "S6.tracked_bloat" and f["location"] == str(scratch.relative_to(repo))
    ]
    assert hits, f"expected non-plan docs/superpowers file to be flagged; got {data.get('findings')}"


def test_structure_s8_script_shebang_executable_parity():
    """Direct CLI scripts should keep shebang and executable bit in sync."""
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s8 = [f for f in data["findings"] if f["id"].startswith("S8")]
    assert s8 == [], f"S8 found script mode drift: {s8}"


def test_structure_s3_flags_missing_markdown_fragment(tmp_path):
    """Internal Markdown links must validate `#fragment` anchors, not just files."""
    repo = _temp_repo(tmp_path)
    name = "bad_anchor.md"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    fake.write_text("# 1. Existing Heading\n\n[bad](#2-missing-heading)\n")
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S3.broken_anchor" and name in f["location"]
    ]
    assert hits, f"expected S3.broken_anchor for {name}; got {data.get('findings')}"


def test_structure_s3_ignores_markdown_link_examples_in_code_spans(tmp_path):
    """Historical examples like ``[§4](#old-heading)`` should not be live links."""
    repo = _temp_repo(tmp_path)
    name = "code_span_anchor.md"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    fake.write_text("# 1. Existing Heading\n\nLiteral example: `[bad](#missing-heading)`.\n")
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"].startswith("S3.") and name in f["location"]
    ]
    assert not hits, f"code-span Markdown link example was treated as live: {hits}"


def test_structure_s3_ignores_markdown_link_examples_in_fenced_code(tmp_path):
    """Fenced snippets often contain example Markdown links that are not live."""
    repo = _temp_repo(tmp_path)
    name = "fenced_anchor.md"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    fake.write_text(
        "# 1. Existing Heading\n\n"
        "```md\n"
        "[bad](#missing-heading)\n"
        "```\n"
    )
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"].startswith("S3.") and name in f["location"]
    ]
    assert not hits, f"fenced Markdown link example was treated as live: {hits}"


def test_structure_s3_checks_notebook_markdown_links(tmp_path):
    """Notebook markdown links should be covered by the same S3 hygiene."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "bad_link.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell("[bad](missing-local-doc.md)\n")]
    nbformat.write(nb, str(fake))
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S3.broken_link" and name in f["location"]
    ]
    assert hits, f"expected S3.broken_link for notebook markdown; got {data.get('findings')}"


def test_structure_s3_checks_nested_docs_markdown_links(tmp_path):
    """Nested docs should be covered by the same S3 link hygiene as shallow docs."""
    repo = _temp_repo(tmp_path)
    nested = repo / "docs" / "maintenance" / "history.md"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("# History\n\n[missing](missing-local-doc.md)\n", encoding="utf-8")

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S3.broken_link" and "docs/maintenance/history.md" in f["location"]
    ]
    assert hits, f"expected S3.broken_link for nested docs markdown; got {data.get('findings')}"


def test_structure_s3_ignores_notebook_markdown_code_span_links(tmp_path):
    """Notebook prose can show Markdown link syntax as a literal example."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "code_span_link.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell("Literal: `[bad](missing-local-doc.md)`\n")]
    nbformat.write(nb, str(fake))
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"].startswith("S3.") and name in f["location"]
    ]
    assert not hits, f"notebook code-span Markdown link was treated as live: {hits}"


def test_docs_d1_known_notebooks_have_required_sections():
    """All tracked notebooks must have their REQUIRED_SECTIONS H1s present.

    Regression guard: if a future edit deletes / reorders an H1 in a tracked
    notebook listed in REQUIRED_SECTIONS, D1.missing_sections fires here.
    Also catches D1.missing_notebook if a listed file gets renamed without
    updating the config.
    """
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    d1 = [f for f in data["findings"] if f["id"].startswith("D1.")]
    assert d1 == [], f"D1 reported issues: {d1}"


def test_docs_d1_unconfigured_active_notebook_is_error(tmp_path):
    """A new active notebook must not bypass docs/E7 checks by being omitted from YAML."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "unconfigured.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell("# 1. Overview\n")]
    nbformat.write(nb, str(fake))
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "D1.unconfigured_notebook" and name in f["location"]
    ]
    assert hits, f"expected D1.unconfigured_notebook for {name}; got {data.get('findings')}"
    assert all(f["severity"] == "error" for f in hits)


def test_docs_d8_terminology_consistency_known_canonicals():
    """The check should mention canonical spellings in its allow-list logic."""
    SCRIPT_TEXT = SCRIPT.read_text()
    for token in ("genai-vanilla", "JupyterHub", "NumPy", "PyTorch"):
        assert token in SCRIPT_TEXT, f"D8 missing canonical {token!r}"


def test_docs_d9_current_numbered_docs_are_consistent():
    """Active numbered docs should use dotted numeric headings consistently."""
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    d9 = [f for f in data["findings"] if f["id"] == "D9.numbered_heading"]
    assert d9 == [], f"D9 reported numbered-heading issues: {d9}"


def test_docs_d9_flags_malformed_numbered_headings(tmp_path):
    """H3 headings need a dotted number plus trailing period, e.g. `3.1.`."""
    repo = _temp_repo(tmp_path)
    readme = repo / ACTIVE_FIXTURE_DIR / "README.md"
    readme.write_text(
        "# Fixture\n\n"
        "## 1. Task summary\n\n"
        "## 2. Why this exists\n\n"
        "## 3. What's in the notebook\n\n"
        "### 3.1 Phase without dotted-number terminator\n\n"
        "## 4. How to run\n\n"
        "## 5. Dependencies\n\n"
        "## 6. Known issues\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "D9.numbered_heading" and "README.md:9" in f["location"]
    ]
    assert hits, f"expected D9.numbered_heading for malformed H3; got {data.get('findings')}"


def test_docs_d10_dependency_ledger_counts_match_current_doc():
    """Package counts and advisory feed-record counts should reconcile."""
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    d10 = [f for f in data["findings"] if f["id"] == "D10.dependency_ledger_count"]
    assert d10 == [], f"D10 reported dependency-ledger issues: {d10}"


def test_docs_d10_flags_dependency_ledger_count_drift(tmp_path):
    """The dependency ledger should not collapse duplicated advisory feed records."""
    repo = _temp_repo(tmp_path)
    docs = repo / "docs"
    docs.mkdir()
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 1. Audit Snapshot\n\n"
        "Result: 2 known vulnerabilities across one resolved package:\n\n"
        "| Package | Manifest Constraint | Audited Resolved Version | Finding Count | Current Disposition |\n"
        "| --- | --- | ---: | ---: | --- |\n"
        "| `torch` | `torch==2.4.1` | `2.4.1` | 2 | Accepted temporarily. |\n\n"
        "| Package | Advisory ID | Feed Records | Fix Versions |\n"
        "| --- | --- | ---: | --- |\n"
        "| `torch` | `PYSEC-2025-41` | 1 | `2.6.0` |\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "D10.dependency_ledger_count"]
    assert hits, f"expected D10.dependency_ledger_count; got {data.get('findings')}"


def test_docs_d11_current_layout_guidance_is_not_stale():
    """Contributor-facing docs should point new tasks at notebooks/<task>/."""
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    d11 = [f for f in data["findings"] if f["id"] == "D11.stale_notebook_layout"]
    assert d11 == [], f"D11 reported stale layout guidance: {d11}"


def test_docs_d11_flags_old_flat_layout_guidance(tmp_path):
    """The verifier should catch the pre-migration top-level task convention."""
    repo = _temp_repo(tmp_path)
    (repo / "README.md").write_text(
        "# Fixture\n\n"
        "## 1. Overview\n\n"
        "Each top-level folder is a self-contained task.\n\n"
        "See archive/README.md for preserved work.\n",
        encoding="utf-8",
    )
    (repo / "CONTRIBUTING.md").write_text(
        "# Contributing\n\n"
        "Use https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/<folder>/<notebook>.ipynb.\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "D11.stale_notebook_layout"]
    assert len(hits) >= 3, f"expected stale-layout findings; got {data.get('findings')}"


def test_comments_phase_a_flags_obvious_state_the_what(tmp_path):
    """Synthetic .py file with a known bad comment should produce a finding.

    The synthetic file lives in an isolated repo root so this test never mutates
    the real checkout.
    """
    repo = _temp_repo(tmp_path)
    name = "state_the_what.py"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    fake.write_text("# import numpy as np\nimport numpy as np\n")
    r = run_verify("--repo-root", str(repo), "--check", "comments", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["check"] == "comments" and name in f["location"]
    ]
    assert hits, f"expected at least one state-the-what flag; got summary={data.get('summary')}"


def test_comments_phase_a_skips_explanatory_comments(tmp_path):
    """A WHY-style comment should NOT be flagged."""
    repo = _temp_repo(tmp_path)
    name = "why.py"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    fake.write_text(
        "# Xavier init keeps variance stable across depths; default torch init blows up here.\n"
        "weight = xavier_init(shape)\n"
    )
    r = run_verify("--repo-root", str(repo), "--check", "comments", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["check"] == "comments" and name in f["location"]
    ]
    assert not hits, f"WHY-style comment falsely flagged: {hits}"


def test_comments_phase_a_skips_parameters_tagged_cells(tmp_path):
    """C.state_the_what must skip papermill `parameters`-tagged cells.

    Their boilerplate (per scripts/inject_smoke_test_cell.py) carries lines
    like `# Set via: papermill -p SMOKE_TEST 1 in.ipynb out.ipynb` that
    document the papermill invocation contract — not state-the-what hits
    on the next code line. Same self-exclusion principle as the
    verify_repo.py-as-scanner skip.
    """
    import nbformat
    repo = _temp_repo(tmp_path)
    name = "params.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell(
        # Comment matches the `^# (initialize|init|set|assign)` rule; without
        # the parameters tag the C check would flag this. The tag must
        # suppress that.
        "# Set via: papermill -p X 1 in.ipynb out.ipynb\nX = 0\n"
    )
    cell.metadata["tags"] = ["parameters"]
    nb.cells = [cell]
    nbformat.write(nb, str(fake))
    r = run_verify("--repo-root", str(repo), "--check", "comments", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["check"] == "comments" and name in f["location"]
    ]
    assert not hits, f"parameters-tagged cell falsely flagged: {hits}"


def test_execution_fast_mode_skips_e1_e2_e3():
    """In --fast mode, slow targets (E1-E3) must not be invoked."""
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    assert "execution" in data["summary"]["checks_run"]
    forbidden_ids = ("E1.tier_a_failed", "E2.tier_b_smoke_failed", "E3.tier_c_smoke_failed")
    for f in data.get("findings", []):
        assert f["id"] not in forbidden_ids, f"slow check ran in --fast mode: {f}"


def test_execution_e5_baseline_missing_warns_not_errors():
    """Before pre-cleanup-baseline tag exists, E5 should warn (not error)."""
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    e5 = [f for f in data["findings"] if f["id"] == "E5.no_baseline"]
    if e5:
        for f in e5:
            assert f["severity"] == "warning", f"E5.no_baseline must be warning, got {f}"


def test_runtime_available_requires_pyg_extension_stack(monkeypatch):
    """Full notebook execution needs the PyG binary extension stack, not just torch_geometric."""
    verify_repo = _load_verify_module()
    present = {"torch", "torch_geometric"}

    def fake_find_spec(name):
        return object() if name in present else None

    monkeypatch.setattr(verify_repo.importlib.util, "find_spec", fake_find_spec)

    assert verify_repo._runtime_available() is False


def test_required_sections_loaded_from_yaml_config():
    """The verify_repo_config.yaml should be the source of truth for the
    REQUIRED_SECTIONS table."""
    import importlib

    scripts_dir = str(REPO / "scripts")
    sys_path_snapshot = list(sys.path)
    sys.path.insert(0, scripts_dir)
    try:
        if "verify_repo" in sys.modules:
            importlib.reload(sys.modules["verify_repo"])
        import verify_repo
        assert isinstance(verify_repo.REQUIRED_SECTIONS, dict)
        for d in verify_repo.ACTIVE_TASK_DIRS:
            assert any(k.startswith(f"notebooks/{d}/") for k in verify_repo.REQUIRED_SECTIONS), (
                f"no entries for {d}"
            )
        phase1 = verify_repo.REQUIRED_SECTIONS.get(
            "notebooks/node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb"
        )
        assert phase1 is not None
        assert "4. Model" not in phase1

        # YAML is the source of truth — compare TIER_A_NOTEBOOKS to what the
        # config file actually declares, not a hardcoded literal.
        import yaml  # PyYAML is a verify_repo runtime dep, so import is safe here
        config_path = REPO / "scripts" / "verify_repo_config.yaml"
        config = yaml.safe_load(config_path.read_text()) or {}
        expected_tier_a = tuple(config.get("tier_a_notebooks", ()))
        assert tuple(verify_repo.TIER_A_NOTEBOOKS) == expected_tier_a
    finally:
        sys.path[:] = sys_path_snapshot


def test_phase_b_export_runs_and_produces_json(tmp_path):
    """--phase-b-out exports candidate comments as JSON; doesn't run full check."""
    out = tmp_path / "candidates.json"
    r = run_verify("--check", "comments", "--phase-b-out", str(out))
    assert r.returncode == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert "schema_version" in data
    assert "candidate_count" in data
    assert "candidates" in data
    assert isinstance(data["candidates"], list)
    for cand in data["candidates"]:
        assert {"location", "comment", "snippet"} <= set(cand.keys())


def test_phase_b_export_does_not_require_check(tmp_path):
    out = tmp_path / "candidates.json"
    r = run_verify("--phase-b-out", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()


def test_e7_papermill_params_tag_check():
    """Notebooks meant to be papermilled with -p should declare a parameters tag."""
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    # E7 is a warning, never an error.
    e7 = [f for f in data["findings"] if f["id"] == "E7.no_papermill_params_tag"]
    for f in e7:
        assert f["severity"] == "warning"


def test_e13_current_active_notebooks_have_no_stale_repo_paths():
    """Active notebook metadata and outputs should not retain pre-rename paths."""
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    e13 = [f for f in data["findings"] if f["id"] == "E13.stale_active_notebook_path"]
    assert e13 == [], f"E13 reported stale active-notebook paths: {e13}"


def test_e13_flags_stale_paths_in_active_notebooks(tmp_path, monkeypatch):
    """The stale-path guard applies to active notebooks, not the archive."""
    verify_repo = _load_verify_module()
    repo = _temp_repo(tmp_path)
    active_dir = repo / "notebooks" / "active-task"
    archive_dir = repo / "notebooks" / "archive" / "old-task"
    active_dir.mkdir(parents=True)
    archive_dir.mkdir(parents=True)
    (active_dir / "notebook.ipynb").write_text(
        '{"outputs":[{"text":["/home/jovyan/work/ml/nnx/src/file.py"]}]}',
        encoding="utf-8",
    )
    (archive_dir / "notebook.ipynb").write_text(
        '{"outputs":[{"text":["/home/jovyan/work/ml/legacy.py"]}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ("active-task",))

    result = verify_repo.check_execution(repo, fast=True)

    hits = [f for f in result.findings if f.id == "E13.stale_active_notebook_path"]
    assert len(hits) == 1
    assert hits[0].location.startswith("notebooks/active-task/notebook.ipynb")


def test_e13_flags_removed_nnx_source_tree_and_host_python_paths(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    repo = _temp_repo(tmp_path)
    active_dir = repo / "notebooks" / "active-task"
    active_dir.mkdir(parents=True)
    (active_dir / "notebook.ipynb").write_text(
        "\n".join([
            '{"outputs":[',
            '  {"text":["/home/jovyan/work/ml-eng-lab/nnx/src/nnx/nn/params/file.py"]},',
            '  {"text":["/Users/alice/.pyenv/versions/3.11/site-packages/pkg/file.py"]}',
            ']}',
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ("active-task",))

    result = verify_repo.check_execution(repo, fast=True)

    hits = [f for f in result.findings if f.id == "E13.stale_active_notebook_path"]
    assert [f.message for f in hits] == [
        "stale active-notebook path artifact: removed in-repo nnx source tree",
        "stale active-notebook path artifact: host-local Python environment path",
    ]


def test_e14_flags_tmp_papermill_output_path(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    import nbformat

    repo = _temp_repo(tmp_path)
    task = "tmp-papermill-task"
    active_dir = repo / "notebooks" / task
    active_dir.mkdir(parents=True)
    nb_path = active_dir / "notebook.ipynb"

    nb = nbformat.v4.new_notebook()
    nb.metadata["papermill"] = {
        "input_path": "notebook.ipynb",
        "output_path": "/tmp/smoke-output.ipynb",
    }
    cell = nbformat.v4.new_code_cell("# parser-friendly comment\nSMOKE_TEST = 0\n")
    cell.metadata["tags"] = ["parameters"]
    nb.cells = [cell]
    nbformat.write(nb, str(nb_path))

    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", (task,))
    monkeypatch.setattr(verify_repo, "REQUIRED_SECTIONS", {str(nb_path.relative_to(repo)): ()})
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())

    result = verify_repo.check_execution(repo, fast=True)

    hits = [f for f in result.findings if f.id == "E14.tmp_papermill_output_path"]
    assert hits
    assert hits[0].location == str(nb_path.relative_to(repo))


def test_e14_flags_source_notebook_papermill_metadata(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    import nbformat

    repo = _temp_repo(tmp_path)
    task = "source-papermill-task"
    active_dir = repo / "notebooks" / task
    active_dir.mkdir(parents=True)
    nb_path = active_dir / "notebook.ipynb"

    nb = nbformat.v4.new_notebook()
    nb.metadata["papermill"] = {
        "input_path": "notebook.ipynb",
        "output_path": str(nb_path.relative_to(repo)),
    }
    cell = nbformat.v4.new_code_cell("# parser-friendly comment\nSMOKE_TEST = 0\n")
    cell.metadata["tags"] = ["parameters"]
    nb.cells = [cell]
    nbformat.write(nb, str(nb_path))

    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", (task,))
    monkeypatch.setattr(verify_repo, "REQUIRED_SECTIONS", {str(nb_path.relative_to(repo)): ()})
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())

    result = verify_repo.check_execution(repo, fast=True)

    hits = [f for f in result.findings if f.id == "E14.source_papermill_metadata"]
    assert hits
    assert hits[0].location == str(nb_path.relative_to(repo))


def _load_verify_module():
    import importlib.util
    if "verify_repo" in sys.modules:
        return sys.modules["verify_repo"]
    spec = importlib.util.spec_from_file_location("verify_repo", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass field resolution needs the module findable in sys.modules,
    # otherwise field-class lookup raises AttributeError on a NoneType.
    sys.modules["verify_repo"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_iter_notebooks_reads_active_tasks_under_notebooks(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    active = tmp_path / "notebooks" / "task-a"
    archive = tmp_path / "notebooks" / "archive" / "old-task"
    old_root = tmp_path / "task-a"
    active.mkdir(parents=True)
    archive.mkdir(parents=True)
    old_root.mkdir()

    (active / "notebook.ipynb").write_text("{}", encoding="utf-8")
    (archive / "notebook.ipynb").write_text("{}", encoding="utf-8")
    (old_root / "notebook.ipynb").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ("task-a",))

    found = [str(p.relative_to(tmp_path)) for p in verify_repo._iter_notebooks(tmp_path)]

    assert found == ["notebooks/task-a/notebook.ipynb"]


def test_baseline_notebook_rel_removes_notebooks_prefix():
    verify_repo = _load_verify_module()
    baseline_rel = "/".join([
        "node_classification-reddit-gnn-pyg",
        "phase3-main-model-training-and-eval-notebook.ipynb",
    ])

    assert (
        verify_repo._baseline_notebook_rel(
            "notebooks/node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook.ipynb"
        )
        == baseline_rel
    )
    assert verify_repo._baseline_notebook_rel("legacy/notebook.ipynb") == "legacy/notebook.ipynb"


def test_assignment_names_ignore_comments_and_strings():
    verify_repo = _load_verify_module()
    names = verify_repo._assignment_names(
        "# COMMENT_ONLY = 1\n"
        "example = 'STRING_ONLY = 1'\n"
        "SMOKE_TEST = 0\n"
        "SMOKE_TEST_EPOCHS: int = 1\n"
        "SMOKE_TEST_SUBSET += 1\n"
        "LEFT, RIGHT = 1, 2\n"
    )

    assert {"SMOKE_TEST", "SMOKE_TEST_EPOCHS", "SMOKE_TEST_SUBSET", "LEFT", "RIGHT"} <= names
    assert "COMMENT_ONLY" not in names
    assert "STRING_ONLY" not in names


def test_e10_flags_parameters_tag_without_smoke_test_assignment(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    import nbformat

    rel = Path("task") / "missing-smoke.ipynb"
    nb_path = tmp_path / rel
    nb_path.parent.mkdir()
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell("OTHER_PARAMETER = 1\n")
    cell.metadata["tags"] = ["parameters"]
    nb.cells = [cell]
    nbformat.write(nb, str(nb_path))

    monkeypatch.setattr(verify_repo, "REQUIRED_SECTIONS", {str(rel): ("1. Any",)})
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())
    result = verify_repo.check_execution(tmp_path, fast=True)

    hits = [f for f in result.findings if f.id == "E10.missing_smoke_test_parameter"]
    assert len(hits) == 1
    assert hits[0].severity == "error"
    assert hits[0].location == str(rel)


def test_e10_smoke_test_parameter_check_clean_current_repo():
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "E10.missing_smoke_test_parameter"]
    assert hits == []


def test_makefile_variable_items_parse_continuation_list(tmp_path):
    verify_repo = _load_verify_module()
    makefile = tmp_path / "Makefile"
    makefile.write_text(
        "OTHER := ignored\n"
        "TIER_A := \\\n"
        "    first/notebook.ipynb \\\n"
        "    second/notebook.ipynb\n"
        "TIER_B := third/notebook.ipynb\n"
    )
    assert verify_repo._makefile_variable_items(tmp_path, "TIER_A") == (
        "first/notebook.ipynb",
        "second/notebook.ipynb",
    )


def test_e11_tier_a_config_matches_makefile():
    verify_repo = _load_verify_module()
    assert verify_repo._makefile_variable_items(REPO, "TIER_A") == tuple(
        verify_repo.TIER_A_NOTEBOOKS
    )
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"].startswith("E11.")]
    assert hits == []


def test_e11_flags_missing_makefile_tier_a(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ("task/notebook.ipynb",))
    result = verify_repo.check_execution(tmp_path, fast=True)
    hits = [f for f in result.findings if f.id == "E11.tier_a_makefile_missing"]
    assert len(hits) == 1
    assert hits[0].severity == "error"


def test_ci_tier_a_artifact_paths_parse_workflow(tmp_path):
    verify_repo = _load_verify_module()
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n"
        "  tier-a-papermill:\n"
        "    steps:\n"
        "      - name: Upload refreshed notebook outputs as artifact\n"
        "        with:\n"
        "          path: |\n"
        "            first/notebook.ipynb\n"
        "            second/notebook.ipynb\n"
    )
    assert verify_repo._ci_tier_a_artifact_paths(tmp_path) == (
        "first/notebook.ipynb",
        "second/notebook.ipynb",
    )


def test_e12_tier_a_artifact_paths_match_config():
    verify_repo = _load_verify_module()
    assert verify_repo._ci_tier_a_artifact_paths(REPO) == tuple(
        verify_repo.TIER_A_NOTEBOOKS
    )
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"].startswith("E12.")]
    assert hits == []


def test_run_helper_timeout_returns_rc_124():
    """_run must catch subprocess.TimeoutExpired and surface rc=124 + a
    diagnostic stderr suffix, so a hung make target produces a clean Finding
    instead of an uncaught traceback."""
    verify_repo = _load_verify_module()
    rc, stdout, stderr = verify_repo._run(["sleep", "5"], REPO, timeout=1)
    assert rc == 124, f"expected rc=124 on timeout, got {rc} (stdout={stdout!r}, stderr={stderr!r})"
    assert "timed out after 1s" in stderr


def test_run_helper_timeout_normalizes_byte_streams(monkeypatch):
    """TimeoutExpired can carry byte stdout/stderr even when subprocess.run used text=True."""
    verify_repo = _load_verify_module()

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs.get("timeout"),
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(verify_repo.subprocess, "run", raise_timeout)
    rc, stdout, stderr = verify_repo._run(["fake"], REPO, timeout=1)
    assert rc == 124
    assert stdout == "partial stdout"
    assert "partial stderr" in stderr
    assert "timed out after 1s" in stderr


def test_run_helper_supplies_default_timeout(monkeypatch):
    """Callers should not have to remember a timeout for short external commands."""
    verify_repo = _load_verify_module()
    seen: dict[str, int | None] = {}

    def fake_run(cmd, cwd, capture_output, text, timeout):
        del cwd, capture_output, text
        seen["timeout"] = timeout
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(verify_repo.subprocess, "run", fake_run)
    rc, _, _ = verify_repo._run(["fake"], REPO)

    assert rc == 0
    assert seen["timeout"] == verify_repo.DEFAULT_SUBPROCESS_TIMEOUT


def test_tier_c_baseline_sources_ignore_parameter_cells():
    verify_repo = _load_verify_module()
    import nbformat

    baseline = nbformat.v4.new_notebook()
    baseline.cells = [
        nbformat.v4.new_code_cell("SMOKE_TEST = 0  # old parser-hostile comment\n"),
        nbformat.v4.new_code_cell("model.train()\n"),
    ]
    baseline.cells[0].metadata["tags"] = ["parameters"]

    head = nbformat.v4.new_notebook()
    head.cells = [
        nbformat.v4.new_code_cell("# parser-friendly comment\nSMOKE_TEST = 0\n"),
        nbformat.v4.new_code_cell("model.train()\n"),
    ]
    head.cells[0].metadata["tags"] = ["parameters"]

    assert verify_repo._code_cell_sources_for_baseline(head) == verify_repo._code_cell_sources_for_baseline(baseline)

    head.cells[1].source = "model.train(n_epochs=1)\n"
    assert verify_repo._code_cell_sources_for_baseline(head) != verify_repo._code_cell_sources_for_baseline(baseline)


def test_parameter_trailing_comment_check_flags_papermill_uninspectable_assignment():
    verify_repo = _load_verify_module()
    import nbformat

    nb = nbformat.v4.new_notebook()
    bad = nbformat.v4.new_code_cell("SMOKE_TEST = 0  # 1 = smoke mode\n")
    bad.metadata["tags"] = ["parameters"]
    good = nbformat.v4.new_code_cell("# 1 = smoke mode\nSMOKE_TEST = 0\n")
    good.metadata["tags"] = ["parameters"]

    nb.cells = [bad]
    findings = verify_repo._parameter_trailing_comment_findings(nb, "fake.ipynb")
    assert [f.id for f in findings] == ["E9.parameter_trailing_comment"]

    nb.cells = [good]
    assert verify_repo._parameter_trailing_comment_findings(nb, "fake.ipynb") == []


def test_s7_forbidden_toplevel_detects_resurrected_common(tmp_path):
    """S7.forbidden_toplevel fires if common/ ever comes back."""
    repo = _temp_repo(tmp_path)
    fake_dir = repo / "common"
    fake_dir.mkdir()
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s7 = [
        f for f in data["findings"]
        if f["id"] == "S7.forbidden_toplevel" and "common" in f["location"]
    ]
    assert s7, "expected S7.forbidden_toplevel to flag resurrected common/"
    for f in s7:
        assert f["severity"] == "error"
