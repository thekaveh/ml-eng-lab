"""Tests for scripts/verify_repo.py — the four-check oracle."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "verify_repo.py"
ACTIVE_FIXTURE_DIR = "image_classification-mnist-ffnn-numpy"


def run_verify(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def _temp_repo(tmp_path: Path) -> Path:
    (tmp_path / ACTIVE_FIXTURE_DIR).mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, text=True, check=True)
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


def test_structure_s1_notebooks_parse(tmp_path):
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


def test_structure_s7_no_pycache_tracked():
    """No __pycache__, .ipynb_checkpoints, .DS_Store should be tracked."""
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s7 = [f for f in data["findings"] if f["id"].startswith("S7")]
    assert s7 == [], f"S7 found tracked bloat: {s7}"


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
            assert any(k.startswith(d) for k in verify_repo.REQUIRED_SECTIONS), (
                f"no entries for {d}"
            )
        phase1 = verify_repo.REQUIRED_SECTIONS.get(
            "node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb"
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
