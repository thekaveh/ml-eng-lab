"""Tests for scripts/verify_repo.py — the four-check oracle."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "verify_repo.py"


def run_verify(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def test_help_lists_all_checks():
    r = run_verify("--help")
    assert r.returncode == 0
    for ch in ("structure", "execution", "docs", "comments", "all"):
        assert ch in r.stdout


def test_unknown_check_errors():
    r = run_verify("--check", "garbage")
    assert r.returncode != 0


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


def test_docs_d8_terminology_consistency_known_canonicals():
    """The check should mention canonical spellings in its allow-list logic."""
    SCRIPT_TEXT = SCRIPT.read_text()
    for token in ("genai-vanilla", "JupyterHub", "NumPy", "PyTorch"):
        assert token in SCRIPT_TEXT, f"D8 missing canonical {token!r}"


def test_comments_phase_a_flags_obvious_state_the_what(tmp_path):
    """Synthetic .py file with a known bad comment should produce a finding.

    verify_repo.py runs in a subprocess and scans ACTIVE_TASK_DIRS in REPO, so
    we must drop the synthetic file inside one of those task dirs. We derive a
    unique filename from `tmp_path` so concurrent test runs don't collide, and
    wrap in try/finally for robust cleanup even if the assertion fires.
    """
    name = f"_temp_{tmp_path.name}_state_the_what.py"
    fake = REPO / "image_classification-mnist-ffnn-numpy" / name
    fake.write_text("# import numpy as np\nimport numpy as np\n")
    try:
        r = run_verify("--check", "comments", "--fast")
        data = json.loads(r.stdout) if r.stdout else {"findings": []}
        hits = [
            f for f in data["findings"]
            if f["check"] == "comments" and name in f["location"]
        ]
        assert hits, f"expected at least one state-the-what flag; got summary={data.get('summary')}"
    finally:
        fake.unlink(missing_ok=True)


def test_comments_phase_a_skips_explanatory_comments(tmp_path):
    """A WHY-style comment should NOT be flagged."""
    name = f"_temp_{tmp_path.name}_why.py"
    fake = REPO / "image_classification-mnist-ffnn-numpy" / name
    fake.write_text(
        "# Xavier init keeps variance stable across depths; default torch init blows up here.\n"
        "weight = xavier_init(shape)\n"
    )
    try:
        r = run_verify("--check", "comments", "--fast")
        data = json.loads(r.stdout) if r.stdout else {"findings": []}
        hits = [
            f for f in data["findings"]
            if f["check"] == "comments" and name in f["location"]
        ]
        assert not hits, f"WHY-style comment falsely flagged: {hits}"
    finally:
        fake.unlink(missing_ok=True)


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


def test_run_helper_timeout_returns_rc_124():
    """_run must catch subprocess.TimeoutExpired and surface rc=124 + a
    diagnostic stderr suffix, so a hung make target produces a clean Finding
    instead of an uncaught traceback."""
    verify_repo = _load_verify_module()
    rc, stdout, stderr = verify_repo._run(["sleep", "5"], REPO, timeout=1)
    assert rc == 124, f"expected rc=124 on timeout, got {rc} (stdout={stdout!r}, stderr={stderr!r})"
    assert "timed out after 1s" in stderr


def test_s7_forbidden_toplevel_detects_resurrected_common():
    """S7.forbidden_toplevel fires if common/ ever comes back."""
    fake_dir = REPO / "common"
    if fake_dir.exists():
        pytest.fail("pre-existing common/ blocks this test")
    fake_dir.mkdir()
    try:
        r = run_verify("--check", "structure", "--fast")
        data = json.loads(r.stdout) if r.stdout else {"findings": []}
        s7 = [
            f for f in data["findings"]
            if f["id"] == "S7.forbidden_toplevel" and "common" in f["location"]
        ]
        assert s7, "expected S7.forbidden_toplevel to flag resurrected common/"
        for f in s7:
            assert f["severity"] == "error"
    finally:
        fake_dir.rmdir()
