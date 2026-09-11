"""Tests for scripts/verify_repo.py — the five-check oracle."""
from __future__ import annotations

import json
import hashlib
import builtins
import copy
import os
import re
import ast
import shlex
import shutil
import subprocess
import sys
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pytest
import yaml

from scripts import verify_repo
from scripts.stamp_notebook_source_hashes import compute_source_hash

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO
SCRIPT = REPO / "scripts" / "verify_repo.py"
ACTIVE_FIXTURE_DIR = "notebooks/image_classification-mnist-ffnn-numpy"
TEST_SUBPROCESS_TIMEOUT = 30
ISSUE62_PLAN = (
    REPO
    / "docs"
    / "superpowers"
    / "plans"
    / "2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md"
)

_ISSUE62_PULL_REQUEST_CHECKOUTS = {
    ".github/workflows/ci.yml": {
        "atlas-consumer-policy",
        "dependency-audit",
        "pytest-repository",
        "pytest-nnx-surface",
        "verify-repo",
        "docs-build",
        "docker-build",
        "tier-a-papermill",
        "smoke-tier-b",
        "smoke-tier-c",
    },
    ".github/workflows/docs.yml": {"check"},
    ".github/workflows/atlas-contract.yml": {"atlas-contract"},
}


def test_dependency_lock_d10_integration_is_clean() -> None:
    assert verify_repo._dependency_lock_findings(REPO_ROOT) == []


def test_dependency_advisory_lock_d10_integration_is_clean() -> None:
    assert verify_repo._dependency_advisory_lock_findings(REPO_ROOT) == []


def _copy_nlp_asset_contract(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    paths = (
        "requirements/nlp-assets.toml",
        "nlp-model-requirements.txt",
        "Makefile",
        "Dockerfile",
        ".github/workflows/ci.yml",
        ".devcontainer/devcontainer.json",
        "notebooks/sentiment_classification-vader-mlp-pytorch/notebook.ipynb",
        "scripts/verify_dependency_locks.py",
        "scripts/atlas_runtime_probe.py",
    )
    for relative in paths:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    return repo


def _mutate_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    mutated = source.replace(old, new, 1)
    assert mutated != source
    path.write_text(mutated, encoding="utf-8")


def _assert_one_nlp_asset_finding(repo: Path, location: str) -> None:
    findings = verify_repo._nlp_asset_contract_findings(repo)
    assert len(findings) == 1
    assert findings[0].id == "D11.nlp_asset_contract"
    assert findings[0].check == "assets"
    assert findings[0].location == location


def test_nlp_asset_d11_clean_control() -> None:
    assert verify_repo._nlp_asset_contract_findings(REPO_ROOT) == []


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("raw.githubusercontent.com/nltk/nltk_data", "example.invalid/nltk/nltk_data"),
        ("8adba4294eef3964", "0adba4294eef3964"),
        ("size = 90486", "size = 90485"),
        ("vader_lexicon/vader_lexicon.txt", "vader_lexicon/other.txt"),
    ),
)
def test_nlp_asset_d11_rejects_manifest_identity_drift(
    tmp_path: Path, old: str, new: str
) -> None:
    repo = _copy_nlp_asset_contract(tmp_path)
    _mutate_once(repo / "requirements/nlp-assets.toml", old, new)

    _assert_one_nlp_asset_finding(repo, "requirements/nlp-assets.toml")


@pytest.mark.parametrize(
    "mutation",
    (
        "\nfrom nltk import download as fetch\nfetch('vader_lexicon')\n",
        "\nnltk.download = lambda *args: None\n",
        "\ngetattr(nltk, 'download')('vader_lexicon')\n",
        "\n!python -m nltk.downloader vader_lexicon\n",
    ),
)
def test_nlp_asset_d11_rejects_notebook_download_bypasses(
    tmp_path: Path, mutation: str
) -> None:
    repo = _copy_nlp_asset_contract(tmp_path)
    path = repo / "notebooks/sentiment_classification-vader-mlp-pytorch/notebook.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    setup = next(
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "sentiment/vader_lexicon.zip" in "".join(cell.get("source", []))
    )
    setup["source"].append(mutation)
    path.write_text(json.dumps(notebook), encoding="utf-8")

    _assert_one_nlp_asset_finding(
        repo,
        "notebooks/sentiment_classification-vader-mlp-pytorch/notebook.ipynb",
    )


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    (
        ("Makefile", "\nverify-nlp-assets:\n", "\nverify-assets:\n"),
        ("Dockerfile", "  && make verify-nlp-assets \\\n", ""),
        (
            ".github/workflows/ci.yml",
            "          make verify-nlp-assets\n",
            "",
        ),
    ),
)
def test_nlp_asset_d11_rejects_missing_consumer_verification(
    tmp_path: Path, relative: str, old: str, new: str
) -> None:
    repo = _copy_nlp_asset_contract(tmp_path)
    _mutate_once(repo / relative, old, new)

    _assert_one_nlp_asset_finding(repo, relative)


def test_nlp_asset_d11_rejects_spacy_identity_drift(tmp_path: Path) -> None:
    repo = _copy_nlp_asset_contract(tmp_path)
    _mutate_once(
        repo / "nlp-model-requirements.txt",
        "en_core_web_sm-3.8.0",
        "en_core_web_sm-3.7.0",
    )

    _assert_one_nlp_asset_finding(repo, "nlp-model-requirements.txt")


def test_nlp_asset_d11_maps_malformed_workflow_to_stable_finding(tmp_path: Path) -> None:
    repo = _copy_nlp_asset_contract(tmp_path)
    (repo / ".github/workflows/ci.yml").write_text("jobs: []\n", encoding="utf-8")

    _assert_one_nlp_asset_finding(repo, ".github/workflows/ci.yml")


def test_dependency_advisory_lock_d10_fails_closed_on_projection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import advisory_baseline

    def fail_projection(repo: Path, projection_root: Path):
        del repo, projection_root
        raise advisory_baseline.AdvisoryBaselineError("injected lock projection drift")

    monkeypatch.setattr(advisory_baseline, "derive_lock_audit_surfaces", fail_projection)

    findings = verify_repo._dependency_advisory_lock_findings(REPO_ROOT)

    assert [finding.id for finding in findings] == ["D10.dependency_advisory_locks"]
    assert "injected lock projection drift" in findings[0].message


def test_dependency_lock_d10_reports_missing_output(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "requirements").mkdir(parents=True)
    from scripts.dependency_locks import load_policy

    policy = load_policy(REPO_ROOT)
    for relative in policy.inputs + policy.outputs:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    shutil.copy2(REPO_ROOT / "requirements/lock-policy.toml", repo / "requirements")
    shutil.copy2(REPO_ROOT / "requirements/image-lock.json", repo / "requirements")
    shutil.copy2(REPO_ROOT / "requirements/nlp-assets.toml", repo / "requirements")
    shutil.copy2(REPO_ROOT / "nlp-model-requirements.txt", repo)
    (repo / "requirements/locks/compiler.txt").unlink()

    findings = verify_repo._dependency_lock_findings(repo)

    assert {finding.id for finding in findings} == {"D10.dependency_locks"}
    assert findings[0].location == "requirements/locks/compiler.txt"


def _assert_pull_request_checkouts_use_synthetic_merge_default(
    workflows: Mapping[str, dict],
) -> None:
    assert set(workflows) == set(_ISSUE62_PULL_REQUEST_CHECKOUTS)
    for path, expected_jobs in _ISSUE62_PULL_REQUEST_CHECKOUTS.items():
        workflow = workflows[path]
        assert "pull_request" in workflow["on"]
        assert set(workflow["jobs"]) == expected_jobs
        for job_name, job in workflow["jobs"].items():
            checkouts = [
                step
                for step in job["steps"]
                if str(step.get("uses", "")).startswith("actions/checkout@")
            ]
            assert len(checkouts) == 1, (path, job_name)
            assert "ref" not in checkouts[0].get("with", {}), (path, job_name)


def _load_issue62_pull_request_workflows() -> dict[str, dict]:
    return {
        path: _load_workflow(REPO / path)
        for path in _ISSUE62_PULL_REQUEST_CHECKOUTS
    }


def test_pull_request_checkouts_preserve_default_synthetic_merge_ref() -> None:
    _assert_pull_request_checkouts_use_synthetic_merge_default(
        _load_issue62_pull_request_workflows()
    )


@pytest.mark.parametrize(
    ("path", "job_name"),
    tuple(
        (path, job_name)
        for path, job_names in _ISSUE62_PULL_REQUEST_CHECKOUTS.items()
        for job_name in sorted(job_names)
    ),
)
@pytest.mark.parametrize(
    "mutation",
    (
        "${{ github.event.pull_request.head.sha }}",
        "${{ github.event.pull_request.base.sha }}",
        "refs/heads/develop",
    ),
    ids=("head", "base", "arbitrary"),
)
def test_pull_request_checkout_contract_rejects_ref_overrides(
    path: str,
    job_name: str,
    mutation: str,
) -> None:
    workflows = _load_issue62_pull_request_workflows()
    checkout = next(
        step
        for step in workflows[path]["jobs"][job_name]["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    checkout.setdefault("with", {})["ref"] = mutation

    with pytest.raises(AssertionError):
        _assert_pull_request_checkouts_use_synthetic_merge_default(workflows)


def _assert_issue62_pr_dual_identity_plan(plan_source: str) -> None:
    task7 = plan_source.split("## 12.22.11 Task 7:", maxsplit=1)[1]
    assert '--commit "$PR_MERGE_SHA"' not in task7
    assert '--commit "$RELEASE_PR_MERGE_SHA"' not in task7
    assert '--commit "$SYNC_PR_TEST_MERGE_SHA"' not in task7
    for source_sha in ("FEATURE_SHA", "DEVELOP_MERGE_SHA", "RELEASE_MERGE_SHA"):
        assert f'--commit "${source_sha}"' in task7
    assert task7.count("python -m scripts.verify_pr_run_evidence") == 3
    for checks in (
        "pr-checks.json",
        "release-pr-checks.json",
        "sync-pr-checks.json",
    ):
        assert f'--checks-json "$FINAL_ROOT/{checks}"' in task7
    for evidence in (
        "feature-pr-run-evidence.json",
        "release-pr-run-evidence.json",
        "sync-pr-run-evidence.json",
    ):
        assert f'--output "$FINAL_ROOT/{evidence}"' in task7
        assert evidence in task7.split("evidence_paths = [", maxsplit=1)[1]
    assert task7.count("potentialMergeCommit") == 3
    assert task7.count("headRepository") == 3
    assert task7.count("--log >") == 9
    assert task7.count('manifest = {"schema": 2, "runs": [') == 3
    assert task7.count('"contaminating_ci"') >= 4
    assert task7.count('"contaminating_pr_run_urls"') >= 7
    assert task7.count('displayTitle,event,headSha,headBranch,createdAt') >= 3
    assert task7.count("--add-label tier-b-smoke") == 2
    assert task7.count(
        'selected = [(run, action) for run, action in ci_actions '
        'if action in {"labeled", "synchronize"}]'
    ) == 2
    assert task7.count("assert len(selected) == 1 and len(opened) <= 1") == 2
    assert task7.count('action in {"opened", "synchronize"}') == 1
    assert task7.count(
        'sync_run_evidence["runs"][0]["action"] in {"opened", "synchronize"}'
    ) == 1
    assert task7.count('url.startswith(item["url"] + "/job/")') == 1
    assert task7.count(
        'url.startswith(sync_run_evidence["runs"][0]["url"] + "/job/")'
    ) == 1
    assert 'by_check = {item["name"]: item for item in checks}' not in task7
    assert 'sync_by_check = {item["name"]: item for item in sync_checks}' not in task7
    for mutation_name in (
        "wrong_pr_source_identity",
        "wrong_pr_merge_identity",
        "wrong_pr_evidence_hash",
        "wrong_pr_check_association",
        "wrong_pr_contaminant_url",
    ):
        assert task7.count(mutation_name) == 3


def _assert_issue62_reuse_queries_select_source_heads(plan_source: str) -> None:
    task7 = plan_source.split("## 12.22.11 Task 7:", maxsplit=1)[1]
    for label, source_sha in (
        ("feature", "FEATURE_SHA"),
        ("release", "DEVELOP_MERGE_SHA"),
    ):
        start = f': > "$FINAL_ROOT/reusable-{label}-pr"'
        end = f'done < "$FINAL_ROOT/current-{label}-prs"'
        block = task7.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]
        assert block.count(f'--commit "${source_sha}" --limit 20') == 1
        assert block.count(
            f'CANDIDATE_RUN=$(python - "$FINAL_ROOT/candidate-{label}-runs.json"'
        ) == 1
        assert block.count(f'"${source_sha}" "$CANDIDATE_PR" <<\'PY\'') == 1
        assert 'if action in {"labeled", "synchronize"}:' in block
        assert 'assert len(selected) == 1' in block
        assert '--commit "$CANDIDATE_MERGE_SHA"' not in block
        assert '"$CANDIDATE_MERGE_SHA" "$CANDIDATE_PR"' not in block


def test_issue62_task7_plan_preserves_pr_source_and_synthetic_identities() -> None:
    plan_source = ISSUE62_PLAN.read_text(encoding="utf-8")
    _assert_issue62_pr_dual_identity_plan(plan_source)
    _assert_issue62_reuse_queries_select_source_heads(plan_source)


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            '--commit "$FEATURE_SHA" --limit 20',
            '--commit "$CANDIDATE_MERGE_SHA" --limit 20',
        ),
        (
            '"$FEATURE_SHA" "$CANDIDATE_PR" <<\'PY\'',
            '"$CANDIDATE_MERGE_SHA" "$CANDIDATE_PR" <<\'PY\'',
        ),
        (
            '--commit "$DEVELOP_MERGE_SHA" --limit 20',
            '--commit "$CANDIDATE_MERGE_SHA" --limit 20',
        ),
        (
            '"$DEVELOP_MERGE_SHA" "$CANDIDATE_PR" <<\'PY\'',
            '"$CANDIDATE_MERGE_SHA" "$CANDIDATE_PR" <<\'PY\'',
        ),
    ),
    ids=("feature-query", "feature-selector", "release-query", "release-selector"),
)
def test_issue62_reuse_query_and_selector_reject_identity_mutations(
    old: str,
    new: str,
) -> None:
    control = ISSUE62_PLAN.read_text(encoding="utf-8")
    mutated = control.replace(old, new, 1)
    assert mutated != control

    with pytest.raises(AssertionError):
        _assert_issue62_reuse_queries_select_source_heads(mutated)


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ('--commit "$FEATURE_SHA"', '--commit "$PR_MERGE_SHA"'),
        (
            "python -m scripts.verify_pr_run_evidence",
            "python -m scripts.verify_smoke_outputs",
        ),
        ("feature-pr-run-evidence.json", "feature-pr-runs.json"),
        ("potentialMergeCommit", "mergeCommit"),
        ("headRepository", "sourceRepository"),
        ("--log >", "--log-failed >"),
        ('manifest = {"schema": 2, "runs": [', 'manifest = {"schema": 1, "runs": ['),
        ('"contaminating_pr_run_urls"', '"ignored_pr_run_urls"'),
        ("--add-label tier-b-smoke", "--remove-label tier-b-smoke"),
        (
            'if action in {"labeled", "synchronize"}]',
            'if action in {"opened", "synchronize"}]',
        ),
        (
            "assert len(selected) == 1 and len(opened) <= 1",
            "assert len(selected) == 1 and len(opened) <= 2",
        ),
        (
            'sync_run_evidence["runs"][0]["action"] in {"opened", "synchronize"}',
            'sync_run_evidence["runs"][0]["action"] == "opened"',
        ),
        ("--checks-json", "--unbound-checks-json"),
        (
            'url.startswith(item["url"] + "/job/")',
            'url.startswith("https://github.com/")',
        ),
        (
            'url.startswith(sync_run_evidence["runs"][0]["url"] + "/job/")',
            'url.startswith("https://github.com/")',
        ),
        ("wrong_pr_source_identity", "wrong_source_identity"),
        ("wrong_pr_merge_identity", "wrong_merge_identity"),
        ("wrong_pr_evidence_hash", "wrong_evidence_hash"),
        ("wrong_pr_check_association", "wrong_check_association"),
    ),
)
def test_issue62_task7_dual_identity_plan_rejects_mutations(old: str, new: str) -> None:
    control = ISSUE62_PLAN.read_text(encoding="utf-8")
    _assert_issue62_pr_dual_identity_plan(control)
    prefix, task7 = control.split("## 12.22.11 Task 7:", maxsplit=1)
    mutated = prefix + "## 12.22.11 Task 7:" + task7.replace(old, new, 1)

    with pytest.raises(AssertionError):
        _assert_issue62_pr_dual_identity_plan(mutated)


def _assert_issue62_qat_debt_plan_selectors(plan_source: str) -> None:
    selectors = tuple(
        re.findall(
            r"tests/nnx_surface/test_quantization_mnist_ffnn_pytorch\.py \\\n"
            r"    -q -k '([^']+)'",
            plan_source,
        )
    )
    assert selectors == ("qat_warning_debt", "qat_warning_debt")
    test_tree = ast.parse(
        (
            REPO
            / "tests"
            / "nnx_surface"
            / "test_quantization_mnist_ffnn_pytorch.py"
        ).read_text(encoding="utf-8")
    )
    debt_tests = tuple(
        node.name
        for node in test_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_qat_warning_debt_")
    )
    assert debt_tests
    assert all(selectors[0] in name for name in debt_tests)


def test_issue62_qat_debt_plan_selectors_cover_every_debt_test_family():
    _assert_issue62_qat_debt_plan_selectors(ISSUE62_PLAN.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "mutation",
    (None, "qat_warning_debt_validator", "qat_warning_debt_key", "qat_warning_debt "),
    ids=("omitted", "validator-only", "key-only", "trailing-space"),
)
def test_issue62_qat_debt_plan_selectors_reject_narrowing_mutations(mutation):
    control = ISSUE62_PLAN.read_text(encoding="utf-8").replace(
        "-q -k 'qat_warning_debt_validator'",
        "-q -k 'qat_warning_debt'",
    )
    _assert_issue62_qat_debt_plan_selectors(control)
    replacement = "-q" if mutation is None else f"-q -k '{mutation}'"
    mutated = control.replace("-q -k 'qat_warning_debt'", replacement, 1)

    with pytest.raises(AssertionError):
        _assert_issue62_qat_debt_plan_selectors(mutated)


def _parse_exact_direct_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([a-z0-9][a-z0-9._-]*)==([^\s\\;@]+)", line, re.IGNORECASE)
        assert match, f"{path.name} must contain only exact package pins: {line!r}"
        package, version = match.groups()
        package = package.lower()
        assert package not in pins, f"{path.name} repeats {package}"
        pins[package] = version
    return pins


def _documentation_pin(package: str) -> str:
    matches = [
        match.group(1)
        for line in (REPO / "docs-requirements.txt").read_text(encoding="utf-8").splitlines()
        if (
            match := re.fullmatch(
                rf"{re.escape(package)}==([^\s\\;@]+) " + re.escape(chr(92)),
                line,
                re.IGNORECASE,
            )
        )
    ]
    assert len(matches) == 1, f"docs-requirements.txt must pin {package} exactly once"
    return matches[0]


def test_atlas_contract_direct_dependencies_match_documentation_pins():
    requirements_path = REPO / "atlas-contract-requirements.txt"

    assert requirements_path.is_file(), "atlas-contract-requirements.txt is missing"
    assert _parse_exact_direct_pins(requirements_path) == {
        "nltk": "3.10.3",
        "pytest": _documentation_pin("pytest"),
        "pyyaml": _documentation_pin("pyyaml"),
        "uv": "0.11.19",
    }


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


def _write_root_governance_manifest(
    repo: Path,
    *,
    source: str = "SUPPORT.md",
    number: str = "13",
    title: str = "Support",
) -> None:
    docs = repo / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "manifest.yaml").write_text(
        "surfaces: [repo, site, wiki]\n"
        "numbering: baked\n"
        "sections:\n"
        "  - id: support\n"
        f"    number: '{number}'\n"
        f"    title: {title}\n"
        f"    source: {source}\n"
        "notebooks: []\n"
        "diagrams: []\n",
        encoding="utf-8",
    )


def test_docs_adapter_skips_synthetic_fixture_without_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_repo, "REQUIRED_SECTIONS", {})

    result = verify_repo.check_docs(tmp_path)

    assert not [finding for finding in result.findings if finding.id == "D10.notebook_infrastructure"]


def test_docs_adapter_reports_invalid_manifest_and_continues_baseline_scans(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(verify_repo, "REQUIRED_SECTIONS", {})
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/manifest.yaml").write_text("sections: [\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "Use the Jupyter Hub deployment.\n",
        encoding="utf-8",
    )

    result = verify_repo.check_docs(tmp_path)

    assert any(
        finding.id == "D9.invalid_manifest"
        and finding.severity == "error"
        and finding.location == "docs/manifest.yaml"
        for finding in result.findings
    ), result.findings
    assert any(
        finding.id == "D8.terminology"
        and finding.location == "README.md:1"
        for finding in result.findings
    ), result.findings


def test_docs_adapter_reports_drift_for_a_real_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_repo, "REQUIRED_SECTIONS", {})
    (tmp_path / "docs/notebooks").mkdir(parents=True)
    (tmp_path / "docs/manifest.yaml").write_text(
        "surfaces: [repo, site, wiki]\nnumbering: baked\nsections:\n  - id: overview\n"
        "    number: '1'\n    title: Overview\n    source: docs/index.md\nnotebooks:\n"
        "  - task: task\n    number: '8.1'\n    family: test\n    depth: full\n"
        "    doc: docs/notebooks/task.md\n    spec: notebooks/task/docs/spec.yaml\ndiagrams: []\n",
        encoding="utf-8",
    )
    (tmp_path / "docs/index.md").write_text("# 1. Overview\n", encoding="utf-8")
    (tmp_path / "docs/notebooks/task.md").write_text("# 8.1 Task\n", encoding="utf-8")
    (tmp_path / "docs/notebook-infrastructure.md").write_text(
        "<!-- atlas-task-contracts:start -->\n| stale |\n<!-- atlas-task-contracts:end -->\n",
        encoding="utf-8",
    )
    (tmp_path / "notebooks/task/docs").mkdir(parents=True)
    (tmp_path / "notebooks/task/docs/spec.yaml").write_text(
        "title: Task\ntier: A\natlas:\n  executor: jupyterhub\n  default_mode: vscode-remote\n"
        "  required_services: [jupyterhub]\n  required_env: []\n  workspace_access: remote\n"
        "  artifact_policy: atlas-jupyter-volume\n  constraints: []\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/verify_repo_config.yaml").write_text("active_task_dirs: [task]\n", encoding="utf-8")

    result = verify_repo.check_docs(tmp_path)

    assert [finding.id for finding in result.findings if finding.id == "D10.notebook_infrastructure"] == ["D10.notebook_infrastructure"]


def test_om_006_is_resolved_by_nnx_verifier_and_ci_contract():
    maintenance = (REPO / "docs/maintenance/overnight-2026-07-04.md").read_text(
        encoding="utf-8"
    )
    om_006 = next(line for line in maintenance.splitlines() if line.startswith("| OM-006 |"))

    assert "| Resolved |" in om_006
    assert "verifier and CI contract" in om_006
    assert "live evidence is recorded on Issue #58 after rollout" in om_006
    assert "already live" not in om_006


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


def test_structure_s3_current_markdown_links_resolve():
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s3 = [f for f in data["findings"] if f["id"].startswith("S3")]
    assert s3 == [], f"S3 found broken current-repository links: {s3}"


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


def test_structure_s1_flags_missing_archive_notebook_cell_id(tmp_path):
    """Archive notebooks must satisfy the same raw nbformat cell-id policy."""
    repo = _temp_repo(tmp_path)
    name = "archive-missing-cell-id.ipynb"
    archive = repo / "notebooks" / "archive" / "old-task" / name
    archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps({
        "cells": [{
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Archived\n",
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
    assert hits, f"expected S1.cell_id for archive {name}; got {data.get('findings')}"


def test_structure_s1_flags_invalid_notebook_schema(tmp_path):
    """Raw notebook schema validation should catch id/minor-version mismatches."""
    repo = _temp_repo(tmp_path)
    name = "invalid-schema.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    fake.write_text(json.dumps({
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "id": "abc123",
            "metadata": {},
            "outputs": [],
            "source": "x = 1\n",
        }],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2,
    }))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S1.schema" and name in f["location"]
    ]
    assert hits, f"expected S1.schema for invalid notebook schema; got {data.get('findings')}"


def test_repo_root_uses_target_repo_config_for_active_dirs(tmp_path):
    """`--repo-root` should verify notebooks listed by that repo's own config."""
    repo = tmp_path
    task_dir = repo / "notebooks" / "custom-task"
    task_dir.mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "verify_repo_config.yaml").write_text(
        "\n".join([
            "active_task_dirs:",
            "  - custom-task",
            "tier_a_notebooks:",
            "  - notebooks/custom-task/notebook.ipynb",
            "",
        ]),
        encoding="utf-8",
    )
    nb_path = task_dir / "notebook.ipynb"
    nb_path.write_text(json.dumps({
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
    subprocess.run(
        ["git", "init"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")

    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S1.cell_id"
        and f["location"] == "notebooks/custom-task/notebook.ipynb:cell[0]"
    ]
    assert hits, f"expected target repo config to include custom notebook; got {data.get('findings')}"


def test_structure_s5_no_common_imports():
    """No `from common.` import anywhere in active task notebooks or scripts."""
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s5 = [f for f in data["findings"] if f["id"].startswith("S5")]
    assert s5 == [], f"S5 found stray common.* imports: {s5}"


def test_structure_s5_flags_common_alias_inside_python_multi_import(tmp_path):
    """A valid first import must not hide a forbidden common import alias."""
    repo = _temp_repo(tmp_path)
    module = repo / "stray_common_multi_import.py"
    module.write_text("import os, common.utils\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", str(module)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S5.common_import"
        and f["location"] == "stray_common_multi_import.py:1"
    ]
    assert hits, f"expected S5.common_import for multi-import alias; got {data.get('findings')}"


def test_structure_s5_flags_common_alias_inside_notebook_multi_import(tmp_path):
    """Notebook cells should get the same multi-import common scan as scripts."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "multi-import-common.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("import os, common.utils\n")
    ]
    nbformat.write(nb, str(fake))
    subprocess.run(
        ["git", "add", "-f", str(fake)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S5.common_import"
        and f["location"] == f"{ACTIVE_FIXTURE_DIR}/{name}:cell[0]:line[1]"
    ]
    assert hits, f"expected S5.common_import for notebook multi-import alias; got {data.get('findings')}"


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


def test_structure_s2_reports_one_based_line_numbers(tmp_path):
    """S2 locations should use the same one-based line convention as other findings."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "one-based-import-line.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "x = 1\nimport definitely_missing_module_for_line_number\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_module_for_line_number" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import; got {data.get('findings')}"
    assert hits[0]["location"].endswith(":cell[0]:line[2]")


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


def test_structure_s2_ignores_acknowledged_runtime_only_imports(tmp_path):
    """Tier-C runtime-container modules should not create recurring local warnings."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "runtime-only-import.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell("import torch_sparse\n")]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "torch_sparse" in f["message"]
    ]
    assert not hits, f"runtime-only import should be acknowledged, got {hits}"


def test_structure_s2_checks_literal_dynamic_imports(tmp_path):
    """Literal importlib/__import__ calls should not bypass unresolved-import checks."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "literal-dynamic-imports.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import importlib\n"
            "importlib.import_module('definitely_missing_dynamic_import')\n"
            "__import__('also_missing_dynamic_import')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    messages = [
        f["message"] for f in data["findings"]
        if f["id"] == "S2.unresolved_import" and name in f["location"]
    ]
    assert any("definitely_missing_dynamic_import" in m for m in messages), data.get("findings")
    assert any("also_missing_dynamic_import" in m for m in messages), data.get("findings")


def test_structure_s2_checks_literal_dynamic_import_aliases(tmp_path):
    """Literal dynamic imports should be checked through common importlib aliases."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "literal-dynamic-import-aliases.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "from importlib import import_module\n"
            "import importlib as il\n"
            "import_module('definitely_missing_from_import_alias')\n"
            "il.import_module('definitely_missing_module_alias')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    messages = [
        f["message"] for f in data["findings"]
        if f["id"] == "S2.unresolved_import" and name in f["location"]
    ]
    assert any("definitely_missing_from_import_alias" in m for m in messages), data.get("findings")
    assert any("definitely_missing_module_alias" in m for m in messages), data.get("findings")


def test_structure_s2_checks_missing_dotted_submodules(tmp_path):
    """An importable top-level package must not hide a missing dotted submodule."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "missing-dotted-submodules.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import json.definitely_missing_submodule_for_s2\n"
            "from json.definitely_missing_from_submodule import VALUE\n"
            "importlib.import_module('json.definitely_missing_dynamic_submodule')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    messages = [
        f["message"] for f in data["findings"]
        if f["id"] == "S2.unresolved_import" and name in f["location"]
    ]
    assert any("json.definitely_missing_submodule_for_s2" in m for m in messages), data.get("findings")
    assert any("json.definitely_missing_from_submodule" in m for m in messages), data.get("findings")
    assert any("json.definitely_missing_dynamic_submodule" in m for m in messages), data.get("findings")


def test_structure_s2_fallback_checks_literal_dynamic_import_aliases(tmp_path):
    """Syntax-error fallback should still check importlib alias calls."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "literal-dynamic-import-alias-fallback.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import importlib as il\n"
            "il.import_module('definitely_missing_alias_fallback')\n"
            "x = [\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_alias_fallback" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import for fallback dynamic alias; got {data.get('findings')}"


def test_structure_s2_fallback_checks_multiline_literal_dynamic_import_aliases(tmp_path):
    """Syntax-error fallback should still track parenthesized importlib aliases."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "literal-dynamic-import-multiline-alias-fallback.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "from importlib import (\n"
            "    import_module as im,\n"
            ")\n"
            "im('definitely_missing_multiline_alias_fallback')\n"
            "if True print('force fallback')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_multiline_alias_fallback" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import for multiline fallback alias; got {data.get('findings')}"


def test_structure_s2_fallback_ignores_parentheses_in_import_comments(tmp_path):
    """Comment text must not break fallback reconstruction of multiline imports."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "fallback-comment-parentheses.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "from importlib import (  # keep ) in comment\n"
            "    import_module,\n"
            ")\n"
            "not valid python ???\n"
            "import_module('definitely_missing_comment_paren_dynamic')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_comment_paren_dynamic" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import with comment paren import alias; got {data.get('findings')}"


def test_structure_s2_fallback_checks_multiline_literal_dynamic_import_calls(tmp_path):
    """Syntax-error fallback should still check multiline import_module calls."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "literal-dynamic-import-multiline-call-fallback.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import importlib\n"
            "importlib.import_module(\n"
            "    'definitely_missing_multiline_call_fallback'\n"
            ")\n"
            "if True print('force fallback')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_multiline_call_fallback" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import for multiline fallback call; got {data.get('findings')}"


def test_structure_s2_fallback_checks_backslash_continued_multi_imports(tmp_path):
    """Syntax-error fallback should still check every backslash-continued import."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "backslash-continued-multi-import-fallback.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import json, \\\n"
            "    definitely_missing_backslash_import\n"
            "if True print('force fallback')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_backslash_import" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import for backslash import; got {data.get('findings')}"


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


def test_structure_s2_flags_dotted_notebook_relative_imports(tmp_path):
    """Sibling helper files must not hide dotted relative imports in notebooks."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "dotted-relative-import.ipynb"
    task_dir = repo / ACTIVE_FIXTURE_DIR
    (task_dir / "helpers.py").write_text("VALUE = 1\n", encoding="utf-8")
    fake = task_dir / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("from .helpers import VALUE\n")
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.relative_import"
        and name in f["location"]
        and "helpers" in f["message"]
    ]
    assert hits, f"expected S2.relative_import for dotted relative import; got {data.get('findings')}"


def test_structure_s2_dedupe_does_not_hide_relative_imports(tmp_path):
    """A normal sibling import must not suppress a later relative-import finding."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "deduped-relative-import.ipynb"
    task_dir = repo / ACTIVE_FIXTURE_DIR
    (task_dir / "helpers.py").write_text("VALUE = 1\n", encoding="utf-8")
    fake = task_dir / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import helpers\nfrom .helpers import VALUE\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.relative_import"
        and name in f["location"]
        and "helpers" in f["message"]
    ]
    assert hits, f"expected relative import despite earlier normal import; got {data.get('findings')}"


def test_structure_s2_fallback_ignores_multiline_string_import_text(tmp_path):
    """Syntax fallback must not scan import-looking text inside multiline strings."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "fallback-string-import-text.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "note = '''\n"
            "import definitely_missing_inside_multiline_string\n"
            "'''\n"
            "if True print('force fallback')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if name in f["location"]
        and "definitely_missing_inside_multiline_string" in f["message"]
    ]
    assert hits == [], f"fallback should ignore multiline string import text; got {hits}"


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


def test_structure_s3_checks_manifest_declared_root_markdown_links(tmp_path):
    """Every manifest-declared root document is part of repository link hygiene."""
    repo = _temp_repo(tmp_path)
    _write_root_governance_manifest(repo)
    (repo / "SUPPORT.md").write_text(
        "# 13. Support\n\n[missing](docs/missing-support-runbook.md)\n",
        encoding="utf-8",
    )

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S3.broken_link" and f["location"] == "SUPPORT.md"
    ]
    assert hits, f"expected S3.broken_link for SUPPORT.md; got {data.get('findings')}"


def test_structure_s3_keeps_scanning_when_docs_manifest_is_invalid(tmp_path):
    """Malformed docs metadata must not hide ordinary repository structure findings."""
    repo = _temp_repo(tmp_path)
    (repo / "docs").mkdir()
    (repo / "docs/manifest.yaml").write_text("sections: [\n", encoding="utf-8")
    (repo / "README.md").write_text("[missing](missing.md)\n", encoding="utf-8")

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}

    assert any(
        finding["id"] == "S3.broken_link" and finding["location"] == "README.md"
        for finding in data["findings"]
    ), data.get("findings")


def test_structure_s3_flags_relative_links_that_escape_repo(tmp_path):
    """Repo docs should not silently validate sibling-directory links."""
    repo = _temp_repo(tmp_path / "repo")
    sibling = tmp_path / "nnx"
    sibling.mkdir()
    changelog = repo / "CHANGELOG.md"
    changelog.write_text("Historical example: (via [`nnx`](../nnx))\n", encoding="utf-8")

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S3.repo_escape_link" and f["location"] == "CHANGELOG.md"
    ]
    assert hits, f"expected repo-escaping link to be flagged; got {data.get('findings')}"


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
    for token in ("JupyterHub", "NumPy", "PyTorch"):
        assert token in SCRIPT_TEXT, f"D8 missing canonical {token!r}"


def test_docs_d8_scans_manifest_declared_root_markdown(tmp_path, monkeypatch):
    """Terminology checks cover arbitrary root governance documents."""
    monkeypatch.setattr(verify_repo, "REQUIRED_SECTIONS", {})
    _write_root_governance_manifest(tmp_path)
    (tmp_path / "SUPPORT.md").write_text(
        "# 13. Support\n\nUse the Jupyter Hub deployment.\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/verify_repo_config.yaml").write_text(
        "active_task_dirs: []\n",
        encoding="utf-8",
    )
    infrastructure = tmp_path / "docs/notebook-infrastructure.md"
    infrastructure.write_text(
        "<!-- atlas-task-contracts:start -->\n"
        "| Task | Tier | Default mode | Workspace access | Required Atlas services | Required environment | Artifact policy | Constraints |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "<!-- atlas-task-contracts:end -->\n",
        encoding="utf-8",
    )

    result = verify_repo.check_docs(tmp_path)

    assert any(
        finding.id == "D8.terminology" and finding.location == "SUPPORT.md:3"
        for finding in result.findings
    ), result.findings


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


def test_docs_d9_flags_malformed_published_docs_page_heading(tmp_path):
    """Published MkDocs pages should be included in numbered-heading checks."""
    repo = _temp_repo(tmp_path)
    page = repo / "docs" / "index.md"
    page.parent.mkdir()
    page.write_text(
        "# 1. Overview\n\n"
        "## 1.1. Nested heading depth on an H2\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "D9.numbered_heading" and f["location"] == "docs/index.md:3"
    ]
    assert hits, f"expected D9.numbered_heading for docs/index.md; got {data.get('findings')}"


def test_docs_d9_flags_malformed_published_diagram_provenance_heading(tmp_path):
    """Published diagram provenance docs should be included in numbered-heading checks."""
    repo = _temp_repo(tmp_path)
    page = repo / "docs" / "diagrams" / "README.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "# Diagram Provenance\n\n"
        "## 1.1. Nested heading depth on an H2\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "D9.numbered_heading" and f["location"] == "docs/diagrams/README.md:3"
    ]
    assert hits, f"expected D9.numbered_heading for docs/diagrams/README.md; got {data.get('findings')}"


def test_docs_d10_dependency_ledger_counts_match_current_doc():
    """Package counts and advisory feed-record counts should reconcile."""
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    d10 = [f for f in data["findings"] if f["id"] == "D10.dependency_ledger_count"]
    assert d10 == [], f"D10 reported dependency-ledger issues: {d10}"


def _advisory_baseline_repo(tmp_path: Path) -> Path:
    repo = _temp_repo(tmp_path)
    (repo / "docs").mkdir()
    shutil.copyfile(REPO / "docs/dependency-contracts.md", repo / "docs/dependency-contracts.md")
    (repo / "security").mkdir()
    shutil.copyfile(
        REPO / "security/accepted-advisories.json",
        repo / "security/accepted-advisories.json",
    )
    return repo


def _d10_advisory_baseline_findings(repo: Path):
    return [
        finding
        for finding in _load_verify_module()._dependency_ledger_findings(repo)
        if finding.id == "D10.dependency_advisory_baseline"
    ]


def _write_canonical_baseline(repo: Path, document: dict) -> None:
    (repo / "security/accepted-advisories.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


_ISSUE63_LEDGER_MARKER = "### 6.1.1.2 Current Issue #63 locked four-surface audit"


def _issue62_ledger_repo(tmp_path: Path) -> Path:
    repo = _advisory_baseline_repo(tmp_path)
    module = _load_verify_module()
    for relative in module._DEPENDENCY_HASH_INPUTS:
        source = REPO / relative
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return repo


def _d10_ids(repo: Path) -> set[str]:
    return {
        finding.id
        for finding in _load_verify_module()._dependency_ledger_findings(repo)
        if finding.id.startswith("D10.dependency_")
    }


def _issue62_section(text: str) -> str:
    start = text.index(_ISSUE63_LEDGER_MARKER)
    following = re.search(r"^#{1,3}[ \t]", text[start + len(_ISSUE63_LEDGER_MARKER):], re.MULTILINE)
    end = (
        start + len(_ISSUE63_LEDGER_MARKER) + following.start()
        if following is not None
        else len(text)
    )
    return text[start:end]


def _replace_issue62_section(repo: Path, mutate) -> None:
    ledger = repo / "docs/dependency-contracts.md"
    original = ledger.read_text(encoding="utf-8")
    section = _issue62_section(original)
    replacement = mutate(section)
    assert replacement != section
    ledger.write_text(original.replace(section, replacement, 1), encoding="utf-8")


def test_dependency_ledger_rejects_missing_or_duplicate_current_issue62_section(tmp_path):
    repo = _issue62_ledger_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    original = ledger.read_text(encoding="utf-8")
    ledger.write_text(
        original.replace(_ISSUE63_LEDGER_MARKER, "### 6.1.1.2 Archived audit", 1),
        encoding="utf-8",
    )
    assert "D10.dependency_ledger_count" in _d10_ids(repo)
    ledger.write_text(original + "\n" + _issue62_section(original), encoding="utf-8")
    assert "D10.dependency_ledger_count" in _d10_ids(repo)


@pytest.mark.parametrize(
    ("needle", "replacement"),
    (
        (
            "| Package | Manifest Constraint | Audited Resolved Version | Finding Count | Current Disposition |\n",
            "",
        ),
        (
            "| Package | Manifest Constraint | Audited Resolved Version | Finding Count | Current Disposition |",
            "| Broken summary header |",
        ),
        ("| --- | --- | ---: | ---: | --- |", "| --- |"),
        ("Result: ", "Result malformed: "),
    ),
)
def test_dependency_ledger_rejects_malformed_result_summary_and_advisory_tables(
    tmp_path, needle, replacement,
):
    repo = _issue62_ledger_repo(tmp_path)
    _replace_issue62_section(repo, lambda section: section.replace(needle, replacement, 1))
    assert "D10.dependency_ledger_count" in _d10_ids(repo)


def test_dependency_ledger_ignores_complete_historical_audit_tables(tmp_path):
    repo = _issue62_ledger_repo(tmp_path)
    assert _d10_ids(repo) == set()
    ledger = repo / "docs/dependency-contracts.md"
    original = ledger.read_text(encoding="utf-8")
    historical = _issue62_section(original).replace(
        _ISSUE63_LEDGER_MARKER,
        "### 6.1.13.1 Archived Issue #61 audit",
        1,
    )
    ledger.write_text(original + "\n## 6.1.13 Archive\n\n" + historical, encoding="utf-8")
    assert _d10_ids(repo) == set()


def test_dependency_ledger_rejects_missing_duplicate_reordered_and_stale_input_hashes(tmp_path):
    repo = _issue62_ledger_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    original = ledger.read_text(encoding="utf-8")
    section = _issue62_section(original)
    digest = hashlib.sha256((repo / "requirements.txt").read_bytes()).hexdigest()
    row = f"| `requirements.txt` | `{digest}` |"
    assert row in section
    next_row = next(
        line for line in section.splitlines()
        if line.startswith("| `torch-core-requirements.txt`")
    )
    mutations = (
        section.replace(row + "\n", "", 1),
        section.replace(row, row + "\n" + row, 1),
        section.replace(row + "\n" + next_row, next_row + "\n" + row, 1),
        section.replace(row, f"| `requirements.txt` | `{'0' * 64}` |", 1),
    )
    for mutated in mutations:
        ledger.write_text(original.replace(section, mutated, 1), encoding="utf-8")
        assert "D10.dependency_input_hash" in _d10_ids(repo)


@pytest.mark.parametrize("target", ("markdown", "json"))
@pytest.mark.parametrize("field", ("package", "advisory_id", "accepted_version", "surfaces"))
def test_dependency_ledger_couples_advisory_identity_version_and_surfaces_to_policy(
    tmp_path, target, field,
):
    repo = _issue62_ledger_repo(tmp_path)
    policy = repo / "security/accepted-advisories.json"
    document = json.loads(policy.read_text(encoding="utf-8"))
    item = next(
        entry for entry in document["accepted_advisories"]
        if entry["package"] == "setuptools"
    )
    replacements = {
        "package": "different-package",
        "advisory_id": "CVE-2099-0000",
        "accepted_version": "0.0.0",
        "surfaces": ["documentation"],
    }
    if target == "json":
        item[field] = replacements[field]
        _write_canonical_baseline(repo, document)
    else:
        def mutate_advisory_row(section: str) -> str:
            row = next(
                line for line in section.splitlines()
                if line.startswith(f"| `{item['package']}` | `{item['advisory_id']}` |")
            )
            replacements_by_field = {
                "package": row.replace(
                    f"| `{item['package']}` |",
                    f"| `{replacements['package']}` |",
                    1,
                ),
                "advisory_id": row.replace(
                    f"| `{item['advisory_id']}` |",
                    f"| `{replacements['advisory_id']}` |",
                    1,
                ),
                "accepted_version": row.replace(
                    f"| `{item['accepted_version']}` |",
                    f"| `{replacements['accepted_version']}` |",
                    1,
                ),
                "surfaces": row.replace(
                    row.rsplit("|", 2)[-2].strip(),
                    "Documentation",
                ),
            }
            return section.replace(row, replacements_by_field[field], 1)

        _replace_issue62_section(repo, mutate_advisory_row)
    assert "D10.dependency_advisory_baseline" in _d10_ids(repo)


def test_dependency_ledger_rejects_advisory_only_package_and_count_drift(tmp_path):
    repo = _issue62_ledger_repo(tmp_path)
    extra = (
        "| `advisory-only` | `CVE-2099-0001` | 1 | None listed | `1.0.0` | "
        "None listed | Combined runtime |"
    )
    def add_advisory_only_row(section: str) -> str:
        final_row = next(
            line for line in section.splitlines()
            if line.startswith("| `torch` | `PYSEC-2025-194` |")
        )
        return section.replace(final_row, f"{final_row}\n{extra}", 1)

    _replace_issue62_section(repo, add_advisory_only_row)
    assert "D10.dependency_ledger_count" in _d10_ids(repo)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda section: section.replace("known vulnerabilities", "known zero vulnerabilities", 1),
        lambda section: section.replace("torch-scatter", "torch-cluster", 1),
        lambda section: section.replace("torch-sparse", "torch-spline-conv", 1),
    ),
)
def test_dependency_ledger_rejects_zero_vulnerability_and_legacy_extension_claims(
    tmp_path, mutation,
):
    repo = _issue62_ledger_repo(tmp_path)
    _replace_issue62_section(repo, mutation)
    assert "D10.dependency_ledger_contract" in _d10_ids(repo)


def test_dependency_ledger_requires_pyg_lib_external_index_limitation(tmp_path):
    repo = _issue62_ledger_repo(tmp_path)
    _replace_issue62_section(
        repo,
        lambda section: section.replace(
            "pyg-lib is an exact external-index wheel outside ordinary PyPI audit coverage; "
            "its version and provenance are verified by `verify_torch_stack`.",
            "pyg-lib is fully covered by pip-audit.",
            1,
        ),
    )
    assert "D10.dependency_pyg_lib_limitation" in _d10_ids(repo)


@pytest.mark.parametrize("tag", ("script", "pre", "style", "textarea"))
@pytest.mark.parametrize("indent", ("", " ", "  ", "   "))
def test_dependency_raw_html_type1_requires_matching_close(tag, indent):
    module = _load_verify_module()
    hidden = _ISSUE63_LEDGER_MARKER
    visible = "### 6.1.1.3 Visible current audit"
    source = f"{indent}<{tag}>\n{hidden}\n\n{hidden}\n</{tag}>\n{visible}\n"
    masked = module._mask_dependency_raw_html(source)
    assert hidden not in masked
    assert masked.count(visible) == 1
    assert masked.count("\n") == source.count("\n")


_COMMONMARK_TYPE6_TAGS = (
    "address", "article", "aside", "base", "basefont", "blockquote", "body", "caption",
    "center", "col", "colgroup", "dd", "details", "dialog", "dir", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "frame", "frameset", "h1", "h2",
    "h3", "h4", "h5", "h6", "head", "header", "hgroup", "hr", "html", "iframe", "legend", "li",
    "link", "main", "menu", "menuitem", "nav", "noframes", "ol", "optgroup", "option", "p",
    "param", "search", "section", "source", "summary", "table", "tbody", "td", "tfoot", "th", "thead",
    "title", "tr", "track", "ul",
)


@pytest.mark.parametrize("tag", _COMMONMARK_TYPE6_TAGS)
@pytest.mark.parametrize("indent", ("", "   "))
def test_dependency_raw_html_type6_uses_blank_termination_without_swallowing_visible(tag, indent):
    module = _load_verify_module()
    hidden = _ISSUE63_LEDGER_MARKER
    visible = "### 6.1.1.3 Visible current audit"
    source = f"{indent}<{tag}>\n</{tag}>\n{hidden}\n\n{visible}\n"
    masked = module._mask_dependency_raw_html(source)
    assert hidden not in masked
    assert visible in masked
    assert masked.count("\n") == source.count("\n")


@pytest.mark.parametrize("indent", ("", " ", "  ", "   "))
def test_dependency_raw_html_hgroup_hides_decoy_but_visible_current_section_is_enforced(
    tmp_path, indent,
):
    repo = _issue62_ledger_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    original = ledger.read_text(encoding="utf-8")
    hidden_decoy = f"{indent}<hgroup>\n{_ISSUE63_LEDGER_MARKER}\n</hgroup>\n\n"
    ledger.write_text(hidden_decoy + original, encoding="utf-8")
    assert _d10_ids(repo) == set()
    ledger.write_text(
        hidden_decoy + original.replace(_ISSUE63_LEDGER_MARKER, "### 6.1.1.2 Removed visible audit", 1),
        encoding="utf-8",
    )
    assert "D10.dependency_ledger_count" in _d10_ids(repo)


def test_dependency_raw_html_four_spaces_remains_markdown_code_not_html():
    module = _load_verify_module()
    hidden = _ISSUE63_LEDGER_MARKER
    source = f"    <div>\n    {hidden}\n\n{hidden}\n"
    published = module._mask_dependency_raw_html(
        module._strip_markdown_code(source, strip_inline=False)
    )
    assert published.count(hidden) == 1


def test_docs_d10_dependency_advisory_baseline_matches_current_doc():
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        finding
        for finding in data["findings"]
        if finding["id"] == "D10.dependency_advisory_baseline"
    ]
    assert hits == [], f"D10 reported advisory-baseline issues: {hits}"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "malformed",
        "unsupported_schema",
        "unknown_key",
        "duplicate_key",
        "duplicate_identity",
        "unsorted_identity",
        "noncanonical_bytes",
    ],
)
def test_docs_d10_dependency_advisory_baseline_fails_closed_for_invalid_policy(tmp_path, mutation):
    repo = _advisory_baseline_repo(tmp_path)
    policy = repo / "security/accepted-advisories.json"
    if mutation == "missing":
        policy.unlink()
    elif mutation == "malformed":
        policy.write_text("{", encoding="utf-8")
    elif mutation == "noncanonical_bytes":
        policy.write_bytes(policy.read_bytes() + b"\n")
    elif mutation == "duplicate_key":
        policy.write_text(
            policy.read_text(encoding="utf-8").replace(
                '  "schema_version": 1,\n',
                '  "schema_version": 1,\n  "schema_version": 1,\n',
                1,
            ),
            encoding="utf-8",
        )
    else:
        document = json.loads(policy.read_text(encoding="utf-8"))
        if mutation == "unsupported_schema":
            document["schema_version"] = 2
        elif mutation == "unknown_key":
            document["unexpected"] = True
        elif mutation == "duplicate_identity":
            document["accepted_advisories"].append(document["accepted_advisories"][0])
        else:
            document["accepted_advisories"].reverse()
        _write_canonical_baseline(repo, document)

    assert _d10_advisory_baseline_findings(repo)


def test_docs_d10_dependency_advisory_baseline_converts_import_os_error(tmp_path, monkeypatch):
    repo = _advisory_baseline_repo(tmp_path)
    original_import = builtins.__import__

    def fail_advisory_loader_import(name, *args, **kwargs):
        if name == "scripts.advisory_baseline":
            raise OSError("injected loader import failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_advisory_loader_import)

    hits = _d10_advisory_baseline_findings(repo)
    assert [hit.message for hit in hits] == ["accepted advisory baseline loader is unavailable"]


def test_docs_d10_dependency_advisory_baseline_converts_loader_os_error(tmp_path, monkeypatch):
    repo = _advisory_baseline_repo(tmp_path)
    from scripts import advisory_baseline

    def fail_load_baseline(_path):
        raise OSError("injected policy read failure")

    monkeypatch.setattr(advisory_baseline, "load_baseline", fail_load_baseline)

    hits = _d10_advisory_baseline_findings(repo)
    assert len(hits) == 1
    assert hits[0].id == "D10.dependency_advisory_baseline"
    assert hits[0].message == "accepted advisory baseline is invalid: injected policy read failure"


@pytest.mark.parametrize("with_infra", [False, True])
def test_docs_d10_dependency_advisory_baseline_flags_missing_ledger(tmp_path, with_infra):
    repo = _temp_repo(tmp_path)
    (repo / "security").mkdir()
    shutil.copyfile(
        REPO / "security/accepted-advisories.json",
        repo / "security/accepted-advisories.json",
    )
    if with_infra:
        (repo / "infra").mkdir()

    hits = _d10_advisory_baseline_findings(repo)
    assert len(hits) == 1
    assert hits[0].message == "current accepted-advisories section is missing"


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("missing", "current accepted-advisories section is missing"),
        ("duplicate", "current accepted-advisories heading must appear exactly once; found 2"),
    ],
)
def test_docs_d10_dependency_advisory_baseline_flags_invalid_current_heading(
    tmp_path, heading, expected
):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    marker = _ISSUE63_LEDGER_MARKER
    if heading == "missing":
        text = text.replace(marker, "### 6.1.1.2 Historical advisories", 1)
    else:
        text += f"\n{marker}\n\nDuplicate.\n"
    ledger.write_text(text, encoding="utf-8")

    hits = _d10_advisory_baseline_findings(repo)
    assert len(hits) == 1
    assert hits[0].message == expected


def test_docs_d10_flags_baseline_advisory_id_drift_dependency_advisory_baseline(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    document = json.loads((repo / "security/accepted-advisories.json").read_text())
    item = next(
        entry for entry in document["accepted_advisories"]
        if entry["package"] == "setuptools"
    )
    item["advisory_id"] = "PYSEC-2099-1"
    _write_canonical_baseline(repo, document)

    assert [finding.message for finding in _d10_advisory_baseline_findings(repo)] == [
        "accepted advisory baseline identity is missing from the current Markdown ledger: "
        "setuptools 81.0.0 PYSEC-2099-1 on "
        "[combined-runtime, torch, documentation, atlas-contract]",
        "current Markdown ledger identity is missing from accepted advisory baseline JSON: "
        "setuptools 81.0.0 PYSEC-2026-3447 on "
        "[combined-runtime, torch, documentation, atlas-contract]",
    ]


def test_docs_d10_flags_baseline_package_drift_dependency_advisory_baseline(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    document = json.loads((repo / "security/accepted-advisories.json").read_text())
    item = next(
        entry for entry in document["accepted_advisories"]
        if entry["package"] == "setuptools"
    )
    item["package"] = "lightning"
    document["accepted_advisories"].sort(
        key=lambda entry: (
            entry["package"], entry["accepted_version"], entry["advisory_id"]
        )
    )
    _write_canonical_baseline(repo, document)

    assert any(
        "accepted advisory baseline identity is missing from the current Markdown ledger: "
        "lightning 81.0.0 PYSEC-2026-3447" in finding.message
        for finding in _d10_advisory_baseline_findings(repo)
    )


def test_docs_d10_flags_baseline_accepted_version_drift_dependency_advisory_baseline(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    document = json.loads((repo / "security/accepted-advisories.json").read_text())
    item = next(
        entry for entry in document["accepted_advisories"]
        if entry["package"] == "setuptools"
    )
    item["accepted_version"] = "9.9.9"
    _write_canonical_baseline(repo, document)

    assert any(
        "accepted advisory baseline identity is missing from the current Markdown ledger: "
        "setuptools 9.9.9 PYSEC-2026-3447" in finding.message
        for finding in _d10_advisory_baseline_findings(repo)
    )


def test_docs_d10_flags_baseline_surface_drift_dependency_advisory_baseline(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    document = json.loads((repo / "security/accepted-advisories.json").read_text())
    item = next(
        entry for entry in document["accepted_advisories"]
        if entry["package"] == "setuptools"
    )
    item["surfaces"] = ["torch"]
    _write_canonical_baseline(repo, document)

    assert any(
        "accepted advisory baseline identity is missing from the current Markdown ledger: "
        "setuptools 81.0.0 PYSEC-2026-3447 on [torch]" == finding.message
        for finding in _d10_advisory_baseline_findings(repo)
    )


def test_docs_d10_collapses_duplicate_raw_rows_for_dependency_advisory_baseline_identity_parity(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    assert _d10_advisory_baseline_findings(repo) == []


def test_docs_d10_excludes_historical_rows_from_dependency_advisory_baseline_parity(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    ledger.write_text(
        ledger.read_text(encoding="utf-8")
        + "\n### Historical records\n\n"
        + "| Package | Advisory ID | Feed Records | Fix Versions | Audited Version | Aliases | Surface |\n"
        + "| --- | --- | ---: | --- | ---: | --- | --- |\n"
        + "| `nltk` | `PYSEC-2099-1` | 1 | None listed | `3.10.3` | None listed | Combined runtime |\n",
        encoding="utf-8",
    )

    assert _d10_advisory_baseline_findings(repo) == []


def test_docs_d10_dependency_advisory_baseline_flags_unknown_current_surface(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    _replace_issue62_section(
        repo,
        lambda section: section.replace(
            "Combined runtime; Torch |",
            "Combined runtime; Unknown |",
            1,
        ),
    )

    assert _d10_advisory_baseline_findings(repo)


def test_docs_d10_dependency_advisory_baseline_flags_noncanonical_surface_order(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    section = _issue62_section(ledger.read_text(encoding="utf-8"))
    canonical_count = section.count("Combined runtime; Torch")
    assert canonical_count > 0

    _replace_issue62_section(
        repo,
        lambda current: current.replace(
            "Combined runtime; Torch",
            "Torch; Combined runtime",
        ),
    )

    hits = _d10_advisory_baseline_findings(repo)
    assert len(hits) == 1
    assert "malformed" in hits[0].message
    assert _issue62_section(ledger.read_text(encoding="utf-8")).count(
        "Torch; Combined runtime"
    ) == canonical_count


@pytest.mark.parametrize("surface", ["Combined runtime; Combined runtime", "Combined runtime; "])
def test_docs_d10_dependency_advisory_baseline_flags_duplicate_or_empty_current_surface(tmp_path, surface):
    repo = _advisory_baseline_repo(tmp_path)
    _replace_issue62_section(
        repo,
        lambda section: section.replace("Combined runtime; Torch", surface, 1),
    )

    hits = _d10_advisory_baseline_findings(repo)
    assert len(hits) == 1
    assert "malformed" in hits[0].message


def test_docs_d10_dependency_advisory_baseline_reports_markdown_only_identity(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    line = (
        "| `torch` | `PYSEC-2099-1` | 1 | None listed | `2.11.0` | None listed | "
        "Combined runtime; Torch |"
    )
    final_row = (
        "| `torch` | `PYSEC-2025-194` | 1 | `2.13.0` | `2.11.0` | "
        "`BIT-pytorch-2025-3000`, `CVE-2025-3000`, `GHSA-rrmf-rvhw-rf47` | Combined runtime; Torch |"
    )
    ledger.write_text(
        ledger.read_text(encoding="utf-8").replace(final_row, f"{final_row}\n{line}"),
        encoding="utf-8",
    )

    messages = [finding.message for finding in _d10_advisory_baseline_findings(repo)]
    assert messages == [
        "current Markdown ledger identity is missing from accepted advisory baseline JSON: "
        "torch 2.11.0 PYSEC-2099-1 on [combined-runtime, torch]"
    ]


def test_docs_d10_dependency_advisory_baseline_reports_policy_only_identity(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    final_row = (
        "| `torch` | `PYSEC-2025-194` | 1 | `2.13.0` | `2.11.0` | "
        "`BIT-pytorch-2025-3000`, `CVE-2025-3000`, `GHSA-rrmf-rvhw-rf47` | Combined runtime; Torch |\n"
    )
    ledger.write_text(ledger.read_text(encoding="utf-8").replace(final_row, ""), encoding="utf-8")

    messages = [finding.message for finding in _d10_advisory_baseline_findings(repo)]
    assert messages == [
        "accepted advisory baseline identity is missing from the current Markdown ledger: "
        "torch 2.11.0 PYSEC-2025-194 on [combined-runtime, torch]"
    ]


def test_docs_d10_html_comment_does_not_satisfy_advisory_baseline(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    ledger.write_text(f"<!--\n{ledger.read_text(encoding='utf-8')}\n-->\n", encoding="utf-8")

    hits = _d10_advisory_baseline_findings(repo)
    assert len(hits) == 1
    assert hits[0].message == "current accepted-advisories section is missing"


@pytest.mark.parametrize("tag", ["script", "style", "pre", "textarea"])
def test_docs_d10_raw_html_block_hides_the_only_advisory_snapshot(tmp_path, tag):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    snapshot = _current_advisory_snapshot(ledger.read_text(encoding="utf-8"))
    ledger.write_text(f"<{tag}>\n{snapshot}\n</{tag}>\n", encoding="utf-8")

    hits = _d10_advisory_baseline_findings(repo)
    assert [hit.message for hit in hits] == ["current accepted-advisories section is missing"]


@pytest.mark.parametrize("tag", ["script", "style", "pre", "textarea"])
def test_docs_d10_raw_html_block_hides_a_duplicate_advisory_snapshot(tmp_path, tag):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(f"{text}\n<{tag}>\n{_current_advisory_snapshot(text)}\n</{tag}>\n", encoding="utf-8")

    assert _d10_advisory_baseline_findings(repo) == []


@pytest.mark.parametrize("opener", ["<script", '<script type="x"'])
def test_docs_d10_partial_raw_html_opener_hides_the_only_advisory_snapshot(tmp_path, opener):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    snapshot = _current_advisory_snapshot(ledger.read_text(encoding="utf-8"))
    ledger.write_text(f"{opener}\n{snapshot}\n</script>\n", encoding="utf-8")

    hits = _d10_advisory_baseline_findings(repo)
    assert [hit.message for hit in hits] == ["current accepted-advisories section is missing"]


@pytest.mark.parametrize("opener", ["<script", '<script type="x"'])
def test_docs_d10_partial_raw_html_opener_hides_a_duplicate_advisory_snapshot(tmp_path, opener):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(f"{text}\n{opener}\n{_current_advisory_snapshot(text)}\n</script>\n", encoding="utf-8")

    assert _d10_advisory_baseline_findings(repo) == []


def test_docs_d10_partial_raw_html_opener_preserves_visible_duplicate_control(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(
        f"<script\nhidden\n</script>\n{text}\n{_current_advisory_snapshot(text)}\n",
        encoding="utf-8",
    )

    hits = _d10_advisory_baseline_findings(repo)
    assert [hit.message for hit in hits] == [
        "current accepted-advisories heading must appear exactly once; found 2"
    ]


def test_docs_d10_four_space_indented_partial_raw_html_is_not_an_opener(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    snapshot = _current_advisory_snapshot(ledger.read_text(encoding="utf-8"))
    ledger.write_text(f"    <script\n{snapshot}\n", encoding="utf-8")

    assert _d10_advisory_baseline_findings(repo) == []


def test_docs_d10_visible_advisory_snapshot_still_counts_after_raw_html_block_control(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(
        f"<script>hidden</script>\n{text}\n{_current_advisory_snapshot(text)}\n",
        encoding="utf-8",
    )

    hits = _d10_advisory_baseline_findings(repo)
    assert [hit.message for hit in hits] == [
        "current accepted-advisories heading must appear exactly once; found 2"
    ]


@pytest.mark.parametrize(
    "fence",
    [
        "````markdown\n```\n{snapshot}\n",
        "```markdown\n~~~\n{snapshot}\n",
    ],
)
def test_docs_d10_unclosed_or_mismatched_fence_hides_advisory_snapshot(tmp_path, fence):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    marker = _ISSUE63_LEDGER_MARKER
    snapshot = marker + text.split(marker, 1)[1]
    ledger.write_text(fence.format(snapshot=snapshot), encoding="utf-8")

    hits = _d10_advisory_baseline_findings(repo)
    assert len(hits) == 1
    assert hits[0].message == "current accepted-advisories section is missing"


def _current_advisory_snapshot(text: str) -> str:
    marker = _ISSUE63_LEDGER_MARKER
    return marker + text.split(marker, 1)[1]


def test_docs_d10_comment_opener_inside_fence_cannot_mask_later_duplicate_heading(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(
        f"{text}\n```markdown\n<!--\n```\n{_current_advisory_snapshot(text)}\n-->\n",
        encoding="utf-8",
    )

    hits = _d10_advisory_baseline_findings(repo)
    assert [hit.message for hit in hits] == [
        "current accepted-advisories heading must appear exactly once; found 2"
    ]


def test_docs_d10_comment_opener_inside_inline_code_cannot_mask_later_duplicate_heading(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(
        f"{text}\n`<!--`\n{_current_advisory_snapshot(text)}\n-->\n",
        encoding="utf-8",
    )

    hits = _d10_advisory_baseline_findings(repo)
    assert [hit.message for hit in hits] == [
        "current accepted-advisories heading must appear exactly once; found 2"
    ]


def test_docs_d10_four_space_fence_pseudo_closer_keeps_advisory_snapshot_hidden(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    snapshot = _current_advisory_snapshot(ledger.read_text(encoding="utf-8"))
    ledger.write_text(f"````markdown\n    ````\n{snapshot}\n", encoding="utf-8")

    hits = _d10_advisory_baseline_findings(repo)
    assert [hit.message for hit in hits] == ["current accepted-advisories section is missing"]


def test_docs_d10_three_space_fence_closer_exposes_advisory_snapshot(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    snapshot = _current_advisory_snapshot(ledger.read_text(encoding="utf-8"))
    ledger.write_text(f"````markdown\n   ````\n{snapshot}\n", encoding="utf-8")

    assert _d10_advisory_baseline_findings(repo) == []


def test_docs_d10_active_comment_ignores_fence_marker_before_later_duplicate_heading(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(
        f"{text}\n<!--\n```markdown\n-->\n{_current_advisory_snapshot(text)}\n",
        encoding="utf-8",
    )

    hits = _d10_advisory_baseline_findings(repo)
    assert [hit.message for hit in hits] == [
        "current accepted-advisories heading must appear exactly once; found 2"
    ]


def test_docs_d10_comment_closing_midline_resumes_live_heading_parsing(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(
        f"{text}\n<!-- ignored --> visible text\n{_current_advisory_snapshot(text)}\n",
        encoding="utf-8",
    )

    hits = _d10_advisory_baseline_findings(repo)
    assert [hit.message for hit in hits] == [
        "current accepted-advisories heading must appear exactly once; found 2"
    ]


def test_docs_d10_multiline_inline_code_comment_delimiter_is_inert_before_duplicate_heading(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(
        f"{text}\n`\n<!--\n`\n{_current_advisory_snapshot(text)}\n-->\n",
        encoding="utf-8",
    )

    hits = _d10_advisory_baseline_findings(repo)
    assert [hit.message for hit in hits] == [
        "current accepted-advisories heading must appear exactly once; found 2"
    ]


def test_markdown_masking_multiline_inline_code_requires_equal_backtick_length():
    masked = verify_repo._strip_markdown_code(
        "``\n<!--\n`\n# hidden\n``\n# visible"
    )

    assert "# hidden" not in masked
    assert "# visible" in masked


def test_docs_d10_multiline_inline_code_hides_only_advisory_snapshot(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    snapshot = _current_advisory_snapshot(ledger.read_text(encoding="utf-8"))
    ledger.write_text(f"prefix ``\n{snapshot}\n``\n", encoding="utf-8")

    hits = _d10_advisory_baseline_findings(repo)
    assert [hit.message for hit in hits] == ["current accepted-advisories section is missing"]


def test_docs_d10_multiline_inline_code_hides_duplicate_advisory_snapshot(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(
        f"{text}\nprefix ``\n{_current_advisory_snapshot(text)}\n``\n",
        encoding="utf-8",
    )

    assert _d10_advisory_baseline_findings(repo) == []


def test_markdown_masking_multiline_inline_code_resumes_after_midline_closer():
    masked = verify_repo._strip_markdown_code(
        "prefix ``\n# hidden\n`` live suffix", strip_inline=False
    )

    assert "# hidden" not in masked
    assert masked.endswith("  live suffix")


def test_docs_d10_invalid_backtick_fence_info_leaves_duplicate_snapshot_live(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(
        f"{text}\n```bad```info\n{_current_advisory_snapshot(text)}\n",
        encoding="utf-8",
    )

    hits = _d10_advisory_baseline_findings(repo)
    assert [hit.message for hit in hits] == [
        "current accepted-advisories heading must appear exactly once; found 2"
    ]


def test_docs_d10_invalid_backtick_fence_composes_with_multiline_inline_code(tmp_path):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(
        f"{text}\n```bad`info\n<!--\n```\n{_current_advisory_snapshot(text)}\n-->\n",
        encoding="utf-8",
    )

    hits = _d10_advisory_baseline_findings(repo)
    assert [hit.message for hit in hits] == [
        "current accepted-advisories heading must appear exactly once; found 2"
    ]


@pytest.mark.parametrize("fence", ["```valid-info", "~~~valid`info"])
def test_docs_d10_valid_fence_info_hides_duplicate_snapshot(tmp_path, fence):
    repo = _advisory_baseline_repo(tmp_path)
    ledger = repo / "docs/dependency-contracts.md"
    text = ledger.read_text(encoding="utf-8")
    closer = fence[:3]
    ledger.write_text(
        f"{text}\n{fence}\n{_current_advisory_snapshot(text)}\n{closer}\n",
        encoding="utf-8",
    )

    assert _d10_advisory_baseline_findings(repo) == []


def _dependency_snapshot(*, summary_count=2, advisory_rows=None):
    rows = advisory_rows or [
        (
            "| `torch` | `PYSEC-2025-41` | 1 | `2.6.0` | `2.4.1` | "
            "`CVE-2025-32434` | Combined runtime; Torch |"
        ),
        (
            "| `torch` | `PYSEC-2025-191` | 1 | `2.7.1rc1` | `2.4.1` | "
            "`CVE-2025-2953` | Combined runtime; Torch |"
        ),
    ]
    return (
        "# 6.1 Dependency Contracts\n\n"
        "## 6.1.1 Audit Snapshot\n\n"
        "### 6.1.1.2 Current Issue #63 locked four-surface audit\n\n"
        f"Result: {summary_count} known vulnerabilities across 1 resolved package.\n\n"
        "| Package | Manifest Constraint | Audited Resolved Version | Finding Count | Current Disposition |\n"
        "| --- | --- | ---: | ---: | --- |\n"
        f"| `torch` | `torch==2.4.1` | `2.4.1` | {summary_count} | Accepted temporarily. |\n\n"
        "| Package | Advisory ID | Feed Records | Fix Versions | Audited Version | Aliases | Surface |\n"
        "| --- | --- | ---: | --- | ---: | --- | --- |\n"
        + "\n".join(rows)
        + "\n"
    )


def _d10_count_findings(tmp_path, text):
    repo = _temp_repo(tmp_path)
    docs = repo / "docs"
    docs.mkdir()
    (docs / "dependency-contracts.md").write_text(text, encoding="utf-8")
    verify_repo = _load_verify_module()
    return [
        finding
        for finding in verify_repo._dependency_ledger_findings(repo)
        if finding.id == "D10.dependency_ledger_count"
    ]


def test_docs_d10_ignores_parser_compatible_historical_rows(tmp_path):
    historical = (
        "\n### 6.1.1.3 Historical reconciliation\n\n"
        "| Package | Manifest Constraint | Audited Resolved Version | Finding Count | Disposition |\n"
        "| --- | --- | ---: | ---: | --- |\n"
        "| `nltk` | `nltk>=3.9.3` | `3.9.4` | 7 | Archived. |\n\n"
        "| Package | Advisory ID | Feed Records | Fix Versions |\n"
        "| --- | --- | ---: | --- |\n"
        "| `nltk` | `CVE-2099-9999` | 7 | none listed |\n"
    )
    assert _d10_count_findings(tmp_path, _dependency_snapshot() + historical) == []


def test_docs_d10_flags_missing_current_advisory_section(tmp_path):
    text = _dependency_snapshot().replace(
        "### 6.1.1.2 Current Issue #63 locked four-surface audit",
        "### 6.1.1.2 Historical advisories",
    )
    findings = _d10_count_findings(tmp_path, text)
    assert any("section is missing" in finding.message for finding in findings)


def test_docs_d10_flags_malformed_current_summary_table(tmp_path):
    text = _dependency_snapshot().replace(
        "| `torch` | `torch==2.4.1` | `2.4.1` | 2 | Accepted temporarily. |\n",
        "",
    )
    findings = _d10_count_findings(tmp_path, text)
    assert any("summary table" in finding.message for finding in findings)


def test_docs_d10_flags_malformed_current_advisory_table(tmp_path):
    text = _dependency_snapshot(advisory_rows=["| no parseable advisory row |"])
    findings = _d10_count_findings(tmp_path, text)
    assert any("advisory table" in finding.message for finding in findings)


def test_docs_d10_flags_headerless_current_summary_table(tmp_path):
    text = _dependency_snapshot().replace(
        "| Package | Manifest Constraint | Audited Resolved Version | Finding Count | Current Disposition |\n"
        "| --- | --- | ---: | ---: | --- |\n",
        "",
    )
    findings = _d10_count_findings(tmp_path, text)
    assert any("summary table" in finding.message for finding in findings)


def test_docs_d10_flags_headerless_current_advisory_table(tmp_path):
    text = _dependency_snapshot().replace(
        "| Package | Advisory ID | Feed Records | Fix Versions | Audited Version | Aliases | Surface |\n"
        "| --- | --- | ---: | --- | ---: | --- | --- |\n",
        "",
    )
    findings = _d10_count_findings(tmp_path, text)
    assert any("advisory table" in finding.message for finding in findings)


def test_docs_d10_flags_malformed_current_summary_separator(tmp_path):
    text = _dependency_snapshot().replace(
        "| --- | --- | ---: | ---: | --- |",
        "| --- | --- | --- | --- | --- |",
    )
    findings = _d10_count_findings(tmp_path, text)
    assert any("summary table" in finding.message for finding in findings)


def test_docs_d10_flags_malformed_current_advisory_separator(tmp_path):
    text = _dependency_snapshot().replace(
        "| --- | --- | ---: | --- | ---: | --- | --- |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    )
    findings = _d10_count_findings(tmp_path, text)
    assert any("advisory table" in finding.message for finding in findings)


def test_docs_d10_flags_duplicate_current_summary_package(tmp_path):
    row = (
        "| `torch` | `torch==2.4.1` | `2.4.1` | 2 | "
        "Accepted temporarily. |"
    )
    text = _dependency_snapshot().replace(row, f"{row}\n{row}")
    findings = _d10_count_findings(tmp_path, text)
    assert any("duplicate" in finding.message for finding in findings)


def test_docs_d10_flags_advisory_package_absent_from_summary(tmp_path):
    torch_row = (
        "| `torch` | `PYSEC-2025-191` | 1 | `2.7.1rc1` | `2.4.1` | "
        "`CVE-2025-2953` | Combined runtime; Torch |"
    )
    nltk_row = (
        "| `nltk` | `PYSEC-2099-1` | 1 | None listed | `3.10.3` | "
        "`CVE-2099-1` | Combined runtime |"
    )
    text = _dependency_snapshot().replace(torch_row, f"{torch_row}\n{nltk_row}")
    text = text.replace(
        "Result: 2 known vulnerabilities", "Result: 3 known vulnerabilities"
    )
    findings = _d10_count_findings(tmp_path, text)
    assert any("absent from audit summary" in finding.message for finding in findings)


def test_docs_d10_flags_duplicate_exact_current_heading(tmp_path):
    text = (
        _dependency_snapshot()
        + "\n### 6.1.1.2 Current Issue #63 locked four-surface audit\n\n"
        + "Duplicate current section.\n"
    )
    findings = _d10_count_findings(tmp_path, text)
    assert any("exactly once" in finding.message for finding in findings)


@pytest.mark.parametrize(
    "duplicate_heading",
        [
            "### 6.1.1.2 Current Issue #63 locked four-surface audit  \t",
            "###\t6.1.1.2  Current\tIssue  #63\tlocked  four-surface   audit",
    ],
)
def test_docs_d10_flags_semantically_duplicate_current_heading(
    tmp_path, duplicate_heading
):
    text = _dependency_snapshot() + f"\n{duplicate_heading}\n\nDuplicate section.\n"
    findings = _d10_count_findings(tmp_path, text)
    assert any("exactly once" in finding.message for finding in findings)


def test_docs_d10_rejects_current_structures_inside_fenced_code(tmp_path):
    heading = _ISSUE63_LEDGER_MARKER
    prefix, body = _dependency_snapshot().split(f"{heading}\n\n", maxsplit=1)
    text = f"{prefix}{heading}\n\n```markdown\n{body}```\n"
    findings = _d10_count_findings(tmp_path, text)
    assert any("summary table" in finding.message for finding in findings)
    assert any("advisory table" in finding.message for finding in findings)
    assert any("Result" in finding.message for finding in findings)


def test_docs_d10_ignores_current_heading_example_inside_fenced_code(tmp_path):
    text = (
        _dependency_snapshot()
        + "\n```markdown\n"
        + "### 6.1.1.2 Current Issue #63 locked four-surface audit\n"
        + "```\n"
    )
    assert _d10_count_findings(tmp_path, text) == []


def test_docs_d10_flags_current_package_count_drift(tmp_path):
    rows = [
        "| `torch` | `PYSEC-2025-41` | 1 | `2.6.0` | `2.4.1` | "
        "`CVE-2025-32434` | Combined runtime; Torch |"
    ]
    findings = _d10_count_findings(
        tmp_path, _dependency_snapshot(summary_count=2, advisory_rows=rows)
    )
    assert any("torch advisory feed-record count" in finding.message for finding in findings)


def test_docs_d10_flags_current_total_count_drift(tmp_path):
    text = _dependency_snapshot().replace(
        "Result: 2 known vulnerabilities", "Result: 3 known vulnerabilities"
    )
    findings = _d10_count_findings(tmp_path, text)
    assert any("advisory feed-record total" in finding.message for finding in findings)


def test_docs_d10_flags_missing_current_total(tmp_path):
    text = _dependency_snapshot().replace(
        "Result: 2 known vulnerabilities across 1 resolved package.\n\n", ""
    )
    findings = _d10_count_findings(tmp_path, text)
    assert any("Result" in finding.message for finding in findings)


@pytest.mark.parametrize(
    "duplicate_total",
    [
        "Result: 2 known vulnerabilities across 1 resolved package.",
        "Result: 3 known vulnerabilities across 1 resolved package.",
    ],
)
def test_docs_d10_flags_duplicate_current_total(tmp_path, duplicate_total):
    text = _dependency_snapshot().replace(
        "Result: 2 known vulnerabilities across 1 resolved package.",
        "Result: 2 known vulnerabilities across 1 resolved package.\n"
        f"{duplicate_total}",
    )
    findings = _d10_count_findings(tmp_path, text)
    assert any("exactly one Result" in finding.message for finding in findings)


def test_docs_d10_current_atlas_infra_gitlink_matches_ledger():
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "D10.dependency_ledger_submodule_sha"
    ]
    assert hits == [], f"D10 reported Atlas infra ledger issues: {hits}"


def test_docs_d10_flags_dependency_ledger_count_drift(tmp_path):
    """The dependency ledger should not collapse duplicated advisory feed records."""
    repo = _temp_repo(tmp_path)
    docs = repo / "docs"
    docs.mkdir()
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 1. Audit Snapshot\n\n"
        "### 6.1.1.2 Current Issue #63 locked four-surface audit\n\n"
        "Result: 2 known vulnerabilities across one resolved package:\n\n"
        "| Package | Manifest Constraint | Audited Resolved Version | Finding Count | Current Disposition |\n"
        "| --- | --- | ---: | ---: | --- |\n"
        "| `torch` | `torch==2.4.1` | `2.4.1` | 2 | Accepted temporarily. |\n\n"
        "| Package | Advisory ID | Feed Records | Fix Versions | Audited Version | Aliases | Surface |\n"
        "| --- | --- | ---: | --- | ---: | --- | --- |\n"
        "| `torch` | `PYSEC-2025-41` | 1 | `2.6.0` | `2.4.1` | "
        "`CVE-2025-32434` | Combined runtime; Torch |\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "D10.dependency_ledger_count"]
    assert hits, f"expected D10.dependency_ledger_count; got {data.get('findings')}"
    assert hits[0]["message"] == (
        "torch advisory feed-record count is 1; expected 2 from audit summary"
    )
    assert hits[0]["detail"] == {"package": "torch", "expected": 2, "actual": 1}


def test_docs_d10_flags_dependency_ledger_submodule_sha_drift(tmp_path, monkeypatch):
    """The Atlas ledger SHA should match the superproject infra gitlink."""
    repo = _temp_repo(tmp_path)
    (repo / "infra").mkdir()
    docs = repo / "docs"
    docs.mkdir()
    ledger_sha = "b96a2924b5d30aa30eddb2fa43f9b7a47fc81bcb"
    gitlink_sha = "163134451a19d024e0e1c0df51139fd8c0a2ca52"
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 7. Atlas Infra Submodule Contract\n\n"
        f"Current Atlas `infra` gitlink SHA: `{ledger_sha}`.\n",
        encoding="utf-8",
    )
    verify_repo = _load_verify_module()

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["git", "ls-files", "--stage", "--", "infra"]:
            return 0, f"160000 {gitlink_sha} 0\tinfra\n", ""
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)

    findings = verify_repo._dependency_ledger_findings(repo)

    hits = [f for f in findings if f.id == "D10.dependency_ledger_submodule_sha"]
    assert hits
    assert hits[0].detail == {"ledger_sha": ledger_sha, "gitlink_sha": gitlink_sha}


def test_docs_d10_flags_missing_dependency_ledger_submodule_sha(tmp_path, monkeypatch):
    """The Atlas ledger must keep a parseable pinned tree-entry SHA."""
    repo = _temp_repo(tmp_path)
    (repo / "infra").mkdir()
    docs = repo / "docs"
    docs.mkdir()
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 7. Atlas Infra Submodule Contract\n\n"
        "Current Atlas `infra` gitlink SHA: `not-a-sha`.\n",
        encoding="utf-8",
    )
    verify_repo = _load_verify_module()

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["git", "ls-files", "--stage", "--", "infra"]:
            return 0, "160000 ba21661e8a63b3727b9c4a14eaf5e61262d4b48e 0\tinfra\n", ""
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)

    findings = verify_repo._dependency_ledger_findings(repo)

    hits = [f for f in findings if f.id == "D10.dependency_ledger_submodule_sha"]
    assert hits
    assert "parseable" in hits[0].message


def test_docs_d10_flags_missing_dependency_ledger_gitlink(tmp_path, monkeypatch):
    """The Atlas ledger SHA must be checked against a parseable gitlink."""
    repo = _temp_repo(tmp_path)
    (repo / "infra").mkdir()
    docs = repo / "docs"
    docs.mkdir()
    ledger_sha = "ba21661e8a63b3727b9c4a14eaf5e61262d4b48e"
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 7. Atlas Infra Submodule Contract\n\n"
        f"Current Atlas `infra` gitlink SHA: `{ledger_sha}`.\n",
        encoding="utf-8",
    )
    verify_repo = _load_verify_module()

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["git", "ls-files", "--stage", "--", "infra"]:
            return 0, "", ""
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)

    findings = verify_repo._dependency_ledger_findings(repo)

    hits = [f for f in findings if f.id == "D10.dependency_ledger_submodule_sha"]
    assert hits
    assert "gitlink" in hits[0].message
    assert hits[0].detail == {"ledger_sha": ledger_sha, "gitlink_sha": None}


def test_docs_d10_requires_atlas_ledger_entry_when_infra_exists(tmp_path):
    repo = _temp_repo(tmp_path)
    (repo / "infra").mkdir()
    docs = repo / "docs"
    docs.mkdir()
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 8. Other dependency record\n\n"
        "The unrelated tree entry is `10f840252404eb5399550f96fbb560153f1a47c7`.\n",
        encoding="utf-8",
    )

    findings = _load_verify_module()._dependency_ledger_findings(repo)

    hits = [f for f in findings if f.id == "D10.dependency_ledger_submodule_sha"]
    assert len(hits) == 1
    assert "Atlas Infra Submodule Contract" in hits[0].message


def test_docs_d10_does_not_use_legacy_sha_for_malformed_atlas_entry(tmp_path, monkeypatch):
    repo = _temp_repo(tmp_path)
    (repo / "infra").mkdir()
    docs = repo / "docs"
    docs.mkdir()
    legacy_sha = "61c7c5103660e2226bf107c115dae42bf46f8374"
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 7. Atlas Infra Submodule Contract\n\n"
        "Current Atlas `infra` gitlink SHA: `not-a-sha`.\n\n"
        "## 8. Other dependency record\n\n"
        f"The repository currently pins tree entry `{legacy_sha}`.\n",
        encoding="utf-8",
    )
    module = _load_verify_module()

    def fail_if_gitlink_checked(cmd, cwd, timeout=None):
        assert cmd != ["git", "ls-files", "--stage", "--", "infra"]
        return 0, "", ""

    monkeypatch.setattr(module, "_run", fail_if_gitlink_checked)

    findings = module._dependency_ledger_findings(repo)

    hits = [f for f in findings if f.id == "D10.dependency_ledger_submodule_sha"]
    assert len(hits) == 1
    assert "parseable" in hits[0].message


def test_docs_d10_reads_atlas_sha_without_capturing_legacy_rollback_sha(tmp_path, monkeypatch):
    repo = _temp_repo(tmp_path)
    (repo / "infra").mkdir()
    docs = repo / "docs"
    docs.mkdir()
    atlas_sha = "61c7c5103660e2226bf107c115dae42bf46f8374"
    legacy_sha = "10f840252404eb5399550f96fbb560153f1a47c7"
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 7. Atlas Infra Submodule Contract\n\n"
        f"Current Atlas `infra` gitlink SHA: `{atlas_sha}`.\n\n"
        "## 8. Other dependency record\n\n"
        f"The repository currently pins tree entry `{legacy_sha}`.\n",
        encoding="utf-8",
    )
    module = _load_verify_module()

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["git", "ls-files", "--stage", "--", "infra"]:
            return 0, f"160000 {atlas_sha} 0\tinfra\n", ""
        return 0, "", ""

    monkeypatch.setattr(module, "_run", fake_run)

    findings = module._dependency_ledger_findings(repo)

    assert not [f for f in findings if f.id == "D10.dependency_ledger_submodule_sha"]


_ATLAS_CURRENT_PIN_DOCS = (
    "README.md",
    "docs/env-setup.md",
    "docs/atlas-pin-bump-runbook.md",
)


def _write_current_atlas_pin_docs(repo: Path, sha: str) -> None:
    marker = f"Current reviewed Atlas pin: `{sha}`.\n"
    for relative in _ATLAS_CURRENT_PIN_DOCS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker, encoding="utf-8")


def _atlas_pin_projection_findings(repo: Path, sha: str):
    return _load_verify_module()._atlas_current_pin_projection_findings(
        repo, gitlink_sha=sha
    )


def test_atlas_current_pin_projection_matches_gitlink(tmp_path):
    sha = "41ba856f7cd35f0b559d6875e08443eac3e98a98"
    _write_current_atlas_pin_docs(tmp_path, sha)

    assert _atlas_pin_projection_findings(tmp_path, sha) == []


@pytest.mark.parametrize("relative", _ATLAS_CURRENT_PIN_DOCS)
def test_atlas_current_pin_projection_rejects_one_stale_surface(tmp_path, relative):
    sha = "41ba856f7cd35f0b559d6875e08443eac3e98a98"
    stale = "61c7c5103660e2226bf107c115dae42bf46f8374"
    _write_current_atlas_pin_docs(tmp_path, sha)
    path = tmp_path / relative
    path.write_text(
        path.read_text(encoding="utf-8").replace(sha, stale),
        encoding="utf-8",
    )

    findings = _atlas_pin_projection_findings(tmp_path, sha)

    assert [finding.id for finding in findings] == ["D10.atlas_current_pin_projection"]
    assert findings[0].location == relative
    assert findings[0].detail == {"matches": [stale], "gitlink_sha": sha}


@pytest.mark.parametrize("mode", ("missing", "malformed", "duplicate"))
def test_atlas_current_pin_projection_rejects_invalid_marker_shape(tmp_path, mode):
    sha = "41ba856f7cd35f0b559d6875e08443eac3e98a98"
    _write_current_atlas_pin_docs(tmp_path, sha)
    path = tmp_path / "README.md"
    marker = f"Current reviewed Atlas pin: `{sha}`.\n"
    replacement = {
        "missing": "Atlas is pinned.\n",
        "malformed": "Current reviewed Atlas pin: `not-a-sha`.\n",
        "duplicate": marker + marker,
    }[mode]
    path.write_text(replacement, encoding="utf-8")

    findings = _atlas_pin_projection_findings(tmp_path, sha)

    assert [finding.location for finding in findings] == ["README.md"]
    assert findings[0].id == "D10.atlas_current_pin_projection"


def test_atlas_current_pin_projection_rejects_partial_document_set(tmp_path):
    sha = "41ba856f7cd35f0b559d6875e08443eac3e98a98"
    (tmp_path / "README.md").write_text(
        f"Current reviewed Atlas pin: `{sha}`.\n", encoding="utf-8"
    )

    findings = _atlas_pin_projection_findings(tmp_path, sha)

    assert [finding.location for finding in findings] == [
        "docs/env-setup.md",
        "docs/atlas-pin-bump-runbook.md",
    ]


def test_atlas_current_pin_projection_ignores_unrelated_minimal_fixture(tmp_path):
    assert _atlas_pin_projection_findings(
        tmp_path, "41ba856f7cd35f0b559d6875e08443eac3e98a98"
    ) == []


def test_real_atlas_current_pin_projections_match_staged_gitlink():
    gitlink = subprocess.run(
        ["git", "rev-parse", ":infra"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    assert _atlas_pin_projection_findings(REPO_ROOT, gitlink) == []


def test_docs_d10_flags_workflow_action_refs_that_are_not_sha_pinned(tmp_path):
    repo = _temp_repo(tmp_path)
    workflow = repo / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n"
        "  test:\n"
        "    steps:\n"
        "      - uses: actions/checkout@v7\n",
        encoding="utf-8",
    )
    docs = repo / "docs"
    docs.mkdir()
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 8. GitHub Actions Pins\n\n"
        "| Action | Reviewed Tag | Pinned SHA |\n"
        "| --- | --- | --- |\n"
        "| `actions/checkout` | `v7` | `9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0` |\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "D10.workflow_action_pin"]
    assert hits, f"expected D10.workflow_action_pin; got {data.get('findings')}"
    assert "actions/checkout@v7" in hits[0]["message"]


def test_docs_d10_flags_yaml_workflow_action_refs_that_are_not_sha_pinned(tmp_path):
    repo = _temp_repo(tmp_path)
    workflow = repo / ".github" / "workflows" / "ci.yaml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n"
        "  test:\n"
        "    steps:\n"
        "      - uses: actions/checkout@v7\n",
        encoding="utf-8",
    )
    docs = repo / "docs"
    docs.mkdir()
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 8. GitHub Actions Pins\n\n"
        "| Action | Reviewed Tag | Pinned SHA |\n"
        "| --- | --- | --- |\n"
        "| `actions/checkout` | `v7` | `9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0` |\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "D10.workflow_action_pin"]
    assert hits, f"expected D10.workflow_action_pin for .yaml workflow; got {data.get('findings')}"
    assert hits[0]["location"] == ".github/workflows/ci.yaml:4"


def test_docs_d10_flags_workflow_action_refs_missing_from_ledger(tmp_path):
    repo = _temp_repo(tmp_path)
    workflow = repo / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n"
        "  test:\n"
        "    steps:\n"
        "      - uses: actions/cache@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa # v5\n",
        encoding="utf-8",
    )
    docs = repo / "docs"
    docs.mkdir()
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 8. GitHub Actions Pins\n\n"
        "| Action | Reviewed Tag | Pinned SHA |\n"
        "| --- | --- | --- |\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "D10.workflow_action_pin"]
    assert hits, f"expected D10.workflow_action_pin; got {data.get('findings')}"
    assert "ledger" in hits[0]["message"]


def test_docs_d10_current_workflow_action_pins_match_ledger():
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "D10.workflow_action_pin"]
    assert hits == []


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


def test_execution_e5_uses_versioned_public_facade_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    expected_tag = "tier-c-deterministic-seeding-atlas-baseline-2026-08-22"
    expected_ref = f"refs/tags/{expected_tag}"
    expected_commit = "35e7903afe45f60e5e30bf8fbd49f7d6463caa6a"
    assert verify_repo.TIER_C_CODE_BASELINE_TAG == expected_tag
    assert verify_repo.TIER_C_CODE_BASELINE_COMMIT == expected_commit

    repo = tmp_path / "repo"
    notebook = (
        repo
        / "notebooks"
        / "node_classification-reddit-gnn-pyg"
        / "phase3-example.ipynb"
    )
    notebook.parent.mkdir(parents=True)
    notebook_text = json.dumps({
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "id": "public-facade-baseline",
            "metadata": {},
            "outputs": [],
            "source": ["from nnx import NNModel\n"],
        }],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    })
    notebook.write_text(notebook_text, encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, timeout=None):
        del cwd, timeout
        calls.append(cmd)
        if cmd == ["git", "cat-file", "-t", expected_ref]:
            return 0, "tag\n", ""
        if cmd == ["git", "rev-parse", "--verify", f"{expected_ref}^{{}}"]:
            return 0, f"{expected_commit}\n", ""
        if cmd[:2] == ["git", "show"]:
            return 0, notebook_text, ""
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(verify_repo, "_run", fake_run)

    assert verify_repo._phase3_code_cells_unchanged(repo) == []
    assert calls == [
        ["git", "cat-file", "-t", expected_ref],
        ["git", "rev-parse", "--verify", f"{expected_ref}^{{}}"],
        [
            "git",
            "show",
            f"{expected_ref}:notebooks/node_classification-reddit-gnn-pyg/phase3-example.ipynb",
        ],
    ]


def test_execution_e5_missing_baseline_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        verify_repo,
        "_run",
        lambda cmd, cwd, timeout=None: (128, "", "missing tag"),
    )

    findings = verify_repo._phase3_code_cells_unchanged(tmp_path)

    assert [(finding.id, finding.severity) for finding in findings] == [
        ("E5.no_baseline", "error"),
    ]


@pytest.mark.parametrize(
    ("object_type", "peeled_commit", "finding_id"),
    [
        ("commit", "35e7903afe45f60e5e30bf8fbd49f7d6463caa6a", "E5.baseline_not_annotated"),
        ("tag", "0" * 40, "E5.baseline_target_changed"),
    ],
)
def test_execution_e5_rejects_non_annotated_or_moved_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    object_type: str,
    peeled_commit: str,
    finding_id: str,
):
    expected_ref = f"refs/tags/{verify_repo.TIER_C_CODE_BASELINE_TAG}"

    def fake_run(cmd, cwd, timeout=None):
        del cwd, timeout
        if cmd == ["git", "cat-file", "-t", expected_ref]:
            return 0, f"{object_type}\n", ""
        if cmd == ["git", "rev-parse", "--verify", f"{expected_ref}^{{}}"]:
            return 0, f"{peeled_commit}\n", ""
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(verify_repo, "_run", fake_run)

    findings = verify_repo._phase3_code_cells_unchanged(tmp_path)

    assert [(finding.id, finding.severity) for finding in findings] == [
        (finding_id, "error"),
    ]


def test_canonical_docs_teach_public_nnx_imports_and_current_tier_c_baseline():
    import_guides = (
        "README.md",
        "CONTRIBUTING.md",
        "docs/conventions.md",
        "docs/nnx-library.md",
    )
    for relative in import_guides:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "from nnx.X import Y" not in text, relative
        assert "from nnx import" in text, relative

    baseline_guides = (
        "CONTRIBUTING.md",
        "docs/conventions.md",
        "docs/env-setup.md",
        "docs/notebooks/node_classification-reddit-gnn-pyg.md",
    )
    for relative in baseline_guides:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert verify_repo.TIER_C_CODE_BASELINE_TAG in text, relative

    conventions = (REPO_ROOT / "docs/conventions.md").read_text(encoding="utf-8")
    assert "historical `pre-cleanup-baseline`" in conventions
    assert verify_repo.TIER_C_CODE_BASELINE_COMMIT in conventions
    assert "fails closed" in conventions


def test_runtime_available_requires_pyg_extension_stack(monkeypatch):
    """Full notebook execution needs the PyG binary extension stack, not just torch_geometric."""
    verify_repo = _load_verify_module()
    present = {"torch", "torch_geometric"}

    def fake_find_spec(name):
        return object() if name in present else None

    monkeypatch.setattr(verify_repo.importlib.util, "find_spec", fake_find_spec)

    assert verify_repo._runtime_available() is False


def _torch_runtime_import_names(repo: Path) -> set[str]:
    tree = ast.parse(
        (repo / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
    )
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "IMPORTS"
            for target in node.targets
        )
    )
    imports = ast.literal_eval(assignment.value)
    assert isinstance(imports, dict)
    return set(imports.values())


def test_issue62_runtime_availability_uses_only_supported_graph_modules():
    required = {"pyg_lib", "torch_scatter", "torch_sparse", "torch_geometric"}
    forbidden = {"torch_cluster", "torch_spline_conv"}
    assert required <= _torch_runtime_import_names(REPO_ROOT)
    assert forbidden.isdisjoint(_torch_runtime_import_names(REPO_ROOT))
    assert verify_repo._RUNTIME_AVAILABLE_IMPORTS == (
        "torch", "torch_geometric", "pyg_lib", "torch_scatter", "torch_sparse",
    )


def _copied_torch_runtime_contract_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for relative in (
        "scripts/verify_torch_stack.py",
        "scripts/verify_repo.py",
        ".github/workflows/ci.yml",
        "Dockerfile",
    ):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((REPO_ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")
    return repo


def _torch_runtime_contract_findings(repo: Path):
    return [
        finding for finding in verify_repo.check_docs(repo).findings
        if finding.id == "D10.torch_runtime_contract"
    ]


def test_torch_runtime_contract_clean_control_has_no_production_finding(tmp_path: Path) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)

    assert _torch_runtime_contract_findings(repo) == []


@pytest.mark.parametrize("forbidden", ("torch_cluster", "torch_spline_conv"))
@pytest.mark.parametrize("declaration", ("IMPORTS", "_RUNTIME_ONLY_MODULES", "_RUNTIME_AVAILABLE_IMPORTS"))
def test_torch_runtime_contract_rejects_legacy_declaration_mutations(
    tmp_path: Path, forbidden: str, declaration: str,
) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    source_path = repo / ("scripts/verify_torch_stack.py" if declaration == "IMPORTS" else "scripts/verify_repo.py")
    source = source_path.read_text(encoding="utf-8")
    anchor = '    "torch-sparse": "torch_sparse",' if declaration == "IMPORTS" else '    "torch_sparse",'
    replacement = anchor + f'\n    "{forbidden}",'
    mutated = source.replace(anchor, replacement, 1)
    assert mutated != source
    source_path.write_text(mutated, encoding="utf-8")

    assert _torch_runtime_contract_findings(repo)


@pytest.mark.parametrize("required", ("torch_geometric", "pyg_lib", "torch_scatter", "torch_sparse"))
@pytest.mark.parametrize("declaration", ("IMPORTS", "_RUNTIME_ONLY_MODULES", "_RUNTIME_AVAILABLE_IMPORTS"))
def test_torch_runtime_contract_rejects_missing_required_modules(
    tmp_path: Path, required: str, declaration: str,
) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    source_path = repo / ("scripts/verify_torch_stack.py" if declaration == "IMPORTS" else "scripts/verify_repo.py")
    source = source_path.read_text(encoding="utf-8")
    anchor = f'    "{required.replace("_", "-")}": "{required}",'
    if declaration != "IMPORTS":
        anchor = f'    "{required}",'
    mutated = source.replace(anchor + "\n", "", 1)
    assert mutated != source
    source_path.write_text(mutated, encoding="utf-8")

    assert _torch_runtime_contract_findings(repo)


def _mutate_ci_run_command(repo: Path, command: str) -> None:
    target = repo / ".github/workflows/ci.yml"
    source = target.read_text(encoding="utf-8")
    anchor = "      - name: Install dependencies\n        run: |\n"
    mutated = source.replace(anchor, anchor + f"          {command}\n", 1)
    assert mutated != source
    target.write_text(mutated, encoding="utf-8")


def _mutate_docker_run_instruction(repo: Path, command: str) -> None:
    target = repo / "Dockerfile"
    source = target.read_text(encoding="utf-8")
    anchor = 'RUN /opt/conda/bin/python -m venv "$VIRTUAL_ENV" \\\n'
    mutated = source.replace(
        anchor,
        f"RUN {command} \\\n  && /opt/conda/bin/python -m venv \"$VIRTUAL_ENV\" \\\n",
        1,
    )
    assert mutated != source
    target.write_text(mutated, encoding="utf-8")


def test_torch_runtime_contract_ignores_historical_python_command_comments(tmp_path: Path) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    _mutate_ci_run_command(repo, '# historical: python -c "import torch_cluster"')
    _mutate_docker_run_instruction(repo, '# historical: python -c "import torch_spline_conv"')

    assert _torch_runtime_contract_findings(repo) == []


@pytest.mark.parametrize(("mutator", "forbidden"), (
    (_mutate_ci_run_command, "torch_cluster"),
    (_mutate_docker_run_instruction, "torch_spline_conv"),
))
def test_torch_runtime_contract_rejects_valid_ci_and_docker_commands(
    tmp_path: Path, mutator, forbidden: str,
) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    mutator(repo, f'python -c "import {forbidden}"')

    assert _torch_runtime_contract_findings(repo)


@pytest.mark.parametrize(("mutator", "command"), (
    (_mutate_ci_run_command, 'python -c "import torch_cluster'),
    (_mutate_docker_run_instruction, 'python -c "import ("'),
))
def test_torch_runtime_contract_rejects_malformed_python_candidates(
    tmp_path: Path, mutator, command: str,
) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    mutator(repo, command)

    assert _torch_runtime_contract_findings(repo)


@pytest.mark.parametrize("name", ("IMPORTS", "_RUNTIME_ONLY_MODULES", "_RUNTIME_AVAILABLE_IMPORTS"))
def test_torch_runtime_contract_rejects_rebound_declarations(tmp_path: Path, name: str) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    target = repo / ("scripts/verify_torch_stack.py" if name == "IMPORTS" else "scripts/verify_repo.py")
    target.write_text(
        target.read_text(encoding="utf-8") + f"\n{name} = {name}\n",
        encoding="utf-8",
    )

    assert _torch_runtime_contract_findings(repo)


def test_torch_runtime_contract_rejects_nonliteral_declaration(tmp_path: Path) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    target = repo / "scripts/verify_repo.py"
    source = target.read_text(encoding="utf-8")
    start = source.index("_RUNTIME_AVAILABLE_IMPORTS = (")
    end = source.index("\n)\n_TORCH_RUNTIME_IMPORTS", start) + 2
    mutated = source[:start] + "_RUNTIME_AVAILABLE_IMPORTS = runtime_names()" + source[end:]
    target.write_text(mutated, encoding="utf-8")

    assert _torch_runtime_contract_findings(repo)


@pytest.mark.parametrize("mutation", (
    "\nif True:\n    IMPORTS = IMPORTS\n",
    "\nif (_RUNTIME_ONLY_MODULES := _RUNTIME_ONLY_MODULES):\n    pass\n",
    "\ndef _shadow(_RUNTIME_AVAILABLE_IMPORTS):\n    return _RUNTIME_AVAILABLE_IMPORTS\n",
))
def test_torch_runtime_contract_rejects_nested_and_nonplain_rebindings(
    tmp_path: Path, mutation: str,
) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    target = repo / ("scripts/verify_torch_stack.py" if "IMPORTS =" in mutation else "scripts/verify_repo.py")
    target.write_text(target.read_text(encoding="utf-8") + mutation, encoding="utf-8")

    assert _torch_runtime_contract_findings(repo)


@pytest.mark.parametrize("replacement", (
    "[\n    \"torch\", \"torch_geometric\", \"pyg_lib\", \"torch_scatter\", \"torch_sparse\",\n]",
    "(\n    \"torch\", \"torch_geometric\", \"pyg_lib\", \"torch_scatter\", \"torch_sparse\", \"torch\",\n)",
    "(\n    \"torch_sparse\", \"torch_scatter\", \"pyg_lib\", \"torch_geometric\", \"torch\",\n)",
))
def test_torch_runtime_contract_requires_exact_available_import_tuple(
    tmp_path: Path, replacement: str,
) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    target = repo / "scripts/verify_repo.py"
    source = target.read_text(encoding="utf-8")
    start = source.index("_RUNTIME_AVAILABLE_IMPORTS = ")
    end = source.index("\n_TORCH_RUNTIME_IMPORTS", start)
    mutated = source[:start] + f"_RUNTIME_AVAILABLE_IMPORTS = {replacement}" + source[end:]
    target.write_text(mutated, encoding="utf-8")

    assert _torch_runtime_contract_findings(repo)


@pytest.mark.parametrize(("target_name", "mutation"), (
    ("IMPORTS", "IMPORTS['legacy'] = 'torch_cluster'"),
    ("IMPORTS", "IMPORTS.update({'legacy': 'torch_cluster'})"),
    ("IMPORTS", "del IMPORTS"),
    ("IMPORTS", "del IMPORTS['torch']"),
    ("IMPORTS", "IMPORTS['torch'] += '_legacy'"),
    ("_RUNTIME_ONLY_MODULES", "_RUNTIME_ONLY_MODULES.add('torch_cluster')"),
    ("_RUNTIME_ONLY_MODULES", "_RUNTIME_ONLY_MODULES.__setitem__(0, 'torch_cluster')"),
    ("_RUNTIME_AVAILABLE_IMPORTS", "del _RUNTIME_AVAILABLE_IMPORTS[0]"),
    ("_RUNTIME_AVAILABLE_IMPORTS", "_RUNTIME_AVAILABLE_IMPORTS.append('torch_cluster')"),
    ("_RUNTIME_AVAILABLE_IMPORTS", "_RUNTIME_AVAILABLE_IMPORTS.value = 'torch_cluster'"),
))
def test_torch_runtime_contract_rejects_executable_declaration_mutations(
    tmp_path: Path, target_name: str, mutation: str,
) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    target = repo / ("scripts/verify_torch_stack.py" if target_name == "IMPORTS" else "scripts/verify_repo.py")
    target.write_text(target.read_text(encoding="utf-8") + f"\n{mutation}\n", encoding="utf-8")

    assert _torch_runtime_contract_findings(repo)


@pytest.mark.parametrize(("target_name", "mutation"), (
    ("IMPORTS", "IMPORTS.__ior__({'legacy': 'torch_cluster'})"),
    ("IMPORTS", "IMPORTS.__class__.__ior__(IMPORTS, {'legacy': 'torch_cluster'})"),
    ("IMPORTS", "dict.__ior__(IMPORTS, {'legacy': 'torch_cluster'})"),
    ("_RUNTIME_ONLY_MODULES", "_RUNTIME_ONLY_MODULES.__iand__({'torch'})"),
    ("_RUNTIME_ONLY_MODULES", "_RUNTIME_ONLY_MODULES.__isub__({'torch'})"),
    ("_RUNTIME_ONLY_MODULES", "_RUNTIME_ONLY_MODULES.__ixor__({'torch'})"),
    ("_RUNTIME_ONLY_MODULES", "set.update(_RUNTIME_ONLY_MODULES, {'torch_cluster'})"),
    ("_RUNTIME_AVAILABLE_IMPORTS", "_RUNTIME_AVAILABLE_IMPORTS.__iadd__(('torch_cluster',))"),
    ("_RUNTIME_AVAILABLE_IMPORTS", "_RUNTIME_AVAILABLE_IMPORTS.__imul__(2)"),
    ("_RUNTIME_AVAILABLE_IMPORTS", "list.append(_RUNTIME_AVAILABLE_IMPORTS, 'torch_cluster')"),
))
def test_torch_runtime_contract_rejects_inplace_and_qualified_declaration_mutations(
    tmp_path: Path, target_name: str, mutation: str,
) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    target = repo / ("scripts/verify_torch_stack.py" if target_name == "IMPORTS" else "scripts/verify_repo.py")
    target.write_text(target.read_text(encoding="utf-8") + f"\n{mutation}\n", encoding="utf-8")

    assert _torch_runtime_contract_findings(repo)


def test_torch_runtime_contract_allows_sink_methods_with_protected_arguments(tmp_path: Path) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    target = repo / "scripts/verify_torch_stack.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + "\nclass Sink:\n"
        + "    def update(self, value):\n        return value\n"
        + "    def append(self, value):\n        return value\n"
        + "    def copy(self, value):\n        return value\n"
        + "sink = Sink()\n"
        + "sink.update(IMPORTS)\n"
        + "sink.append(IMPORTS)\n"
        + "sink.copy(IMPORTS)\n",
        encoding="utf-8",
    )

    assert _torch_runtime_contract_findings(repo) == []


@pytest.mark.parametrize(("target_name", "mutation"), (
    ("IMPORTS", "import operator\noperator.ior(IMPORTS, {'legacy': 'torch_cluster'})"),
    ("IMPORTS", "import operator as op\nop.setitem(IMPORTS, 'legacy', 'torch_cluster')"),
    ("IMPORTS", "import operator\noperator.delitem(IMPORTS, 'torch')"),
    ("_RUNTIME_ONLY_MODULES", "import operator\noperator.iand(_RUNTIME_ONLY_MODULES, {'torch'})"),
    ("_RUNTIME_AVAILABLE_IMPORTS", "import operator\noperator.iadd(_RUNTIME_AVAILABLE_IMPORTS, ('torch_cluster',))"),
))
def test_torch_runtime_contract_rejects_operator_declaration_mutators(
    tmp_path: Path, target_name: str, mutation: str,
) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    target = repo / ("scripts/verify_torch_stack.py" if target_name == "IMPORTS" else "scripts/verify_repo.py")
    target.write_text(target.read_text(encoding="utf-8") + f"\n{mutation}\n", encoding="utf-8")

    assert _torch_runtime_contract_findings(repo)


@pytest.mark.parametrize(("target_name", "mutation"), (
    ("IMPORTS", "from operator import ior\nior(IMPORTS, {'legacy': 'torch_cluster'})"),
    ("IMPORTS", "from operator import setitem as put\nput(IMPORTS, 'legacy', 'torch_cluster')"),
    ("IMPORTS", "from operator import delitem\ndelitem(IMPORTS, 'torch')"),
    ("_RUNTIME_ONLY_MODULES", "from operator import ixor\nixor(_RUNTIME_ONLY_MODULES, {'torch'})"),
    ("_RUNTIME_AVAILABLE_IMPORTS", "from operator import iconcat\niconcat(_RUNTIME_AVAILABLE_IMPORTS, ('torch_cluster',))"),
))
def test_torch_runtime_contract_rejects_from_operator_declaration_mutators(
    tmp_path: Path, target_name: str, mutation: str,
) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    target = repo / ("scripts/verify_torch_stack.py" if target_name == "IMPORTS" else "scripts/verify_repo.py")
    target.write_text(target.read_text(encoding="utf-8") + f"\n{mutation}\n", encoding="utf-8")

    assert _torch_runtime_contract_findings(repo)


def test_torch_runtime_contract_rejects_operator_star_import(tmp_path: Path) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    target = repo / "scripts/verify_torch_stack.py"
    target.write_text(
        target.read_text(encoding="utf-8") + "\nfrom operator import *\n",
        encoding="utf-8",
    )

    assert _torch_runtime_contract_findings(repo)


def test_torch_runtime_contract_allows_nonmutating_operator_functions(tmp_path: Path) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    target = repo / "scripts/verify_torch_stack.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + "\nimport operator\n"
        + "operator.getitem(IMPORTS, 'torch')\n"
        + "operator.contains(IMPORTS, 'torch')\n"
        + "operator.length_hint(IMPORTS)\n",
        encoding="utf-8",
    )

    assert _torch_runtime_contract_findings(repo) == []


def test_torch_runtime_contract_allows_nonmutating_from_operator_functions(tmp_path: Path) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    target = repo / "scripts/verify_torch_stack.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + "\nfrom operator import contains, getitem as lookup, length_hint\n"
        + "lookup(IMPORTS, 'torch')\n"
        + "contains(IMPORTS, 'torch')\n"
        + "length_hint(IMPORTS)\n",
        encoding="utf-8",
    )

    assert _torch_runtime_contract_findings(repo) == []


def test_torch_runtime_contract_allows_ordinary_declaration_reads(tmp_path: Path) -> None:
    repo = _copied_torch_runtime_contract_repo(tmp_path)
    stack_source = repo / "scripts/verify_torch_stack.py"
    stack_source.write_text(
        stack_source.read_text(encoding="utf-8")
        + "\nif 'torch' in IMPORTS:\n    selected_import = IMPORTS['torch']\n"
        + "read_import = IMPORTS.get('torch')\n",
        encoding="utf-8",
    )
    repo_source = repo / "scripts/verify_repo.py"
    repo_source.write_text(
        repo_source.read_text(encoding="utf-8")
        + "\nfor runtime_name in _RUNTIME_ONLY_MODULES:\n    pass\n"
        + "first_runtime = _RUNTIME_AVAILABLE_IMPORTS[0]\n"
        + "qualified_import_read = dict.get(IMPORTS, 'torch')\n"
        + "iterator = IMPORTS.__iter__()\n"
        + "qualified_set_read = set.isdisjoint(_RUNTIME_ONLY_MODULES, {'torch'})\n"
        + "qualified_tuple_read = tuple.count(_RUNTIME_AVAILABLE_IMPORTS, 'torch')\n",
        encoding="utf-8",
    )

    assert _torch_runtime_contract_findings(repo) == []


def test_full_execution_uses_temporary_tier_a_outputs(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    repo = _temp_repo(tmp_path)
    calls: list[list[str]] = []

    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ())
    monkeypatch.setattr(verify_repo, "REQUIRED_SECTIONS", {})
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())
    monkeypatch.setattr(verify_repo, "_phase3_code_cells_unchanged", lambda _repo: [])
    monkeypatch.setattr(verify_repo, "_runtime_available", lambda: True)

    def fake_run(cmd, cwd, timeout=None):
        del cwd, timeout
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)

    verify_repo.check_execution(repo, fast=False)

    assert ["make", "smoke-tier-a"] in calls
    assert ["make", "check-tier-a-artifacts"] in calls
    assert ["make", "check-tier-a-clean"] in calls
    assert ["make", "run-tier-a"] not in calls


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


@pytest.mark.parametrize(
    "saved_path",
    (
        "/private/tmp/worktree/runs/2026-08-22_12-00-00",
        "/runs/2026-08-22_12-00-00",
        r"C:\Users\alice\worktree\runs\2026-08-22_12-00-00",
        r"\\server\share\runs\2026-08-22_12-00-00",
    ),
)
def test_e13_flags_absolute_nnx_run_saved_paths(tmp_path, monkeypatch, saved_path):
    """NNx completion output must not preserve an execution-host run path."""
    verify_repo = _load_verify_module()
    repo = _temp_repo(tmp_path)
    active_dir = repo / "notebooks" / "active-task"
    active_dir.mkdir(parents=True)
    (active_dir / "notebook.ipynb").write_text(
        json.dumps({"outputs": [{"text": [f"Run saved to {saved_path}"]}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ("active-task",))

    result = verify_repo.check_execution(repo, fast=True)

    hits = [f for f in result.findings if f.id == "E13.stale_active_notebook_path"]
    assert [f.message for f in hits] == [
        "stale active-notebook path artifact: absolute NNx saved-run output",
    ]


@pytest.mark.parametrize(
    "saved_path",
    ("runs/2026-08-22_12-00-00", "./runs/2026-08-22_12-00-00"),
)
def test_e13_allows_portable_nnx_run_saved_paths(tmp_path, monkeypatch, saved_path):
    verify_repo = _load_verify_module()
    repo = _temp_repo(tmp_path)
    active_dir = repo / "notebooks" / "active-task"
    active_dir.mkdir(parents=True)
    (active_dir / "notebook.ipynb").write_text(
        json.dumps({"outputs": [{"text": [f"Run saved to {saved_path}"]}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ("active-task",))

    result = verify_repo.check_execution(repo, fast=True)

    hits = [f for f in result.findings if f.id == "E13.stale_active_notebook_path"]
    assert hits == []


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


def _write_valid_atlas_verifier_fixture(repo: Path) -> None:
    (repo / "compose").mkdir(exist_ok=True)
    (repo / "infra").mkdir(exist_ok=True)
    scripts = repo / "scripts"
    scripts.mkdir(exist_ok=True)
    (repo / "atlas.consumer.yml").write_text(
        "name: ml-eng-lab\n"
        "project_name: ml-eng-lab\n"
        "profile: dev\n"
        "brand:\n"
        "  name: ML Eng Lab\n"
        "env:\n"
        "  file: ./atlas.env.user\n"
        "  values:\n"
        "    BASE_PORT: auto\n"
        "    JUPYTERHUB_SOURCE: container\n"
        "    LLM_PROVIDER_SOURCE: ollama-localhost\n"
        "compose_overlays:\n"
        "  - ./compose/ml-eng-lab-atlas.yml\n",
        encoding="utf-8",
    )
    (repo / "atlas.env.user.example").write_text(
        "ML_ENG_LAB_REPO_PATH=/absolute/path/to/ml-eng-lab\n",
        encoding="utf-8",
    )
    (repo / "compose/ml-eng-lab-atlas.yml").write_text(
        "services:\n  jupyterhub: {}\n",
        encoding="utf-8",
    )
    for name in ("atlas-up.sh", "atlas-down.sh", "atlas-connect.sh"):
        script = scripts / name
        script.write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
        script.chmod(0o755)


def _prepare_atlas_execution_check(monkeypatch, module, repo: Path, active_tasks=()):
    monkeypatch.setattr(module, "ACTIVE_TASK_DIRS", tuple(active_tasks))
    monkeypatch.setattr(module, "REQUIRED_SECTIONS", {})
    monkeypatch.setattr(module, "TIER_A_NOTEBOOKS", ())
    monkeypatch.setattr(module, "_phase3_code_cells_unchanged", lambda _repo: [])

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["git", "submodule", "status", "--", "infra"]:
            return 0, " 61c7c5103660e2226bf107c115dae42bf46f8374 infra\n", ""
        if cmd == ["git", "status", "--porcelain", "--", "."]:
            assert cwd == repo / "infra"
            return 0, "", ""
        if cmd == ["which", "shellcheck"]:
            return 1, "", ""
        return 0, "", ""

    monkeypatch.setattr(module, "_run", fake_run)


@pytest.mark.parametrize(
    ("manifest_text", "message"),
    [
        (None, "missing"),
        ("name: [unterminated\n", "valid YAML"),
    ],
)
def test_e15_flags_missing_or_malformed_atlas_manifest(
    tmp_path, monkeypatch, manifest_text, message
):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    if manifest_text is None:
        (repo / "atlas.consumer.yml").unlink()
    else:
        (repo / "atlas.consumer.yml").write_text(manifest_text, encoding="utf-8")
    _prepare_atlas_execution_check(monkeypatch, module, repo)

    result = module.check_execution(repo, fast=True)

    hits = [finding for finding in result.findings if finding.id == "E15.atlas_manifest"]
    assert any(finding.location == "atlas.consumer.yml" and message in finding.message for finding in hits)


def test_e15_flags_illegal_manifest_track(tmp_path, monkeypatch):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    manifest = repo / "atlas.consumer.yml"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "track: ml-eng\n", encoding="utf-8")
    _prepare_atlas_execution_check(monkeypatch, module, repo)

    result = module.check_execution(repo, fast=True)

    hits = [finding for finding in result.findings if finding.id == "E15.atlas_manifest"]
    assert any(finding.location == "atlas.consumer.yml" and "track" in finding.message for finding in hits)


@pytest.mark.parametrize(
    "mutation",
    [
        "",
        "    LLM_PROVIDER_SOURCE: auto\n",
        "    LLM_PROVIDER_SOURCE: ollama-container-cpu\n",
        "    LLM_PROVIDER_SOURCE: ollama-container-gpu\n",
        "    COMFYUI_SOURCE: container-cpu\n",
        "    COMFYUI_SOURCE: container-gpu\n",
    ],
)
def test_e15_rejects_non_native_or_containerized_ai_sources(
    tmp_path, monkeypatch, mutation
):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    manifest = repo / "atlas.consumer.yml"
    text = manifest.read_text(encoding="utf-8")
    if mutation:
        text = text.replace("    LLM_PROVIDER_SOURCE: ollama-localhost\n", mutation)
    else:
        text = text.replace("    LLM_PROVIDER_SOURCE: ollama-localhost\n", "")
    manifest.write_text(text, encoding="utf-8")
    _prepare_atlas_execution_check(monkeypatch, module, repo)

    result = module.check_execution(repo, fast=True)

    hits = [finding for finding in result.findings if finding.id == "E15.atlas_manifest"]
    assert any(finding.location == "atlas.consumer.yml" for finding in hits)


@pytest.mark.parametrize(
    "missing_path",
    [
        "atlas.env.user.example",
        "compose/ml-eng-lab-atlas.yml",
        "scripts/atlas-up.sh",
        "scripts/atlas-down.sh",
        "scripts/atlas-connect.sh",
    ],
)
def test_e15_flags_missing_atlas_contract_files(tmp_path, monkeypatch, missing_path):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    (repo / missing_path).unlink()
    _prepare_atlas_execution_check(monkeypatch, module, repo)

    result = module.check_execution(repo, fast=True)

    hits = [finding for finding in result.findings if finding.id == "E15.atlas_manifest"]
    assert any(finding.location == missing_path and "missing" in finding.message for finding in hits)


def test_e15_flags_non_executable_atlas_lifecycle_script(tmp_path, monkeypatch):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    script = repo / "scripts/atlas-connect.sh"
    script.chmod(0o644)
    _prepare_atlas_execution_check(monkeypatch, module, repo)

    result = module.check_execution(repo, fast=True)

    hits = [finding for finding in result.findings if finding.id == "E15.atlas_manifest"]
    assert any(finding.location == "scripts/atlas-connect.sh" and "executable" in finding.message for finding in hits)


def test_e16_uses_shared_parser_for_invalid_active_task_metadata(tmp_path, monkeypatch):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    task = "atlas-task"
    (repo / "scripts/verify_repo_config.yaml").write_text(
        f"active_task_dirs: [{task}]\n",
        encoding="utf-8",
    )
    (repo / "docs/notebooks").mkdir(parents=True)
    (repo / "docs/notebooks/atlas-task.md").write_text("# Task\n", encoding="utf-8")
    (repo / "docs/manifest.yaml").write_text(
        "surfaces: [repo, site, wiki]\n"
        "numbering: baked\n"
        "sections: []\n"
        "notebooks:\n"
        f"  - task: {task}\n"
        '    number: "1"\n'
        "    family: test\n"
        "    depth: full\n"
        "    doc: docs/notebooks/atlas-task.md\n"
        "    spec: notebooks/atlas-task/docs/spec.yaml\n"
        "diagrams: []\n",
        encoding="utf-8",
    )
    spec = repo / "notebooks/atlas-task/docs/spec.yaml"
    spec.parent.mkdir(parents=True)
    spec.write_text(
        "title: Atlas task\n"
        "tier: A\n"
        "atlas:\n"
        "  executor: jupyterhub\n"
        "  default_mode: vscode-remote\n"
        "  required_services: [jupyterhub]\n"
        "  required_env: []\n"
        "  workspace_access: local\n"
        "  artifact_policy: atlas-jupyter-volume\n"
        "  constraints: []\n",
        encoding="utf-8",
    )
    _prepare_atlas_execution_check(monkeypatch, module, repo, active_tasks=(task,))

    result = module.check_execution(repo, fast=True)

    hits = [finding for finding in result.findings if finding.id == "E16.atlas_task_metadata"]
    assert len(hits) == 1
    assert hits[0].location == "notebooks/**/docs/spec.yaml"
    assert "workspace_access" in hits[0].message


def test_e17_flags_port_literal_in_integration_code_and_notebook_code_cell(
    tmp_path, monkeypatch
):
    import nbformat

    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    task = "atlas-task"
    (repo / "scripts/atlas-up.sh").write_text(
        'MLFLOW_HOST="127.0.0.1:63040"\n',
        encoding="utf-8",
    )
    notebook_path = repo / f"notebooks/{task}/notebook.ipynb"
    notebook_path.parent.mkdir(parents=True)
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell('spark = "http://localhost:63030"\n'),
    ]
    nbformat.write(notebook, notebook_path)
    _prepare_atlas_execution_check(monkeypatch, module, repo, active_tasks=(task,))

    result = module.check_execution(repo, fast=True)

    hits = [finding for finding in result.findings if finding.id == "E17.atlas_hardcoded_endpoint"]
    assert {finding.location for finding in hits} == {
        "scripts/atlas-up.sh:1",
        "notebooks/atlas-task/notebook.ipynb:cell[0]:line[1]",
    }
    assert {finding.detail["endpoint"] for finding in hits} == {
        "127.0.0.1:63040",
        "http://localhost:63030",
    }


def test_e17_excludes_docs_tests_history_notebook_prose_and_harmless_examples(
    tmp_path, monkeypatch
):
    import nbformat

    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    task = "atlas-task"
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs/example.md").write_text("Try http://localhost:63094\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests/test_example.py").write_text(
        'EXAMPLE = "http://127.0.0.1:63040"\n',
        encoding="utf-8",
    )
    (repo / "scripts/docs").mkdir()
    (repo / "scripts/docs/example.py").write_text(
        '"""Documentation example: http://localhost:63094."""\n',
        encoding="utf-8",
    )
    (repo / "scripts/atlas-integration.py").write_text(
        '"""Prose example: http://127.0.0.1:63040."""\n'
        "# Historical example: http://localhost:63094\n"
        'TEMPLATE = "http://localhost:<port>"\n',
        encoding="utf-8",
    )
    notebook_path = repo / f"notebooks/{task}/notebook.ipynb"
    notebook_path.parent.mkdir(parents=True)
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell("Try http://localhost:63094"),
        nbformat.v4.new_code_cell(
            "# Historical example: http://127.0.0.1:63040\n"
            'template = "http://localhost:<port>"\n'
        ),
    ]
    notebook.cells[1].outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="http://localhost:63094\n")
    ]
    nbformat.write(notebook, notebook_path)
    archive_path = repo / "notebooks/archive/old/notebook.ipynb"
    archive_path.parent.mkdir(parents=True)
    archived = nbformat.v4.new_notebook()
    archived.cells = [nbformat.v4.new_code_cell('url = "http://localhost:63094"\n')]
    nbformat.write(archived, archive_path)
    _prepare_atlas_execution_check(monkeypatch, module, repo, active_tasks=(task,))

    result = module.check_execution(repo, fast=True)

    assert not [
        finding for finding in result.findings if finding.id == "E17.atlas_hardcoded_endpoint"
    ]


def test_e17_ignores_docstring_after_ipython_magic_in_active_notebook(
    tmp_path, monkeypatch
):
    import nbformat

    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    task = "atlas-task"
    notebook_path = repo / f"notebooks/{task}/notebook.ipynb"
    notebook_path.parent.mkdir(parents=True)
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell(
            "%matplotlib inline\n"
            '"""Prose example: http://localhost:63094."""\n'
        ),
    ]
    nbformat.write(notebook, notebook_path)
    _prepare_atlas_execution_check(monkeypatch, module, repo, active_tasks=(task,))

    result = module.check_execution(repo, fast=True)

    assert not [
        finding for finding in result.findings if finding.id == "E17.atlas_hardcoded_endpoint"
    ]


def test_e17_checks_endpoint_after_same_line_docstring(tmp_path, monkeypatch):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    (repo / "scripts/atlas-integration.py").write_text(
        '"""Prose."""; endpoint = "http://127.0.0.1:63040"\n',
        encoding="utf-8",
    )
    _prepare_atlas_execution_check(monkeypatch, module, repo, active_tasks=())

    result = module.check_execution(repo, fast=True)

    hits = [
        finding for finding in result.findings
        if finding.id == "E17.atlas_hardcoded_endpoint"
    ]
    assert [(finding.location, finding.detail["endpoint"]) for finding in hits] == [
        ("scripts/atlas-integration.py:1", "http://127.0.0.1:63040"),
    ]


def _load_workflow(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


_SHELL_SEPARATORS = frozenset((";", "&&", "||", "|"))
_MAKE_MUTATION_TARGETS = frozenset(
    ("install-torch-stack", "codespace-setup", "nlp-assets")
)
_SHELL_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.DOTALL)
_LOCK_INPUT_MANIFESTS = (
    "requirements.txt",
    "bootstrap-requirements.txt",
    "compiler-requirements.txt",
    "nlp-model-requirements.txt",
    "torch-core-requirements.txt",
    "torch-ecosystem-requirements.txt",
    "torch-requirements.txt",
    "torch-audit-requirements.txt",
    "pyg-extension-audit-requirements.txt",
    "vulnerability-audit-requirements.txt",
    "atlas-contract-requirements.txt",
    "docs-requirements.in",
)
_STACK_CACHE_MANIFESTS = (
    "requirements/lock-policy.toml",
    "requirements/locks/bootstrap.txt",
    "requirements/locks/linux-x86_64/core.txt",
    "requirements/locks/linux-x86_64/runtime.txt",
    "requirements/locks/linux-x86_64/root.txt",
    "bootstrap-requirements.txt",
    "compiler-requirements.txt",
    "nlp-model-requirements.txt",
    "requirements.txt",
    "torch-core-requirements.txt",
    "torch-ecosystem-requirements.txt",
    "torch-requirements.txt",
    "torch-audit-requirements.txt",
    "pyg-extension-audit-requirements.txt",
    "vulnerability-audit-requirements.txt",
    "atlas-contract-requirements.txt",
    "docs-requirements.in",
    "docs-requirements.txt",
)
_ATLAS_CACHE_MANIFESTS = (
    "requirements/lock-policy.toml",
    "requirements/locks/bootstrap.txt",
    "requirements/locks/atlas-contract.txt",
    *_LOCK_INPUT_MANIFESTS,
)
_DOCS_CACHE_MANIFESTS = (
    "requirements/lock-policy.toml",
    "requirements/locks/bootstrap.txt",
    "docs-requirements.txt",
    *_LOCK_INPUT_MANIFESTS,
)
_AUDIT_CACHE_MANIFESTS = (
    "requirements/lock-policy.toml",
    "requirements/locks/bootstrap.txt",
    "requirements/locks/compiler.txt",
    "requirements/locks/audit.txt",
    "requirements/locks/atlas-contract.txt",
    "requirements/locks/darwin-arm64/core.txt",
    "requirements/locks/darwin-arm64/runtime.txt",
    "requirements/locks/darwin-arm64/root.txt",
    "requirements/locks/linux-x86_64/core.txt",
    "requirements/locks/linux-x86_64/runtime.txt",
    "requirements/locks/linux-x86_64/root.txt",
    "requirements/locks/linux-aarch64/core.txt",
    "requirements/locks/linux-aarch64/runtime.txt",
    "requirements/locks/linux-aarch64/root.txt",
    *_LOCK_INPUT_MANIFESTS,
    "docs-requirements.txt",
)


def _cache_text(paths: Sequence[str]) -> str:
    return "\n".join(paths) + "\n"


@dataclass(frozen=True)
class ShellCommand:
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    wrappers: tuple[str, ...]


def _parse_shell_command(argv: Sequence[str]) -> ShellCommand:
    tokens = list(argv)
    environment: dict[str, str] = {}
    wrappers: list[str] = []
    while tokens:
        if tokens[0] in {"sudo", "env"}:
            wrappers.append(tokens.pop(0))
            continue
        if _SHELL_ASSIGNMENT_RE.fullmatch(tokens[0]):
            name, value = tokens.pop(0).split("=", 1)
            environment[name] = value
            continue
        break
    return ShellCommand(tuple(tokens), environment, tuple(wrappers))


def _shell_commands(source: str) -> tuple[ShellCommand, ...]:
    logical = source.replace("\\\n", " ").replace("\n", ";")
    lexer = shlex.shlex(logical, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    commands: list[ShellCommand] = []
    current: list[str] = []
    for token in lexer:
        if token in _SHELL_SEPARATORS:
            if current:
                command = _parse_shell_command(current)
                if command.argv:
                    commands.append(command)
                current = []
        else:
            current.append(token)
    if current:
        command = _parse_shell_command(current)
        if command.argv:
            commands.append(command)
    return tuple(commands)


def _shell_argvs(source: str) -> tuple[tuple[str, ...], ...]:
    return tuple(command.argv for command in _shell_commands(source))


def _attribute_chain(node: ast.AST) -> tuple[str, ...]:
    names: list[str] = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    return tuple(reversed(names))


def _python_c_downloads_data(program: str) -> bool:
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return True
    return any(
        isinstance(node, ast.Call)
        and _attribute_chain(node.func) in (("nltk", "download"), ("spacy", "download"))
        for node in ast.walk(tree)
    )


def _is_package_or_data_change(argv: tuple[str, ...]) -> bool:
    if not argv:
        return False
    executable = Path(argv[0].replace("$(PYTHON)", "python")).name
    if executable in {"pip", "pip3"}:
        return len(argv) > 1 and argv[1] == "install"
    if executable == "uv":
        return len(argv) > 2 and argv[1:3] == ("pip", "install")
    if executable in {"apt", "apt-get", "conda"}:
        return "install" in argv[1:]
    if executable in {"make", "$(MAKE)"}:
        return any(
            token in _MAKE_MUTATION_TARGETS or token.startswith("install")
            for token in argv[1:]
        )
    if executable == "spacy":
        return len(argv) > 1 and argv[1] == "download"
    if executable == "nltk":
        return len(argv) > 1 and argv[1] in {"download", "downloader"}
    if executable.startswith("python"):
        if len(argv) > 3 and argv[1:3] == ("-m", "pip"):
            return argv[3] == "install"
        if len(argv) > 3 and argv[1:3] == ("-m", "spacy"):
            return argv[3] == "download"
        if len(argv) > 2 and argv[1:3] in {
            ("-m", "nltk"),
            ("-m", "nltk.downloader"),
        }:
            return True
        if len(argv) > 2 and argv[1] == "-c":
            return _python_c_downloads_data(argv[2])
    return False


def _assert_final_install_order(commands: tuple[str, ...], workload: str) -> None:
    argvs = tuple(argv for source in commands for argv in _shell_argvs(source))
    installers = [
        index
        for index, argv in enumerate(argvs)
        if argv == ("make", "install-torch-stack")
    ]
    assert len(installers) == 1
    changes = [index for index, argv in enumerate(argvs) if _is_package_or_data_change(argv)]
    assert installers[0] in changes
    pip_check = argvs.index(("python", "-m", "pip", "check"))
    stack = argvs.index(("make", "verify-torch-stack"))
    nnx = argvs.index(("make", "verify-nnx-install"))
    workload_argv = next(argv for argv in argvs if shlex.join(argv) == workload)
    work = argvs.index(workload_argv)
    assert installers[0] <= max(changes) < pip_check < stack < nnx < work
    assert not any(_is_package_or_data_change(argv) for argv in argvs[pip_check:work])


_WARNING_ACTIONS = ("default", "error", "ignore", "always", "module", "once")
_FORBIDDEN_WARNING_ARGV = frozenset(
    (
        "--disable-warnings",
        "--disable-pytest-warnings",
    )
)


def _warning_action(specification: str) -> str:
    action = specification.split(":", 1)[0].strip().lower()
    if not action:
        return "default"
    if action == "all":
        return "always"
    matches = tuple(candidate for candidate in _WARNING_ACTIONS if candidate.startswith(action))
    assert len(matches) == 1, specification
    return matches[0]


def _warning_actions(argv: Sequence[str]) -> tuple[str, ...]:
    actions: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in {"-W", "--pythonwarnings"}:
            assert index + 1 < len(argv), argv
            actions.append(_warning_action(argv[index + 1]))
            index += 2
            continue
        if token.startswith("--pythonwarnings="):
            actions.append(_warning_action(token.split("=", 1)[1]))
            index += 1
            continue
        if token.startswith("-W"):
            actions.append(_warning_action(token[2:]))
        index += 1
    return tuple(actions)


def _pythonwarnings_actions(value: object) -> tuple[str, ...]:
    assert isinstance(value, str) and value, value
    return tuple(_warning_action(part) for part in value.split(","))


def _pytest_plugin_options(argv: Sequence[str]) -> tuple[str, ...]:
    plugins: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "-p":
            assert index + 1 < len(argv), argv
            plugins.append(argv[index + 1])
            index += 2
            continue
        if token.startswith("-p"):
            plugins.append(token[2:])
        index += 1
    return tuple(plugins)


def _assert_no_warning_bypass(argv: Sequence[str]) -> None:
    assert _FORBIDDEN_WARNING_ARGV.isdisjoint(argv)
    assert "no:warnings" not in _pytest_plugin_options(argv)
    assert not any("filterwarnings=" in token for token in argv)


def _environment_warning_actions(env: object) -> tuple[str, ...]:
    if env is None:
        return ()
    assert isinstance(env, dict), env
    actions: list[str] = []
    if "PYTHONWARNINGS" in env:
        actions.extend(_pythonwarnings_actions(env["PYTHONWARNINGS"]))
    if "PYTEST_ADDOPTS" in env:
        assert isinstance(env["PYTEST_ADDOPTS"], str), env["PYTEST_ADDOPTS"]
        addopts = tuple(shlex.split(env["PYTEST_ADDOPTS"]))
        _assert_no_warning_bypass(addopts)
        actions.extend(_warning_actions(addopts))
    return tuple(actions)


def _assert_warning_error_command(
    argv: tuple[str, ...],
    *environments: object,
) -> None:
    _assert_no_warning_bypass(argv)
    command_actions = _warning_actions(argv)
    environment_actions = tuple(
        action
        for env in environments
        for action in _environment_warning_actions(env)
    )
    assert (
        sum(
            argv[index : index + 2] == ("-W", "error")
            for index in range(len(argv) - 1)
        )
        == 1
    ), argv
    assert command_actions == ("error",), command_actions
    assert command_actions + environment_actions == ("error",), (
        command_actions,
        environment_actions,
    )


def _assert_nnx_warning_contract(workflow: dict[str, object]) -> None:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["pytest-nnx-surface"]
    assert isinstance(job, dict)
    steps = job["steps"]
    assert isinstance(steps, list)
    step = next(item for item in steps if item.get("name") == "Run NNx-surface tests")
    pytest_commands = tuple(
        command
        for command in _shell_commands(step["run"])
        if command.argv and Path(command.argv[0]).name == "pytest"
    )
    assert len(pytest_commands) == 1, pytest_commands
    command = pytest_commands[0]
    _assert_warning_error_command(
        command.argv,
        command.environment,
        workflow.get("env"),
        job.get("env"),
        step.get("env"),
    )


_RUNTIME_JOB_WORKLOADS = {
    "pytest-repository": "make test",
    "pytest-nnx-surface": (
        "pytest -p no:cacheprovider -W error --junitxml=/tmp/nnx-surface.xml "
        "tests/nnx_surface -v"
    ),
    "verify-repo": "make verify",
    "tier-a-papermill": "make smoke-tier-a",
    "smoke-tier-b": "make smoke-tier-b",
    "smoke-tier-c": "make smoke-tier-c",
}


def _job_run_commands(workflow: dict, job_name: str) -> tuple[str, ...]:
    return tuple(
        step["run"]
        for step in workflow["jobs"][job_name]["steps"]
        if "run" in step
    )


def _unquoted_pipe_operators(source: str) -> tuple[str, ...]:
    operators: list[str] = []
    quote: str | None = None
    escaped = False
    in_comment = False
    at_word_start = True
    index = 0
    while index < len(source):
        character = source[index]
        if in_comment:
            if character == "\n":
                in_comment = False
                at_word_start = True
            index += 1
            continue
        if quote is not None:
            if quote == '"' and escaped:
                escaped = False
            elif quote == '"' and character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            index += 1
            continue
        if escaped:
            escaped = False
            at_word_start = False
            index += 1
            continue
        if character == "\\":
            escaped = True
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
            at_word_start = False
            index += 1
            continue
        if character == "#" and at_word_start:
            in_comment = True
            index += 1
            continue
        if character == "|":
            if index + 1 < len(source) and source[index + 1] == "|":
                operators.append("||")
                index += 2
            else:
                operators.append("|")
                index += 1
            at_word_start = True
            continue
        at_word_start = character.isspace() or character in ";&()"
        index += 1
    return tuple(operators)


def _assert_no_failure_masking(step: Mapping[str, object]) -> None:
    assert "continue-on-error" not in step
    assert "shell" not in step
    source = step.get("run")
    if source is None:
        return
    assert isinstance(source, str)
    operators = _unquoted_pipe_operators(source)
    assert "||" not in operators
    assert "|" not in operators
    for argv in _shell_argvs(source):
        assert argv[:2] != ("set", "+e")
        assert argv[:3] != ("set", "+o", "errexit")


def _assert_runtime_job_install_contract(workflow: dict, job_name: str) -> None:
    job = workflow["jobs"][job_name]
    assert "services" not in job
    assert "container" not in job
    assert "continue-on-error" not in job
    assert "defaults" not in job
    for step in job["steps"]:
        _assert_no_failure_masking(step)
    setup = next(
        step
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/setup-python@")
    )
    cache_paths = tuple(setup["with"]["cache-dependency-path"].splitlines())
    assert len(cache_paths) == len(set(cache_paths))
    assert set(_STACK_CACHE_MANIFESTS) <= set(cache_paths)
    assert set(cache_paths) == set(_STACK_CACHE_MANIFESTS)
    commands = _job_run_commands(workflow, job_name)
    _assert_final_install_order(commands, _RUNTIME_JOB_WORKLOADS[job_name])
    all_commands = tuple(
        command for source in commands for command in _shell_commands(source)
    )
    warning_environments = (
        workflow.get("env"),
        job.get("env"),
        *(step.get("env") for step in job["steps"]),
    )
    assert all(
        not _environment_warning_actions(environment)
        for environment in warning_environments
    )
    assert all(
        not ({"PYTHONWARNINGS", "PYTEST_ADDOPTS"} & set(command.environment))
        for command in all_commands
    )
    for command in all_commands:
        if job_name == "pytest-nnx-surface" and Path(command.argv[0]).name == "pytest":
            _assert_warning_error_command(command.argv, command.environment)
        else:
            _assert_no_warning_bypass(command.argv)
            assert not _warning_actions(command.argv)
    forbidden_executables = {"jupyter", "jupyterhub", "ollama", "comfyui"}
    assert all(
        Path(command.argv[0]).name not in forbidden_executables
        and command.argv[:2] not in {
            ("docker", "compose"),
            ("docker-compose", "up"),
            ("make", "atlas-setup"),
            ("make", "atlas-up"),
        }
        for command in all_commands
    )
    checkout = next(step for step in job["steps"] if step.get("name") == "Checkout")
    if job_name == "verify-repo":
        assert checkout.get("with") == {
            "persist-credentials": "false",
            "fetch-depth": "0",
            "submodules": "recursive",
        }
    elif job_name == "pytest-repository":
        assert checkout.get("with") == {
            "persist-credentials": "false",
            "submodules": "recursive",
        }
    else:
        assert "submodules" not in checkout.get("with", {})


def _assert_nnx_junit_contract(workflow: dict) -> None:
    step = next(
        item
        for item in workflow["jobs"]["pytest-nnx-surface"]["steps"]
        if item.get("name") == "Run NNx-surface tests"
    )
    assert _shell_argvs(step["run"]) == (
        (
            "pytest",
            "-p",
            "no:cacheprovider",
            "-W",
            "error",
            "--junitxml=/tmp/nnx-surface.xml",
            "tests/nnx_surface",
            "-v",
        ),
        (
            "python",
            "-m",
            "scripts.verify_junit",
            "/tmp/nnx-surface.xml",
        ),
    )


_TIER_OUTPUT_CONTRACTS = {
    "tier-a-papermill": ("a", "/tmp/ml-tier-a"),
    "smoke-tier-b": ("b", "/tmp/ml-smoke"),
    "smoke-tier-c": ("c", "/tmp/ml-smoke"),
}


def _assert_tier_output_contract(workflow: dict, job_name: str) -> None:
    argvs = tuple(
        argv
        for source in _job_run_commands(workflow, job_name)
        for argv in _shell_argvs(source)
    )
    workload = tuple(shlex.split(_RUNTIME_JOB_WORKLOADS[job_name]))
    tier, root = _TIER_OUTPUT_CONTRACTS[job_name]
    oracle = (
        "python",
        "-m",
        "scripts.verify_smoke_outputs",
        "--tier",
        tier,
        "--root",
        root,
    )
    assert argvs.count(workload) == 1
    assert argvs.count(oracle) == 1
    assert argvs.index(oracle) == argvs.index(workload) + 1


@pytest.mark.parametrize(
    "command",
    (
        "sudo apt install libcairo2",
        "sudo apt-get install -y libcairo2",
        "env PIP_NO_INDEX=1 python -m pip install package",
        "sudo env PIP_NO_INDEX=1 python -m pip install package",
    ),
)
def test_package_change_classifier_normalizes_wrappers(command):
    (argv,) = _shell_argvs(command)
    assert _is_package_or_data_change(argv)


def test_shell_parser_preserves_inline_warning_environment_and_wrappers():
    (command,) = _shell_commands(
        "sudo env PYTHONWARNINGS=ignore "
        "PYTEST_ADDOPTS='--pythonwarnings default' pytest -W error tests/nnx_surface"
    )
    assert command.argv == ("pytest", "-W", "error", "tests/nnx_surface")
    assert command.environment == {
        "PYTHONWARNINGS": "ignore",
        "PYTEST_ADDOPTS": "--pythonwarnings default",
    }
    assert command.wrappers == ("sudo", "env")


def test_shell_parser_handles_line_continuations_newlines_and_assignments():
    assert _shell_argvs("python -m pip check \\\n&& make verify-torch-stack\nmake verify-nnx-install") == (
        ("python", "-m", "pip", "check"),
        ("make", "verify-torch-stack"),
        ("make", "verify-nnx-install"),
    )
    (command,) = _shell_commands("env MODE=test sudo make verify-torch-stack")
    assert command.argv == ("make", "verify-torch-stack")
    assert command.environment == {"MODE": "test"}
    assert command.wrappers == ("env", "sudo")


@pytest.mark.parametrize("job_name", tuple(_RUNTIME_JOB_WORKLOADS))
def test_ci_runtime_jobs_use_final_install_order_and_complete_cache_manifest(job_name):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    _assert_runtime_job_install_contract(workflow, job_name)


def test_ci_verify_repo_submodule_contract_initializes_recursive_checkout():
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    checkout = next(
        step
        for step in workflow["jobs"]["verify-repo"]["steps"]
        if step.get("name") == "Checkout"
    )

    assert checkout["with"] == {
        "persist-credentials": "false",
        "fetch-depth": "0",
        "submodules": "recursive",
    }
    _assert_runtime_job_install_contract(workflow, "verify-repo")


def test_ci_repository_suite_initializes_recursive_checkout_for_atlas_projection_tests():
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    checkout = next(
        step
        for step in workflow["jobs"]["pytest-repository"]["steps"]
        if step.get("name") == "Checkout"
    )

    assert checkout["with"] == {
        "persist-credentials": "false",
        "submodules": "recursive",
    }
    _assert_runtime_job_install_contract(workflow, "pytest-repository")


def _assert_ci_atlas_consumer_policy_recursive_checkout(workflow: dict) -> None:
    checkout = next(
        step
        for step in workflow["jobs"]["atlas-consumer-policy"]["steps"]
        if step.get("name") == "Checkout"
    )

    assert checkout["with"] == {
        "persist-credentials": "false",
        "submodules": "recursive",
    }


def test_ci_atlas_consumer_policy_initializes_recursive_checkout():
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    _assert_ci_atlas_consumer_policy_recursive_checkout(workflow)


@pytest.mark.parametrize(
    "mutation",
    (None, False, True, "false", "true", "recursive "),
    ids=("omitted", "false-bool", "true-bool", "false-string", "true-string", "spaced"),
)
def test_ci_atlas_consumer_policy_rejects_nonrecursive_checkout(mutation):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    checkout = next(
        step
        for step in workflow["jobs"]["atlas-consumer-policy"]["steps"]
        if step.get("name") == "Checkout"
    )
    checkout["with"]["submodules"] = "recursive"

    if mutation is None:
        checkout["with"].pop("submodules")
    else:
        checkout["with"]["submodules"] = mutation
    with pytest.raises(AssertionError):
        _assert_ci_atlas_consumer_policy_recursive_checkout(workflow)


@pytest.mark.parametrize(
    "mutation",
    (None, False, True, "false", "true", "recursive "),
    ids=("omitted", "false-bool", "true-bool", "false-string", "true-string", "spaced"),
)
def test_ci_verify_repo_submodule_contract_rejects_nonrecursive_mutations(mutation):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    checkout = next(
        step
        for step in workflow["jobs"]["verify-repo"]["steps"]
        if step.get("name") == "Checkout"
    )
    checkout["with"]["submodules"] = "recursive"
    _assert_runtime_job_install_contract(workflow, "verify-repo")

    if mutation is None:
        checkout["with"].pop("submodules")
    else:
        checkout["with"]["submodules"] = mutation
    with pytest.raises(AssertionError):
        _assert_runtime_job_install_contract(workflow, "verify-repo")


@pytest.mark.parametrize(
    "mutation",
    (None, False, True, "false", "true", "recursive "),
    ids=("omitted", "false-bool", "true-bool", "false-string", "true-string", "spaced"),
)
def test_ci_repository_suite_rejects_nonrecursive_checkout_mutations(mutation):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    checkout = next(
        step
        for step in workflow["jobs"]["pytest-repository"]["steps"]
        if step.get("name") == "Checkout"
    )
    checkout["with"]["submodules"] = "recursive"
    _assert_runtime_job_install_contract(workflow, "pytest-repository")

    if mutation is None:
        checkout["with"].pop("submodules")
    else:
        checkout["with"]["submodules"] = mutation
    with pytest.raises(AssertionError):
        _assert_runtime_job_install_contract(workflow, "pytest-repository")


@pytest.mark.parametrize(
    "job_name",
    tuple(
        name
        for name in _RUNTIME_JOB_WORKLOADS
        if name not in {"pytest-repository", "verify-repo"}
    ),
)
def test_ci_verify_repo_submodule_contract_preserves_other_runtime_checkouts(job_name):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    checkout = next(
        step
        for step in workflow["jobs"][job_name]["steps"]
        if step.get("name") == "Checkout"
    )
    assert "submodules" not in checkout.get("with", {})
    checkout["with"]["submodules"] = "recursive"

    with pytest.raises(AssertionError):
        _assert_runtime_job_install_contract(workflow, job_name)


@pytest.mark.parametrize("job_name", tuple(_RUNTIME_JOB_WORKLOADS))
def test_ci_runtime_job_contract_rejects_job_continue_on_error(job_name):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    workflow["jobs"][job_name]["continue-on-error"] = "true"

    with pytest.raises(AssertionError):
        _assert_runtime_job_install_contract(workflow, job_name)


@pytest.mark.parametrize("job_name", tuple(_RUNTIME_JOB_WORKLOADS))
def test_ci_runtime_job_contract_rejects_continue_on_error_on_every_step(job_name):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    steps = workflow["jobs"][job_name]["steps"]
    assert steps

    for step_index in range(len(steps)):
        mutated = copy.deepcopy(workflow)
        step = mutated["jobs"][job_name]["steps"][step_index]
        original = copy.deepcopy(step)
        step["continue-on-error"] = "true"
        assert step != original
        with pytest.raises(AssertionError):
            _assert_runtime_job_install_contract(mutated, job_name)


@pytest.mark.parametrize("job_name", tuple(_RUNTIME_JOB_WORKLOADS))
@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("continue-on-error", "true"),
        ("shell", "bash {0} || true"),
        ("run-suffix", " || true"),
        ("run-suffix", " || :"),
        ("run-prefix", "set +e\n"),
        ("run-prefix", "set +o errexit\n"),
    ),
)
def test_ci_runtime_job_contract_rejects_failure_masking_on_every_run_step(
    job_name,
    mutation,
    value,
):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    run_indexes = tuple(
        index
        for index, step in enumerate(workflow["jobs"][job_name]["steps"])
        if "run" in step
    )
    assert run_indexes

    for step_index in run_indexes:
        mutated = copy.deepcopy(workflow)
        step = mutated["jobs"][job_name]["steps"][step_index]
        original = copy.deepcopy(step)
        if mutation == "run-suffix":
            step["run"] = f"{step['run']}{value}"
        elif mutation == "run-prefix":
            step["run"] = f"{value}{step['run']}"
        else:
            step[mutation] = value
        assert step != original
        with pytest.raises(AssertionError):
            _assert_runtime_job_install_contract(mutated, job_name)


@pytest.mark.parametrize(
    ("job_name", "step_name", "command"),
    (
        ("verify-repo", "Run repo verifier", "make verify"),
        (
            "tier-a-papermill",
            "Verify Tier-A notebook output contract",
            "python -m scripts.verify_smoke_outputs --tier a --root /tmp/ml-tier-a",
        ),
        ("pytest-repository", "Install dependencies", "make install-torch-stack"),
        (
            "pytest-nnx-surface",
            "Check and verify canonical Torch and NNx stack",
            "make verify-torch-stack",
        ),
        ("smoke-tier-b", "Smoke-run Tier-B notebooks", "make smoke-tier-b"),
    ),
)
def test_ci_runtime_job_contract_rejects_unquoted_pipeline_masking(
    job_name,
    step_name,
    command,
):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    step = next(
        item
        for item in workflow["jobs"][job_name]["steps"]
        if item.get("name") == step_name
    )
    original = step["run"]
    assert original.count(command) == 1
    step["run"] = original.replace(command, f"{command} | cat", 1)
    assert step["run"] != original

    with pytest.raises(AssertionError):
        _assert_runtime_job_install_contract(workflow, job_name)


@pytest.mark.parametrize(
    "source",
    (
        "make verify",
        "python -m pip check\nmake verify-torch-stack\nmake verify-nnx-install",
        "printf '%s\\n' 'verification | complete'",
        "printf '%s\\n' 'fallback || complete'",
        "printf '%s\\n' '|'",
        "printf '%s\\n' '||'",
        'python -c "print(\'workload | complete\')"',
    ),
)
def test_ci_failure_masking_contract_accepts_ordinary_commands(source):
    _assert_no_failure_masking({"run": source})


@pytest.mark.parametrize("job_name", tuple(_RUNTIME_JOB_WORKLOADS))
@pytest.mark.parametrize("manifest", _STACK_CACHE_MANIFESTS)
def test_ci_runtime_cache_manifest_rejects_each_omission(job_name, manifest):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    setup = next(
        step
        for step in workflow["jobs"][job_name]["steps"]
        if str(step.get("uses", "")).startswith("actions/setup-python@")
    )
    original = setup["with"]["cache-dependency-path"]
    mutated = original.replace(f"{manifest}\n", "")
    assert mutated != original
    setup["with"]["cache-dependency-path"] = mutated
    with pytest.raises(AssertionError):
        _assert_runtime_job_install_contract(workflow, job_name)


def _ordered_install_fixture() -> tuple[str, ...]:
    return (
        "sudo apt-get install -y libcairo2",
        "make install-torch-stack",
        "python -m pip install -r docs-requirements.txt",
        "make nlp-assets",
        "make verify-nlp-assets",
        "python -m pip check",
        "make verify-torch-stack",
        "make verify-nnx-install",
        "make test",
    )


def test_final_install_order_accepts_allowed_setup_before_final_verification():
    _assert_final_install_order(_ordered_install_fixture(), "make test")


@pytest.mark.parametrize(
    "late_change",
    (
        "pip install package",
        "pip3 install package",
        "python -m pip install package",
        "uv pip install package",
        "apt install package",
        "apt-get install package",
        "conda install package",
        "python -m spacy download model",
        "spacy download model",
        "nltk download vader_lexicon",
        "python -m nltk.downloader vader_lexicon",
        "sudo apt install package",
        "sudo apt-get install package",
        "env PIP_NO_INDEX=1 python -m pip install package",
        'python -c "import nltk; nltk.download(\'vader_lexicon\')"',
        "make install-extra",
        "make nlp-assets",
        "make codespace-setup",
    ),
)
def test_final_install_order_rejects_every_late_package_or_data_change(late_change):
    commands = list(_ordered_install_fixture())
    commands.insert(-1, late_change)
    with pytest.raises(AssertionError):
        _assert_final_install_order(tuple(commands), "make test")


@pytest.mark.parametrize("position", (5, 6, 7))
def test_final_install_order_rejects_duplicate_installer_at_each_verification_boundary(position):
    commands = list(_ordered_install_fixture())
    commands.insert(position, "make install-torch-stack")
    with pytest.raises(AssertionError):
        _assert_final_install_order(tuple(commands), "make test")


@pytest.mark.parametrize(
    "suffix",
    (
        "-W ignore",
        "-Wignore",
        "-Wignore::UserWarning",
        "-Wdefault",
        "-Wignore::DeprecationWarning",
        "-W once",
        "-Wonce",
        "-W module",
        "-Wmodule",
        "-W always",
        "-Walways",
        "-Werror",
        "--pythonwarnings ignore",
        "--pythonwarnings=default",
        "--pythonwarnings=ignore::DeprecationWarning",
        "-p no:warnings",
        "-pno:warnings",
        "--disable-warnings",
    ),
)
def test_nnx_ci_rejects_appended_warning_cli_actions(suffix):
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    mutated = copy.deepcopy(workflow)
    step = next(
        item
        for item in mutated["jobs"]["pytest-nnx-surface"]["steps"]
        if item.get("name") == "Run NNx-surface tests"
    )
    original = step["run"]
    step["run"] = original.replace("-W error", f"-W error {suffix}", 1)
    assert step["run"] != original and "-W error" in step["run"]
    with pytest.raises(AssertionError):
        _assert_nnx_warning_contract(mutated)


@pytest.mark.parametrize("level", ("workflow", "job", "step"))
@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("PYTHONWARNINGS", "ignore"),
        ("PYTHONWARNINGS", "default"),
        ("PYTHONWARNINGS", "ignore::DeprecationWarning"),
        ("PYTHONWARNINGS", "once"),
        ("PYTHONWARNINGS", "module"),
        ("PYTHONWARNINGS", "always"),
        ("PYTHONWARNINGS", "error"),
        ("PYTEST_ADDOPTS", "-W ignore"),
        ("PYTEST_ADDOPTS", "-Wdefault"),
        ("PYTEST_ADDOPTS", "-Wignore::DeprecationWarning"),
        ("PYTEST_ADDOPTS", "-W once"),
        ("PYTEST_ADDOPTS", "-Wmodule"),
        ("PYTEST_ADDOPTS", "-Walways"),
        ("PYTEST_ADDOPTS", "-Werror"),
        ("PYTEST_ADDOPTS", "-p no:warnings"),
        ("PYTEST_ADDOPTS", "-pno:warnings"),
        ("PYTEST_ADDOPTS", "--disable-warnings"),
    ),
)
def test_nnx_ci_rejects_appended_warning_environment(level, name, value):
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    mutated = copy.deepcopy(workflow)
    job = mutated["jobs"]["pytest-nnx-surface"]
    step = next(item for item in job["steps"] if item.get("name") == "Run NNx-surface tests")
    owner = {"workflow": mutated, "job": job, "step": step}[level]
    owner.setdefault("env", {})[name] = value
    assert "-W error" in step["run"]
    with pytest.raises(AssertionError):
        _assert_nnx_warning_contract(mutated)


@pytest.mark.parametrize(
    "prefix",
    (
        "PYTHONWARNINGS=ignore",
        "PYTEST_ADDOPTS='-W ignore'",
        "PYTEST_ADDOPTS='-p no:warnings'",
        "env PYTHONWARNINGS=ignore::DeprecationWarning",
        "env PYTEST_ADDOPTS='-Wdefault'",
        "env PYTEST_ADDOPTS=-pno:warnings",
        "sudo env PYTEST_ADDOPTS='-Wignore::DeprecationWarning'",
        "sudo env PYTEST_ADDOPTS='-p no:warnings'",
    ),
)
def test_nnx_ci_rejects_inline_warning_environment(prefix):
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    mutated = copy.deepcopy(workflow)
    step = next(
        item
        for item in mutated["jobs"]["pytest-nnx-surface"]["steps"]
        if item.get("name") == "Run NNx-surface tests"
    )
    original = step["run"]
    step["run"] = original.replace("pytest -p", f"{prefix} pytest -p", 1)
    assert step["run"] != original and "-W error" in step["run"]
    with pytest.raises(AssertionError):
        _assert_nnx_warning_contract(mutated)


def test_nnx_ci_warning_error_contract_accepts_only_original_error_action():
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    _assert_nnx_warning_contract(workflow)
    _assert_nnx_junit_contract(workflow)
    assert _warning_actions(("pytest", "-Werror")) == ("error",)
    assert _warning_actions(("pytest", "-W", "error")) == ("error",)
    assert _warning_actions(("pytest", "--pythonwarnings", "ignore")) == ("ignore",)
    assert _warning_actions(("pytest", "--pythonwarnings=default")) == ("default",)
    assert _pytest_plugin_options(("pytest", "-p", "no:warnings")) == ("no:warnings",)
    assert _pytest_plugin_options(("pytest", "-pno:warnings")) == ("no:warnings",)
    _assert_no_warning_bypass(("pytest", "-p", "no:cacheprovider", "-W", "error"))
    _assert_warning_error_command(("pytest", "-W", "error"))
    for argv in (
        ("pytest", "-Werror"),
        ("pytest", "-W", "error", "-W", "ignore"),
        ("pytest", "-Werror", "-Wdefault"),
        ("pytest", "-W", "error", "-Wignore::DeprecationWarning"),
        ("pytest", "-W", "error", "--pythonwarnings", "ignore"),
        ("pytest", "-W", "error", "--pythonwarnings=default"),
        ("pytest", "-W", "error", "-p", "no:warnings"),
        ("pytest", "-W", "error", "-pno:warnings"),
    ):
        with pytest.raises(AssertionError):
            _assert_warning_error_command(argv)


@pytest.mark.parametrize("job_name", tuple(_TIER_OUTPUT_CONTRACTS))
def test_ci_tier_output_oracle_follows_matching_workload(job_name):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    _assert_tier_output_contract(workflow, job_name)


@pytest.mark.parametrize("job_name", tuple(_TIER_OUTPUT_CONTRACTS))
@pytest.mark.parametrize("mutation", ("missing", "wrong-root", "duplicate-workload", "late-install"))
def test_ci_tier_output_oracle_rejects_order_mutations(job_name, mutation):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    steps = workflow["jobs"][job_name]["steps"]
    workload_name = {
        "tier-a-papermill": "Run Tier-A notebooks (papermill)",
        "smoke-tier-b": "Smoke-run Tier-B notebooks",
        "smoke-tier-c": "Smoke-run Tier-C notebooks",
    }[job_name]
    workload_index = next(
        index for index, step in enumerate(steps) if step.get("name") == workload_name
    )
    oracle_index = workload_index + 1
    if mutation == "missing":
        steps.pop(oracle_index)
    elif mutation == "wrong-root":
        steps[oracle_index]["run"] = steps[oracle_index]["run"].replace(
            _TIER_OUTPUT_CONTRACTS[job_name][1], "/tmp/wrong"
        )
    elif mutation == "duplicate-workload":
        steps.insert(oracle_index, copy.deepcopy(steps[workload_index]))
    else:
        steps.insert(oracle_index, {"name": "Late install", "run": "pip install package"})
    with pytest.raises(AssertionError):
        _assert_tier_output_contract(workflow, job_name)


_DEPENDENCY_AUDIT_JOB = {
    "name": "dependency-audit",
    "runs-on": "ubuntu-24.04",
    "timeout-minutes": "20",
    "steps": [
        {
            "name": "Checkout",
            "uses": "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0",
            "with": {"persist-credentials": "false", "fetch-depth": "0"},
        },
        {
            "name": "Set up Python 3.11",
            "uses": "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
            "with": {
                "python-version": "3.11.15",
                "cache": "pip",
                "cache-dependency-path": _cache_text(_AUDIT_CACHE_MANIFESTS),
            },
        },
        {
            "name": "Install locked audit environment",
            "run": "make install-bootstrap\nmake install-audit-lock\npython -m pip check\n",
        },
        {
            "name": "Verify committed dependency locks offline",
            "run": "make verify-dependency-locks",
        },
        {
            "name": "Compare dependency advisories with accepted baseline",
            "run": "make audit-advisories",
        },
        {
            "name": "Select networked dependency checks",
            "id": "dependency-scope",
            "env": {
                "BASE_SHA": "${{ github.event.pull_request.base.sha || github.event.before }}",
                "HEAD_SHA": "${{ github.event.pull_request.head.sha || github.sha }}",
            },
            "run": (
                "python -m scripts.verify_dependency_locks \\\n"
                "  --classify-event \"${{ github.event_name }}\" \\\n"
                "  --base \"$BASE_SHA\" \\\n"
                "  --head \"$HEAD_SHA\" \\\n"
                "  --github-output \"$GITHUB_OUTPUT\"\n"
            ),
        },
        {
            "name": "Install locked compiler",
            "if": "steps.dependency-scope.outputs.lock-check == 'true'",
            "run": "make install-compiler-lock\npython -m pip check\n",
        },
        {
            "name": "Regenerate and compare dependency locks",
            "if": "steps.dependency-scope.outputs.lock-check == 'true'",
            "run": "make lock-check",
        },
        {
            "name": "Validate immutable image identities",
            "if": "steps.dependency-scope.outputs.image-check == 'true'",
            "run": "make image-lock-check",
        },
    ],
}


def _assert_dependency_audit_job_contract(workflow: dict) -> None:
    assert "defaults" not in workflow
    assert "env" not in workflow
    assert "dependency-audit" in workflow["jobs"]

    job = workflow["jobs"]["dependency-audit"]
    assert set(job) == {"name", "runs-on", "timeout-minutes", "steps"}
    assert job == _DEPENDENCY_AUDIT_JOB


def _assert_dependency_audit_ci_envelope(workflow: dict) -> None:
    assert workflow["on"] == {
        "push": {"branches": ["develop", "main"]},
        "pull_request": {
            "branches": ["develop", "main"],
            "types": ["opened", "synchronize", "reopened", "labeled"],
        },
        "workflow_dispatch": "",
        "schedule": [{"cron": "0 7 * * 1"}],
    }
    assert workflow["run-name"] == (
        "CI / ${{ github.event_name }} / "
        "${{ github.event.action || 'none' }} / PR "
        "${{ github.event.pull_request.number || 0 }}"
    )
    assert workflow["permissions"] == {"contents": "read"}


def test_ci_dependency_audit_job_contract():
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")

    _assert_dependency_audit_ci_envelope(workflow)
    _assert_dependency_audit_job_contract(workflow)


@pytest.mark.parametrize(
    "mutation",
    (
        "remove-schedule",
        "remove-dispatch",
        "change-cron",
        "add-write-permission",
        "remove-pr-types",
        "remove-labeled-type",
        "wrong-pr-type",
        "wrong-run-name",
    ),
)
def test_ci_dependency_audit_envelope_rejects_trigger_or_permission_mutations(mutation):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    if mutation == "remove-schedule":
        del workflow["on"]["schedule"]
    elif mutation == "remove-dispatch":
        del workflow["on"]["workflow_dispatch"]
    elif mutation == "change-cron":
        workflow["on"]["schedule"][0]["cron"] = "17 3 * * 1"
    elif mutation == "remove-pr-types":
        del workflow["on"]["pull_request"]["types"]
    elif mutation == "remove-labeled-type":
        workflow["on"]["pull_request"]["types"].remove("labeled")
    elif mutation == "wrong-pr-type":
        workflow["on"]["pull_request"]["types"][-1] = "closed"
    elif mutation == "wrong-run-name":
        workflow["run-name"] = "CI"
    else:
        workflow["permissions"]["pull-requests"] = "write"

    with pytest.raises(AssertionError):
        _assert_dependency_audit_ci_envelope(workflow)


@pytest.mark.parametrize("mutation", ("deleted", "renamed"))
def test_ci_dependency_audit_job_contract_rejects_deleted_or_renamed_job(mutation):
    workflow = {"jobs": {"dependency-audit": deepcopy(_DEPENDENCY_AUDIT_JOB)}}
    job = workflow["jobs"].pop("dependency-audit")
    if mutation == "renamed":
        workflow["jobs"]["renamed-dependency-audit"] = job

    with pytest.raises(AssertionError):
        _assert_dependency_audit_job_contract(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("if", "github.ref == 'refs/heads/main'"),
        ("needs", "verify-repo"),
        ("services", {"postgres": {"image": "postgres"}}),
        ("container", "python:3.11"),
        ("env", {"PYTEST_ADDOPTS": "-q"}),
        ("defaults", {"run": {"shell": "bash"}}),
        ("continue-on-error", "true"),
    ],
)
def test_ci_dependency_audit_job_contract_rejects_job_level_controls(field, value):
    workflow = {"jobs": {"dependency-audit": deepcopy(_DEPENDENCY_AUDIT_JOB)}}
    workflow["jobs"]["dependency-audit"][field] = value

    with pytest.raises(AssertionError):
        _assert_dependency_audit_job_contract(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("if", "github.ref == 'refs/heads/main'"),
        ("continue-on-error", "true"),
        ("shell", "bash {0} || true"),
        ("env", {"PYTEST_ADDOPTS": "-q"}),
    ],
)
def test_ci_dependency_audit_job_contract_rejects_step_level_controls(field, value):
    workflow = {"jobs": {"dependency-audit": deepcopy(_DEPENDENCY_AUDIT_JOB)}}
    workflow["jobs"]["dependency-audit"]["steps"][-1][field] = value

    with pytest.raises(AssertionError):
        _assert_dependency_audit_job_contract(workflow)


@pytest.mark.parametrize("mutation", ("extra", "reordered"))
def test_ci_dependency_audit_job_contract_rejects_changed_step_inventory(mutation):
    workflow = {"jobs": {"dependency-audit": deepcopy(_DEPENDENCY_AUDIT_JOB)}}
    steps = workflow["jobs"]["dependency-audit"]["steps"]
    if mutation == "extra":
        steps.append({"name": "Extra validation", "run": "true"})
    else:
        steps[0], steps[1] = steps[1], steps[0]

    with pytest.raises(AssertionError):
        _assert_dependency_audit_job_contract(workflow)


@pytest.mark.parametrize("step_index", range(len(_DEPENDENCY_AUDIT_JOB["steps"])))
def test_ci_dependency_audit_job_contract_rejects_omitted_step(step_index):
    workflow = {"jobs": {"dependency-audit": deepcopy(_DEPENDENCY_AUDIT_JOB)}}
    del workflow["jobs"]["dependency-audit"]["steps"][step_index]

    with pytest.raises(AssertionError):
        _assert_dependency_audit_job_contract(workflow)


def test_ci_dependency_audit_job_contract_rejects_checkout_submodules():
    workflow = {"jobs": {"dependency-audit": deepcopy(_DEPENDENCY_AUDIT_JOB)}}
    workflow["jobs"]["dependency-audit"]["steps"][0]["with"]["submodules"] = "recursive"

    with pytest.raises(AssertionError):
        _assert_dependency_audit_job_contract(workflow)


@pytest.mark.parametrize("manifest", _AUDIT_CACHE_MANIFESTS)
def test_ci_dependency_audit_job_contract_rejects_missing_cache_manifest(manifest):
    workflow = {"jobs": {"dependency-audit": deepcopy(_DEPENDENCY_AUDIT_JOB)}}
    setup = workflow["jobs"]["dependency-audit"]["steps"][1]
    original = setup["with"]["cache-dependency-path"]
    mutated = original.replace(f"{manifest}\n", "")
    assert mutated != original
    setup["with"]["cache-dependency-path"] = mutated

    with pytest.raises(AssertionError):
        _assert_dependency_audit_job_contract(workflow)


@pytest.mark.parametrize(
    "command",
    (
        "pip install -r vulnerability-audit-requirements.txt",
        "python -m pip install pip-audit",
        "python -m pip install pip-audit==2.10.0",
        "python -m pip install -r requirements.txt",
        "python -m pip install -r vulnerability-audit-requirements.txt --ignore-vuln CVE-0000",
    ),
)
def test_ci_dependency_audit_job_contract_rejects_alternate_tool_install(command):
    workflow = {"jobs": {"dependency-audit": deepcopy(_DEPENDENCY_AUDIT_JOB)}}
    workflow["jobs"]["dependency-audit"]["steps"][2]["run"] = command

    with pytest.raises(AssertionError):
        _assert_dependency_audit_job_contract(workflow)


@pytest.mark.parametrize(
    "command",
    (
        "make audit-advisories --ignore-vuln CVE-0000",
        "python -m pip_audit -r requirements.txt",
        "make audit-advisories || true",
    ),
)
def test_ci_dependency_audit_job_contract_rejects_alternate_audit_command(command):
    workflow = {"jobs": {"dependency-audit": deepcopy(_DEPENDENCY_AUDIT_JOB)}}
    workflow["jobs"]["dependency-audit"]["steps"][3]["run"] = command

    with pytest.raises(AssertionError):
        _assert_dependency_audit_job_contract(workflow)


@pytest.mark.parametrize(
    "live_command",
    (
        "docker ps",
        "docker compose ps",
        "make atlas-setup",
        "make atlas-up",
        "make atlas-down",
        "./scripts/atlas-up.sh",
        "jupyterhub --version",
        "ollama serve",
        "comfyui --version",
        "curl localhost:8000",
        "curl 127.0.0.1:8000",
    ),
)
def test_ci_dependency_audit_job_contract_rejects_live_runtime_commands(live_command):
    workflow = {"jobs": {"dependency-audit": deepcopy(_DEPENDENCY_AUDIT_JOB)}}
    audit = workflow["jobs"]["dependency-audit"]["steps"][3]
    audit["run"] = f"{audit['run']}\n{live_command}"

    with pytest.raises(AssertionError):
        _assert_dependency_audit_job_contract(workflow)


_ATLAS_CONSUMER_POLICY_JOB = {
    "name": "atlas-consumer-policy",
    "runs-on": "ubuntu-24.04",
    "timeout-minutes": "15",
    "steps": [
        {
            "name": "Checkout",
            "uses": "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0",
            "with": {
                "persist-credentials": "false",
                "submodules": "recursive",
            },
        },
        {
            "name": "Set up Python 3.11",
            "uses": "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
            "with": {
                "python-version": "3.11.15",
                "cache": "pip",
                "cache-dependency-path": _cache_text(_ATLAS_CACHE_MANIFESTS),
            },
        },
        {
            "name": "Install focused Atlas contract dependencies",
            "run": (
                "make install-bootstrap\n"
                "make install-atlas-contract-lock\n"
                "python -m pip check\n"
            ),
        },
        {
            "name": "ShellCheck parent-owned Atlas wrappers",
            "run": (
                "shellcheck scripts/atlas-up.sh scripts/atlas-down.sh "
                "scripts/atlas-connect.sh scripts/lib/atlas-dotenv.sh"
            ),
        },
        {
            "name": "Run Atlas consumer policy tests",
            "run": "make test-atlas-consumer",
        },
    ],
}


def _valid_atlas_consumer_policy_workflow() -> dict:
    return {"jobs": {"atlas-consumer-policy": deepcopy(_ATLAS_CONSUMER_POLICY_JOB)}}


def _assert_atlas_consumer_policy_contract(workflow: dict) -> None:
    assert "defaults" not in workflow
    assert "env" not in workflow
    assert "atlas-consumer-policy" in workflow["jobs"]

    job = workflow["jobs"]["atlas-consumer-policy"]
    assert set(job) == {"name", "runs-on", "timeout-minutes", "steps"}
    assert job["name"] == "atlas-consumer-policy"
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == "15"
    assert job["steps"] == _ATLAS_CONSUMER_POLICY_JOB["steps"]

    command_body = "\n".join(
        step["run"] for step in job["steps"] if "run" in step
    ).lower()
    for forbidden in (
        "docker",
        "ollama serve",
        "make atlas-up",
        "make atlas-down",
        "curl",
        "localhost",
        "127.0.0.1",
    ):
        assert forbidden not in command_body


def test_atlas_consumer_policy_contract_is_exact_and_unconditional():
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")

    _assert_atlas_consumer_policy_contract(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("defaults", {"run": {"shell": "bash"}}),
        ("env", {"PYTEST_ADDOPTS": "-k smoke"}),
    ],
)
def test_atlas_consumer_policy_contract_rejects_workflow_level_controls(
    field,
    value,
):
    workflow = _valid_atlas_consumer_policy_workflow()
    workflow[field] = value

    with pytest.raises(AssertionError):
        _assert_atlas_consumer_policy_contract(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("defaults", {"run": {"shell": "bash"}}),
        ("env", {"PYTEST_ADDOPTS": "-k smoke"}),
        ("if", "github.ref == 'refs/heads/main'"),
        ("needs", "verify-repo"),
        ("services", {"ollama": {"image": "ollama/ollama"}}),
        ("container", "python:3.11"),
        ("continue-on-error", "true"),
    ],
)
def test_atlas_consumer_policy_contract_rejects_job_level_controls(field, value):
    workflow = _valid_atlas_consumer_policy_workflow()
    workflow["jobs"]["atlas-consumer-policy"][field] = value

    with pytest.raises(AssertionError):
        _assert_atlas_consumer_policy_contract(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("if", "github.ref == 'refs/heads/main'"),
        ("env", {"PYTEST_ADDOPTS": "-q"}),
        ("continue-on-error", "true"),
        ("timeout-minutes", "5"),
        ("shell", "bash {0} || true"),
    ],
)
def test_atlas_consumer_policy_contract_rejects_step_level_controls(field, value):
    workflow = _valid_atlas_consumer_policy_workflow()
    workflow["jobs"]["atlas-consumer-policy"]["steps"][-1][field] = value

    with pytest.raises(AssertionError):
        _assert_atlas_consumer_policy_contract(workflow)


@pytest.mark.parametrize("mutation", ["extra", "reordered"])
def test_atlas_consumer_policy_contract_rejects_changed_step_inventory(mutation):
    workflow = _valid_atlas_consumer_policy_workflow()
    steps = workflow["jobs"]["atlas-consumer-policy"]["steps"]
    if mutation == "extra":
        steps.append({"name": "Extra", "run": "true"})
    else:
        steps[0], steps[1] = steps[1], steps[0]

    with pytest.raises(AssertionError):
        _assert_atlas_consumer_policy_contract(workflow)


def test_atlas_consumer_policy_contract_rejects_nonrecursive_checkout():
    workflow = _valid_atlas_consumer_policy_workflow()
    checkout = workflow["jobs"]["atlas-consumer-policy"]["steps"][0]
    checkout["with"]["submodules"] = "false"

    with pytest.raises(AssertionError):
        _assert_atlas_consumer_policy_contract(workflow)


@pytest.mark.parametrize(
    "live_command",
    [
        "docker ps",
        "ollama serve",
        "make atlas-up",
        "make atlas-down",
        "curl http://example.invalid",
        "probe localhost",
        "probe 127.0.0.1",
    ],
)
def test_atlas_consumer_policy_contract_rejects_live_run_step_mutations(
    live_command,
):
    workflow = _valid_atlas_consumer_policy_workflow()
    install = workflow["jobs"]["atlas-consumer-policy"]["steps"][2]
    install["run"] = f"{install['run']}\n{live_command}"

    with pytest.raises(AssertionError):
        _assert_atlas_consumer_policy_contract(workflow)


_ATLAS_CONTRACT_PATHS = (
    ".gitmodules",
    "infra",
    "atlas.consumer.yml",
    "atlas.env.user.example",
    "compose/ml-eng-lab-atlas.yml",
    "scripts/atlas-*.sh",
    "scripts/atlas_runtime_probe.py",
    "scripts/lib/atlas-dotenv.sh",
    "scripts/docs/notebook_infrastructure.py",
    "docs/notebook-infrastructure.md",
    "docs/atlas-pin-bump-runbook.md",
    "docs/dependency-contracts.md",
    "notebooks/**/docs/spec.yaml",
    "scripts/verify_repo.py",
    "scripts/verify_repo_config.yaml",
    "tests/test_verify_repo.py",
    "tests/test_atlas_*.py",
    "tests/test_makefile_contract.py",
    "atlas-contract-requirements.txt",
    "bootstrap-requirements.txt",
    "requirements/lock-policy.toml",
    "requirements/locks/bootstrap.txt",
    "requirements/locks/atlas-contract.txt",
    "scripts/install_locked_requirements.py",
    "scripts/verify_dependency_locks.py",
    "Makefile",
    ".github/workflows/atlas-contract.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/docs.yml",
)

_ATLAS_CONTRACT_STEPS = [
    {
        "name": "Checkout",
        "uses": "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0",
        "with": {
            "persist-credentials": "false",
            "submodules": "recursive",
        },
    },
    {
        "name": "Set up Python 3.11",
        "uses": "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
        "with": {
            "python-version": "3.11.15",
            "cache": "pip",
            "cache-dependency-path": _cache_text(_ATLAS_CACHE_MANIFESTS),
        },
    },
    {
        "name": "Install pinned Atlas runner",
        "run": (
            "make install-bootstrap\n"
            "make install-atlas-contract-lock\n"
            "python -m pip check\n"
        ),
    },
    {
        "name": "Validate the non-live Atlas consumer contract",
        "shell": "bash",
        "run": """set -euo pipefail
cp infra/.env.example infra/.env
printf 'ML_ENG_LAB_REPO_PATH=%s\\n' "$GITHUB_WORKSPACE" > atlas.env.user
(
  cd infra
  ./start.sh env backfill
  ./start.sh --consumer ../atlas.consumer.yml compose validate
  ./start.sh --consumer ../atlas.consumer.yml doctor --format json
)
infra_status="$(git -C infra status --porcelain --untracked-files=all --ignored=no)"
if [[ -n "$infra_status" ]]; then
  printf '%s\\n' "Atlas validation changed tracked or non-ignored infra files:" >&2
  printf '%s\\n' "$infra_status" >&2
  exit 1
fi
""",
    },
]


def _valid_atlas_contract_workflow() -> dict:
    return {
        "name": "Atlas contract",
        "on": {
            "pull_request": {"paths": list(_ATLAS_CONTRACT_PATHS)},
            "workflow_dispatch": "",
        },
        "permissions": {"contents": "read"},
        "jobs": {
            "atlas-contract": {
                "runs-on": "ubuntu-24.04",
                "timeout-minutes": "15",
                "steps": deepcopy(_ATLAS_CONTRACT_STEPS),
            },
        },
    }


def _assert_atlas_contract_workflow_contract(workflow: dict) -> None:
    assert set(workflow) == {"name", "on", "permissions", "jobs"}
    assert workflow["name"] == "Atlas contract"
    assert workflow["on"] == {
        "pull_request": {"paths": list(_ATLAS_CONTRACT_PATHS)},
        "workflow_dispatch": "",
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"atlas-contract"}

    job = workflow["jobs"]["atlas-contract"]
    assert set(job) == {"runs-on", "timeout-minutes", "steps"}
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == "15"
    assert job["steps"] == _ATLAS_CONTRACT_STEPS

    command_body = "\n".join(
        step["run"] for step in job["steps"] if "run" in step
    )
    for forbidden in (
        "make atlas-contract",
        "./scripts/atlas-up.sh",
        "--detach",
        "--track",
        "endpoints ",
        "atlas-connect",
        "docker ",
        "ollama serve",
        "make atlas-up",
        "make atlas-down",
        "curl ",
        "localhost:",
        "127.0.0.1:",
    ):
        assert forbidden not in command_body.lower()


def test_atlas_contract_workflow_contract_is_exact_and_non_live():
    workflow = _load_workflow(REPO / ".github/workflows/atlas-contract.yml")

    _assert_atlas_contract_workflow_contract(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("defaults", {"run": {"shell": "bash"}}),
        ("env", {"PYTEST_ADDOPTS": "-k atlas"}),
    ],
)
def test_atlas_contract_workflow_rejects_workflow_level_controls(field, value):
    workflow = _valid_atlas_contract_workflow()
    workflow[field] = value

    with pytest.raises(AssertionError):
        _assert_atlas_contract_workflow_contract(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("defaults", {"run": {"shell": "bash"}}),
        ("env", {"PYTEST_ADDOPTS": "-k atlas"}),
        ("if", "github.ref == 'refs/heads/main'"),
        ("needs", "verify-repo"),
        ("services", {"ollama": {"image": "ollama/ollama"}}),
        ("container", "python:3.11"),
        ("continue-on-error", "true"),
    ],
)
def test_atlas_contract_workflow_rejects_job_level_controls(field, value):
    workflow = _valid_atlas_contract_workflow()
    workflow["jobs"]["atlas-contract"][field] = value

    with pytest.raises(AssertionError):
        _assert_atlas_contract_workflow_contract(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("if", "github.ref == 'refs/heads/main'"),
        ("env", {"PYTEST_ADDOPTS": "-q"}),
        ("shell", "bash {0} || true"),
        ("continue-on-error", "true"),
        ("timeout-minutes", "5"),
    ],
)
def test_atlas_contract_workflow_rejects_step_level_controls(field, value):
    workflow = _valid_atlas_contract_workflow()
    workflow["jobs"]["atlas-contract"]["steps"][2][field] = value

    with pytest.raises(AssertionError):
        _assert_atlas_contract_workflow_contract(workflow)


@pytest.mark.parametrize("mutation", ["extra", "reordered"])
def test_atlas_contract_workflow_rejects_changed_step_inventory(mutation):
    workflow = _valid_atlas_contract_workflow()
    steps = workflow["jobs"]["atlas-contract"]["steps"]
    if mutation == "extra":
        steps.append({"name": "Extra", "run": "true"})
    else:
        steps[0], steps[1] = steps[1], steps[0]

    with pytest.raises(AssertionError):
        _assert_atlas_contract_workflow_contract(workflow)


@pytest.mark.parametrize("step_index", [0, 1])
def test_atlas_contract_workflow_rejects_unpinned_action_mutations(step_index):
    workflow = _valid_atlas_contract_workflow()
    step = workflow["jobs"]["atlas-contract"]["steps"][step_index]
    step["uses"] = step["uses"].split("@", 1)[0] + "@main"

    with pytest.raises(AssertionError):
        _assert_atlas_contract_workflow_contract(workflow)


def test_atlas_contract_workflow_rejects_uv_pin_mutation():
    workflow = _valid_atlas_contract_workflow()
    install = workflow["jobs"]["atlas-contract"]["steps"][2]
    install["run"] = "python -m pip install uv==0.11.18"

    with pytest.raises(AssertionError):
        _assert_atlas_contract_workflow_contract(workflow)


def test_atlas_contract_workflow_rejects_validation_body_drift():
    workflow = _valid_atlas_contract_workflow()
    validation = workflow["jobs"]["atlas-contract"]["steps"][3]
    validation["run"] = f"{validation['run']}printf '%s\\n' done\n"

    with pytest.raises(AssertionError):
        _assert_atlas_contract_workflow_contract(workflow)


@pytest.mark.parametrize(
    "parent_command",
    [
        "pip install -r atlas-contract-requirements.txt",
        "shellcheck scripts/atlas-up.sh scripts/atlas-down.sh",
        "pytest tests/test_atlas_consumer_contract.py",
        "make test-atlas-consumer",
    ],
)
def test_atlas_contract_workflow_rejects_parent_policy_boundary_collapse(
    parent_command,
):
    workflow = _valid_atlas_contract_workflow()
    validation = workflow["jobs"]["atlas-contract"]["steps"][3]
    validation["run"] = f"{validation['run']}{parent_command}\n"

    with pytest.raises(AssertionError):
        _assert_atlas_contract_workflow_contract(workflow)


@pytest.mark.parametrize(
    "live_command",
    [
        "docker ps",
        "ollama serve",
        "make atlas-up",
        "make atlas-down",
        "curl http://example.invalid",
        "probe localhost:63030",
        "probe 127.0.0.1:63040",
    ],
)
def test_atlas_contract_workflow_rejects_live_runtime_commands(live_command):
    workflow = _valid_atlas_contract_workflow()
    validation = workflow["jobs"]["atlas-contract"]["steps"][3]
    validation["run"] = f"{validation['run']}{live_command}\n"

    with pytest.raises(AssertionError):
        _assert_atlas_contract_workflow_contract(workflow)


def test_ci_runs_repository_workflow_contract_tests():
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    steps = workflow["jobs"]["verify-repo"]["steps"]
    contract_tests = next(
        step
        for step in steps
        if step.get("name") == "Test repository workflow contracts"
    )
    assert contract_tests["run"] == (
        "pytest "
        "tests/test_verify_repo.py::test_ci_repository_test_contract_enforces_canonical_nnx_wheel "
        "tests/test_verify_repo.py::test_ci_nnx_surface_job_enforces_canonical_wheel_contract "
        "tests/test_verify_repo.py::test_ci_dependency_audit_job_contract -q\n"
        "pytest tests/test_verify_repo.py -q -k "
        "'atlas_consumer_policy_contract or "
        "atlas_contract_workflow or "
        "dependency_audit or "
        "atlas_docs_preserve_mounted_workspace_and_track_ownership or "
        "ci_covers_gitflow_pr_targets or "
        "ci_tier_a_uses_temporary_outputs_and_preserves_sources or "
        "documentation_workflows_install_cairo_and_gate_pages_inputs or "
        "documentation_direct_dependencies_are_exactly_pinned or "
        "docs_workflow_covers_atlas_metadata_inputs_and_parser_tests or "
        "ci_verify_repo_submodule_contract or "
        "ci_runs_repository_workflow_contract_tests or "
        "ci_runs_complete_repository_test_contract or "
        "ci_repository_test_contract_enforces_canonical_nnx_wheel or "
        "ci_nnx_surface_job_enforces_canonical_wheel_contract or "
        "repository_test_collection_boundary_is_explicit'\n"
    )


@pytest.mark.parametrize(
    "positive_test",
    (
        "test_ci_repository_test_contract_enforces_canonical_nnx_wheel",
        "test_ci_nnx_surface_job_enforces_canonical_wheel_contract",
        "test_ci_dependency_audit_job_contract",
    ),
    ids=("repository-gate", "focused-gate", "dependency-audit-gate"),
)
@pytest.mark.parametrize("mutation", ("delete", "rename"))
def test_ci_workflow_contract_self_test_resists_positive_test_deletion(
    tmp_path: Path, positive_test: str, mutation: str
):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    command = next(
        step["run"]
        for step in workflow["jobs"]["verify-repo"]["steps"]
        if step.get("name") == "Test repository workflow contracts"
    ).splitlines()[0]
    source = (REPO / "tests" / "test_verify_repo.py").read_text(encoding="utf-8")
    function_header = f"def {positive_test}():"
    assert source.count(function_header) == 1
    assert "tests/test_verify_repo.py::test_ci_dependency_audit_job_contract" in command

    test_file = tmp_path / "tests" / "test_verify_repo.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(source, encoding="utf-8")
    workflow_path = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow_path.parent.mkdir(parents=True)
    workflow_path.write_text(yaml.safe_dump(workflow), encoding="utf-8")

    environment = os.environ.copy()
    source_path = str(REPO)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else f"{source_path}{os.pathsep}{existing_pythonpath}"
    )
    control = subprocess.run(
        shlex.split(command),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
        env=environment,
    )
    assert control.returncode == 0, control.stdout + control.stderr

    if mutation == "rename":
        source = source.replace(function_header, f"def removed_{positive_test}():", 1)
    else:
        source = source.replace(function_header, f"def _deleted_{positive_test}():", 1)
    test_file.write_text(source, encoding="utf-8")

    result = subprocess.run(
        shlex.split(command),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
        env=environment,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert positive_test in output
    assert "not found:" in output.lower()
    assert "modulenotfounderror" not in output.lower()
    assert "importerror" not in output.lower()
    assert "error collecting" not in output.lower()


def _assert_no_nnx_environment_overrides(workflow: dict) -> None:
    forbidden = {"NNX_ALLOW_EDITABLE", "PYTHONPATH"}
    assert forbidden.isdisjoint(workflow.get("env", {}))
    for job in workflow["jobs"].values():
        assert forbidden.isdisjoint(job.get("env", {}))
        for step in job.get("steps", []):
            assert forbidden.isdisjoint(step.get("env", {}))

    def semantic_scalars(value, *, command_context=False):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield from semantic_scalars(
                    nested,
                    command_context=command_context or key in {"run", "shell", "with"},
                )
        elif isinstance(value, list):
            for nested in value:
                yield from semantic_scalars(nested, command_context=command_context)
        elif command_context and isinstance(value, str):
            yield value

    variables = "|".join(sorted(map(re.escape, forbidden)))
    variable_use = re.compile(
        rf"(?:"
        rf"(?<![A-Za-z0-9_])(?:{variables})\s*\+?="
        rf"|\bexport[ \t]+(?:[A-Za-z_][A-Za-z0-9_]*(?:=[^\s;|&]*)?[ \t]+)*"
        rf"(?:{variables})(?:\s*=|\b)"
        rf"|\$(?:{variables})\b"
        rf"|\$\{{(?:{variables})(?:[^}}]*)\}}"
        rf"|\$\{{\{{\s*env\s*(?:\.\s*(?:{variables})\b|\[\s*['\"](?:{variables})['\"]\s*\])"
        rf")"
    )
    assert all(not variable_use.search(value) for value in semantic_scalars(workflow))


def _assert_complete_repository_test_contract(workflow: dict) -> None:
    _assert_no_nnx_environment_overrides(workflow)
    assert "defaults" not in workflow
    assert "env" not in workflow

    job = workflow["jobs"]["pytest-repository"]

    assert set(job) == {"name", "runs-on", "timeout-minutes", "steps"}
    assert job["name"] == "pytest-repository"
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == "15"
    assert "if" not in job
    assert "continue-on-error" not in job
    assert "services" not in job
    assert "container" not in job

    assert job["steps"] == [
        {
            "name": "Checkout",
            "uses": "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0",
            "with": {
                "persist-credentials": "false",
                "submodules": "recursive",
            },
        },
        {
            "name": "Install system dependencies for cairosvg",
            "run": "sudo apt-get update && sudo apt-get install -y libcairo2",
        },
        {
            "name": "Set up Python 3.11",
            "uses": "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
            "with": {
                "python-version": "3.11.15",
                "cache": "pip",
                "cache-dependency-path": _cache_text(_STACK_CACHE_MANIFESTS),
            },
        },
        {
            "name": "Install dependencies",
            "run": "make install-torch-stack",
        },
        {
            "name": "Check and verify canonical Torch and NNx stack",
            "run": (
                "python -m pip check\n"
                "make verify-torch-stack\n"
                "make verify-nnx-install\n"
            ),
        },
        {
            "name": "Run complete repository tests",
            "run": "make test",
        },
    ]


def test_ci_runs_complete_repository_test_contract():
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")

    _assert_complete_repository_test_contract(workflow)


def test_ci_repository_test_contract_enforces_canonical_nnx_wheel():
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")

    _assert_complete_repository_test_contract(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("needs", "verify-repo"),
        ("defaults", {"run": {"shell": "bash"}}),
        ("env", {"PYTEST_ADDOPTS": "-k smoke"}),
    ],
)
def test_ci_runs_complete_repository_test_contract_rejects_job_level_controls(
    field,
    value,
):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    workflow["jobs"]["pytest-repository"][field] = value

    with pytest.raises(AssertionError):
        _assert_complete_repository_test_contract(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("defaults", {"run": {"shell": "bash"}}),
        ("env", {"PYTEST_ADDOPTS": "-k smoke"}),
        ("env", {"BASH_ENV": "/tmp/ci-env"}),
        ("env", {"PATH": "/tmp/bin"}),
    ],
)
def test_ci_runs_complete_repository_test_contract_rejects_workflow_level_controls(
    field,
    value,
):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    workflow[field] = value

    with pytest.raises(AssertionError):
        _assert_complete_repository_test_contract(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("if", "github.ref == 'refs/heads/main'"),
        ("shell", "bash {0} || true"),
    ],
)
def test_ci_runs_complete_repository_test_contract_rejects_conditional_or_masked_step(
    field,
    value,
):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    complete = next(
        step
        for step in workflow["jobs"]["pytest-repository"]["steps"]
        if step.get("name") == "Run complete repository tests"
    )
    complete[field] = value

    with pytest.raises(AssertionError):
        _assert_complete_repository_test_contract(workflow)


@pytest.mark.parametrize(
    ("step_name", "field", "value"),
    [
        ("Checkout", "env", {"ACTIONS_STEP_DEBUG": "true"}),
        (
            "Install system dependencies for cairosvg",
            "continue-on-error",
            "true",
        ),
        ("Set up Python 3.11", "if", "github.ref == 'refs/heads/main'"),
        ("Set up Python 3.11", "shell", "bash {0} || true"),
        ("Install dependencies", "if", "github.ref == 'refs/heads/main'"),
        ("Install dependencies", "shell", "bash {0} || true"),
        ("Run complete repository tests", "env", {"PYTEST_ADDOPTS": "-q"}),
    ],
)
def test_ci_runs_complete_repository_test_contract_rejects_extra_step_metadata(
    step_name,
    field,
    value,
):
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    step = next(
        step
        for step in workflow["jobs"]["pytest-repository"]["steps"]
        if step.get("name") == step_name
    )
    step[field] = value

    with pytest.raises(AssertionError):
        _assert_complete_repository_test_contract(workflow)


def test_ci_runs_complete_repository_test_contract_rejects_false_cairo_echo():
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    cairo = next(
        step
        for step in workflow["jobs"]["pytest-repository"]["steps"]
        if step.get("name") == "Install system dependencies for cairosvg"
    )
    cairo["run"] = "echo libcairo2"

    with pytest.raises(AssertionError):
        _assert_complete_repository_test_contract(workflow)


def _assert_nnx_surface_job_contract(workflow: dict) -> None:
    _assert_no_nnx_environment_overrides(workflow)
    assert "defaults" not in workflow
    assert "env" not in workflow

    job = workflow["jobs"]["pytest-nnx-surface"]

    assert set(job) == {"runs-on", "timeout-minutes", "steps"}
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == "15"
    assert job["steps"] == [
        {
            "name": "Checkout",
            "uses": "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0",
            "with": {"persist-credentials": "false"},
        },
        {
            "name": "Set up Python 3.11",
            "uses": "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
            "with": {
                "python-version": "3.11.15",
                "cache": "pip",
                "cache-dependency-path": _cache_text(_STACK_CACHE_MANIFESTS),
            },
        },
        {"name": "Install dependencies", "run": "make install-torch-stack"},
        {"name": "Lint (ruff check)", "run": "make lint"},
        {
            "name": "Check and verify canonical Torch and NNx stack",
            "run": (
                "python -m pip check\n"
                "make verify-torch-stack\n"
                "make verify-nnx-install\n"
            ),
        },
        {
            "name": "Run NNx-surface tests",
            "run": (
                "pytest -p no:cacheprovider -W error "
                "--junitxml=/tmp/nnx-surface.xml tests/nnx_surface -v\n"
                "python -m scripts.verify_junit /tmp/nnx-surface.xml\n"
            ),
        },
    ]


def test_ci_nnx_surface_job_enforces_canonical_wheel_contract():
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")

    _assert_nnx_surface_job_contract(workflow)


def _valid_nnx_contract_workflow() -> dict:
    return _load_workflow(REPO / ".github/workflows/ci.yml")


@pytest.mark.parametrize("variable", ["NNX_ALLOW_EDITABLE", "PYTHONPATH"])
def test_ci_nnx_contract_rejects_provenance_environment_overrides_in_other_jobs(variable):
    workflow = _valid_nnx_contract_workflow()
    workflow["jobs"]["atlas-consumer-policy"]["env"] = {variable: "1"}

    with pytest.raises(AssertionError):
        _assert_nnx_surface_job_contract(workflow)


@pytest.mark.parametrize("variable", ["NNX_ALLOW_EDITABLE", "PYTHONPATH"])
@pytest.mark.parametrize(
    ("field", "value_template"),
    [
        ("run", "{variable}=1 make test-atlas-consumer"),
        ("run", "env {variable}=1 make test-atlas-consumer"),
        ("run", "export {variable}=1\nmake test-atlas-consumer"),
        ("run", "export {variable}\nmake test-atlas-consumer"),
        ("run", "export OTHER_VARIABLE {variable}\nmake test-atlas-consumer"),
        ("run", 'printf \'%s\\n\' "${{{variable}}}"'),
        ("run", 'printf \'%s\\n\' "${variable}"'),
        ("run", 'printf \'%s\\n\' "${{{{ env.{variable} }}}}"'),
        ("shell", "env {variable}=1 bash -e {{0}}"),
        ("with", "{variable}=1 make test-atlas-consumer"),
    ],
)
def test_ci_nnx_surface_job_enforces_canonical_wheel_contract_rejects_inline_environment_escapes(
    variable,
    field,
    value_template,
):
    workflow = _valid_nnx_contract_workflow()
    step = workflow["jobs"]["atlas-consumer-policy"]["steps"][-1]
    value = value_template.format(variable=variable)
    if field == "with":
        step[field] = {"args": value}
    else:
        step[field] = value

    with pytest.raises(AssertionError):
        _assert_nnx_surface_job_contract(workflow)


@pytest.mark.parametrize(
    "command",
    [
        "NNX_ALLOW_EDITABLE+=1 make test-atlas-consumer",
        "PYTHONPATH+=/tmp/escape make test-atlas-consumer",
    ],
)
def test_ci_nnx_surface_job_enforces_canonical_wheel_contract_rejects_compound_environment_overrides(
    command,
):
    workflow = _valid_nnx_contract_workflow()
    step = workflow["jobs"]["atlas-consumer-policy"]["steps"][-1]
    step["run"] = command

    with pytest.raises(AssertionError):
        _assert_nnx_surface_job_contract(workflow)


def test_ci_nnx_surface_job_enforces_canonical_wheel_contract_allows_identifier_prose():
    workflow = _valid_nnx_contract_workflow()
    steps = workflow["jobs"]["atlas-consumer-policy"]["steps"]
    steps[-1]["name"] = "Test NNX_ALLOW_EDITABLE and PYTHONPATH policy"
    steps[-1]["run"] = (
        "pytest tests/test_verify_repo.py -q "
        "-k 'NNX_ALLOW_EDITABLE or PYTHONPATH'"
    )
    steps[0]["with"]["policy-note"] = (
        "NNX_ALLOW_EDITABLE and PYTHONPATH are forbidden in CI"
    )

    _assert_nnx_surface_job_contract(workflow)


@pytest.mark.parametrize(
    ("job_name", "assert_contract"),
    [
        ("pytest-repository", _assert_complete_repository_test_contract),
        ("pytest-nnx-surface", _assert_nnx_surface_job_contract),
    ],
)
@pytest.mark.parametrize(
    "install_command",
    [
        "python -m pip install -r requirements.txt",
        "python -m pip install --only-binary=:all: -r requirements.txt",
        "python -m pip install -e .",
        "python -m pip install git+https://example.invalid/thekaveh-nnx.git",
    ],
)
def test_ci_nnx_jobs_reject_noncanonical_install_commands(
    job_name,
    assert_contract,
    install_command,
):
    workflow = _valid_nnx_contract_workflow()
    install = next(
        step
        for step in workflow["jobs"][job_name]["steps"]
        if step.get("name") == "Install dependencies"
    )
    install["run"] = f"{install['run']}\n{install_command}"

    with pytest.raises(AssertionError):
        assert_contract(workflow)


@pytest.mark.parametrize(
    ("scope", "variable"),
    [
        ("workflow", "NNX_ALLOW_EDITABLE"),
        ("workflow", "PYTHONPATH"),
        ("job", "NNX_ALLOW_EDITABLE"),
        ("job", "PYTHONPATH"),
        ("step", "NNX_ALLOW_EDITABLE"),
        ("step", "PYTHONPATH"),
    ],
)
@pytest.mark.parametrize(
    ("job_name", "assert_contract"),
    [
        ("pytest-repository", _assert_complete_repository_test_contract),
        ("pytest-nnx-surface", _assert_nnx_surface_job_contract),
    ],
)
def test_ci_nnx_jobs_reject_provenance_environment_overrides(
    job_name,
    assert_contract,
    scope,
    variable,
):
    workflow = _valid_nnx_contract_workflow()
    job = workflow["jobs"][job_name]
    if scope == "workflow":
        workflow["env"] = {variable: "1"}
    elif scope == "job":
        job["env"] = {variable: "1"}
    else:
        verifier = next(
            step
            for step in job["steps"]
            if step.get("name") == "Check and verify canonical Torch and NNx stack"
        )
        verifier["env"] = {variable: "1"}

    with pytest.raises(AssertionError):
        assert_contract(workflow)


@pytest.mark.parametrize(
    ("job_name", "assert_contract"),
    [
        ("pytest-repository", _assert_complete_repository_test_contract),
        ("pytest-nnx-surface", _assert_nnx_surface_job_contract),
    ],
)
def test_ci_nnx_jobs_reject_removed_or_reordered_verifier(job_name, assert_contract):
    workflow = _valid_nnx_contract_workflow()
    steps = workflow["jobs"][job_name]["steps"]
    verifier_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Check and verify canonical Torch and NNx stack"
    )
    verifier = steps.pop(verifier_index)

    with pytest.raises(AssertionError):
        assert_contract(workflow)

    steps.insert(verifier_index, verifier)
    install_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Install dependencies"
    )
    steps.insert(install_index, steps.pop(verifier_index))

    with pytest.raises(AssertionError):
        assert_contract(workflow)


@pytest.mark.parametrize(
    ("job_name", "assert_contract", "test_step_name"),
    [
        (
            "pytest-repository",
            _assert_complete_repository_test_contract,
            "Run complete repository tests",
        ),
        ("pytest-nnx-surface", _assert_nnx_surface_job_contract, "Run NNx-surface tests"),
    ],
)
@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("job", "if", "github.ref == 'refs/heads/main'"),
        ("job", "services", {"postgres": {"image": "postgres"}}),
        ("job", "container", "python:3.11"),
        ("verifier", "continue-on-error", "true"),
        ("verifier", "shell", "bash {0} || true"),
        ("test", "if", "github.ref == 'refs/heads/main'"),
        ("test", "run", "pytest -q"),
    ],
)
def test_ci_nnx_jobs_reject_controls_and_weakened_workloads(
    job_name,
    assert_contract,
    test_step_name,
    target,
    field,
    value,
):
    workflow = _valid_nnx_contract_workflow()
    job = workflow["jobs"][job_name]
    if target == "job":
        job[field] = value
    else:
        step_name = (
            "Check and verify canonical Torch and NNx stack"
            if target == "verifier"
            else test_step_name
        )
        step = next(step for step in job["steps"] if step.get("name") == step_name)
        step[field] = value

    with pytest.raises(AssertionError):
        assert_contract(workflow)


@pytest.mark.parametrize(
    ("job_name", "assert_contract"),
    [
        ("pytest-repository", _assert_complete_repository_test_contract),
        ("pytest-nnx-surface", _assert_nnx_surface_job_contract),
    ],
)
def test_ci_nnx_jobs_reject_extra_steps(job_name, assert_contract):
    workflow = _valid_nnx_contract_workflow()
    job = workflow["jobs"][job_name]
    job["steps"].insert(-1, {"name": "Extra validation", "run": "true"})

    with pytest.raises(AssertionError):
        assert_contract(workflow)


def _assert_repository_test_collection_boundary(repo: Path) -> None:
    assert not [
        name
        for name in (
            "pytest.ini",
            ".pytest.ini",
            "pytest.toml",
            ".pytest.toml",
            "tox.ini",
            "setup.cfg",
        )
        if (repo / name).exists()
    ]

    config = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    pytest_config = config["tool"]["pytest"]["ini_options"]

    assert pytest_config["testpaths"] == ["tests"]
    assert {"infra", "notebooks/archive", ".venv"} <= set(
        pytest_config["norecursedirs"]
    )
    assert pytest_config.get("addopts", "") in ("", [])

    make = subprocess.run(
        ["make", "--no-print-directory", "-n", "test"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    assert make.stdout.splitlines() == ["pytest tests/ -v"]


def test_repository_test_collection_boundary_is_explicit():
    _assert_repository_test_collection_boundary(REPO)


def _copy_repository_test_collection_contract(tmp_path: Path) -> Path:
    for name in ("Makefile", "pyproject.toml"):
        (tmp_path / name).write_text(
            (REPO / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return tmp_path


def test_repository_test_collection_boundary_is_explicit_for_effective_make_target(
    tmp_path,
):
    repo = _copy_repository_test_collection_contract(tmp_path)
    with (repo / "Makefile").open("a", encoding="utf-8") as makefile:
        makefile.write("\ntest:\n\tpytest tests/test_verify_repo.py -v\n")

    with pytest.raises(AssertionError):
        _assert_repository_test_collection_boundary(repo)


@pytest.mark.parametrize("addopts", ["-k smoke", "--collect-only"])
def test_repository_test_collection_boundary_is_explicit_without_nonempty_addopts(
    tmp_path,
    addopts,
):
    repo = _copy_repository_test_collection_contract(tmp_path)
    pyproject = repo / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'pythonpath = ["."]',
            f'pythonpath = ["."]\naddopts = "{addopts}"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError):
        _assert_repository_test_collection_boundary(repo)


def test_repository_test_collection_boundary_is_explicit_with_empty_addopts(tmp_path):
    repo = _copy_repository_test_collection_contract(tmp_path)
    pyproject = repo / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'pythonpath = ["."]',
            'pythonpath = ["."]\naddopts = ""',
        ),
        encoding="utf-8",
    )

    _assert_repository_test_collection_boundary(repo)


@pytest.mark.parametrize(
    "config_name",
    [
        "pytest.ini",
        ".pytest.ini",
        "pytest.toml",
        ".pytest.toml",
        "tox.ini",
        "setup.cfg",
    ],
)
def test_repository_test_collection_boundary_is_explicit_without_higher_precedence_config(
    tmp_path,
    config_name,
):
    repo = _copy_repository_test_collection_contract(tmp_path)
    (repo / config_name).write_text("[pytest]\ntestpaths = selected\n", encoding="utf-8")

    with pytest.raises(AssertionError):
        _assert_repository_test_collection_boundary(repo)


def test_ci_covers_gitflow_pr_targets():
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")

    assert set(workflow["on"]["push"]["branches"]) == {"develop", "main"}
    assert set(workflow["on"]["pull_request"]["branches"]) == {"develop", "main"}
    assert workflow["on"]["pull_request"]["types"] == [
        "opened",
        "synchronize",
        "reopened",
        "labeled",
    ]


def test_documentation_workflows_install_cairo_and_gate_pages_inputs():
    ci = _load_workflow(REPO / ".github/workflows/ci.yml")
    docs = _load_workflow(REPO / ".github/workflows/docs.yml")
    pages = _load_workflow(REPO / ".github/workflows/pages.yml")

    for steps in (
        ci["jobs"]["docs-build"]["steps"],
        docs["jobs"]["check"]["steps"],
        pages["jobs"]["build"]["steps"],
        pages["jobs"]["wiki"]["steps"],
    ):
        assert any("libcairo2" in step.get("run", "") for step in steps)

    required_paths = {
        "*.md",
        ".gitmodules",
        "infra",
        "atlas.consumer.yml",
        "atlas.env.user.example",
        "compose/**",
        "scripts/atlas-*.sh",
        "scripts/lib/atlas-dotenv.sh",
        "docs-requirements.in",
        ".github/workflows/pages.yml",
        "security/accepted-advisories.json",
        "vulnerability-audit-requirements.txt",
        "torch-audit-requirements.txt",
        "pyg-extension-audit-requirements.txt",
        "scripts/advisory_baseline.py",
        "tests/test_advisory_baseline.py",
    }
    assert required_paths <= set(docs["on"]["pull_request"]["paths"])
    assert any(
        step.get("run") == "make docs-check"
        for step in pages["jobs"]["build"]["steps"]
    )


def test_documentation_direct_dependencies_are_exactly_pinned():
    requirements = {
        line
        for line in (REPO / "docs-requirements.in").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }

    assert requirements == {
        "mkdocs-material==9.7.7",
        "pyyaml==6.0.3",
        "cairosvg==2.9.0",
        "ruff==0.9.10",
        "pytest==9.0.3",
    }


def test_ci_tier_a_uses_temporary_outputs_and_preserves_sources():
    verify_repo = _load_verify_module()
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    steps = workflow["jobs"]["tier-a-papermill"]["steps"]

    execute = next(step for step in steps if step.get("name") == "Run Tier-A notebooks (papermill)")
    artifacts = next(
        step for step in steps if step.get("name") == "Check Tier-A temporary notebook outputs"
    )
    clean = next(step for step in steps if step.get("name") == "Check Tier-A source notebooks are unchanged")
    artifact = next(step for step in steps if step.get("name") == "Upload refreshed notebook outputs as artifact")
    artifact_paths = tuple(
        line.strip()
        for line in artifact["with"]["path"].splitlines()
        if line.strip()
    )

    assert execute["run"] == "make smoke-tier-a"
    assert artifacts["run"] == "make check-tier-a-artifacts"
    assert clean["run"] == "make check-tier-a-clean"
    assert artifact["with"]["if-no-files-found"] == "error"
    assert artifact_paths == tuple(
        f"/tmp/ml-tier-a/{notebook}" for notebook in verify_repo.TIER_A_NOTEBOOKS
    )
    assert "TIER_A_OUT ?= /tmp/ml-tier-a" in (REPO / "Makefile").read_text(encoding="utf-8")


_TIER_NNX_CONTRACTS = {
    "tier-a-papermill": {
        "workload": {"name": "Run Tier-A notebooks (papermill)", "run": "make smoke-tier-a"},
        "install": (
            "make install-torch-stack\n"
            "make nlp-assets\n"
            "make verify-nlp-assets\n"
        ),
    },
    "smoke-tier-b": {
        "workload": {"name": "Smoke-run Tier-B notebooks", "run": "make smoke-tier-b"},
        "install": "make install-torch-stack",
    },
    "smoke-tier-c": {
        "workload": {"name": "Smoke-run Tier-C notebooks", "run": "make smoke-tier-c"},
        "install": "make install-torch-stack",
    },
}
_LIVE_SERVICE_COMMANDS = (
    "docker run",
    "docker compose",
    "docker-compose",
    "atlas-up",
    "jupyterhub",
    "ollama",
    "comfyui",
)


def _assert_tier_nnx_provenance_contract(workflow: dict, job_name: str) -> None:
    _assert_no_nnx_environment_overrides(workflow)
    _assert_runtime_job_install_contract(workflow, job_name)
    _assert_tier_output_contract(workflow, job_name)
    job = workflow["jobs"][job_name]
    contract = _TIER_NNX_CONTRACTS[job_name]

    expected_conditions = {
        "tier-a-papermill": None,
        "smoke-tier-b": (
            "github.event_name == 'workflow_dispatch'\n"
            "|| github.event_name == 'schedule'\n"
            "|| contains(github.event.pull_request.labels.*.name, 'tier-b-smoke')\n"
        ),
        "smoke-tier-c": (
            "github.event_name == 'workflow_dispatch' || "
            "github.event_name == 'schedule'"
        ),
    }
    expected_condition = expected_conditions[job_name]
    if expected_condition is None:
        assert "if" not in job
    else:
        assert job["if"] == expected_condition

    assert "container" not in job
    assert "services" not in job
    assert "continue-on-error" not in job
    assert all("continue-on-error" not in step for step in job["steps"])
    for step in job["steps"]:
        run = step.get("run", "").lower()
        assert not any(command in run for command in _LIVE_SERVICE_COMMANDS)

    install_index = next(
        index
        for index, step in enumerate(job["steps"])
        if step.get("name") == "Install dependencies"
    )
    assert job["steps"][install_index] == {
        "name": "Install dependencies",
        "run": contract["install"],
    }
    verifier_index = next(
        index
        for index, step in enumerate(job["steps"])
        if step.get("name") == "Check and verify canonical Torch and NNx stack"
    )
    assert job["steps"][verifier_index] == {
        "name": "Check and verify canonical Torch and NNx stack",
        "run": (
            "python -m pip check\n"
            "make verify-torch-stack\n"
            "make verify-nnx-install\n"
        ),
    }
    assert job["steps"][verifier_index + 1] == contract["workload"]
    assert all(
        "pip install" not in step.get("run", "").lower()
        for step in job["steps"][verifier_index + 1 :]
    )
    uploads = [
        step
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    if job_name == "tier-a-papermill":
        assert len(uploads) == 1
        assert uploads[0]["if"] == "always()"
        assert uploads[0]["with"]["if-no-files-found"] == "error"
    else:
        assert not uploads


@pytest.mark.parametrize("job_name", tuple(_TIER_NNX_CONTRACTS))
def test_ci_tier_nnx_provenance_contract(job_name):
    _assert_tier_nnx_provenance_contract(
        _load_workflow(REPO / ".github/workflows/ci.yml"),
        job_name,
    )


@pytest.mark.parametrize("job_name", tuple(_TIER_NNX_CONTRACTS))
@pytest.mark.parametrize(
    "mutation",
    (
        "removed_verifier",
        "late_install",
        "atlas",
        "ollama_container",
        "docker_compose",
        "jupyterhub",
        "comfyui",
        "services",
        "container",
    ),
)
def test_ci_tier_nnx_provenance_contract_rejects_mutations(job_name, mutation):
    workflow = deepcopy(_load_workflow(REPO / ".github/workflows/ci.yml"))
    _assert_tier_nnx_provenance_contract(workflow, job_name)
    job = workflow["jobs"][job_name]
    steps = job["steps"]
    verifier_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Check and verify canonical Torch and NNx stack"
    )
    if mutation == "removed_verifier":
        steps.pop(verifier_index)
    elif mutation == "late_install":
        steps.insert(verifier_index + 1, {"name": "Late install", "run": "pip install -r requirements.txt"})
    elif mutation == "atlas":
        steps.insert(verifier_index, {"name": "Start runtime", "run": "make atlas-up"})
    elif mutation == "ollama_container":
        steps.insert(verifier_index, {"name": "Start model", "run": "docker run -d ollama/ollama"})
    elif mutation == "docker_compose":
        steps.insert(verifier_index, {"name": "Start dependencies", "run": "docker compose up -d"})
    elif mutation == "jupyterhub":
        steps.insert(verifier_index, {"name": "Start notebook", "run": "jupyterhub"})
    elif mutation == "comfyui":
        steps.insert(verifier_index, {"name": "Start image UI", "run": "comfyui --listen"})
    elif mutation == "services":
        job["services"] = {"cache": {"image": "redis"}}
    elif mutation == "container":
        job["container"] = "python:3.11"
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")

    with pytest.raises((AssertionError, StopIteration, ValueError)):
        _assert_tier_nnx_provenance_contract(workflow, job_name)


def test_atlas_docs_preserve_mounted_workspace_and_track_ownership():
    numpy_spec = yaml.safe_load(
        (REPO / "notebooks/image_classification-mnist-ffnn-numpy/docs/spec.yaml").read_text(
            encoding="utf-8"
        )
    )
    constraint = " ".join(numpy_spec["atlas"]["constraints"])
    numpy_readme = (REPO / "notebooks/image_classification-mnist-ffnn-numpy/README.md").read_text(
        encoding="utf-8"
    )
    jupyterhub = (REPO / "docs/jupyterhub-integration.md").read_text(encoding="utf-8")
    vscode = (REPO / "docs/vscode-remote-access.md").read_text(encoding="utf-8")
    environment = (REPO / "docs/env-setup.md").read_text(encoding="utf-8")
    contributing = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")

    mounted_editor = "Browser JupyterLab or VS Code attached to the JupyterHub container"
    assert numpy_spec["atlas"]["default_mode"] == "mounted-workspace"
    assert numpy_spec["atlas"]["workspace_access"] == "mounted-required"
    assert mounted_editor in constraint
    for document in (jupyterhub, vscode, contributing, numpy_readme):
        assert mounted_editor in " ".join(document.split())

    track_owner = "`scripts/atlas-up.sh` supplies `--track ml-eng`"
    assert track_owner in jupyterhub
    assert track_owner in environment
    assert "`atlas:` mapping" in contributing
    assert "`make docs-sync-notebook-infrastructure`" in contributing
    assert "future-service admission" in contributing


def test_atlas_consumer_policy_docs_define_ci_boundaries():
    expected_phrases = {
        "CONTRIBUTING.md": (
            "`make test-atlas-consumer`",
            "`atlas-consumer-policy` is unconditional on every pull request and is "
            "intended to be a required gate",
            "`atlas-contract` remains a separate, path-scoped, non-required direct "
            "validator of the recursive `infra/` submodule",
            ),
            "docs/conventions.md": (
                "The `atlas-consumer-policy` job",
                "sets up exact Python 3.11.15, installs the bootstrap lock and hash-required "
                "Atlas contract lock",
                "routine execution consumes `requirements/locks/atlas-contract.txt`",
            "`shellcheck scripts/atlas-up.sh scripts/atlas-down.sh "
            "scripts/atlas-connect.sh scripts/lib/atlas-dotenv.sh`",
            "`make test-atlas-consumer`",
            "does not start, stop, or contact Atlas, JupyterHub, Ollama, ComfyUI, "
            "Docker Compose, or unrelated containers",
            "complete `make test`",
        ),
            "docs/jupyterhub-integration.md": (
                "Changes to the parent wrapper, runtime probe, dotenv helper, Atlas policy "
                "tests, or focused dependency input/lock reach both checks",
            "`atlas-consumer-policy` runs unconditionally on every pull request and is "
            "intended to be required",
            "path-scoped `atlas-contract` directly validates the recursive submodule and "
            "is not a required check",
            "CI never starts or contacts live services",
            "`ollama-localhost` is the only allowed Ollama source",
            "The only allowed ComfyUI modes are `disabled`, `localhost`, and "
            "`managed-localhost-MPS`",
            "containerized Ollama and ComfyUI sources remain prohibited",
        ),
        "docs/architecture.md": (
            "unconditional `atlas-consumer-policy` job is intended to be a required gate",
            "path-scoped `atlas-contract` remains the non-required direct "
            "recursive-submodule validator",
        ),
        "CHANGELOG.md": (
            "`atlas-consumer-policy` gate now runs unconditionally on every pull request",
            "path-scoped, non-required `atlas-contract` direct validator",
            "`make test-atlas-consumer`",
            "never starts or contacts live services",
        ),
    }

    for relative_path, phrases in expected_phrases.items():
        content = " ".join(
            (REPO / relative_path).read_text(encoding="utf-8").split()
        )
        for phrase in phrases:
            assert phrase in content, f"{relative_path} is missing {phrase!r}"


def test_docs_workflow_covers_atlas_metadata_inputs_and_parser_tests():
    workflow = _load_workflow(REPO / ".github/workflows/docs.yml")
    paths = set(workflow["on"]["pull_request"]["paths"])
    assert {
        "docs/manifest.yaml",
        "notebooks/**/docs/spec.yaml",
        "scripts/docs/notebook_infrastructure.py",
        "tests/test_notebook_infrastructure.py",
    } <= paths
    steps = workflow["jobs"]["check"]["steps"]
    unit_tests = next(step for step in steps if step.get("name") == "Unit tests (docs scripts)")
    assert "tests/test_notebook_infrastructure.py" in unit_tests["run"].split()


def test_docs_workflow_watches_all_root_markdown():
    workflow = _load_workflow(REPO / ".github/workflows/docs.yml")

    assert "*.md" in set(workflow["on"]["pull_request"]["paths"])


def test_e6_shellcheck_targets_include_only_parent_owned_scripts():
    verify_repo = _load_verify_module()

    targets = {
        str(path.relative_to(REPO))
        for path in verify_repo._shellcheck_targets(REPO)
    }

    assert "scripts/atlas-up.sh" in targets
    assert "scripts/atlas-down.sh" in targets
    assert "scripts/atlas-connect.sh" in targets
    assert not any(target.startswith(("infra/", "vendor/")) for target in targets)


def test_e6_flags_required_parent_shellcheck_target_without_executable_bit(
    tmp_path, monkeypatch
):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    script = repo / "scripts/atlas-down.sh"
    script.chmod(0o644)
    _prepare_atlas_execution_check(monkeypatch, module, repo)

    result = module.check_execution(repo, fast=True)

    hits = [
        finding
        for finding in result.findings
        if finding.id == "E6.shellcheck_target_not_executable"
    ]
    assert [finding.location for finding in hits] == ["scripts/atlas-down.sh"]


def test_e6_flags_missing_required_parent_shellcheck_targets(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    repo = _temp_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()

    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ())
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())
    monkeypatch.setattr(verify_repo, "_phase3_code_cells_unchanged", lambda _repo: [])

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["which", "shellcheck"]:
            return 0, "/usr/bin/shellcheck\n", ""
        if cmd and cmd[0] == "shellcheck":
            return 0, "", ""
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)

    result = verify_repo.check_execution(repo, fast=True)

    hits = [f for f in result.findings if f.id == "E6.shellcheck_target_missing"]
    assert hits
    assert {
        "scripts/atlas-up.sh",
        "scripts/atlas-down.sh",
        "scripts/atlas-connect.sh",
    } == {f.location for f in hits}


def test_e6_flags_missing_required_parent_shellcheck_targets_without_shellcheck(
    tmp_path, monkeypatch
):
    verify_repo = _load_verify_module()
    repo = _temp_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()

    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ())
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())
    monkeypatch.setattr(verify_repo, "_phase3_code_cells_unchanged", lambda _repo: [])

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["which", "shellcheck"]:
            return 1, "", ""
        assert not (cmd and cmd[0] == "shellcheck")
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)

    result = verify_repo.check_execution(repo, fast=True)

    missing_binary = [f for f in result.findings if f.id == "E6.shellcheck_missing"]
    assert len(missing_binary) == 1

    missing_targets = [
        f for f in result.findings if f.id == "E6.shellcheck_target_missing"
    ]
    assert {
        "scripts/atlas-up.sh",
        "scripts/atlas-down.sh",
        "scripts/atlas-connect.sh",
    } == {f.location for f in missing_targets}


def test_e6_flags_dirty_required_submodule(monkeypatch):
    verify_repo = _load_verify_module()

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["git", "submodule", "status", "--", "infra"]:
            return 0, "+163134451a19d024e0e1c0df51139fd8c0a2ca52 infra\n", ""
        if cmd == ["which", "shellcheck"]:
            return 1, "", ""
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)
    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ())
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())
    monkeypatch.setattr(verify_repo, "_phase3_code_cells_unchanged", lambda _repo: [])

    result = verify_repo.check_execution(REPO, fast=True)

    hits = [f for f in result.findings if f.id == "E6.submodule_dirty"]
    assert hits
    assert hits[0].location == "infra"


def test_e6_flags_required_submodule_with_modified_worktree(monkeypatch):
    verify_repo = _load_verify_module()
    submodule_cwd = REPO / "infra"

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["git", "submodule", "status", "--", "infra"]:
            return 0, " 163134451a19d024e0e1c0df51139fd8c0a2ca52 infra\n", ""
        if cmd == ["git", "status", "--porcelain", "--", "."]:
            assert cwd == submodule_cwd
            return 0, " M services/jupyterhub/build/requirements.txt\n", ""
        if cmd == ["which", "shellcheck"]:
            return 1, "", ""
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)
    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ())
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())
    monkeypatch.setattr(verify_repo, "_phase3_code_cells_unchanged", lambda _repo: [])

    result = verify_repo.check_execution(REPO, fast=True)

    hits = [f for f in result.findings if f.id == "E6.submodule_dirty"]
    assert hits
    assert hits[0].location == "infra"
    assert "local modifications" in hits[0].message


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


def _write_source_hash_notebook(path: Path, cells: list) -> None:
    import nbformat

    notebook = nbformat.v4.new_notebook()
    notebook.cells = cells
    nbformat.write(notebook, str(path))


def test_source_hash_findings_fail_closed_for_all_active_cell_types(tmp_path, monkeypatch):
    import nbformat

    notebook_path = tmp_path / "notebooks" / "task-a" / "freshness.ipynb"
    notebook_path.parent.mkdir(parents=True)
    output = nbformat.v4.new_output("stream", name="stdout", text="retained\n")
    missing = nbformat.v4.new_code_cell("missing = 1\n", outputs=[output])
    invalid = nbformat.v4.new_code_cell("invalid = 1\n", outputs=[output])
    invalid.metadata["source_hash"] = "not-a-sha256"
    stale = nbformat.v4.new_code_cell("current = 1\n", outputs=[output])
    stale.metadata["source_hash"] = compute_source_hash("old = 1\n")
    orphan = nbformat.v4.new_code_cell("orphan = 1\n")
    orphan.metadata["source_hash"] = "anything"
    markdown_orphan = nbformat.v4.new_markdown_cell("# Markdown")
    markdown_orphan.metadata["source_hash"] = "anything"
    raw_orphan = nbformat.v4.new_raw_cell("raw")
    raw_orphan.metadata["source_hash"] = "anything"
    current = nbformat.v4.new_code_cell("current = 1\n", outputs=[output])
    current.metadata["source_hash"] = compute_source_hash(current.source)
    _write_source_hash_notebook(
        notebook_path,
        [missing, invalid, stale, orphan, markdown_orphan, raw_orphan, current],
    )
    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ("task-a",))

    findings = verify_repo._source_hash_findings(tmp_path)

    assert [(finding.id, finding.location) for finding in findings] == [
        ("E8.source_hash_missing", "notebooks/task-a/freshness.ipynb:cell[0]"),
        ("E8.source_hash_invalid", "notebooks/task-a/freshness.ipynb:cell[1]"),
        ("E8.stale_output", "notebooks/task-a/freshness.ipynb:cell[2]"),
        ("E8.source_hash_orphan", "notebooks/task-a/freshness.ipynb:cell[3]"),
        ("E8.source_hash_orphan", "notebooks/task-a/freshness.ipynb:cell[4]"),
        ("E8.source_hash_orphan", "notebooks/task-a/freshness.ipynb:cell[5]"),
    ]
    assert {finding.severity for finding in findings} == {"error"}


def test_source_hash_findings_ignore_archived_invalid_marker(tmp_path, monkeypatch):
    import nbformat

    active = tmp_path / "notebooks" / "task-a" / "current.ipynb"
    archive = tmp_path / "notebooks" / "archive" / "old-task" / "legacy.ipynb"
    active.parent.mkdir(parents=True)
    archive.parent.mkdir(parents=True)
    output = nbformat.v4.new_output("stream", name="stdout", text="retained\n")
    current = nbformat.v4.new_code_cell("current = 1\n", outputs=[output])
    current.metadata["source_hash"] = compute_source_hash(current.source)
    archived = nbformat.v4.new_code_cell("old = 1\n", outputs=[output])
    archived.metadata["source_hash"] = "not-a-sha256"
    _write_source_hash_notebook(active, [current])
    _write_source_hash_notebook(archive, [archived])
    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ("task-a",))

    assert verify_repo._source_hash_findings(tmp_path) == []


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


def test_tier_a_excludes_external_reddit2_download_notebook():
    verify_repo = _load_verify_module()
    reddit_phase1 = "notebooks/node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb"

    assert reddit_phase1 not in verify_repo.TIER_A_NOTEBOOKS
    assert reddit_phase1 not in verify_repo._makefile_variable_items(REPO, "TIER_A")


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
        f"/tmp/ml-tier-a/{notebook}" for notebook in verify_repo.TIER_A_NOTEBOOKS
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


_ISSUE63_WORKFLOW_CACHE_ROLES = {
    ("ci.yml", "atlas-consumer-policy"): _ATLAS_CACHE_MANIFESTS,
    ("ci.yml", "dependency-audit"): _AUDIT_CACHE_MANIFESTS,
    ("ci.yml", "pytest-repository"): _STACK_CACHE_MANIFESTS,
    ("ci.yml", "pytest-nnx-surface"): _STACK_CACHE_MANIFESTS,
    ("ci.yml", "verify-repo"): _STACK_CACHE_MANIFESTS,
    ("ci.yml", "docs-build"): _DOCS_CACHE_MANIFESTS,
    ("ci.yml", "tier-a-papermill"): _STACK_CACHE_MANIFESTS,
    ("ci.yml", "smoke-tier-b"): _STACK_CACHE_MANIFESTS,
    ("ci.yml", "smoke-tier-c"): _STACK_CACHE_MANIFESTS,
    ("docs.yml", "check"): _DOCS_CACHE_MANIFESTS,
    ("pages.yml", "build"): _DOCS_CACHE_MANIFESTS,
    ("pages.yml", "wiki"): _DOCS_CACHE_MANIFESTS,
    ("atlas-contract.yml", "atlas-contract"): _ATLAS_CACHE_MANIFESTS,
}


def _assert_issue63_locked_workflow_consumers(workflows: Mapping[str, dict]) -> None:
    assert set(workflows) == {"ci.yml", "docs.yml", "pages.yml", "atlas-contract.yml"}
    for (filename, job_name), cache_paths in _ISSUE63_WORKFLOW_CACHE_ROLES.items():
        job = workflows[filename]["jobs"][job_name]
        setup = next(
            step
            for step in job["steps"]
            if str(step.get("uses", "")).startswith("actions/setup-python@")
        )
        assert setup["with"] == {
            "python-version": "3.11.15",
            "cache": "pip",
            "cache-dependency-path": _cache_text(cache_paths),
        }
        commands = tuple(
            command
            for step in job["steps"]
            for command in _shell_commands(step.get("run", ""))
        )
        assert not any(
            Path(command.argv[0]).name in {"pip", "pip3"}
            or (
                command.argv[:3] == ("python", "-m", "pip")
                and len(command.argv) > 3
                and command.argv[3] == "install"
            )
            for command in commands
        )

    ci = workflows["ci.yml"]
    for job_name in _RUNTIME_JOB_WORKLOADS:
        commands = _job_run_commands(ci, job_name)
        assert sum(
            command.argv == ("make", "install-torch-stack")
            for source in commands
            for command in _shell_commands(source)
        ) == 1
    dedicated_roles = {
        ("ci.yml", "atlas-consumer-policy"): "make install-atlas-contract-lock",
        ("ci.yml", "dependency-audit"): "make install-audit-lock",
        ("ci.yml", "docs-build"): "make install-docs-lock",
        ("docs.yml", "check"): "make install-docs-lock",
        ("pages.yml", "build"): "make install-docs-lock",
        ("pages.yml", "wiki"): "make install-docs-lock",
        ("atlas-contract.yml", "atlas-contract"): "make install-atlas-contract-lock",
    }
    for (filename, job_name), role_command in dedicated_roles.items():
        source = "\n".join(
            step.get("run", "") for step in workflows[filename]["jobs"][job_name]["steps"]
        )
        assert source.count("make install-bootstrap") == 1
        assert source.count(role_command) == 1
    _assert_dependency_audit_job_contract(ci)


def test_issue63_workflows_consume_only_exact_locked_roles() -> None:
    workflows = {
        name: _load_workflow(REPO / ".github/workflows" / name)
        for name in ("ci.yml", "docs.yml", "pages.yml", "atlas-contract.yml")
    }

    _assert_issue63_locked_workflow_consumers(workflows)


@pytest.mark.parametrize(
    ("filename", "job_name", "mutation"),
    (
        ("ci.yml", "pytest-repository", "python"),
        ("ci.yml", "smoke-tier-b", "cache"),
        ("docs.yml", "check", "source-install"),
        ("atlas-contract.yml", "atlas-contract", "role"),
    ),
)
def test_issue63_workflow_lock_contract_rejects_mutations(
    filename: str, job_name: str, mutation: str
) -> None:
    workflows = {
        name: _load_workflow(REPO / ".github/workflows" / name)
        for name in ("ci.yml", "docs.yml", "pages.yml", "atlas-contract.yml")
    }
    job = workflows[filename]["jobs"][job_name]
    setup = next(
        step
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/setup-python@")
    )
    if mutation == "python":
        setup["with"]["python-version"] = "3.11"
    elif mutation == "cache":
        source = setup["with"]["cache-dependency-path"]
        mutated = source.replace("requirements/locks/bootstrap.txt\n", "", 1)
        assert mutated != source
        setup["with"]["cache-dependency-path"] = mutated
    elif mutation == "source-install":
        install = next(step for step in job["steps"] if "Install" in step.get("name", ""))
        install["run"] = "pip install -r docs-requirements.txt"
    else:
        install = next(
            step for step in job["steps"] if step.get("name") == "Install pinned Atlas runner"
        )
        install["run"] = install["run"].replace(
            "make install-atlas-contract-lock", "make install-docs-lock"
        )

    with pytest.raises(AssertionError):
        _assert_issue63_locked_workflow_consumers(workflows)
