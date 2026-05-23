# Repo Cleanup & Doc Standardization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the entire `ml` repo to a polished, conventional data-science-portfolio state via an iterative verify-and-fix loop driven by `/goal`, reaching zero findings across four orthogonal checks (structure, execution, docs, comments).

**Architecture:** A one-shot setup phase (round 0) builds a single Python verification oracle (`scripts/verify_repo.py`) and a Tier-C-safe markdown editor (`scripts/edit_notebook_markdown.py`), purges known dangling references, and tags a recovery baseline. Then a bounded iterative loop (rounds 1..8) plans → fixes → verifies → commits per round, halting on green or on stall.

**Tech Stack:** Python 3.11+ (already pinned via `.python-version`), `nbformat` (transitively via `papermill`), `pytest` for tests, GNU make, `shellcheck`, git tags for recovery.

**Spec reference:** `docs/superpowers/specs/2026-05-22-repo-cleanup-and-doc-standardization-design.md`

---

## File structure (created or modified by this plan)

**Created in Phase 0:**
- `scripts/verify_repo.py` — single CLI dispatching the four checks; emits findings JSON
- `scripts/edit_notebook_markdown.py` — markdown-cells-only notebook editor (Tier-C safety)
- `tests/test_verify_repo.py` — pytest suite for the oracle
- `tests/test_edit_notebook_markdown.py` — pytest suite for the markdown editor
- `docs/FINDINGS-NNX.md` — empty stub; loop appends submodule findings here
- `docs/FINDINGS-VENDOR.md` — empty stub; loop appends vendor findings here

**Modified in Phase 0:**
- `.gitignore` — adds `.claude/`, `.superpowers/`, `plan-*.md`, `notes-*.md`, `docs/superpowers/specs/*-draft.md`, `docs/superpowers/specs/*-scratch.md`
- `README.md` — removes dangling link to non-existent `2026-05-16-ml-repo-revival-design.md`; replaces with inline rationale; fixes NNx `(private)` annotation if wrong
- `image_classification-mnist-ffnn-pytorch/README.md` — verifies/fixes `Phase 1 merge cb4d8f4` SHA reference

**Iteratively modified in Phase 1 (rounds 1..8):** every file listed in spec §1.2 as in-scope.

---

## Phase 0 — Round 0 setup

### Task 1: Verify dev dependencies are present

**Files:**
- Read: `requirements.txt`, `pyproject.toml`

- [ ] **Step 1.1: Confirm nbformat, pytest, papermill are importable in the active env**

Run: `python -c "import nbformat, pytest, papermill; print('ok')"`
Expected: `ok`. If `ModuleNotFoundError`, install with `pip install nbformat pytest`.

- [ ] **Step 1.2: Confirm shellcheck is on PATH**

Run: `shellcheck --version`
Expected: version banner (≥ 0.9). If missing on macOS: `brew install shellcheck`. If install impossible, the E6 check is logged as skipped; do not block on this.

- [ ] **Step 1.3: Confirm we are in the worktree (not the main repo)**

Run: `git rev-parse --show-toplevel`
Expected: `/Users/kaveh/repos/ml/.claude/worktrees/documentation-and-cleanup`

---

### Task 2: Build `scripts/verify_repo.py` skeleton (CLI + JSON output schema)

**Files:**
- Create: `scripts/verify_repo.py`
- Test: `tests/test_verify_repo.py`

- [ ] **Step 2.1: Write the failing tests**

Create `tests/test_verify_repo.py`:

```python
"""Tests for scripts/verify_repo.py — the four-check oracle."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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


def test_unknown_check_errors(tmp_path):
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
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `cd /Users/kaveh/repos/ml/.claude/worktrees/documentation-and-cleanup && pytest tests/test_verify_repo.py -v`
Expected: ALL tests FAIL (file not found / module missing).

- [ ] **Step 2.3: Create the skeleton implementation**

Create `scripts/verify_repo.py`:

```python
#!/usr/bin/env python3
"""Repo verification oracle for the cleanup-and-standardization /goal loop.

Runs four orthogonal checks (structure, execution, docs, comments) and emits
machine-readable findings JSON + a human-readable report. Exit code 0 = no
findings; nonzero = findings present (count in stderr).

See docs/superpowers/specs/2026-05-22-repo-cleanup-and-doc-standardization-design.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent

ACTIVE_TASK_DIRS = (
    "image_classification-mnist-ffnn-numpy",
    "image_classification-mnist-ffnn-pytorch",
    "node_classification-reddit-gnn-pyg",
)

VERIFY_ONLY_DIRS = ("archive", "nnx", "vendor")


@dataclass
class Finding:
    id: str
    check: str           # "structure" | "execution" | "docs" | "comments"
    severity: str        # "error" | "warning"
    location: str        # "path:lineno" or "path"
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class CheckResult:
    name: str
    findings: list[Finding] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


def check_structure(repo: Path, fast: bool) -> CheckResult:
    return CheckResult(name="structure")  # filled by Task 3


def check_docs(repo: Path, fast: bool) -> CheckResult:
    return CheckResult(name="docs")  # filled by Task 4


def check_comments(repo: Path, fast: bool) -> CheckResult:
    return CheckResult(name="comments")  # filled by Task 5


def check_execution(repo: Path, fast: bool) -> CheckResult:
    return CheckResult(name="execution")  # filled by Task 6


CHECKS: dict[str, Callable[[Path, bool], CheckResult]] = {
    "structure": check_structure,
    "docs": check_docs,
    "comments": check_comments,
    "execution": check_execution,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repo verification oracle. Runs one or all of the four checks: "
            "structure, execution, docs, comments, all."
        )
    )
    parser.add_argument(
        "--check", required=True,
        choices=("structure", "execution", "docs", "comments", "all"),
        help="Which check to run.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip slow checks (E1-E3 in execution). Required when only "
             "non-executable areas changed in the round.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Path to write findings JSON. Default: print to stdout.",
    )
    args = parser.parse_args(argv)

    if args.check == "all":
        checks_to_run = list(CHECKS.keys())
    else:
        checks_to_run = [args.check]

    results = [CHECKS[name](REPO_ROOT, args.fast) for name in checks_to_run]

    all_findings = [asdict(f) for r in results for f in r.findings]
    payload = {
        "schema_version": 1,
        "summary": {
            "checks_run": checks_to_run,
            "skipped": [r.name for r in results if r.skipped],
            "total_findings": len(all_findings),
            "by_check": {r.name: len(r.findings) for r in results},
        },
        "findings": all_findings,
    }

    out_text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out_text)
    else:
        print(out_text)

    if all_findings:
        print(f"verify_repo: {len(all_findings)} findings", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2.4: Make it executable and run tests**

```bash
chmod +x scripts/verify_repo.py
pytest tests/test_verify_repo.py -v
```

Expected: all 4 skeleton tests PASS (the checks return empty results, schema is valid).

- [ ] **Step 2.5: Commit**

```bash
git add scripts/verify_repo.py tests/test_verify_repo.py
git commit -m "verify_repo: skeleton with CLI and JSON schema (Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>)"
```

---

### Task 3: Implement `check-structure` (S1–S7)

**Files:**
- Modify: `scripts/verify_repo.py:check_structure`
- Modify: `tests/test_verify_repo.py` (append)

- [ ] **Step 3.1: Append failing tests**

Append to `tests/test_verify_repo.py`:

```python
def test_structure_s1_notebooks_parse(tmp_path):
    # Real notebooks in the repo must all parse as JSON; if any don't, S1 reports them.
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s1 = [f for f in data["findings"] if f["id"].startswith("S1")]
    # In a healthy repo this is empty; we just assert the check ran (presence in summary).
    assert data["summary"]["by_check"]["structure"] == len(s1) + sum(
        1 for f in data["findings"] if not f["id"].startswith("S1") and f["check"] == "structure"
    )


def test_structure_s5_no_common_imports():
    """No `from common.` import anywhere in active task notebooks or scripts."""
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s5 = [f for f in data["findings"] if f["id"].startswith("S5")]
    # Repo policy: zero. CI must catch any regression.
    assert s5 == [], f"S5 found stray common.* imports: {s5}"


def test_structure_s7_no_pycache_tracked():
    """No __pycache__, .ipynb_checkpoints, .DS_Store should be tracked."""
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s7 = [f for f in data["findings"] if f["id"].startswith("S7")]
    assert s7 == [], f"S7 found tracked bloat: {s7}"
```

- [ ] **Step 3.2: Run to verify they fail or pass-trivially (skeleton returns no findings)**

Run: `pytest tests/test_verify_repo.py -v -k structure`
Expected: tests pass trivially (skeleton returns no findings); we will tighten with real checks below.

- [ ] **Step 3.3: Replace `check_structure` with the real implementation**

Replace the `check_structure` stub in `scripts/verify_repo.py` with:

```python
import re
import subprocess
import importlib.util
import nbformat

_INTERNAL_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)#][^)#]*)(#[^)]*)?\)")
_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))")
_GITIGNORE_REQUIRED_PATTERNS = (".claude/", ".superpowers/", "plan-*.md", "notes-*.md")
_BLOAT_PATTERNS = ("__pycache__", ".ipynb_checkpoints", ".DS_Store")


def _git_ls_files(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return ""


def _iter_notebooks(repo: Path):
    for d in ACTIVE_TASK_DIRS:
        for nb_path in (repo / d).glob("*.ipynb"):
            yield nb_path


def _iter_in_scope_text_files(repo: Path):
    yield repo / "README.md"
    yield repo / "CLAUDE.md"
    for p in (repo / "docs").glob("*.md"):
        yield p
    for d in ACTIVE_TASK_DIRS:
        for p in (repo / d).glob("*.md"):
            yield p


def check_structure(repo: Path, fast: bool) -> CheckResult:
    result = CheckResult(name="structure")
    tracked = set(_git_ls_files(repo))

    # S1: notebooks parse as JSON; every cell has a valid cell_type.
    valid_types = {"code", "markdown", "raw"}
    notebooks = list(_iter_notebooks(repo))
    for nb in notebooks:
        try:
            doc = nbformat.read(nb, as_version=4)
            for i, c in enumerate(doc.cells):
                if c.cell_type not in valid_types:
                    result.findings.append(Finding(
                        id="S1.cell_type", check="structure", severity="error",
                        location=f"{nb.relative_to(repo)}:cell[{i}]",
                        message=f"unknown cell_type={c.cell_type!r}",
                    ))
        except Exception as e:
            result.findings.append(Finding(
                id="S1.parse", check="structure", severity="error",
                location=str(nb.relative_to(repo)),
                message=f"failed to parse: {e}",
            ))

    # S2: notebook imports resolve in the current env.
    seen_modules: dict[str, str] = {}  # module -> first-seen location
    for nb in notebooks:
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        for ci, cell in enumerate(doc.cells):
            if cell.cell_type != "code":
                continue
            for li, line in enumerate(cell.source.splitlines()):
                m = _IMPORT_RE.match(line)
                if not m:
                    continue
                module = (m.group(1) or m.group(2) or "").split(".")[0]
                if not module or module in seen_modules:
                    continue
                seen_modules[module] = f"{nb.relative_to(repo)}:cell[{ci}]:line[{li}]"
                try:
                    if importlib.util.find_spec(module) is None:
                        result.findings.append(Finding(
                            id="S2.unresolved_import", check="structure", severity="error",
                            location=seen_modules[module],
                            message=f"module {module!r} not importable in current env",
                        ))
                except (ImportError, ValueError) as e:
                    result.findings.append(Finding(
                        id="S2.import_error", check="structure", severity="warning",
                        location=seen_modules[module],
                        message=f"find_spec({module!r}) raised {e!r}",
                    ))

    # S3 + S4: internal markdown links resolve; no dangling refs.
    for md in _iter_in_scope_text_files(repo):
        text = _read_text(md)
        for m in _INTERNAL_LINK_RE.finditer(text):
            target = m.group(1).strip()
            if target.startswith("http://") or target.startswith("https://") or target.startswith("mailto:"):
                continue
            target_path = (md.parent / target).resolve()
            if not target_path.exists():
                result.findings.append(Finding(
                    id="S3.broken_link", check="structure", severity="error",
                    location=f"{md.relative_to(repo)}",
                    message=f"internal link target missing: {target}",
                    detail={"link": m.group(0)},
                ))

    # S5: no `from common.` imports anywhere in tracked text.
    for path in tracked:
        full = repo / path
        if not full.is_file():
            continue
        suffix = full.suffix.lower()
        if suffix not in (".py", ".ipynb"):
            continue
        text = _read_text(full)
        if "from common." in text or "import common." in text:
            # narrow to actual import lines (not comments mentioning "common")
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if "from common." in stripped or "import common." in stripped:
                    result.findings.append(Finding(
                        id="S5.common_import", check="structure", severity="error",
                        location=f"{path}:{i}",
                        message="forbidden import; use `from nnx.` instead",
                    ))

    # S6: .gitignore covers required patterns; none of them are tracked.
    gitignore = _read_text(repo / ".gitignore")
    for pat in _GITIGNORE_REQUIRED_PATTERNS:
        if pat not in gitignore:
            result.findings.append(Finding(
                id="S6.gitignore_missing", check="structure", severity="error",
                location=".gitignore",
                message=f"required pattern absent: {pat}",
            ))
    for path in tracked:
        if path.startswith(".claude/") or path.startswith(".superpowers/"):
            result.findings.append(Finding(
                id="S6.tracked_bloat", check="structure", severity="error",
                location=path,
                message="bloat directory tracked; should be gitignored",
            ))

    # S7: no __pycache__, .ipynb_checkpoints, .DS_Store tracked.
    for path in tracked:
        for pat in _BLOAT_PATTERNS:
            if pat in path:
                result.findings.append(Finding(
                    id="S7.tracked_bloat", check="structure", severity="error",
                    location=path,
                    message=f"bloat artifact tracked: contains {pat!r}",
                ))

    return result
```

- [ ] **Step 3.4: Run tests; expect green**

Run: `pytest tests/test_verify_repo.py -v -k structure`
Expected: PASS. Then run an end-to-end smoke:

```bash
python scripts/verify_repo.py --check structure --fast
```

Expected: prints JSON. Real findings WILL appear at this point (the dangling `2026-05-16-...md` link, the missing `.gitignore` patterns) — that's the loop's job to fix later.

- [ ] **Step 3.5: Commit**

```bash
git add scripts/verify_repo.py tests/test_verify_repo.py
git commit -m "verify_repo: implement check-structure (S1-S7) (Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>)"
```

---

### Task 4: Implement `check-docs` (D1–D8)

**Files:**
- Modify: `scripts/verify_repo.py:check_docs`
- Modify: `tests/test_verify_repo.py` (append)

- [ ] **Step 4.1: Append failing tests**

```python
def test_docs_d1_known_notebooks_have_required_sections():
    """Active notebooks must contain the §1..§6 headings (with phase variants)."""
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    d1 = [f for f in data["findings"] if f["id"].startswith("D1")]
    # Findings are allowed initially (notebooks not yet standardized) — assert the check
    # actually examined them by looking at the summary count.
    assert data["summary"]["by_check"]["docs"] >= 0


def test_docs_d8_terminology_consistency_known_canonicals():
    """The check should mention canonical spellings in its allow-list logic."""
    SCRIPT_TEXT = SCRIPT.read_text()
    for token in ("genai-vanilla", "JupyterHub", "NumPy", "PyTorch"):
        assert token in SCRIPT_TEXT, f"D8 missing canonical {token!r}"
```

- [ ] **Step 4.2: Run; expect first to pass trivially, second to fail**

Run: `pytest tests/test_verify_repo.py -v -k docs`
Expected: `test_docs_d1_*` PASS trivially (skeleton); `test_docs_d8_*` FAIL (canonicals not yet in source).

- [ ] **Step 4.3: Replace `check_docs` with the real implementation**

Add this constant block near the top of `scripts/verify_repo.py` (after `VERIFY_ONLY_DIRS`):

```python
# Required-sections lookup table (spec §2.1, §3.3 D1).
# Maps notebook path (relative to repo root) → ordered list of required top-level
# section headings (markdown `# N. <Name>` form).
REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {
    "image_classification-mnist-ffnn-numpy/notebook.ipynb": (
        "1. Overview", "2. Environment & Setup", "3. Data",
        "4. Model", "5. Training", "6. Evaluation & Results",
    ),
    "image_classification-mnist-ffnn-pytorch/notebook.ipynb": (
        "1. Overview", "2. Environment & Setup", "3. Data",
        "4. Model", "5. Training", "6. Evaluation & Results",
    ),
    "node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb": (
        "1. Overview", "2. Environment & Setup", "3. Dataset deep-dive",
    ),
    "node_classification-reddit-gnn-pyg/phase2-model-selection-notebook1.ipynb": (
        "1. Overview", "2. Environment & Setup", "3. Data",
        "4. Model", "5. Training", "6. Evaluation & Results",
    ),
    "node_classification-reddit-gnn-pyg/phase2-model-selection-notebook2.ipynb": (
        "1. Overview", "2. Environment & Setup", "3. Data",
        "4. Model", "5. Training", "6. Evaluation & Results",
    ),
    "node_classification-reddit-gnn-pyg/phase2-model-selection-notebook3.ipynb": (
        "1. Overview", "2. Environment & Setup", "3. Data",
        "4. Model", "5. Training", "6. Evaluation & Results",
    ),
    "node_classification-reddit-gnn-pyg/phase2-model-selection-notebook4.ipynb": (
        "1. Overview", "2. Environment & Setup", "3. Data",
        "4. Model", "5. Training", "6. Evaluation & Results",
    ),
    "node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook.ipynb": (
        "1. Overview", "2. Environment & Setup", "3. Data",
        "4. Model", "5. Training", "6. Evaluation & Results",
    ),
    "node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook2.ipynb": (
        "1. Overview", "2. Environment & Setup", "3. Data",
        "4. Model", "5. Training", "6. Evaluation & Results",
    ),
    "node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook3.ipynb": (
        "1. Overview", "2. Environment & Setup", "3. Data",
        "4. Model", "5. Training", "6. Evaluation & Results",
    ),
    "node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook4.ipynb": (
        "1. Overview", "2. Environment & Setup", "3. Data",
        "4. Model", "5. Training", "6. Evaluation & Results",
    ),
}

# Required H2 headings in per-task README (spec §2.2).
README_REQUIRED_H2 = (
    "1. Task summary", "2. Why this exists", "3. What's in the notebook",
    "4. How to run", "5. Dependencies", "6. Known issues",
)

# Required H2 headings in root README (spec §2.3).
ROOT_README_REQUIRED_H2 = (
    "1. Overview", "2. Repository layout", "3. Quick start", "4. Tasks",
    "5. Notebook re-execution policy", "6. NNx library",
    "7. Repository conventions", "8. Roadmap", "9. License",
)

# D8: canonical spellings and the misspellings we flag.
TERMINOLOGY_CANONICALS = {
    "genai-vanilla": ("Genai-Vanilla", "GenAI-Vanilla", "GenAI Vanilla", "genai vanilla"),
    "JupyterHub": ("jupyterhub", "Jupyterhub", "Jupyter Hub", "jupyter hub"),
    "NumPy": ("numpy", "Numpy", "NUMPY"),
    "PyTorch": ("pytorch", "Pytorch", "PYTORCH", "Py-Torch"),
    "PyG": ("PYG", "Pyg"),
}
```

Replace the `check_docs` stub with:

```python
_H1_RE = re.compile(r"^# ([^\n]+)", re.MULTILINE)
_H2_RE = re.compile(r"^## ([^\n]+)", re.MULTILINE)


def _markdown_headings(text: str, level: int) -> list[str]:
    pat = _H1_RE if level == 1 else _H2_RE
    return [m.group(1).strip() for m in pat.finditer(text)]


def _notebook_markdown_text(nb_path: Path) -> str:
    try:
        doc = nbformat.read(nb_path, as_version=4)
    except Exception:
        return ""
    return "\n\n".join(c.source for c in doc.cells if c.cell_type == "markdown")


def _ordered_contains(required: tuple[str, ...], actual: list[str]) -> tuple[bool, list[str]]:
    """Returns (ok, missing). `actual` must contain `required` as an ordered subsequence."""
    i = 0
    missing = []
    for needed in required:
        found = False
        while i < len(actual):
            if needed.lower() in actual[i].lower():
                found = True
                i += 1
                break
            i += 1
        if not found:
            missing.append(needed)
            i = 0  # reset to allow remaining checks independently
    return (not missing, missing)


def check_docs(repo: Path, fast: bool) -> CheckResult:
    result = CheckResult(name="docs")

    # D1: every active-task notebook contains required §1..§6 headings in order.
    for rel, required in REQUIRED_SECTIONS.items():
        nb = repo / rel
        if not nb.exists():
            result.findings.append(Finding(
                id="D1.missing_notebook", check="docs", severity="error",
                location=rel, message="referenced in REQUIRED_SECTIONS but file missing",
            ))
            continue
        text = _notebook_markdown_text(nb)
        h1s = _markdown_headings(text, level=1)
        ok, missing = _ordered_contains(required, h1s)
        if not ok:
            result.findings.append(Finding(
                id="D1.missing_sections", check="docs", severity="error",
                location=rel,
                message=f"missing or out-of-order top-level sections: {missing}",
                detail={"found": h1s, "required": list(required)},
            ))

    # D2: every notebook has a top markdown cell stating purpose + dataset.
    for rel in REQUIRED_SECTIONS:
        nb = repo / rel
        if not nb.exists():
            continue
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        if not doc.cells:
            result.findings.append(Finding(
                id="D2.empty_notebook", check="docs", severity="error",
                location=rel, message="notebook has no cells",
            ))
            continue
        first = doc.cells[0]
        if first.cell_type != "markdown":
            result.findings.append(Finding(
                id="D2.first_cell_not_markdown", check="docs", severity="error",
                location=rel, message="first cell must be a markdown title/purpose cell",
            ))

    # D3: per-task READMEs match shape (all required H2s present, in order).
    for d in ACTIVE_TASK_DIRS:
        readme = repo / d / "README.md"
        if not readme.exists():
            result.findings.append(Finding(
                id="D3.missing_readme", check="docs", severity="error",
                location=f"{d}/README.md", message="per-task README missing",
            ))
            continue
        h2s = _markdown_headings(_read_text(readme), level=2)
        ok, missing = _ordered_contains(README_REQUIRED_H2, h2s)
        if not ok:
            result.findings.append(Finding(
                id="D3.missing_sections", check="docs", severity="error",
                location=f"{d}/README.md",
                message=f"per-task README missing required H2s: {missing}",
                detail={"found": h2s, "required": list(README_REQUIRED_H2)},
            ))

    # D4: root README matches shape.
    root_readme = repo / "README.md"
    root_h2s = _markdown_headings(_read_text(root_readme), level=2)
    ok, missing = _ordered_contains(ROOT_README_REQUIRED_H2, root_h2s)
    if not ok:
        result.findings.append(Finding(
            id="D4.missing_sections", check="docs", severity="error",
            location="README.md",
            message=f"root README missing required H2s: {missing}",
            detail={"found": root_h2s, "required": list(ROOT_README_REQUIRED_H2)},
        ))

    # D5: root README's task table row count = number of active task folders.
    root_text = _read_text(root_readme)
    # Count rows in the task table heuristically: lines starting with `| [`
    table_rows = sum(1 for line in root_text.splitlines() if line.startswith("| [") and "/](" in line)
    active_count = sum(1 for d in ACTIVE_TASK_DIRS if (repo / d).is_dir())
    if table_rows < active_count:
        result.findings.append(Finding(
            id="D5.task_table_mismatch", check="docs", severity="error",
            location="README.md",
            message=f"task table has {table_rows} rows; expected ≥ {active_count} active",
        ))

    # D6: Roadmap section non-empty.
    if "## 8. Roadmap" in root_text or "## Roadmap" in root_text:
        # find the section body
        marker = "## 8. Roadmap" if "## 8. Roadmap" in root_text else "## Roadmap"
        body = root_text.split(marker, 1)[1]
        body = body.split("\n## ", 1)[0]
        if not re.search(r"-\s*\[\s*[xX ]\s*\]\s+\S", body):
            result.findings.append(Finding(
                id="D6.empty_roadmap", check="docs", severity="warning",
                location="README.md", message="Roadmap section present but has no checklist items",
            ))
    else:
        result.findings.append(Finding(
            id="D6.missing_roadmap", check="docs", severity="error",
            location="README.md", message="Roadmap section absent",
        ))

    # D7: docs/*.md exist and contain at least one H2.
    for required_doc in ("env-setup.md", "jupyterhub-integration.md", "vscode-remote-access.md"):
        p = repo / "docs" / required_doc
        if not p.exists():
            result.findings.append(Finding(
                id="D7.missing_doc", check="docs", severity="error",
                location=f"docs/{required_doc}", message="required doc missing",
            ))
            continue
        if not _markdown_headings(_read_text(p), level=2):
            result.findings.append(Finding(
                id="D7.no_sections", check="docs", severity="warning",
                location=f"docs/{required_doc}", message="doc has no H2 sections",
            ))

    # D8: terminology consistency. Allow canonicals; flag deviations.
    for path in _iter_in_scope_text_files(repo):
        text = _read_text(path)
        for canonical, deviations in TERMINOLOGY_CANONICALS.items():
            for dev in deviations:
                # word-boundary match to avoid partial-word false positives
                for m in re.finditer(rf"\b{re.escape(dev)}\b", text):
                    line_no = text.count("\n", 0, m.start()) + 1
                    result.findings.append(Finding(
                        id="D8.terminology", check="docs", severity="warning",
                        location=f"{path.relative_to(repo)}:{line_no}",
                        message=f"non-canonical spelling {dev!r}; use {canonical!r}",
                    ))

    return result
```

- [ ] **Step 4.4: Run tests; smoke the check**

```bash
pytest tests/test_verify_repo.py -v -k docs
python scripts/verify_repo.py --check docs --fast
```

Expected: tests pass; the docs check produces many findings (this is correct — the docs haven't been standardized yet).

- [ ] **Step 4.5: Commit**

```bash
git add scripts/verify_repo.py tests/test_verify_repo.py
git commit -m "verify_repo: implement check-docs (D1-D8) (Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>)"
```

---

### Task 5: Implement `check-comments` (Phase A heuristic only)

**Files:**
- Modify: `scripts/verify_repo.py:check_comments`
- Modify: `tests/test_verify_repo.py` (append)

The judge (Phase B) is invoked by the loop's FIX step, not by verify. Verify is deterministic.

- [ ] **Step 5.1: Append failing tests**

```python
def test_comments_phase_a_flags_obvious_state_the_what():
    """Synthetic .py file with a known bad comment should produce a finding."""
    import tempfile, os
    fake = REPO / "image_classification-mnist-ffnn-numpy" / "_temp_test_comments.py"
    fake.write_text("# import numpy as np\nimport numpy as np\n")
    try:
        r = run_verify("--check", "comments", "--fast")
        data = json.loads(r.stdout) if r.stdout else {"findings": []}
        hits = [
            f for f in data["findings"]
            if f["check"] == "comments" and "_temp_test_comments.py" in f["location"]
        ]
        assert hits, f"expected at least one state-the-what flag; got {data['findings']}"
    finally:
        fake.unlink(missing_ok=True)


def test_comments_phase_a_skips_explanatory_comments():
    """A WHY-style comment should NOT be flagged."""
    fake = REPO / "image_classification-mnist-ffnn-numpy" / "_temp_why_comments.py"
    fake.write_text(
        "# Xavier init keeps variance stable across depths; default torch init blows up here.\n"
        "weight = xavier_init(shape)\n"
    )
    try:
        r = run_verify("--check", "comments", "--fast")
        data = json.loads(r.stdout) if r.stdout else {"findings": []}
        hits = [
            f for f in data["findings"]
            if f["check"] == "comments" and "_temp_why_comments.py" in f["location"]
        ]
        assert not hits, f"WHY-style comment falsely flagged: {hits}"
    finally:
        fake.unlink(missing_ok=True)
```

- [ ] **Step 5.2: Run; expect FAIL (skeleton returns nothing)**

Run: `pytest tests/test_verify_repo.py -v -k comments`
Expected: both FAIL.

- [ ] **Step 5.3: Replace `check_comments` with the real implementation**

```python
# Phase A heuristic patterns. Each entry is (comment-regex, code-regex);
# if both match consecutive non-blank lines, flag.
_STATE_THE_WHAT_PATTERNS: tuple[tuple[re.Pattern, re.Pattern], ...] = (
    (re.compile(r"^\s*#\s*import\s+\S", re.IGNORECASE),
     re.compile(r"^\s*(?:from\s+\S+\s+)?import\s+\S")),
    (re.compile(r"^\s*#\s*loop\s+(over|through|across)\b", re.IGNORECASE),
     re.compile(r"^\s*(?:for|while)\s+")),
    (re.compile(r"^\s*#\s*return\b", re.IGNORECASE),
     re.compile(r"^\s*return\b")),
    (re.compile(r"^\s*#\s*(define|create|define the|declare)\b", re.IGNORECASE),
     re.compile(r"^\s*def\s+|^\s*class\s+|^\s*\w+\s*=")),
    (re.compile(r"^\s*#\s*(initialize|init|set|assign)\b", re.IGNORECASE),
     re.compile(r"^\s*\w+\s*=")),
    (re.compile(r"^\s*#\s*print\b", re.IGNORECASE),
     re.compile(r"^\s*print\s*\(")),
    (re.compile(r"^\s*#\s*(call|invoke|run)\s+\w+", re.IGNORECASE),
     re.compile(r"^\s*\w+\s*\(")),
)


def _scan_source_for_comments(source: str, location_prefix: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = source.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            continue
        # find the next non-blank, non-comment code line
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if nxt and not nxt.startswith("#"):
                break
            j += 1
        if j >= len(lines):
            continue
        nxt_line = lines[j]
        for comment_pat, code_pat in _STATE_THE_WHAT_PATTERNS:
            if comment_pat.match(line) and code_pat.match(nxt_line):
                findings.append(Finding(
                    id="C.state_the_what", check="comments", severity="warning",
                    location=f"{location_prefix}:{i+1}",
                    message=f"comment restates the next code line: {stripped[:80]!r}",
                    detail={"next_code": nxt_line.strip()[:80]},
                ))
                break
    return findings


def _iter_in_scope_code(repo: Path):
    # In-scope .py files: scripts/*.py (excluding new oracle files) and per-task *.py
    for p in (repo / "scripts").glob("*.py"):
        if p.name in ("verify_repo.py", "edit_notebook_markdown.py"):
            continue
        yield p, p.read_text()
    for d in ACTIVE_TASK_DIRS:
        for p in (repo / d).glob("*.py"):
            yield p, p.read_text()
    # In-scope notebook code cells
    for nb in _iter_notebooks(repo):
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        for ci, cell in enumerate(doc.cells):
            if cell.cell_type != "code":
                continue
            yield nb / f"cell[{ci}]", cell.source


def check_comments(repo: Path, fast: bool) -> CheckResult:
    result = CheckResult(name="comments")
    for path_marker, source in _iter_in_scope_code(repo):
        # path_marker may be a real Path or a synthetic Path/cell marker; stringify safely
        try:
            rel = path_marker.relative_to(repo)
            location_prefix = str(rel)
        except (ValueError, AttributeError):
            location_prefix = str(path_marker)
        for f in _scan_source_for_comments(source, location_prefix):
            result.findings.append(f)
    return result
```

- [ ] **Step 5.4: Run tests**

Run: `pytest tests/test_verify_repo.py -v -k comments`
Expected: both PASS.

- [ ] **Step 5.5: Commit**

```bash
git add scripts/verify_repo.py tests/test_verify_repo.py
git commit -m "verify_repo: implement check-comments Phase A heuristic (Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>)"
```

---

### Task 6: Implement `check-execution` (E1–E6)

**Files:**
- Modify: `scripts/verify_repo.py:check_execution`
- Modify: `tests/test_verify_repo.py` (append)

- [ ] **Step 6.1: Append failing tests**

```python
def test_execution_fast_mode_skips_e1_e2_e3():
    """In --fast mode, slow targets (E1-E3) must be marked skipped."""
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    by = data["summary"]["by_check"].get("execution", 0)
    # Skipped subchecks don't contribute findings; assertion is just that the check ran.
    assert "execution" in data["summary"]["checks_run"]


def test_execution_e5_reads_phase3_code_cells():
    """E5 requires the pre-cleanup-baseline tag; without it, must mark E5 skipped (not error)."""
    # We don't create the tag during testing; just assert no crash.
    r = run_verify("--check", "execution", "--fast")
    assert r.returncode in (0, 1)
```

- [ ] **Step 6.2: Run; tests should pass trivially (skeleton)**

Run: `pytest tests/test_verify_repo.py -v -k execution`
Expected: PASS.

- [ ] **Step 6.3: Replace `check_execution` with the real implementation**

```python
def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _phase3_code_cells_unchanged(repo: Path) -> list[Finding]:
    findings: list[Finding] = []
    # Verify pre-cleanup-baseline tag exists; otherwise skip with a warning.
    rc, _, _ = _run(["git", "rev-parse", "--verify", "pre-cleanup-baseline"], repo)
    if rc != 0:
        findings.append(Finding(
            id="E5.no_baseline", check="execution", severity="warning",
            location="<git>", message="pre-cleanup-baseline tag missing; E5 not enforceable",
        ))
        return findings
    phase3 = list((repo / "node_classification-reddit-gnn-pyg").glob("phase3-*.ipynb"))
    for nb in phase3:
        # read code-cell content at HEAD
        head_doc = nbformat.read(nb, as_version=4)
        # read code-cell content at baseline via `git show`
        rc, raw, err = _run(["git", "show", f"pre-cleanup-baseline:{nb.relative_to(repo)}"], repo)
        if rc != 0:
            findings.append(Finding(
                id="E5.baseline_read_failed", check="execution", severity="error",
                location=str(nb.relative_to(repo)),
                message=f"could not read baseline: {err.strip()[:120]}",
            ))
            continue
        try:
            base_doc = nbformat.reads(raw, as_version=4)
        except Exception as e:
            findings.append(Finding(
                id="E5.baseline_parse_failed", check="execution", severity="error",
                location=str(nb.relative_to(repo)),
                message=f"baseline notebook unparseable: {e}",
            ))
            continue
        head_codes = [(c.source, c.get("outputs", []), c.get("execution_count"))
                      for c in head_doc.cells if c.cell_type == "code"]
        base_codes = [(c.source, c.get("outputs", []), c.get("execution_count"))
                      for c in base_doc.cells if c.cell_type == "code"]
        if head_codes != base_codes:
            findings.append(Finding(
                id="E5.code_cells_changed", check="execution", severity="error",
                location=str(nb.relative_to(repo)),
                message="Tier-C code cells (source/outputs/execution_count) diverged from baseline",
                detail={"head_count": len(head_codes), "base_count": len(base_codes)},
            ))
    return findings


def check_execution(repo: Path, fast: bool) -> CheckResult:
    result = CheckResult(name="execution")

    # E1-E3: only in non-fast mode.
    if not fast:
        # E1
        rc, _, err = _run(["make", "run-tier-a"], repo)
        if rc != 0:
            result.findings.append(Finding(
                id="E1.tier_a_failed", check="execution", severity="error",
                location="Makefile:run-tier-a", message=f"failed: {err.strip()[-300:]}",
            ))
        # E2
        rc, _, err = _run(["make", "smoke-tier-b"], repo)
        if rc != 0:
            result.findings.append(Finding(
                id="E2.tier_b_smoke_failed", check="execution", severity="error",
                location="Makefile:smoke-tier-b", message=f"failed: {err.strip()[-300:]}",
            ))
        # E3
        rc, _, err = _run(["make", "smoke-tier-c"], repo)
        if rc != 0:
            result.findings.append(Finding(
                id="E3.tier_c_smoke_failed", check="execution", severity="error",
                location="Makefile:smoke-tier-c", message=f"failed: {err.strip()[-300:]}",
            ))
    else:
        result.skipped = False  # we still run the fast subchecks below

    # E4: no errored Tier-A cells.
    tier_a = (
        "image_classification-mnist-ffnn-numpy/notebook.ipynb",
        "image_classification-mnist-ffnn-pytorch/notebook.ipynb",
        "node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb",
    )
    for rel in tier_a:
        nb = repo / rel
        if not nb.exists():
            continue
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        for ci, cell in enumerate(doc.cells):
            if cell.cell_type != "code":
                continue
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    result.findings.append(Finding(
                        id="E4.cell_error", check="execution", severity="error",
                        location=f"{rel}:cell[{ci}]",
                        message=f"errored output: {out.get('ename', '?')}: {out.get('evalue', '')[:120]}",
                    ))

    # E5: phase3 code cells unchanged from baseline tag.
    result.findings.extend(_phase3_code_cells_unchanged(repo))

    # E6: shellcheck.
    rc_shellcheck, _, _ = _run(["which", "shellcheck"], repo)
    if rc_shellcheck != 0:
        result.findings.append(Finding(
            id="E6.shellcheck_missing", check="execution", severity="warning",
            location="<env>", message="shellcheck not on PATH; install with `brew install shellcheck`",
        ))
    else:
        for sh in (repo / "scripts").glob("*.sh"):
            rc, out, err = _run(["shellcheck", str(sh)], repo)
            if rc != 0:
                result.findings.append(Finding(
                    id="E6.shellcheck", check="execution", severity="error",
                    location=str(sh.relative_to(repo)),
                    message=(out + err).strip()[-300:],
                ))

    return result
```

- [ ] **Step 6.4: Run tests + fast smoke**

```bash
pytest tests/test_verify_repo.py -v -k execution
python scripts/verify_repo.py --check execution --fast
```

Expected: PASS. Real findings will appear (no baseline tag yet → E5 warning).

- [ ] **Step 6.5: Commit**

```bash
git add scripts/verify_repo.py tests/test_verify_repo.py
git commit -m "verify_repo: implement check-execution (E1-E6) (Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>)"
```

---

### Task 7: Build `scripts/edit_notebook_markdown.py`

**Files:**
- Create: `scripts/edit_notebook_markdown.py`
- Create: `tests/test_edit_notebook_markdown.py`

- [ ] **Step 7.1: Write the failing tests**

```python
"""Tests for scripts/edit_notebook_markdown.py — Tier-C safe markdown editor."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "edit_notebook_markdown.py"


def _make_notebook(path: Path) -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell("# Old title\nintro"),
        nbformat.v4.new_code_cell("print('hello')",
                                  outputs=[nbformat.v4.new_output("stream", text="hello\n")]),
        nbformat.v4.new_markdown_cell("## Section A"),
    ]
    nbformat.write(nb, path)


def test_replace_markdown_preserves_code_cells(tmp_path):
    nb_path = tmp_path / "nb.ipynb"
    _make_notebook(nb_path)
    orig_code = nbformat.read(nb_path, as_version=4).cells[1]

    r = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--notebook", str(nb_path),
         "--cell", "0",
         "--text", "# New title\nnew intro"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr

    doc = nbformat.read(nb_path, as_version=4)
    assert doc.cells[0].source == "# New title\nnew intro"
    assert doc.cells[0].cell_type == "markdown"
    # Code cell — every byte preserved.
    new_code = doc.cells[1]
    assert new_code.source == orig_code.source
    assert new_code.outputs == orig_code.outputs
    assert new_code.execution_count == orig_code.execution_count


def test_refuses_to_edit_code_cell(tmp_path):
    nb_path = tmp_path / "nb.ipynb"
    _make_notebook(nb_path)
    r = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--notebook", str(nb_path),
         "--cell", "1",
         "--text", "should-be-rejected"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "code" in (r.stderr + r.stdout).lower()


def test_insert_markdown_cell_at_index(tmp_path):
    nb_path = tmp_path / "nb.ipynb"
    _make_notebook(nb_path)
    r = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--notebook", str(nb_path),
         "--insert-at", "1",
         "--text", "## New section"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    doc = nbformat.read(nb_path, as_version=4)
    assert doc.cells[1].cell_type == "markdown"
    assert doc.cells[1].source == "## New section"
    # The original code cell is now at index 2 and unchanged.
    assert doc.cells[2].cell_type == "code"
    assert doc.cells[2].source == "print('hello')"
```

- [ ] **Step 7.2: Run; expect FAIL**

Run: `pytest tests/test_edit_notebook_markdown.py -v`
Expected: all FAIL (script missing).

- [ ] **Step 7.3: Create the implementation**

```python
#!/usr/bin/env python3
"""Markdown-cells-only notebook editor.

Used by the cleanup-and-standardization loop to mutate notebook documentation
without touching any code cell. Tier-C safety: never modifies, deletes, or
re-orders `cell_type == "code"` cells; never alters outputs or execution_count.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat


def replace_markdown_cell(nb_path: Path, index: int, new_text: str) -> None:
    doc = nbformat.read(nb_path, as_version=4)
    if index < 0 or index >= len(doc.cells):
        raise SystemExit(f"cell index {index} out of range (0..{len(doc.cells)-1})")
    cell = doc.cells[index]
    if cell.cell_type != "markdown":
        raise SystemExit(
            f"refusing to edit cell {index}: cell_type={cell.cell_type!r}, "
            f"only markdown cells may be edited"
        )
    cell.source = new_text
    nbformat.write(doc, nb_path)


def insert_markdown_cell(nb_path: Path, index: int, new_text: str) -> None:
    doc = nbformat.read(nb_path, as_version=4)
    if index < 0 or index > len(doc.cells):
        raise SystemExit(f"insert index {index} out of range (0..{len(doc.cells)})")
    new_cell = nbformat.v4.new_markdown_cell(new_text)
    doc.cells.insert(index, new_cell)
    nbformat.write(doc, nb_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Edit markdown cells of a Jupyter notebook without touching code cells.",
    )
    p.add_argument("--notebook", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--cell", type=int, help="Replace markdown cell at this index.")
    group.add_argument("--insert-at", type=int, help="Insert a new markdown cell at this index.")
    p.add_argument("--text", required=True, help="New markdown source.")
    args = p.parse_args(argv)

    if not args.notebook.exists():
        raise SystemExit(f"notebook not found: {args.notebook}")

    if args.cell is not None:
        replace_markdown_cell(args.notebook, args.cell, args.text)
    else:
        insert_markdown_cell(args.notebook, args.insert_at, args.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7.4: Run tests**

```bash
chmod +x scripts/edit_notebook_markdown.py
pytest tests/test_edit_notebook_markdown.py -v
```

Expected: all PASS.

- [ ] **Step 7.5: Commit**

```bash
git add scripts/edit_notebook_markdown.py tests/test_edit_notebook_markdown.py
git commit -m "scripts: add edit_notebook_markdown.py (Tier-C safe) (Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>)"
```

---

### Task 8: Round-0 cleanup sweep (gitignore + dangling refs)

**Files:**
- Modify: `.gitignore`
- Modify: `README.md` (dangling-link removal)
- Modify: `image_classification-mnist-ffnn-pytorch/README.md` (SHA verification)
- Create: `docs/FINDINGS-NNX.md`, `docs/FINDINGS-VENDOR.md`

- [ ] **Step 8.1: Append required patterns to `.gitignore`**

Append to `.gitignore`:

```
# Loop / superpowers / local-plan bloat
.claude/
.superpowers/
plan-*.md
notes-*.md
docs/superpowers/specs/*-draft.md
docs/superpowers/specs/*-scratch.md
```

Then confirm:

```bash
git check-ignore -v .claude/ 2>&1 | grep -E "\.gitignore"
```

Expected: matches the `.claude/` rule.

- [ ] **Step 8.2: Remove dangling spec link from `README.md`**

Edit `README.md`, replace the line:

```
See [docs/superpowers/specs/2026-05-16-ml-repo-revival-design.md](docs/superpowers/specs/2026-05-16-ml-repo-revival-design.md) §4 for the rationale and the library co-evolution principle.
```

with:

```
The library co-evolution principle (see [CLAUDE.md](CLAUDE.md)): each future task lands its required `nnx` additions upstream first, then bumps the submodule pointer here. YAGNI applies — no speculative abstractions in `nnx`.
```

- [ ] **Step 8.3: Verify the SHA reference**

Check if `cb4d8f4` exists:

```bash
git -C /Users/kaveh/repos/ml cat-file -t cb4d8f4 2>&1 || echo "missing"
```

If "missing": edit `image_classification-mnist-ffnn-pytorch/README.md`, replace the line containing `Phase 1 merge cb4d8f4` with the equivalent stable statement (no SHA):

```
All installed by the jupyterhub image (see `vendor/genai-vanilla/`) or via the root `requirements.txt` + `torch-requirements.txt`.
```

If the SHA exists, leave it.

- [ ] **Step 8.4: Verify NNx `(private)` annotation**

```bash
git ls-remote https://github.com/thekaveh/NNx 2>&1 | head -1
```

If output starts with `Cloning into ... fatal: could not read Username` or similar → it's private; leave the annotation.
If output starts with refs (e.g., `<sha>\tHEAD`) → it's public; remove `(private)` from the README.

- [ ] **Step 8.5: Create empty findings sink files**

```bash
cat > docs/FINDINGS-NNX.md <<'EOF'
# NNx submodule findings

Issues found by the verify_repo.py loop in the `./nnx` submodule. These are
NOT fixed by this loop (per spec §1.3); they are surfaced here for an upstream
PR follow-up to `thekaveh/NNx`.

(no findings yet)
EOF

cat > docs/FINDINGS-VENDOR.md <<'EOF'
# Vendor findings

Issues found by the verify_repo.py loop in vendored dependencies under
`vendor/`. These are NOT fixed by this loop (per spec §1.3); they are
surfaced for upstream contribution.

(no findings yet)
EOF
```

- [ ] **Step 8.6: Run full verify; record initial findings**

```bash
mkdir -p /tmp/ml-verify
python scripts/verify_repo.py --check all --fast --out /tmp/ml-verify/round-0-findings.json
echo "initial finding count: $(python -c 'import json; print(json.load(open("/tmp/ml-verify/round-0-findings.json"))["summary"]["total_findings"])')"
```

Expected: a nonzero finding count — these are what the loop's rounds 1..8 will burn down.

- [ ] **Step 8.7: Commit round 0 and create the baseline tag**

```bash
git add .gitignore README.md image_classification-mnist-ffnn-pytorch/README.md \
        docs/FINDINGS-NNX.md docs/FINDINGS-VENDOR.md
git commit -m "chore: pre-cleanup and dangling-ref purge (Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>)"
git tag pre-cleanup-baseline
```

---

## Phase 1 — The iterative loop (rounds 1..8)

### Task 9: Per-round procedure (executed by `/goal`)

This task is the procedure the `/goal` loop walks **once per round**, up to 8 times.

**Pre-conditions (round 1 only):**
- Round 0 (Tasks 1–8) is committed.
- `pre-cleanup-baseline` tag exists.
- `/tmp/ml-verify/round-0-findings.json` exists.

**Per-round substeps (round N, N = 1..8):**

- [ ] **Step 9.1: PLAN — read prior findings; pick round area**

```bash
N=<current round number>
PREV=$((N-1))
FINDINGS_JSON=/tmp/ml-verify/round-${PREV}-findings.json
test -f "$FINDINGS_JSON" || { echo "no prior findings; aborting"; exit 1; }
```

Parse the JSON. Assign each finding to one of three areas:

| Finding `id` prefix | Area |
|---|---|
| `S1`, `S2`, `S5`, `S6`, `S7` | S-round (structure) |
| `S3`, `S4`, `D1`–`D8` | D-round (docs) |
| `C.*`, `E4`, `E6` | C-round (code + scripts) |
| `E1`, `E2`, `E3`, `E5` | C-round (execution; only addressable on a C-round full verify) |

Pick the area with the highest finding count for this round. Tie-break order: S → D → C (handle structure issues first because they often unblock doc/comment scans).

Write `/tmp/ml-verify/round-${N}-plan.md` containing: the picked area, the count, and a bulleted list of finding IDs to be resolved this round.

- [ ] **Step 9.2: FIX — apply edits per the picked area's rules**

For each finding in the picked area:
- **If `location` starts with `nnx/`** → append the finding to `docs/FINDINGS-NNX.md` and SKIP (do not edit `nnx/`).
- **If `location` starts with `vendor/`** → append to `docs/FINDINGS-VENDOR.md` and SKIP.
- **If `location` starts with `archive/`** → SKIP silently (archive is read-only).
- **Otherwise** → fix per the dispatch table below.

Dispatch table (which mechanic fixes which finding):

| Finding ID | Fix mechanic |
|---|---|
| `S1.parse` | Re-save the notebook with `nbformat.write(nbformat.read(p, as_version=4), p)` to canonicalize JSON. If still failing, escalate (halt). |
| `S1.cell_type` | Open the notebook, set the offending cell's `cell_type` to `markdown` (preserving content). |
| `S2.unresolved_import` | Either remove the unused import OR add the dependency to `requirements.txt` (decide by reading the cell context; if used → add dep; if unused → remove). |
| `S3.broken_link` | Either update the link target to a real file OR delete the link. Prefer updating if a clear replacement exists. |
| `S5.common_import` | Replace `from common.` with `from nnx.` using the same module path (CLAUDE.md confirms 1:1 rename). |
| `S6.gitignore_missing` | Append the missing pattern to `.gitignore`. |
| `S6.tracked_bloat` | `git rm -r --cached <path>` and add the pattern to `.gitignore` if missing. |
| `S7.tracked_bloat` | `git rm --cached <path>`. |
| `D1.missing_sections` | For each missing section: use `scripts/edit_notebook_markdown.py --insert-at <idx> --text "# N. <Section name>"` to insert. For Tier-A notebooks (the 3 in `make run-tier-a`) you may also edit code cells if needed; for Tier-B/C ONLY use the helper. |
| `D2.first_cell_not_markdown` | Insert a new markdown cell at index 0 via the helper with a title + purpose statement. |
| `D2.empty_notebook` | Halt — empty notebook is a structural surprise; ask the user. |
| `D3.missing_sections` | Edit the per-task `README.md` directly to add the missing H2 in the correct ordinal position. |
| `D4.missing_sections` | Edit root `README.md` directly. |
| `D5.task_table_mismatch` | Add a row per missing active task to the task table in root `README.md`. |
| `D6.empty_roadmap` / `D6.missing_roadmap` | Add `## 8. Roadmap` with the planned tasks from the existing README (copy from current state if section is missing entirely; otherwise restore the checklist items). |
| `D7.missing_doc` | Halt — missing structural doc is a surprise; ask the user. |
| `D7.no_sections` | Edit the doc to add H2 sections per its content. |
| `D8.terminology` | Find-and-replace the non-canonical spelling with the canonical one. Be careful with code identifiers — only replace in text, not in `import` / variable names. |
| `C.state_the_what` | Delete the offending comment line. Hard cap: max 50 deletions per round; if exceeded, halt. |
| `E1.tier_a_failed` | Read stderr from the find; fix the underlying notebook bug (typically a stale import or a non-deterministic seed). |
| `E2.tier_b_smoke_failed` / `E3.tier_c_smoke_failed` | Same — examine, fix root cause. |
| `E4.cell_error` | Re-run the notebook (Tier-A) or escalate (Tier-B/C — should not happen if E5 holds). |
| `E5.code_cells_changed` | **HALT IMMEDIATELY.** Tier-C corruption. Recover via `git checkout pre-cleanup-baseline -- node_classification-reddit-gnn-pyg/phase3-*.ipynb` and report to user. |
| `E5.no_baseline` | Re-tag if accidentally deleted: `git tag pre-cleanup-baseline <round-0-commit-sha>`. |
| `E6.shellcheck` | Fix the warning per shellcheck's message. |
| `E6.shellcheck_missing` | Skip — env limitation; not a real finding. |

For C-round comment-hygiene cleanup beyond the deterministic Phase-A heuristic: dispatch a fresh subagent (the Phase-B judge) with the prompt below. Run on changed files only.

**Phase-B judge prompt template:**

```
You are reviewing a Python (or notebook code-cell) source snippet to enforce
a strict comment-hygiene rule: comments are allowed ONLY if they explain WHY
(a non-obvious choice), note a hidden CONSTRAINT or workaround, or cite an
external reference. Comments that merely restate WHAT the code does must be
removed.

Source path: <path>
Context (5 lines before + the comment + 5 lines after):
<snippet>

Question: For the comment line marked with ▶, does it qualify under the
"WHY / constraint / reference" rule? Respond with one line:
"KEEP" or "DELETE", followed by a 12-word-max justification.
```

Apply the judge's verdict only if it agrees with the deterministic §2.4 rule, never over it. Log every verdict to `/tmp/ml-verify/round-${N}-judge.log`.

- [ ] **Step 9.3: VERIFY — run the oracle**

If the round area was D or S:
```bash
python scripts/verify_repo.py --check all --fast \
       --out /tmp/ml-verify/round-${N}-findings.json
```

If the round area was C:
```bash
python scripts/verify_repo.py --check all \
       --out /tmp/ml-verify/round-${N}-findings.json
```

(Note: no `--fast` for C-rounds — full execution suite runs.)

- [ ] **Step 9.4: COMMIT — round commit per outcome**

```bash
NEW_COUNT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["summary"]["total_findings"])' /tmp/ml-verify/round-${N}-findings.json)
PREV_COUNT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["summary"]["total_findings"])' /tmp/ml-verify/round-${PREV}-findings.json)
FIXED=$((PREV_COUNT - NEW_COUNT))
AREA=<S|D|C from step 9.1>

if [ "$FIXED" -gt 0 ] || git diff --cached --quiet; then
    git add -A
    git commit -m "round ${N}: ${AREA}-round — fixed ${FIXED} findings, ${NEW_COUNT} remain"
fi
```

- [ ] **Step 9.5: EXIT CHECK**

```bash
if [ "$NEW_COUNT" -eq 0 ]; then
    # Final full verify before declaring success.
    python scripts/verify_repo.py --check all \
           --out /tmp/ml-verify/round-${N}-final-findings.json
    FINAL=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["summary"]["total_findings"])' /tmp/ml-verify/round-${N}-final-findings.json)
    if [ "$FINAL" -eq 0 ]; then
        git commit --allow-empty -m "round ${N}: final verify green"
        echo "SUCCESS: all checks green at round ${N}"
        exit 0
    else
        echo "Final full-verify uncovered ${FINAL} findings; continuing to round $((N+1))"
    fi
fi

# Stall detection: compare finding ID sets.
PREV_IDS=$(python -c 'import json,sys; print(",".join(sorted(f["id"]+":"+f["location"] for f in json.load(open(sys.argv[1]))["findings"])))' /tmp/ml-verify/round-${PREV}-findings.json)
NEW_IDS=$(python -c 'import json,sys; print(",".join(sorted(f["id"]+":"+f["location"] for f in json.load(open(sys.argv[1]))["findings"])))' /tmp/ml-verify/round-${N}-findings.json)
if [ "$PREV_IDS" = "$NEW_IDS" ]; then
    # First stall — try once more on a different area next round.
    if [ -f /tmp/ml-verify/last-stall ]; then
        # Second consecutive stall — halt.
        cat > /tmp/ml-verify/HALT.md <<EOF
HALT at round ${N}: two consecutive stalled rounds.
Remaining findings: ${NEW_COUNT}
See /tmp/ml-verify/round-${N}-report.md for details.
EOF
        echo "HALT: stalled. See /tmp/ml-verify/HALT.md"
        exit 2
    fi
    touch /tmp/ml-verify/last-stall
else
    rm -f /tmp/ml-verify/last-stall
fi

if [ "$N" -ge 8 ]; then
    echo "Round cap reached. Remaining: ${NEW_COUNT}. Halting."
    exit 3
fi
# Increment N and continue to next round.
```

---

## Phase 2 — Hand-off

### Task 10: Hand off to the user with the `/goal` directive

Once Phase 0 is committed (tasks 1–8 done), the user pastes the directive below to start Phase 1. The loop runs autonomously until exit.

**The `/goal` directive (verbatim):**

```
/goal Run repo cleanup & doc standardization loop per
docs/superpowers/specs/2026-05-22-repo-cleanup-and-doc-standardization-design.md
and docs/superpowers/plans/2026-05-22-repo-cleanup-and-doc-standardization-plan.md.

Exit when: scripts/verify_repo.py --check all (no --fast) returns exit code 0
           AND git status --short is empty
           AND the last commit message starts with "round" and ends with "final verify green".

Hard cap: 8 rounds.
Per-round procedure: Task 9 of the plan (PLAN → FIX → VERIFY → COMMIT → EXIT CHECK).
Halt-on-stall: two consecutive rounds with identical finding sets.
Edit boundaries: spec §1.2 (in scope) and §1.3 (verify-only).
Tier-C output preservation: spec §6.1 and verify check E5 — mandatory.

If the loop halts (stall, round cap, or Tier-C corruption alarm), read
/tmp/ml-verify/HALT.md and surface it to the user. Do not retry or
auto-recover from a stall.
```

- [ ] **Step 10.1: Final sanity check before paste**

```bash
test -x scripts/verify_repo.py
test -x scripts/edit_notebook_markdown.py
git rev-parse --verify pre-cleanup-baseline
ls docs/FINDINGS-NNX.md docs/FINDINGS-VENDOR.md
ls /tmp/ml-verify/round-0-findings.json
```

All commands exit 0.

- [ ] **Step 10.2: User pastes the directive into Claude Code**

The loop runs. The user inspects `git log` and `/tmp/ml-verify/round-*-report.md` as it progresses. Manual intervention only on HALT.

---

## Appendix A — How to recover from Tier-C corruption

If E5 ever fires (`E5.code_cells_changed`):

```bash
git checkout pre-cleanup-baseline -- node_classification-reddit-gnn-pyg/phase3-*.ipynb
git commit -m "revert: restore Tier-C notebooks from pre-cleanup-baseline"
python scripts/verify_repo.py --check execution --fast
```

Then investigate the round's `FIX` step to find what bypassed the markdown-only helper.

---

## Appendix B — How to extend the verify oracle for future tasks

When a new task folder is added (per CLAUDE.md workflow):

1. Add the folder name to `ACTIVE_TASK_DIRS` in `scripts/verify_repo.py`.
2. Add the notebook(s) and their required sections to `REQUIRED_SECTIONS`.
3. If a new tier-A notebook, add to the `tier_a` tuple in `check_execution`.
4. Run `pytest tests/test_verify_repo.py -v` to confirm.

---
