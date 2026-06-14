#!/usr/bin/env python3
"""Repo verification oracle.

Runs four orthogonal checks (structure, execution, docs, comments) and emits
machine-readable findings JSON + a human-readable report. Exit code 0 = no
error-severity findings (warnings are allowed and reported but don't fail
the run); 1 = at least one error finding (counts on stderr).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, asdict
from pathlib import Path

import nbformat

try:
    import yaml as _yaml  # PyYAML
except ImportError:
    _yaml = None

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "verify_repo_config.yaml"


def _load_config() -> dict:
    if _yaml is None or not CONFIG_PATH.exists():
        raise RuntimeError(
            "verify_repo_config.yaml is required; install PyYAML and ensure "
            "the file exists."
        )
    return _yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


_CONFIG = _load_config()

_active_task_dirs_raw = _CONFIG.get("active_task_dirs")
if not _active_task_dirs_raw:
    raise RuntimeError(
        "verify_repo_config.yaml is missing the required 'active_task_dirs' key."
    )
ACTIVE_TASK_DIRS = tuple(_active_task_dirs_raw)

VERIFY_ONLY_DIRS = ("archive", "vendor")


def _required_sections_from_config() -> dict[str, tuple[str, ...]]:
    raw = _CONFIG.get("required_sections")
    if not raw:
        return {}
    return {k: tuple(v) for k, v in raw.items()}


REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = _required_sections_from_config()

_tier_a_raw = _CONFIG.get("tier_a_notebooks")
if not _tier_a_raw:
    raise RuntimeError(
        "verify_repo_config.yaml is missing the required 'tier_a_notebooks' key."
    )
TIER_A_NOTEBOOKS = tuple(_tier_a_raw)

README_REQUIRED_H2 = (
    "1. Task summary", "2. Why this exists", "3. What's in the notebook",
    "4. How to run", "5. Dependencies", "6. Known issues",
)

ROOT_README_REQUIRED_H2 = (
    "1. Overview", "2. Repository layout", "3. Quick start", "4. Tasks",
    "5. Notebook re-execution policy", "6. NNx library",
    "7. Repository conventions", "8. Roadmap", "9. License",
)

TERMINOLOGY_CANONICALS = {
    "genai-vanilla": ("Genai-Vanilla", "GenAI-Vanilla", "GenAI Vanilla", "genai vanilla"),
    "JupyterHub": ("Jupyterhub", "Jupyter Hub", "jupyter hub"),
    "NumPy": ("Numpy", "NUMPY"),
    "PyTorch": ("Pytorch", "PYTORCH", "Py-Torch"),
    "PyG": ("PYG", "Pyg"),
}


@dataclass
class Finding:
    id: str
    check: str
    severity: str
    location: str
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class CheckResult:
    name: str
    findings: list[Finding] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


_INTERNAL_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)#][^)#]*)(#[^)]*)?\)")
_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))")
_GITIGNORE_REQUIRED_PATTERNS = (
    "docs/superpowers/",
    "plan-*.md", "notes-*.md", "audit-*.md",
    ".mypy_cache/", ".trunk/", ".vscode/",
)
_BLOAT_PATTERNS = (
    "__pycache__", ".ipynb_checkpoints", ".DS_Store",
    ".mypy_cache", ".pytest_cache",
)
# Top-level dirs that should not exist at all (either tracked or untracked).
_FORBIDDEN_TOPLEVEL_DIRS = ("common",)

# Modules expected to be available in the genai-vanilla jupyterhub runtime but
# not necessarily in the verifier's lightweight venv. S2 reports these as
# warnings rather than errors when missing locally.
_RUNTIME_ONLY_MODULES = frozenset({
    "numpy",
    "torch", "torchvision", "torch_geometric", "torch_sparse", "torch_scatter",
    "torch_cluster", "torch_spline_conv", "pyg_lib",
    "matplotlib", "seaborn", "pandas", "sklearn", "scipy",
    "networkx", "community",
    "nnx",
    "tqdm",
})


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


def _iter_notebooks(repo: Path) -> Iterator[Path]:
    for d in ACTIVE_TASK_DIRS:
        for nb_path in (repo / d).glob("*.ipynb"):
            yield nb_path


def _iter_in_scope_text_files(repo: Path) -> Iterator[Path]:
    yield repo / "README.md"
    yield repo / "CONTRIBUTING.md"
    yield repo / "CHANGELOG.md"
    for p in (repo / "docs").glob("*.md"):
        yield p
    for d in ACTIVE_TASK_DIRS:
        for p in (repo / d).glob("*.md"):
            yield p


def check_structure(repo: Path) -> CheckResult:
    result = CheckResult(name="structure")
    tracked = set(_git_ls_files(repo))

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

    for nb in notebooks:
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        sibling_modules = {
            p.stem for p in nb.parent.glob("*.py") if p.stem != "__init__"
        }
        seen_in_notebook: set[str] = set()
        for ci, cell in enumerate(doc.cells):
            if cell.cell_type != "code":
                continue
            for li, line in enumerate(cell.source.splitlines()):
                m = _IMPORT_RE.match(line)
                if not m:
                    continue
                module = (m.group(1) or m.group(2) or "").split(".")[0]
                if not module or module in seen_in_notebook:
                    continue
                seen_in_notebook.add(module)
                if module in sibling_modules:
                    continue
                location = f"{nb.relative_to(repo)}:cell[{ci}]:line[{li}]"
                try:
                    spec = importlib.util.find_spec(module)
                except (ImportError, ValueError) as e:
                    result.findings.append(Finding(
                        id="S2.import_error", check="structure", severity="warning",
                        location=location,
                        message=f"find_spec({module!r}) raised {e!r}",
                    ))
                    continue
                if spec is None:
                    severity = "warning" if module in _RUNTIME_ONLY_MODULES else "error"
                    result.findings.append(Finding(
                        id="S2.unresolved_import", check="structure", severity=severity,
                        location=location,
                        message=(
                            f"module {module!r} not importable in verifier env"
                            + (" (expected only in runtime container)"
                               if module in _RUNTIME_ONLY_MODULES else "")
                        ),
                    ))

    for md in _iter_in_scope_text_files(repo):
        text = _read_text(md)
        for m in _INTERNAL_LINK_RE.finditer(text):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            target_path = (md.parent / target).resolve()
            if not target_path.exists():
                result.findings.append(Finding(
                    id="S3.broken_link", check="structure", severity="error",
                    location=f"{md.relative_to(repo)}",
                    message=f"internal link target missing: {target}",
                    detail={"link": m.group(0)},
                ))

    _COMMON_IMPORT_RE = re.compile(r"^\s*(?:from\s+common(?:\.\w+)*\s+import\b|import\s+common(?:\.\w+)*)")
    for path in tracked:
        if path.startswith(("tests/", "archive/", "vendor/")):
            continue
        full = repo / path
        if not full.is_file():
            continue
        suffix = full.suffix.lower()
        if suffix == ".py":
            for i, line in enumerate(_read_text(full).splitlines(), 1):
                if _COMMON_IMPORT_RE.match(line):
                    result.findings.append(Finding(
                        id="S5.common_import", check="structure", severity="error",
                        location=f"{path}:{i}",
                        message="forbidden import; use `from nnx.` instead",
                    ))
        elif suffix == ".ipynb":
            try:
                doc = nbformat.read(full, as_version=4)
            except Exception:
                continue
            for ci, cell in enumerate(doc.cells):
                if cell.cell_type != "code":
                    continue
                for li, line in enumerate(cell.source.splitlines(), 1):
                    if _COMMON_IMPORT_RE.match(line):
                        result.findings.append(Finding(
                            id="S5.common_import", check="structure", severity="error",
                            location=f"{path}:cell[{ci}]:line[{li}]",
                            message="forbidden import; use `from nnx.` instead",
                        ))

    gitignore_lines = {
        line.strip() for line in _read_text(repo / ".gitignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for pat in _GITIGNORE_REQUIRED_PATTERNS:
        if pat not in gitignore_lines:
            result.findings.append(Finding(
                id="S6.gitignore_missing", check="structure", severity="error",
                location=".gitignore",
                message=f"required pattern absent: {pat}",
            ))
    for path in tracked:
        if path.startswith(("docs/superpowers/",)):
            result.findings.append(Finding(
                id="S6.tracked_bloat", check="structure", severity="error",
                location=path,
                message="bloat directory tracked; should be gitignored",
            ))

    for path in tracked:
        for pat in _BLOAT_PATTERNS:
            if pat in path:
                result.findings.append(Finding(
                    id="S7.tracked_bloat", check="structure", severity="error",
                    location=path,
                    message=f"bloat artifact tracked: contains {pat!r}",
                ))

    for d in _FORBIDDEN_TOPLEVEL_DIRS:
        if (repo / d).exists():
            result.findings.append(Finding(
                id="S7.forbidden_toplevel", check="structure", severity="error",
                location=d,
                message=(
                    "forbidden top-level directory exists (tracked or not); "
                    "violates repo conventions — see CONTRIBUTING.md"
                ),
            ))

    return result


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
    missing: list[str] = []
    needed_idx = 0
    actual_idx = 0
    while needed_idx < len(required) and actual_idx < len(actual):
        if required[needed_idx].lower() in actual[actual_idx].lower():
            needed_idx += 1
        actual_idx += 1
    while needed_idx < len(required):
        missing.append(required[needed_idx])
        needed_idx += 1
    return (not missing, missing)


def check_docs(repo: Path) -> CheckResult:
    result = CheckResult(name="docs")

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

    root_text = _read_text(root_readme)
    table_rows = sum(
        1 for line in root_text.splitlines()
        if line.startswith("| [") and "/](" in line
    )
    active_count = sum(1 for d in ACTIVE_TASK_DIRS if (repo / d).is_dir())
    if table_rows < active_count:
        result.findings.append(Finding(
            id="D5.task_table_mismatch", check="docs", severity="error",
            location="README.md",
            message=f"task table has {table_rows} rows; expected >= {active_count} active",
        ))

    roadmap_marker = None
    for candidate in ("## 8. Roadmap", "## Roadmap"):
        if candidate in root_text:
            roadmap_marker = candidate
            break
    if roadmap_marker is None:
        result.findings.append(Finding(
            id="D6.missing_roadmap", check="docs", severity="error",
            location="README.md", message="Roadmap section absent",
        ))
    else:
        body = root_text.split(roadmap_marker, 1)[1]
        body = body.split("\n## ", 1)[0]
        if not re.search(r"-\s*\[\s*[xX ]\s*\]\s+\S", body):
            result.findings.append(Finding(
                id="D6.empty_roadmap", check="docs", severity="warning",
                location="README.md",
                message="Roadmap section present but has no checklist items",
            ))

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

    for path in _iter_in_scope_text_files(repo):
        text = _read_text(path)
        for canonical, deviations in TERMINOLOGY_CANONICALS.items():
            for dev in deviations:
                for m in re.finditer(rf"\b{re.escape(dev)}\b", text):
                    line_no = text.count("\n", 0, m.start()) + 1
                    result.findings.append(Finding(
                        id="D8.terminology", check="docs", severity="warning",
                        location=f"{path.relative_to(repo)}:{line_no}",
                        message=f"non-canonical spelling {dev!r}; use {canonical!r}",
                    ))

    return result


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
    # verify_repo.py is the scanner itself; scanning its own source produces
    # spurious C.state_the_what hits on its rule-matching helpers. The other
    # scripts under scripts/ are in scope.
    for p in (repo / "scripts").glob("*.py"):
        if p.name == "verify_repo.py":
            continue
        yield p, _read_text(p)
    for d in ACTIVE_TASK_DIRS:
        for p in (repo / d).glob("*.py"):
            yield p, _read_text(p)
    for nb in _iter_notebooks(repo):
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        for ci, cell in enumerate(doc.cells):
            if cell.cell_type != "code":
                continue
            # Papermill `parameters`-tagged cells carry convention-bound
            # boilerplate (see scripts/inject_smoke_test_cell.py). Their
            # leading comments document the papermill -p invocation
            # contract — they're documentation, not state-the-what hits.
            # Same self-exclusion principle as the verify_repo.py skip
            # above.
            tags = cell.get("metadata", {}).get("tags") or []
            if "parameters" in tags:
                continue
            marker = nb.with_name(f"{nb.name}#cell[{ci}]")
            yield marker, cell.source


def check_comments(repo: Path) -> CheckResult:
    result = CheckResult(name="comments")
    for path_marker, source in _iter_in_scope_code(repo):
        try:
            rel = path_marker.relative_to(repo)
            location_prefix = str(rel)
        except (ValueError, AttributeError):
            location_prefix = str(path_marker)
        for f in _scan_source_for_comments(source, location_prefix):
            result.findings.append(f)
    return result


def export_phase_b_candidates(repo: Path, out_path: Path) -> int:
    """Phase-B LLM judge input.

    Phase A (the deterministic heuristic above) catches obvious state-the-what
    comments. Phase B is meant to send the *survivors* — comments that look
    plausible but might still be redundant — to an LLM judge.

    This function exports the candidates: for each comment line that survived
    Phase A, the 5 lines before, the comment line, and the 5 lines after.
    A calling agent (or the /goal loop) reads this JSON, dispatches a subagent
    per file with the prompt below, and applies the verdict.

    Judge prompt template:

        You are reviewing a Python source snippet to enforce a strict
        comment-hygiene rule: comments are allowed ONLY if they explain WHY
        (a non-obvious choice), note a hidden CONSTRAINT or workaround, or
        cite an external reference. Comments that merely restate WHAT the
        code does must be removed.

        Source path: <path>
        Context (5 before, comment, 5 after; comment marked ▶):
        <snippet>

        Respond with: "KEEP" or "DELETE", a colon, then a 12-word-max
        justification.

    Returns the candidate count.
    """
    candidates = []
    for path_marker, source in _iter_in_scope_code(repo):
        try:
            rel = path_marker.relative_to(repo)
            location_prefix = str(rel)
        except (ValueError, AttributeError):
            location_prefix = str(path_marker)
        # Phase A scanner reports state-the-what — opposite filter wanted here:
        # comments that DIDN'T match the heuristic but exist anyway.
        a_flagged_lines = {
            int(f.location.rsplit(":", 1)[-1])
            for f in _scan_source_for_comments(source, location_prefix)
        }
        lines = source.splitlines()
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if not stripped.startswith("#"):
                continue
            if (i + 1) in a_flagged_lines:
                continue  # already flagged by Phase A
            # 5 lines of context on each side.
            start = max(0, i - 5)
            end = min(len(lines), i + 6)
            snippet = "\n".join(
                ("▶ " if j == i else "  ") + lines[j]
                for j in range(start, end)
            )
            candidates.append({
                "location": f"{location_prefix}:{i+1}",
                "comment": stripped,
                "snippet": snippet,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "schema_version": 1,
            "candidate_count": len(candidates),
            "candidates": candidates,
        }, indent=2),
        encoding="utf-8",
    )
    return len(candidates)


def _run(cmd: list[str], cwd: Path, timeout: int | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # rc=124 mirrors GNU `timeout(1)`: callers already branch on rc != 0
        # to surface a Finding, so a hung make target produces a clean
        # error rather than crashing the verifier.
        return 124, e.stdout or "", (e.stderr or "") + f"\n[verify_repo] timed out after {timeout}s"
    return proc.returncode, proc.stdout, proc.stderr


def _phase3_code_cells_unchanged(repo: Path) -> list[Finding]:
    findings: list[Finding] = []
    rc, _, _ = _run(["git", "rev-parse", "--verify", "pre-cleanup-baseline"], repo)
    if rc != 0:
        findings.append(Finding(
            id="E5.no_baseline", check="execution", severity="warning",
            location="<git>",
            message="pre-cleanup-baseline tag missing; E5 not enforceable",
        ))
        return findings
    phase3 = list((repo / "node_classification-reddit-gnn-pyg").glob("phase3-*.ipynb"))
    for nb in phase3:
        try:
            head_doc = nbformat.read(nb, as_version=4)
        except Exception as e:
            findings.append(Finding(
                id="E5.head_parse_failed", check="execution", severity="error",
                location=str(nb.relative_to(repo)),
                message=f"HEAD notebook unparseable: {e}",
            ))
            continue
        rc, raw, err = _run(
            ["git", "show", f"pre-cleanup-baseline:{nb.relative_to(repo)}"], repo,
        )
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
        head_codes = [c.source for c in head_doc.cells if c.cell_type == "code"]
        base_codes = [c.source for c in base_doc.cells if c.cell_type == "code"]
        if head_codes != base_codes:
            findings.append(Finding(
                id="E5.code_cells_changed", check="execution", severity="error",
                location=str(nb.relative_to(repo)),
                message="Tier-C code cells diverged from baseline",
                detail={"head_count": len(head_codes), "base_count": len(base_codes)},
            ))
    return findings


def _runtime_available() -> bool:
    """True when the heavyweight ML runtime (torch, PyG) is importable in this env.

    The Tier-A/B/C papermill targets exercise notebooks that import torch,
    torch_geometric, etc. When these are missing, running the make targets
    fails with environment errors that have nothing to do with the notebooks'
    correctness — so we downgrade E1-E3 to env-limited skips (warning), not
    errors. The full execution check is meaningful only in the genai-vanilla
    container or an equivalent fully-provisioned env.
    """
    for canary in ("torch", "torch_geometric"):
        if importlib.util.find_spec(canary) is None:
            return False
    return True


def check_execution(repo: Path, fast: bool) -> CheckResult:
    result = CheckResult(name="execution")

    if not fast:
        if not _runtime_available():
            result.findings.append(Finding(
                id="E1-3.runtime_unavailable", check="execution", severity="warning",
                location="<env>",
                message=(
                    "torch / torch_geometric not importable in verifier env; "
                    "Tier-A/B/C papermill targets skipped. Run verify inside "
                    "the genai-vanilla container for full execution coverage."
                ),
            ))
        else:
            # Timeouts mirror the CI caps in .github/workflows/ci.yml
            # (tier-a-papermill 90 min, smoke-tier-b/c 180 min each). Without
            # local caps a hung papermill cell blocks the verifier indefinitely.
            rc, _, err = _run(["make", "run-tier-a"], repo, timeout=5400)
            if rc != 0:
                result.findings.append(Finding(
                    id="E1.tier_a_failed", check="execution", severity="error",
                    location="Makefile:run-tier-a",
                    message=f"failed: {err.strip()[-300:]}",
                ))
            rc, _, err = _run(["make", "smoke-tier-b"], repo, timeout=10800)
            if rc != 0:
                result.findings.append(Finding(
                    id="E2.tier_b_smoke_failed", check="execution", severity="error",
                    location="Makefile:smoke-tier-b",
                    message=f"failed: {err.strip()[-300:]}",
                ))
            rc, _, err = _run(["make", "smoke-tier-c"], repo, timeout=10800)
            if rc != 0:
                result.findings.append(Finding(
                    id="E3.tier_c_smoke_failed", check="execution", severity="error",
                    location="Makefile:smoke-tier-c",
                    message=f"failed: {err.strip()[-300:]}",
                ))

    for rel in TIER_A_NOTEBOOKS:
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
                        message=(
                            f"errored output: {out.get('ename', '?')}: "
                            f"{str(out.get('evalue', ''))[:120]}"
                        ),
                    ))

    # V7: every notebook scheduled in REQUIRED_SECTIONS that's also a
    # papermill target (Tier-A/B/C) must have a cell tagged 'parameters'.
    # Without the tag, `papermill -p NAME val` silently no-ops.
    for rel in REQUIRED_SECTIONS:
        nb = repo / rel
        if not nb.exists():
            continue
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        has_params_tag = any(
            "parameters" in (c.get("metadata", {}).get("tags") or [])
            for c in doc.cells
        )
        if not has_params_tag:
            # Tag missing → papermill parameterization won't work for this
            # notebook. Warning rather than error because some notebooks
            # legitimately don't accept parameters.
            result.findings.append(Finding(
                id="E7.no_papermill_params_tag", check="execution", severity="warning",
                location=rel,
                message=(
                    "no cell tagged 'parameters'; papermill -p will silently "
                    "no-op against this notebook"
                ),
            ))

    # V6: Tier-A notebook outputs should match the current source. Cheap check:
    # for each code cell that has outputs, source byte-hash should match the
    # hash recorded in the cell's `metadata.source_hash` field if present.
    # We don't enforce — only flag drift when a freshness marker exists.
    # (No-op if the marker is absent, which it currently always is. The marker
    # gets written by a future post-execution hook; this check pre-positions
    # the verifier for that.)
    for rel in TIER_A_NOTEBOOKS:
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
            recorded = cell.get("metadata", {}).get("source_hash")
            if recorded is None:
                continue
            current = hashlib.sha256(cell.source.encode("utf-8")).hexdigest()
            if recorded != current:
                result.findings.append(Finding(
                    id="E8.stale_output", check="execution", severity="warning",
                    location=f"{rel}:cell[{ci}]",
                    message="cell source changed since last execution; re-run to refresh outputs",
                ))

    result.findings.extend(_phase3_code_cells_unchanged(repo))

    rc_shellcheck, _, _ = _run(["which", "shellcheck"], repo)
    if rc_shellcheck != 0:
        result.findings.append(Finding(
            id="E6.shellcheck_missing", check="execution", severity="warning",
            location="<env>",
            message="shellcheck not on PATH; install with `brew install shellcheck`",
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


CHECKS: dict[str, Callable[..., CheckResult]] = {
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
    parser.add_argument(
        "--phase-b-out", type=Path, default=None,
        help=(
            "Path to write Phase-B comment-hygiene candidates JSON (the input "
            "to the LLM judge subagent). When set, only this is produced; "
            "the main check loop is skipped."
        ),
    )
    args = parser.parse_args(argv)

    if args.phase_b_out is not None:
        count = export_phase_b_candidates(REPO_ROOT, args.phase_b_out)
        print(f"verify_repo: {count} Phase-B candidates → {args.phase_b_out}", file=sys.stderr)
        return 0

    if args.check == "all":
        checks_to_run = list(CHECKS.keys())
    else:
        checks_to_run = [args.check]

    # Only check_execution respects --fast; the other three never read it.
    results = [
        CHECKS[name](REPO_ROOT, args.fast) if name == "execution" else CHECKS[name](REPO_ROOT)
        for name in checks_to_run
    ]

    all_findings = [asdict(f) for r in results for f in r.findings]
    error_count = sum(1 for f in all_findings if f["severity"] == "error")
    warning_count = sum(1 for f in all_findings if f["severity"] == "warning")
    payload = {
        "schema_version": 1,
        "summary": {
            "checks_run": checks_to_run,
            "skipped": [r.name for r in results if r.skipped],
            "total_findings": len(all_findings),
            "errors": error_count,
            "warnings": warning_count,
            "by_check": {r.name: len(r.findings) for r in results},
            "by_check_errors": {
                r.name: sum(1 for f in r.findings if f.severity == "error")
                for r in results
            },
        },
        "findings": all_findings,
    }

    out_text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out_text, encoding="utf-8")
    else:
        print(out_text)

    if error_count:
        print(
            f"verify_repo: {error_count} errors, {warning_count} warnings",
            file=sys.stderr,
        )
        return 1
    if warning_count:
        print(f"verify_repo: 0 errors, {warning_count} warnings", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
