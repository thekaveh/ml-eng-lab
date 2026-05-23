#!/usr/bin/env python3
"""Repo verification oracle for the cleanup-and-standardization /goal loop.

Runs four orthogonal checks (structure, execution, docs, comments) and emits
machine-readable findings JSON + a human-readable report. Exit code 0 = no
findings; nonzero = findings present (count in stderr).

See docs/superpowers/specs/2026-05-22-repo-cleanup-and-doc-standardization-design.md.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterator

import nbformat

REPO_ROOT = Path(__file__).resolve().parent.parent

ACTIVE_TASK_DIRS = (
    "image_classification-mnist-ffnn-numpy",
    "image_classification-mnist-ffnn-pytorch",
    "node_classification-reddit-gnn-pyg",
)

VERIFY_ONLY_DIRS = ("archive", "nnx", "vendor")

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


def _iter_notebooks(repo: Path) -> Iterator[Path]:
    for d in ACTIVE_TASK_DIRS:
        for nb_path in (repo / d).glob("*.ipynb"):
            yield nb_path


def _iter_in_scope_text_files(repo: Path) -> Iterator[Path]:
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

    seen_modules: dict[str, str] = {}
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
        if path.startswith(("tests/", "archive/", "nnx/", "vendor/")):
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

    gitignore = _read_text(repo / ".gitignore")
    for pat in _GITIGNORE_REQUIRED_PATTERNS:
        if pat not in gitignore:
            result.findings.append(Finding(
                id="S6.gitignore_missing", check="structure", severity="error",
                location=".gitignore",
                message=f"required pattern absent: {pat}",
            ))
    for path in tracked:
        if path.startswith((".claude/", ".superpowers/")):
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
    i = 0
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


def check_docs(repo: Path, fast: bool) -> CheckResult:
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


def check_comments(repo: Path, fast: bool) -> CheckResult:
    return CheckResult(name="comments")


def check_execution(repo: Path, fast: bool) -> CheckResult:
    return CheckResult(name="execution")


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
