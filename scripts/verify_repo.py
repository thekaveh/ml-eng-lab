#!/usr/bin/env python3
"""Repo verification oracle.

Runs four orthogonal checks (structure, execution, docs, comments) and emits
machine-readable findings JSON + a human-readable report. Exit code 0 = no
error-severity findings (warnings are allowed and reported but don't fail
the run); 1 = at least one error finding (counts on stderr).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import tokenize
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, asdict
from pathlib import Path
from urllib.parse import unquote

import nbformat

try:
    import yaml as _yaml  # PyYAML
except ImportError:
    _yaml = None

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "verify_repo_config.yaml"
_HELP_REQUESTED = any(arg in ("-h", "--help") for arg in sys.argv[1:])


def _load_config(config_path: Path = CONFIG_PATH) -> dict:
    if _yaml is None or not config_path.exists():
        if _HELP_REQUESTED:
            return {}
        raise RuntimeError(
            "verify_repo_config.yaml is required; install PyYAML and ensure "
            "the file exists."
        )
    return _yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _active_task_dirs_from_config(config: dict) -> tuple[str, ...]:
    raw = config.get("active_task_dirs")
    if not raw:
        if _HELP_REQUESTED:
            raw = ()
        else:
            raise RuntimeError(
                "verify_repo_config.yaml is missing the required 'active_task_dirs' key."
            )
    return tuple(raw)


def _required_sections_from_config(config: dict) -> dict[str, tuple[str, ...]]:
    raw = config.get("required_sections")
    if not raw:
        return {}
    return {k: tuple(v) for k, v in raw.items()}


def _tier_a_notebooks_from_config(config: dict) -> tuple[str, ...]:
    raw = config.get("tier_a_notebooks")
    if not raw:
        if _HELP_REQUESTED:
            raw = ()
        else:
            raise RuntimeError(
                "verify_repo_config.yaml is missing the required 'tier_a_notebooks' key."
            )
    return tuple(raw)


def _apply_config(config: dict) -> None:
    global _CONFIG, ACTIVE_TASK_DIRS, REQUIRED_SECTIONS, TIER_A_NOTEBOOKS
    _CONFIG = config
    ACTIVE_TASK_DIRS = _active_task_dirs_from_config(config)
    REQUIRED_SECTIONS = _required_sections_from_config(config)
    TIER_A_NOTEBOOKS = _tier_a_notebooks_from_config(config)


_CONFIG: dict = {}
NOTEBOOK_ROOT = Path("notebooks")
DEFAULT_SUBPROCESS_TIMEOUT = 120
ACTIVE_TASK_DIRS: tuple[str, ...] = ()
REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {}
TIER_A_NOTEBOOKS: tuple[str, ...] = ()
_apply_config(_load_config())

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


@dataclass(frozen=True)
class ImportedModule:
    module: str
    line: int
    relative: bool = False


_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
_INLINE_CODE_RE = re.compile(r"`+[^`]*`+")
_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))")
_NON_PYTHON_CELL_MAGICS = frozenset({
    "bash",
    "html",
    "javascript",
    "js",
    "latex",
    "perl",
    "ruby",
    "script",
    "sh",
    "svg",
    "writefile",
})
_GITIGNORE_REQUIRED_PATTERNS = (
    "docs/superpowers/",
    "plan-*.md", "notes-*.md", "audit-*.md",
    ".mypy_cache/", ".trunk/", ".vscode/",
)
_TRACKED_SUPERPOWERS_DOC_PREFIXES = (
    "docs/superpowers/specs/",
    "docs/superpowers/plans/",
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


def _cell_magic_name(line: str) -> str:
    stripped = line.lstrip()
    if not stripped.startswith("%%"):
        return ""
    return stripped[2:].split(None, 1)[0].strip().lower()


def _literal_dynamic_import(
    node: ast.AST,
    importlib_aliases: set[str] | None = None,
    import_module_aliases: set[str] | None = None,
) -> str:
    if not isinstance(node, ast.Call) or not node.args:
        return ""
    importlib_aliases = importlib_aliases or {"importlib"}
    import_module_aliases = import_module_aliases or set()
    func = node.func
    if isinstance(func, ast.Name):
        is_import = func.id == "__import__" or func.id in import_module_aliases
    elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        is_import = func.value.id in importlib_aliases and func.attr == "import_module"
    else:
        is_import = False
    if not is_import:
        return ""
    first_arg = node.args[0]
    if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
        return ""
    return first_arg.value


def _fallback_statement(lines: list[str], start: int) -> tuple[str, int]:
    statement_lines = [lines[start]]
    balance = lines[start].count("(") - lines[start].count(")")
    end = start
    while (balance > 0 or statement_lines[-1].rstrip().endswith("\\")) and end + 1 < len(lines):
        end += 1
        next_line = lines[end]
        statement_lines.append(next_line)
        balance += next_line.count("(") - next_line.count(")")
    return "\n".join(statement_lines), end


def _imported_modules_from_source(source: str) -> Iterator[ImportedModule]:
    """Yield top-level imported module names and one-based line numbers."""
    for line in source.splitlines():
        if not line.strip():
            continue
        if _cell_magic_name(line) in _NON_PYTHON_CELL_MAGICS:
            return
        break

    cleaned_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            cleaned_lines.append("")
        else:
            cleaned_lines.append(line)
    cleaned_source = "\n".join(cleaned_lines)

    try:
        tree = ast.parse(cleaned_source)
    except SyntaxError:
        fallback_lines = _blank_multiline_string_lines(cleaned_source)
        importlib_aliases = {"importlib"}
        import_module_aliases: set[str] = set()
        idx = 0
        while idx < len(fallback_lines):
            li = idx + 1
            line = fallback_lines[idx]
            try:
                line_tree = ast.parse(line)
            except SyntaxError:
                statement, end_idx = _fallback_statement(fallback_lines, idx)
                if end_idx > idx:
                    try:
                        line_tree = ast.parse(statement)
                    except SyntaxError:
                        line_tree = None
                    if line_tree is not None:
                        line_importlib_aliases, line_import_module_aliases = _importlib_aliases(line_tree)
                        importlib_aliases.update(line_importlib_aliases)
                        import_module_aliases.update(line_import_module_aliases)
                        for node in ast.walk(line_tree):
                            location = li + getattr(node, "lineno", 1) - 1
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    module = alias.name
                                    if module:
                                        yield ImportedModule(module=module, line=location)
                            elif isinstance(node, ast.ImportFrom):
                                if node.level:
                                    module = node.module or ", ".join(alias.name for alias in node.names)
                                    yield ImportedModule(module=module or ".", line=location, relative=True)
                                elif node.module:
                                    module = node.module
                                    if module:
                                        yield ImportedModule(module=module, line=location)
                            elif module := _literal_dynamic_import(node, importlib_aliases, import_module_aliases):
                                yield ImportedModule(module=module, line=location)
                        idx = end_idx + 1
                        continue
                m = _IMPORT_RE.match(line)
                if not m:
                    idx += 1
                    continue
                module = m.group(1) or m.group(2) or ""
                if module:
                    yield ImportedModule(module=module, line=li)
                idx += 1
                continue
            line_importlib_aliases, line_import_module_aliases = _importlib_aliases(line_tree)
            importlib_aliases.update(line_importlib_aliases)
            import_module_aliases.update(line_import_module_aliases)
            for node in ast.walk(line_tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                        if module:
                            yield ImportedModule(module=module, line=li)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        module = node.module or ", ".join(alias.name for alias in node.names)
                        yield ImportedModule(module=module or ".", line=li, relative=True)
                    elif node.module:
                        module = node.module
                        if module:
                            yield ImportedModule(module=module, line=li)
                elif module := _literal_dynamic_import(node, importlib_aliases, import_module_aliases):
                    yield ImportedModule(module=module, line=li)
            idx += 1
        return

    importlib_aliases, import_module_aliases = _importlib_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                if module:
                    yield ImportedModule(module=module, line=node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                module = node.module or ", ".join(alias.name for alias in node.names)
                yield ImportedModule(
                    module=module or ".",
                    line=node.lineno,
                    relative=True,
                )
            elif node.module:
                module = node.module
                if module:
                    yield ImportedModule(module=module, line=node.lineno)
        elif module := _literal_dynamic_import(node, importlib_aliases, import_module_aliases):
            yield ImportedModule(module=module, line=node.lineno)


def _importlib_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    importlib_aliases = {"importlib"}
    import_module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    import_module_aliases.add(alias.asname or alias.name)
    return importlib_aliases, import_module_aliases


def _blank_multiline_string_lines(source: str) -> list[str]:
    lines = source.splitlines()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type != tokenize.STRING:
                continue
            start_line, _ = tok.start
            end_line, _ = tok.end
            if end_line <= start_line:
                continue
            for idx in range(start_line - 1, min(end_line, len(lines))):
                lines[idx] = ""
    except tokenize.TokenError:
        pass
    return lines


def _git_ls_files(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True,
        timeout=DEFAULT_SUBPROCESS_TIMEOUT,
    )
    return out.stdout.splitlines()


def _is_allowed_tracked_superpowers_doc(path: str) -> bool:
    if not path.endswith(".md"):
        return False
    for prefix in _TRACKED_SUPERPOWERS_DOC_PREFIXES:
        if path.startswith(prefix) and "/" not in path.removeprefix(prefix):
            return True
    return False


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return ""


def _strip_markdown_code(text: str) -> str:
    stripped: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if re.match(r"^\s*(?:```|~~~)", line):
            in_fence = not in_fence
            stripped.append(" " * len(line))
            continue
        if in_fence:
            stripped.append(" " * len(line))
            continue
        stripped.append(_INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), line))
    return "\n".join(stripped)


def _github_markdown_slug(heading: str) -> str:
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = re.sub(r"[`*_~\[\]]", "", heading).strip().lower()
    heading = re.sub(r"[^a-z0-9\s-]", "", heading)
    heading = re.sub(r"\s+", "-", heading).strip("-")
    return heading


def _markdown_heading_slugs(text: str) -> set[str]:
    counts: dict[str, int] = {}
    slugs: set[str] = set()
    for line in text.splitlines():
        m = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if not m:
            continue
        base = _github_markdown_slug(m.group(1))
        if not base:
            continue
        count = counts.get(base, 0)
        counts[base] = count + 1
        slugs.add(base if count == 0 else f"{base}-{count}")
    return slugs


def _split_markdown_link_target(target: str) -> tuple[str, str]:
    target = target.strip().strip("<>")
    target = target.split()[0] if target else ""
    if "#" not in target:
        return target, ""
    path_part, fragment = target.split("#", 1)
    return path_part, unquote(fragment)


def _iter_notebooks(repo: Path) -> Iterator[Path]:
    for d in ACTIVE_TASK_DIRS:
        for nb_path in _active_task_path(repo, d).glob("*.ipynb"):
            yield nb_path


def _active_task_path(repo: Path, task: str) -> Path:
    return repo / NOTEBOOK_ROOT / task


def _notebook_rel(path: Path, repo: Path) -> str:
    return str(path.relative_to(repo))


def _baseline_notebook_rel(rel: str) -> str:
    prefix = f"{NOTEBOOK_ROOT.as_posix()}/"
    if rel.startswith(prefix):
        return rel.removeprefix(prefix)
    return rel


def _iter_in_scope_text_files(repo: Path) -> Iterator[Path]:
    yield repo / "README.md"
    yield repo / "CONTRIBUTING.md"
    yield repo / "CHANGELOG.md"
    for p in sorted((repo / "docs").rglob("*.md")):
        if p.relative_to(repo).as_posix().startswith("docs/superpowers/"):
            continue
        yield p
    for d in ACTIVE_TASK_DIRS:
        for p in _active_task_path(repo, d).glob("*.md"):
            yield p


def _iter_in_scope_markdown_documents(repo: Path) -> Iterator[tuple[Path, Path, str]]:
    for md in _iter_in_scope_text_files(repo):
        yield md, md.parent, _read_text(md)
    for nb_path in _iter_notebooks(repo):
        try:
            doc = nbformat.read(nb_path, as_version=4)
        except Exception:
            continue
        text = "\n\n".join(
            cell.source for cell in doc.cells if cell.cell_type == "markdown"
        )
        yield nb_path, nb_path.parent, text


def _required_shellcheck_targets(repo: Path) -> tuple[Path, ...]:
    return (
        repo / "vendor" / "genai-vanilla" / "start.sh",
        repo / "vendor" / "genai-vanilla" / "stop.sh",
        repo / "vendor" / "genai-vanilla" / "bootstrapper" / "_run.sh",
        repo / "vendor" / "genai-vanilla" / "services" / "jupyterhub" / "build" / "scripts" / "startup.sh",
    )


def _shellcheck_targets(repo: Path) -> tuple[Path, ...]:
    local_scripts = tuple(sorted((repo / "scripts").glob("*.sh")))
    return tuple(path for path in (*local_scripts, *_required_shellcheck_targets(repo)) if path.exists())


def _required_submodule_paths() -> tuple[str, ...]:
    return ("vendor/genai-vanilla",)


def check_structure(repo: Path) -> CheckResult:
    result = CheckResult(name="structure")
    tracked = set(_git_ls_files(repo))

    valid_types = {"code", "markdown", "raw"}
    notebooks = list(_iter_notebooks(repo))
    for nb in notebooks:
        try:
            raw_doc = json.loads(nb.read_text(encoding="utf-8"))
            for i, c in enumerate(raw_doc.get("cells", [])):
                if "id" not in c:
                    result.findings.append(Finding(
                        id="S1.cell_id", check="structure", severity="error",
                        location=f"{nb.relative_to(repo)}:cell[{i}]",
                        message="cell is missing required nbformat v4 id",
                    ))
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
            for imported in _imported_modules_from_source(cell.source):
                module = imported.module
                module_root = module.split(".", 1)[0]
                li = imported.line
                location = f"{nb.relative_to(repo)}:cell[{ci}]:line[{li}]"
                if imported.relative:
                    result.findings.append(Finding(
                        id="S2.relative_import",
                        check="structure",
                        severity="error",
                        location=location,
                        message=(
                            "notebook uses a relative import that will not resolve "
                            f"reliably in a top-to-bottom kernel run: {module!r}"
                        ),
                    ))
                    continue
                if not module or module in seen_in_notebook:
                    continue
                seen_in_notebook.add(module)
                if module_root in sibling_modules:
                    continue
                if module_root in _RUNTIME_ONLY_MODULES:
                    continue
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
                    severity = "warning" if module_root in _RUNTIME_ONLY_MODULES else "error"
                    result.findings.append(Finding(
                        id="S2.unresolved_import", check="structure", severity=severity,
                        location=location,
                        message=(
                            f"module {module!r} not importable in verifier env"
                            + (" (expected only in runtime container)"
                               if module_root in _RUNTIME_ONLY_MODULES else "")
                        ),
                    ))

    for doc_path, base_dir, raw_text in _iter_in_scope_markdown_documents(repo):
        text = _strip_markdown_code(raw_text)
        for m in _MARKDOWN_LINK_RE.finditer(text):
            path_part, fragment = _split_markdown_link_target(m.group(1))
            target = path_part
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not target and not fragment:
                continue
            target_path = (base_dir / target).resolve() if target else doc_path.resolve()
            try:
                target_path.relative_to(repo.resolve())
            except ValueError:
                result.findings.append(Finding(
                    id="S3.repo_escape_link", check="structure", severity="error",
                    location=f"{doc_path.relative_to(repo)}",
                    message=f"internal link escapes repository root: {target}",
                    detail={"link": m.group(0)},
                ))
                continue
            if not target_path.exists():
                result.findings.append(Finding(
                    id="S3.broken_link", check="structure", severity="error",
                    location=f"{doc_path.relative_to(repo)}",
                    message=f"internal link target missing: {target}",
                    detail={"link": m.group(0)},
                ))
                continue
            if fragment and target_path.suffix.lower() == ".md":
                slugs = _markdown_heading_slugs(_read_text(target_path))
                if fragment not in slugs:
                    result.findings.append(Finding(
                        id="S3.broken_anchor", check="structure", severity="error",
                        location=f"{doc_path.relative_to(repo)}",
                        message=f"internal link anchor missing: #{fragment}",
                        detail={"link": m.group(0), "target": str(target_path.relative_to(repo))},
                    ))

    _COMMON_IMPORT_RE = re.compile(r"^\s*(?:from\s+common(?:\.\w+)*\s+import\b|import\s+common(?:\.\w+)*)")
    for path in tracked:
        if path.startswith(("tests/", "notebooks/archive/", "vendor/")):
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
        if path.startswith(("docs/superpowers/",)) and not _is_allowed_tracked_superpowers_doc(path):
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

    for script in sorted((repo / "scripts").glob("*.py")):
        if not script.is_file():
            continue
        rel = str(script.relative_to(repo))
        text = _read_text(script)
        has_shebang = text.startswith("#!")
        executable = bool(script.stat().st_mode & 0o111)
        if has_shebang != executable:
            result.findings.append(Finding(
                id="S8.script_executable_mismatch",
                check="structure",
                severity="error",
                location=rel,
                message=(
                    "script shebang and executable bit disagree; keep both "
                    "present for direct CLI scripts or both absent for modules"
                ),
                detail={"has_shebang": has_shebang, "executable": executable},
            ))

    return result


_H1_RE = re.compile(r"^# ([^\n]+)", re.MULTILINE)
_H2_RE = re.compile(r"^## ([^\n]+)", re.MULTILINE)
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+\S")
_STALE_LAYOUT_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    (
        "flat top-level task-folder guidance",
        re.compile(r"\b(?:top-level folder|top-level folders|flat top-level layout|<21 active task folders>)\b"),
    ),
    (
        "old root archive guidance",
        re.compile(r"(?<!notebooks/)archive/(?:README\.md)?"),
    ),
    (
        "old nbviewer placeholder without notebooks prefix",
        re.compile(r"nbviewer\.org/github/thekaveh/ml-eng-lab/(?:blob|tree)/main/<folder>"),
    ),
)
_STALE_ACTIVE_NOTEBOOK_PATHS: tuple[tuple[str, re.Pattern], ...] = (
    (
        "old local repo path",
        re.compile(r"/Users/[^/\s]+/repos/ml(?!-eng-lab)\b"),
    ),
    (
        "old JupyterHub repo path",
        re.compile(r"/home/jovyan/work/ml(?!-eng-lab)\b"),
    ),
    (
        "old Codespaces repo path",
        re.compile(r"/workspaces/ml(?!-eng-lab)\b"),
    ),
    (
        "old GitHub repo URL",
        re.compile(r"github\.com/thekaveh/ml-lab\b"),
    ),
    (
        "removed in-repo nnx source tree",
        re.compile(r"/(?:home/jovyan/work|workspaces)/ml-eng-lab/nnx/src/nnx\b"),
    ),
    (
        "host-local Python environment path",
        re.compile(r"/Users/[^/\s]+/\.pyenv\b"),
    ),
)


def _markdown_headings(text: str, level: int) -> list[str]:
    pat = _H1_RE if level == 1 else _H2_RE
    return [m.group(1).strip() for m in pat.finditer(text)]


def _iter_markdown_headings(text: str) -> Iterator[tuple[int, int, str]]:
    in_fence = False
    for line_no, line in enumerate(text.splitlines(), 1):
        if re.match(r"^\s*(?:```|~~~)", line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _MARKDOWN_HEADING_RE.match(line)
        if not m:
            continue
        title = m.group(2).strip().rstrip("#").strip()
        yield line_no, len(m.group(1)), title


def _iter_numbered_doc_files(repo: Path) -> Iterator[Path]:
    for rel in ("README.md", "CONTRIBUTING.md"):
        path = repo / rel
        if path.exists():
            yield path
    for rel in (
        "FINDINGS-NNX.md",
        "FINDINGS-VENDOR.md",
        "dependency-contracts.md",
        "env-setup.md",
        "jupyterhub-integration.md",
        "vscode-remote-access.md",
    ):
        path = repo / "docs" / rel
        if path.exists():
            yield path
    maintenance_dir = repo / "docs" / "maintenance"
    if maintenance_dir.exists():
        yield from sorted(maintenance_dir.glob("*.md"))
    for d in ACTIVE_TASK_DIRS:
        path = _active_task_path(repo, d) / "README.md"
        if path.exists():
            yield path


def _numbered_heading_findings(repo: Path, path: Path) -> list[Finding]:
    findings: list[Finding] = []
    for line_no, level, title in _iter_markdown_headings(_read_text(path)):
        if level == 1:
            continue
        m = _NUMBERED_HEADING_RE.match(title)
        if not m:
            findings.append(Finding(
                id="D9.numbered_heading", check="docs", severity="error",
                location=f"{path.relative_to(repo)}:{line_no}",
                message="numbered-doc heading must start with '<number>. '",
                detail={"heading": title},
            ))
            continue
        expected_depth = level - 1
        actual_depth = len(m.group(1).split("."))
        if actual_depth != expected_depth:
            findings.append(Finding(
                id="D9.numbered_heading", check="docs", severity="error",
                location=f"{path.relative_to(repo)}:{line_no}",
                message=(
                    f"numbered-doc H{level} must use {expected_depth} "
                    f"numeric component(s)"
                ),
                detail={"heading": title},
            ))
    return findings


def _stale_layout_guidance_findings(repo: Path) -> list[Finding]:
    findings: list[Finding] = []
    for rel in ("README.md", "CONTRIBUTING.md"):
        path = repo / rel
        if not path.exists():
            continue
        text = _strip_markdown_code(_read_text(path))
        for label, pattern in _STALE_LAYOUT_PATTERNS:
            for m in pattern.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                findings.append(Finding(
                    id="D11.stale_notebook_layout",
                    check="docs",
                    severity="error",
                    location=f"{rel}:{line_no}",
                    message=f"stale pre-notebooks/ layout guidance: {label}",
                    detail={"match": m.group(0)},
                ))
    return findings


def _dependency_ledger_findings(repo: Path) -> list[Finding]:
    path = repo / "docs" / "dependency-contracts.md"
    if not path.exists():
        return []
    text = _read_text(path)
    package_counts = {
        package: int(count)
        for package, count in re.findall(
            r"^\| `([^`]+)` \| `[^`]+` \| `[^`]+` \| (\d+) \|", text, re.M
        )
    }
    advisory_counts: dict[str, int] = {}
    for package, count in re.findall(
        r"^\| `([^`]+)` \| `(?:PYSEC|CVE)-[^`]+` \| (\d+) \|", text, re.M
    ):
        advisory_counts[package] = advisory_counts.get(package, 0) + int(count)

    findings: list[Finding] = []
    for package, expected in package_counts.items():
        actual = advisory_counts.get(package, 0)
        if actual != expected:
            findings.append(Finding(
                id="D10.dependency_ledger_count", check="docs", severity="error",
                location="docs/dependency-contracts.md",
                message=(
                    f"{package} advisory feed-record count is {actual}; "
                    f"expected {expected} from audit summary"
                ),
                detail={"package": package, "expected": expected, "actual": actual},
            ))

    total_match = re.search(r"Result: (\d+) known vulnerabilities", text)
    if total_match:
        expected_total = int(total_match.group(1))
        actual_total = sum(advisory_counts.values())
        if actual_total != expected_total:
            findings.append(Finding(
                id="D10.dependency_ledger_count", check="docs", severity="error",
                location="docs/dependency-contracts.md",
                message=(
                    f"advisory feed-record total is {actual_total}; "
                    f"expected {expected_total} from audit summary"
                ),
                detail={"expected": expected_total, "actual": actual_total},
            ))
    if "vendor/genai-vanilla" in text:
        ledger_sha_match = re.search(r"currently pins tree entry\s+`([0-9a-f]{40})`", text)
        if not ledger_sha_match:
            findings.append(Finding(
                id="D10.dependency_ledger_submodule_sha",
                check="docs",
                severity="error",
                location="docs/dependency-contracts.md",
                message=(
                    "genai-vanilla ledger must include a parseable 40-character "
                    "pinned tree-entry SHA"
                ),
            ))
            return findings
        ledger_sha = ledger_sha_match.group(1)
        rc, out, _err = _run(["git", "ls-files", "--stage", "--", "vendor/genai-vanilla"], repo)
        gitlink_match = re.search(r"160000 ([0-9a-f]{40}) \d+\s+vendor/genai-vanilla", out)
        if rc != 0 or not gitlink_match:
            findings.append(Finding(
                id="D10.dependency_ledger_submodule_sha",
                check="docs",
                severity="error",
                location="docs/dependency-contracts.md",
                message="genai-vanilla ledger SHA cannot be compared to a parseable superproject gitlink",
                detail={"ledger_sha": ledger_sha, "gitlink_sha": None},
            ))
            return findings
        gitlink_sha = gitlink_match.group(1)
        if ledger_sha != gitlink_sha:
            findings.append(Finding(
                id="D10.dependency_ledger_submodule_sha",
                check="docs",
                severity="error",
                location="docs/dependency-contracts.md",
                message=(
                    "genai-vanilla ledger SHA does not match the superproject "
                    "gitlink"
                ),
                detail={"ledger_sha": ledger_sha, "gitlink_sha": gitlink_sha},
            ))
    return findings


def _workflow_action_pin_ledger(text: str) -> dict[str, tuple[str, str]]:
    return {
        action: (tag, sha)
        for action, tag, sha in re.findall(
            r"^\| `([^`]+)` \| `([^`]+)` \| `([0-9a-f]{40})` \|", text, re.M
        )
    }


def _workflow_action_pin_findings(repo: Path) -> list[Finding]:
    ledger_path = repo / "docs" / "dependency-contracts.md"
    if not ledger_path.exists():
        return []
    ledger = _workflow_action_pin_ledger(_read_text(ledger_path))
    findings: list[Finding] = []
    for workflow in sorted((repo / ".github" / "workflows").glob("*.yml")):
        for line_no, line in enumerate(_read_text(workflow).splitlines(), start=1):
            m = re.search(r"\buses:\s*([^\s#]+)(?:\s*#\s*(\S+))?", line)
            if not m:
                continue
            uses_ref = m.group(1).strip("'\"")
            tag_comment = (m.group(2) or "").strip()
            if uses_ref.startswith(("./", "../")):
                continue
            if "@" not in uses_ref:
                findings.append(Finding(
                    id="D10.workflow_action_pin",
                    check="docs",
                    severity="error",
                    location=f"{workflow.relative_to(repo)}:{line_no}",
                    message=f"workflow action reference must include @ref: {uses_ref}",
                ))
                continue
            action, ref = uses_ref.rsplit("@", 1)
            if not re.fullmatch(r"[0-9a-f]{40}", ref):
                findings.append(Finding(
                    id="D10.workflow_action_pin",
                    check="docs",
                    severity="error",
                    location=f"{workflow.relative_to(repo)}:{line_no}",
                    message=f"workflow action reference must be pinned to a full SHA: {uses_ref}",
                    detail={"action": action, "ref": ref},
                ))
                continue
            if action not in ledger:
                findings.append(Finding(
                    id="D10.workflow_action_pin",
                    check="docs",
                    severity="error",
                    location=f"{workflow.relative_to(repo)}:{line_no}",
                    message=f"workflow action is SHA-pinned but missing from dependency ledger: {action}",
                    detail={"action": action, "sha": ref},
                ))
                continue
            ledger_tag, ledger_sha = ledger[action]
            if ref != ledger_sha or tag_comment != ledger_tag:
                findings.append(Finding(
                    id="D10.workflow_action_pin",
                    check="docs",
                    severity="error",
                    location=f"{workflow.relative_to(repo)}:{line_no}",
                    message=(
                        "workflow action SHA/comment must match dependency ledger "
                        f"for {action}"
                    ),
                    detail={
                        "action": action,
                        "workflow_sha": ref,
                        "workflow_tag_comment": tag_comment,
                        "ledger_sha": ledger_sha,
                        "ledger_tag": ledger_tag,
                    },
                ))
    return findings


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

    configured_notebooks = set(REQUIRED_SECTIONS)
    for nb in _iter_notebooks(repo):
        rel = _notebook_rel(nb, repo)
        if rel not in configured_notebooks:
            result.findings.append(Finding(
                id="D1.unconfigured_notebook", check="docs", severity="error",
                location=rel,
                message=(
                    "active notebook is missing from verify_repo_config.yaml "
                    "required_sections; docs and papermill-parameter checks "
                    "would otherwise skip it"
                ),
            ))

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
        readme = _active_task_path(repo, d) / "README.md"
        readme_rel = f"{NOTEBOOK_ROOT.as_posix()}/{d}/README.md"
        if not readme.exists():
            result.findings.append(Finding(
                id="D3.missing_readme", check="docs", severity="error",
                location=readme_rel, message="per-task README missing",
            ))
            continue
        h2s = _markdown_headings(_read_text(readme), level=2)
        ok, missing = _ordered_contains(README_REQUIRED_H2, h2s)
        if not ok:
            result.findings.append(Finding(
                id="D3.missing_sections", check="docs", severity="error",
                location=readme_rel,
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
    active_count = sum(1 for d in ACTIVE_TASK_DIRS if _active_task_path(repo, d).is_dir())
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

    for path in _iter_numbered_doc_files(repo):
        result.findings.extend(_numbered_heading_findings(repo, path))

    result.findings.extend(_dependency_ledger_findings(repo))
    result.findings.extend(_workflow_action_pin_findings(repo))
    result.findings.extend(_stale_layout_guidance_findings(repo))

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
        for p in _active_task_path(repo, d).glob("*.py"):
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


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _cell_tags(cell) -> set[str]:
    return set(cell.get("metadata", {}).get("tags") or [])


def _is_parameters_cell(cell) -> bool:
    tags = _cell_tags(cell)
    return "parameters" in tags or "injected-parameters" in tags


def _code_cell_sources_for_baseline(doc) -> list[str]:
    return [
        cell.source
        for cell in doc.cells
        if cell.cell_type == "code" and not _is_parameters_cell(cell)
    ]


def _parameter_trailing_comment_findings(doc, rel: str) -> list[Finding]:
    findings: list[Finding] = []
    bad_line_re = re.compile(r"^\s*[A-Za-z_]\w*\s*=.*#.*=")
    for ci, cell in enumerate(doc.cells):
        if cell.cell_type != "code" or not _is_parameters_cell(cell):
            continue
        for li, line in enumerate(cell.source.splitlines(), start=1):
            if bad_line_re.search(line):
                findings.append(Finding(
                    id="E9.parameter_trailing_comment",
                    check="execution",
                    severity="error",
                    location=f"{rel}:cell[{ci}]:line[{li}]",
                    message=(
                        "parameters cell assignment has a trailing comment with "
                        "'='; papermill 2.7 cannot inspect it reliably"
                    ),
                ))
    return findings


def _assignment_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()

    def collect(target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                collect(elt)

    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets.append(node.target)
        elif isinstance(node, ast.AugAssign):
            targets.append(node.target)
        for target in targets:
            collect(target)
    return names


def _parameters_assignment_names(doc) -> set[str]:
    names: set[str] = set()
    for cell in doc.cells:
        if cell.cell_type == "code" and _is_parameters_cell(cell):
            names.update(_assignment_names(cell.source))
    return names


def _makefile_variable_items(repo: Path, name: str) -> tuple[str, ...]:
    lines = _read_text(repo / "Makefile").splitlines()
    items: list[str] = []
    collecting = False
    prefix = f"{name} :="
    for line in lines:
        stripped = line.strip()
        if not collecting:
            if not stripped.startswith(prefix):
                continue
            collecting = True
            stripped = stripped[len(prefix):].strip()
        if stripped.endswith("\\"):
            stripped = stripped[:-1].strip()
            keep_collecting = True
        else:
            keep_collecting = False
        if stripped:
            items.extend(stripped.split())
        if collecting and not keep_collecting:
            break
    return tuple(items)


def _ci_tier_a_artifact_paths(repo: Path) -> tuple[str, ...]:
    workflow = repo / ".github" / "workflows" / "ci.yml"
    if _yaml is None or not workflow.exists():
        return ()
    data = _yaml.safe_load(workflow.read_text(encoding="utf-8")) or {}
    steps = data.get("jobs", {}).get("tier-a-papermill", {}).get("steps", [])
    for step in steps:
        if step.get("name") != "Upload refreshed notebook outputs as artifact":
            continue
        raw_path = step.get("with", {}).get("path", "")
        return tuple(
            line.strip()
            for line in str(raw_path).splitlines()
            if line.strip()
        )
    return ()


def _run(
    cmd: list[str], cwd: Path, timeout: int | None = DEFAULT_SUBPROCESS_TIMEOUT
) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # rc=124 mirrors GNU `timeout(1)`: callers already branch on rc != 0
        # to surface a Finding, so a hung make target produces a clean
        # error rather than crashing the verifier.
        stdout = _subprocess_text(e.stdout)
        stderr = _subprocess_text(e.stderr)
        return 124, stdout, stderr + f"\n[verify_repo] timed out after {timeout}s"
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
    phase3 = list(_active_task_path(repo, "node_classification-reddit-gnn-pyg").glob("phase3-*.ipynb"))
    for nb in phase3:
        rel = str(nb.relative_to(repo))
        baseline_rel = _baseline_notebook_rel(rel)
        try:
            head_doc = nbformat.read(nb, as_version=4)
        except Exception as e:
            findings.append(Finding(
                id="E5.head_parse_failed", check="execution", severity="error",
                location=rel,
                message=f"HEAD notebook unparseable: {e}",
            ))
            continue
        rc, raw, err = _run(
            ["git", "show", f"pre-cleanup-baseline:{baseline_rel}"], repo,
        )
        if rc != 0:
            findings.append(Finding(
                id="E5.baseline_read_failed", check="execution", severity="error",
                location=rel,
                message=f"could not read baseline: {err.strip()[:120]}",
            ))
            continue
        try:
            base_doc = nbformat.reads(raw, as_version=4)
        except Exception as e:
            findings.append(Finding(
                id="E5.baseline_parse_failed", check="execution", severity="error",
                location=rel,
                message=f"baseline notebook unparseable: {e}",
            ))
            continue
        head_codes = _code_cell_sources_for_baseline(head_doc)
        base_codes = _code_cell_sources_for_baseline(base_doc)
        if head_codes != base_codes:
            findings.append(Finding(
                id="E5.code_cells_changed", check="execution", severity="error",
                location=rel,
                message="Tier-C code cells diverged from baseline",
                detail={"head_count": len(head_codes), "base_count": len(base_codes)},
            ))
    return findings


def _runtime_available() -> bool:
    """True when the heavyweight ML runtime (torch, PyG) is importable in this env.

    The Tier-A/B/C papermill targets exercise notebooks that import torch,
    torch_geometric, and PyG's compiled extension stack. When these are
    missing, running the make targets fails with environment errors that have
    nothing to do with the notebooks' correctness — so we downgrade E1-E3 to
    env-limited skips (warning), not errors. The full execution check is
    meaningful only in the genai-vanilla container or an equivalent
    fully-provisioned env.
    """
    for canary in (
        "torch",
        "torch_geometric",
        "torch_sparse",
        "torch_scatter",
        "torch_cluster",
        "torch_spline_conv",
        "pyg_lib",
    ):
        if importlib.util.find_spec(canary) is None:
            return False
    return True


def check_execution(repo: Path, fast: bool) -> CheckResult:
    result = CheckResult(name="execution")

    make_tier_a = _makefile_variable_items(repo, "TIER_A")
    if not make_tier_a:
        result.findings.append(Finding(
            id="E11.tier_a_makefile_missing",
            check="execution",
            severity="error",
            location="Makefile:TIER_A",
            message="Makefile TIER_A is missing or empty; Tier-A execution contract is unenforceable",
        ))
    elif make_tier_a != TIER_A_NOTEBOOKS:
        result.findings.append(Finding(
            id="E11.tier_a_config_drift",
            check="execution",
            severity="error",
            location="Makefile:TIER_A",
            message="Makefile TIER_A drifted from scripts/verify_repo_config.yaml tier_a_notebooks",
            detail={
                "makefile_only": sorted(set(make_tier_a) - set(TIER_A_NOTEBOOKS)),
                "config_only": sorted(set(TIER_A_NOTEBOOKS) - set(make_tier_a)),
            },
        ))

    ci_tier_a_artifacts = _ci_tier_a_artifact_paths(repo)
    if not ci_tier_a_artifacts:
        result.findings.append(Finding(
            id="E12.tier_a_artifact_paths_missing",
            check="execution",
            severity="error",
            location=".github/workflows/ci.yml:tier-a-papermill",
            message="Tier-A artifact upload paths are missing or empty",
        ))
    elif ci_tier_a_artifacts != TIER_A_NOTEBOOKS:
        result.findings.append(Finding(
            id="E12.tier_a_artifact_paths_drift",
            check="execution",
            severity="error",
            location=".github/workflows/ci.yml:tier-a-papermill",
            message="Tier-A artifact upload paths drifted from verifier config",
            detail={
                "artifact_only": sorted(set(ci_tier_a_artifacts) - set(TIER_A_NOTEBOOKS)),
                "config_only": sorted(set(TIER_A_NOTEBOOKS) - set(ci_tier_a_artifacts)),
            },
        ))

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
        elif "SMOKE_TEST" not in _parameters_assignment_names(doc):
            result.findings.append(Finding(
                id="E10.missing_smoke_test_parameter",
                check="execution",
                severity="error",
                location=rel,
                message=(
                    "parameters-tagged cell does not assign SMOKE_TEST; "
                    "make smoke targets pass `-p SMOKE_TEST 1`"
                ),
            ))
        result.findings.extend(_parameter_trailing_comment_findings(doc, rel))

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

    for nb in _iter_notebooks(repo):
        rel = _notebook_rel(nb, repo)
        text = _read_text(nb)
        for label, pattern in _STALE_ACTIVE_NOTEBOOK_PATHS:
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                result.findings.append(Finding(
                    id="E13.stale_active_notebook_path",
                    check="execution",
                    severity="warning",
                    location=f"{rel}:line[{line_no}]",
                    message=f"stale active-notebook path artifact: {label}",
                ))
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        papermill_meta = doc.get("metadata", {}).get("papermill") or {}
        output_path = str(papermill_meta.get("output_path", ""))
        if papermill_meta:
            result.findings.append(Finding(
                id="E14.source_papermill_metadata",
                check="execution",
                severity="warning",
                location=rel,
                message=(
                    "active source notebook carries top-level papermill metadata; "
                    "strip generated-run metadata before committing"
                ),
            ))
        if output_path.startswith("/tmp/"):
            result.findings.append(Finding(
                id="E14.tmp_papermill_output_path",
                check="execution",
                severity="warning",
                location=rel,
                message=(
                    "notebook metadata.papermill.output_path points at /tmp; "
                    "strip or refresh papermill metadata before committing"
                ),
            ))

    result.findings.extend(_phase3_code_cells_unchanged(repo))

    for submodule in _required_submodule_paths():
        rc, out, err = _run(["git", "submodule", "status", "--", submodule], repo)
        if rc != 0:
            result.findings.append(Finding(
                id="E6.submodule_status",
                check="execution",
                severity="warning",
                location=submodule,
                message=f"could not inspect required submodule status: {(out + err).strip()[-300:]}",
            ))
            continue
        status = out.strip()
        if status.startswith(("+", "-", "U")):
            result.findings.append(Finding(
                id="E6.submodule_dirty",
                check="execution",
                severity="error",
                location=submodule,
                message=(
                    "required submodule checkout does not match the superproject "
                    "gitlink; stage the intended gitlink or run git submodule update"
                ),
                detail={"status": status},
            ))
            continue
        submodule_repo = repo / submodule
        rc, out, err = _run(["git", "status", "--porcelain", "--", "."], submodule_repo)
        if rc != 0:
            result.findings.append(Finding(
                id="E6.submodule_status",
                check="execution",
                severity="warning",
                location=submodule,
                message=f"could not inspect required submodule worktree: {(out + err).strip()[-300:]}",
            ))
            continue
        worktree_status = out.strip()
        if worktree_status:
            result.findings.append(Finding(
                id="E6.submodule_dirty",
                check="execution",
                severity="error",
                location=submodule,
                message=(
                    "required submodule checkout has local modifications; commit, "
                    "stash, or discard them before recording consumed-contract parity"
                ),
                detail={"status": worktree_status},
            ))

    for sh in _required_shellcheck_targets(repo):
        if not sh.exists():
            result.findings.append(Finding(
                id="E6.shellcheck_target_missing",
                check="execution",
                severity="error",
                location=str(sh.relative_to(repo)),
                message=(
                    "required consumed shellcheck target is missing; "
                    "initialize submodules or update the consumed contract"
                ),
            ))

    rc_shellcheck, _, _ = _run(["which", "shellcheck"], repo)
    if rc_shellcheck != 0:
        result.findings.append(Finding(
            id="E6.shellcheck_missing", check="execution", severity="warning",
            location="<env>",
            message="shellcheck not on PATH; install with `brew install shellcheck`",
        ))
    else:
        for sh in _shellcheck_targets(repo):
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
        "--check",
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
        "--repo-root", type=Path, default=REPO_ROOT,
        help=argparse.SUPPRESS,
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

    if args.check is None and args.phase_b_out is None:
        parser.error("--check is required unless --phase-b-out is used")

    repo_root = args.repo_root.resolve()
    repo_config_path = repo_root / "scripts" / "verify_repo_config.yaml"
    if repo_config_path.exists() and repo_config_path.resolve() != CONFIG_PATH.resolve():
        _apply_config(_load_config(repo_config_path))

    if args.phase_b_out is not None:
        count = export_phase_b_candidates(repo_root, args.phase_b_out)
        print(f"verify_repo: {count} Phase-B candidates → {args.phase_b_out}", file=sys.stderr)
        return 0

    if args.check == "all":
        checks_to_run = list(CHECKS.keys())
    else:
        checks_to_run = [args.check]

    # Only check_execution respects --fast; the other three never read it.
    results = [
        CHECKS[name](repo_root, args.fast) if name == "execution" else CHECKS[name](repo_root)
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
