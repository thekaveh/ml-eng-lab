#!/usr/bin/env python3
"""Rewrite `from common.X` → `from nnx.X` in Jupyter notebooks.

Handles both modern (`from common.nn.X`) and old-style flat (`from common.X`)
imports. Used once during the ml-eng-lab revival to migrate notebooks to the new
NNx submodule package layout. Idempotent: re-running on already-rewritten
notebooks is a no-op.

Usage:
    python scripts/rewrite_imports.py path/to/notebook.ipynb [more.ipynb ...]
"""
from __future__ import annotations
import io
import json
import re
import sys
import tokenize
from collections.abc import Callable
from pathlib import Path

USAGE = "Usage: rewrite_imports.py NOTEBOOK [NOTEBOOK ...]"

# Order matters. Longer/more-specific patterns FIRST so they win over
# shorter prefixes.
SIMPLE_MAPPINGS: list[tuple[str, str]] = [
    # Old-style flat imports → new nested paths:
    ("from common.feed_fwd_nn",   "from nnx.nn.net.feed_fwd_nn"),
    ("from common.graph_att_nn",  "from nnx.nn.net.graph_att_nn"),
    ("from common.graph_conv_nn", "from nnx.nn.net.graph_conv_nn"),
    ("from common.graph_sage_nn", "from nnx.nn.net.graph_sage_nn"),
    ("from common.nn_model",      "from nnx.nn.nn_model"),
    ("from common.nn_params",     "from nnx.nn.params.nn_params"),
    # Modern nested imports → just rename root.
    ("from common.nn.",           "from nnx.nn."),
    ("from common.utils",         "from nnx.utils"),
    ("from common.vis_utils",     "from nnx.vis_utils"),
    # bare `import common.X`:
    ("import common.",            "import nnx."),
]

# Multi-import splits (old-style modules that combined classes now in separate files).
SPLIT_PATTERNS: list[tuple[re.Pattern[str], Callable[[re.Match[str]], list[str]]]] = [
    # `from common.nn_model import NNModel, NNTrainParams` → two lines (modern paths)
    (
        re.compile(r"^(\s*)from common\.nn_model import NNModel,\s*NNTrainParams(\s*(?:#.*)?)$"),
        lambda m: [
            f"{m.group(1)}from nnx.nn.nn_model import NNModel{m.group(2)}",
            f"{m.group(1)}from nnx.nn.params.nn_train_params import NNTrainParams{m.group(2)}",
        ],
    ),
    (
        re.compile(r"^(\s*)from common\.nn_model import NNTrainParams,\s*NNModel(\s*(?:#.*)?)$"),
        lambda m: [
            f"{m.group(1)}from nnx.nn.params.nn_train_params import NNTrainParams{m.group(2)}",
            f"{m.group(1)}from nnx.nn.nn_model import NNModel{m.group(2)}",
        ],
    ),
]

# 2026-05-27 — symbol consolidation that the 2026-05-16 migration missed.
#
# nnx merged the per-net {FeedFwdNN, GraphAtt, GraphConv, GraphSage}Params
# dataclasses into a single NNParams with `n_heads: Optional[int]` (a
# deliberate KISS over per-class typing). The original rewrite_imports
# handled module paths but not symbol names; this class closes that miss.
#
# Two kinds of rewrites:
#   * IMPORT lines that pull in one of the deprecated symbols — split or
#     reshape so the deprecated symbol is dropped (and NNParams is imported
#     if not already imported in the cell).
#   * CALL-SITE substitution: rename `OldNameParams(` → `NNParams(` in any
#     executable code line; comments and string literals are preserved.
#
DEPRECATED_PARAM_NAMES: list[str] = [
    "FeedFwdNNParams",
    "GraphAttNNParams",
    "GraphConvNNParams",
    "GraphSageNNParams",
]

NON_PYTHON_CELL_MAGICS = frozenset({
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


def _non_python_cell_magic(lines: list[str]) -> bool:
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            continue
        if not stripped.startswith("%%"):
            return False
        return stripped[2:].split(None, 1)[0].strip().lower() in NON_PYTHON_CELL_MAGICS
    return False


def _multiline_string_line_numbers(lines: list[str]) -> set[int]:
    protected: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO("".join(lines)).readline)
        for token in tokens:
            if token.type != tokenize.STRING:
                continue
            start_line, _ = token.start
            end_line, _ = token.end
            if end_line > start_line:
                protected.update(range(start_line, end_line + 1))
    except tokenize.TokenError:
        return protected
    return protected


def _symbol_without_inline_comment(symbol: str) -> str:
    return symbol.split("#", 1)[0].strip()


def _nnparams_replacement_for_symbol(symbol: str) -> str | None:
    symbol = _symbol_without_inline_comment(symbol)
    symbol = symbol.rstrip(")").strip()
    m = re.match(r"^([A-Za-z_]\w*)(?:\s+as\s+([A-Za-z_]\w*))?,?$", symbol)
    if not m or m.group(1) not in DEPRECATED_PARAM_NAMES:
        return None
    alias = m.group(2)
    return f"NNParams as {alias}" if alias else "NNParams"


def _closing_paren_line(line: str) -> str:
    indent = re.match(r"^(\s*)", line).group(1)
    suffix = "\n" if line.endswith("\n") else ""
    trailing = ""
    m = re.match(r"^\s*\)([^\n]*?)(\n?)$", line)
    if m and "#" in m.group(1):
        trailing = m.group(1).rstrip()
    return f"{indent}){trailing}{suffix}"


def _single_line_parenthesized_import(line: str) -> str:
    m = re.match(r"^(\s*from\s+[\w.]+\s+import\s+)\((.+)\)(\s*(?:#.*)?)(\n?)$", line)
    if not m:
        return line
    if not any(_nnparams_replacement_for_symbol(part) for part in m.group(2).split(",")):
        return line
    return f"{m.group(1)}{m.group(2).strip()}{m.group(3)}{m.group(4)}"


def _rewrite_common_import_aliases(line: str) -> str:
    m = re.match(r"^(\s*)import\s+(.+?)(\n?)$", line)
    if not m:
        return line
    indent, aliases_text, newline = m.group(1), m.group(2), m.group(3)
    code_text, comment_marker, comment_text = aliases_text.partition("#")
    aliases = [part.strip() for part in code_text.split(",")]
    if not aliases or any(not alias for alias in aliases):
        return line

    changed = False
    rewritten_aliases: list[str] = []
    for alias in aliases:
        rewritten = re.sub(r"^common(?=\.|\s+as\b|$)", "nnx", alias)
        changed = changed or rewritten != alias
        rewritten_aliases.append(rewritten)
    if not changed:
        return line

    trailing = ""
    if comment_marker:
        before_comment = code_text[len(code_text.rstrip()) :]
        trailing = f"{before_comment or ' '}#{comment_text}"
    return f"{indent}import {', '.join(rewritten_aliases)}{trailing}{newline}"


def _nnparams_import_requirements(symbols: str) -> set[str]:
    requirements = set()
    for part in (p.strip() for p in symbols.split(",") if p.strip()):
        part = _symbol_without_inline_comment(part)
        m = re.match(r"^NNParams(?:\s+as\s+([A-Za-z_]\w*))?$", part)
        if m:
            requirements.add(f"NNParams as {m.group(1)}" if m.group(1) else "NNParams")
    return requirements


def _rewrite_parenthesized_import_member_line(line: str) -> tuple[list[str], list[str], bool]:
    stripped_nl = line.rstrip("\n")
    closes_import = ")" in stripped_nl
    symbol_text = stripped_nl.split(")", 1)[0] if closes_import else stripped_nl
    indent = re.match(r"^(\s*)", line).group(1)
    content = symbol_text[len(indent) :]
    kept_lines: list[str] = []
    nnparams_imports: list[str] = []
    if not content.strip():
        return kept_lines, nnparams_imports, closes_import
    if content.lstrip().startswith("#"):
        suffix = "\n" if line.endswith("\n") else ""
        kept_lines.append(line if not closes_import else f"{indent}{content}{suffix}")
        return kept_lines, nnparams_imports, closes_import

    code_text = content.split("#", 1)[0]
    parts = [p.strip() for p in code_text.split(",") if p.strip()]
    has_deprecated = any(_nnparams_replacement_for_symbol(part) for part in parts)
    if not has_deprecated:
        if closes_import:
            kept = code_text.rstrip().rstrip(",").strip()
            if kept:
                kept_lines.append(f"{indent}{kept},\n")
        else:
            kept_lines.append(line)
        return kept_lines, nnparams_imports, closes_import

    for part in parts:
        if replacement := _nnparams_replacement_for_symbol(part):
            nnparams_imports.append(replacement)
        else:
            kept_lines.append(f"{indent}{part.rstrip(',')},\n")
    return kept_lines, nnparams_imports, closes_import


def _drop_deprecated_from_import(line: str) -> tuple[str, list[str], bool]:
    """Given a `from ... import A, B[, ...]` line, drop any deprecated
    Params names from the symbol list. Returns (new_line,
    nnparams_import_symbols, changed).

    If all symbols were deprecated, returns ("", True) — caller should
    omit the line entirely. NNParams import injection is handled at the
    cell level, not here.
    """
    m = re.match(r"^(\s*)from\s+([\w.]+)\s+import\s+(.+?)(\s*)$", line.rstrip("\n"))
    if not m:
        return line, [], False
    indent, module, symbols, trailing = m.group(1), m.group(2), m.group(3), m.group(4)
    parts = [p.strip() for p in symbols.split(",") if p.strip()]
    nnparams_imports = [replacement for p in parts if (replacement := _nnparams_replacement_for_symbol(p))]
    kept = [p for p in parts if _nnparams_replacement_for_symbol(p) is None]
    if kept == parts:
        return line, [], False
    if not kept:
        return "", nnparams_imports, True
    new_symbols = ", ".join(kept)
    had_nl = line.endswith("\n")
    rebuilt = f"{indent}from {module} import {new_symbols}{trailing}"
    return rebuilt + ("\n" if had_nl else ""), nnparams_imports, True


def _rewrite_param_name_references(line: str) -> tuple[str, bool]:
    """Rewrite executable unqualified deprecated Params references to `NNParams`.

    Tokenization keeps comments and string literals untouched, so prose-only
    cells do not gain executable imports. Qualified attributes and declaration
    names are left alone.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
    except tokenize.TokenError:
        close_parens = ")" * max(1, line.count("(") - line.count(")"))
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(line.rstrip("\n") + close_parens + "\n").readline))
        except tokenize.TokenError:
            return line, False

    line_offsets = [0]
    for physical_line in line.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(physical_line))

    def absolute_offset(position: tuple[int, int]) -> int:
        line_no, column = position
        if 1 <= line_no <= len(line_offsets):
            return line_offsets[line_no - 1] + column
        return column

    replacements: list[tuple[int, int]] = []
    for idx, token in enumerate(tokens):
        if token.type != tokenize.NAME or token.string not in DEPRECATED_PARAM_NAMES:
            continue
        prev_token = next(
            (
                candidate
                for candidate in reversed(tokens[:idx])
                if candidate.type not in {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}
            ),
            None,
        )
        if prev_token is not None and prev_token.string in {".", "class", "def"}:
            continue
        replacements.append((absolute_offset(token.start), absolute_offset(token.end)))

    if not replacements:
        return line, False

    new_line = line
    for start, end in reversed(replacements):
        new_line = f"{new_line[:start]}NNParams{new_line[end:]}"
    return new_line, True


def _rewrite_call_sites_across_continuations(lines: list[str]) -> tuple[list[str], bool]:
    rewritten: list[str] = []
    changed = False
    idx = 0
    while idx < len(lines):
        chunk = [lines[idx]]
        while chunk[-1].rstrip("\n").rstrip().endswith("\\") and idx + 1 < len(lines):
            idx += 1
            chunk.append(lines[idx])
        if len(chunk) > 1:
            new_chunk, chunk_changed = _rewrite_param_name_references("".join(chunk))
            if chunk_changed:
                changed = True
                rewritten.extend(new_chunk.splitlines(keepends=True))
            else:
                rewritten.extend(chunk)
        else:
            rewritten.extend(chunk)
        idx += 1
    return rewritten, changed


def _nnparams_import_insert_index(lines: list[str]) -> int:
    if lines and lines[0].lstrip().startswith("%%"):
        return 1
    return 0


def _is_nnparams_parenthesized_import(line: str) -> bool:
    return bool(re.match(r"^\s*from\s+nnx\.nn\.params\.nn_params\s+import\s+\(", line))


def _parenthesized_import_opener(line: str) -> tuple[str, str] | None:
    m = re.match(r"^(\s*from\s+[\w.]+\s+import\s+\()(.*?)(\n?)$", line)
    if not m:
        return None
    opener = f"{m.group(1)}{m.group(3)}"
    member_text = m.group(2)
    return opener, member_text


def rewrite_lines(source_lines: list[str]) -> list[str]:
    """Apply all rewrites to a list of source lines (each preserving its trailing \\n if present)."""
    if _non_python_cell_magic(source_lines):
        return source_lines
    out: list[str] = []
    needed_nnparams_imports: set[str] = set()
    existing_nnparams_imports: set[str] = set()
    protected_lines = _multiline_string_line_numbers(source_lines)
    in_parenthesized_import = False
    parenthesized_import_open = ""
    parenthesized_import_kept: list[str] = []
    parenthesized_import_raw: list[str] = []
    parenthesized_import_changed = False
    for line_no, line in enumerate(source_lines, 1):
        stripped_nl = line.rstrip("\n")
        had_nl = line.endswith("\n")
        if line_no in protected_lines:
            out.append(line)
            continue
        if in_parenthesized_import:
            parenthesized_import_raw.append(line)
            kept_lines, nnparams_imports, closes_import = _rewrite_parenthesized_import_member_line(line)
            needed_nnparams_imports.update(nnparams_imports)
            parenthesized_import_changed = parenthesized_import_changed or bool(nnparams_imports)
            parenthesized_import_kept.extend(kept_lines)
            if _is_nnparams_parenthesized_import(parenthesized_import_open):
                for kept_line in kept_lines:
                    if "NNParams" in kept_line:
                        existing_nnparams_imports.update(_nnparams_import_requirements(kept_line.strip().rstrip(",")))
            if closes_import:
                if not parenthesized_import_changed:
                    out.extend(parenthesized_import_raw)
                    if _is_nnparams_parenthesized_import(parenthesized_import_open):
                        for raw_line in parenthesized_import_raw:
                            if "NNParams" in raw_line:
                                existing_nnparams_imports.update(
                                    _nnparams_import_requirements(raw_line.split(")", 1)[0].strip().rstrip(","))
                                )
                elif parenthesized_import_kept:
                    out.append(parenthesized_import_open)
                    out.extend(parenthesized_import_kept)
                    out.append(_closing_paren_line(line))
                in_parenthesized_import = False
                parenthesized_import_open = ""
                parenthesized_import_kept = []
                parenthesized_import_raw = []
                parenthesized_import_changed = False
                continue
            continue
        # Try split patterns first
        split_applied = False
        for pattern, builder in SPLIT_PATTERNS:
            m = pattern.match(stripped_nl)
            if m:
                new_lines = builder(m)
                for nl in new_lines[:-1]:
                    out.append(nl.rstrip("\n") + "\n")
                last = new_lines[-1].rstrip("\n")
                out.append(last + ("\n" if had_nl else ""))
                split_applied = True
                break
        if split_applied:
            continue
        # Apply simple prefix mappings only to real import statements.
        new_line = line
        if new_line.lstrip().startswith(("from common", "import common.")):
            for old, new in SIMPLE_MAPPINGS:
                if old in new_line:
                    new_line = new_line.replace(old, new)
        new_line = _rewrite_common_import_aliases(new_line)
        new_line = _single_line_parenthesized_import(new_line)
        if new_line.lstrip().startswith("from ") and " import (" in new_line:
            in_parenthesized_import = True
            parenthesized_import_raw = [new_line]
            parenthesized_import_changed = False
            opener_parts = _parenthesized_import_opener(new_line)
            parenthesized_import_open = opener_parts[0] if opener_parts else new_line
            parenthesized_import_kept = []
            if opener_parts and opener_parts[1].strip():
                opener_indent = re.match(r"^(\s*)", new_line).group(1)
                member_line = f"{opener_indent}    {opener_parts[1]}"
                kept_lines, nnparams_imports, closes_import = _rewrite_parenthesized_import_member_line(member_line)
                needed_nnparams_imports.update(nnparams_imports)
                parenthesized_import_changed = parenthesized_import_changed or bool(nnparams_imports)
                parenthesized_import_kept.extend(kept_lines)
                if _is_nnparams_parenthesized_import(parenthesized_import_open):
                    for kept_line in kept_lines:
                        if "NNParams" in kept_line:
                            existing_nnparams_imports.update(_nnparams_import_requirements(kept_line.strip().rstrip(",")))
                if closes_import:
                    if not parenthesized_import_changed:
                        out.extend(parenthesized_import_raw)
                        if _is_nnparams_parenthesized_import(parenthesized_import_open):
                            for raw_line in parenthesized_import_raw:
                                if "NNParams" in raw_line:
                                    existing_nnparams_imports.update(
                                        _nnparams_import_requirements(raw_line.split(")", 1)[0].strip().rstrip(","))
                                    )
                    elif parenthesized_import_kept:
                        out.append(parenthesized_import_open)
                        out.extend(parenthesized_import_kept)
                        out.append(_closing_paren_line(new_line))
                    in_parenthesized_import = False
                    parenthesized_import_open = ""
                    parenthesized_import_kept = []
                    parenthesized_import_raw = []
                    parenthesized_import_changed = False
            continue
        # 2026-05-27: drop deprecated per-net Params from import lines
        if new_line.lstrip().startswith("from "):
            rewritten, nnparams_imports, ch = _drop_deprecated_from_import(new_line)
            if ch:
                # Dropping a deprecated symbol means the cell now needs NNParams.
                needed_nnparams_imports.update(nnparams_imports)
                # If the whole line vanished (all symbols deprecated), skip emitting it.
                if rewritten == "":
                    continue
                new_line = rewritten
        # 2026-05-27: rewrite executable uses of deprecated per-net Params.
        new_line, param_reference_changed = _rewrite_param_name_references(new_line)
        if param_reference_changed:
            needed_nnparams_imports.add("NNParams")
        # Track whether NNParams is being imported in this cell already.
        # Match only real `from ... import NNParams[,...]` lines, not
        # comments / strings that happen to contain the substring.
        nnparams_import = re.match(r"^\s*from\s+nnx\.nn\.params\.nn_params\s+import\s+(.+?)(\s*)$", new_line.rstrip("\n"))
        if nnparams_import:
            existing_nnparams_imports.update(_nnparams_import_requirements(nnparams_import.group(1)))
        out.append(new_line)
    out, continuation_call_site_changed = _rewrite_call_sites_across_continuations(out)
    if continuation_call_site_changed:
        needed_nnparams_imports.add("NNParams")
    # If any call site or rewrite needs NNParams but the cell never imports it,
    # inject one.
    missing_nnparams_imports = sorted(needed_nnparams_imports - existing_nnparams_imports)
    if missing_nnparams_imports:
        out.insert(
            _nnparams_import_insert_index(out),
            f"from nnx.nn.params.nn_params import {', '.join(missing_nnparams_imports)}\n",
        )
    return out


def process_notebook(path: Path) -> bool:
    """Process one notebook in-place. Returns True if any cells changed."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    any_changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, str):
            original_list = source.splitlines(keepends=True)
        else:
            original_list = "".join(str(chunk) for chunk in source).splitlines(keepends=True)
        new_list = rewrite_lines(original_list)
        if new_list != original_list:
            cell["source"] = new_list
            any_changed = True
    if any_changed:
        path.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return any_changed


def main(argv: list[str]) -> int:
    if not argv:
        print(USAGE, file=sys.stderr)
        return 2
    if argv == ["--help"] or argv == ["-h"]:
        print(USAGE)
        return 0
    changed_count = 0
    missing_count = 0
    for arg in argv:
        p = Path(arg)
        if not p.exists():
            print(f"SKIP (not found): {p}", file=sys.stderr)
            missing_count += 1
            continue
        if process_notebook(p):
            print(f"REWRITTEN: {p}")
            changed_count += 1
        else:
            print(f"unchanged: {p}")
    print(f"\n{changed_count} notebook(s) rewritten.")
    if missing_count:
        print(f"{missing_count} input path(s) were missing.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
