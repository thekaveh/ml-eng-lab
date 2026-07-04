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
    return f"{indent}){suffix}"


def _single_line_parenthesized_import(line: str) -> str:
    m = re.match(r"^(\s*from\s+[\w.]+\s+import\s+)\((.+)\)(\s*(?:#.*)?)(\n?)$", line)
    if not m:
        return line
    return f"{m.group(1)}{m.group(2).strip()}{m.group(3)}{m.group(4)}"


def _imported_symbol_bindings(symbols: str) -> set[str]:
    bindings = set()
    for part in (p.strip() for p in symbols.split(",") if p.strip()):
        part = _symbol_without_inline_comment(part)
        m = re.match(r"^([A-Za-z_]\w*)(?:\s+as\s+([A-Za-z_]\w*))?$", part)
        if m:
            bindings.add(m.group(2) or m.group(1))
    return bindings


def _rewrite_parenthesized_import_member_line(line: str) -> tuple[list[str], list[str], bool]:
    stripped_nl = line.rstrip("\n")
    closes_import = ")" in stripped_nl
    symbol_text = stripped_nl.split(")", 1)[0] if closes_import else stripped_nl
    indent = re.match(r"^(\s*)", line).group(1)
    kept_lines: list[str] = []
    nnparams_imports: list[str] = []
    for part in (p.strip() for p in symbol_text.split(",") if p.strip()):
        if part.startswith("#"):
            continue
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


def _rewrite_call_sites(line: str) -> tuple[str, bool]:
    """Rewrite `OldNameParams(` → `NNParams(` on this line.

    Tokenization keeps comments and string literals untouched, so prose-only
    cells do not gain executable imports.
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
        next_token = next(
            (
                candidate
                for candidate in tokens[idx + 1:]
                if candidate.type not in {tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER}
            ),
            None,
        )
        if next_token is not None and next_token.string == "(":
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
            new_chunk, chunk_changed = _rewrite_call_sites("".join(chunk))
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
    out: list[str] = []
    needed_nnparams_imports: set[str] = set()
    existing_nnparams_imports: set[str] = set()
    in_parenthesized_import = False
    parenthesized_import_open = ""
    parenthesized_import_kept: list[str] = []
    for line in source_lines:
        stripped_nl = line.rstrip("\n")
        had_nl = line.endswith("\n")
        if in_parenthesized_import:
            kept_lines, nnparams_imports, closes_import = _rewrite_parenthesized_import_member_line(line)
            needed_nnparams_imports.update(nnparams_imports)
            parenthesized_import_kept.extend(kept_lines)
            if _is_nnparams_parenthesized_import(parenthesized_import_open):
                for kept_line in kept_lines:
                    if "NNParams" in kept_line:
                        existing_nnparams_imports.update(_imported_symbol_bindings(kept_line.strip().rstrip(",")))
            if closes_import:
                if parenthesized_import_kept:
                    out.append(parenthesized_import_open)
                    out.extend(parenthesized_import_kept)
                    out.append(_closing_paren_line(line))
                in_parenthesized_import = False
                parenthesized_import_open = ""
                parenthesized_import_kept = []
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
        new_line = _single_line_parenthesized_import(new_line)
        if new_line.lstrip().startswith("from ") and " import (" in new_line:
            in_parenthesized_import = True
            opener_parts = _parenthesized_import_opener(new_line)
            parenthesized_import_open = opener_parts[0] if opener_parts else new_line
            parenthesized_import_kept = []
            if opener_parts and opener_parts[1].strip():
                opener_indent = re.match(r"^(\s*)", new_line).group(1)
                member_line = f"{opener_indent}    {opener_parts[1]}"
                kept_lines, nnparams_imports, closes_import = _rewrite_parenthesized_import_member_line(member_line)
                needed_nnparams_imports.update(nnparams_imports)
                parenthesized_import_kept.extend(kept_lines)
                if _is_nnparams_parenthesized_import(parenthesized_import_open):
                    for kept_line in kept_lines:
                        if "NNParams" in kept_line:
                            existing_nnparams_imports.update(_imported_symbol_bindings(kept_line.strip().rstrip(",")))
                if closes_import:
                    if parenthesized_import_kept:
                        out.append(parenthesized_import_open)
                        out.extend(parenthesized_import_kept)
                        out.append(_closing_paren_line(new_line))
                    in_parenthesized_import = False
                    parenthesized_import_open = ""
                    parenthesized_import_kept = []
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
        # 2026-05-27: rewrite call sites OldNameParams( → NNParams(
        new_line, call_site_changed = _rewrite_call_sites(new_line)
        if call_site_changed:
            needed_nnparams_imports.add("NNParams")
        # Track whether NNParams is being imported in this cell already.
        # Match only real `from ... import NNParams[,...]` lines, not
        # comments / strings that happen to contain the substring.
        nnparams_import = re.match(r"^\s*from\s+nnx\.nn\.params\.nn_params\s+import\s+(.+?)(\s*)$", new_line.rstrip("\n"))
        if nnparams_import:
            existing_nnparams_imports.update(_imported_symbol_bindings(nnparams_import.group(1)))
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
    for arg in argv:
        p = Path(arg)
        if not p.exists():
            print(f"SKIP (not found): {p}", file=sys.stderr)
            continue
        if process_notebook(p):
            print(f"REWRITTEN: {p}")
            changed_count += 1
        else:
            print(f"unchanged: {p}")
    print(f"\n{changed_count} notebook(s) rewritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
