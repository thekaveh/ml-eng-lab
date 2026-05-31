#!/usr/bin/env python3
"""Rewrite `from common.X` → `from nnx.X` in Jupyter notebooks.

Handles both modern (`from common.nn.X`) and old-style flat (`from common.X`)
imports. Used once during the ml-lab revival to migrate notebooks to the new
NNx submodule package layout. Idempotent: re-running on already-rewritten
notebooks is a no-op.

Usage:
    python scripts/rewrite_imports.py path/to/notebook.ipynb [more.ipynb ...]
"""
from __future__ import annotations
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

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
        re.compile(r"^(\s*)from common\.nn_model import NNModel,\s*NNTrainParams(\s*)$"),
        lambda m: [
            f"{m.group(1)}from nnx.nn.nn_model import NNModel{m.group(2)}",
            f"{m.group(1)}from nnx.nn.params.nn_train_params import NNTrainParams{m.group(2)}",
        ],
    ),
    (
        re.compile(r"^(\s*)from common\.nn_model import NNTrainParams,\s*NNModel(\s*)$"),
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
#     code line (including commented-out lines, for consistency).
#
DEPRECATED_PARAM_NAMES: list[str] = [
    "FeedFwdNNParams",
    "GraphAttNNParams",
    "GraphConvNNParams",
    "GraphSageNNParams",
]


def _drop_deprecated_from_import(line: str) -> tuple[str, bool]:
    """Given a `from ... import A, B[, ...]` line, drop any deprecated
    Params names from the symbol list. Returns (new_line, changed).

    If all symbols were deprecated, returns ("", True) — caller should
    omit the line entirely. NNParams import injection is handled at the
    cell level, not here.
    """
    m = re.match(r"^(\s*)from\s+([\w.]+)\s+import\s+(.+?)(\s*)$", line.rstrip("\n"))
    if not m:
        return line, False
    indent, module, symbols, trailing = m.group(1), m.group(2), m.group(3), m.group(4)
    parts = [p.strip() for p in symbols.split(",") if p.strip()]
    kept = [p for p in parts if p not in DEPRECATED_PARAM_NAMES]
    if kept == parts:
        return line, False
    if not kept:
        return "", True
    new_symbols = ", ".join(kept)
    had_nl = line.endswith("\n")
    rebuilt = f"{indent}from {module} import {new_symbols}{trailing}"
    return rebuilt + ("\n" if had_nl else ""), True


def _rewrite_call_sites(line: str) -> tuple[str, bool]:
    """Rewrite `OldNameParams(` → `NNParams(` on this line.

    Uses word-boundary regex so identifiers that merely contain a
    deprecated name as a substring (e.g. `MyGraphAttNNParamsHelper`)
    are left alone.
    """
    new_line = line
    changed = False
    for old in DEPRECATED_PARAM_NAMES:
        if old in new_line:
            replaced = re.sub(rf"\b{re.escape(old)}\b", "NNParams", new_line)
            if replaced != new_line:
                changed = True
                new_line = replaced
    return new_line, changed


def rewrite_lines(source_lines: list[str]) -> list[str]:
    """Apply all rewrites to a list of source lines (each preserving its trailing \\n if present)."""
    out: list[str] = []
    cell_emits_nnparams_import = False  # tracks whether we already have an NNParams import in this cell after rewrites
    needs_nnparams = False
    for line in source_lines:
        stripped_nl = line.rstrip("\n")
        had_nl = line.endswith("\n")
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
        # Apply simple prefix mappings
        new_line = line
        for old, new in SIMPLE_MAPPINGS:
            if old in new_line:
                new_line = new_line.replace(old, new)
        # 2026-05-27: drop deprecated per-net Params from import lines
        if new_line.lstrip().startswith("from "):
            rewritten, ch = _drop_deprecated_from_import(new_line)
            if ch:
                # Dropping a deprecated symbol means the cell now needs NNParams.
                needs_nnparams = True
                # If the whole line vanished (all symbols deprecated), skip emitting it.
                if rewritten == "":
                    continue
                new_line = rewritten
        # 2026-05-27: rewrite call sites OldNameParams( → NNParams(
        new_line, call_site_changed = _rewrite_call_sites(new_line)
        if call_site_changed:
            needs_nnparams = True
        # Track whether NNParams is being imported in this cell already.
        # Match only real `from ... import NNParams[,...]` lines, not
        # comments / strings that happen to contain the substring.
        if re.match(r"^\s*from\s+nnx\.nn\.params\.nn_params\s+import\b", new_line) or \
           re.match(r"^\s*from\s+\S+\s+import\s+.*\bNNParams\b", new_line):
            cell_emits_nnparams_import = True
        out.append(new_line)
    # If any call site or rewrite needs NNParams but the cell never imports it,
    # inject one.
    if needs_nnparams and not cell_emits_nnparams_import:
        out.insert(0, "from nnx.nn.params.nn_params import NNParams\n")
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
            original_list = list(source)
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
        print("Usage: rewrite_imports.py NOTEBOOK [NOTEBOOK ...]", file=sys.stderr)
        return 2
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
