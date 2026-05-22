#!/usr/bin/env python3
"""Rewrite `from common.X` → `from nnx.X` in Jupyter notebooks.

Handles both modern (`from common.nn.X`) and old-style flat (`from common.X`)
imports. Used once during the ml-repo revival to migrate notebooks to the new
NNx submodule package layout. Idempotent: re-running on already-rewritten
notebooks is a no-op.

Usage:
    python scripts/rewrite_imports.py path/to/notebook.ipynb [more.ipynb ...]
"""
from __future__ import annotations
import json
import re
import sys
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
SPLIT_PATTERNS: list[tuple[re.Pattern[str], callable]] = [
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


def rewrite_lines(source_lines: list[str]) -> list[str]:
    """Apply all rewrites to a list of source lines (each preserving its trailing \\n if present)."""
    out: list[str] = []
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
        out.append(new_line)
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
