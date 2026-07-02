#!/usr/bin/env python3
"""Ensure each notebook has a `parameters`-tagged SMOKE_TEST hook.

Idempotent when a tagged cell already assigns `SMOKE_TEST`; otherwise augments
the first existing `parameters` cell, or inserts a new one before the first code cell.

Usage:
    python scripts/inject_smoke_test_cell.py NOTEBOOK [...]
"""
from __future__ import annotations
import ast
import json
import sys
from pathlib import Path

SMOKE_CELL_SOURCE = [
    "# Papermill parameters cell. Default values used when run interactively.\n",
    "# Set via: papermill -p SMOKE_TEST 1 in.ipynb out.ipynb\n",
    "# SMOKE_TEST: 1 = run a tiny smoke version of this notebook\n",
    "SMOKE_TEST = 0\n",
    "# SMOKE_TEST_EPOCHS: max epochs when SMOKE_TEST=1\n",
    "SMOKE_TEST_EPOCHS = 1\n",
    "# SMOKE_TEST_SUBSET: max samples when SMOKE_TEST=1\n",
    "SMOKE_TEST_SUBSET = 256\n",
]
# NOTE: comments must be on their own lines, NOT trailing the assignments.
# Papermill 2.7.0 switched to AST-based parameter-cell parsing; the older
# "SMOKE_TEST = 0  # 1 = run smoke" trailing-comment form trips its parser
# (the second `=` inside the comment confuses the assignment detector) and
# emits "Unable to parse line N" + "Passed unknown parameter: SMOKE_TEST"
# warnings on every smoke-tier-{b,c} run. 2026-06-15 weekly cron surfaced
# this. Comments-on-their-own-lines is the parser-friendly form.


def _cell_source_text(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _assignment_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets.append(node.target)
        elif isinstance(node, ast.AugAssign):
            targets.append(node.target)
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def parameters_cell_with_smoke_test(nb: dict) -> bool:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        if "parameters" in tags and "SMOKE_TEST" in _assignment_names(_cell_source_text(cell)):
            return True
    return False


def first_parameters_cell(nb: dict) -> dict | None:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        if "parameters" in tags:
            return cell
    return None


def _unique_cell_id(nb: dict, preferred: str = "smoke-params") -> str:
    existing = {
        cell.get("id")
        for cell in nb.get("cells", [])
        if isinstance(cell, dict) and cell.get("id")
    }
    if preferred not in existing:
        return preferred
    suffix = 2
    while f"{preferred}-{suffix}" in existing:
        suffix += 1
    return f"{preferred}-{suffix}"


def make_parameters_cell(cell_id: str = "smoke-params") -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {"tags": ["parameters"]},
        "outputs": [],
        "source": SMOKE_CELL_SOURCE,
    }


def find_insert_index(nb: dict) -> int:
    """Insert after leading markdown cells, before the first code cell."""
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            return i
    return 0


def process(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    if parameters_cell_with_smoke_test(nb):
        return "unchanged (SMOKE_TEST parameters cell present)"
    existing = first_parameters_cell(nb)
    if existing is not None:
        existing_source = existing.get("source", [])
        if isinstance(existing_source, str):
            existing_source = [existing_source]
        existing["source"] = SMOKE_CELL_SOURCE + ["\n"] + existing_source
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return "augmented existing parameters cell with SMOKE_TEST"
    idx = find_insert_index(nb)
    nb["cells"].insert(idx, make_parameters_cell(_unique_cell_id(nb)))
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return f"injected at cell index {idx}"


def main(argv):
    if not argv:
        print("Usage: inject_smoke_test_cell.py NOTEBOOK [...]", file=sys.stderr)
        return 2
    for arg in argv:
        p = Path(arg)
        if not p.exists():
            print(f"SKIP: {p}", file=sys.stderr)
            continue
        result = process(p)
        print(f"{p}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
