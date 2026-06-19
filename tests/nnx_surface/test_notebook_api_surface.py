"""Static guards over committed notebooks — catch defects the papermill tiers miss.

Background
----------
The nnx 0.2.0 PyPI migration (2026-06-14) split the plotting helpers
(`multi_line_plot`, `scatter_plot`, `get_scatter_plot_vm`, ...) off
`nnx.utils.Utils` onto `nnx.vis_utils.VisUtils`. Seven node-classification
notebooks kept calling them as `Utils.multi_line_plot(...)`, which raises
``AttributeError`` at runtime. `verify_repo.py --check structure` passed clean
(it resolves *imports*, not attribute access), so the breakage only surfaced in
the weekly `smoke-tier-b` / `smoke-tier-c` cron — and stayed hidden between
merges because those jobs run on `schedule` only. See CHANGELOG 2026-06-19.

These tests close that gap with cheap, execution-free static scans that run in
CI's fast `make test-nnx-surface` job on *every* PR:

1. ``test_no_visutils_method_called_via_Utils`` — the migration guard. The
   forbidden-method set is derived live from the real nnx surface, so it tracks
   future Utils/VisUtils reshuffles automatically.
2. ``test_no_committed_error_outputs`` — no committed error/traceback outputs
   (e.g. a stray ``KeyboardInterrupt`` from a manually-aborted run).
3. ``test_no_transient_worktree_paths`` — no transient ``.claude/worktrees``
   dev paths leaked into committed cell outputs.

Each scan is paired with a synthetic-notebook unit test proving the checker
actually fires, so a green suite means "checked", not "vacuously passed".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nnx.utils import Utils
from nnx.vis_utils import VisUtils

# Repo root resolved from this file (the autouse conftest fixture chdirs tests
# into a tmp_path, so cwd is NOT the repo root — never rely on it here).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Bare `Utils.<attr>` access, excluding the `VisUtils.` suffix-match.
_UTILS_ATTR_RE = re.compile(r"(?<![A-Za-z0-9_])Utils\.([A-Za-z_]\w*)")
# Transient per-worktree path that must never be committed in an output.
_TRANSIENT_PATH_RE = re.compile(r"\.claude/worktrees/")


def _public_attrs(cls: object) -> set[str]:
    return {n for n in dir(cls) if not n.startswith("_")}


def _visutils_only_methods() -> set[str]:
    """Methods that live on VisUtils but NOT on Utils — illegal as ``Utils.<m>``."""
    return _public_attrs(VisUtils) - _public_attrs(Utils)


def _active_notebooks() -> list[Path]:
    """Every committed notebook except the archived codexglue set + checkpoints."""
    return sorted(
        p
        for p in REPO_ROOT.rglob("*.ipynb")
        if "archive/" not in p.relative_to(REPO_ROOT).as_posix()
        and ".ipynb_checkpoints" not in p.parts
    )


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]


def _live_lines(cell: dict) -> list[str]:
    """Source lines that are not pure comments (commented-out historical code is
    deliberately preserved verbatim in some Tier-C cells and must not be flagged)."""
    return [ln for ln in cell.get("source", []) if not ln.lstrip().startswith("#")]


def _output_text(cell: dict) -> str:
    chunks: list[str] = []
    for o in cell.get("outputs", []):
        txt = o.get("text", "")
        chunks.append("".join(txt) if isinstance(txt, list) else str(txt))
        tp = o.get("data", {}).get("text/plain", "")
        chunks.append("".join(tp) if isinstance(tp, list) else str(tp))
    return "\n".join(chunks)


# --- scan checkers (also exercised directly by the synthetic unit tests) -----

def find_misplaced_utils_attrs(nb: dict, forbidden: set[str]) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        for line in _live_lines(cell):
            for m in _UTILS_ATTR_RE.finditer(line):
                if m.group(1) in forbidden:
                    out.append(f"code_cell[{idx}]: Utils.{m.group(1)} (moved to VisUtils)")
    return out


def find_error_outputs(nb: dict) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        for o in cell.get("outputs", []):
            if o.get("output_type") == "error":
                out.append(f"code_cell[{idx}]: committed {o.get('ename', 'error')} output")
    return out


def find_transient_paths(nb: dict) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        if _TRANSIENT_PATH_RE.search(_output_text(cell)):
            out.append(f"code_cell[{idx}]: '.claude/worktrees' path leaked into output")
    return out


# --- real-notebook scans (parametrized per notebook) -------------------------

_NOTEBOOKS = _active_notebooks()
_IDS = [p.relative_to(REPO_ROOT).as_posix() for p in _NOTEBOOKS]


def test_active_notebooks_discovered():
    """Guard against the glob silently matching nothing (which would make every
    parametrized scan vacuously pass)."""
    assert len(_NOTEBOOKS) >= 25, f"expected the full active notebook set, found {len(_NOTEBOOKS)}"


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_no_visutils_method_called_via_Utils(nb_path: Path):
    forbidden = _visutils_only_methods()
    assert forbidden, "expected VisUtils to expose methods absent from Utils"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_misplaced_utils_attrs(nb, forbidden)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} calls VisUtils-only method(s) via Utils:\n  "
        + "\n  ".join(violations)
    )


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_no_committed_error_outputs(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_error_outputs(nb)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} has committed error output(s):\n  "
        + "\n  ".join(violations)
    )


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_no_transient_worktree_paths(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_transient_paths(nb)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} leaks transient worktree path(s):\n  "
        + "\n  ".join(violations)
    )


# --- self-validation: prove each checker fires on a known-bad notebook -------

def _synthetic_nb(code_cell: dict) -> dict:
    return {"cells": [code_cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}


def test_migration_guard_catches_bad_call():
    forbidden = _visutils_only_methods()
    assert "multi_line_plot" in forbidden, "fixture assumes multi_line_plot is VisUtils-only"
    bad = _synthetic_nb({"cell_type": "code", "source": ["Utils.multi_line_plot(x=[1])\n"], "outputs": []})
    assert find_misplaced_utils_attrs(bad, forbidden)


def test_migration_guard_allows_correct_call_and_real_utils_methods():
    forbidden = _visutils_only_methods()
    good = _synthetic_nb({
        "cell_type": "code",
        "source": ["VisUtils.multi_line_plot(x=[1])\n", "Utils.print_table(rows)\n"],
        "outputs": [],
    })
    assert not find_misplaced_utils_attrs(good, forbidden)


def test_migration_guard_ignores_commented_out_reference():
    forbidden = _visutils_only_methods()
    commented = _synthetic_nb({
        "cell_type": "code",
        "source": ["# original: Utils.scatter_plot(...)  preserved for reference\n"],
        "outputs": [],
    })
    assert not find_misplaced_utils_attrs(commented, forbidden)


def test_error_output_guard_catches_traceback():
    bad = _synthetic_nb({
        "cell_type": "code", "source": ["train()\n"],
        "outputs": [{"output_type": "error", "ename": "KeyboardInterrupt", "evalue": "", "traceback": []}],
    })
    assert find_error_outputs(bad)


def test_transient_path_guard_catches_leak():
    bad = _synthetic_nb({
        "cell_type": "code", "source": ["run.save()\n"],
        "outputs": [{"output_type": "stream", "name": "stdout",
                     "text": ["Run saved to /Users/x/.claude/worktrees/wt/runs/abc\n"]}],
    })
    assert find_transient_paths(bad)
