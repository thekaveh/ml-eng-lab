"""Tests for scripts/inject_smoke_test_cell.py — papermill `parameters` cell injector."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "inject_smoke_test_cell.py"


def _make_notebook(
    path: Path, *, with_params: bool = False, params_source: str = "SMOKE_TEST = 0\n"
) -> None:
    nb = nbformat.v4.new_notebook()
    cells: list = [
        nbformat.v4.new_markdown_cell("# Title\nintro"),
    ]
    if with_params:
        params_cell = nbformat.v4.new_code_cell(params_source)
        params_cell.metadata["tags"] = ["parameters"]
        cells.append(params_cell)
    cells.append(nbformat.v4.new_code_cell("import torch\nprint('hi')\n"))
    nb.cells = cells
    nbformat.write(nb, path)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True,
    )


def test_inject_adds_parameters_cell_when_missing(tmp_path):
    nb_path = tmp_path / "nb.ipynb"
    _make_notebook(nb_path, with_params=False)
    r = _run(str(nb_path))
    assert r.returncode == 0, r.stderr
    nb = nbformat.read(nb_path, as_version=4)
    params_cells = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(params_cells) == 1
    assert "SMOKE_TEST" in params_cells[0].source
    # Inserted before the first code cell (markdown stays on top).
    assert nb.cells[0].cell_type == "markdown"
    assert nb.cells[1].cell_type == "code"
    assert nb.cells[1].id == "smoke-params"
    assert "parameters" in nb.cells[1].metadata.get("tags", [])


def test_inject_is_idempotent_when_parameters_cell_exists(tmp_path):
    nb_path = tmp_path / "nb.ipynb"
    _make_notebook(nb_path, with_params=True)
    before = nb_path.read_bytes()
    r = _run(str(nb_path))
    assert r.returncode == 0, r.stderr
    assert "unchanged" in r.stdout
    assert nb_path.read_bytes() == before


def test_inject_augments_parameters_cell_missing_smoke_test(tmp_path):
    nb_path = tmp_path / "nb.ipynb"
    _make_notebook(nb_path, with_params=True, params_source="OTHER_PARAMETER = 1\n")
    r = _run(str(nb_path))
    assert r.returncode == 0, r.stderr
    assert "augmented existing parameters cell" in r.stdout

    nb = nbformat.read(nb_path, as_version=4)
    params_cells = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(params_cells) == 1
    assert "SMOKE_TEST = 0" in params_cells[0].source
    assert "OTHER_PARAMETER = 1" in params_cells[0].source


def test_main_returns_2_when_no_argv():
    r = _run()
    assert r.returncode == 2
    assert "Usage" in r.stderr


def test_smoke_cell_source_is_papermill_2_7_parser_friendly():
    """Regression guard: SMOKE_CELL_SOURCE must not put trailing comments on
    assignment lines.

    Papermill 2.7.0 switched to AST-based parameter-cell parsing; lines of the
    form `NAME = 0  # 1 = ...` (assignment with a trailing comment that itself
    contains `=`) trip its parser and emit "Unable to parse line N" + "Passed
    unknown parameter: SMOKE_TEST" warnings on every smoke-tier-{b,c} run.
    2026-06-15 weekly cron surfaced this. Comments must live on their own lines
    above the assignment.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("inject_smoke_test_cell", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for line in mod.SMOKE_CELL_SOURCE:
        s = line.rstrip("\n").rstrip()
        if not s or s.lstrip().startswith("#"):
            continue
        # Assignment line. Must NOT have an inline trailing comment.
        assert "#" not in s, (
            f"SMOKE_CELL_SOURCE assignment has trailing comment "
            f"(papermill 2.7+ can't parse): {line!r}"
        )


def test_missing_file_is_skipped_not_fatal(tmp_path):
    real = tmp_path / "real.ipynb"
    _make_notebook(real, with_params=False)
    missing = tmp_path / "missing.ipynb"
    r = _run(str(missing), str(real))
    assert r.returncode == 0, r.stderr
    assert "SKIP" in r.stderr
    # The real file still got injected.
    nb = nbformat.read(real, as_version=4)
    assert any("parameters" in c.metadata.get("tags", []) for c in nb.cells)
