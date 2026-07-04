"""Tests for scripts/edit_notebook_markdown.py — Tier-C safe markdown editor."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "edit_notebook_markdown.py"
TEST_SUBPROCESS_TIMEOUT = 30


def _make_notebook(path: Path) -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell("# Old title\nintro"),
        nbformat.v4.new_code_cell(
            "print('hello')",
            outputs=[nbformat.v4.new_output("stream", text="hello\n")],
        ),
        nbformat.v4.new_markdown_cell("## Section A"),
    ]
    nbformat.write(nb, path)


def test_replace_markdown_preserves_code_cells(tmp_path):
    nb_path = tmp_path / "nb.ipynb"
    _make_notebook(nb_path)
    orig_code = nbformat.read(nb_path, as_version=4).cells[1]

    r = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--notebook", str(nb_path),
         "--cell", "0",
         "--text", "# New title\nnew intro"],
        capture_output=True, text=True, timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    assert r.returncode == 0, r.stderr

    doc = nbformat.read(nb_path, as_version=4)
    assert doc.cells[0].source == "# New title\nnew intro"
    assert doc.cells[0].cell_type == "markdown"
    new_code = doc.cells[1]
    assert new_code.source == orig_code.source
    assert new_code.outputs == orig_code.outputs
    assert new_code.execution_count == orig_code.execution_count


def test_refuses_to_edit_code_cell(tmp_path):
    nb_path = tmp_path / "nb.ipynb"
    _make_notebook(nb_path)
    r = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--notebook", str(nb_path),
         "--cell", "1",
         "--text", "should-be-rejected"],
        capture_output=True, text=True, timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    assert r.returncode != 0
    assert "code" in (r.stderr + r.stdout).lower()


def test_insert_markdown_cell_at_index(tmp_path):
    nb_path = tmp_path / "nb.ipynb"
    _make_notebook(nb_path)
    r = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--notebook", str(nb_path),
         "--insert-at", "1",
         "--text", "## New section"],
        capture_output=True, text=True, timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    assert r.returncode == 0, r.stderr
    doc = nbformat.read(nb_path, as_version=4)
    assert doc.cells[1].cell_type == "markdown"
    assert doc.cells[1].source == "## New section"
    assert doc.cells[2].cell_type == "code"
    assert doc.cells[2].source == "print('hello')"
