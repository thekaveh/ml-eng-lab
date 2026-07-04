"""Unit tests for scripts/rewrite_imports.py.

Covers (a) the original module-path rewrites (sanity regression), and
(b) the new symbol-consolidation rewrites for {FeedFwdNN, GraphAtt,
GraphConv, GraphSage}Params → NNParams added 2026-05-27.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TEST_SUBPROCESS_TIMEOUT = 30


def _make_notebook(tmp_path: Path, name: str, cells: list[dict]) -> Path:
    """Write a minimal nbformat-4 notebook to disk and return its path."""
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    p = tmp_path / name
    p.write_text(json.dumps(nb, indent=1) + "\n")
    return p


def _code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def _run(path: Path) -> None:
    """Run rewrite_imports.py on the given notebook path."""
    repo_root = Path(__file__).resolve().parent.parent
    subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "rewrite_imports.py"), str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )


def _cell_source(path: Path, idx: int) -> str:
    nb = json.loads(path.read_text())
    return "".join(nb["cells"][idx]["source"])


# ----- Existing rule coverage (sanity regression) ---------------------------

def test_help_exits_zero_and_prints_usage():
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "rewrite_imports.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "NOTEBOOK" in result.stdout

def test_module_path_rewrite_common_to_nnx(tmp_path):
    p = _make_notebook(tmp_path, "old.ipynb", [
        _code_cell("from common.nn_model import NNModel\n"),
    ])
    _run(p)
    assert "from nnx.nn.nn_model import NNModel" in _cell_source(p, 0)


def test_module_path_rewrite_common_alias_inside_multi_import(tmp_path):
    src = "import os, common.utils as common_utils  # migrate this alias\n"
    p = _make_notebook(tmp_path, "multi_import_common_alias.ipynb", [_code_cell(src)])
    _run(p)
    assert _cell_source(p, 0) == "import os, nnx.utils as common_utils  # migrate this alias\n"


def test_idempotent_on_already_rewritten(tmp_path):
    src = "from nnx.nn.nn_model import NNModel\n"
    p = _make_notebook(tmp_path, "ok.ipynb", [_code_cell(src)])
    _run(p)
    assert _cell_source(p, 0) == src


def test_module_path_rewrite_ignores_comments_and_strings(tmp_path):
    src = (
        "# from common.nn_model import NNModel\n"
        "example = 'from common.nn_model import NNModel'\n"
        "from common.nn_model import NNModel\n"
    )
    p = _make_notebook(tmp_path, "comments_strings.ipynb", [_code_cell(src)])

    _run(p)

    rewritten = _cell_source(p, 0)
    assert "# from common.nn_model import NNModel" in rewritten
    assert "example = 'from common.nn_model import NNModel'" in rewritten
    assert "from nnx.nn.nn_model import NNModel" in rewritten


def test_module_path_rewrite_preserves_multiline_strings(tmp_path):
    src = (
        'example = """\n'
        "from common.nn_model import NNModel\n"
        "import os, common.utils as common_utils\n"
        "GraphAttNNParams(n_heads=2, input_dim=4, output_dim=2)\n"
        '"""\n'
    )
    p = _make_notebook(tmp_path, "multiline_string_common_imports.ipynb", [_code_cell(src)])
    _run(p)
    assert _cell_source(p, 0) == src


def test_module_path_rewrite_skips_non_python_cell_magic(tmp_path):
    src = "%%bash\nimport os, common.utils as common_utils\n"
    p = _make_notebook(tmp_path, "bash_cell_common_import.ipynb", [_code_cell(src)])
    _run(p)
    assert _cell_source(p, 0) == src


# ----- NEW: symbol-consolidation rules --------------------------------------

def test_graph_att_params_import_consolidates_to_nnparams(tmp_path):
    """`from nnx.nn.net.graph_att_nn import GraphAttNN, GraphAttNNParams`
    → `from nnx.nn.net.graph_att_nn import GraphAttNN`
    + `from nnx.nn.params.nn_params import NNParams`
    (the NNParams import is omitted if already present in the same cell)."""
    p = _make_notebook(tmp_path, "gat.ipynb", [
        _code_cell("from nnx.nn.net.graph_att_nn import GraphAttNN, GraphAttNNParams\n"),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "GraphAttNNParams" not in src, f"GraphAttNNParams should be gone; got: {src!r}"
    assert "from nnx.nn.net.graph_att_nn import GraphAttNN" in src
    assert "NNParams" in src  # either via import or remaining context


def test_aliased_graph_att_params_import_consolidates_to_aliased_nnparams(tmp_path):
    p = _make_notebook(tmp_path, "gat_alias.ipynb", [
        _code_cell("from nnx.nn.net.graph_att_nn import GraphAttNNParams as GATParams\n"),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "from nnx.nn.net.graph_att_nn import NNParams" not in src
    assert "from nnx.nn.params.nn_params import NNParams as GATParams" in src
    assert "GraphAttNNParams" not in src


def test_existing_aliased_nnparams_import_suppresses_identical_injection(tmp_path):
    src = (
        "from nnx.nn.params.nn_params import NNParams as GATParams\n"
        "from nnx.nn.net.graph_att_nn import GraphAttNNParams as GATParams\n"
        "params = GATParams(n_heads=2, input_dim=4, output_dim=2)\n"
    )
    p = _make_notebook(tmp_path, "existing_aliased_nnparams.ipynb", [_code_cell(src)])
    _run(p)
    rewritten = _cell_source(p, 0)
    assert rewritten.count("from nnx.nn.params.nn_params import NNParams as GATParams") == 1
    assert "GraphAttNNParams" not in rewritten


def test_parenthesized_graph_att_params_import_consolidates_to_nnparams(tmp_path):
    p = _make_notebook(tmp_path, "gat_parenthesized.ipynb", [
        _code_cell(
            "from nnx.nn.net.graph_att_nn import (\n"
            "    GraphAttNN,\n"
            "    GraphAttNNParams,\n"
            ")\n"
        ),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "from nnx.nn.net.graph_att_nn import NNParams" not in src
    assert "GraphAttNNParams" not in src
    assert "from nnx.nn.net.graph_att_nn import (\n    GraphAttNN,\n)" in src
    assert "from nnx.nn.params.nn_params import NNParams" in src


def test_parenthesized_graph_att_params_import_with_inline_comment(tmp_path):
    p = _make_notebook(tmp_path, "gat_parenthesized_comment.ipynb", [
        _code_cell(
            "from nnx.nn.net.graph_att_nn import (\n"
            "    GraphAttNN,\n"
            "    GraphAttNNParams,  # old params alias\n"
            ")\n"
        ),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "GraphAttNNParams" not in src
    assert "old params alias" not in src
    assert "from nnx.nn.net.graph_att_nn import (\n    GraphAttNN,\n)" in src
    assert "from nnx.nn.params.nn_params import NNParams" in src
    compile(src, "<rewritten-cell>", "exec")


def test_parenthesized_graph_att_params_import_with_closing_paren_same_line(tmp_path):
    p = _make_notebook(tmp_path, "gat_parenthesized_closing_same_line.ipynb", [
        _code_cell(
            "from nnx.nn.net.graph_att_nn import (\n"
            "    GraphAttNN,\n"
            "    GraphAttNNParams)\n"
        ),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "GraphAttNNParams" not in src
    assert "from nnx.nn.net.graph_att_nn import (\n    GraphAttNN,\n    )" in src
    assert "from nnx.nn.params.nn_params import NNParams" in src
    compile(src, "<rewritten-cell>", "exec")


def test_parenthesized_mixed_closing_line_drops_only_deprecated_params(tmp_path):
    p = _make_notebook(tmp_path, "gat_parenthesized_mixed_closing.ipynb", [
        _code_cell(
            "from nnx.nn.net.graph_att_nn import (\n"
            "    GraphAttNN, GraphAttNNParams)\n"
        ),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "GraphAttNNParams" not in src
    assert "from nnx.nn.net.graph_att_nn import (\n    GraphAttNN,\n    )" in src
    assert "from nnx.nn.params.nn_params import NNParams" in src
    compile(src, "<rewritten-cell>", "exec")


def test_parenthesized_mixed_closing_line_keeps_symbol_after_params(tmp_path):
    p = _make_notebook(tmp_path, "gat_parenthesized_mixed_closing_params_first.ipynb", [
        _code_cell(
            "from nnx.nn.net.graph_att_nn import (\n"
            "    GraphAttNNParams, GraphAttNN)\n"
        ),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "GraphAttNNParams" not in src
    assert "from nnx.nn.net.graph_att_nn import (\n    GraphAttNN,\n    )" in src
    assert "from nnx.nn.params.nn_params import NNParams" in src
    compile(src, "<rewritten-cell>", "exec")


def test_same_line_closing_parenthesized_nnparams_import_prevents_duplicate_injection(tmp_path):
    p = _make_notebook(tmp_path, "parenthesized_nnparams_and_call.ipynb", [
        _code_cell(
            "from nnx.nn.params.nn_params import (\n"
            "    NNParams)\n"
            "params = GraphAttNNParams(n_heads=4, input_dim=10, output_dim=2)\n"
        ),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert src.count("from nnx.nn.params.nn_params import") == 1
    assert "params = NNParams(n_heads=4, input_dim=10, output_dim=2)" in src
    assert "GraphAttNNParams" not in src
    compile(src, "<rewritten-cell>", "exec")


def test_parenthesized_opener_line_members_are_rewritten(tmp_path):
    p = _make_notebook(tmp_path, "gat_parenthesized_opener_members.ipynb", [
        _code_cell(
            "from nnx.nn.net.graph_att_nn import (GraphAttNNParams,\n"
            "    GraphAttNN,\n"
            ")\n"
        ),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "GraphAttNNParams" not in src
    assert "from nnx.nn.net.graph_att_nn import (\n    GraphAttNN,\n)" in src
    assert "from nnx.nn.params.nn_params import NNParams" in src
    compile(src, "<rewritten-cell>", "exec")


def test_single_line_parenthesized_params_import_is_preserved_and_rewritten(tmp_path):
    p = _make_notebook(tmp_path, "gat_parenthesized_single_line.ipynb", [
        _code_cell("from nnx.nn.net.graph_att_nn import (GraphAttNN, GraphAttNNParams)\n"),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "GraphAttNNParams" not in src
    assert "from nnx.nn.net.graph_att_nn import GraphAttNN" in src
    assert "from nnx.nn.params.nn_params import NNParams" in src
    compile(src, "<rewritten-cell>", "exec")


def test_single_line_parenthesized_params_only_import_drops_empty_import(tmp_path):
    p = _make_notebook(tmp_path, "gat_parenthesized_single_line_only_params.ipynb", [
        _code_cell("from nnx.nn.net.graph_att_nn import (GraphAttNNParams)\n"),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "from nnx.nn.net.graph_att_nn" not in src
    assert "GraphAttNNParams" not in src
    assert "from nnx.nn.params.nn_params import NNParams" in src
    compile(src, "<rewritten-cell>", "exec")


def test_multiline_source_chunk_is_split_before_rewriting(tmp_path):
    nb = {
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from nnx.nn.net.graph_att_nn import (\n    GraphAttNN,\n    GraphAttNNParams,\n)\nprint(GraphAttNN)\n"
            ],
        }],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    p = tmp_path / "multiline_chunk.ipynb"
    p.write_text(json.dumps(nb, indent=1) + "\n")

    _run(p)

    src = _cell_source(p, 0)
    assert "GraphAttNNParams" not in src
    assert "GraphAttNN" in src
    assert "print(GraphAttNN)" in src
    assert "from nnx.nn.params.nn_params import NNParams" in src
    compile(src, "<rewritten-cell>", "exec")


def test_source_chunks_are_joined_before_rewriting_logical_lines(tmp_path):
    nb = {
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from common.",
                "nn_model import NNModel, NNTrainParams\n",
                "print(NNModel, NNTrainParams)\n",
            ],
        }],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    p = tmp_path / "split_logical_line_chunk.ipynb"
    p.write_text(json.dumps(nb, indent=1) + "\n")

    _run(p)

    src = _cell_source(p, 0)
    assert "from common.nn_model" not in src
    assert "from nnx.nn.nn_model import NNModel" in src
    assert "from nnx.nn.params.nn_train_params import NNTrainParams" in src
    assert "print(NNModel, NNTrainParams)" in src
    compile(src, "<rewritten-cell>", "exec")


def test_parenthesized_params_only_import_drops_empty_block(tmp_path):
    p = _make_notebook(tmp_path, "gat_parenthesized_only_params.ipynb", [
        _code_cell(
            "from nnx.nn.net.graph_att_nn import (\n"
            "    GraphAttNNParams,\n"
            ")\n"
        ),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "from nnx.nn.net.graph_att_nn import (" not in src
    assert "GraphAttNNParams" not in src
    assert "from nnx.nn.params.nn_params import NNParams" in src
    compile(src, "<rewritten-cell>", "exec")


def test_aliased_nnparams_import_does_not_suppress_bare_nnparams_for_call_site(tmp_path):
    p = _make_notebook(tmp_path, "gat_alias_and_call.ipynb", [
        _code_cell(
            "from nnx.nn.params.nn_params import NNParams as GATParams\n"
            "params = GraphAttNNParams(n_heads=4, input_dim=10, output_dim=2)\n"
        ),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "from nnx.nn.params.nn_params import NNParams\n" in src
    assert "from nnx.nn.params.nn_params import NNParams as GATParams\n" in src
    assert "params = NNParams(n_heads=4, input_dim=10, output_dim=2)" in src
    assert "GraphAttNNParams" not in src


def test_graph_att_params_call_site_renamed_to_nnparams(tmp_path):
    """`GraphAttNNParams(n_heads=..., ...)` call sites rewrite to `NNParams(n_heads=..., ...)`."""
    p = _make_notebook(tmp_path, "gat_call.ipynb", [
        _code_cell(
            "from nnx.nn.net.graph_att_nn import GraphAttNN, GraphAttNNParams\n"
            "net = GraphAttNN(params=GraphAttNNParams(\n"
            "    n_heads=4, dropout_prob=0.25, hidden_dims=[128], input_dim=10, output_dim=2,\n"
            "))\n"
        ),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "GraphAttNNParams(" not in src
    assert "NNParams(" in src
    assert "n_heads=4" in src  # call-site args preserved verbatim


def test_non_call_param_references_are_rewritten_with_import(tmp_path):
    p = _make_notebook(tmp_path, "gat_non_call_refs.ipynb", [
        _code_cell(
            "from nnx.nn.net.graph_att_nn import GraphAttNNParams\n"
            "ParamsClass = GraphAttNNParams\n"
            "is_old = isinstance(params, GraphAttNNParams)\n"
        ),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "GraphAttNNParams" not in src
    assert "from nnx.nn.params.nn_params import NNParams" in src
    assert "ParamsClass = NNParams" in src
    assert "is_old = isinstance(params, NNParams)" in src
    compile(src, "<rewritten-cell>", "exec")


def test_params_call_site_with_backslash_continuation_renamed_to_nnparams(tmp_path):
    p = _make_notebook(tmp_path, "gat_call_backslash.ipynb", [
        _code_cell(
            "from nnx.nn.net.graph_att_nn import GraphAttNNParams\n"
            "params = GraphAttNNParams \\\n"
            "    (n_heads=4, input_dim=10, output_dim=2)\n"
        ),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "GraphAttNNParams" not in src
    assert "params = NNParams \\\n    (n_heads=4, input_dim=10, output_dim=2)" in src
    compile(src, "<rewritten-cell>", "exec")


def test_qualified_and_declaration_param_names_are_not_rewritten(tmp_path):
    p = _make_notebook(tmp_path, "qualified_and_decl_names.ipynb", [
        _code_cell(
            "import legacy\n"
            "qualified = legacy.GraphAttNNParams(n_heads=4)\n"
            "class GraphAttNNParams(BaseParams):\n"
            "    pass\n"
            "def GraphConvNNParams(value):\n"
            "    return value\n"
        ),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "legacy.GraphAttNNParams(n_heads=4)" in src
    assert "class GraphAttNNParams(BaseParams):" in src
    assert "def GraphConvNNParams(value):" in src
    assert "from nnx.nn.params.nn_params import NNParams" not in src
    compile(src, "<rewritten-cell>", "exec")


def test_nnparams_import_inserted_after_cell_magic(tmp_path):
    p = _make_notebook(tmp_path, "gat_call_cell_magic.ipynb", [
        _code_cell(
            "%%time\n"
            "params = GraphAttNNParams(n_heads=4, input_dim=10, output_dim=2)\n"
        ),
    ])
    _run(p)
    src = _cell_source(p, 0)
    lines = src.splitlines(keepends=True)
    assert lines[0] == "%%time\n"
    assert lines[1] == "from nnx.nn.params.nn_params import NNParams\n"
    assert "params = NNParams(n_heads=4, input_dim=10, output_dim=2)" in src
    assert "GraphAttNNParams" not in src


def test_other_per_net_params_renamed_defensively(tmp_path):
    """Defensive rules: even if no notebook currently uses these, the rewriter
    handles them so future audits don't repeat the miss."""
    for old_name in ("FeedFwdNNParams", "GraphConvNNParams", "GraphSageNNParams"):
        p = _make_notebook(tmp_path, f"{old_name}.ipynb", [
            _code_cell(f"x = {old_name}(input_dim=4, output_dim=2, dropout_prob=0.1)\n"),
        ])
        _run(p)
        src = _cell_source(p, 0)
        assert old_name not in src, f"{old_name} should be rewritten; got: {src!r}"
        assert "NNParams(" in src


def test_commented_out_and_string_call_sites_are_preserved(tmp_path):
    """Historical prose must not gain executable NNParams imports."""
    src = (
        "# x = GraphAttNNParams(n_heads=2, input_dim=4, output_dim=2)\n"
        "example = 'GraphAttNNParams(n_heads=2, input_dim=4, output_dim=2)'\n"
    )
    p = _make_notebook(tmp_path, "commented.ipynb", [_code_cell(src)])
    _run(p)
    assert _cell_source(p, 0) == src


def test_parenthesized_non_deprecated_import_preserves_comments(tmp_path):
    src = (
        "from some.module import (\n"
        "    # needed for registry side effect\n"
        "    UsefulSymbol,  # keep this inline note\n"
        ")\n"
    )
    p = _make_notebook(tmp_path, "non_deprecated_parenthesized_import.ipynb", [_code_cell(src)])
    _run(p)
    assert _cell_source(p, 0) == src


def test_single_line_parenthesized_non_deprecated_import_is_preserved(tmp_path):
    src = "from some.module import (UsefulSymbol)\n"
    p = _make_notebook(tmp_path, "single_line_non_deprecated_parenthesized_import.ipynb", [_code_cell(src)])
    _run(p)
    assert _cell_source(p, 0) == src


def test_parenthesized_rewrite_preserves_kept_member_comments(tmp_path):
    src = (
        "from nnx.nn.net.graph_att_nn import (\n"
        "    # concrete network class\n"
        "    GraphAttNN,  # keep this inline note\n"
        "    GraphAttNNParams,  # old params alias\n"
        ")\n"
    )
    p = _make_notebook(tmp_path, "parenthesized_rewrite_preserves_comments.ipynb", [_code_cell(src)])
    _run(p)
    rewritten = _cell_source(p, 0)
    assert "GraphAttNNParams" not in rewritten
    assert "old params alias" not in rewritten
    assert (
        "from nnx.nn.net.graph_att_nn import (\n"
        "    # concrete network class\n"
        "    GraphAttNN,  # keep this inline note\n"
        ")\n"
    ) in rewritten
    assert "from nnx.nn.params.nn_params import NNParams" in rewritten
    compile(rewritten, "<rewritten-cell>", "exec")


def test_parenthesized_rewrite_preserves_closing_paren_comment(tmp_path):
    src = (
        "from nnx.nn.net.graph_att_nn import (\n"
        "    GraphAttNN,\n"
        "    GraphAttNNParams,\n"
        ")  # keep close note\n"
    )
    p = _make_notebook(tmp_path, "parenthesized_rewrite_preserves_closing_comment.ipynb", [_code_cell(src)])
    _run(p)
    rewritten = _cell_source(p, 0)
    assert ")  # keep close note" in rewritten
    assert "GraphAttNNParams" not in rewritten
    assert "from nnx.nn.params.nn_params import NNParams" in rewritten


def test_common_nn_model_split_import_with_inline_comment(tmp_path):
    p = _make_notebook(tmp_path, "common_nn_model_comment.ipynb", [
        _code_cell("from common.nn_model import NNModel, NNTrainParams  # training params\n"),
    ])
    _run(p)
    src = _cell_source(p, 0)
    assert "from nnx.nn.nn_model import NNModel" in src
    assert "from nnx.nn.params.nn_train_params import NNTrainParams" in src
    assert "from nnx.nn.nn_model import NNModel, NNTrainParams" not in src
    compile(src, "<rewritten-cell>", "exec")


def test_missing_input_path_returns_nonzero(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    missing = tmp_path / "missing.ipynb"
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "rewrite_imports.py"), str(missing)],
        check=False,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    assert result.returncode == 1
    assert "SKIP (not found)" in result.stderr
