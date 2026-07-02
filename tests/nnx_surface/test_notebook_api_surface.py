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

import ast
import inspect
import json
import re
from pathlib import Path

import pytest

import nnx
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


def _source_lines(cell: dict) -> list[str]:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source.splitlines(keepends=True)
    return list(source)


def _live_lines(cell: dict) -> list[str]:
    """Source lines that are not pure comments (commented-out historical code is
    deliberately preserved verbatim in some Tier-C cells and must not be flagged)."""
    return [ln for ln in _source_lines(cell) if not ln.lstrip().startswith("#")]


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


def test_migration_guard_catches_string_source_bad_call():
    forbidden = _visutils_only_methods()
    bad = _synthetic_nb({"cell_type": "code", "source": "Utils.multi_line_plot(x=[1])\n", "outputs": []})
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


# --- nnx constructor signature-completeness guard ----------------------------
#
# The Utils->VisUtils break wasn't the only thing the nnx 0.2.0 migration left
# stranded: NNOptimParams gained a required keyword-only `momentum` arg, so seven
# notebooks calling NNOptimParams(name=, max_lr=, weight_decay=) raised TypeError
# at execution. Attribute-surface scanning can't see that — this guard parses
# each code cell's AST and checks every call to an nnx `NN*` constructor supplies
# all of that constructor's required keyword-only params (resolved live from the
# real nnx signatures, so it tracks future signature changes). Calls with
# positional args or **kwargs unpacking are skipped (not statically resolvable).


def _nnx_required_kwonly_params() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name in dir(nnx):
        obj = getattr(nnx, name)
        if not (inspect.isclass(obj) and name.startswith("NN")):
            continue
        try:
            sig = inspect.signature(obj.__init__)
        except (ValueError, TypeError):
            continue
        out[name] = {
            pname for pname, p in sig.parameters.items()
            if p.kind is p.KEYWORD_ONLY and p.default is p.empty and pname != "self"
        }
    return out


def _nnx_ctor_accepted_params() -> dict[str, set[str]]:
    """All accepted kwarg names per top-level nnx `NN*` constructor, resolved
    live from the INSTALLED nnx. Classes whose ``__init__`` accepts ``**kwargs``
    are omitted (their kwarg set is unbounded → can't validate).

    NOTE: this resolves against whatever nnx is installed, so a kwarg that
    exists only in a local *dev* checkout but not in the released
    ``thekaveh-nnx`` PyPI build will pass locally and FAIL in CI — which is
    exactly the point: it converts dev-vs-release drift (e.g. an unreleased
    ``NNGraphDataset(seed=)``) into a fast pytest-nnx-surface failure instead of
    a slow smoke-tier-b/c crash.
    """
    out: dict[str, set[str]] = {}
    for name in dir(nnx):
        obj = getattr(nnx, name)
        if not (inspect.isclass(obj) and name.startswith("NN")):
            continue
        try:
            sig = inspect.signature(obj.__init__)
        except (ValueError, TypeError):
            continue
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            continue
        out[name] = {pname for pname, p in sig.parameters.items() if pname != "self"}
    return out


def find_nnx_unknown_kwargs(nb: dict, accepted: dict[str, set[str]]) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        lines = [ln for ln in "".join(cell.get("source", [])).splitlines()
                 if not ln.lstrip().startswith(("%", "!"))]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name not in accepted:
                continue
            if any(kw.arg is None for kw in node.keywords):
                continue  # **kwargs unpacking — can't statically resolve
            bad = {kw.arg for kw in node.keywords if kw.arg} - accepted[name]
            if bad:
                out.append(f"code_cell[{idx}]: {name}(...) unknown kwarg(s) {sorted(bad)} (not in installed nnx signature)")
    return out


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_nnx_constructor_calls_use_known_kwargs(nb_path: Path):
    accepted = _nnx_ctor_accepted_params()
    assert accepted.get("NNGraphDataset"), "expected NNGraphDataset to resolve a signature"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_nnx_unknown_kwargs(nb, accepted)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} calls an nnx constructor with a kwarg absent from the "
        f"installed nnx (dev-vs-release drift?):\n  " + "\n  ".join(violations)
    )


def test_nnx_unknown_kwarg_guard_catches_bad_kwarg():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": ["d = NNGraphDataset(ds_class=R, n_neighbors=[2], totally_made_up_kwarg=1)\n"],
        "outputs": [],
    })
    assert find_nnx_unknown_kwargs(bad, {"NNGraphDataset": {"ds_class", "n_neighbors", "n_workers", "transform", "batch_sizes", "root_dir"}})


def test_nnx_unknown_kwarg_guard_allows_real_kwargs():
    good = _synthetic_nb({
        "cell_type": "code",
        "source": ["d = NNGraphDataset(ds_class=R, n_neighbors=[2], n_workers=4, transform=t)\n"],
        "outputs": [],
    })
    assert not find_nnx_unknown_kwargs(good, {"NNGraphDataset": {"ds_class", "n_neighbors", "n_workers", "transform", "batch_sizes", "root_dir"}})


def _called_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def find_signature_violations(nb: dict, required: dict[str, set[str]]) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        # drop ipython magics / shell-escapes that aren't valid python
        lines = [ln for ln in "".join(cell.get("source", [])).splitlines()
                 if not ln.lstrip().startswith(("%", "!"))]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name not in required:
                continue
            if node.args or any(kw.arg is None for kw in node.keywords):
                continue  # positional / **kwargs — can't statically verify
            provided = {kw.arg for kw in node.keywords}
            missing = required[name] - provided
            if missing:
                out.append(f"code_cell[{idx}]: {name}(...) missing required kwarg(s) {sorted(missing)}")
    return out


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_nnx_constructor_calls_supply_required_kwargs(nb_path: Path):
    required = _nnx_required_kwonly_params()
    assert required.get("NNOptimParams"), "expected NNOptimParams to have required keyword-only params"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_signature_violations(nb, required)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} calls an nnx constructor with missing required kwarg(s):\n  "
        + "\n  ".join(violations)
    )


def test_signature_guard_catches_missing_momentum():
    required = _nnx_required_kwonly_params()
    assert "momentum" in required["NNOptimParams"], "fixture assumes momentum is required"
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": ["p = NNOptimParams(name=Optims.ADAM, max_lr=1e-2, weight_decay=5e-4)\n"],
        "outputs": [],
    })
    assert find_signature_violations(bad, required)


def test_signature_guard_allows_complete_call():
    required = _nnx_required_kwonly_params()
    good = _synthetic_nb({
        "cell_type": "code",
        "source": ["p = NNOptimParams(name=Optims.ADAM, max_lr=1e-2, weight_decay=5e-4, momentum=(0.9, 0.999))\n"],
        "outputs": [],
    })
    assert not find_signature_violations(good, required)


# --- v0.2.0 stale-API guards (nnx usage-conformance review, 2026-06-29) -------
#
# The nnx 0.2.0 data model removed the flat per-iteration metric fields and the
# snapshot fields from the intermediate data point, and `NNModel.train` returns
# a single `NNRun` (no tuple to unpack). Seven node-classification notebooks
# (phase2 nb1-3, phase3 nb1-4) plus the image_classification baseline still
# referenced the stale shapes — `idp.train_loss` / `idp.val_error` raise
# `AttributeError`, and `NNRun.load("best")` is not the v0.2.0 idiom (load
# takes a real run id; the BEST checkpoint is reached via
# `NNCheckpoint.load(run=<id>, type=Checkpoints.BEST)` or `run.checkpoints()`).
# These execution-free scans catch the stale shapes on EVERY PR — the phase2/3
# notebooks live in the smoke-only Tier-B/C lanes, so the papermill tiers don't
# exercise them on a normal PR.

# Attribute access of a removed flat IDP metric field (NOT the nested
# `train_edp.loss` / `val_edp.error` form). The leading `.` requires attribute
# access; the trailing `\b` keeps `train_loss_history`-style names from matching.
_STALE_IDP_FIELD_RE = re.compile(
    r"\.(train_loss|train_error|val_loss|val_error|snapshot_x|snapshot_y_hat|snapshot_y)\b"
)
# `NNRun.load("best")` / `NNRun.load('best')` — `"best"` is not a run id.
_NNRUN_LOAD_BEST_RE = re.compile(r"""NNRun\.load\(\s*["']best["']""")


def find_stale_idp_fields(nb: dict) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        for line in _live_lines(cell):
            for m in _STALE_IDP_FIELD_RE.finditer(line):
                out.append(f"code_cell[{idx}]: .{m.group(1)} (removed; use train_edp/val_edp.<loss|error>)")
    return out


def find_nnrun_load_best(nb: dict) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        for line in _live_lines(cell):
            if _NNRUN_LOAD_BEST_RE.search(line):
                out.append(f"code_cell[{idx}]: NNRun.load(\"best\") (use NNCheckpoint.load(run=<id>, type=Checkpoints.BEST))")
    return out


def find_sparse_tensor_edge_index_drops(nb: dict) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        lines = [ln for ln in "".join(_live_lines(cell)).splitlines()
                 if not ln.lstrip().startswith(("%", "!"))]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _called_name(node) != "ToSparseTensor":
                continue
            preserves_edge_index = any(
                kw.arg == "remove_edge_index"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
                for kw in node.keywords
            )
            if not preserves_edge_index:
                out.append(f"code_cell[{idx}]: ToSparseTensor(...) drops edge_index by default")
    return out


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_no_stale_flat_idp_fields(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_stale_idp_fields(nb)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} accesses removed flat IDP/snapshot field(s):\n  "
        + "\n  ".join(violations)
    )


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_no_nnrun_load_best(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_nnrun_load_best(nb)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} calls NNRun.load(\"best\"):\n  "
        + "\n  ".join(violations)
    )


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_no_tosparsetensor_default_edge_index_drop(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_sparse_tensor_edge_index_drops(nb)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} calls ToSparseTensor without preserving edge_index:\n  "
        + "\n  ".join(violations)
    )


def test_stale_idp_guard_catches_flat_fields():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": ["losses = [idp.train_loss for idp in run.idps]\n", "e = min(run.idps, key=lambda i: i.val_error).val_error\n"],
        "outputs": [],
    })
    assert len(find_stale_idp_fields(bad)) >= 2


def test_stale_idp_guard_catches_string_source_flat_fields():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": "losses = [idp.train_loss for idp in run.idps]\n",
        "outputs": [],
    })
    assert find_stale_idp_fields(bad)


def test_stale_idp_guard_allows_nested_form_and_similar_names():
    good = _synthetic_nb({
        "cell_type": "code",
        "source": [
            "losses = [idp.train_edp.loss for idp in run.idps]\n",
            "errs = [i.val_edp.error for i in run.idps if i.val_edp is not None]\n",
            "history = run.train_loss_history\n",  # different attr — must NOT match
        ],
        "outputs": [],
    })
    assert not find_stale_idp_fields(good)


def test_stale_idp_guard_ignores_commented_snapshot_block():
    commented = _synthetic_nb({
        "cell_type": "code",
        "source": ["# if idp.snapshot_y_hat is None: continue  (legacy, removed)\n"],
        "outputs": [],
    })
    assert not find_stale_idp_fields(commented)


def test_nnrun_load_best_guard_catches_call():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": ["for c in NNRun.load(\"best\").checkpoints():\n", "    pass\n"],
        "outputs": [],
    })
    assert find_nnrun_load_best(bad)


def test_nnrun_load_best_guard_catches_string_source_call():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": "run = NNRun.load(\"best\")\n",
        "outputs": [],
    })
    assert find_nnrun_load_best(bad)


def test_nnrun_load_best_guard_allows_real_id_load():
    good = _synthetic_nb({
        "cell_type": "code",
        "source": ["run = NNRun.load(top_runs[0].id)\n", "ckpt = NNCheckpoint.load(run=top_runs[0].id, type=Checkpoints.BEST)\n"],
        "outputs": [],
    })
    assert not find_nnrun_load_best(good)


def test_tosparsetensor_guard_catches_default_edge_index_drop():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": ["transform = pyg.transforms.ToSparseTensor()\n"],
        "outputs": [],
    })
    assert find_sparse_tensor_edge_index_drops(bad)


def test_tosparsetensor_guard_allows_preserving_edge_index():
    good = _synthetic_nb({
        "cell_type": "code",
        "source": ["transform = pyg.transforms.ToSparseTensor(remove_edge_index=False)\n"],
        "outputs": [],
    })
    assert not find_sparse_tensor_edge_index_drops(good)


def test_tosparsetensor_guard_catches_multiline_default_edge_index_drop():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": [
            "transform = pyg.transforms.ToSparseTensor(\n",
            "    fill_cache=False,\n",
            ")\n",
        ],
        "outputs": [],
    })
    assert find_sparse_tensor_edge_index_drops(bad)


def test_tosparsetensor_guard_catches_string_source_default_edge_index_drop():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": "# keep this historical note\ntransform = pyg.transforms.ToSparseTensor()\n",
        "outputs": [],
    })
    assert find_sparse_tensor_edge_index_drops(bad)


def test_tosparsetensor_guard_allows_spaced_keyword_assignment():
    good = _synthetic_nb({
        "cell_type": "code",
        "source": ["transform = pyg.transforms.ToSparseTensor(remove_edge_index = False)\n"],
        "outputs": [],
    })
    assert not find_sparse_tensor_edge_index_drops(good)


def test_tosparsetensor_guard_allows_multiline_preserving_edge_index():
    good = _synthetic_nb({
        "cell_type": "code",
        "source": [
            "transform = pyg.transforms.ToSparseTensor(\n",
            "    fill_cache=False,\n",
            "    remove_edge_index=False,\n",
            ")\n",
        ],
        "outputs": [],
    })
    assert not find_sparse_tensor_edge_index_drops(good)


# --- VisUtils call-signature guard (nnx plotting-API drift) -------------------
#
# The plotting helpers on `nnx.vis_utils.VisUtils` are a recurring source of
# silent drift: a wide nnx version bump renames/removes a kwarg (e.g.
# `scatter_plot(figsize=)` → `fig_size=`), and the only callers that hit it are
# the smoke-only Tier-B/C node-classification notebooks, which the per-PR
# papermill tier never executes — so the `TypeError` only surfaces on the weekly
# cron (or never, if an earlier cell already crashes). This AST guard validates
# every `VisUtils.<m>(...)` call's keyword-arg NAMES against the live signature,
# so a renamed/removed kwarg fails on every PR. (It can't check positional
# structure like `yss_legend`'s (group_labels, line_labels) tuple — that's a
# value shape, not a kwarg name.)


def _visutils_method_params() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name in dir(VisUtils):
        if name.startswith("_"):
            continue
        obj = getattr(VisUtils, name)
        if not callable(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (ValueError, TypeError):
            continue
        # Skip methods that accept **kwargs — their kwarg set is unbounded.
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            continue
        out[name] = {p.name for p in sig.parameters.values() if p.name != "self"}
    return out


def find_visutils_kwarg_violations(nb: dict, params: dict[str, set[str]]) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        lines = [ln for ln in "".join(cell.get("source", [])).splitlines()
                 if not ln.lstrip().startswith(("%", "!"))]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "VisUtils"):
                continue
            if f.attr not in params:
                continue
            provided = {kw.arg for kw in node.keywords if kw.arg}
            bad = provided - params[f.attr]
            if bad:
                out.append(f"code_cell[{idx}]: VisUtils.{f.attr}(...) invalid kwarg(s) {sorted(bad)}")
    return out


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_visutils_calls_use_valid_kwargs(nb_path: Path):
    params = _visutils_method_params()
    assert params.get("multi_line_plot"), "expected VisUtils.multi_line_plot to resolve a signature"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_visutils_kwarg_violations(nb, params)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} calls VisUtils with invalid kwarg(s):\n  "
        + "\n  ".join(violations)
    )


def test_visutils_guard_catches_bad_kwarg():
    params = _visutils_method_params()
    assert "figsize" not in params.get("scatter_plot", set()), "fixture assumes scatter_plot uses fig_size, not figsize"
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": ["VisUtils.scatter_plot(vm=vm, figsize=(25, 20))\n"],
        "outputs": [],
    })
    assert find_visutils_kwarg_violations(bad, params)


def test_visutils_guard_allows_valid_kwargs():
    params = _visutils_method_params()
    good = _synthetic_nb({
        "cell_type": "code",
        "source": ["VisUtils.scatter_plot(vm=vm, fig_size=(25, 20))\n", "VisUtils.multi_line_plot(x=x, yss=y, title=t, yss_legend=g, x_axis_label='a', y_axis_label='b')\n"],
        "outputs": [],
    })
    assert not find_visutils_kwarg_violations(good, params)
