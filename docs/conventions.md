# 5. Repository conventions

This page is the canonical reference for how the ml-eng-lab repository is
organized, how its notebooks are kept runnable, which gates a change must pass,
and how commits and documentation flow. It expands README §7 and
`CONTRIBUTING.md` into a single self-contained reference; the two root files
remain the quick-start summaries, this page is the durable detail. For
environment-specific runtime paths see [env-setup.md](env-setup.md); for the
dependency ledger behind the pins referenced here see
[dependency-contracts.md](dependency-contracts.md); for the system view of the
documentation pipeline see [architecture.md](architecture.md).

## 5.1. Task-folder layout & naming

Every active experiment lives in its own self-contained directory directly
under `notebooks/`, named with the four-segment convention:

```
notebooks/<task>-<dataset>-<model>-<framework>/
```

For example `image_classification-mnist-ffnn-numpy/`,
`node_classification-reddit-gnn-pyg/`, or
`text_classification-agnews-spacy-mlp-pytorch/`. The four segments are
lower-snake-case and joined with single dashes; the framework segment is the
concrete toolkit (`pytorch`, `pyg`, `numpy`, `sklearn`), not a family name.

The layout rules that follow from this:

- **No `tasks/` subdirectory.** Active task folders sit directly under
  `notebooks/`. Family-prefixed groupings (`vision/`, `nlp/`, `gnn/`) are
  explicitly forbidden — the task name itself carries the family.
- **No local shared-code directory.** Shared library code lives in the
  `thekaveh-nnx` package installed from PyPI (see §5.5 of the README and
  [dependency-contracts.md](dependency-contracts.md) §6). A former in-repo
  `common/` was removed during the 2026-06-14 PyPI migration;
  `scripts/verify_repo.py` enforces its absence via the `S7.forbidden_toplevel`
  structure check. Notebooks import via `from nnx.X import Y`.
- **Self-contained folders.** Each task folder carries its own `README.md`
  (purpose, dataset, what's in the notebook(s)) and one or more notebooks.
  Multi-notebook tasks (e.g. the reddit GNN task) keep `phase1-*` /
  `phase2-*` / `phase3-*` notebooks together in the same folder.
- **Notebook source hierarchy.** A standard task notebook opens with a top
  markdown cell stating purpose and dataset, then follows the canonical
  §1–§6 sections: Overview / Setup / Data / Model / Training / Evaluation &
  Results. Phase-1 exploration notebooks use a variant: §1, §2, §3 Dataset
  deep-dive. `scripts/verify_repo_config.yaml`'s `required_sections` map
  pins the expected section list per notebook; the structure check enforces it.
- **nbviewer rendering tip.** GitHub's notebook renderer fails on cells with
  large embedded matplotlib PNGs. Each task README includes a tip block
  pointing to the nbviewer mirror (`https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/notebooks/<folder>/<notebook>.ipynb`),
  or to the folder view for multi-notebook tasks.
- **`notebooks/archive/` is read-only.** It holds preserved Aug-2023
  codexglue summarization experiments. Never edit or re-execute its notebooks.

### Adding a new task folder

The recipe (condensed from `CONTRIBUTING.md` §3) is:

1. Survey `thekaveh/NNx`'s `src/nnx/` for reusable primitives before writing
   any new library code.
2. If new primitives are needed, land them upstream in `thekaveh/NNx` first
   (PR + smoke test), wait for the next NNx PyPI release, then bump the
   `thekaveh-nnx[lm]==X.Y.Z` pin in `requirements.txt` here. Do not fork nnx
   behavior into the notebook.
3. Scaffold `notebooks/<task>-<dataset>-<model>-<framework>/` with a
   `README.md` (use `notebooks/node_classification-reddit-gnn-pyg/README.md`
   as the template) and the notebook(s), including the nbviewer tip.
4. Register every active notebook in `required_sections` in
   `scripts/verify_repo_config.yaml` (copy the canonical six-section block for
   a standard task).
5. If the notebook is Tier-A, also add its path to `tier_a_notebooks` in the
   same YAML **and** to the `TIER_A` list in `Makefile` (the CI workflow
   uploads the same list as an artifact — keep them in sync).
6. Add the task to the root README's active task table (§4.1).
7. Tick the matching roadmap entry in README §8.
8. YAGNI on nnx: only land a library feature when a concrete task needs it.

## 5.2. Notebook execution tiers

Notebooks are tiered by execution cost, and the tier decides both the local
re-run command and what CI exercises. The Makefile owns the authoritative
per-tier notebook lists (`TIER_A`, `TIER_B`, `TIER_C`).

| Tier | Cost | Re-run policy | Local target |
|---|---|---|---|
| **A** | Cheap (<5 min) | Re-executed in place; outputs refreshed and committed. | `make run-tier-a` |
| **B** | Moderate (model-selection sweeps) | Original outputs preserved. Smoke-run with `SMOKE_TEST=1` to `/tmp/`. | `make smoke-tier-b` |
| **C** | Expensive (main GPU training) | Historical Aug-2023 GPU outputs preserved as artifact. Smoke-run with `SMOKE_TEST=1` to `/tmp/`. | `make smoke-tier-c` |

### The `SMOKE_TEST` papermill parameter

Tier-B and Tier-C notebooks are gated by an injected papermill `parameters`
cell that defines `SMOKE_TEST = 0` (full run). The smoke targets pass
`-p SMOKE_TEST 1`, which each notebook reads to shrink its workload: the
parameterized `image_classification-mnist-ffnn-pytorch` notebook reduces its
sweep, the reddit phase2 notebooks run smoke-truncated epochs/subsets (notebook4
also reduces fanout via `n_neighbors=[5,5]`), and the phase3 notebooks take the
CPU path. `scripts/inject_smoke_test_cell.py` adds this cell when promoting a
notebook to Tier-B/C, and `tests/test_inject_smoke_test_cell.py` guards the
injected shape against papermill parser drift.

### What CI runs

- **Tier-A, every PR and every push to `main`:** the `tier-a-papermill` job
  runs `make run-tier-a`, then `make check-tier-a-clean` to fail if execution
  changed any tracked output. Refreshed outputs are uploaded as a 7-day
  artifact so a maintainer can recover them without a local re-run. The job
  has a 90-minute cap: Linux GH runners are roughly 3–4× slower than a macOS
  M-series CPU for the hand-coded numpy training loop in
  `image_classification-mnist-ffnn-numpy`.
- **Tier-B:** runs on the weekly schedule, on `workflow_dispatch`, and on PRs
  labeled `tier-b-smoke`. Writes to `/tmp/ml-smoke`; never touches committed
  outputs.
- **Tier-C:** runs on the weekly schedule and on `workflow_dispatch` only.
- **Both smoke tiers** execute each notebook from its own task directory
  (papermill `cwd` = the notebook's folder), so relative paths behave like an
  interactive run. Training/evaluation may therefore create ignored
  task-local `./data/` or `./runs/` artifacts even when source outputs are
  preserved.

### Tier-C output preservation

Tier-C phase3 notebooks are locked to the `pre-cleanup-baseline` git tag for
their **code-cell source**. The execution check `E5` diffs each Tier-C
notebook's code cells against that tag and errors on any mismatch; markdown
cells and embedded outputs are deliberately **not** compared, so wording fixes
are safe. Edit phase3 markdown via `scripts/edit_notebook_markdown.py` rather
than by hand. Force-tagging `pre-cleanup-baseline` requires explicit approval.

## 5.3. Validation gates

A change is not ready until four independent gates pass. Each catches a
different class of regression, and CI runs them as separate jobs so a failure
is attributable.

### Repo verifier — `make verify`

`scripts/verify_repo.py --check all --fast` runs four checks (the `--fast`
flag only affects the execution check):

- **Structure (`S`)** — task-folder naming, the absence of a `tasks/` subdir
  and of a re-introduced `common/` (`S7.forbidden_toplevel`), no tracked
  bloat (`S7.tracked_bloat`), expected top-level layout.
- **Docs (`D`)** — `check_docs` validates the generated documentation tree
  for self-containment, completeness against `docs/manifest.yaml`, and the
  absence of placeholder text.
- **Comments (`C`)** — comment/code invariants the repo relies on.
- **Execution (`E`)** — in `--fast` mode this is skipped; in full mode it
  runs the Tier-A/B/C papermill smoke. `E5` is the Tier-C `pre-cleanup-baseline`
  code-cell equality gate described above.

Exit code 0 means zero **error**-severity findings; warnings are
informational. The verifier is the source of truth for "is the repo
internally consistent" and runs in CI as the `verify-repo` job.

### Pytest — `make test`

Runs `pytest tests/ -v`. The test tree covers the docs pipeline
(`test_manifest`, `test_links`, `test_transforms`, `test_render_diagrams`,
`test_build_docs`, `test_wiki`, `test_check_docs`, `test_push_wiki`), the
verifier and helper scripts (`test_verify_repo`,
`test_inject_smoke_test_cell`, `test_edit_notebook_markdown`,
`test_rewrite_imports`), the Makefile contract (`test_makefile_contract`),
and NNx surface guards (`tests/nnx_surface/`). CI runs the NNx-surface subset
on every PR as the `pytest-nnx-surface` job (`make test-nnx-surface`) — these
guards inspect the installed `nnx` import surface and are exact
release-contract evidence only when `nnx` resolves from the pinned PyPI wheel,
not from an editable checkout (see [dependency-contracts.md](dependency-contracts.md)
§6).

### Lint — `make lint`

`ruff check .` using the `[tool.ruff]` config in `pyproject.toml`: line length
120, target py311, rules `E`/`F`/`W` selected, `E501` (line too long) ignored
because much of the code is notebook-derived and under gradual cleanup.
Tier-C phase3 notebooks carry per-file ignores because their source is locked
to the baseline tag.

### Docs gate — `make docs-check`

Render diagrams (`scripts/docs.render_diagrams`) → run `check_docs` →
`mkdocs build --strict`. The dedicated `.github/workflows/docs.yml` workflow
runs this plus `ruff check scripts/docs/` and the docs-script unit tests on
any PR touching `docs/`, notebook READMEs, `mkdocs.yml`, `scripts/docs/`,
`docs-requirements.txt`, or the `Makefile`. This is the gate that catches
broken manifest entries, leaked placeholders, and non-self-contained generated
pages before they reach the published site.

## 5.4. Commit & PR workflow

- **Branch off `main`.** Open a feature branch off the current `main` HEAD.
  An `origin/develop` integration branch exists for batching larger
  workstreams — for example the documentation overhaul landed as PR #30 into
  `develop`, then `develop` merged into `main` — but either path requires a
  pull request.
- **PRs are required on `main`.** Branch protection (set 2026-05-29) requires
  a pull request, allows zero approvals, and forbids force-push and deletion.
  This lets the solo maintainer self-merge while keeping every change on the
  reviewable PR queue.
- **Conventional commit messages.** Use the `type(scope): subject` form
  (`feat`, `fix`, `docs`, `ci`, `chore`, `test`, `refactor`). The recent
  history is consistent with this — `feat(docs): 20 notebook deep-dives`,
  `fix(docs): push_wiki defaults a commit identity`, `ci(docs): install ruff
  + pytest in the docs gate`. Scope the message to the change.
- **One concern per PR.** Don't bundle unrelated cleanup with a feature.
  Tier-C notebook re-execution, if ever needed, belongs in its own PR —
  preserved outputs are intentional and rare to touch.
- **CHANGELOG is the durable record.** A PR description goes stale after
  merge; `CHANGELOG.md` (Keep-a-Changelog format) is the long-term history.
  Wrap one workstream per branch and keep going on the same branch; record
  the outcome in the CHANGELOG rather than editing old PR bodies.

### Pre-PR checklist

1. `make verify` (fast, <30 s) — must exit 0.
2. `make test` locally; CI additionally runs `make test-nnx-surface`.
3. `make lint`.
4. If you touched a notebook: re-run it at the right tier (`make run-tier-a`,
   `make smoke-tier-b`, `make smoke-tier-c`). For Tier-A, also run
   `make check-tier-a-clean` to confirm outputs are committed.
5. If you touched docs: `make docs-check`.

## 5.5. Documentation convention

ml-eng-lab projects one canonical documentation source into two additional,
derived surfaces. The canonical source is the only one a human edits.

- **Canonical source** — the hand-authored files under `docs/` plus the
  per-task notebook READMEs, indexed by `docs/manifest.yaml`. This is where
  content is written and reviewed. This page, [architecture.md](architecture.md),
  [env-setup.md](env-setup.md), [dependency-contracts.md](dependency-contracts.md),
  and the other `docs/*.md` files all live here.
- **Generated MkDocs site** — `scripts/docs/build_docs --site` renders the
  manifest into a site input under `generated/`, then `mkdocs build --strict`
  produces `site/`. Published to GitHub Pages by the Pages workflow. Built
  locally with `make docs-build` or previewed with `make docs-serve`.
- **Generated GitHub wiki** — `scripts/docs/build_docs --wiki` renders the
  same manifest into wiki Markdown, and `scripts/docs/push_wiki.py` pushes it
  to the repo's wiki (which renders from `master`, not the repo default
  `main`). Previewed locally with `make docs-wiki` (a `--check` dry run).

The three rules that follow from this:

1. **Never hand-edit the generated trees.** `generated/`, the root `mkdocs.yml`,
   and `site/` are gitignored and rebuilt on every change. If a page looks
   wrong in the site or wiki, fix the canonical `docs/` source (or the
   generator in `scripts/docs/`), not the rendered output.
2. **The manifest drives both derived surfaces.** Adding a new `docs/*.md`
   page means adding an entry to `docs/manifest.yaml`; the `check_docs`
   completeness check fails if a manifest page is missing from the source or
   vice versa.
3. **The README links only to in-repo files.** The three surfaces are
   deliberately independent: a reader of the raw repo (GitHub source view,
   clone) never depends on the generated site or wiki to follow a link. The
   generated surfaces link among themselves; the canonical README does not
   link into them.

The documentation gate (§5.3) enforces self-containment (every generated page
must resolve its assets without leaving the site), completeness (manifest ↔
source agreement), and the absence of placeholder text — so the three-surface
pipeline stays in sync without manual reconciliation.
