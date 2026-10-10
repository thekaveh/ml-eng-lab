# 5 Repository conventions

This page is the canonical reference for how the ml-eng-lab repository is
organized, how its notebooks are kept runnable, which gates a change must pass,
and how commits and documentation flow. It expands README §7 and
`CONTRIBUTING.md` into a single self-contained reference; the two root files
remain the quick-start summaries, this page is the durable detail. For
environment-specific runtime paths see [env-setup.md](env-setup.md); for the
dependency ledger behind the pins referenced here see
[dependency-contracts.md](dependency-contracts.md); for the system view of the
documentation pipeline see [architecture.md](architecture.md).

## 5.1 Task-folder layout & naming

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
  [dependency-contracts.md](dependency-contracts.md) §6.1.6). A former in-repo
  `common/` was removed during the 2026-06-14 PyPI migration;
  `scripts/verify_repo.py` enforces its absence via the `S7.forbidden_toplevel`
  structure check. Notebooks import supported symbols through the public facade,
  for example `from nnx import NNModel, NNParams`; equivalent deep package paths
  are not notebook-facing API.
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

### 5.1.1 Adding a new task folder

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
   same YAML **and** to the `TIER_A` list in `Makefile`. CI mirrors that list
   under `/tmp/ml-tier-a/` before upload; keep the transformed workflow list
   in sync as well.
6. Add the task to the root README's active task table (§4.1).
7. Tick the matching roadmap entry in README §8.
8. YAGNI on nnx: only land a library feature when a concrete task needs it.

## 5.2 Notebook execution tiers

Notebooks are tiered by execution cost, and the tier decides both the local
re-run command and what CI exercises. The Makefile owns the authoritative
per-tier notebook lists (`TIER_A`, `TIER_B`, `TIER_C`).

![Notebook execution sequence](diagrams/img/notebook-sequence.png)

The sequence separates task-local execution from repository-wide validation. That boundary keeps
ignored run artifacts near their task while allowing CI to verify source notebooks without
mutating their committed outputs.

| Tier | Cost | Re-run policy | Local target |
|---|---|---|---|
| **A** | Cheap (<5 min) | `make run-tier-a` deliberately refreshes a committed snapshot; CI smoke-runs to `/tmp/ml-tier-a` without rewriting it. | `make run-tier-a` / `make smoke-tier-a` |
| **B** | Moderate (model-selection sweeps) | Original outputs preserved. Smoke-run with `SMOKE_TEST=1` to `/tmp/`. | `make smoke-tier-b` |
| **C** | Expensive (main GPU training) | Historical Aug-2023 GPU outputs preserved as artifact. Smoke-run with `SMOKE_TEST=1` to `/tmp/`. | `make smoke-tier-c` |

### 5.2.1 The `SMOKE_TEST` papermill parameter

Tier-B and Tier-C notebooks are gated by an injected papermill `parameters`
cell that defines `SMOKE_TEST = 0` (full run). The smoke targets pass
`-p SMOKE_TEST 1`, which each notebook reads to shrink its workload: the
parameterized `image_classification-mnist-ffnn-pytorch` notebook reduces its
sweep, the reddit phase2 notebooks run smoke-truncated epochs/subsets (notebook4
also reduces fanout via `n_neighbors=[5,5]`), and the phase3 notebooks take the
CPU path. `scripts/inject_smoke_test_cell.py` adds this cell when promoting a
notebook to Tier-B/C, and `tests/test_inject_smoke_test_cell.py` guards the
injected shape against papermill parser drift.

### 5.2.2 What CI runs

- **Complete pytest, every PR and every push to `develop` or `main`:** the
  unconditional `pytest-repository` job installs the runtime and documentation
  dependency sets, runs exact `make test`, and has a 15-minute cap. It does not
  initialize Atlas and accepts only explicit, reason-bearing test skips.
- **Tier-A, every PR and every push to `develop` or `main`:** the `tier-a-papermill` job
  runs `make smoke-tier-a`, writes fresh copies under `/tmp/ml-tier-a`, then
  runs `make check-tier-a-artifacts` and `make check-tier-a-clean` to prove
  every output exists and execution did not rewrite tracked source notebooks.
  Those fresh copies are uploaded as a 7-day artifact; committed notebook
  outputs are refreshed only deliberately with `make run-tier-a`. The job has
  a 90-minute cap: Linux GH runners are roughly 3–4× slower than a macOS
  M-series CPU for the hand-coded numpy training loop in
  `image_classification-mnist-ffnn-numpy`.
- **Tier-B:** runs on the weekly schedule, on `workflow_dispatch`, and on PRs
  labeled `tier-b-smoke`. Writes to `/tmp/ml-smoke`; never touches committed
  outputs.
- **Tier-C:** runs on the weekly schedule and on `workflow_dispatch` only.
- **All smoke targets** execute each notebook from its own task directory
  (papermill `cwd` = the notebook's folder), so relative paths behave like an
  interactive run. Training/evaluation may therefore create ignored
  task-local `./data/` or `./runs/` artifacts even when source outputs are
  preserved.

### 5.2.3 Tier-C output preservation

Tier-C phase3 notebooks are locked to the immutable annotated
`tier-c-deterministic-seeding-atlas-baseline-2026-08-22` git tag at commit
`35e7903afe45f60e5e30bf8fbd49f7d6463caa6a` for their **code-cell source**.
The execution check `E5` fails closed if that fully qualified tag is missing,
lightweight, moved, or unreadable, and diffs each Tier-C notebook's code cells
against it; markdown cells and embedded outputs are deliberately **not**
compared, so wording fixes are safe. Edit phase3 markdown via
`scripts/edit_notebook_markdown.py` rather than by hand. The historical `pre-cleanup-baseline`
remains an unchanged rollback anchor; never move either published baseline tag.

### 5.2.4 Notebook output freshness

For every output-bearing code cell in a notebook enumerated by the authoritative
`active_task_dirs` inventory, `cell.metadata.source_hash` is a bare lowercase 64-character
SHA-256 digest of the exact UTF-8 logical code-cell source. A JSON `source` string is used as-is;
a JSON `source` array concatenates its string elements with no separator. Whitespace, comments,
magics, line endings, and syntax are not normalized.

Output bytes are never hashed. The marker says retained outputs correspond to the source now in
the cell; it does not say that nondeterministic output rendering is byte-reproducible. E8 reports
error severity for missing, malformed, stale, or orphan markers across all active notebooks.
Any `metadata.source_hash` on an outputless code, markdown or raw cell is
`E8.source_hash_orphan`; normal stamping removes all such markers.

The exception boundary is deliberately narrow: `notebooks/archive/` is immutable historical
material outside the active inventory, and outputless code cells are structurally inapplicable and
must not carry a marker. There is no tag, path, wildcard, or general exemption for an
output-bearing active cell.

`make run-tier-a`, `make smoke-tier-a`, `make smoke-tier-b`, and `make smoke-tier-c` invoke
`scripts/stamp_notebook_source_hashes.py` only after successful Papermill execution. Papermill
inputs may inherit prior source hashes, so merely skipping that success stamper is not sufficient
when execution fails. On a nonzero exit, each Make target checks whether its in-place or temporary
artifact exists and invokes `--clear` to validate the failed/partial notebook and atomically remove
every cell's `metadata.source_hash`; the success stamper is never invoked. The target preserves
the execution failure even when cleanup also fails. This caught-failure boundary cannot run after
an uncatchable host or process kill, so it does not claim crash durability beyond the atomic file
replacement itself.

Clear mode requires one or more explicit notebook paths and rejects `--all-active`. For intentional
maintenance repair, use `python scripts/stamp_notebook_source_hashes.py --all-active`; it is not a
replacement for executing changed source. A success-stamper failure still fails the Make target.
Both modes validate the raw nbformat-4 schema on the parsed JSON without normalization or coercion
before mutation. Rollback is separable: reverting verifier/execution enforcement does not rewrite
outputs, while reverting the metadata migration alone makes E8 fail.

## 5.3 Validation gates

A change is not ready until the repository verifier, complete pytest contract,
Ruff, and documentation gate pass. CI keeps complete pytest separate from faster
focused and publication signals so failures remain attributable.

### 5.3.1 Repo verifier — `make verify`

`scripts/verify_repo.py --check all --fast` runs five checks (the `--fast`
flag only affects the execution check):

- **Structure (`S`)** — task-folder naming, the absence of a `tasks/` subdir
  and of a re-introduced `common/` (`S7.forbidden_toplevel`), no tracked
  bloat (`S7.tracked_bloat`), expected top-level layout.
- **Assets (`D11`)** — the exact VADER manifest and the install/verify ordering
  shared by Make, Docker, CI, Codespaces, the sentiment notebook, and Atlas.
- **Docs (`D`)** — `check_docs` validates the generated documentation tree
  for self-containment, completeness against `docs/manifest.yaml`, and the
  absence of placeholder text.
- **Comments (`C`)** — comment/code invariants the repo relies on.
- **Execution (`E`)** — in `--fast` mode this is skipped; in full mode it
  runs the Tier-A/B/C papermill smoke. `E5` is the Tier-C
  `tier-c-deterministic-seeding-atlas-baseline-2026-08-22` code-cell equality gate described
  above.

Exit code 0 means zero **error**-severity findings; warnings are
informational. The verifier is the source of truth for "is the repo
internally consistent" and runs in CI as the `verify-repo` job.

### 5.3.2 Pytest — `make test`

Runs `pytest tests/ -v`. The test tree covers the docs pipeline
(`test_manifest`, `test_links`, `test_transforms`, `test_render_diagrams`,
`test_build_docs`, `test_wiki`, `test_check_docs`, `test_push_wiki`), the
verifier and helper scripts (`test_verify_repo`,
`test_inject_smoke_test_cell`, `test_edit_notebook_markdown`,
`test_rewrite_imports`), the Makefile contract (`test_makefile_contract`),
and NNx surface guards (`tests/nnx_surface/`). The pytest configuration's
`testpaths = ["tests"]` plus its `infra`, `notebooks/archive`, and `.venv`
exclusions define collection; no fixed test count is contractual.

Both NNx-consuming pytest jobs install the exact hash-required Linux lock selected by
`requirements/lock-policy.toml`; the NNx wheel is binary-only within that lock. After dependency
installation and immediately before their test workload, both run `make verify-nnx-install`: the
unconditional `pytest-repository` job then runs `make test`, while `pytest-nnx-surface` runs
`make test-nnx-surface`. No package installation may intervene between verification and either
workload.

`pytest-repository` is the required merge-blocking complete-suite contract. Its setup installs
`libcairo2`, the selected runtime lock, and the hash-required documentation lock; cache identity
includes the lock policy, source inputs, and committed locks. `pytest-nnx-surface` is the focused
diagnostic NNx/PyPI compatibility and Ruff signal. Both jobs validate the canonical released wheel
with `make verify-nnx-install`; editable results are development-surface evidence only (see
[dependency-contracts.md](dependency-contracts.md) §6).

### 5.3.3 Atlas consumer policy — `make test-atlas-consumer`

The `atlas-consumer-policy` job runs unconditionally on every pull request and is intended to be a
required gate. It recursively checks out the exact `infra/` gitlink so projection tests inspect the
real pinned Atlas files, sets up exact Python 3.11.15, installs the bootstrap lock and hash-required
Atlas contract lock, runs ShellCheck, and executes the focused Make target.
`atlas-contract-requirements.txt` remains the human-authored input and includes exact
`nltk==3.10.3` for the VADER runtime-probe tests; routine execution consumes
`requirements/locks/atlas-contract.txt`.

The fourth step runs
`shellcheck scripts/atlas-up.sh scripts/atlas-down.sh scripts/atlas-connect.sh scripts/lib/atlas-dotenv.sh`.
The fifth runs exact `make test-atlas-consumer`, which selects the consumer contract, lifecycle,
runtime-probe, and Makefile-contract test modules. The job does not start, stop, or contact Atlas,
JupyterHub, Ollama, ComfyUI, Docker Compose, or unrelated containers. It is a focused parent-policy
signal inside complete `make test`, not a replacement for the required `pytest-repository` job.

The separate `atlas-contract` workflow is path-scoped and non-required. It recursively checks out
`infra/` and directly validates the Atlas consumer manifest against that pinned submodule when an
Atlas input changes; it is not the unconditional parent-policy gate.

### 5.3.4 Dependency advisory policy — `make audit-advisories`

`make audit-advisories` runs the unsuppressed comparison of the combined-runtime, Torch,
documentation, and parent-owned Atlas-contract surfaces against
`security/accepted-advisories.json`. New primary IDs and accepted-version drift fail; a disappeared
accepted ID is a reconciliation notice, not proof of remediation. Update the JSON policy and the
current Markdown ledger rows together through review. The focused audit does not initialize Atlas
or start any service.

All four audit surfaces use exact lock-derived, no-resolve projections. The combined-runtime and
Torch surfaces also carry explicit non-PyPI PyG-extension evidence instead of asking PyPI to
re-resolve those platform wheels. The gate validates each projection against the selected locks
before network access.

The unconditional `dependency-audit` job is isolated from `pytest-repository` and
`atlas-consumer-policy` for attribution and timeout isolation. The controller makes it the third
required GitHub context alongside those two existing contexts; the job itself does not make a
ruleset change. Issue #63 supplies and verifies the complete lock family consumed by this job.

### 5.3.5 Lint — `make lint`

`ruff check .` using the `[tool.ruff]` config in `pyproject.toml`: line length
120, target py311, rules `E`/`F`/`W` selected, `E501` (line too long) ignored
because much of the code is notebook-derived and under gradual cleanup.
Tier-C phase3 notebooks carry per-file ignores because their source is locked
to the baseline tag.

### 5.3.6 Docs gate — `make docs-check`

Render diagrams (`scripts/docs.render_diagrams`) → run `check_docs` →
`mkdocs build --strict`. The dedicated `.github/workflows/docs.yml` workflow
runs this plus `ruff check scripts/docs/` and the docs-script unit tests on
any PR touching `docs/`, notebook READMEs, `mkdocs.yml`, `scripts/docs/`,
`docs-requirements.txt`, or the `Makefile`. This is the gate that catches
broken manifest entries, leaked placeholders, and non-self-contained generated
pages before they reach the published site.

## 5.4 Commit & PR workflow

- **Branch off `develop`.** Open a feature branch from the current `develop`
  HEAD and merge it there through a pull request. Promote the resulting
  `develop` state to `main` through a separate pull request; do not merge a
  feature branch directly to `main`.
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

### 5.4.1 Pre-PR checklist

1. `make verify` (fast, <30 s) — must exit 0.
2. `make test` locally; CI reruns that complete contract as `pytest-repository` and also retains the focused `pytest-nnx-surface` signal.
3. If you changed the Atlas consumer policy, run `make test-atlas-consumer` before opening the PR.
4. If you changed an audited manifest or accepted advisory, run `make audit-advisories`.
5. `make lint`.
6. If you touched a notebook: re-run it at the right tier (`make run-tier-a`
   for an intentional Tier-A snapshot refresh, `make smoke-tier-a` for a
   non-mutating Tier-A execution, `make smoke-tier-b`, `make smoke-tier-c`).
   After `make smoke-tier-a`, run `make check-tier-a-artifacts` and
   `make check-tier-a-clean` to confirm the generated copies exist and sources
   remained unchanged.
7. If you touched docs: `make docs-check`.

## 5.5 Documentation convention

ml-eng-lab projects a canonical documentation source set into two additional,
derived surfaces. Canonical sources are the only documentation files a human edits.

- **Canonical sources** — the root `README.md`, manifest-declared root governance
  pages such as `SECURITY.md`, the hand-authored files under `docs/`, and the
  per-task notebook READMEs. `docs/manifest.yaml` indexes every page projected
  into the site and wiki, including root governance pages. This is where content
  is written and reviewed. This page, [architecture.md](architecture.md),
  [env-setup.md](env-setup.md), [dependency-contracts.md](dependency-contracts.md),
  and the other `docs/*.md` files all live in the canonical source set.
- **Generated MkDocs site** — `python -m scripts.docs.build_docs --site` renders the
  manifest into a site input under `generated/`, then `mkdocs build --strict`
  produces `site/`. Published to GitHub Pages by the Pages workflow. Built
  locally with `make docs-build` or previewed with `make docs-serve`.
- **Generated GitHub wiki** — `python -m scripts.docs.build_docs --wiki` renders the
  same manifest into wiki Markdown, and `scripts/docs/push_wiki.py` pushes it
  to the repo's wiki (which renders from `master`, not the repo default
  `main`). Previewed locally with `make docs-wiki` (a `--check` dry run).

The three rules that follow from this:

1. **Never hand-edit the generated trees.** `generated/`, the root `mkdocs.yml`,
   and `site/` are gitignored and rebuilt on every change. If a page looks
   wrong in the site or wiki, fix the canonical source (or the generator in
   `scripts/docs/`), not the rendered output.
2. **The manifest drives both derived surfaces.** Adding a new `docs/*.md`
   page or root governance page means adding an entry to `docs/manifest.yaml`;
   the `check_docs` completeness check fails if a canonical `docs/*.md` page is
   undeclared or if a declared source is missing.
3. **Root entry and governance pages link only to in-repo files.** The three
   surfaces are deliberately independent: a reader of the raw repo (GitHub
   source view, clone) never depends on the generated site or wiki to follow a
   link. The generated surfaces link among themselves; canonical root pages do
   not link into them.

The documentation gate (§5.3) enforces self-containment (every generated page
must resolve its assets without leaving the site), completeness (manifest ↔
source agreement), and the absence of placeholder text — so the three-surface
pipeline stays in sync without manual reconciliation.

## 5.6 Repository layout

```
ml-eng-lab/
├── README.md                                  (project entry point)
├── CONTRIBUTING.md                            (workflow + conventions)
├── SECURITY.md                                (private reporting + support policy)
├── CHANGELOG.md                               (release notes)
├── Makefile                                   (papermill tier targets)
├── security/                                  (accepted-advisory policy)
├── docs/                                      (env/runtime docs, dependency contracts, findings, maintenance log)
├── requirements.txt + torch-*.txt + pyg-extension-audit-requirements.txt (runtime pins plus audit projections; thekaveh-nnx[lm]==0.2.0)
├── infra/                                     (Atlas git submodule; pinned infrastructure)
├── atlas.consumer.yml                         (ml-eng Atlas consumer contract)
├── compose/                                   (parent-owned Atlas compose overlays)
├── scripts/                                   (Atlas lifecycle, verifier, notebook edit/import helpers)
├── tests/                                     (pytest: nnx_surface contract + verifier + helpers)
└── notebooks/                                 (21 active task folders plus notebooks/archive/)
```

See [Changelog](../CHANGELOG.md) for release history; per-task folders are linked from the root README active catalog, and secondary docs are linked from the documentation navigation.


## 5.7 Roadmap

The `tabular_classification-iris-mlp-pytorch` task added in 2026-05-28 seeds the `tabular_classification-titanic-xgboost-sklearn` roadmap entry below.

Future tasks planned (each will become a new `notebooks/<task>/` directory):

- [ ] `image_classification-cifar10-resnet-pytorch`
- [ ] `tabular_classification-titanic-xgboost-sklearn`
- [ ] `text_classification-imdb-distilbert-hf` — distinct from the shipped `notebooks/text_classification-agnews-spacy-mlp-pytorch/` (pre-transformer baseline); this entry is specifically the DistilBERT fine-tune / PEFT continuation.
- [ ] `link_prediction-citation-graphsage-pyg` — distinct from the shipped `notebooks/link_prediction-karate-graphsage-pyg/` (small-graph smoke); this entry is on a real citation network.
- [ ] `time_series_forecasting-electricity-tft-pytorch`
- [ ] `anomaly_detection-creditcard-autoencoder-pytorch`
- [ ] `recommendation-movielens-mf-pytorch`
- [ ] `generative-mnist-vae-pytorch` — distinct from the shipped `notebooks/diffusion-mnist-ddpm-pytorch/`; VAEs and diffusion are different generative families.
- [ ] `reinforcement_learning-cartpole-dqn-pytorch`
- [x] `diffusion-mnist-ddpm-pytorch` — shipped 2026-05-29 in PR #4.

Adding a new task: see the "Adding a new task folder" section in [CONTRIBUTING.md](../CONTRIBUTING.md).
