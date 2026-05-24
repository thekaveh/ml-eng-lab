# Changelog

This repo follows [Keep a Changelog](https://keepachangelog.com/). Date format: YYYY-MM-DD.

## [Unreleased]

### Added
- `scripts/verify_repo.py` — four-check verification oracle (structure, docs, comments, execution).
- `scripts/verify_repo_config.yaml` — required-sections + Tier-A list pulled out of code; new tasks edit YAML.
- `scripts/verify_repo.py --phase-b-out PATH` — exports surviving comment candidates as JSON for a calling agent to dispatch the LLM judge.
- `scripts/edit_notebook_markdown.py` — Tier-C-safe markdown-cell editor.
- New verifier checks: `S7.forbidden_toplevel` (catches resurrected `common/`), `E7.no_papermill_params_tag` (catches missing `parameters` tag), `E8.stale_output` (output-source-hash drift; no-op until the post-execution hook lands).
- `tests/` — pytest suite for the verifier and the markdown editor (20 tests).
- `docs/FINDINGS-NNX.md`, `docs/FINDINGS-VENDOR.md` — issue sinks for the read-only submodules.
- Canonical hierarchical-section template in every active notebook (`#1 Overview` → `#6 Evaluation & Results`).
- `CONTRIBUTING.md`, `CHANGELOG.md`.
- CI: weekly schedule trigger for `smoke-tier-b` and `smoke-tier-c` jobs.

### Fixed
- `image_classification-mnist-ffnn-numpy/notebook.ipynb`: cell where `net2_idps = net1.train_and_validate()` now correctly trains `net2`. The shallow-vs-deep comparison was silently broken since 2023.
- `image_classification-mnist-ffnn-numpy/linear_layer.py`: `np.matrix.copy(W)` → `np.ndarray.copy(W)` (deprecated API).
- `image_classification-mnist-ffnn-numpy/funcs.py`: deleted dead `relu` and `relu_prime` (only `parametric_relu*` is used).
- `node_classification-reddit-gnn-pyg/README.md`: phase-3 epoch counts corrected (1000 → 2000 for notebooks 2/3/4); phase-2 notebook-1 sweep dimensions corrected (1 optimizer × 2 lrs × 2 dropouts, not 2 optimizers).
- `image_classification-mnist-ffnn-numpy/README.md`: "ReLU" clarified to "parametric ReLU with α=0.01" (matches code).

### Removed
- `common/` — leftover from the pre-nnx era; violated CLAUDE.md.
- `.DS_Store` at repo root.

### Changed
- All per-task READMEs and the root README follow a canonical H2 hierarchy.
- `.gitignore` broadened: covers `docs/superpowers/`, `.mypy_cache/`, `.trunk/`, `.vscode/`, `.pytest_cache/`, `plan-*.md`, `notes-*.md`, `audit-*.md`.
- nnx submodule pointer bumped to `ae4e2f4` (thekaveh/NNx#1 + #2): see "nnx via submodule" below.

### nnx via submodule
The `nnx` submodule (thekaveh/NNx) advanced two releases in this cycle:
- **`thekaveh/NNx#1`**: fixes `Losses.MEAN_SQUARED_ERROR`/`BINARY_CROSS_ENTROPY` swap and the `NNIterationDataPoint.from_state()` KeyError on `val_edp=None`.
- **`thekaveh/NNx#2`**: `NNModel.train()` extracted into 6 helpers; new `Callback` ABC with `EarlyStopping`/`LRMonitor`/`ModelCheckpoint`; `Schedulers` enum (Step/Cosine/OneCycle/LinearWarmup); opt-in mixed-precision via `NNModelParams.mixed_precision`; `VisUtils.confusion_matrix` + `classification_report`; new `GraphNNBase` removes ~95% duplication across the three GNN modules; type hints added to all Enum `__call__`s.

## 2026-05-22 — repo cleanup + doc standardization loop

Iterative /goal-driven verify-and-fix loop converged in 2 rounds. Established the verification oracle, canonical documentation hierarchy, and `pre-cleanup-baseline` recovery tag. 26 doc-conformance errors driven to 0.

## 2026-05-16 — Phase 3 ml repo revival

- `nnx/` extracted as a git submodule pointing at [`thekaveh/NNx`](https://github.com/thekaveh/NNx).
- Notebooks rewritten to `from nnx.X import Y` (was `from common.X`).
- `Makefile` introduced with Tier-A / Tier-B / Tier-C papermill targets.
- `SMOKE_TEST` parameter cell injected into long-running notebooks.
- Per-task READMEs and root README written.

## 2026-05-15 — Phase 2 nnx extraction

`common/` lifted out into a standalone PyTorch toolkit (`nnx`) at [`thekaveh/NNx`](https://github.com/thekaveh/NNx).

## 2026-05-12 — Phase 1 jupyterhub ml-capable runtime

`vendor/genai-vanilla/` added as the primary jupyterhub runtime, with ml-specific overrides under `deploy/`.

## 2023-08 — Original experiments

Aug-2023 GNN training runs (phase-3 notebooks) — preserved outputs are part of the artifact; do not re-execute.
