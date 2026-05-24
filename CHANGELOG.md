# Changelog

This repo follows [Keep a Changelog](https://keepachangelog.com/). Date format: YYYY-MM-DD.

## [Unreleased]

### Added
- `scripts/verify_repo.py` — four-check verification oracle (structure, docs, comments, execution).
- `scripts/edit_notebook_markdown.py` — Tier-C-safe markdown-cell editor.
- `tests/` — pytest suite for the verifier and the markdown editor.
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
