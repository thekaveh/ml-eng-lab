# Changelog

This repo follows [Keep a Changelog](https://keepachangelog.com/). Date format: YYYY-MM-DD.

## [Unreleased]

### Added
- `quantization-mnist-ffnn-pytorch/` — new Tier-A single-notebook task. First in-repo demo of `nnx.quantize_int8` (PTQ INT8 weight-only via torchao) + `nnx.qat_train_step_factory` / `nnx.QATLifecycleCallback` with `qat_config="8da4w"` (8-bit dynamic activation + 4-bit weight QAT). Trains FP32 baseline, quantizes it post-hoc (PTQ), separately trains a fresh model end-to-end with the QAT callback, then compares val loss / val accuracy / pickled state-dict size / per-batch CPU forward latency side-by-side. Asserts QAT callback advanced through `is_prepared` + `is_converted` and the converted net contains `Int8DynActInt4WeightLinear` modules. Hidden widths pinned to multiples of 32 (`[128, 64]`) to fit the int4 default groupsize cleanly without `padding_allowed=True`. Notebook runs in ~27 s on CPU.
- `model_surgery-mnist-ffnn-pytorch/` — new Tier-A single-notebook task. First in-repo demo of the `nnx.surgery` namespace: trains a small `FeedFwdNN` baseline on MNIST, then asserts that `nnx.widen` and `nnx.deepen` produce a forward output equal to the original within `atol=1e-5` on a probe batch (the Net2Net function-preservation contract), then races warm-start (post-surgery) vs cold-start (random init at the wider shape) resume training. Notebook runs in ~45 s on CPU; baseline pinned to `Activations.RELU` because `nnx.deepen`'s identity-init insertion is function-preserving only for ReLU. Registered in `Makefile` `TIER_A`, `scripts/verify_repo_config.yaml` (`active_task_dirs` + `tier_a_notebooks` + `required_sections`), and the root README §4 task table.
- **`nnx` submodule pointer bumped 6ce1122 → 5d1a746**: pulls in the megamerge (#29 — 12 sub-projects: Transformer fork + tokenizer + generation, PEFT family DoRA/IA3/Prefix/Prompt, paradigms DPO/JEPA/born-again/feature-KD/MoE, MoE net, quantization PTQ/QAT via torchao, pruning, model surgery, viz namespace, ONNX-dynamo opt-in, hub interop) plus the ONNX single-ndarray fix (#30) and the post-megamerge maintenance pass (#31). 128 new top-level exports under `nnx.*`. All pre-existing exports preserved — existing ml-lab task notebooks unchanged.
- `tabular_classification-iris-mlp-pytorch/` — new Tier-A single-notebook task. First in-repo use of `nnx.NNTabularDataset` (pandas → DataLoader plumbing) on the iris dataset, comparing three FFN topologies (`[]` linear / `[8]` / `[16, 8]`) side-by-side with `VisUtils.multi_line_plot` overlay and per-candidate `VisUtils.confusion_matrix`. Per-task README, notebook with six narrative sections (§1–§6), pytest-nnx-surface module covering the call chain.
- `tests/nnx_surface/` — pytest NNx-surface layer (`conftest.py` with shared `tiny_tabular_batch` / `tiny_image_batch` / `tiny_graph_data` fixtures + an autouse `_nnx_run_isolation` fixture that pins seeds and chdir's every test to `tmp_path`; `test_image_classification_mnist_ffnn_pytorch.py`; `test_node_classification_reddit_gnn_pyg.py` with parametrized SAGE/CONV smoke + GAT-with-`n_heads` consolidation regression + `NNParams.state()` round-trip; `test_tabular_classification_iris_mlp_pytorch.py` covering the iris NNTabularDataset → FeedFwdNN → NNModel call chain). Sub-30s; gates every PR via the new `pytest-nnx-surface` CI job. GNN forward-pass tests skip cleanly when `pyg-lib` / `torch-sparse` aren't installed (Linux-x86 wheel-only).
- `.github/workflows/ci.yml` — new `pytest-nnx-surface` job; Tier-B papermill smoke also fires on PRs labeled `tier-b-smoke`.
- `scripts/rewrite_imports.py` — symbol-consolidation pass: `{FeedFwdNN, GraphAtt, GraphConv, GraphSage}Params` → `NNParams` at both import and call sites (with word-boundary regex; trailing-comma-safe; auto-injects `from nnx.nn.params.nn_params import NNParams` at cell top when needed). Closes the 2026-05-16 audit miss.
- `tests/test_rewrite_imports.py` — 6 tests covering the original module-path rewrites + the new symbol consolidation (including a commented-out call-site case).
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
- `node_classification-reddit-gnn-pyg/phase2-model-selection-notebook{1,2,3}.ipynb` cell[4] — rewrote model + train_params construction to the canonical pattern (was using `NNModel(device=, net=)` and `NNTrainParams(learning_rate=, weight_decay=)`, neither of which exist in any nnx version). Banner imports for `Optims`, `NNOptimParams`, `NNModelParams`, `Devices`, `Losses`, `Nets` added per notebook. Tier-B papermill smoke now passes.
- `node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook{,2,3,4}.ipynb` cells [6] + [8] — same canonical rewrite, plus dropping invented `snapshot_x`/`snapshot_interval` kwargs (these never existed in nnx; the `Checkpoints` lifecycle is the replacement). cell[13] snapshot-consumer visualizations commented out with a leading note (followup to adapt to the `NNCheckpoint` lifecycle). Tier-C papermill smoke now passes.
- `node_classification-reddit-gnn-pyg/phase{2,3}-*.ipynb` — applied `rewrite_imports.py`'s new symbol-consolidation pass; `GraphAttNNParams` references removed; `from nnx.nn.params.nn_params import NNParams` injected at cell top where needed.
- `image_classification-mnist-ffnn-numpy/notebook.ipynb`: cell where `net2_idps = net1.train_and_validate()` now correctly trains `net2`. The shallow-vs-deep comparison was silently broken since 2023.
- `image_classification-mnist-ffnn-numpy/linear_layer.py`: `np.matrix.copy(W)` → `np.ndarray.copy(W)` (deprecated API).
- `image_classification-mnist-ffnn-numpy/funcs.py`: deleted dead `relu` and `relu_prime` (only `parametric_relu*` is used).
- `node_classification-reddit-gnn-pyg/README.md`: phase-3 epoch counts corrected (1000 → 2000 for notebooks 2/3/4); phase-2 notebook-1 sweep dimensions corrected (1 optimizer × 2 lrs × 2 dropouts, not 2 optimizers).
- `image_classification-mnist-ffnn-numpy/README.md`: "ReLU" clarified to "parametric ReLU with α=0.01" (matches code).
- Maintenance batch:
  - `tabular_classification-iris-mlp-pytorch/notebook.ipynb` — §1/§4/§6 narrative claims corrected to match measured numbers.
  - `node_classification-reddit-gnn-pyg/phase3-*.ipynb` cell[7] — `net.unpack_batch` → `model.net.unpack_batch`.
  - `node_classification-reddit-gnn-pyg/phase3-*.ipynb` train cells now honor the papermill `SMOKE_TEST_EPOCHS` parameter (previously hardcoded `n_epochs`).
  - `image_classification-mnist-ffnn-numpy/notebook.ipynb`, `image_classification-mnist-ffnn-pytorch/notebook.ipynb`, `node_classification-reddit-gnn-pyg/phase1-*.ipynb` — leftover papermill `injected-parameters` cells removed.
- Overnight maintenance pass (`worktree-overnight-cleanup` branch):
  - `scripts/verify_repo.py` — drop 2 redundant `f` prefixes and 1 unused loop index (ruff F541/F841).
  - `image_classification-mnist-ffnn-numpy/*.py` + `notebook.ipynb` — ruff cleanup (trailing whitespace, EOF newlines, useless semicolon, unused `fig =` assignment with `plt.figure()` side-effect preserved).
  - `image_classification-mnist-ffnn-pytorch/README.md` §1 — `nnx.nn.net.FeedFwdNN` was an invalid import path (the class lives at `nnx.nn.net.feed_fwd_nn.FeedFwdNN` and re-exports as `nnx.FeedFwdNN`); fixed to match the iris README.
  - `docs/env-setup.md` §1 — number subsections 1.1/1.2 to match the repo-wide hierarchical convention.
  - `README.md` §3.2 — match `docs/env-setup.md` §2 docker run (quote `$(pwd)`, add `--shm-size=4g` minimum for GNN notebooks; cross-link).
  - `requirements.txt` — add `pytest` so `make test` (per CONTRIBUTING §2) works after a fresh `pip install -r requirements.txt`; add trailing newline.
  - Active notebook cells (excluding Tier-C phase3, source-locked by check E5) — ruff cleanup: dropped unused imports (`torch_sparse.SparseTensor`, `torch.utils.data.{DataLoader,SubsetRandomSampler}`, `torch.nn.functional`, duplicate `tqdm`, per-net class imports `Graph{Conv,Sage,Att}NN` + `FeedFwdNN` dispatched through `NNModel(Nets.X)` instead) in phase1 + phase2-{1,2,3}; bare `f` prefixes (F541) in mnist-pytorch + phase2-{1,2,3,4} `title=` kwargs and iris `print(…)`; one E701 single-line `if … : continue` split. phase2-notebook4 patched at raw-JSON level to avoid 17k lines of float-precision output re-serialization noise.
  - `tests/test_inject_smoke_test_cell.py` — new (4 tests, brings parity with sibling-script test coverage): happy-path injection inserts a `parameters`-tagged cell before the first code cell; idempotent when one already exists; `main` returns 2 on empty argv; missing files are skipped without aborting the run.
  - `image_classification-mnist-ffnn-pytorch/README.md` + `tabular_classification-iris-mlp-pytorch/README.md` §4 — backtick `make` so "Tier-A `make` target" reads as a command reference.
  - `scripts/verify_repo_config.yaml` — one-line comment noting `tier_a_notebooks` must stay in sync with Makefile's `TIER_A`.
  - `README.md` §3.1 quick-start — replace standalone `scripts/setup-in-jupyter.sh` with the `docker exec -it <jupyterhub-container> /home/jovyan/work/ml-lab/scripts/setup-in-jupyter.sh` form already canonical in `docs/env-setup.md` §1.2 and `docs/jupyterhub-integration.md`; the prior shape sat next to the host-shell `start-jupyterhub.sh` and would have invited the reader to run it in the wrong shell.
  - `requirements.txt` — promote `tqdm` (used by `image_classification-mnist-ffnn-numpy/feed_fwd_nn.py` for the training-loop progress bar) and `nbformat` (used by `scripts/{verify_repo,edit_notebook_markdown,inject_smoke_test_cell}.py` and the test suite) from transitive (via papermill / jupyter) to direct declarations.

### Removed
- `common/` — leftover from the pre-nnx era; replaced by the `nnx` submodule.
- `.DS_Store` at repo root.

### Changed
- `scripts/verify_repo.py` — YAML is now required (no silent fallback); `fast` kwarg dropped from `check_structure` / `check_docs` / `check_comments` (was dead); `typing.{Callable,Iterator}` → `collections.abc.{Callable,Iterator}`.
- `Makefile` `run-tier-a` no longer hardcodes `SMOKE_TEST=1` — it now does the full refresh the help text always promised.
- `requirements.txt` cleanup: `dataclasses` (stdlib backport) and `openapi` (unused/typo) removed; `torch` pin moved into `torch-requirements.txt`.
- `Dockerfile` pinned to `quay.io/jupyter/datascience-notebook:python-3.11`; install order swapped to match CI (torch-requirements first); unused NLTK / spaCy downloads dropped.
- `.github/workflows/ci.yml` — iris notebook added to the Tier-A artifact upload; `cache: pip` extended to the smoke jobs.
- New `Makefile` targets `make test` and `make verify` wrapping the CI invocations.
- All per-task READMEs and the root README follow a canonical H2 hierarchy.
- `.gitignore` broadened: covers `docs/superpowers/`, `.mypy_cache/`, `.trunk/`, `.vscode/`, `.pytest_cache/`, `plan-*.md`, `notes-*.md`, `audit-*.md`.
- `nnx` submodule pointer bumped from `4ec08aa` to `6ce1122` — 21-commit hop adding `train_step_fn` hook on `NNModel.train`, fine-tuning infra (`freeze`/`unfreeze`/`load_pretrained`/`NNParamGroupSpec`), multi-optimizer `Trainer` (GAN/actor-critic/EBM), diffusion (`NoiseSchedulers`/`DiffusionMLP`/`sample`), training paradigms (`kd`/`simclr`/`mixup`/`cutmix` step factories), PEFT (`LoRALinear`/`apply_lora_to`/`AdapterLayer`), seeding (`nnx.set_seed`/`dataloader_worker_init_fn`/`env_snapshot`), `NNTabularDataset` (pandas DataFrame → loaders), opt-in `TensorBoardCallback`/`WandbCallback`, ONNX export (`NNModel.to_onnx`), and back-compat additions (`PredictResult` NamedTuple, `evaluate(extra_metrics=)`). Earlier in this cycle the pointer also moved through `ae4e2f4` (thekaveh/NNx#1 + #2): see "nnx via submodule" below.
- `pre-cleanup-baseline` tag rolled forward to incorporate the audit-miss fix (deliberate, audited; verifier check E5 continues enforcing Tier-C **code-cell source** equality from the new tag forward — markdown and embedded outputs are not compared).

### nnx via submodule
The `nnx` submodule (thekaveh/NNx) advanced two releases in this cycle:
- **`thekaveh/NNx#1`**: fixes `Losses.MEAN_SQUARED_ERROR`/`BINARY_CROSS_ENTROPY` swap and the `NNIterationDataPoint.from_state()` KeyError on `val_edp=None`.
- **`thekaveh/NNx#2`**: `NNModel.train()` extracted into 6 helpers; new `Callback` ABC with `EarlyStopping`/`LRMonitor`/`ModelCheckpoint`; `Schedulers` enum (Step/Cosine/OneCycle/LinearWarmup); opt-in mixed-precision via `NNModelParams.mixed_precision`; `VisUtils.confusion_matrix` + `classification_report`; new `GraphNNBase` removes ~95% duplication across the three GNN modules; type hints added to all Enum `__call__`s.

## 2026-05-22 — repo cleanup + doc standardization loop

Iterative /goal-driven verify-and-fix loop converged in 2 rounds. Established the verification oracle, canonical documentation hierarchy, and `pre-cleanup-baseline` recovery tag. 26 doc-conformance errors driven to 0.

## 2026-05-16 — Phase 3 ml-lab repo revival

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
