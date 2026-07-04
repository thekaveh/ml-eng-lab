# Contributing

A short guide for adding new notebook experiment folders and modifying shared code in this lab.

## 1. Conventions

- This is a notebook-driven ML lab. Each active task is a self-contained directory under `notebooks/` using the `[task]-[dataset]-[model]-[framework]` naming convention. Do not introduce a `tasks/` subdirectory or family-prefixed dirs (`vision/`, `nlp/`, ...).
- Shared library code lives in **`thekaveh-nnx`** — the PyTorch toolkit installed from PyPI ([source: `thekaveh/NNx`](https://github.com/thekaveh/NNx)), pinned in `requirements.txt` to `thekaveh-nnx[lm]==0.2.0` (since 2026-06-14). Notebooks import via `from nnx.X import Y`. Do not reintroduce a local `common/` directory — `scripts/verify_repo.py` enforces this via `S7.forbidden_toplevel`.
- The `notebooks/archive/` directory holds preserved-as-is experiments. Read-only.
- New notebooks should include a top markdown cell stating purpose and dataset, plus the canonical §1–§6 hierarchy (Overview / Setup / Data / Model / Training / Evaluation & Results). Phase-1 exploration notebooks use a variant: §1, §2, §3 Dataset deep-dive.

## 2. Workflow

1. Open a feature branch off `main`.
2. Make your change.
3. Run `make verify` (wraps `python scripts/verify_repo.py --check all --fast`) — must exit 0 (no error-severity findings; warnings are OK).
4. Run `make test` (wraps `pytest tests/`) locally. CI also runs `pytest tests/nnx_surface` as the per-PR `pytest-nnx-surface` gate.
5. If you touched a notebook, re-run it (Tier-A: `make run-tier-a`; Tier-B: `make smoke-tier-b`; Tier-C: `make smoke-tier-c`). Tier-C **code cells** must remain identical to the `pre-cleanup-baseline` tag — verify check E5 enforces this (markdown and embedded outputs are not compared).
6. Open a PR. CI runs Tier-A automatically; Tier-B runs on schedule, on `workflow_dispatch`, and on PRs labeled `tier-b-smoke`; Tier-C runs on schedule and on `workflow_dispatch`.

## 3. Adding a new task folder

Convention: active experiment directory named `notebooks/[task]-[dataset]-[model]-[framework]/`.

1. Survey [`thekaveh/NNx`'s `src/nnx/`](https://github.com/thekaveh/NNx/tree/main/src/nnx) for reusable primitives.
2. Identify gaps. If you need new primitives, **land them in [`thekaveh/NNx`](https://github.com/thekaveh/NNx) first** (open a PR upstream), wait for the next NNx PyPI release, then bump `requirements.txt`'s `thekaveh-nnx` version pin here.
3. Scaffold the new task folder with a `README.md` (use [`notebooks/node_classification-reddit-gnn-pyg/README.md`](notebooks/node_classification-reddit-gnn-pyg/README.md) as template) and notebook(s). At the top of §3 "What's in the notebook(s)", include the nbviewer tip — GitHub's notebook renderer chokes on cells with large embedded matplotlib PNGs:

   ```markdown
   > **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/notebooks/<folder>/<notebook>.ipynb) for full rendering.
   ```

   For folders with multiple notebooks, link to the folder view at `https://nbviewer.org/github/thekaveh/ml-eng-lab/tree/main/notebooks/<folder>/` instead.
4. Add every active notebook to `required_sections` in [`scripts/verify_repo_config.yaml`](scripts/verify_repo_config.yaml); ordinary task notebooks should copy the canonical six-section block.
5. If Tier-A, add the notebook path to `tier_a_notebooks` in the same YAML and to `TIER_A` in [`Makefile`](Makefile).
6. Update the root README's task table.
7. Tick the box on the root README roadmap.
8. YAGNI: don't add abstractions to `nnx` speculatively. Only land features when a concrete task needs them.

## 4. Modifying shared code

- **`thekaveh-nnx` is a PyPI dep.** Don't bump the `requirements.txt` pin without a corresponding upstream release on [`thekaveh/NNx`](https://github.com/thekaveh/NNx). Workflow:
  1. Open a PR against `thekaveh/NNx` with the new feature + a smoke test.
  2. After merge, wait for the next NNx PyPI release (or, for editable iteration: clone `thekaveh/NNx` outside the ml-eng-lab tree and `pip install -e <path>[lm]` into your venv).
  3. Bump `thekaveh-nnx[lm]==X.Y.Z` in ml-eng-lab's `requirements.txt` to the new version; open a PR here. Tier-A papermill CI re-runs the Tier-A list against the new version; run `make smoke-tier-b`, `make smoke-tier-c`, and manual quantization validation when the NNx change touches those surfaces.
- **`vendor/genai-vanilla/` is vendored.** Don't edit it directly. The ml-specific compose override lives in [`deploy/`](deploy/) — never commit override files inside `vendor/genai-vanilla/`.
- **`notebooks/archive/` is read-only.** Preserved Aug-2023 work.

Found an issue in the `thekaveh-nnx` library? Append to [docs/FINDINGS-NNX.md](docs/FINDINGS-NNX.md) (and open an upstream issue at [`thekaveh/NNx`](https://github.com/thekaveh/NNx/issues)). Same for `vendor/genai-vanilla`: [docs/FINDINGS-VENDOR.md](docs/FINDINGS-VENDOR.md).

## 5. Running notebooks

Primary runtime: the `genai-vanilla` stack. As of genai-vanilla `448333d`, the image natively ships the ml-eng-lab dependency set, `thekaveh-nnx[lm]==0.2.0`, and the two NLP model assets. Pull and rebuild older genai-vanilla images if they still reference the defunct `nnx-pytorch[lm]` distribution name. The wrapper-and-bind-mount is required for the from-scratch `image_classification-mnist-ffnn-numpy` notebook and for host-side data/runs persistence; the quantization notebook remains manual-only under `torch>=2.5` + `torchao>=0.17`.

- **Default (standalone genai-vanilla)** — `cd ~/repos/genai-vanilla && ./start.sh`, then point VS Code Mode 2 at the token URL.
- **Persistence variant (wrapper + bind-mount)** — `scripts/start-jupyterhub.sh` from the ml-eng-lab repo root (NOT `cd vendor/genai-vanilla && ./start.sh` directly — the wrapper sets `ML_REPO_PATH` and `COMPOSE_FILE` to layer the override).
- **Editable-iteration on NNx itself** — clone `thekaveh/NNx` outside the ml-eng-lab tree, then `pip install -e <path>[lm]` into your venv to override the PyPI install. No in-repo override script.
- **Zero-click cloud dev (GitHub Codespaces)** — `Code → Codespaces → Create codespace on main` on github.com/thekaveh/ml-eng-lab. `.devcontainer/devcontainer.json`'s `postCreateCommand` runs `make codespace-setup` (full pip install + NLP assets, ~2-3 min one-time). See [README.md §3.4](README.md#34-github-codespaces-zero-click-cloud-dev) for the motivation + scenario list (and the GPU + persistence caveats).
- Full two-path walkthrough: [docs/jupyterhub-integration.md](docs/jupyterhub-integration.md).

### 5.1. One-time NLP-task setup

Two Tier-A tasks need a model + a lexicon that `pip install -r requirements.txt` doesn't pull on its own. Run these once after the venv is set up (CI runs them automatically in `.github/workflows/ci.yml`'s `tier-a-papermill` job):

```bash
# spaCy English model — needed by text_classification-agnews-spacy-mlp-pytorch
# and sentiment_classification-vader-mlp-pytorch
python -m spacy download en_core_web_sm

# NLTK VADER lexicon — needed by sentiment_classification-vader-mlp-pytorch
# (the notebook also has a lazy fallback download, but pre-downloading avoids
# the per-run delay)
python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"
```

## 6. Verification

`scripts/verify_repo.py` is the repo's four-check oracle. Run before commits / PRs:

- `python scripts/verify_repo.py --check all --fast` — structure, docs, comments, env-limited execution. Fast (<30s).
- `python scripts/verify_repo.py --check all` — adds the full Tier-A/B/C papermill smoke. Requires the genai-vanilla container or an equivalent fully-provisioned env.

Exit code 0 iff zero error-severity findings; warnings are informational. Tier-C **code-cell source** equality with the `pre-cleanup-baseline` git tag is enforced by check E5 (markdown / outputs are not compared). Edits to phase3 markdown cells should still use `scripts/edit_notebook_markdown.py` for safety.

### 6.1. Helper scripts

- `scripts/verify_repo.py` — the four-check oracle described above.
- `scripts/edit_notebook_markdown.py` — Tier-C-safe markdown-cell editor (changes a single markdown cell's source in-place).
- `scripts/inject_smoke_test_cell.py` — adds a papermill `parameters`-tagged cell (`SMOKE_TEST = 0`) to a notebook. Use when promoting a notebook to Tier-B / Tier-C so `make smoke-tier-b/c` can truncate via `-p SMOKE_TEST 1`.
- `scripts/rewrite_imports.py` — applies the `common/* → nnx/*` module-path rewrite plus the per-net-Params consolidation (`{FeedFwdNN, GraphAtt, GraphConv, GraphSage}Params → NNParams`). Idempotent; safe to re-run.

## 7. One concern per PR

- Don't bundle unrelated cleanup with a feature change.
- Tier-C notebook re-execution belongs in its own PR if you ever need to (rare; preserved outputs are intentional).
