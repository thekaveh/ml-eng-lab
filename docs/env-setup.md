# Environment setup

Four paths, pick whichever fits the moment.

## 1. genai-vanilla jupyterhub (recommended)

As of genai-vanilla `c2ffe8d` (pinned in `vendor/genai-vanilla`), the `jupyterhub` image natively ships the ml-eng-lab dependency set — `thekaveh-nnx[lm]==0.2.0`, `python-louvain`, `nltk`, `spacy`, `torchao`, and `prettytable` — plus the `en_core_web_sm` spaCy model and `vader_lexicon` NLTK corpus baked at image-build time. Two paths, pick by need.

### 1.1. Default — standalone genai-vanilla + VS Code Mode 2

A current genai-vanilla checkout covers the tier-covered ml-eng-lab notebooks except the from-scratch `notebooks/image_classification-mnist-ffnn-numpy/`, which imports sibling `.py` modules from its own folder and needs the §1.2 wrapper-and-bind-mount path. The quantization notebook remains manual-only under `torch>=2.5` + `torchao>=0.17`.

```bash
cd ~/repos/genai-vanilla && ./start.sh
```

Then open any ml-eng-lab notebook locally in VS Code and `Cmd-Shift-P` → **Jupyter: Specify Jupyter Server for Connections** → `http://localhost:63081/?token=<JUPYTERHUB_TOKEN>`. See [vscode-remote-access.md](vscode-remote-access.md) Mode 2 for the token-retrieval detail.

Trade-off: notebook code that does `pd.read_csv("./data/foo.csv")` or `NNRun.save()` writes to the kernel's CWD inside the container (`/home/jovyan/`), not to your host repo. Artifacts land in the `jupyterhub-data` named volume — opaque to `git status`, wiped by `docker volume rm`. Acceptable for Tier-A demos with small re-downloadable datasets; not acceptable when you want host-side persistence (see §1.2).

### 1.2. Persistence variant — wrapper script + bind-mount

Use when you want any of:
- Datasets + `runs/` checkpoints to land on your host filesystem (visible in `git status`, survives `docker compose down -v`).
- The from-scratch `notebooks/image_classification-mnist-ffnn-numpy/notebook.ipynb` to work (sibling `.py` modules).
- A workflow where you `git commit` notebook edits + dataset downloads from inside the container.

```bash
git submodule update --init --recursive      # one-time, for vendor/genai-vanilla
scripts/start-jupyterhub.sh                  # each session
```

The wrapper layers `deploy/genai-vanilla-jupyterhub.override.yml` onto the `vendor/genai-vanilla` submodule's compose, bind-mounting the repo at `/home/jovyan/work/ml-eng-lab/`. It mounts an empty `.ssh` directory by default; set `HOST_SSH_DIR=/path/to/keys` only when you explicitly want host SSH keys mounted read-only. See [jupyterhub-integration.md](jupyterhub-integration.md) for the full two-path walkthrough.

## 2. Local Docker

```bash
docker build -t ml-eng-lab .                   # uses the in-repo Dockerfile
docker run -p 8888:8888 -v "$(pwd):/home/jovyan/work" --shm-size=4g ml-eng-lab
```

Open `http://localhost:8888/?token=<token>` (token printed at startup).

Notes:
- Image is CPU-only.
- `--shm-size=4g` is the minimum for the GNN notebooks; for serious GNN training, increase to 16-50g.

## 3. Local Python venv

```bash
python -m venv .venv && source .venv/bin/activate
make install-torch-stack
pip install -r requirements.txt              # pulls thekaveh-nnx[lm]==0.2.0 from PyPI

# One-time downloads for the two Tier-A NLP tasks (text_classification-agnews-spacy-mlp
# and sentiment_classification-vader-mlp). pip install doesn't pull these.
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

jupyter lab
```

Caveats:
- PyG wheels: torch + torch_geometric must match. The core pins in `torch-core-requirements.txt` and PyG pins in `torch-requirements.txt` are tested against the `--find-links` wheel index at `https://data.pyg.org/whl/torch-2.4.0+cpu.html`.
- macOS Apple Silicon: PyG wheels for `arm64` are not always available at that index. If pip falls back to source builds, expect ~15 min compile time and Xcode CLT installed.
- The `docker build`/`docker run` path in §2 above bakes the spaCy + NLTK downloads into the image, so the venv-only path above is the only one that needs them done manually. See [`CONTRIBUTING.md`](https://github.com/thekaveh/ml-eng-lab/blob/main/CONTRIBUTING.md) §5.1 for the same instructions in the contributor workflow.

## 4. GitHub Codespaces (zero-click cloud dev)

Click **Code → Codespaces → Create codespace on main** on github.com/thekaveh/ml-eng-lab. The repo ships [`.devcontainer/devcontainer.json`](https://github.com/thekaveh/ml-eng-lab/blob/main/.devcontainer/devcontainer.json) which declaratively defines the runtime:

- **Base image**: `mcr.microsoft.com/devcontainers/python:3.11-bookworm` (Python 3.11, matches the version pin in `.python-version` + CI).
- **`postCreateCommand`**: `make codespace-setup` — runs the Torch-first install target, `pip install -r requirements.txt`, and `make nlp-assets` in the same order CI uses. ~2-3 min one-time per Codespace.
- **VS Code extensions** preinstalled: `ms-python.python`, `ms-toolsai.jupyter` + `jupyter-cell-tags` (makes the papermill `parameters` tag visible) + `jupyter-keymap` + `jupyter-renderers`.
- **Repo location**: `/workspaces/ml-eng-lab` — auto-cloned, persistent across kernel restarts within the Codespace. The `image_classification-mnist-ffnn-numpy` notebook's sibling `.py` imports resolve here natively.

**Editor choice**: open notebooks in the browser-based VS Code that Codespaces ships with, or set JupyterLab as your default editor at [github.com/settings/codespaces → Editor preference → JupyterLab](https://github.com/settings/codespaces) for single-click access.

**Caveats**:
- **GPU**: deprecated 2025-08-29 (Azure NCv3 retirement); see §5 below. Notebooks here run on CPU only.
- **Quantization notebook**: still won't run — same `torch.int1` / `torch==2.4.1` incompatibility documented in its task README.
- **Persistence**: `./data/` and `./runs/` content is lost when the Codespace is deleted. Commit anything you want to keep.

See [README.md §3.4](https://github.com/thekaveh/ml-eng-lab/blob/main/README.md#34-github-codespaces-zero-click-cloud-dev) for the full motivation + scenario list (why this 4th path exists alongside §1 / §2 / §3, what it does and doesn't solve).

## 5. GPU notes

The current setup is CPU-only. No GPU image variant is shipped. For GPU training:
- Tier-C GNN notebooks were originally trained on GPU (Aug 2023 outputs preserved).
- For new GPU runs, use a cloud GPU box with `torch.cuda` available, or set up a separate GPU-enabled jupyterhub variant (out of scope here).

## 6. Tier mapping

The authoritative list lives in `Makefile` (`TIER_A` / `TIER_B` / `TIER_C` variables) and `scripts/verify_repo_config.yaml` (`tier_a_notebooks`). The lists below are mirrored from there; if they drift, the Makefile + YAML win.

- **Tier-A** (`make run-tier-a`, runs in CI on every PR):
  - `notebooks/image_classification-mnist-ffnn-numpy/notebook.ipynb`
  - `notebooks/node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb`
  - `notebooks/tabular_classification-iris-mlp-pytorch/notebook.ipynb`
  - `notebooks/model_surgery-mnist-ffnn-pytorch/notebook.ipynb`
  - `notebooks/pruning-mnist-ffnn-pytorch/notebook.ipynb`
  - `notebooks/knowledge_distillation-mnist-ffnn-pytorch/notebook.ipynb`
  - `notebooks/text_generation-tinyshakespeare-transformer-pytorch/notebook.ipynb`
  - `notebooks/peft-mnist-to-fmnist-dora-vs-lora-pytorch/notebook.ipynb`
  - `notebooks/dim_reduction-iris-autoencoder-pytorch/notebook.ipynb`
  - `notebooks/tabular_regression-diabetes-mlp-pytorch/notebook.ipynb`
  - `notebooks/diffusion-mnist-ddpm-pytorch/notebook.ipynb`
  - `notebooks/moe-fmnist-mixture-of-experts-pytorch/notebook.ipynb`
  - `notebooks/clustering-iris-kmeans-vs-ae-pytorch/notebook.ipynb`
  - `notebooks/link_prediction-karate-graphsage-pyg/notebook.ipynb`
  - `notebooks/community_detection-karate-louvain-vs-gnn-pyg/notebook.ipynb`
  - `notebooks/text_classification-agnews-spacy-mlp-pytorch/notebook.ipynb`
  - `notebooks/sentiment_classification-vader-mlp-pytorch/notebook.ipynb`
  - `notebooks/preference_alignment-toy-dpo-pytorch/notebook.ipynb`
  - `notebooks/self_supervised-fmnist-jepa-pytorch/notebook.ipynb`
- **Tier-B** (`make smoke-tier-b`; CI runs on weekly cron, `workflow_dispatch`, or PRs labeled `tier-b-smoke`; passes `-p SMOKE_TEST 1` and writes to /tmp; the parameterized mnist-pytorch notebook shrinks its sweep, and the 4 phase2 reddit notebooks run smoke-truncated epochs/subsets, with notebook4 also reducing fanout):
  - `notebooks/image_classification-mnist-ffnn-pytorch/notebook.ipynb` (full `[9 hidden_dims × 2 dropouts × 500 epochs]` sweep — `~17 min macOS / >90 min Linux`; moved out of Tier-A per [issue #7](https://github.com/thekaveh/ml-eng-lab/issues/7))
  - `notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook{1,2,3,4}.ipynb`
- **Manual-only** (excluded from Tier-A/B/C; cannot run in ml-eng-lab's pinned environment):
  - `notebooks/quantization-mnist-ffnn-pytorch/notebook.ipynb` (torchao ≥ 0.9.0 — the earliest version with `Int8WeightOnlyConfig` — references `torch.int1` at import time, which requires `torch ≥ 2.5`; ml-eng-lab pins `torch==2.4.1` for genai-vanilla image-parity. Was Tier-A until 2026-06-02 (#10), Tier-B until 2026-06-16 (`Makefile` TIER_B header comment explains the cron-failure-driven removal). Run locally under `torch>=2.5` + `torchao>=0.17`.)
- **Tier-C** (`make smoke-tier-c`; CI runs on weekly cron or `workflow_dispatch`; writes to /tmp):
  - `notebooks/node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook{,2,3,4}.ipynb`
