# Environment setup

Three paths, pick whichever fits the moment.

## 1. genai-vanilla jupyterhub (recommended)

As of genai-vanilla `cbad341` (PR #26, 2026-06-02), the `jupyterhub` image natively ships the full ml-lab dep set — `nnx-pytorch` + `python-louvain` + `nltk` + `spacy` + `torchao` + `prettytable`, plus the `en_core_web_sm` spaCy model + `vader_lexicon` NLTK corpus baked at image-build time. Two paths, pick by need.

### 1.1. Default — standalone genai-vanilla + VS Code Mode 2

Works for **27 of 29 ml-lab notebooks** (every Tier-A/B/C notebook except the from-scratch `image_classification-mnist-ffnn-numpy/`, which imports sibling `.py` modules from its own folder, AND `text_generation-tinyshakespeare-transformer-pytorch/`, whose `NNTokenizerParams` needs the `[lm]` extra — the standalone genai-vanilla image currently bakes `nnx-pytorch` without extras; bumps back to 28/29 once the upstream image picks up `nnx-pytorch[lm]`, tracked as a follow-up to issue #12).

```bash
cd ~/repos/genai-vanilla && ./start.sh
```

Then open any ml-lab notebook locally in VS Code and `Cmd-Shift-P` → **Jupyter: Specify Jupyter Server for Connections** → `http://localhost:63081/?token=<JUPYTERHUB_TOKEN>`. See [vscode-remote-access.md](vscode-remote-access.md) Mode 2 for the token-retrieval detail.

Trade-off: notebook code that does `pd.read_csv("./data/foo.csv")` or `NNRun.save()` writes to the kernel's CWD inside the container (`/home/jovyan/`), not to your host repo. Artifacts land in the `jupyterhub-data` named volume — opaque to `git status`, wiped by `docker volume rm`. Acceptable for Tier-A demos with small re-downloadable datasets; not acceptable when you want host-side persistence (see §1.2).

### 1.2. Persistence variant — wrapper script + bind-mount

Use when you want any of:
- Datasets + `runs/` checkpoints to land on your host filesystem (visible in `git status`, survives `docker compose down -v`).
- The from-scratch `image_classification-mnist-ffnn-numpy/notebook.ipynb` to work (sibling `.py` modules).
- A workflow where you `git commit` notebook edits + dataset downloads from inside the container.

```bash
git submodule update --init --recursive      # one-time
scripts/start-jupyterhub.sh                  # each session
```

The wrapper layers `deploy/genai-vanilla-jupyterhub.override.yml` onto the submodule's genai-vanilla compose, bind-mounting the repo at `/home/jovyan/work/ml-lab/`.

`scripts/setup-in-jupyter.sh` is **optional** and only relevant when actively hacking on `nnx` — it overrides the image's pip-installed `nnx-pytorch` with an editable install pointing at the bind-mounted `nnx/` submodule. See [jupyterhub-integration.md](jupyterhub-integration.md) for the full two-path walkthrough.

## 2. Local Docker

```bash
docker build -t ml-lab .                   # uses the in-repo Dockerfile
docker run -p 8888:8888 -v "$(pwd):/home/jovyan/work" --shm-size=4g ml-lab
```

Open `http://localhost:8888/?token=<token>` (token printed at startup).

Notes:
- Image is CPU-only.
- `--shm-size=4g` is the minimum for the GNN notebooks; for serious GNN training, increase to 16-50g.

## 3. Local Python venv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r torch-requirements.txt
pip install -r requirements.txt
git submodule update --init --recursive    # if not done at clone time

# One-time downloads for the two Tier-A NLP tasks (text_classification-agnews-spacy-mlp
# and sentiment_classification-vader-mlp). pip install doesn't pull these.
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

jupyter lab
```

Caveats:
- PyG wheels: torch + torch_geometric must match. The pins in `torch-requirements.txt` are tested against the `--find-links` wheel index at `https://data.pyg.org/whl/torch-2.4.0+cpu.html`.
- macOS Apple Silicon: PyG wheels for `arm64` are not always available at that index. If pip falls back to source builds, expect ~15 min compile time and Xcode CLT installed.
- The `docker build`/`docker run` path in §2 above bakes the spaCy + NLTK downloads into the image, so the venv-only path above is the only one that needs them done manually. See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) §5.1 for the same instructions in the contributor workflow.

## 4. GPU notes

The current setup is CPU-only. No GPU image variant is shipped. For GPU training:
- Tier-C GNN notebooks were originally trained on GPU (Aug 2023 outputs preserved).
- For new GPU runs, use a cloud GPU box with `torch.cuda` available, or set up a separate GPU-enabled jupyterhub variant (out of scope here).

## 5. Tier mapping

The authoritative list lives in `Makefile` (`TIER_A` / `TIER_B` / `TIER_C` variables) and `scripts/verify_repo_config.yaml` (`tier_a_notebooks`). The lists below are mirrored from there; if they drift, the Makefile + YAML win.

- **Tier-A** (`make run-tier-a`, runs in CI on every PR):
  - `image_classification-mnist-ffnn-numpy/notebook.ipynb`
  - `node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb`
  - `tabular_classification-iris-mlp-pytorch/notebook.ipynb`
  - `model_surgery-mnist-ffnn-pytorch/notebook.ipynb`
  - `pruning-mnist-ffnn-pytorch/notebook.ipynb`
  - `knowledge_distillation-mnist-ffnn-pytorch/notebook.ipynb`
  - `text_generation-tinyshakespeare-transformer-pytorch/notebook.ipynb`
  - `peft-mnist-to-fmnist-dora-vs-lora-pytorch/notebook.ipynb`
  - `dim_reduction-iris-autoencoder-pytorch/notebook.ipynb`
  - `tabular_regression-diabetes-mlp-pytorch/notebook.ipynb`
  - `diffusion-mnist-ddpm-pytorch/notebook.ipynb`
  - `moe-fmnist-mixture-of-experts-pytorch/notebook.ipynb`
  - `clustering-iris-kmeans-vs-ae-pytorch/notebook.ipynb`
  - `link_prediction-karate-graphsage-pyg/notebook.ipynb`
  - `community_detection-karate-louvain-vs-gnn-pyg/notebook.ipynb`
  - `text_classification-agnews-spacy-mlp-pytorch/notebook.ipynb`
  - `sentiment_classification-vader-mlp-pytorch/notebook.ipynb`
  - `preference_alignment-toy-dpo-pytorch/notebook.ipynb`
  - `self_supervised-fmnist-jepa-pytorch/notebook.ipynb`
- **Tier-B** (`make smoke-tier-b`, on-demand + weekly cron, writes to /tmp):
  - `image_classification-mnist-ffnn-pytorch/notebook.ipynb` (full `[9 hidden_dims × 500 epochs]` sweep — `~17 min macOS / >90 min Linux`; moved out of Tier-A per [issue #7](https://github.com/thekaveh/ml-lab/issues/7))
  - `quantization-mnist-ffnn-pytorch/notebook.ipynb` (torchao ≥ 0.9.0 — the earliest version with `Int8WeightOnlyConfig` — references `torch.int1` at import time, which requires `torch ≥ 2.5`; ml-lab pins `torch==2.4.1` for genai-vanilla image-parity. Moved out of Tier-A per [issue #10](https://github.com/thekaveh/ml-lab/issues/10).)
  - `node_classification-reddit-gnn-pyg/phase2-model-selection-notebook{1,2,3,4}.ipynb`
- **Tier-C** (`make smoke-tier-c`, on-demand, writes to /tmp):
  - `node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook{,2,3,4}.ipynb`
