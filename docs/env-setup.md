# Environment setup

Three paths, pick whichever fits the moment.

## 1. genai-vanilla jupyterhub (recommended)

This repo vendors [`genai-vanilla`](https://github.com/thekaveh/genai-vanilla) as a git submodule at `vendor/genai-vanilla` (pinned to `main`). The ml-specific docker compose override lives in `deploy/genai-vanilla-jupyterhub.override.yml` and is applied via a wrapper script.

### 1.1. One-time setup

```bash
git submodule update --init --recursive
```

### 1.2. Each session

```bash
scripts/start-jupyterhub.sh
```

The first time a container is created (or after image rebuild), run inside it:

```bash
docker exec -it <jupyterhub-container> /home/jovyan/work/ml-lab/scripts/setup-in-jupyter.sh
```

This installs nnx editable so notebook imports resolve. See [jupyterhub-integration.md](jupyterhub-integration.md) for the details.

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

- **Tier A** (`make run-tier-a`, runs in CI on every PR):
  - `image_classification-mnist-ffnn-numpy/notebook.ipynb`
  - `image_classification-mnist-ffnn-pytorch/notebook.ipynb`
  - `node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb`
  - `tabular_classification-iris-mlp-pytorch/notebook.ipynb`
  - `model_surgery-mnist-ffnn-pytorch/notebook.ipynb`
  - `quantization-mnist-ffnn-pytorch/notebook.ipynb`
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
- **Tier B** (`make smoke-tier-b`, on-demand, writes to /tmp):
  - `node_classification-reddit-gnn-pyg/phase2-model-selection-notebook{1,2,3,4}.ipynb`
- **Tier C** (`make smoke-tier-c`, on-demand, writes to /tmp):
  - `node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook{,2,3,4}.ipynb`
