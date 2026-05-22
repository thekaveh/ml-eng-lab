# Environment setup

Three paths, pick whichever fits the moment.

## genai-vanilla jupyterhub (recommended)

Prerequisite: a working [`genai-vanilla`](https://github.com/thekaveh/genai-vanilla) checkout. Its jupyterhub service was broadened to be DS/ML-capable at commit `cb4d8f4` (May 2026). All of torch / pytorch-lightning / torch_geometric / torchmetrics are baked into the image.

### One-time setup

```bash
# In the ml repo:
ml/scripts/link-jupyter-override.sh        # symlinks deploy/...override.yml into genai-vanilla
```

Set these in your genai-vanilla `.env`:
```bash
ML_REPO_PATH=/Users/kaveh/repos/ml
HOST_SSH_DIR=$HOME/.ssh                    # so git operations work inside the container
```

### Each session

```bash
cd /path/to/genai-vanilla && ./start.sh    # the override file auto-applies
```

The first time a container is created (or after image rebuild), run inside it:

```bash
docker exec -it <jupyterhub-container> /home/jovyan/work/ml/scripts/setup-in-jupyter.sh
# Or attach via VS Code and run scripts/setup-in-jupyter.sh from the integrated terminal.
```

This `pip install -e`s the nnx submodule so notebook imports resolve. The editable install persists in the named `jupyterhub-data` volume across container restarts; only needs to be re-run if the image is rebuilt.

See [jupyterhub-integration.md](jupyterhub-integration.md) for the details.

## Local Docker

```bash
docker build -t ml-lab .                   # uses the in-repo Dockerfile
docker run -p 8888:8888 -v "$(pwd):/home/jovyan/work" --shm-size=4g ml-lab
```

Open `http://localhost:8888/?token=<token>` (token printed at startup).

Notes:
- Image is CPU-only.
- `--shm-size=4g` is the minimum for the GNN notebooks; for serious GNN training, increase to 16-50g.

## Local Python venv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r torch-requirements.txt
pip install -r requirements.txt
git submodule update --init --recursive    # if not done at clone time
jupyter lab
```

Caveats:
- PyG wheels: torch + torch_geometric must match. The pins in `torch-requirements.txt` are tested against the `--find-links` wheel index at `https://data.pyg.org/whl/torch-2.4.0+cpu.html`.
- macOS Apple Silicon: PyG wheels for `arm64` are not always available at that index. If pip falls back to source builds, expect ~15 min compile time and Xcode CLT installed.

## GPU notes

The current setup is CPU-only. No GPU image variant is shipped. For GPU training:
- Tier-C GNN notebooks were originally trained on GPU (Aug 2023 outputs preserved).
- For new GPU runs, use a cloud GPU box with `torch.cuda` available, or set up a separate GPU-enabled jupyterhub variant (out of scope here).

## Tier mapping

- **Tier A** (`make run-tier-a`, runs in CI):
  - `image_classification-mnist-ffnn-numpy/notebook.ipynb`
  - `image_classification-mnist-ffnn-pytorch/notebook.ipynb`
  - `node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb`
- **Tier B** (`make smoke-tier-b`, on-demand, writes to /tmp):
  - `node_classification-reddit-gnn-pyg/phase2-model-selection-notebook[1-4].ipynb`
- **Tier C** (`make smoke-tier-c`, on-demand, writes to /tmp):
  - `node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook[1-4].ipynb`
