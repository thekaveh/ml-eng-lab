# ml — personal ML lab

A multi-project repository of machine-learning task demonstrations. Each top-level folder is a self-contained task following the convention `[task]-[dataset]-[model]-[framework]`. Each folder contains its own notebook(s), README, data directory (gitignored), and runs directory (gitignored).

A shared PyTorch toolkit (`nnx`, included here as a git submodule) provides reusable training-loop, dataset, and visualization primitives that the notebooks consume.

This repo serves three overlapping purposes:
- **Personal lab**: a place to prototype new ML tasks quickly.
- **Portfolio**: each task folder reads as a standalone demonstration of a technique.
- **Educational resource**: notebooks include narrative explanations alongside code.

## Tasks

| Folder | Task | Dataset | Model | Framework | Status |
|---|---|---|---|---|---|
| [image_classification-mnist-ffnn-numpy/](image_classification-mnist-ffnn-numpy/) | Image classification | MNIST | Feed-forward NN (from scratch) | NumPy | active |
| [image_classification-mnist-ffnn-pytorch/](image_classification-mnist-ffnn-pytorch/) | Image classification | MNIST | Feed-forward NN | PyTorch (via nnx) | active |
| [node_classification-reddit-gnn-pyg/](node_classification-reddit-gnn-pyg/) | Node classification | Reddit | GNN (GraphConv, GraphSAGE, GAT) | PyTorch Geometric (via nnx) | active |
| [archive/codexglue_summarization/](archive/codexglue_summarization/) | Code summarization (23 experiments) | CodeXGLUE | Transformers | HuggingFace | archived |

## Quick start

Three ways to run these notebooks, in increasing order of "I want my own machine to do the work":

### 1. genai-vanilla jupyterhub (Recommended)

This repo vendors `genai-vanilla` as a submodule at `vendor/genai-vanilla`, so the integration is automatic:

```bash
git submodule update --init --recursive
cd vendor/genai-vanilla && ./start.sh
# Then attach VS Code or browse to http://localhost:63081
ml/scripts/setup-in-jupyter.sh         # one-time per container: pip install -e nnx
```

See [docs/jupyterhub-integration.md](docs/jupyterhub-integration.md) and [docs/vscode-remote-access.md](docs/vscode-remote-access.md).

### 2. Local Docker

```bash
docker build -t ml-lab .
docker run -p 8888:8888 -v $(pwd):/home/jovyan/work ml-lab
```

### 3. Local venv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r torch-requirements.txt
pip install -r requirements.txt
jupyter lab
```

See [docs/env-setup.md](docs/env-setup.md) for environment details.

## Notebook re-execution policy

Notebooks are tiered by execution cost:

| Tier | What it is | Re-run policy |
|---|---|---|
| **A** | Cheap (<5 min) | `make run-tier-a` re-runs and refreshes outputs. Verified in CI on every PR. |
| **B** | Moderate (model-selection sweeps) | Original outputs preserved. Imports updated for nnx; environment validates against `make smoke-tier-b` (`workflow_dispatch`). |
| **C** | Expensive (main GPU training) | Original outputs from Aug 2023 preserved. A `SMOKE_TEST` parameter cell allows quick env validation via `make smoke-tier-c` without overwriting preserved outputs. |

See [docs/env-setup.md](docs/env-setup.md) for the tier mapping.

## NNx library

The shared toolkit lives as a git submodule at [`./nnx`](./nnx) → `thekaveh/NNx` (private). Clone with submodules:

```bash
git clone --recurse-submodules <this repo>
# or if already cloned:
git submodule update --init --recursive
```

The library is installed editable via `pip install -e ./nnx` (part of `requirements.txt`). Notebooks import via `from nnx.nn... import ...`.

## Roadmap

Future tasks planned (each will become a new top-level folder):

- [ ] `image_classification-cifar10-resnet-pytorch`
- [ ] `tabular_classification-titanic-xgboost-sklearn`
- [ ] `text_classification-imdb-distilbert-hf`
- [ ] `link_prediction-citation-graphsage-pyg`
- [ ] `time_series_forecasting-electricity-tft-pytorch`
- [ ] `anomaly_detection-creditcard-autoencoder-pytorch`
- [ ] `recommendation-movielens-mf-pytorch`
- [ ] `generative-mnist-vae-pytorch`
- [ ] `reinforcement_learning-cartpole-dqn-pytorch`
- [ ] `diffusion-mnist-ddpm-pytorch`

See [docs/superpowers/specs/2026-05-16-ml-repo-revival-design.md](docs/superpowers/specs/2026-05-16-ml-repo-revival-design.md) §4 for the rationale and the library co-evolution principle.

## License

MIT. See [LICENSE](LICENSE).
