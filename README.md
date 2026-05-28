# ml-lab — personal ML lab

A multi-project repository of machine-learning task demonstrations, organized as a portfolio of self-contained ML experiments. Each top-level folder follows the convention `[task]-[dataset]-[model]-[framework]` and contains its own notebook(s), README, data directory (gitignored), and runs directory (gitignored).

## 1. Overview

This repo serves three overlapping purposes:

- **Personal lab** — a place to prototype new ML tasks quickly.
- **Portfolio** — each task folder reads as a standalone demonstration of a technique.
- **Educational resource** — notebooks include narrative explanations alongside code.

A shared PyTorch toolkit (`nnx`, included here as a git submodule) provides reusable training-loop, dataset, and visualization primitives that the notebooks consume. Library and tasks co-evolve: each new task lands its required `nnx` additions upstream first, then bumps the submodule pointer here. YAGNI applies — no speculative abstractions in `nnx`.

## 2. Repository layout

```
ml-lab/
├── README.md                                  (this file)
├── CONTRIBUTING.md                            (workflow + conventions)
├── CHANGELOG.md                               (release notes)
├── Makefile                                   (papermill tier targets)
├── docs/                                      (env, jupyterhub, vscode-remote, FINDINGS-NNX, FINDINGS-VENDOR)
├── scripts/                                   (start, setup, verify, helpers)
├── deploy/                                    (genai-vanilla compose override)
├── tests/                                     (pytest: nnx_surface contract + verifier + helpers)
├── nnx/                                       (git submodule → thekaveh/NNx)
├── vendor/genai-vanilla/                      (git submodule, JupyterHub stack)
├── archive/                                   (preserved-as-is experiments)
├── image_classification-mnist-ffnn-numpy/      ┐
├── image_classification-mnist-ffnn-pytorch/    │
├── node_classification-reddit-gnn-pyg/         │ active task folders
└── tabular_classification-iris-mlp-pytorch/    ┘
```

See [CHANGELOG.md](CHANGELOG.md) for release history; per-folder docs are linked from [§10 Other documentation](#10-other-documentation).

## 3. Quick start

Three ways to run these notebooks, in increasing order of "I want my own machine to do the work."

### 3.1 genai-vanilla jupyterhub (recommended)

This repo vendors `genai-vanilla` as a submodule (pinned to its `main`). The ml-specific compose override lives in `deploy/` and is applied via a wrapper script:

```bash
git submodule update --init --recursive
scripts/start-jupyterhub.sh
# Then attach VS Code or browse to http://localhost:63081
scripts/setup-in-jupyter.sh   # one-time per container, inside the container
```

See [docs/jupyterhub-integration.md](docs/jupyterhub-integration.md) and [docs/vscode-remote-access.md](docs/vscode-remote-access.md).

### 3.2 Local Docker

```bash
docker build -t ml-lab .
docker run -p 8888:8888 -v $(pwd):/home/jovyan/work ml-lab
```

### 3.3 Local venv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r torch-requirements.txt
pip install -r requirements.txt
jupyter lab
```

See [docs/env-setup.md](docs/env-setup.md) for environment details.

## 4. Tasks

### 4.1 Active

| Folder | Task | Dataset | Model | Framework |
|---|---|---|---|---|
| [image_classification-mnist-ffnn-numpy/](image_classification-mnist-ffnn-numpy/) | Image classification | MNIST | Feed-forward NN (from scratch) | NumPy |
| [image_classification-mnist-ffnn-pytorch/](image_classification-mnist-ffnn-pytorch/) | Image classification | MNIST | Feed-forward NN | PyTorch (via nnx) |
| [node_classification-reddit-gnn-pyg/](node_classification-reddit-gnn-pyg/) | Node classification | Reddit2 | GNN (GraphConv, GraphSAGE, GAT) | PyTorch Geometric (via nnx) |
| [tabular_classification-iris-mlp-pytorch/](tabular_classification-iris-mlp-pytorch/) | Tabular classification | Iris | Feed-forward NN | PyTorch (via nnx) |

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [Browse this repo on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/tree/main/) for full rendering of any notebook.

### 4.2 Archived

| Folder | Task | Dataset | Model | Framework |
|---|---|---|---|---|
| [archive/codexglue_summarization/](archive/codexglue_summarization/) | Code summarization (22 experiments) | CodeXGLUE | Transformers | HuggingFace |

### 4.3 Planned

See [§8 Roadmap](#8-roadmap).

## 5. Notebook re-execution policy

Notebooks are tiered by execution cost:

| Tier | What it is | Re-run policy |
|---|---|---|
| **A** | Cheap (<5 min) | `make run-tier-a` re-runs and refreshes outputs. Verified in CI on every PR. Tier-A notebooks also accept a `SMOKE_TEST` papermill parameter (default `0` = full run). |
| **B** | Moderate (model-selection sweeps) | Original outputs preserved. `make smoke-tier-b` writes to `/tmp/`. |
| **C** | Expensive (main GPU training) | Historical Aug-2023 GPU training-run outputs preserved as artifact. `make smoke-tier-c` runs CPU with `SMOKE_TEST=1` to validate the pipeline without overwriting outputs. |

See [docs/env-setup.md](docs/env-setup.md) for the tier mapping.

## 6. NNx library

Throughout this README, `NNx` refers to the [GitHub project](https://github.com/thekaveh/NNx); the importable Python package is lowercase `nnx`.

The shared toolkit lives as a git submodule at [`./nnx`](./nnx) → [`thekaveh/NNx`](https://github.com/thekaveh/NNx). Clone with submodules:

```bash
git clone --recurse-submodules <this repo>
# or if already cloned:
git submodule update --init --recursive
```

The library is installed editable via `pip install -e ./nnx` (part of `requirements.txt`). Notebooks import via `from nnx.X import Y`.

To extend `nnx` for a new task:

1. Branch in the submodule: `cd nnx && git checkout -b feature-name`.
2. Add the feature plus a smoke test, commit, push to `thekaveh/NNx`.
3. From the ml-lab repo: `cd nnx && git pull && cd .. && git add nnx`.
4. Commit the submodule pointer bump in ml-lab.

## 7. Repository conventions

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow. Key points:

- Each top-level folder is a self-contained task (`[task]-[dataset]-[model]-[framework]`). No `tasks/` subdirectory.
- Shared library code lives in `nnx/`, not a local `common/`.
- Notebooks are saved with executed cells (outputs included) for active tasks.
- Tier-C notebooks have their Aug-2023 outputs preserved; never re-execute them in place.
- `archive/` is read-only.

## 8. Roadmap

The `tabular_classification-iris-mlp-pytorch` task added in 2026-05-28 seeds the `tabular_classification-titanic-xgboost-sklearn` roadmap entry below.

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

Adding a new task: see the "Adding a new task folder" section in [CONTRIBUTING.md](CONTRIBUTING.md).

## 9. License

MIT. See [LICENSE](LICENSE).

## 10. Other documentation

The README is the entry point; the items below are the hub's index of secondary documentation.

### 10.1. Workflow + history

- [CONTRIBUTING.md](CONTRIBUTING.md) — workflow, conventions, "Adding a new task folder" recipe, verifier+pytest gates.
- [CHANGELOG.md](CHANGELOG.md) — Keep-a-Changelog release notes.

### 10.2. Environment + runtimes

- [docs/env-setup.md](docs/env-setup.md) — environment variables, Python version, Tier mapping.
- [docs/jupyterhub-integration.md](docs/jupyterhub-integration.md) — primary runtime (vendored `genai-vanilla` JupyterHub stack).
- [docs/vscode-remote-access.md](docs/vscode-remote-access.md) — VS Code remote-attach modes.

### 10.3. Submodule issue sinks

- [docs/FINDINGS-NNX.md](docs/FINDINGS-NNX.md) — issue log for the `nnx` submodule (append findings here; do not edit nnx directly via this repo).
- [docs/FINDINGS-VENDOR.md](docs/FINDINGS-VENDOR.md) — same, for `vendor/genai-vanilla`.

### 10.4. Archive

- [archive/README.md](archive/README.md) — preserved Aug-2023 codexglue summarization experiments (22 runs); read-only.
