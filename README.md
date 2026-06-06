# ml-lab — personal ML lab

A multi-project repository of machine-learning task demonstrations, organized as a portfolio of self-contained ML experiments. Each top-level folder follows the convention `[task]-[dataset]-[model]-[framework]` and contains its own notebook(s), README, data directory (gitignored), and runs directory (gitignored).

## 1. Overview

This repo serves three overlapping purposes:

- **Personal lab** — a place to prototype new ML tasks quickly.
- **Portfolio** — each task folder reads as a standalone demonstration of a technique.
- **Educational resource** — notebooks include narrative explanations alongside code.

**Paradigms covered** (see [§4.1](#41-active) for the per-task mapping): image classification (numpy from-scratch + PyTorch FFNN), tabular classification + regression, GNNs (`pytorch-geometric` GraphSAGE / GraphConv / GAT), NLP (spaCy + NLTK pipelines, BPE tokenizer), transformer LM with sampling stack, diffusion (DDPM), preference alignment (DPO), self-supervised (I-JEPA), Mixture-of-Experts, PEFT (LoRA / DoRA), quantization (PTQ + QAT), pruning, knowledge distillation, model surgery (Net2Net), autoencoders, clustering.

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
└── <21 active task folders>                   ([task]-[dataset]-[model]-[framework]/ — full list in §4.1)
```

See [CHANGELOG.md](CHANGELOG.md) for release history; per-folder docs are linked from [§10 Other documentation](#10-other-documentation).

## 3. Quick start

Three ways to run these notebooks, in increasing order of "I want my own machine to do the work."

### 3.1 genai-vanilla jupyterhub (recommended)

As of genai-vanilla `cbad341` (PR #26, 2026-06-02), the `jupyterhub` image natively ships the full ml-lab dep set (`nnx-pytorch` + 5 pip pkgs + 2 NLP model assets). Two paths, pick by need:

**Default — standalone genai-vanilla + VS Code Mode 2** (works for 26 of 29 ml-lab notebooks; the three exceptions: (a) `image_classification-mnist-ffnn-numpy/notebook.ipynb` imports sibling `.py` modules from its own folder and needs the wrapper-and-bind-mount §2 path below; (b) `text_generation-tinyshakespeare-transformer-pytorch/notebook.ipynb` and (c) `preference_alignment-toy-dpo-pytorch/notebook.ipynb` both call `train_bpe`/`NNTokenizerParams` and need the standalone image bumped to `nnx-pytorch[lm]` before they work on this path; coverage jumps to 28/29 once that lands, tracked as a follow-up to issue #12):

```bash
cd ~/repos/genai-vanilla && ./start.sh
# Open any ml-lab notebook locally in VS Code, then:
# Cmd-Shift-P → Jupyter: Specify Jupyter Server for Connections →
#   http://localhost:63081/?token=<JUPYTERHUB_TOKEN>
```

**Persistence variant — wrapper script + bind-mount** (required for the from-scratch `image_classification-mnist-ffnn-numpy` notebook + host-side `./data/`/`./runs/` persistence):

```bash
git submodule update --init --recursive
scripts/start-jupyterhub.sh
# Optional: docker exec -it <jupyterhub-container> /home/jovyan/work/ml-lab/scripts/setup-in-jupyter.sh
#   (only if hacking on nnx — overrides the image's pip-installed nnx with editable install)
```

See [docs/jupyterhub-integration.md](docs/jupyterhub-integration.md) (full two-path walkthrough) and [docs/vscode-remote-access.md](docs/vscode-remote-access.md).

### 3.2 Local Docker

```bash
docker build -t ml-lab .
docker run -p 8888:8888 -v "$(pwd):/home/jovyan/work" --shm-size=4g ml-lab
```

`--shm-size=4g` is the minimum for the GNN notebooks; see [docs/env-setup.md](docs/env-setup.md) §2 for more.

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
| [model_surgery-mnist-ffnn-pytorch/](model_surgery-mnist-ffnn-pytorch/) | Model surgery (Net2Net) | MNIST | Feed-forward NN | PyTorch (via nnx) |
| [quantization-mnist-ffnn-pytorch/](quantization-mnist-ffnn-pytorch/) | Quantization (PTQ + QAT) | MNIST | Feed-forward NN | PyTorch (via nnx) + torchao |
| [pruning-mnist-ffnn-pytorch/](pruning-mnist-ffnn-pytorch/) | Pruning (magnitude sparsity sweep) | MNIST | Feed-forward NN | PyTorch (via nnx) |
| [knowledge_distillation-mnist-ffnn-pytorch/](knowledge_distillation-mnist-ffnn-pytorch/) | Knowledge distillation (born-again) | MNIST | Feed-forward NN | PyTorch (via nnx) |
| [text_generation-tinyshakespeare-transformer-pytorch/](text_generation-tinyshakespeare-transformer-pytorch/) | Text generation (autoregressive LM) | TinyShakespeare (embedded) | Decoder-only transformer | PyTorch (via nnx) |
| [peft-mnist-to-fmnist-dora-vs-lora-pytorch/](peft-mnist-to-fmnist-dora-vs-lora-pytorch/) | PEFT cross-task adaptation (LoRA vs DoRA) | MNIST → Fashion-MNIST | Feed-forward NN + LoRA / DoRA adapters | PyTorch (via nnx) |
| [dim_reduction-iris-autoencoder-pytorch/](dim_reduction-iris-autoencoder-pytorch/) | Dimensionality reduction (PCA vs autoencoder) | Iris | Autoencoder (FFN with input_dim==output_dim) | PyTorch (via nnx) + sklearn |
| [tabular_regression-diabetes-mlp-pytorch/](tabular_regression-diabetes-mlp-pytorch/) | Tabular regression | Diabetes | Feed-forward MLP + sklearn baselines | PyTorch (via nnx) + sklearn |
| [diffusion-mnist-ddpm-pytorch/](diffusion-mnist-ddpm-pytorch/) | Generative (DDPM diffusion) | MNIST | DiffusionMLP denoiser (no U-Net) | PyTorch (via nnx) |
| [moe-fmnist-mixture-of-experts-pytorch/](moe-fmnist-mixture-of-experts-pytorch/) | Mixture-of-Experts classification | Fashion-MNIST | FeedFwdNN + MoELinear (4 experts, top-2 routing) | PyTorch (via nnx) |
| [clustering-iris-kmeans-vs-ae-pytorch/](clustering-iris-kmeans-vs-ae-pytorch/) | Unsupervised clustering | Iris | KMeans on raw features vs on AE latent | PyTorch (via nnx) + sklearn |
| [link_prediction-karate-graphsage-pyg/](link_prediction-karate-graphsage-pyg/) | Link prediction (GNN encoder) | Zachary Karate Club | GraphSAGE + dot-product scorer | PyTorch Geometric |
| [community_detection-karate-louvain-vs-gnn-pyg/](community_detection-karate-louvain-vs-gnn-pyg/) | Community detection (classical vs GNN) | Zachary Karate Club | Louvain vs GraphSAGE+KMeans | PyTorch Geometric + python-louvain |
| [text_classification-agnews-spacy-mlp-pytorch/](text_classification-agnews-spacy-mlp-pytorch/) | Text classification (4-topic) | Embedded AG-News-style corpus | spaCy + bag-of-words + MLP | PyTorch (via nnx) + spaCy + sklearn |
| [sentiment_classification-vader-mlp-pytorch/](sentiment_classification-vader-mlp-pytorch/) | Sentiment classification (rule vs neural) | Embedded review corpus | VADER (lexicon) vs MLP | PyTorch (via nnx) + nltk + spaCy + sklearn |
| [preference_alignment-toy-dpo-pytorch/](preference_alignment-toy-dpo-pytorch/) | Preference alignment (DPO) | Embedded 16-triplet preference corpus | Tiny TransformerNN (ref + policy) | PyTorch (via nnx) |
| [self_supervised-fmnist-jepa-pytorch/](self_supervised-fmnist-jepa-pytorch/) | Self-supervised (I-JEPA) + linear probe | Fashion-MNIST | ViT + EMA target + JEPA predictor | PyTorch (via nnx) |

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

The library is installed editable via `pip install -e './nnx[lm]'` (part of `requirements.txt`). The `[lm]` extra pulls the BPE tokenizer + datasets backbone for the two notebooks that call `train_bpe`/`NNTokenizerParams` (`text_generation-tinyshakespeare-transformer-pytorch/notebook.ipynb` and `preference_alignment-toy-dpo-pytorch/notebook.ipynb`); without it both `ImportError` (issue #12). Notebooks import via `from nnx.X import Y`.

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
- [ ] `text_classification-imdb-distilbert-hf` — distinct from the shipped `text_classification-agnews-spacy-mlp-pytorch/` (pre-transformer baseline); this entry is specifically the DistilBERT fine-tune / PEFT continuation.
- [ ] `link_prediction-citation-graphsage-pyg` — distinct from the shipped `link_prediction-karate-graphsage-pyg/` (small-graph smoke); this entry is on a real citation network.
- [ ] `time_series_forecasting-electricity-tft-pytorch`
- [ ] `anomaly_detection-creditcard-autoencoder-pytorch`
- [ ] `recommendation-movielens-mf-pytorch`
- [ ] `generative-mnist-vae-pytorch` — distinct from the shipped `diffusion-mnist-ddpm-pytorch/`; VAEs and diffusion are different generative families.
- [ ] `reinforcement_learning-cartpole-dqn-pytorch`
- [x] `diffusion-mnist-ddpm-pytorch` — shipped 2026-05-29 in PR #4.

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
