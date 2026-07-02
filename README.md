# ml-lab — personal ML lab

A multi-project repository of machine-learning task demonstrations, organized as a portfolio of self-contained ML experiments. Each top-level folder follows the convention `[task]-[dataset]-[model]-[framework]` and contains its own notebook(s), README, data directory (gitignored), and runs directory (gitignored).

## 1. Overview

This repo serves three overlapping purposes:

- **Personal lab** — a place to prototype new ML tasks quickly.
- **Portfolio** — each task folder reads as a standalone demonstration of a technique.
- **Educational resource** — notebooks include narrative explanations alongside code.

**Paradigms covered** (see [§4.1](#41-active) for the per-task mapping): image classification (numpy from-scratch + PyTorch FFNN), tabular classification + regression, GNNs on graphs (`pytorch-geometric` GraphSAGE / GraphConv / GAT — node classification, link prediction, community detection), NLP (spaCy + NLTK pipelines, BPE tokenizer), transformer LM with sampling stack, diffusion (DDPM), preference alignment (DPO), self-supervised (I-JEPA), Mixture-of-Experts, PEFT (LoRA / DoRA), quantization (PTQ + QAT), pruning, knowledge distillation, model surgery (Net2Net), autoencoders, clustering.

A shared PyTorch toolkit (`nnx`, [`thekaveh-nnx`](https://pypi.org/project/thekaveh-nnx/) on PyPI) provides reusable training-loop, dataset, and visualization primitives that the notebooks consume. Library and tasks co-evolve: each new task lands its required `nnx` additions upstream first ([`thekaveh/NNx`](https://github.com/thekaveh/NNx)), then ml-lab bumps the pinned version here. YAGNI applies — no speculative abstractions in `nnx`.

## 2. Repository layout

```
ml-lab/
├── README.md                                  (this file)
├── CONTRIBUTING.md                            (workflow + conventions)
├── CHANGELOG.md                               (release notes)
├── Makefile                                   (papermill tier targets)
├── docs/                                      (env/runtime docs, dependency contracts, findings, maintenance log)
├── requirements.txt + torch-*.txt             (pip deps; thekaveh-nnx[lm]==0.2.0)
├── scripts/                                   (jupyterhub start, verifier, notebook edit/import helpers)
├── deploy/                                    (genai-vanilla compose override)
├── tests/                                     (pytest: nnx_surface contract + verifier + helpers)
├── vendor/genai-vanilla/                      (git submodule, JupyterHub stack)
├── archive/                                   (preserved-as-is experiments)
└── <21 active task folders>                   ([task]-[dataset]-[model]-[framework]/ — full list in §4.1)
```

See [CHANGELOG.md](CHANGELOG.md) for release history; per-task folders are linked from [§4.1 Active](#41-active), and secondary docs are linked from [§10 Other documentation](#10-other-documentation).

## 3. Quick start

Four ways to run these notebooks, in increasing order of "I want my own machine to do the work."

### 3.1. genai-vanilla jupyterhub (recommended)

As of genai-vanilla `cbad341` (PR #26, 2026-06-02), the `jupyterhub` image natively ships the ml-lab dep set + the 2 NLP model assets. The image bakes the now-defunct `nnx-pytorch[lm]` PyPI name; a coordinated upstream bump to `thekaveh-nnx[lm]==0.2.0` is needed before this path covers the tier-covered notebooks on a fresh build (tracked as a follow-up to the 2026-06-14 PyPI migration). Two paths, pick by need:

**Default — standalone genai-vanilla + VS Code Mode 2** (works for tier-covered notebooks once the genai-vanilla image bumps to `thekaveh-nnx[lm]==0.2.0`; one tier-covered exception remains regardless: `image_classification-mnist-ffnn-numpy/notebook.ipynb` imports sibling `.py` modules from its own folder and needs the persistence variant below. The quantization notebook is still manual-only under `torch>=2.5` + `torchao>=0.17`):

```bash
cd ~/repos/genai-vanilla && ./start.sh
# Open any ml-lab notebook locally in VS Code, then:
# Cmd-Shift-P → Jupyter: Specify Jupyter Server for Connections →
#   http://localhost:63081/?token=<JUPYTERHUB_TOKEN>
```

**Persistence variant — wrapper script + bind-mount** (required for the from-scratch `image_classification-mnist-ffnn-numpy` notebook + host-side `./data/`/`./runs/` persistence):

```bash
git submodule update --init --recursive   # one-time, for vendor/genai-vanilla
scripts/start-jupyterhub.sh
```

See [docs/jupyterhub-integration.md](docs/jupyterhub-integration.md) (full two-path walkthrough) and [docs/vscode-remote-access.md](docs/vscode-remote-access.md).

### 3.2. Local Docker

```bash
docker build -t ml-lab .
docker run -p 8888:8888 -v "$(pwd):/home/jovyan/work" --shm-size=4g ml-lab
```

`--shm-size=4g` is the minimum for the GNN notebooks; see [docs/env-setup.md](docs/env-setup.md) §2 for more.

### 3.3. Local venv

```bash
python -m venv .venv && source .venv/bin/activate
make install-torch-stack
pip install -r requirements.txt   # pulls thekaveh-nnx[lm]==0.2.0 from PyPI
make nlp-assets  # one-time spaCy + NLTK assets used by the 2 NLP Tier-A notebooks
jupyter lab
```

See [docs/env-setup.md](docs/env-setup.md) for environment details.

### 3.4. GitHub Codespaces (zero-click cloud dev)

Click **Code → Codespaces → Create codespace on main** on [github.com/thekaveh/ml-lab](https://github.com/thekaveh/ml-lab). After ~2-3 minutes of one-time dep install you have a browser-based VS Code (or JupyterLab — see below) with the 21 active task folders available and 28 of 29 active notebooks runnable under the pinned environment.

**Why this path was added.** The §3.1 / §3.2 / §3.3 paths each require ~10-15 minutes of first-time setup on a new machine (Docker pulls, `git submodule update --init --recursive` for `vendor/genai-vanilla`, pip installs against the requirements manifests, `make nlp-assets` predownloads for spaCy + NLTK). They also each have a coupling cost: §3.1 depends on the genai-vanilla image's pip layer staying in sync with ml-lab's `requirements.txt` (the [`nnx-pytorch[lm]` → `thekaveh-nnx[lm]==0.2.0` follow-up](CHANGELOG.md) is a long-running example of what happens when it drifts); §3.2 and §3.3 require local Docker / a working venv on the dev's machine. Codespaces eliminates both: the `.devcontainer/devcontainer.json` declaratively bakes the install recipe (so the dep set is auto-synced to `requirements.txt`, `torch-core-requirements.txt`, and `torch-requirements.txt` during Codespace creation via `postCreateCommand`, with no image-rebuild loop), and the repo is auto-cloned into `/workspaces/ml-lab` inside the container.

**Scenarios this supports**:
- Onboarding a new contributor — they click "Create codespace" and have a working env in ~2-3 minutes, no local install at all.
- Running a notebook on a beefier machine without local install (the smallest Codespace machine is 2-core / 8 GB RAM — comparable to a low-end laptop, sufficient for every Tier-A notebook; bump to 4-core / 16 GB if any Tier-B sweep feels slow).
- Quick demo / drive-by experiment without polluting the local Python env.
- The `image_classification-mnist-ffnn-numpy/notebook.ipynb` edge case (it imports sibling `.py` modules from its own folder) works natively — Codespaces clones the repo into the container's `/workspaces/ml-lab`, so the kernel sees those files without needing the §3.1 wrapper-and-bind-mount path's `scripts/start-jupyterhub.sh`.

**Scenarios this does NOT support**:
- GPU workloads — GitHub deprecated GPU Codespaces 2025-08-29 (Azure NCv3 retirement). The few GPU-benefiting notebooks (heaviest is `self_supervised-fmnist-jepa-pytorch`) still run on CPU here, just slowly; for real GPU you want a separate path (Modal `function.spawn`, a self-hosted GPU box behind Jupyter Enterprise Gateway, or Vertex AI Workbench / Colab Enterprise).
- Data persistence across Codespace deletions — anything written to `./data/` or `./runs/` is gone when the Codespace is deleted (Codespaces are intended to be cheap and disposable). Commit any results you want to keep, or use Codespaces' "prebuild" feature if dep install time becomes a bottleneck.
- The quantization-mnist-ffnn-pytorch notebook still won't run here — it has the same `torch.int1` vs `torch==2.4.1` incompatibility documented in its task README and in [docs/dependency-contracts.md](docs/dependency-contracts.md) (manual-only).

**How to use**:

1. On [github.com/thekaveh/ml-lab](https://github.com/thekaveh/ml-lab) → green **Code** button → **Codespaces** tab → **Create codespace on main**.
2. Wait ~2-3 min for `postCreateCommand` to run `make codespace-setup` (= Torch-first dependency install + `make nlp-assets`). Progress is visible in the terminal panel.
3. Open any notebook. You can either:
   - **Stay in VS Code (browser)** — the Jupyter / Python extensions are preinstalled per the devcontainer config; works for the 28 tier-covered active notebooks. The quantization notebook is manual-only under `torch>=2.5`.
   - **Switch to JupyterLab** — click the dropdown next to "Open" on github.com → choose JupyterLab. To make JupyterLab the single-click default for all your codespaces, go to [github.com/settings/codespaces → Editor preference → JupyterLab](https://github.com/settings/codespaces).

See [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) for the exact image + extension set, and [`Makefile`](Makefile) `codespace-setup` target for the Codespaces/venv install recipe. The §3.2 Docker path bakes the same Torch-first dependency order into [`Dockerfile`](Dockerfile). Free-tier Codespaces (60 core-hours/month on personal accounts, 90 on Pro) is enough for typical solo-maintainer usage.

## 4. Tasks

### 4.1. Active

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

### 4.2. Archived

| Folder | Task | Dataset | Model | Framework |
|---|---|---|---|---|
| [archive/codexglue_summarization/](archive/codexglue_summarization/) | Code summarization (22 experiments) | CodeXGLUE | Transformers | HuggingFace |

### 4.3. Planned

See [§8 Roadmap](#8-roadmap).

## 5. Notebook re-execution policy

Notebooks are tiered by execution cost:

| Tier | What it is | Re-run policy |
|---|---|---|
| **A** | Cheap (<5 min) | `make run-tier-a` re-runs and refreshes outputs. Verified in CI on every PR. Tier-A notebooks also accept a `SMOKE_TEST` papermill parameter (default `0` = full run). |
| **B** | Moderate (model-selection sweeps) | Original outputs preserved. `make smoke-tier-b` runs `SMOKE_TEST=1` and writes to `/tmp/`: the parameterized `image_classification-mnist-ffnn-pytorch` notebook shrinks its sweep, and the 4 phase2 reddit notebooks run smoke-truncated epochs/subsets (notebook4 also reduces fanout). |
| **C** | Expensive (main GPU training) | Historical Aug-2023 GPU training-run outputs preserved as artifact. `make smoke-tier-c` runs CPU with `SMOKE_TEST=1` to validate the pipeline without overwriting outputs. |

See [docs/env-setup.md](docs/env-setup.md) for the tier mapping.

## 6. NNx library

Throughout this README, `NNx` refers to the [GitHub project](https://github.com/thekaveh/NNx); the importable Python package is lowercase `nnx`; the PyPI distribution is [`thekaveh-nnx`](https://pypi.org/project/thekaveh-nnx/).

The library is consumed via PyPI — `thekaveh-nnx[lm]==0.2.0` is pinned in `requirements.txt` (since 2026-06-14, replacing the prior git-submodule editable install). The `[lm]` extra pulls the BPE tokenizer + datasets backbone for the two notebooks that call `train_bpe`/`NNTokenizerParams` (`text_generation-tinyshakespeare-transformer-pytorch/notebook.ipynb` and `preference_alignment-toy-dpo-pytorch/notebook.ipynb`); without it both `ImportError` (issue #12). Notebooks import via `from nnx.X import Y` exactly as before — only the distribution name and install mechanism changed.

To extend `nnx` for a new task:

1. Open a PR against [`thekaveh/NNx`](https://github.com/thekaveh/NNx) with the new feature + a smoke test.
2. After merge, wait for the next NNx release cut (or, for editable iteration during the design phase: clone `thekaveh/NNx` outside the ml-lab tree and `pip install -e <path-to-clone>[lm]` into your venv).
3. Bump the pinned version in `requirements.txt` here (e.g. `thekaveh-nnx[lm]==0.2.1`); open a PR. Tier-A papermill CI re-runs the Tier-A list against the new version; run `make smoke-tier-b`, `make smoke-tier-c`, and manual quantization validation when the NNx change touches those surfaces — same validation discipline as the prior submodule-pointer-bump workflow.

## 7. Repository conventions

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow. Key points:

- Each top-level folder is a self-contained task (`[task]-[dataset]-[model]-[framework]`). No `tasks/` subdirectory.
- Shared library code lives in `nnx` (the PyPI-installed `thekaveh-nnx` package), not a local `common/`.
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

- [docs/env-setup.md](docs/env-setup.md) — the four setup paths (jupyterhub / Docker / venv / Codespaces), GPU notes, Tier mapping.
- [docs/jupyterhub-integration.md](docs/jupyterhub-integration.md) — primary runtime (vendored `genai-vanilla` JupyterHub stack).
- [docs/vscode-remote-access.md](docs/vscode-remote-access.md) — VS Code remote-attach modes.
- [docs/dependency-contracts.md](docs/dependency-contracts.md) — dependency audit ledger, Torch-stack pin rationale, manual-only quantization contract, and external asset notes.
- [docs/maintenance/overnight-2026-07-02.md](docs/maintenance/overnight-2026-07-02.md) — current overnight maintenance pass log and issue tracker.

### 10.3. Issue sinks for external code

- [docs/FINDINGS-NNX.md](docs/FINDINGS-NNX.md) — issue log for the `thekaveh-nnx` library (append findings here; do not edit nnx directly via this repo — fixes land upstream at [`thekaveh/NNx`](https://github.com/thekaveh/NNx)).
- [docs/FINDINGS-VENDOR.md](docs/FINDINGS-VENDOR.md) — same, for the `vendor/genai-vanilla` submodule.

### 10.4. Archive

- [archive/README.md](archive/README.md) — preserved Aug-2023 codexglue summarization experiments (22 runs); read-only.
