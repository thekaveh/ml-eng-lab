<p align="center">
  <img src="docs/assets/ml-eng-lab-poster.png" alt="ML Eng Lab — notebooks, systems, and reproducibility" width="100%">
</p>

<h1 align="center">ML ENG LAB</h1>

<p align="center"><strong>Local notebooks. Remote Atlas execution. Explicit infrastructure contracts.</strong></p>

<p align="center">
  <sub><strong>Core ML</strong></sub><br>
  <img alt="Python" src="docs/assets/badges/python.svg"> <img alt="Jupyter" src="docs/assets/badges/jupyter.svg"> <img alt="NumPy" src="docs/assets/badges/numpy.svg"> <img alt="pandas" src="docs/assets/badges/pandas.svg"> <img alt="PyTorch" src="docs/assets/badges/pytorch.svg"> <img alt="PyTorch Geometric" src="docs/assets/badges/pytorch-geometric.svg"> <img alt="scikit-learn" src="docs/assets/badges/scikit-learn.svg">
</p>

<p align="center">
  <sub><strong>NLP and graphs</strong></sub><br>
  <img alt="spaCy" src="docs/assets/badges/spacy.svg"> <img alt="NLTK" src="docs/assets/badges/nltk.svg"> <img alt="NetworkX" src="docs/assets/badges/networkx.svg">
</p>

<p align="center">
  <sub><strong>Runtime</strong></sub><br>
  <img alt="Atlas" src="docs/assets/badges/atlas.svg"> <img alt="Docker" src="docs/assets/badges/docker.svg"> <img alt="VS Code" src="docs/assets/badges/vscode.svg"> <img alt="GitHub Codespaces" src="docs/assets/badges/github-codespaces.svg">
</p>

<p align="center">
  <sub><strong>Engineering</strong></sub><br>
  <img alt="NNx" src="docs/assets/badges/nnx.svg"> <img alt="Papermill" src="docs/assets/badges/papermill.svg"> <img alt="pytest" src="docs/assets/badges/pytest.svg"> <img alt="Ruff" src="docs/assets/badges/ruff.svg"> <img alt="GitHub Actions" src="docs/assets/badges/github-actions.svg">
</p>

<!-- project-summary:start -->
ml-eng-lab is a portfolio of machine-learning notebook experiments. Explore
classification, graphs, language models and other techniques through narrative
notebooks with committed results. The collection contains 21 active task folders
and 29 active notebooks; older experiments are kept separately in a read-only archive.

Use Atlas JupyterHub with local VS Code, or choose a supported local environment.
Each task declares its runtime requirements. Start with the small Iris
classification notebook, then use the task catalog and deeper guides below.
<!-- project-summary:end -->

## 1. Capabilities

This repo serves three overlapping purposes:

- **Personal lab** — a place to prototype new ML tasks quickly.
- **Portfolio** — each task folder reads as a standalone demonstration of a technique.
- **Educational resource** — notebooks include narrative explanations alongside code.

**Paradigms covered** (see [§4.1](#41-active) for the per-task mapping): image classification (numpy from-scratch + PyTorch FFNN), tabular classification + regression, GNNs on graphs (`pytorch-geometric` GraphSAGE / GraphConv / GAT — node classification, link prediction, community detection), NLP (spaCy + NLTK pipelines, BPE tokenizer), transformer LM with sampling stack, diffusion (DDPM), preference alignment (DPO), self-supervised (I-JEPA), Mixture-of-Experts, PEFT (LoRA / DoRA), quantization (PTQ + QAT), pruning, knowledge distillation, model surgery (Net2Net), autoencoders, clustering.

A shared PyTorch toolkit (`nnx`, [`thekaveh-nnx`](https://pypi.org/project/thekaveh-nnx/) on PyPI) provides reusable training-loop, dataset, and visualization primitives that the notebooks consume. Library and tasks co-evolve: each new task lands its required `nnx` additions upstream first ([`thekaveh/NNx`](https://github.com/thekaveh/NNx)), then ml-eng-lab bumps the pinned version here. YAGNI applies — no speculative abstractions in `nnx`.

## 2. Requirements and checkout

Choose an environment before running its commands:

- All local paths require Git, Make and Bash. Clone the repository and run commands from its root.
- Atlas and local Docker require a running Docker Engine with Compose v2. Atlas also requires host-native Ollama answering on loopback; do not substitute an Ollama container.
- The qualified local venv uses Python 3.11.15 on Darwin arm64, Linux x86_64 or Linux aarch64. Native Windows is not a qualified lock target; use a supported Linux environment instead.
- Codespaces requires a GitHub account with available capacity. Download/build time, RAM and disk requirements vary by task.

```bash
git clone https://github.com/thekaveh/ml-eng-lab.git
cd ml-eng-lab
```

See [environment setup](docs/env-setup.md) for platform, GPU and resource details
before selecting a path. These commands can download packages, images and data;
starting Atlas changes your local runtime state.

## 3. Quick start

Four ways to run these notebooks, ordered from managed runtime to local execution.

### 3.1. Atlas JupyterHub + local VS Code (recommended)

Atlas is the direct successor to the previous infrastructure seam. This repository consumes it as
the pinned `infra/` submodule and starts the `ml-eng` track.

Current reviewed Atlas pin: `41ba856f7cd35f0b559d6875e08443eac3e98a98`.

The default workflow keeps notebooks and VS Code on the host while execution uses
the running Atlas JupyterHub kernel. The NumPy MNIST task's `mounted-workspace` mode is
`mounted-required`, so run it from Browser JupyterLab or VS Code attached to the JupyterHub
container at `/home/jovyan/work/ml-eng-lab` rather than relying on a host-local notebook's
remote-kernel CWD.

```bash
git submodule update --init --recursive  # one-time after clone or a pin update
make atlas-setup

# In a separate terminal, only if the host-native daemon is not already running:
ollama serve

# Return to this terminal after the daemon is ready.
make atlas-up
make atlas-connect
```

`make atlas-connect` prints a token-bearing local URL only to an interactive
terminal. In VS Code: open a local notebook, run **Jupyter: Specify Jupyter Server for
Connections**, choose **Existing Jupyter Server**, paste that URL, then choose the remote kernel.
Treat the URL as a password; do not commit or paste it into documentation, and reconnect after
Atlas or JupyterHub restarts instead of relying on a saved URL.

Atlas is intentionally configured with `LLM_PROVIDER_SOURCE=ollama-localhost`: use the native
host Ollama daemon, never an Ollama Docker container. ComfyUI is off for this `ml-eng` consumer
until a task has an approved need for a host-native source. See
[docs/jupyterhub-integration.md](docs/jupyterhub-integration.md) and
[docs/vscode-remote-access.md](docs/vscode-remote-access.md) for the lifecycle, persistence,
and fallback details.

### 3.2. Local Docker

```bash
docker build -t ml-eng-lab .
docker run -p 8888:8888 -v "$(pwd):/home/jovyan/work" --shm-size=4g ml-eng-lab
```

`--shm-size=4g` is the minimum for the GNN notebooks; see [docs/env-setup.md](docs/env-setup.md) §2 for more.

### 3.3. Local venv

```bash
python3.11 -m venv .venv && source .venv/bin/activate
make install-torch-stack
make nlp-assets
make verify-nlp-assets
python -m pip check
make verify-torch-stack
make verify-nnx-install
jupyter lab
```

`make install-torch-stack` selects the exact Darwin arm64, Linux x86_64, or Linux aarch64
hash-required lock declared by `requirements/lock-policy.toml`; it does not resolve the human input
manifests at install time. The result is reproducible for the qualified platform lock, not one
cross-platform binary environment.

For dependency maintenance, use `make verify-dependency-locks` for the offline
policy/input/lock check, `make lock-check` for networked byte regeneration, and
`make image-lock-check` for registry-backed image digest verification. Lock regeneration uses the
exact resolver cutoff in `requirements/lock-policy.toml`; advancing it is a reviewed dependency
update rather than an ambient response to newly published packages.

The root lock already installs the exact hash-required spaCy model wheel. `make nlp-assets`
downloads only NLTK's official VADER ZIP, verifies its URL, byte size, SHA-256, and sole archive
member, then installs it under `NLTK_DATA`. `make verify-nlp-assets` repeats those checks offline;
it never downloads or starts Atlas.

The supported CPU matrix is torch==2.11.0, torchvision==0.26.0,
torch_geometric==2.8.0.post1, pyg-lib==0.8.0+pt211, torch-scatter==2.1.2+pt211,
torch-sparse==0.6.18+pt211, torchao==0.18.0, and thekaveh-nnx[lm]==0.2.0; Linux wheels use the
+pt211cpu local tag. The verifier exercises scatter, sparse, and real sampler canaries, including
preferred pyg-lib sampling and the forced torch-sparse fallback.

See [docs/env-setup.md](docs/env-setup.md) for environment details.

### 3.4. GitHub Codespaces

On the repository page, select **Code → Codespaces → Create codespace on main**.
Wait for dependency setup, then open a notebook in browser VS Code or JupyterLab.
Setup time varies with image-cache and network state.
The environment is CPU-only. Files under `data/` and `runs/` are lost when the
Codespace is deleted; preserve results you need before deletion.

See [Codespaces usage and limits](docs/env-setup.md#414-github-codespaces) for
setup, machine sizing and supported scenarios.

### 3.5. Run your first notebook

After your chosen runtime is ready, open
`notebooks/tabular_classification-iris-mlp-pytorch/notebook.ipynb` and select
its Python kernel. Run the cells from top to bottom. This small CPU example
uses scikit-learn's bundled Iris data, so it does not require a dataset download.

Inspect **Evaluation & Results**: compare the candidate metrics and confusion
matrices, then read the final verdict. The notebook trains models and writes
artifacts; a full rerun replaces the notebook's displayed results when saved.
Use a working copy if you want to preserve committed outputs. The
[Iris walkthrough](docs/notebooks/tabular_classification-iris-mlp-pytorch.md)
explains the model comparison. Consult each task's recorded execution conditions.

## 4. Tasks

### 4.1. Active

| Folder | Task | Dataset | Model | Framework |
|---|---|---|---|---|
| [notebooks/image_classification-mnist-ffnn-numpy/](notebooks/image_classification-mnist-ffnn-numpy/) | Image classification | MNIST | Feed-forward NN (from scratch) | NumPy |
| [notebooks/image_classification-mnist-ffnn-pytorch/](notebooks/image_classification-mnist-ffnn-pytorch/) | Image classification | MNIST | Feed-forward NN | PyTorch (via nnx) |
| [notebooks/node_classification-reddit-gnn-pyg/](notebooks/node_classification-reddit-gnn-pyg/) | Node classification | Reddit2 | GNN (GraphConv, GraphSAGE, GAT) | PyTorch Geometric (via nnx) |
| [notebooks/tabular_classification-iris-mlp-pytorch/](notebooks/tabular_classification-iris-mlp-pytorch/) | Tabular classification | Iris | Feed-forward NN | PyTorch (via nnx) |
| [notebooks/model_surgery-mnist-ffnn-pytorch/](notebooks/model_surgery-mnist-ffnn-pytorch/) | Model surgery (Net2Net) | MNIST | Feed-forward NN | PyTorch (via nnx) |
| [notebooks/quantization-mnist-ffnn-pytorch/](notebooks/quantization-mnist-ffnn-pytorch/) | Quantization (PTQ + QAT) | MNIST | Feed-forward NN | PyTorch (via nnx) + torchao |
| [notebooks/pruning-mnist-ffnn-pytorch/](notebooks/pruning-mnist-ffnn-pytorch/) | Pruning (magnitude sparsity sweep) | MNIST | Feed-forward NN | PyTorch (via nnx) |
| [notebooks/knowledge_distillation-mnist-ffnn-pytorch/](notebooks/knowledge_distillation-mnist-ffnn-pytorch/) | Knowledge distillation (born-again) | MNIST | Feed-forward NN | PyTorch (via nnx) |
| [notebooks/text_generation-tinyshakespeare-transformer-pytorch/](notebooks/text_generation-tinyshakespeare-transformer-pytorch/) | Text generation (autoregressive LM) | TinyShakespeare (embedded) | Decoder-only transformer | PyTorch (via nnx) |
| [notebooks/peft-mnist-to-fmnist-dora-vs-lora-pytorch/](notebooks/peft-mnist-to-fmnist-dora-vs-lora-pytorch/) | PEFT cross-task adaptation (LoRA vs DoRA) | MNIST → Fashion-MNIST | Feed-forward NN + LoRA / DoRA adapters | PyTorch (via nnx) |
| [notebooks/dim_reduction-iris-autoencoder-pytorch/](notebooks/dim_reduction-iris-autoencoder-pytorch/) | Dimensionality reduction (PCA vs autoencoder) | Iris | Autoencoder (FFN with input_dim==output_dim) | PyTorch (via nnx) + sklearn |
| [notebooks/tabular_regression-diabetes-mlp-pytorch/](notebooks/tabular_regression-diabetes-mlp-pytorch/) | Tabular regression | Diabetes | Feed-forward MLP + sklearn baselines | PyTorch (via nnx) + sklearn |
| [notebooks/diffusion-mnist-ddpm-pytorch/](notebooks/diffusion-mnist-ddpm-pytorch/) | Generative (DDPM diffusion) | MNIST | DiffusionMLP denoiser (no U-Net) | PyTorch (via nnx) |
| [notebooks/moe-fmnist-mixture-of-experts-pytorch/](notebooks/moe-fmnist-mixture-of-experts-pytorch/) | Mixture-of-Experts classification | Fashion-MNIST | FeedFwdNN + MoELinear (4 experts, top-2 routing) | PyTorch (via nnx) |
| [notebooks/clustering-iris-kmeans-vs-ae-pytorch/](notebooks/clustering-iris-kmeans-vs-ae-pytorch/) | Unsupervised clustering | Iris | KMeans on raw features vs on AE latent | PyTorch (via nnx) + sklearn |
| [notebooks/link_prediction-karate-graphsage-pyg/](notebooks/link_prediction-karate-graphsage-pyg/) | Link prediction (GNN encoder) | Zachary Karate Club | GraphSAGE + dot-product scorer | PyTorch Geometric |
| [notebooks/community_detection-karate-louvain-vs-gnn-pyg/](notebooks/community_detection-karate-louvain-vs-gnn-pyg/) | Community detection (classical vs GNN) | Zachary Karate Club | Louvain vs GraphSAGE+KMeans | PyTorch Geometric + python-louvain |
| [notebooks/text_classification-agnews-spacy-mlp-pytorch/](notebooks/text_classification-agnews-spacy-mlp-pytorch/) | Text classification (4-topic) | Embedded AG-News-style corpus | spaCy + bag-of-words + MLP | PyTorch (via nnx) + spaCy + sklearn |
| [notebooks/sentiment_classification-vader-mlp-pytorch/](notebooks/sentiment_classification-vader-mlp-pytorch/) | Sentiment classification (rule vs neural) | Embedded review corpus | VADER (lexicon) vs MLP | PyTorch (via nnx) + nltk + spaCy + sklearn |
| [notebooks/preference_alignment-toy-dpo-pytorch/](notebooks/preference_alignment-toy-dpo-pytorch/) | Preference alignment (DPO) | Embedded 16-triplet preference corpus | Tiny TransformerNN (ref + policy) | PyTorch (via nnx) |
| [notebooks/self_supervised-fmnist-jepa-pytorch/](notebooks/self_supervised-fmnist-jepa-pytorch/) | Self-supervised (I-JEPA) + linear probe | Fashion-MNIST | ViT + EMA target + JEPA predictor | PyTorch (via nnx) |

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [Browse this repo on nbviewer](https://nbviewer.org/github/thekaveh/ml-eng-lab/tree/main/) for full rendering of any notebook.

### 4.2. Archived

| Folder | Task | Dataset | Model | Framework |
|---|---|---|---|---|
| [notebooks/archive/codexglue_summarization/](notebooks/archive/codexglue_summarization/) | Code summarization (22 experiments) | CodeXGLUE | Transformers | HuggingFace |

### 4.3. Planned

See [§8 Planned work](#8-planned-work).

## 5. Notebook re-execution policy

Notebooks are tiered by execution cost:

| Tier | What it is | Re-run policy |
|---|---|---|
| **A** | Cheap (<5 min) | `make run-tier-a` deliberately refreshes committed snapshots. CI uses non-mutating `make smoke-tier-a`, which writes fresh artifacts to `/tmp/ml-tier-a`. Tier-A notebooks also accept a `SMOKE_TEST` papermill parameter (default `0` = full run). |
| **B** | Moderate (model-selection sweeps) | Original outputs preserved. `make smoke-tier-b` runs `SMOKE_TEST=1` and writes to `/tmp/`: the parameterized `image_classification-mnist-ffnn-pytorch` notebook shrinks its sweep, and the 4 phase2 reddit notebooks run smoke-truncated epochs/subsets (notebook4 also reduces fanout). |
| **C** | Expensive (main GPU training) | Historical Aug-2023 GPU training-run outputs preserved as artifact. `make smoke-tier-c` runs CPU with `SMOKE_TEST=1` to validate the pipeline without overwriting outputs. |

Tier-A CI writes executed notebook copies under `/tmp/ml-tier-a`; Tier-B/C smoke targets write under `/tmp/ml-smoke`. Papermill intentionally runs each notebook from its own task directory so relative paths behave like an interactive run. Training and evaluation may therefore create ignored task-local `./data/` or `./runs/` artifacts even when source notebook outputs are preserved; committed output text such as `Run saved to ./runs/...` describes that notebook-local runtime location, not files guaranteed to exist in a clean checkout.

Every retained-output code cell in an active notebook also carries a source-freshness marker. The
marker proves that retained outputs correspond to the current source, not that nondeterministic
output rendering is byte-reproducible; output bytes are never part of that claim. Normal execution
targets stamp successful notebooks automatically. See [the repository conventions](docs/conventions.md#524-notebook-output-freshness)
for the exact marker and verification contract.

See [docs/env-setup.md](docs/env-setup.md) for the tier mapping.

## 6. NNx library

Throughout this README, `NNx` refers to the [GitHub project](https://github.com/thekaveh/NNx); the importable Python package is lowercase `nnx`; the PyPI distribution is [`thekaveh-nnx`](https://pypi.org/project/thekaveh-nnx/).

The current contract installs `thekaveh-nnx[lm]==0.2.0` from PyPI. The
recommended Atlas runtime also pins 0.2.0. The `[lm]` extra supplies tokenizer
and dataset support required by the two language-model notebooks.

Import supported symbols through the public facade, such as
`from nnx import NNModel, NNParams`. Do not use deep module paths when a
public export exists. Verify the released-wheel installation with
`make verify-nnx-install`.

For intentional upstream development, install a separate NNx checkout editable
and run `NNX_ALLOW_EDITABLE=1 make test-nnx-surface`. This is development
validation, not released-wheel evidence. The [NNx guide](docs/nnx-library.md)
and [canonical dependency contract](docs/dependency-contracts.md) preserve extension
procedures, the completed 0.2.2 review and the reason for retaining 0.2.0.

## 7. Repository conventions

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow. Key points:

- Each active task is a self-contained directory under `notebooks/` using the `[task]-[dataset]-[model]-[framework]` naming convention. No `tasks/` subdirectory.
- Shared library code lives in `nnx` (the PyPI-installed `thekaveh-nnx` package), not a local `common/`.
- Active notebooks normally retain executed-cell outputs; documented long-running or preservation
  exceptions may intentionally omit them.
- Tier-C notebooks have their Aug-2023 outputs preserved; never re-execute them in place.
- `notebooks/archive/` is read-only.

## 8. Planned work

See the [roadmap](docs/conventions.md#57-roadmap) for proposed tasks and
[Contributing](CONTRIBUTING.md) for the task-admission workflow. Planned tasks
are separate from the active catalog above.

## 9. License

MIT. See [LICENSE](LICENSE).

## 10. Other documentation

The README is the entry point; the items below are the hub's index of secondary documentation.

### 10.1. Governance, workflow + history

- [SECURITY.md](SECURITY.md) — private vulnerability reporting, supported versions, coordinated disclosure, and dependency handling.
- [CONTRIBUTING.md](CONTRIBUTING.md) — workflow, conventions, "Adding a new task folder" recipe, verifier+pytest gates.
- [CHANGELOG.md](CHANGELOG.md) — Keep-a-Changelog release notes.

### 10.2. Environment + runtimes

- [docs/env-setup.md](docs/env-setup.md) — the four setup paths (Atlas / Docker / venv / Codespaces), GPU notes, Tier mapping.
- [docs/jupyterhub-integration.md](docs/jupyterhub-integration.md) — Atlas JupyterHub lifecycle and ownership boundary.
- [docs/vscode-remote-access.md](docs/vscode-remote-access.md) — local VS Code remote-kernel path and browser fallback.
- [docs/atlas-pin-bump-runbook.md](docs/atlas-pin-bump-runbook.md) — reviewed Atlas pin-bump and future-service admission runbook.
- [docs/dependency-contracts.md](docs/dependency-contracts.md) — dependency audit ledger, local/CI Torch contract, Atlas runtime evidence, and automated quantization contract.
- [docs/architecture.md](docs/architecture.md) — system/context view for the notebook lab, verifier, CI, runtime environments, and documentation delivery.
- [docs/diagrams/README.md](docs/diagrams/README.md) — provenance and regeneration contract for embedded architecture diagrams.
- [docs/maintenance/overnight-2026-07-04.md](docs/maintenance/overnight-2026-07-04.md) — current overnight maintenance pass log and issue tracker.
- [docs/maintenance/overnight-2026-07-02.md](docs/maintenance/overnight-2026-07-02.md) — historical overnight maintenance run that reached its hard cap.
- [docs/maintenance/notebooks-reorganization-design.md](docs/maintenance/notebooks-reorganization-design.md) — completed design record for the `notebooks/<task>/` layout and archive move.
- [docs/maintenance/notebooks-reorganization-implementation.md](docs/maintenance/notebooks-reorganization-implementation.md) — completed implementation record for the notebook/archive reorganization and runtime-path contract.

### 10.3. Issue sinks for external code

- [docs/FINDINGS-NNX.md](docs/FINDINGS-NNX.md) — issue log for the `thekaveh-nnx` library (append findings here; do not edit nnx directly via this repo — fixes land upstream at [`thekaveh/NNx`](https://github.com/thekaveh/NNx)).
- [docs/FINDINGS-ATLAS.md](docs/FINDINGS-ATLAS.md) — Atlas-consumer findings; fixes to Atlas itself belong upstream.

### 10.4. Archive

- [notebooks/archive/README.md](notebooks/archive/README.md) — preserved Aug-2023 codexglue summarization experiments (22 runs); read-only.
