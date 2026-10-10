# 4.1 Environment setup

Four paths are supported. Atlas is the default because it gives local VS Code a reproducible,
running JupyterHub kernel while retaining the checked-out notebooks on the host.

Before local commands, install Git, Make and Bash. Atlas/Docker paths require
a running Docker Engine with Compose v2. The qualified venv uses Python 3.11.15
on Darwin arm64, Linux x86_64 or Linux aarch64; native Windows is not a qualified
lock target. Codespaces requires an account with available capacity.

From a fresh local checkout:

```bash
git clone https://github.com/thekaveh/ml-eng-lab.git
cd ml-eng-lab
```

Commands below run from this checkout root. Package, image and data downloads
need network access. Check the chosen task's RAM, disk and GPU requirements first.

## 4.1.1 Atlas JupyterHub + local VS Code (recommended)

Atlas is the direct successor to the previous infrastructure seam. This repository consumes Atlas
as the pinned `infra/` submodule.

Current reviewed Atlas pin: `41ba856f7cd35f0b559d6875e08443eac3e98a98`.

The lifecycle wrapper `scripts/atlas-up.sh` supplies `--track ml-eng`, while `atlas.consumer.yml` declares
`BASE_PORT=auto`, a container JupyterHub, and native host Ollama. The parent-owned overlay
bind-mounts this checkout at `/home/jovyan/work/ml-eng-lab`. That mounted path is available to
Browser JupyterLab or an attached JupyterHub container; it is not assumed by every host-local
notebook paired with a remote kernel.

### 4.1.1.1 Start the runtime

```bash
git submodule update --init --recursive
make atlas-setup

# In a separate terminal, only if the host daemon is not already running:
ollama serve

# Return here after the daemon is ready.
make atlas-up
make atlas-connect
```

`make atlas-setup` creates ignored, machine-local environment files without replacing existing
ones. `make atlas-up` validates the consumer contract and starts Atlas in detached mode. It refuses
to proceed unless native Ollama answers on the loopback address. `make atlas-connect` prints a
token-bearing URL only to an interactive terminal; follow its VS Code instructions rather than
recording the URL in a shell history, issue, or document.

### 4.1.1.2 Source and persistence policy

- **Ollama:** `LLM_PROVIDER_SOURCE=ollama-localhost` is mandatory. Use the host-native Ollama
  daemon; never run an Ollama Docker container for this consumer.
- **ComfyUI:** the `ml-eng` track leaves it disabled. A future approved task can request only a
  host-native source (`localhost` or managed Apple-silicon MPS); automatic and container sources
  are prohibited.
- **Ports:** `BASE_PORT=auto` avoids collisions with sibling Atlas consumers. Do not hard-code a
  published port in a notebook or docs; use `make atlas-connect` for the current endpoint.
- **Artifacts:** the default task contract keeps remote-workspace artifacts on the Atlas Jupyter
  volume. The NumPy MNIST task is explicitly `mounted-workspace` / `mounted-required`: use Browser
  JupyterLab or VS Code attached to the JupyterHub container from `/home/jovyan/work/ml-eng-lab`,
  so its ignored `data/` and `runs/` directories remain in this checkout.

### 4.1.1.3 Stop and clean up

```bash
make atlas-down          # normal shutdown; preserves Atlas volumes
COLD=1 make atlas-down   # deliberate destructive teardown of Atlas volumes
```

The normal stop is the expected development command. Use the cold form only when intentionally
discarding persisted Atlas data. Never edit files inside `infra/`; consumer configuration belongs
in the parent repository. See [atlas-pin-bump-runbook.md](atlas-pin-bump-runbook.md) for pin
updates and future service admission.

## 4.1.2 Local Docker

```bash
docker build -t ml-eng-lab .
docker run -p 8888:8888 -v "$(pwd):/home/jovyan/work" --shm-size=4g ml-eng-lab
```

Open the token URL printed at startup. This is a self-contained notebook image, not the Atlas
runtime; use it when you specifically need an isolated local image. `--shm-size=4g` is the minimum
for the GNN notebooks; serious GNN training may need 16–50 GiB.

## 4.1.3 Local Python venv

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

Use Python 3.11.15 for the qualified host flow. `make install-torch-stack` selects the exact
hash-required target under `requirements/locks/` using `requirements/lock-policy.toml`; the
installer ends with binary-only `thekaveh-nnx[lm]==0.2.0`. This is reproducible for that qualified
platform lock, not a claim that Darwin and Linux share identical wheel bytes.

The spaCy model is
already locked as `en-core-web-sm==3.8.0`; `make nlp-assets` downloads only the VADER lexicon.
It accepts only the official NLTK data ZIP described by `requirements/nlp-assets.toml`, verifies
its byte size, SHA-256, and sole member, and installs it under the explicit `NLTK_DATA` root.

`make verify-nlp-assets` is offline and fails on a missing, corrupt, substituted, or symlinked
asset. After that last data download, package and data state are frozen through pip-check, Torch,
NNx, and NLP verification and the workload.

A clean install needs network access to the official
URL; verification of an already installed valid ZIP does not. Neither command starts Atlas.

## 4.1.4 GitHub Codespaces

Create a codespace on `main` from GitHub. `.devcontainer/devcontainer.json` runs
`make codespace-setup`, which installs the local/CI dependency contract and NLP assets in the
auto-cloned `/workspaces/ml-eng-lab` checkout. The exact Linux lock and tag-plus-index devcontainer
identity are governed by `requirements/lock-policy.toml` and `requirements/image-lock.json`. Use
browser VS Code or JupyterLab as appropriate.

Codespaces is CPU-only and disposable: data/ and runs/ are lost when a codespace is deleted. The
quantization notebook uses the same canonical stack and its bounded PTQ/QAT plus checkpoint
contract runs through Tier B.

### 4.1.4.1 Usage and supported scenarios

**Scenarios this supports**:
- Onboarding a new contributor — they click "Create codespace" and receive a working environment after the one-time dependency install, with no local installation.
- Running a notebook on a larger host without local install (the smallest Codespace machine is 2-core / 8 GB RAM — comparable to a low-end laptop, sufficient for every Tier-A notebook; bump to 4-core / 16 GB if any Tier-B sweep feels slow).
- Short exploratory run without polluting the local Python env.
- The `notebooks/image_classification-mnist-ffnn-numpy/notebook.ipynb` edge case (it imports sibling `.py` modules from its own folder) works natively because Codespaces clones the repo into `/workspaces/ml-eng-lab`.
- The `quantization-mnist-ffnn-pytorch` notebook is Tier B. Its one-epoch smoke proves PTQ, QAT conversion, exact QAT checkpoint reconstruction, and a machine-readable output contract; its full path retains three epochs.

**Scenarios this does NOT support**:
- GPU workloads — this repository's Codespaces configuration is CPU-oriented. Heavier notebooks can run slowly; GPU work requires a separately validated GPU environment.
- Data persistence across Codespace deletions — anything written to `./data/` or `./runs/` is gone when the Codespace is deleted (Codespaces are intended to be cheap and disposable). Commit any results you want to keep, or use Codespaces' "prebuild" feature if dep install time becomes a bottleneck.
**How to use**:

1. On [github.com/thekaveh/ml-eng-lab](https://github.com/thekaveh/ml-eng-lab) → green **Code** button → **Codespaces** tab → **Create codespace on main**.
2. Wait for `postCreateCommand` to run `make codespace-setup` (= Torch-first dependency install + `make nlp-assets`). Progress is visible in the terminal panel.
3. Open any notebook. You can either:
   - **Stay in VS Code (browser)** — the Jupyter / Python extensions are preinstalled per the devcontainer config and work for all 29 active notebooks. Use `make smoke-tier-b` for the bounded quantization execution contract.
   - **Switch to JupyterLab** — click the dropdown next to "Open" on github.com → choose JupyterLab. To make JupyterLab the single-click default for all your codespaces, go to [github.com/settings/codespaces → Editor preference → JupyterLab](https://github.com/settings/codespaces).

See `.devcontainer/devcontainer.json` for the exact image + extension set, and `Makefile` `codespace-setup` target for the Codespaces/venv install recipe. The §4.1.2 Docker path bakes the same Torch-first dependency order into `Dockerfile`. Available Codespaces quota and billing depend on your account plan; check your account before allocating capacity.


## 4.1.5 GPU notes

The current checked-in local Docker and Codespaces paths are CPU-oriented. For a new GPU run, use
an environment with a matching CUDA Torch/PyG stack and record the compatibility result. Do not
silently repoint the Atlas consumer to a containerized Ollama or ComfyUI service to obtain GPU
access; service admission is a separate reviewed change.

## 4.1.6 Tier mapping

The authoritative lists are `Makefile` (`TIER_A`, `TIER_B`, and `TIER_C`) and
`scripts/verify_repo_config.yaml` (`tier_a_notebooks`).

- **Tier A:** `make run-tier-a` intentionally refreshes an in-place snapshot. CI instead runs
  `make smoke-tier-a`, writes generated copies to `/tmp/ml-tier-a`, verifies every expected copy,
  checks that tracked source notebooks remain unchanged, and uploads those copies as an artifact.
- **Tier B:** `make smoke-tier-b` writes parameterized smoke outputs to `/tmp/ml-smoke`. The
  quantization task runs one FP32 epoch plus one QAT epoch, reconstructs the saved shadow
  checkpoint exactly, proves final torchao conversion, and emits a verified semantic marker.
- **Tier C:** `make smoke-tier-c` runs the expensive Reddit training pipelines in smoke mode;
  preserved code cells remain guarded by E5 against the immutable
  `tier-c-deterministic-seeding-atlas-baseline-2026-08-22` tag.
