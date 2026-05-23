# JupyterHub integration

The recommended runtime for these notebooks is the `jupyterhub` service in the [`genai-vanilla`](https://github.com/thekaveh/genai-vanilla) stack. That service's image is DS/ML-capable (PyTorch + PyG + Lightning baked in) as of `genai-vanilla@cb4d8f4`.

This repo vendors a snapshot of genai-vanilla as a git submodule at [`vendor/genai-vanilla`](../vendor/genai-vanilla), pinned to genai-vanilla's `main` branch. The ml-specific docker compose override lives in this repo at [`deploy/genai-vanilla-jupyterhub.override.yml`](../deploy/genai-vanilla-jupyterhub.override.yml) and is layered onto the stack via a wrapper script ([`scripts/start-jupyterhub.sh`](../scripts/start-jupyterhub.sh)) that sets `COMPOSE_FILE` and invokes the submodule's `./start.sh`.

## Why this layout

The two repos have orthogonal release cycles. The submodule pin pulls a known-good genai-vanilla snapshot; the override file in ml describes ml's deployment needs. genai-vanilla's `main` stays clean — no ml-specific files. Bumping the submodule is one command (`git submodule update --remote`).

## Setup

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/thekaveh/ml.git
# Or, if already cloned:
git submodule update --init --recursive
```

## Run

```bash
/Users/kaveh/repos/ml/scripts/start-jupyterhub.sh
```

The wrapper sets `ML_REPO_PATH` (the ml repo root) and `HOST_SSH_DIR` (defaults to `~/.ssh`), exports `COMPOSE_FILE` to layer the override onto genai-vanilla's base compose, and execs genai-vanilla's `./start.sh`. From the running container's perspective, the ml repo is at `/home/jovyan/work/ml`.

To run with custom paths:

```bash
HOST_SSH_DIR=/path/to/keys ml/scripts/start-jupyterhub.sh
```

## One-time per container: install nnx editable

```bash
docker exec -it <project>-jupyterhub /home/jovyan/work/ml/scripts/setup-in-jupyter.sh
```

Or attach with VS Code and run from the integrated terminal. The editable install persists in the named `jupyterhub-data` volume.

## Bumping the submodule

Standard submodule workflow:

```bash
cd /Users/kaveh/repos/ml/vendor/genai-vanilla
git fetch origin
git checkout main
git pull origin main
cd ../..
git add vendor/genai-vanilla
git commit -m "ml: bump genai-vanilla submodule to <new-sha>"
```

## Tested against

Submodule pinned to a commit on genai-vanilla `main` that includes the DS/ML-capable image (Phase 1, `cb4d8f4` or later).

## Common failure modes

- **`ModuleNotFoundError: No module named 'nnx'`** — `setup-in-jupyter.sh` wasn't run for this container instance.
- **Submodule not found at vendor/genai-vanilla/** — run `git submodule update --init --recursive` at the ml repo root.
- **`ML_REPO_PATH variable is not set`** during compose up — you ran `cd vendor/genai-vanilla && ./start.sh` directly instead of using the wrapper. Use `scripts/start-jupyterhub.sh`.
- **Notebook hangs at first cell** — stack service didn't come up. Check `docker compose ps` (from inside `vendor/genai-vanilla/`).
