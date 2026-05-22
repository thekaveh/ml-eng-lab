# JupyterHub integration

The recommended runtime for these notebooks is the `jupyterhub` service in the [`genai-vanilla`](https://github.com/thekaveh/genai-vanilla) stack. That service's image is DS/ML-capable (PyTorch + PyG + Lightning baked in) as of `genai-vanilla@cb4d8f4`.

This repo vendors a snapshot of genai-vanilla as a git submodule at [`vendor/genai-vanilla`](../vendor/genai-vanilla). The submodule is pinned to the `ml-integration` branch of `thekaveh/genai-vanilla`, which is identical to `main` except for a single committed `docker-compose.override.yml` that bind-mounts this ml repo into the jupyterhub container.

## Why this layout

The two repos have orthogonal release cycles. Vendoring genai-vanilla as a submodule means:

- Everything ml's runtime needs lives inside ml's git boundary.
- The override file is a normal versioned file inside the submodule — no symlinks, no setup scripts to remember.
- ml records a specific genai-vanilla SHA via the submodule pointer — reproducible.
- genai-vanilla's `main` branch stays generic; only the `ml-integration` branch carries the override.

The standalone `/Users/kaveh/repos/genai-vanilla/` checkout is retained for genai-vanilla-only work and is independent of this submodule.

## Setup

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/thekaveh/ml.git
# Or, if already cloned:
git submodule update --init --recursive
```

Optional `.env` configuration in `vendor/genai-vanilla/.env` (the override has working defaults if you don't):

```bash
ML_REPO_PATH=../..               # default: ../.. (= ml repo root from vendor/genai-vanilla/)
HOST_SSH_DIR=$HOME/.ssh          # default: ~/.ssh
```

## Run

```bash
cd /Users/kaveh/repos/ml/vendor/genai-vanilla
./start.sh
```

The override applies automatically. Inside the container, the ml repo is at `/home/jovyan/work/ml`.

## One-time per container: install nnx editable

```bash
docker exec -it <project>-jupyterhub /Users/kaveh/repos/ml/scripts/setup-in-jupyter.sh
# Or attach with VS Code and run scripts/setup-in-jupyter.sh from the terminal.
```

Wait — that path is the host path. Inside the container, the script lives at `/home/jovyan/work/ml/scripts/setup-in-jupyter.sh`:

```bash
docker exec -it <project>-jupyterhub /home/jovyan/work/ml/scripts/setup-in-jupyter.sh
```

The editable install persists in the named `jupyterhub-data` volume across container restarts. Only needs re-running after an image rebuild.

## Bumping the submodule

When genai-vanilla's main advances and you want to incorporate those changes:

```bash
cd /Users/kaveh/repos/ml/vendor/genai-vanilla
git checkout ml-integration
git fetch origin
git merge origin/main             # rebase + push if you prefer linear history
git push origin ml-integration
cd ../..
git add vendor/genai-vanilla
git commit -m "ml: bump genai-vanilla submodule to <new-sha>"
```

## Tested against

Submodule pinned to a commit on `ml-integration` that includes the ML-capable image changes from `genai-vanilla@cb4d8f4` plus the integration override.

## Things to know

- **First-container setup**: the editable nnx install persists in the named `jupyterhub-data` volume. After an image rebuild (`docker compose build jupyterhub --no-cache`), re-run `setup-in-jupyter.sh`.
- **Git operations**: the host's `~/.ssh` is mounted read-only at `/home/jovyan/.ssh`. `git push` works with the host identity.
- **Orphaned containers**: if `./start.sh` reports name conflicts, run `./stop.sh` first; if names remain, remove with `docker rm -f <name>`.

## Common failure modes

- **`ModuleNotFoundError: No module named 'nnx'`** — `setup-in-jupyter.sh` wasn't run for this container instance.
- **`from nnx.nn.net.feed_fwd_nn import FeedFwdNN` fails** at install time — submodule not initialized. Run `git submodule update --init --recursive` at the ml repo root.
- **Notebook hangs at first cell** — likely waiting for a stack service that didn't come up. Check `docker compose ps`.
