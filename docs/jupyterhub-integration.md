# JupyterHub integration

The recommended runtime for these notebooks is the `jupyterhub` service in the [`genai-vanilla`](https://github.com/thekaveh/genai-vanilla) stack. That service's image is DS/ML-capable (PyTorch + PyG + Lightning baked in) as of `genai-vanilla@cb4d8f40c3862b82511b010ae467638d846f8f9c`.

## Why this runtime

- The image contains the right pinned versions of torch, torch_geometric, pytorch-lightning, torchmetrics. No local install required.
- The genai-vanilla stack also runs LiteLLM, Weaviate, Neo4j, Supabase — useful for any LLM/RAG-flavored future tasks.
- One running container; no environment drift between sessions.

## The deployment overlay

The ml repo is mounted into the running jupyterhub container via a per-user Docker Compose override file. The override lives in this repo (at [`deploy/genai-vanilla-jupyterhub.override.yml`](../deploy/genai-vanilla-jupyterhub.override.yml)) — versioned alongside the rest of the project — and is symlinked into the local genai-vanilla checkout where Docker Compose auto-discovers it.

### Why an overlay file

genai-vanilla's tree should not know about ml-specific paths (`/Users/kaveh/repos/ml/...`). Putting the mount in the upstream `services/jupyterhub/compose.yml` would couple genai-vanilla to one specific consumer. The override pattern keeps genai-vanilla generic.

### Setup

```bash
# 1. Set env vars in genai-vanilla/.env:
echo "ML_REPO_PATH=/Users/kaveh/repos/ml" >> /path/to/genai-vanilla/.env
echo "HOST_SSH_DIR=$HOME/.ssh" >> /path/to/genai-vanilla/.env

# 2. Symlink the overlay into the genai-vanilla checkout:
/Users/kaveh/repos/ml/scripts/link-jupyter-override.sh

# 3. Start the stack (override auto-applies):
cd /path/to/genai-vanilla && ./start.sh

# 4. Inside the running container, install nnx editable:
docker exec -it <project>-jupyterhub /home/jovyan/work/ml/scripts/setup-in-jupyter.sh
# (Or attach with VS Code and run the script from the terminal.)
```

## Tested against

This integration is tested against `genai-vanilla@cb4d8f40c3862b82511b010ae467638d846f8f9c` (Phase 1 merge).

When genai-vanilla advances, re-test the integration and update the SHA recorded here.

## Things to know

- **First-container setup**: `setup-in-jupyter.sh` does a `pip install -e /home/jovyan/work/ml/nnx`. The editable install persists in the named `jupyterhub-data` volume, so it survives container restarts. After an image rebuild (`docker compose build jupyterhub --no-cache`), re-run the script.
- **Git operations**: the host's `~/.ssh` is mounted read-only at `/home/jovyan/.ssh`. `git push` works with the host identity. Configure name/email via `git config --global` inside the container if needed.
- **Orphaned containers**: if `./start.sh` reports name conflicts, run `./stop.sh` first; if names remain, remove with `docker rm -f <name>`. Then re-start.
- **Image rebuilds**: run from the genai-vanilla root with `.env` loaded, e.g. `cd /path/to/genai-vanilla && docker compose build jupyterhub`. Building from a different directory may produce the wrong image tag (project-name-prefixed).

## Common failure modes

- **`ModuleNotFoundError: No module named 'nnx'`** — `setup-in-jupyter.sh` wasn't run for this container instance. Run it.
- **`from nnx.nn.net.feed_fwd_nn import FeedFwdNN` fails** with import error — submodule not initialized on the host. Run `git submodule update --init --recursive` in the ml repo.
- **Notebook hangs at first cell** — likely waiting for a stack service that didn't come up. Check `docker compose ps` for unhealthy services in the genai-vanilla stack.
