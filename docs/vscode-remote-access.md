# VS Code remote access to the jupyterhub container

Three modes, pick by use case. Mode 2 is the default for most ml-eng-lab work as of genai-vanilla `cbad341` (PR #26) — the image ships ml-eng-lab's external dependencies and NLP assets, while `nnx` still needs the `thekaveh-nnx[lm]==0.2.0` image bump or the per-session install workaround described below.

## 1. Mode 1 — Attach to Running Container

Install extension: **Dev Containers** (`ms-vscode-remote.remote-containers`).

After the genai-vanilla stack is up:
1. Open the Docker view in VS Code (left sidebar; install the Docker extension if missing).
2. Find the running `<project>-jupyterhub` container.
3. Right-click → **Attach Visual Studio Code**.

A new VS Code window opens inside the container. The container's CWD is `/home/jovyan/work/`. On the §2-wrapper path of [jupyterhub-integration.md](jupyterhub-integration.md#2-persistence-path-wrapper-script-bind-mount), the ml-eng-lab repo is bind-mounted at `/home/jovyan/work/ml-eng-lab/`; open that folder.

What works inside:
- Native VS Code notebook UI with kernel = `python3` (the container's interpreter, with external deps/assets installed via genai-vanilla PR #26 and `nnx` caveated as in §2).
- Integrated terminal with `git`, `pip`, etc.
- If using the §2-wrapper path, `/home/jovyan/.ssh` is empty by default. Set `HOST_SSH_DIR=/path/to/keys` before `scripts/start-jupyterhub.sh` only when you want to opt into a read-only host-key mount for `git push`.

Use this when you want the full container shell experience.

## 2. Mode 2 — Connect to Remote Jupyter Server (default)

Install extension: **Jupyter** (`ms-toolsai.jupyter`).

After the stack is up:
1. Open the local `.ipynb` file in VS Code (your host machine's path: e.g. `~/repos/ml-eng-lab/...`).
2. `Cmd-Shift-P` → **Jupyter: Specify Jupyter Server for Connections** → paste:
   ```
   http://localhost:<JUPYTERHUB_PORT>/?token=<JUPYTERHUB_TOKEN>
   ```
   `JUPYTERHUB_PORT` and `JUPYTERHUB_TOKEN` are env vars in genai-vanilla's `.env`. The shipped default port is `63081`. The token is empty by default — set it to a fixed value before `./start.sh`, otherwise the container's `start-notebook.sh` generates a random token on every boot that you'd have to scrape from `docker logs <project>-jupyterhub | grep token`.
3. The kernel now runs in the container; the `.ipynb` file is local.

**Coverage:** Once the genai-vanilla image bumps its baked pip layer to `thekaveh-nnx[lm]==0.2.0` (the 2026-06-14 ml-eng-lab PyPI-migration follow-up; the image currently bakes the now-defunct `nnx-pytorch` distribution name), this works out of the box for the tier-covered notebooks except `notebooks/image_classification-mnist-ffnn-numpy/notebook.ipynb`, which imports 8 sibling `.py` modules from its own task folder (`consts`, `feed_fwd_nn`, `linear_layer`, etc.) that aren't pip-installable and require the ml-eng-lab repo to be accessible inside the container. The quantization notebook remains manual-only under `torch>=2.5` + `torchao>=0.17`. For the numpy notebook, use the §2-wrapper path of [jupyterhub-integration.md](jupyterhub-integration.md#2-persistence-path-wrapper-script-bind-mount) (bind-mount the repo) and open the notebook from `/home/jovyan/work/ml-eng-lab/notebooks/image_classification-mnist-ffnn-numpy/notebook.ipynb`. Until the image bumps, every Mode-2 session importing `nnx` needs a one-shot `docker exec ... pip install thekaveh-nnx[lm]==0.2.0` in the running jupyterhub container.

**Relative paths:** Notebook code that does `pd.read_csv("./data/foo.csv")` or `NNRun.save()` resolves against the kernel's CWD inside the container (`/home/jovyan/`), not your host repo. On the standalone-genai-vanilla path, those artifacts land in the `jupyterhub-data` named volume. On the wrapper-and-bind-mount path, they land in your host repo. Pick the path based on whether you want host-side persistence — see [jupyterhub-integration.md §1 vs §2](jupyterhub-integration.md#1-default-path-standalone-genai-vanilla-vs-code-mode-2).

## 3. Mode 3 — Browser JupyterLab

The simplest path. After the stack is up:
- Open `http://localhost:<JUPYTERHUB_PORT>/?token=<JUPYTERHUB_TOKEN>` in a browser
- Navigate to `work/ml-eng-lab/...` (wrapper-and-bind-mount path) or upload notebooks individually (standalone path)
- The `jupyterlab-git` extension (shipped in the image) handles git operations

Use this for quick edits, demos, or when VS Code is overkill.

## 4. Not pursued

- **Remote-SSH** — requires an SSH server in the container. Extra surface area for no benefit over Mode 1.
- **`.devcontainer.json` reopen-in-container for the existing JupyterHub container** — not applicable; Mode 1 attaches to the long-lived running container. The repo's `.devcontainer/devcontainer.json` is a separate Codespaces/local-devcontainer path that builds its own environment.
