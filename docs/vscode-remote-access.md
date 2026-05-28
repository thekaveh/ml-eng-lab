# VS Code remote access to the jupyterhub container

Three modes, pick by use case.

## 1. Mode 1 — Attach to Running Container (Recommended)

Install extension: **Dev Containers** (ms-vscode-remote.remote-containers).

After the genai-vanilla stack is up:
1. Open the Docker view in VS Code (left sidebar; install the Docker extension if missing).
2. Find the running `<project>-jupyterhub` container.
3. Right-click → **Attach Visual Studio Code**.

A new VS Code window opens inside the container. The ml-lab repo is at `/home/jovyan/work/ml-lab`. Open that folder.

What works inside:
- Native VS Code notebook UI with kernel = `python3` (the container's interpreter, with all deps installed).
- Integrated terminal with `git`, `pip`, etc.
- The host's `~/.ssh` is mounted read-only at `/home/jovyan/.ssh`; `git push` works with the host identity.

This is the best ergonomic match for "edit + commit, no local install".

## 2. Mode 2 — Connect to Remote Jupyter Server

Install extension: **Jupyter** (ms-toolsai.jupyter).

After the stack is up:
1. Open the local `.ipynb` file in VS Code (your host machine's path: e.g. `~/repos/ml-lab/...`).
2. `Cmd-Shift-P` → **Jupyter: Specify Jupyter Server for Connections** → paste:
   ```
   http://localhost:63081/?token=<JUPYTER_TOKEN>
   ```
   The token is in genai-vanilla's `.env` as `JUPYTERHUB_TOKEN`.
3. The kernel now runs in the container; the file is local.

Notes:
- This works because the ml-lab repo is bind-mounted: the path tree exists on both sides. Relative paths in notebooks (`./data/foo.csv`) resolve correctly because the kernel's CWD is set by VS Code to the notebook's directory inside the container.
- Useful when you want to keep using your local VS Code window's extensions and context.

## 3. Mode 3 — Browser Jupyter Lab

The simplest path. After the stack is up:
- Open `http://localhost:63081/?token=<JUPYTER_TOKEN>`
- Navigate to `work/ml-lab/...`
- The `jupyterlab-git` extension (shipped in the image) handles git operations.

Use this for quick edits, demos, or when VS Code is overkill.

## 4. Not pursued

- **Remote-SSH** — requires an SSH server in the container. Extra surface area for no benefit over Mode 1.
- **`.devcontainer.json` reopen-in-container** — would rebuild a new image. We have a long-lived running container already; Mode 1's attach is simpler.
